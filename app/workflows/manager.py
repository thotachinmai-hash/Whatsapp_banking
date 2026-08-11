from typing import Any
import re

from app.logger import get_logger
from app.workflows.constants import (
    WORKFLOW_CHEQUE,
    WORKFLOW_LOAN,
    WORKFLOW_KYC,
    WORKFLOW_ONBOARDING,
    WORKFLOW_TRANSFER,
    STEP_COLLECT_AADHAAR,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_KYC_FORM,
    STEP_UPLOAD_CHEQUE,
    STEP_SELECT_BENEFICIARY,
)
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow, update_workflow_data
from app.workflows.memory import complete_workflow
from app.workflows.memory import set_workflow_step
from app.services.registration_gate import GREETING_KEYWORDS
from app.conversation.responses.transfer import render_insufficient_balance
from app.conversation.responses.common import render_goodbye, render_main_menu_list, render_workflow_boundary_with_step, render_workflow_step_hint
from app.conversation.responses.cheque import render_cheque_deposit_started
from app.conversation.responses.kyc import render_kyc_update_started
from app.conversation.intent.rules import BANKING_DOMAIN_KEYWORDS
from app.services.llm_understanding import answer_side_question, detect_step_or_workflow_jump, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, StructuredResponse
from app.conversation.workflow_adapter import start_workflow_directly

from app.workflows.processors.cheque import ChequeWorkflowProcessor
from app.workflows.processors.loan import LoanWorkflowHandler, detect_loan_type_from_text, loan_type_list_prompt
from app.workflows.processors.kyc import KYCWorkflowHandler
from app.workflows.processors.onboarding import OnboardingWorkflowHandler
from app.workflows.processors.transfer import TransferWorkflowProcessor, has_transferable_balance, start_transfer_from_text

logger = get_logger(__name__)

# Shared human-readable label per workflow type — used both when asking
# "do you want to stop your {label}?" and in the final cancellation
# message, so the two always agree with each other.
_WORKFLOW_LABELS = {
    WORKFLOW_CHEQUE: "Cheque deposit",
    WORKFLOW_TRANSFER: "Money transfer",
    WORKFLOW_LOAN: "Loan application",
    WORKFLOW_KYC: "KYC update",
}


class WorkflowManager:
    """
    Routes incoming requests to the appropriate workflow handler
    if the customer has an active workflow.
    """

    def __init__(self):

        self.cheque_handler = ChequeWorkflowProcessor()
        self.loan_handler = LoanWorkflowHandler()
        self.kyc_handler = KYCWorkflowHandler()
        self.onboarding_handler = OnboardingWorkflowHandler()
        self.transfer_handler = TransferWorkflowProcessor()

    async def handle(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        """
        Handle an active workflow.

        Returns:
        {
            "handled": True/False,
            "response": "..."
        }
        """

        workflow = get_workflow(phone_number)

        if workflow is None:

            logger.info(
                f"[{trace_id}] No active workflow found | phone={phone_number[-4:]}"
            )

            if _is_cancel_command(query) or _is_closing_word(query):
                # Nothing is active to cancel — the customer already left a
                # workflow (or never started one) and is now explicitly
                # saying they're done (an exit word, or a natural closing
                # phrase like "thanks, that's all" / "bye"). Say goodbye
                # rather than pushing the main menu back at them again,
                # which reads as ignoring what they just said.
                logger.info(f"[{trace_id}] Exit acknowledged | phone={phone_number[-4:]} | trigger={query[:20]!r}")
                return {"handled": True, "response": render_goodbye()}
            if _is_back_command(query):
                from app.database import get_customer_by_phone
                customer = get_customer_by_phone(phone_number)
                name = customer.get("full_name", "there") if customer else "there"
                return {"handled": True, "response": render_main_menu_list(name, greeting=False)}
            return {"handled": False, "response": None}

        workflow_type = workflow["type"]

        # Resolve a pending "do you want to stop?" confirmation (see the
        # cancel/closing-word branch below) before anything else this turn.
        if workflow.get("data", {}).get("pending_stop_confirmation"):
            return _resolve_pending_stop(workflow, workflow_type, phone_number, query, trace_id, self.transfer_handler)

        # An explicit stop/cancel word, or a natural closing phrase ("thanks,
        # that's all", "bye"), must not silently abandon real in-progress
        # work (a beneficiary already picked, a document already uploaded).
        # Ask once, then act on whatever they reply via the branch above —
        # onboarding is the one exception (nothing of value is lost by
        # restarting it, and there's no menu to "continue" back into yet).
        if _is_cancel_command(query) or _is_closing_word(query):
            if workflow_type == WORKFLOW_ONBOARDING:
                complete_workflow(phone_number)
                logger.info(
                    f"[{trace_id}] Onboarding cancelled | "
                    f"phone={phone_number[-4:]} | trigger={query[:20]!r}"
                )
                from app.conversation.responses.onboarding import render_registration_cancelled
                return {"handled": True, "response": render_registration_cancelled()}

            label = _WORKFLOW_LABELS.get(workflow_type, "request")
            update_workflow_data(phone_number, {
                "pending_stop_confirmation": True,
                "pending_stop_was_closing": _is_closing_word(query),
            })
            logger.info(
                f"[{trace_id}] Workflow stop requested, confirming | type={workflow_type} | "
                f"phone={phone_number[-4:]} | trigger={query[:20]!r}"
            )
            return {
                "handled": True,
                "response": StructuredResponse.buttons_of(
                    f"You have an active {label.lower()} in progress. Would you like to "
                    "continue, or stop here?",
                    [InteractiveButton(id="continue", title="Continue"), InteractiveButton(id="stop", title="Stop")],
                ),
            }

        # A bare greeting ("hi") sent out of scope for the current step is
        # treated as an immediate restart, not a stop request needing
        # confirmation — it doesn't read as "I want to abandon this."
        if _is_greeting_word(query):
            complete_workflow(phone_number)

            if workflow_type == WORKFLOW_ONBOARDING:
                logger.info(
                    f"[{trace_id}] Onboarding interrupted and restarted | "
                    f"phone={phone_number[-4:]} | trigger={query[:20]!r}"
                )
                new_workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
                create_workflow(phone_number, new_workflow)
                from app.services.menu import build_onboarding_welcome_message
                return {"handled": True, "response": build_onboarding_welcome_message()}

            label = _WORKFLOW_LABELS.get(workflow_type, "This request")
            from app.database import get_customer_by_phone
            customer = get_customer_by_phone(phone_number)
            name = customer.get("full_name", "there") if customer else "there"
            logger.info(
                f"[{trace_id}] Workflow interrupted | type={workflow_type} | "
                f"phone={phone_number[-4:]} | trigger={query[:20]!r}"
            )
            return {
                "handled": True,
                "response": render_main_menu_list(
                    name, greeting=False, prefix=f"✅ {label} cancelled. Nothing was submitted or changed."
                ),
            }

        if _is_back_command(query) and workflow_type != WORKFLOW_TRANSFER:
            return _handle_back_for_workflow(workflow, phone_number)

        # Questions and general banking requests are allowed during every
        # workflow. Answer them without losing the active flow or asking the
        # customer to restart it. Data-entry/help questions for cheque and
        # loan workflows remain with their processors so they can explain the
        # exact missing fields.
        if _is_conversational_query(query):
            if not _is_allowed_for_workflow(workflow_type, query):
                return {
                    "handled": True,
                    "response": _workflow_boundary_message(workflow_type, workflow.get("step")),
                }
            if not _is_workflow_help_query(workflow_type, query):
                logger.info(
                    f"Workflow-related question | workflow={workflow_type} | phone={phone_number[-4:]}"
                )
                if is_llm_fallback_enabled():
                    answer = answer_side_question(query, workflow_type, workflow.get("step"), trace_id)
                    if answer:
                        hint = render_workflow_step_hint(workflow_type, workflow.get("step"))
                        logger.info(
                            f"[{trace_id}] Side question answered mid-workflow, resuming step | "
                            f"workflow={workflow_type} | phone={phone_number[-4:]}"
                        )
                        return {
                            "handled": True,
                            "response": f"{answer}\n\n{hint}" if hint else answer,
                        }
                # LLM unavailable/disabled/declined to answer — fall back to
                # the original behavior: let the router/LLM answer it next,
                # workflow state left untouched, resuming on the next message.
                return {"handled": False, "response": None, "reprocess_query": query}

        # An incomplete document workflow must not swallow unrelated requests.
        # Ask for explicit confirmation before abandoning the customer's data.
        pending = workflow.get("data", {}).get("pending_interrupt")
        if pending:
            # Older sessions may still contain this field from the previous
            # interruption-confirmation behavior. Workflow isolation no
            # longer uses it, so remove it before handling this message.
            from app.workflows.memory import clear_workflow_data
            clear_workflow_data(phone_number, "pending_interrupt")

        if (
            workflow_type in {WORKFLOW_CHEQUE, WORKFLOW_LOAN, WORKFLOW_KYC, WORKFLOW_TRANSFER}
            and not _is_current_workflow_input(workflow, query)
            and _looks_like_new_service_request(query, workflow_type)
        ):
            if is_llm_fallback_enabled():
                jump = detect_step_or_workflow_jump(query, workflow_type, workflow.get("step"), trace_id)
                if jump and jump.target_workflow:
                    label = _WORKFLOW_LABELS.get(workflow_type, "request")
                    target_label = _WORKFLOW_LABELS.get(jump.target_workflow, jump.target_workflow)
                    update_workflow_data(phone_number, {
                        "pending_stop_confirmation": True,
                        "pending_stop_was_closing": False,
                        "pending_jump_workflow": jump.target_workflow,
                        "pending_jump_query": query,
                    })
                    logger.info(
                        f"[{trace_id}] Workflow jump requested, confirming | from={workflow_type} | "
                        f"to={jump.target_workflow} | phone={phone_number[-4:]}"
                    )
                    return {
                        "handled": True,
                        "response": StructuredResponse.buttons_of(
                            f"You have an active {label.lower()} in progress. Would you like to "
                            f"continue that, or switch to {target_label.lower()} instead?",
                            [
                                InteractiveButton(id="continue", title="Continue"),
                                InteractiveButton(id="switch", title=f"Switch to {target_label}"[:20]),
                            ],
                        ),
                    }
            return {"handled": True, "response": _workflow_boundary_message(workflow_type, workflow.get("step"))}

        logger.info(
            f"[{trace_id}] Active workflow found | "
            f"type={workflow_type} | "
            f"step={workflow['step']} | "
            f"phone={phone_number[-4:]}"
        )

        if workflow_type == WORKFLOW_CHEQUE:

            return await self.cheque_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type == WORKFLOW_LOAN:

            return await self.loan_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type == WORKFLOW_KYC:

            return await self.kyc_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type == WORKFLOW_ONBOARDING:

            pending_service_query = workflow.get("data", {}).get("pending_service_query")

            result = await self.onboarding_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

            # Registration just finished successfully (no workflow left AND
            # a customer record now exists — cancelling/declining also
            # clears the workflow, so the customer check is what tells the
            # two apart) and the customer originally asked for a real
            # service before we made them register — resume that request
            # now instead of leaving them to ask again from scratch. See
            # registration_gate.py, where this gets stashed.
            if pending_service_query and get_workflow(phone_number) is None:
                from app.database import get_customer_by_phone
                if not get_customer_by_phone(phone_number):
                    return result
                resumed = self.start_requested(phone_number, pending_service_query, trace_id=trace_id)
                if resumed.get("handled"):
                    logger.info(
                        f"[{trace_id}] Resumed pending service after registration | "
                        f"phone={phone_number[-4:]}"
                    )
                    result["response"] = f"{result['response']}\n\n{resumed['response']}"

            return result

        elif workflow_type == WORKFLOW_TRANSFER:
            return self.transfer_handler.handle(workflow, phone_number, query, trace_id=trace_id)

        logger.warning(
            f"Unknown workflow type: {workflow_type}"
        )

        return {
            "handled": False,
            "response": None,
        }

    def start_requested(self, phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        """Start deterministic workflows without depending on an LLM intent call."""
        normalized = query.strip().lower()
        logger.info(
            f"[{trace_id}] Checking for deterministic workflow start | "
            f"phone={phone_number[-4:]} | query={query[:30]!r}"
        )
        menu_actions = {
            "1": "transfer",
            "2": "balance",
            "3": "transactions",
            "4": "cheque",
            "5": "cheque status",
            "6": "loan",
            "7": "kyc",
        }
        if normalized in menu_actions:
            action = menu_actions[normalized]
            if action == "transfer":
                if not has_transferable_balance(phone_number):
                    logger.info(f"[{trace_id}] Transfer blocked | reason=zero_balance | phone={phone_number[-4:]}")
                    return {"handled": True, "response": _insufficient_balance_message()}
                workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY)
                create_workflow(phone_number, workflow)
                return {"handled": True, "response": self.transfer_handler._beneficiary_prompt(phone_number)["response"]}
            if action == "cheque":
                workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
                create_workflow(phone_number, workflow)
                return {"handled": True, "response": render_cheque_deposit_started()}
            if action == "loan":
                workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
                create_workflow(phone_number, workflow)
                return {"handled": True, "response": loan_type_list_prompt(
                    "\U0001F4DD Let's get your loan application going! What kind of loan are you after?"
                )}
            if action == "kyc":
                workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
                create_workflow(phone_number, workflow)
                return {"handled": True, "response": render_kyc_update_started()}
            return {"handled": False, "response": None, "reprocess_query": f"check my {action}"}
        lookup_words = (
            "status", "list", "show", "my ", "all ", "details", "information",
            "progress", "submitted", "associated", "application", "applications",
            "what happened", "where is", "track",
        )
        is_lookup = any(word in normalized for word in lookup_words)
        if any(word in normalized for word in ("cheque", "check deposit", "deposit a check")) and not is_lookup:
            workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": "🧾 *Cheque deposit started*\n\nPlease upload a clear cheque image to continue.\n\nReply *Cancel* to stop.",
            }
        if any(word in normalized for word in ("loan", "borrow", "finance")) and not is_lookup:
            workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
            create_workflow(phone_number, workflow)
            # If the loan type was already stated in this same message
            # ("I'd like a personal loan"), skip straight past the "which
            # loan type?" step instead of asking again.
            loan_type = detect_loan_type_from_text(query)
            if loan_type:
                return self.loan_handler._select_type(workflow, phone_number, query, trace_id)
            return {"handled": True, "response": loan_type_list_prompt(
                "\U0001F4DD *Loan application started* — what kind of loan are you after?"
            )}
        if any(word in normalized for word in ("kyc", "know your customer", "update my details")):
            workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": (
                    "📄 *KYC update started*\n\nPlease upload a clear KYC form or document.\n\n"
                    "Required details: full name, date of birth, address, Aadhaar number, and PAN number.\n\nReply *Cancel* to stop."
                ),
            }
        if any(word in normalized for word in ("transfer", "send money", "pay someone", "make a payment")) and not is_lookup:
            if not has_transferable_balance(phone_number):
                logger.info(f"[{trace_id}] Transfer blocked | reason=zero_balance | phone={phone_number[-4:]}")
                return {"handled": True, "response": _insufficient_balance_message()}
            return start_transfer_from_text(phone_number, query, self.transfer_handler, trace_id)
        return {"handled": False, "response": None}


def _insufficient_balance_message() -> str:
    return render_insufficient_balance()


def _looks_like_new_service_request(query: str, workflow_type: str | None = None) -> bool:
    """A generic banking word alone isn't enough to conclude the customer
    wants to abandon their active workflow for something else — "account"
    and "transfer" are exactly the vocabulary a transfer's own beneficiary/
    amount answers naturally use ("account number 12345"), so matching
    them unconditionally used to reject perfectly valid workflow input with
    the boundary message instead of ever reaching the processor. Only a
    term that ISN'T already part of the active workflow's own topic (see
    _WORKFLOW_ON_TOPIC_TERMS, shared with _is_allowed_for_workflow below)
    counts as a genuine pivot to a different service."""
    text = query.strip().lower()
    on_topic = _WORKFLOW_ON_TOPIC_TERMS.get(workflow_type, ()) if workflow_type else ()
    return any(
        term in text and not any(topic_term in term or term in topic_term for topic_term in on_topic)
        for term in (
            "balance", "transaction", "statement", "loan", "kyc", "cheque", "menu",
            "help", "account", "transfer", "service",
        )
    )


def _is_current_workflow_input(workflow: dict[str, Any], query: str) -> bool:
    """Do not treat a field correction as a request to abandon its workflow."""
    if workflow.get("type") != WORKFLOW_LOAN:
        return False
    fields = (
        "account", "account number", "applicant", "name", "income", "monthly income",
        "salary", "employment", "amount", "loan amount", "requested", "tenure",
        "loan tenure", "purpose", "loan purpose",
    )
    return any(
        any(line.lstrip().lower().startswith(f"{field}:") for field in fields)
        for line in query.splitlines()
    )


def _is_conversational_query(query: str) -> bool:
    """Recognize questions that should be answered without interrupting a
    flow. Deliberately question-marker-led (not just "contains a banking
    word") — a workflow input value can easily contain a banking word
    ("send 500" while entering a transfer amount, an employer field
    reading "Bank of ..."), so only text that actually READS like a
    question gets this far. See _is_allowed_for_workflow below for the
    Task-10-follow-up widening (a genuine question about a DIFFERENT
    banking topic than the current workflow is now still answered, rather
    than rejected with the boundary message)."""
    text = query.strip().lower()
    if not text:
        return False
    question_markers = (
        "?", "what ", "what's", "why ", "how ", "which ", "where ", "can you", "could you",
        # Imperative lookup phrasing ("list my beneficiaries", "show my
        # saved payees") reads as a request, not a question, but must be
        # recognized the same way — otherwise it gets swallowed as literal
        # input by whatever step the workflow happens to be on (this is
        # exactly what let "List all my saved beneficiary" get stripped
        # and saved as a beneficiary's account number).
        "list ", "list my", "show me", "show my", "who are my", "who is my",
    )
    banking_terms = (
        "expense", "expenses", "spent", "spending", "trip", "goa", "travel",
        "balance", "transaction", "statement", "status", "details", "meaning",
        "help", "explain", "next step", "my account", "my loan", "my cheque",
        "beneficiar", "saved payee", "saved payees",
    )
    return any(marker in text for marker in question_markers) or any(term in text for term in banking_terms)


def _is_workflow_help_query(workflow_type: str, query: str) -> bool:
    """Leave field-specific questions with the workflow's precise explainer."""
    text = query.strip().lower()
    if workflow_type == WORKFLOW_CHEQUE:
        return any(term in text for term in (
            "error", "wrong", "mismatch", "missing", "payee", "amount", "clear image",
            "upload", "what happened",
        ))
    if workflow_type == WORKFLOW_LOAN:
        return any(term in text for term in (
            "required", "account", "applicant", "income", "salary", "employment", "amount",
            "tenure", "purpose", "form", "field",
        ))
    return False


# Shared by _is_allowed_for_workflow (keep a conversational question inside
# the active workflow's subject area) and _looks_like_new_service_request
# (decide whether workflow input that ISN'T phrased as a question is a
# pivot to a different service) — one table of what each workflow's own
# vocabulary is, so the two checks can't disagree with each other the way
# they used to (the boundary check didn't know "account" is normal transfer
# vocabulary and was rejecting valid "account number 12345" answers).
_WORKFLOW_ON_TOPIC_TERMS = {
    WORKFLOW_TRANSFER: (
        "transfer", "send", "pay", "payment", "beneficiar", "recipient", "amount",
        "account", "otp", "one time", "verification", "source", "recipient", "money",
    ),
    WORKFLOW_CHEQUE: (
        "cheque", "check", "deposit", "payee", "amount", "bank", "branch", "image",
        "upload", "status", "request", "error", "wrong", "missing", "mismatch",
    ),
    WORKFLOW_LOAN: (
        "loan", "borrow", "application", "account", "applicant", "income", "salary",
        "employment", "amount", "tenure", "purpose", "form", "status", "request", "field",
    ),
    WORKFLOW_KYC: (
        "kyc", "identity", "document", "aadhaar", "aadhar", "pan", "name", "address",
        "date of birth", "dob", "update", "field",
    ),
    WORKFLOW_ONBOARDING: (
        "register", "registration", "name", "aadhaar", "aadhar", "pan", "address",
        "document", "account", "profile", "field",
    ),
}


def _is_allowed_for_workflow(workflow_type: str, query: str) -> bool:
    """Keep conversational questions inside the workflow's subject area."""
    text = query.strip().lower()
    terms = _WORKFLOW_ON_TOPIC_TERMS
    if workflow_type in terms and any(term in text for term in terms[workflow_type]):
        return True
    if bool(re.fullmatch(r"\d+(?:\.\d+)?", text)):
        return True
    # Task 10 follow-up: a genuine banking question about a DIFFERENT topic
    # than the active workflow (e.g. "what's the loan interest rate" asked
    # mid-transfer) is still answered rather than rejected — only text with
    # no banking-domain content at all (small talk, general knowledge)
    # falls through to the boundary message below. This is only reached
    # for text _is_conversational_query already decided reads like a
    # question, so it doesn't divert ordinary field input (an amount, a
    # name, "yes"/"no") away from the workflow's own processor.
    return any(term in text for term in BANKING_DOMAIN_KEYWORDS)


def _workflow_boundary_message(workflow_type: str, step: str | None = None) -> str:
    """Task 10, Parts 9/10: explain the CURRENT step instead of the rigid
    "I can answer questions only about this request here." — the workflow
    is never restarted and the customer is never sent to the main menu."""
    return render_workflow_boundary_with_step(workflow_type, step)


def _is_greeting_word(query: str) -> bool:
    """
    Recognize a bare greeting/menu word (hi, hello, menu, help, ...) sent
    out of scope for the current workflow step. Reuses the same keyword set
    the registration gate uses to detect a fresh "hi" from an unregistered
    number, so the two behave consistently.
    """
    normalized = query.strip().lower().strip("!.? ")
    return normalized in GREETING_KEYWORDS


def _is_cancel_command(query: str) -> bool:
    """Recognize short stop commands consistently across every workflow."""
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    # Word-boundary matching (Task 10) — a naive substring check here
    # false-positived on ordinary banking questions: "spend" contains
    # "end", and "this"/"it" are common words, so "What did I spend
    # this month?" was being misread as a cancel command.
    stop_words = re.search(r"\b(cancel|stop|end)\b", text)
    action_words = re.search(
        r"\b(process|application|request|cheque|check|loan|transfer|this|it)\b", text
    )
    broad_stop = bool(stop_words and action_words)
    return broad_stop or text in {
        "cancel", "cancel it", "cancel this", "stop", "stop it", "exit",
        "quit", "end", "end this", "never mind", "no thanks",
    } or text.startswith("cancel ") or text.startswith("stop ") or text in {
        "please cancel", "please stop", "i want to cancel", "i dont want this",
        "i do not want this", "forget it", "leave it", "not interested",
        "dont continue", "do not continue", "no longer want this",
        "i want to stop", "i want to exit", "i want to quit", "id like to stop",
        "im done with this", "i am done with this", "i changed my mind",
        "ive changed my mind", "not now",
    }


def _is_closing_word(query: str) -> bool:
    """Recognize the customer signalling they're done with the
    conversation altogether ("bye", "thanks, that's all") — distinct from
    _is_cancel_command's "stop this operation" words, since the two need
    different follow-up: after stopping an active workflow, a closing word
    means the reply should be a goodbye, not the main menu (see
    pending_stop_was_closing in _resolve_pending_stop)."""
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    return text in {
        "bye", "goodbye", "good bye", "see you", "see ya", "thank you",
        "thanks", "thank you very much", "thanks a lot", "thankyou", "ty",
        "thats all", "thats it", "thats all for now", "nothing else",
        "no more questions", "im done", "i am done", "all done",
        "that will be all", "thatll be all", "im good", "i am good",
        "no thank you", "no more",
    }


_RESUME_RE = re.compile(r"\b(continue|resume|keep going|carry on|go ?ahead|proceed)\b", re.I)
_CONFIRM_STOP_RE = re.compile(r"\b(stop|cancel|end|quit|exit|switch)\b", re.I)


def _interpret_stop_or_continue(text: str) -> str | None:
    """Answer to "would you like to continue, or stop here?" — deliberately
    NOT app.workflows.nlu.interpret_confirmation: that treats "stop"/
    "cancel" as a generic denial (right for "reply YES to confirm or NO to
    cancel" prompts), which is backwards here since "stop" is the option
    being AGREED to, not rejected. "continue"/"proceed"/"go ahead" (and a
    bare yes) mean keep going; "stop"/"cancel"/"end"/"quit"/"exit" (and a
    bare no) mean actually stop."""
    normalized = re.sub(r"[^a-z ]", " ", text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    wants_continue = bool(_RESUME_RE.search(normalized))
    wants_stop = bool(_CONFIRM_STOP_RE.search(normalized))
    if wants_continue and not wants_stop:
        return "continue"
    if wants_stop and not wants_continue:
        return "stop"
    if normalized in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "y"}:
        return "continue"
    if normalized in {"no", "nope", "nah", "n"}:
        return "stop"
    return None


def _resolve_pending_stop(
    workflow: dict[str, Any],
    workflow_type: str,
    phone_number: str,
    query: str,
    trace_id: str,
    transfer_handler: Any = None,
) -> dict[str, Any]:
    """Act on the customer's answer to "do you want to stop your {X}, or
    continue?" (asked by the cancel/closing-word branch, and by the
    workflow-jump branch, in handle()). "continue" resumes at the exact
    step they were on; "stop" actually cancels — either ending the
    conversation/returning to the menu (plain cancel/closing-word trigger),
    or, if this confirmation was asked because the customer's message
    looked like a request for a DIFFERENT workflow (pending_jump_workflow),
    starting that workflow instead of just cancelling to the menu."""
    label = _WORKFLOW_LABELS.get(workflow_type, "request")
    answer = _interpret_stop_or_continue(query)
    jump_workflow = workflow.get("data", {}).get("pending_jump_workflow")
    jump_query = workflow.get("data", {}).get("pending_jump_query", "")

    if answer == "continue":
        update_workflow_data(phone_number, {
            "pending_stop_confirmation": False,
            "pending_stop_was_closing": False,
            "pending_jump_workflow": None,
            "pending_jump_query": None,
        })
        logger.info(f"[{trace_id}] Workflow stop declined, resuming | type={workflow_type} | phone={phone_number[-4:]}")
        hint = render_workflow_step_hint(workflow_type, workflow.get("step"))
        resume_text = f"No problem, let's continue with your {label.lower()}."
        return {"handled": True, "response": f"{resume_text}\n\n{hint}" if hint else resume_text}

    if answer == "stop":
        was_closing = bool(workflow.get("data", {}).get("pending_stop_was_closing"))
        complete_workflow(phone_number)
        logger.info(
            f"[{trace_id}] Workflow stop confirmed | type={workflow_type} | "
            f"phone={phone_number[-4:]} | closing={was_closing} | jump_to={jump_workflow or 'none'}"
        )
        if jump_workflow:
            started = start_workflow_directly(
                jump_workflow, phone_number, transfer_handler=transfer_handler,
                query=jump_query, trace_id=trace_id,
            )
            if started and started.get("handled"):
                return started
        if was_closing:
            return {"handled": True, "response": render_goodbye()}
        from app.database import get_customer_by_phone
        customer = get_customer_by_phone(phone_number)
        name = customer.get("full_name", "there") if customer else "there"
        return {
            "handled": True,
            "response": render_main_menu_list(
                name, greeting=False, prefix=f"✅ {label} cancelled. Nothing was submitted or changed."
            ),
        }

    return {
        "handled": True,
        "response": (
            f"Sorry, just to double check — would you like to continue with your "
            f"{label.lower()}, or stop here? Reply *continue* or *stop*."
        ),
    }


def _is_back_command(query: str) -> bool:
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    return text in {"back", "b", "go back", "please go back", "previous", "previous step"}


def _handle_back_for_workflow(workflow: dict[str, Any], phone_number: str) -> dict[str, Any]:
    """Move non-transfer workflows back one step before their processors run."""
    step = workflow.get("step")
    previous = {
        "CORRECT_CHEQUE": ("UPLOAD_CHEQUE", "🧾 Back to cheque upload. Please send a clear cheque image."),
        "UPLOAD_CHEQUE": (None, "🧾 You are already at the first cheque step. Please upload an image or reply *Cancel*."),
        "UPLOAD_LOAN_FORM": ("SELECT_LOAN_TYPE", "📝 Back to loan type. Reply 1 Personal, 2 Home, 3 Vehicle, or 4 Education."),
        "CONFIRM_LOAN": ("UPLOAD_LOAN_FORM", "📝 Back to your loan form. Please upload it or reply with corrections."),
        "SELECT_LOAN_TYPE": (None, "📝 You are already at the first loan step. Reply *Cancel* to stop."),
        "CONFIRM_KYC": ("UPLOAD_KYC_FORM", "📄 Back to KYC details. Please upload the document or reply with corrections."),
        "UPLOAD_KYC_FORM": (None, "📄 You are already at the first KYC step. Reply *Cancel* to stop."),
        "COLLECT_AADHAAR": (None, "🪪 You are already at the first registration step. Please upload a clear image of your Aadhaar card."),
        "COLLECT_PAN": ("COLLECT_AADHAAR", "🪪 Back to Aadhaar. Please upload the Aadhaar card image."),
        "CONFIRM_REGISTRATION": ("COLLECT_PAN", "🪪 Back to PAN. Please upload the PAN card image."),
        "SELECT_ACCOUNT_TYPE": ("CONFIRM_REGISTRATION", "🔎 Back to confirmation. Please review your registration details."),
    }.get(step, (None, "You are already at the first step. Reply *Cancel* to stop."))
    if previous[0]:
        set_workflow_step(phone_number, previous[0])
    return {"handled": True, "response": previous[1] + "\n\nReply *Back* or *Cancel* anytime."}
