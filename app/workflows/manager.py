from typing import Any
import re

from app.logger import get_logger
from app.workflows.constants import (
    WORKFLOW_ADD_ACCOUNT,
    WORKFLOW_CHEQUE,
    WORKFLOW_LOAN,
    WORKFLOW_KYC,
    WORKFLOW_ONBOARDING,
    WORKFLOW_TRANSFER,
    WORKFLOW_VIEW_TRANSACTIONS,
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
from app.conversation.responses.common import render_goodbye, render_main_menu_list, render_workflow_boundary_with_step, render_workflow_step_hint, with_nav_buttons
from app.conversation.responses.cheque import render_cheque_deposit_started
from app.conversation.responses.kyc import render_kyc_update_started
from app.conversation.intent.rules import BANKING_DOMAIN_KEYWORDS
from app.services.llm_understanding import (
    answer_side_question,
    detect_soft_decline,
    is_llm_fallback_enabled,
)
from app.conversation.renderer import InteractiveButton, StructuredResponse
from app.conversation.workflow_adapter import start_workflow_directly
from app.conversation.intent.llm_routing import classify_and_route_llm_sync
from app.conversation.intent.models import CONFIDENCE_HIGH, WORKFLOW_EXECUTING_INTENTS
from app.conversation.router import get_workflow_for_intent

from app.workflows.processors.cheque import ChequeWorkflowProcessor
from app.workflows.processors.loan import LOAN_TYPES, LoanWorkflowHandler, detect_loan_type_from_text, loan_type_list_prompt
from app.workflows.processors.kyc import KYCWorkflowHandler
from app.workflows.processors.onboarding import OnboardingWorkflowHandler, start_add_account_workflow
from app.workflows.processors.transfer import TransferWorkflowProcessor, has_transferable_balance, start_transfer_from_text
from app.workflows.processors.transactions import ViewTransactionsWorkflowHandler, start_view_transactions

logger = get_logger(__name__)

# Shared human-readable label per workflow type — used both when asking
# "do you want to stop your {label}?" and in the final cancellation
# message, so the two always agree with each other.
_WORKFLOW_LABELS = {
    WORKFLOW_CHEQUE: "Cheque deposit",
    WORKFLOW_TRANSFER: "Money transfer",
    WORKFLOW_LOAN: "Loan application",
    WORKFLOW_KYC: "KYC update",
    WORKFLOW_ADD_ACCOUNT: "Account creation",
    WORKFLOW_VIEW_TRANSACTIONS: "View transactions",
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
        self.transactions_handler = ViewTransactionsWorkflowHandler()

    def handle(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
        trace_id: str = "",
        intent_result: Any = None,
    ) -> dict[str, Any]:
        """
        Handle an active workflow.

        Returns:
        {
            "handled": True/False,
            "response": "..."
        }
        """
        query = str(query or "")

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

        # Context-aware cancellation: "never mind, apply for a loan for me"
        # must NOT read as pure cancellation just because it starts with a
        # cancel phrase — the rest of the sentence names a clear, different
        # request. Confirmed live (this is the exact reported regression):
        # _is_cancel_command()'s "never mind" match used to fire
        # unconditionally, before the generic switch check below ever got a
        # chance to see the loan request that followed it.
        #
        # The fix reuses the SAME already-computed intent_result and the
        # SAME confidence/workflow-mapping logic the generic switch check
        # below already applies — not a new keyword or a pairwise rule, and
        # not English-specific (once classify_workflow_request is replaced
        # by LLM-based classification in a future step, intent_result comes
        # from that instead, and this check keeps working unchanged). Only
        # when the message ALSO confidently names a different workflow does
        # cancellation defer to the switch path; a bare "never mind, don't
        # open the account" (naming no new request) still cancels exactly
        # as before.
        has_confident_switch_intent = (
            intent_result is not None
            and intent_result.intent in WORKFLOW_EXECUTING_INTENTS
            and intent_result.confidence >= CONFIDENCE_HIGH
            and not _is_current_workflow_input(workflow, query)
            and get_workflow_for_intent(intent_result.intent) not in (None, workflow_type)
        )

        # An explicit stop/cancel word, or a natural closing phrase ("thanks,
        # that's all", "bye"), must not silently abandon real in-progress
        # work (a beneficiary already picked, a document already uploaded).
        # Ask once, then act on whatever they reply via the branch above —
        # onboarding is the one exception (nothing of value is lost by
        # restarting it, and there's no menu to "continue" back into yet).
        #
        # The rigid regex above only catches an explicit cancel/stop/end
        # word — a natural phrasing like "I don't want to go with the
        # transfer right now" has none of those and was falling through
        # completely unrecognized, leaving the workflow silently active
        # with zero acknowledgment. The LLM fallback below catches that,
        # gated by a cheap keyword pre-filter so it's never on the hot
        # path for an ordinary field answer (an amount, an account
        # number) — see _looks_like_possible_decline.
        is_soft_decline = (
            is_llm_fallback_enabled()
            and _looks_like_possible_decline(query)
            and detect_soft_decline(query, workflow_type, trace_id)
        )
        if (_is_cancel_command(query) or _is_closing_word(query) or is_soft_decline) and not has_confident_switch_intent:
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

        # Generic, any-workflow-to-any-workflow switch: reuses
        # has_confident_switch_intent (computed above, before the
        # cancellation check, so a message like "never mind, apply for a
        # loan for me" resolves as a switch rather than a cancellation) —
        # the SAME intent_result app/conversation/manager.py already
        # computed for this message via classify_intent() (before it even
        # knew a workflow was active), not a second, separate
        # classification, and not a per-workflow-pair keyword table. This
        # runs BEFORE the older, narrower jump detection below (which stays
        # as a fallback for phrasing the fast rules don't catch) and before
        # the free-text question/side-answer handling, since a clear new
        # request should win over both.
        #
        # CONFIDENCE_HIGH matches the exact bar a fresh (no-workflow)
        # START_WORKFLOW decision already requires (see router.py) — this
        # doesn't lower the bar for starting a financial workflow, it only
        # extends the SAME bar to also apply while a different one is
        # active.
        if has_confident_switch_intent:
            target_workflow = get_workflow_for_intent(intent_result.intent)
            switched = _switch_workflow(
                phone_number, workflow_type, target_workflow, query, trace_id, self.transfer_handler
            )
            if switched is not None:
                return switched

        # Questions and general banking requests are allowed during every
        # workflow. Answer them without losing the active flow or asking the
        # customer to restart it. Data-entry/help questions for cheque and
        # loan workflows remain with their processors so they can explain the
        # exact missing fields.
        if _is_conversational_query(query):
            if not _is_allowed_for_workflow(workflow_type, query):
                if workflow_type == WORKFLOW_TRANSFER:
                    update_workflow_data(phone_number, {
                        "pending_stop_confirmation": True,
                        "pending_stop_was_closing": False,
                        "pending_jump_workflow": None,
                        "pending_jump_query": query,
                    })
                    logger.info(
                        f"[{trace_id}] Transfer interruption confirmed, awaiting continue/stop | "
                        f"phone={phone_number[-4:]} | query={query[:30]!r}"
                    )
                    return {
                        "handled": True,
                        "response": StructuredResponse.buttons_of(
                            "You have an active money transfer in progress. Would you like to continue, or stop here?",
                            [InteractiveButton(id="continue", title="Continue"), InteractiveButton(id="stop", title="Stop")],
                        ),
                    }
                return {
                    "handled": True,
                    "response": _workflow_boundary_message(workflow_type, workflow.get("step")),
                }
            if not _is_workflow_help_query(workflow_type, query):
                logger.info(
                    f"Workflow-related question | workflow={workflow_type} | phone={phone_number[-4:]}"
                )
                # answer_side_question has NO tools bound — it's a fast,
                # general-knowledge-only shortcut, and is explicitly told
                # to decline (reply NONE) anything needing real customer
                # data so THAT falls through to reprocess_query below and
                # gets answered by the real LLM+tools agent instead. In
                # practice the model doesn't reliably comply — "get my
                # balance" mid-transfer got a hallucinated "check the
                # mobile app" non-answer instead of declining, even though
                # the real number was already shown earlier in the same
                # conversation. Rather than trust that self-judgment every
                # time, skip it outright for anything that looks like a
                # personal-data lookup — go straight to the real agent,
                # which actually has the tools to answer correctly.
                if is_llm_fallback_enabled() and not _looks_like_data_request(query):
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
        elif (
            workflow_type in {WORKFLOW_CHEQUE, WORKFLOW_LOAN, WORKFLOW_KYC, WORKFLOW_TRANSFER}
            and not _is_current_workflow_input(workflow, query)
            and not _looks_like_protocol_id(query)
        ):
            # _is_conversational_query() above is deliberately English-
            # question-marker-led ("?", "what", "how", ...), so a genuine
            # side question with no "?" and no English question word --
            # native script ("நா bank ఖాతాలో ఎంత...") OR fully-ASCII
            # romanized text ("Naa bank khatalo entha dabbu undi") -- fails
            # it completely and used to fall straight through to the step
            # processor as literal field input. Confirmed live via a real
            # test conversation (scripts/_real_log_cases.json).
            #
            # This ALSO absorbs what used to be a separate pivot-detection
            # call further down in this method (Step 6/7): one LLM
            # consultation now answers both "is this a side question?" and
            # "does this want a different workflow?" -- two separate call
            # sites here would have meant two LLM calls for the same
            # message once Step 7 removed the keyword pre-filter that used
            # to (accidentally) keep them from both firing together.
            #
            # Deliberately unconditional (no LLM_FALLBACK_ENABLED gate,
            # unlike answer_side_question/detect_soft_decline below/above,
            # which stay opt-in -- they're a separate concern this
            # migration didn't validate). This mechanism specifically WAS
            # validated this session (scripts/shadow_eval.py's 101-case
            # corpus + real traffic replay) and is the actual target of
            # this migration step, so it is now the live, authoritative
            # behavior rather than a shadow/opt-in one. Gating this with a
            # keyword/script/length heuristic instead of just the
            # structural guards above would reintroduce a differently-
            # shaped blind spot -- not a new keyword table.
            decision = classify_and_route_llm_sync(query, workflow_type, workflow.get("step"), trace_id)

            if decision and decision.action in ("TOOL", "RAG") and decision.certainty != "low":
                # certainty != "low" mirrors the same gate the SWITCH branch
                # below already applies (there: == "high"). Confirmed live
                # this was a real gap: "send 500" mid-transfer at
                # SELECT_BENEFICIARY (an ordinary amount-shaped answer, not
                # a question) got classified TOOL at certainty="low" by the
                # LLM itself -- diverting it as a side question would have
                # silently swallowed a real field answer instead of letting
                # the transfer processor parse it.
                logger.info(
                    f"[{trace_id}] Non-English/unmarked side question recognized via LLM fallback | "
                    f"workflow={workflow_type} | phone={phone_number[-4:]}"
                )
                return {"handled": False, "response": None, "reprocess_query": query}

            if decision and decision.action == "SWITCH":
                candidate = decision.resolved_target_workflow()
                if candidate and candidate != workflow_type:
                    label = _WORKFLOW_LABELS.get(workflow_type, "request")
                    target_label = _WORKFLOW_LABELS.get(candidate, candidate)
                    update_workflow_data(phone_number, {
                        "pending_stop_confirmation": True,
                        "pending_stop_was_closing": False,
                        "pending_jump_workflow": candidate,
                        "pending_jump_query": query,
                    })
                    logger.info(
                        f"[{trace_id}] Workflow jump requested, confirming | from={workflow_type} | "
                        f"to={candidate} | phone={phone_number[-4:]}"
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
            # Not recognized as a side question or a genuine pivot
            # (CONTINUE/CORRECT/CANCEL/CLARIFY/low certainty/no decision) --
            # fall through to normal step-processor handling for this
            # workflow, exactly as before either of these two fixes. This
            # is deliberately NOT a boundary-message rejection: once Step 7
            # removed the keyword pre-filter that used to narrowly gate
            # this check, treating every non-match as an off-topic boundary
            # violation would have wrongly rejected ordinary free-text
            # field answers the LLM correctly recognized as CONTINUE.

        # An incomplete document workflow must not swallow unrelated requests.
        # Ask for explicit confirmation before abandoning the customer's data.
        pending = workflow.get("data", {}).get("pending_interrupt")
        if pending:
            # Older sessions may still contain this field from the previous
            # interruption-confirmation behavior. Workflow isolation no
            # longer uses it, so remove it before handling this message.
            from app.workflows.memory import clear_workflow_data
            clear_workflow_data(phone_number, "pending_interrupt")

        logger.info(
            f"[{trace_id}] Active workflow found | "
            f"type={workflow_type} | "
            f"step={workflow['step']} | "
            f"phone={phone_number[-4:]}"
        )

        if workflow_type == WORKFLOW_CHEQUE:

            return self.cheque_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type == WORKFLOW_LOAN:

            return self.loan_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type == WORKFLOW_KYC:

            return self.kyc_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )

        elif workflow_type in (WORKFLOW_ONBOARDING, WORKFLOW_ADD_ACCOUNT):

            # pending_service_query is only ever stashed by registration_gate.py
            # for a genuinely unregistered customer (see below) — an
            # add-account workflow never sets it, so this whole resume
            # branch is naturally a no-op for WORKFLOW_ADD_ACCOUNT.
            pending_service_query = workflow.get("data", {}).get("pending_service_query")

            result = self.onboarding_handler.handle(
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

        elif workflow_type == WORKFLOW_VIEW_TRANSACTIONS:
            return self.transactions_handler.handle(workflow, phone_number, query, trace_id=trace_id)

        logger.warning(
            f"Unknown workflow type: {workflow_type}"
        )

        return {
            "handled": False,
            "response": None,
        }

    def start_requested(self, phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        """Start deterministic workflows without depending on an LLM intent call."""
        query = str(query or "")
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
            "8": "create_account",
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
            if action == "create_account":
                return start_add_account_workflow(phone_number, trace_id)
            if action == "transactions":
                return start_view_transactions(phone_number, trace_id)
            return {"handled": False, "response": None, "reprocess_query": f"check my {action}"}
        if normalized in LOAN_TYPES:
            # A tap on the "Choose loan type" list row, arriving with no
            # active workflow to interpret it. WhatsApp interactive
            # messages stay tappable forever, so a customer naturally
            # reuses this same list later — to try a different type after
            # already picking one, or after already finishing that
            # application entirely — and that reply has nowhere to land
            # (confirmed live: it fell through to a generic "I'm here to
            # help" message instead). The row id already fully states
            # what was chosen, so treat it exactly like typing "I'd like a
            # <type> loan" — start a fresh application with that type
            # instead of dead-ending.
            logger.info(
                f"[{trace_id}] Loan-type tap with no active workflow, starting fresh | "
                f"phone={phone_number[-4:]} | loan_type={LOAN_TYPES[normalized]}"
            )
            workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
            create_workflow(phone_number, workflow)
            return self.loan_handler._select_type(workflow, phone_number, normalized, trace_id)
        # A destructive verb ("delete my KYC request") must never fall
        # through to a keyword-triggered workflow *start* just because the
        # rest of the sentence contains "kyc"/"cheque"/"loan" — confirmed
        # live: "Delete my KYC request." was silently force-starting a new
        # KYC upload instead of saying this isn't something the app
        # supports. None of these operations exist anywhere in this
        # codebase (no delete/cancel-after-submit workflow for any
        # request type), so this is never a false negative.
        if any(word in normalized for word in ("delete", "remove", "erase")):
            return {
                "handled": True,
                "response": (
                    "I'm not able to delete or remove a submitted request through chat. "
                    "Please contact support for that. Is there anything else I can help with?"
                ),
            }
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
        if any(word in normalized for word in ("kyc", "know your customer", "update my details")) and not is_lookup:
            workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": (
                    "📄 *KYC update started*\n\nPlease upload a clear photo of one of: Aadhaar card, "
                    "PAN card, Passport, Voter ID, or Driving Licence.\n\nReply *Cancel* to stop."
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


# REMOVED (Step 7 of the LLM-first routing migration): _looks_like_new_service_request()
# and _looks_non_ascii(), the two English-keyword-ish pre-filters that used
# to gate whether a pivot check even ran (see the jump-detection block
# above, which now runs classify_and_route_llm_sync() directly behind only
# structural guards -- workflow type, not-already-literal-field-input, and
# LLM_FALLBACK_ENABLED). Both were confirmed live to miss real pivots
# (scripts/shadow_eval.py's en_te_mixed_switch case) since gating an LLM
# understanding step with a semantic keyword guess just reintroduces the
# same blind spot one layer up. _WORKFLOW_ON_TOPIC_TERMS below is still
# used by _is_allowed_for_workflow() -- not removed.


def _looks_like_protocol_id(query: str) -> bool:
    """Button/list taps and other menu-generated ids ("lt_home",
    "acct_yes") are a fixed machine protocol, not natural language -- they
    must never reach an LLM classification call (button/list/menu/digit
    handling is required to stay deterministic). Confirmed live: a real
    "lt_home" loan-type tap reaching the mid-workflow LLM check produced
    flaky, non-deterministic results because there is no natural-language
    meaning for the model to actually classify.

    A single whitespace-free token containing an underscore is the
    structural signature every such id already shares in this codebase
    (WhatsApp interactive replies are always exactly one id, never a
    sentence, and every namespaced id here uses "prefix_value") -- this is
    a protocol-shape check, not a keyword table: it says nothing about
    what the id means, only that it isn't prose."""
    text = query.strip()
    return bool(text) and " " not in text and "_" in text


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


# Used by _is_allowed_for_workflow (keep a conversational question inside
# the active workflow's subject area) — one table of what each workflow's
# own vocabulary is. Previously also shared with the now-removed
# _looks_like_new_service_request() (see the REMOVED note above
# _is_current_workflow_input) — kept here since _is_allowed_for_workflow
# still needs it.
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
    WORKFLOW_ADD_ACCOUNT: (
        "account", "aadhaar", "aadhar", "pan", "address", "document", "profile",
        "field", "savings", "current", "salary",
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


def _workflow_boundary_message(workflow_type: str, step: str | None = None) -> StructuredResponse:
    """Task 10, Parts 9/10: explain the CURRENT step instead of the rigid
    "I can answer questions only about this request here." — the workflow
    is never restarted and the customer is never sent to the main menu.

    A customer who lands here again and again (an off-topic or ambiguous
    reply, repeatedly) previously had no way out except knowing the exact
    word to type — now Back/Cancel/Main Menu are tappable buttons on this
    same message, not just text in the hint. See with_nav_buttons()."""
    return with_nav_buttons(render_workflow_boundary_with_step(workflow_type, step))


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
    if any(ord(ch) > 127 for ch in query):
        # The a-z stripping below discards every non-Latin character, so a
        # compound message like "never mind, मुझे लोन चाहिए" would otherwise
        # collapse to exactly "never mind" and match the broad "never mind"
        # substring check below even though a real, different request
        # follows in native script -- the same class of bug just fixed in
        # classify_hard_navigation() (app/conversation/intent/rules.py) for
        # the pre-workflow case. This rule was never able to recognize a
        # PURE native-script cancel phrase anyway (every pattern below is
        # English), so deferring here loses nothing that used to work --
        # it only stops a false-positive on content this function can't
        # read. Later mechanisms (the mid-workflow LLM-based switch check
        # a few lines below in handle(), when enabled) get a real chance to
        # understand the full sentence instead.
        return False
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    # Word-boundary matching (Task 10) — a naive substring check here
    # false-positived on ordinary banking questions: "spend" contains
    # "end", and "this"/"it" are common words, so "What did I spend
    # this month?" was being misread as a cancel command.
    stop_words = re.search(r"\b(cancel|stop|end)\b", text)
    action_words = re.search(
        r"\b(process|application|request|cheque|check|loan|transfer|this|it)\b", text
    )
    # "never mind" alone is already in the exact-match set below, but only
    # when it's the ENTIRE message — "never mind, don't open the account"
    # (confirmed live, scripts/shadow_eval.py's acct_cancel case) has more
    # text after it and fell through both checks. "never mind" is an
    # unambiguous abandonment signal in this context regardless of what
    # follows, and — unlike a financial confirmation — a false positive
    # here only means re-asking, not an unauthorized action, so a plain
    # substring match is an acceptable, low-risk widening.
    broad_stop = bool((stop_words and action_words) or "never mind" in text)
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


_DATA_REQUEST_RE = re.compile(
    r"\b(my|the)\s+(balance|account|accounts|transaction|transactions|statement|"
    r"beneficiar\w*|payee\w*|cheque\w*|check\w*|loan\w*|kyc|transfer\w*|spend\w*|"
    r"expense\w*)\b|"
    r"\b(check|show|list|get|see|view)\s+(my|the)\b",
    re.I,
)


def _looks_like_data_request(text: str) -> bool:
    """A message that reads like a request for the customer's OWN real
    data (balance, transactions, beneficiaries, statuses, ...) — must
    never be handed to answer_side_question (no tools, general knowledge
    only) even when LLM fallback is enabled; see its call site above for
    why. Deliberately broad/cheap (a plain keyword regex, not an LLM
    call) — a false positive here just means this particular question
    goes through the slightly slower but reliable real-agent path
    instead of the fast general-knowledge one; a false negative would
    mean a real-data question risks getting a hallucinated non-answer,
    which is the actual bug this exists to prevent."""
    return bool(_DATA_REQUEST_RE.search(text.lower()))


_POSSIBLE_DECLINE_RE = re.compile(
    r"\b(don'?t|dont|not|never ?mind|no more|hold on|hold off|"
    r"wait|later|forget it|changed my mind|another time|some other time)\b",
    re.I,
)


def _looks_like_possible_decline(text: str) -> bool:
    """Cheap pre-filter before the expensive LLM soft-decline check
    (detect_soft_decline, a ~1-2s reasoning-model call) — only messages
    containing a negation/hesitation word are worth asking the LLM about
    at all. Without this, every ordinary field answer during a workflow
    (an amount, an account number, a name) would pay that latency just to
    get "no" back. False positives here just mean one wasted LLM call
    that (correctly) says "not a decline" — false negatives mean this
    fix doesn't catch a genuinely oddly-phrased decline, same limitation
    _is_cancel_command already accepted for its own keyword set."""
    return bool(_POSSIBLE_DECLINE_RE.search(text.lower()))


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
        if workflow_type == WORKFLOW_TRANSFER and jump_query and not jump_workflow:
            return {
                "handled": True,
                "response": "I will proceed with current request before addressing new request.",
            }
        hint = render_workflow_step_hint(workflow_type, workflow.get("step"))
        resume_text = f"No problem, let's continue with your {label.lower()}."
        return {"handled": True, "response": f"{resume_text}\n\n{hint}" if hint else resume_text}

    if answer == "stop":
        was_closing = bool(workflow.get("data", {}).get("pending_stop_was_closing"))
        pending_query = jump_query if not jump_workflow else None
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
        if pending_query:
            return {"handled": False, "response": None, "reprocess_query": pending_query}
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
        "CONFIRM_LOAN_ACCOUNT": ("SELECT_LOAN_TYPE", "📝 Back to loan type. Reply 1 Personal, 2 Home, 3 Vehicle, or 4 Education."),
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


def _switch_workflow(
    phone_number: str,
    from_workflow: str,
    to_workflow: str,
    query: str,
    trace_id: str,
    transfer_handler: Any,
) -> dict[str, Any] | None:
    """Abandon `from_workflow` and start `to_workflow` in its place — the
    one, generic, any-to-any mechanism behind WorkflowManager.handle()'s
    switch check above. Reuses start_workflow_directly (app/conversation/
    workflow_adapter.py), the SAME starter a fresh no-active-workflow
    START_WORKFLOW decision already uses for every workflow type, so
    there is no second, workflow-type-specific starting path to keep in
    sync.

    Clears the structured workflow record only — the conversation's
    session history and ConversationContext are untouched by this call,
    so useful context (what the customer already said) isn't lost, even
    though the old workflow's in-progress, unconfirmed answers are. This
    matches the existing behavior of every other workflow-abandoning path
    in this file (the greeting-word restart, the confirmed-stop path) —
    nothing is ever committed until a workflow's own final confirmation
    step, so abandoning mid-flow loses no real banking action, only
    not-yet-submitted form state.

    Returns None (caller falls through to the existing behavior) if the
    target workflow type isn't one start_workflow_directly can start —
    defensive; every WORKFLOW_EXECUTING_INTENTS value maps to a type it
    handles today.

    Deliberately does NOT clear `from_workflow` up front. start_workflow_directly's
    own create_workflow() call (when it actually starts one) already
    overwrites the same per-phone Redis record, so a genuine switch needs
    no separate delete — and some starters legitimately answer without
    creating a workflow at all (insufficient balance for a transfer, no
    account types left to add): clearing first would abandon the old
    workflow even on THAT path, losing real in-progress state for no
    reason. Checking the actual post-call state instead keeps the old
    workflow fully intact whenever no new one was actually created."""
    started = start_workflow_directly(
        to_workflow, phone_number, transfer_handler=transfer_handler, query=query, trace_id=trace_id,
    )
    if started is None or not started.get("handled"):
        return None

    new_workflow = get_workflow(phone_number)
    if not new_workflow or new_workflow.get("type") != to_workflow:
        # Answered informationally without starting a workflow (e.g.
        # "insufficient balance" / "no account types left") -- nothing
        # switched, so no "pausing" note and the old workflow (if any) is
        # still exactly as it was.
        return started

    from_label = _WORKFLOW_LABELS.get(from_workflow, "previous request")
    logger.info(
        f"[{trace_id}] Workflow switched | from={from_workflow} | to={to_workflow} | "
        f"phone={phone_number[-4:]} | trigger={query[:30]!r}"
    )
    switch_note = f"No problem — pausing your {from_label.lower()} for this.\n\n"
    started["response"] = _prefix_response_text(started["response"], switch_note)
    return started


def _prefix_response_text(response: Any, prefix: str) -> Any:
    """Prepend `prefix` to a response's visible text, whether it's a plain
    string or a StructuredResponse (list/buttons) — used only by
    _switch_workflow so the "pausing your X" note survives regardless of
    which shape the target workflow's own starter happens to return."""
    if isinstance(response, str):
        return prefix + response
    if isinstance(response, StructuredResponse):
        response.text = prefix + response.text
        return response
    return response
