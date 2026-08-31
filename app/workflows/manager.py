from typing import Any, Optional
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
from app.conversation.responses.transfer import render_insufficient_balance
from app.conversation.responses.common import render_goodbye, render_main_menu_list, render_workflow_boundary_with_step, render_workflow_step_hint, with_nav_buttons
from app.conversation.responses.cheque import render_cheque_deposit_started
from app.conversation.responses.kyc import render_kyc_update_started
from app.conversation.renderer import InteractiveButton, StructuredResponse
from app.conversation.workflow_adapter import start_workflow_directly
from app.conversation.intent.llm_routing import LLMRoutingDecision, classify_and_route_llm_sync

from app.workflows.processors.cheque import ChequeWorkflowProcessor
from app.workflows.processors.loan import LOAN_TYPES, LoanWorkflowHandler, loan_type_list_prompt
from app.workflows.processors.kyc import KYCWorkflowHandler
from app.workflows.processors.onboarding import OnboardingWorkflowHandler, start_add_account_workflow
from app.workflows.processors.transfer import TransferWorkflowProcessor, has_transferable_balance
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

    Whether a mid-workflow message should switch/cancel/answer a side
    question, or genuinely continue/correct the active step, is decided
    ONCE by the LLM router (app/conversation/intent/llm_routing.py) —
    either by app/conversation/manager.py before this class is ever
    called, or (only when no decision was computed upstream, e.g. this
    class is invoked directly) lazily here via classify_and_route_llm_sync.
    This class itself never runs a second, independent semantic
    classifier — only literal protocol checks (button/list/menu ids,
    digit taps, exact cancel/back phrases) and the deterministic workflow
    step processors.
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
        llm_decision: Optional[LLMRoutingDecision] = None,
    ) -> dict[str, Any]:
        """
        Handle an active workflow.

        `llm_decision`, if given, is the SAME single LLM routing decision
        app/conversation/manager.py already computed for this message this
        turn (or None if a deterministic pre-filter/protocol shortcut made
        a call unnecessary) — never a second, independent classification.

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
        # This is a literal button-tap reply ("continue"/"stop"/"switch"),
        # not a semantic classification — no LLM call needed.
        if workflow.get("data", {}).get("pending_stop_confirmation"):
            return _resolve_pending_stop(workflow, workflow_type, phone_number, query, trace_id, self.transfer_handler)

        # A literal protocol/field input for the CURRENT workflow (a
        # document upload, a button/list tap id, a bare digit, a bare
        # yes/no/confirm answering an active CONFIRM_* step, or a value
        # shaped like the field this exact step is collecting) always
        # belongs to the active step's own processor — never diverted, and
        # never needs an LLM call to know that. This keeps the hot,
        # high-volume document/data-entry workflows (onboarding, KYC,
        # cheque, loan field collection) at zero extra LLM calls, exactly
        # like before this migration; only a genuine free-text reply
        # (a name, an address, a real side question) pays for the single
        # LLM routing call below.
        is_protocol_input = (
            parsed_document is not None
            or _looks_like_protocol_id(query)
            or _is_current_workflow_input(workflow, query)
            or bool(re.fullmatch(r"\d+(?:[.,]\d+)?", query.strip()))
            or (
                (workflow.get("step") or "").upper().startswith("CONFIRM")
                and re.sub(r"[^a-z]", "", query.strip().lower()) in {"yes", "y", "confirm", "no", "n"}
            )
        )

        # The LLM-based switch/cancel/side-question detection below applies
        # to every real, workflow-owning operation a customer can switch
        # to/from ("this must work for every supported workflow
        # combination") — cheque/loan/kyc/transfer/add_account. Onboarding
        # (first-time registration) and view_transactions (a single-shot
        # lookup, not a multi-step flow) are excluded: registration is a
        # mandatory prerequisite gate a customer cannot "switch away from"
        # into another operation (registration_gate.py already owns that
        # boundary), and a bare field answer there (an account type, a
        # name) must never be second-guessed as a pivot — those two types
        # only ever get the deterministic checks below (menu/restart word,
        # literal cancel, back) before falling through to their processor.
        is_llm_eligible_workflow = workflow_type in (
            WORKFLOW_CHEQUE, WORKFLOW_LOAN, WORKFLOW_KYC, WORKFLOW_TRANSFER, WORKFLOW_ADD_ACCOUNT,
        )

        if not is_protocol_input and is_llm_eligible_workflow and _is_back_command(query) and workflow_type != WORKFLOW_TRANSFER:
            # Literal "back" is hard-navigation protocol -- deterministic,
            # checked before paying for an LLM call.
            return _handle_back_for_workflow(workflow, phone_number)

        if not is_protocol_input and is_llm_eligible_workflow:
            # A literal "menu"/"start over" word, or an explicit, short
            # cancel/closing phrase, is hard-navigation protocol (exact
            # phrase, deterministic — see rules._MENU_WORDS/_RESTART_WORDS)
            # — checked before paying for an LLM call. A message this
            # short and literal can't simultaneously be naming a different,
            # confident workflow request, so no switch-intent check is
            # needed to safely act on it immediately.
            is_menu_or_restart = _is_menu_or_restart_word(query)
            is_deterministic_cancel = _is_cancel_command(query) or _is_closing_word(query)

            # Get (or reuse) the single LLM routing decision for this
            # message. app/conversation/manager.py already computes one
            # for almost every non-protocol message before calling this
            # method — llm_decision is only None here for a caller that
            # invokes WorkflowManager.handle() directly (e.g. some tests),
            # or a literal menu/restart/cancel/back word that doesn't need
            # one.
            needed_llm_call = llm_decision is None and not is_menu_or_restart and not is_deterministic_cancel
            if needed_llm_call:
                llm_decision = classify_and_route_llm_sync(query, workflow_type, workflow.get("step"), trace_id)

            # A bare greeting ("hi"), or a literal "menu"/"start over"
            # word, sent out of scope for the current step is treated as
            # an immediate restart, not a stop request needing
            # confirmation — it doesn't read as "I want to abandon this."
            if is_menu_or_restart or (llm_decision is not None and llm_decision.action == "GREETING"):
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

            has_confident_switch_intent = (
                llm_decision is not None
                and llm_decision.action in ("SWITCH", "START_WORKFLOW")
                and llm_decision.certainty == "high"
                and llm_decision.resolved_target_workflow() not in (None, workflow_type)
            )
            is_llm_cancel = llm_decision is not None and llm_decision.action == "CANCEL"

            # An explicit stop/cancel word, a natural closing phrase
            # ("thanks, that's all", "bye"), or the LLM recognizing a
            # natural-language decline ("I don't want to do this right
            # now") must not silently abandon real in-progress work (a
            # beneficiary already picked, a document already uploaded).
            # Ask once, then act on whatever they reply via the branch
            # above — onboarding is the one exception (nothing of value is
            # lost by restarting it, and there's no menu to "continue" back
            # into yet). has_confident_switch_intent only ever applies to
            # the LLM-decided case (is_deterministic_cancel messages are
            # too short/literal to also carry a countervailing switch, by
            # construction — see the check above).
            if (is_deterministic_cancel or is_llm_cancel) and not has_confident_switch_intent:
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

            # (a literal "back" was already handled above, before the LLM
            # call, for every eligible workflow type)

            # Generic, any-workflow-to-any-workflow switch — the single LLM
            # decision computed above (or upstream) is the sole source of
            # truth; CONFIDENCE_HIGH-equivalent ("certainty == high") is
            # the same bar a fresh (no-workflow) START_WORKFLOW decision
            # already requires.
            if has_confident_switch_intent:
                target_workflow = llm_decision.resolved_target_workflow()
                switched = _switch_workflow(
                    phone_number, workflow_type, target_workflow, query, trace_id, self.transfer_handler
                )
                if switched is not None:
                    return switched

            if llm_decision is not None and llm_decision.action in ("TOOL", "RAG") and llm_decision.certainty != "low":
                # A genuine side question (balance/transactions/status/RAG)
                # asked mid-workflow — answer it via the real LLM+tools
                # agent (which has real data access) without losing the
                # active flow. certainty != "low" mirrors the switch bar
                # above: an ordinary field answer ("500" mid-transfer)
                # sometimes gets classified TOOL at low certainty by the
                # model itself — diverting that would silently swallow a
                # real field answer instead of letting the step processor
                # parse it.
                logger.info(
                    f"[{trace_id}] Side question recognized via LLM router | "
                    f"workflow={workflow_type} | phone={phone_number[-4:]}"
                )
                return {"handled": False, "response": None, "reprocess_query": query}

            if llm_decision is not None and llm_decision.action == "SWITCH":
                candidate = llm_decision.resolved_target_workflow()
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
            if llm_decision is not None and llm_decision.action == "OUT_OF_SCOPE" and workflow_type == WORKFLOW_TRANSFER:
                # Transfer has no useful per-step "explain this" hint the
                # way document workflows do — a genuinely off-topic
                # message here asks continue-or-stop instead, matching the
                # existing pending-stop confirmation UX.
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

            if llm_decision is not None and llm_decision.action in ("OUT_OF_SCOPE", "CLARIFY"):
                # Genuinely unrelated to banking (OUT_OF_SCOPE), or a
                # "what should I do?"-style request for help with the
                # current step the model isn't confident enough to answer
                # as a real TOOL/RAG question (CLARIFY) — either way,
                # explain the current step rather than silently feeding
                # the text to the step processor as if it were a field
                # answer, and never abandon or restart the workflow.
                return {
                    "handled": True,
                    "response": with_nav_buttons(render_workflow_boundary_with_step(workflow_type, workflow.get("step"))),
                }

            if needed_llm_call and llm_decision is None:
                # The LLM call was attempted (this wasn't a literal
                # menu/cancel word) and failed/unavailable — let the
                # router/LLM+tools agent try next turn rather than
                # silently feeding unrelated text to the step processor
                # as if it were a field answer.
                return {"handled": False, "response": None, "reprocess_query": query}

            # Not recognized as a side question, a pivot, or out-of-scope
            # (CONTINUE/CORRECT/CLARIFY/low certainty) — fall through to
            # normal step-processor handling for this workflow.

        elif not is_protocol_input:
            # A short, linear, single-purpose workflow (onboarding,
            # add_account, view_transactions) — only the deterministic,
            # zero-LLM-call checks apply; an ordinary field answer here
            # (an account type, a name) is never second-guessed as a
            # pivot. See is_llm_eligible_workflow's comment above.
            if _is_menu_or_restart_word(query):
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
                return {
                    "handled": True,
                    "response": render_main_menu_list(
                        name, greeting=False, prefix=f"✅ {label} cancelled. Nothing was submitted or changed."
                    ),
                }

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
                return {
                    "handled": True,
                    "response": StructuredResponse.buttons_of(
                        f"You have an active {label.lower()} in progress. Would you like to "
                        "continue, or stop here?",
                        [InteractiveButton(id="continue", title="Continue"), InteractiveButton(id="stop", title="Stop")],
                    ),
                }

            if _is_back_command(query) and workflow_type != WORKFLOW_TRANSFER:
                return _handle_back_for_workflow(workflow, phone_number)

        # An incomplete document workflow must not swallow unrelated requests.
        # Ask for explicit confirmation before abandoning the customer's data.
        pending = workflow.get("data", {}).get("pending_interrupt")
        if pending:
            # Older sessions may still contain this field from a previous
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
                resumed = self.resume_pending_request(phone_number, pending_service_query, trace_id=trace_id)
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
        """Start a workflow from a literal button/list/menu-digit tap —
        pure protocol dispatch, never a semantic classification. Free-text
        workflow starts ("I want a loan", "send 500 to Priya") are decided
        by the LLM router in app/conversation/manager.py, which then calls
        app/conversation/workflow_adapter.py::start_workflow_directly()
        directly — this method exists only for the two literal-tap cases
        that carry their own menu context worth preserving (a menu-digit
        reply that isn't itself the target workflow's own free-text
        phrasing, and a loan-type list row tapped with no active workflow).
        """
        query = str(query or "")
        normalized = query.strip().lower()
        logger.info(
            f"[{trace_id}] Checking for protocol workflow start | "
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
            # application entirely — and that reply has nowhere to land.
            # The row id already fully states what was chosen, so treat it
            # exactly like typing "I'd like a <type> loan" — start a fresh
            # application with that type instead of dead-ending.
            logger.info(
                f"[{trace_id}] Loan-type tap with no active workflow, starting fresh | "
                f"phone={phone_number[-4:]} | loan_type={LOAN_TYPES[normalized]}"
            )
            workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
            create_workflow(phone_number, workflow)
            return self.loan_handler._select_type(workflow, phone_number, normalized, trace_id)
        return {"handled": False, "response": None}

    def resume_pending_request(self, phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        """Resume a service request that had to wait for registration to
        finish (see registration_gate.py's pending_service_query). The
        original message may be free text ("I want a loan"), so this
        makes ONE fresh LLM routing call for it (the message was never
        classified before — registration_gate intercepted it pre-LLM) and
        dispatches via the same start_workflow_directly() adapter every
        other free-text workflow start uses."""
        protocol = self.start_requested(phone_number, query, trace_id=trace_id)
        if protocol["handled"]:
            return protocol
        decision = classify_and_route_llm_sync(query, None, None, trace_id)
        if decision is None:
            return {"handled": False, "response": None}
        target = decision.resolved_target_workflow()
        if decision.action in ("START_WORKFLOW", "SWITCH") and target:
            started = start_workflow_directly(
                target, phone_number, transfer_handler=self.transfer_handler, query=query, trace_id=trace_id,
            )
            if started:
                return started
        return {"handled": False, "response": None}


def _insufficient_balance_message() -> str:
    return render_insufficient_balance()


def _looks_like_protocol_id(query: str) -> bool:
    """Button/list taps and other menu-generated ids ("lt_home",
    "acct_yes") are a fixed machine protocol, not natural language -- they
    must never reach an LLM classification call (button/list/menu/digit
    handling is required to stay deterministic).

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


_MENU_OR_RESTART_WORDS = {
    "menu", "main menu", "show menu", "display menu", "show me the menu",
    "take me to the main menu", "open menu", "home",
    "start over", "start again", "restart", "begin again",
}


def _is_menu_or_restart_word(query: str) -> bool:
    """Literal, exact-phrase menu/restart protocol — mirrors
    rules._MENU_WORDS/_RESTART_WORDS (app/conversation/intent/rules.py).
    Kept as its own local check (like _is_cancel_command/_is_back_command
    below) rather than imported, since this module and rules.py already
    each maintain their own copy of every hard-navigation word set."""
    text = re.sub(r"[^a-z0-9 ]", "", query.strip().lower())
    return text in _MENU_OR_RESTART_WORDS


def _is_cancel_command(query: str) -> bool:
    """Recognize an explicit, literal stop/cancel word or fixed phrase —
    hard navigation protocol, kept deterministic. Natural-language
    cancellation phrased any other way is the LLM router's job (action
    "CANCEL")."""
    if any(ord(ch) > 127 for ch in query):
        return False
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    return text in {
        "cancel", "cancel it", "cancel this", "stop", "stop it", "exit",
        "quit", "end", "end this", "never mind", "no thanks",
    } or text.startswith("cancel ") or text.startswith("stop ")


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
    though the old workflow's in-progress, unconfirmed answers are.

    Returns None (caller falls through to the existing behavior) if the
    target workflow type isn't one start_workflow_directly can start."""
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
