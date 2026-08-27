import asyncio
import os
import re
import time
import json
from datetime import date
from typing import Annotated, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.utils import convert_to_secret_str
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from app.agent.tools import (
    tool_get_account_balance,
    tool_get_last_transactions,
    tool_start_cheque_workflow,
    tool_check_cheque_status,
    tool_check_loan_status,
    tool_check_transfer_status,
    tool_check_kyc_status,
    tool_list_beneficiaries,
    tool_get_spend_summary,
    tool_get_loan_product_info,
    tool_search_bank_documents,
    tool_start_transfer_workflow,
    tool_start_loan_workflow,
    tool_start_kyc_workflow,
)
from app.memory import get_session_history
from app.services.sarvam_client import get_fast_model
from app.metrics import log_tool_call
from app.logger import get_logger
from app.workflows.manager import WorkflowManager
from app.workflows.memory import get_workflow
from app.conversation.context_store import ConversationContextStore
from app.conversation.manager import ConversationManager
from app.conversation.renderer import InteractiveListRow, InteractiveListSection, ResponseLike, StructuredResponse
from app.conversation.responses.errors import render_agent_error
from app.conversation.responses.common import with_nav_buttons
load_dotenv()
logger = get_logger(__name__)

workflow_manager = WorkflowManager()
conversation_context_store = ConversationContextStore()
conversation_manager = ConversationManager(
    workflow_manager=workflow_manager,
    context_store=conversation_context_store,
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    phone_number: str
    trace_id: str


def get_llm() -> ChatOpenAI:
    model = get_fast_model()
    api_key = os.getenv("SARVAM_API_KEY", "")
    return ChatOpenAI(
        model=model,
        api_key=convert_to_secret_str(api_key),
        base_url="https://api.sarvam.ai/v1",
        # Sarvam authenticates with this header instead of the standard
        # `Authorization: Bearer` header the OpenAI SDK sends by default.
        default_headers={"api-subscription-key": api_key},
        temperature=0,
        # sarvam-105b-conversations (get_fast_model()) benchmarked 2-3x
        # faster than sarvam-105b on this exact tool-calling + compound
        # conditional reasoning shape, with correct tool selection and
        # correct threshold arithmetic in testing, and without
        # sarvam-105b's hidden-reasoning-token-starvation failure mode (see
        # get_fast_model()'s docstring) — so max_tokens=6500 here is a
        # generous cap for replies that cite multiple tool results, not a
        # load-bearing workaround the way it was for sarvam-105b.
        max_tokens=6500,
        # reasoning_effort="medium" — kept for the compound/conditional
        # banking reasoning this agent does (check a balance, evaluate a
        # threshold, decide whether to act), unlike the simple single-shot
        # classification/detection calls elsewhere (llm_understanding.py,
        # language.py, classifier.py, document_parser.py), which stay at
        # "low" since more effort there is just latency with no quality
        # gain. Confirmed in testing that "low" also reasons correctly on
        # this model, but "medium" was equal-or-faster overall in that same
        # test, so there's no latency reason to drop it.
        model_kwargs={"reasoning_effort": "medium"},
    )


def _timed_tool(name: str, fn, trace_id: str):
    """Wrap a tool's underlying function so its actual DB/RAG execution
    time (not the LLM's tool-selection overhead) is measured and fed into
    /metrics — one choke point here instead of instrumenting all 14 tool
    functions individually in tools.py."""
    def wrapped(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        log_tool_call(name, (time.time() - start) * 1000, trace_id)
        return result
    return wrapped


def make_tools(trace_id: str,phone_number: str,) -> list:
    tools = [
        StructuredTool.from_function(
            func=lambda account_number="": tool_get_account_balance(account_number, phone_number, trace_id),
            name="get_account_balance",
            description="Get the current balance. For a registered customer, omit account_number and use the account linked to their phone; never ask repeatedly for an account number."
        ),
        StructuredTool.from_function(
            func=lambda account_number="", limit=5, start_date=None, end_date=None, transaction_type=None, category=None: tool_get_last_transactions(
                account_number, limit, start_date, end_date, transaction_type, category, trace_id, phone_number
            ),
            name="get_last_transactions",
            description="""
            Get transactions for a bank account. Defaults to the last 5. For a registered customer, omit account_number to use their linked account.

            Optionally filter by start_date/end_date (YYYY-MM-DD),
            transaction_type ("credit" or "debit"), and/or category — use
            the customer's own everyday word (e.g. "food", "groceries",
            "bills", "rent", "salary", "transport", "entertainment",
            "shopping"); common synonyms are mapped automatically.
            """
        ),
        StructuredTool.from_function(
            func=lambda account_number="", start_date=None, end_date=None, category=None: tool_get_spend_summary(
                account_number, start_date, end_date, category, trace_id, phone_number
            ),
            name="get_spend_summary",
            description="""
            Get a spend summary grouped by category for a bank account, and
            cumulatively total it when a category is given.

            Use for questions like "how much did I spend on food" or "how
            much did I spend on bills this month" or "what's my spending
            breakdown". Pass whatever everyday word the customer used for
            category (e.g. "food", "groceries", "travel", "shopping",
            "rent") — common synonyms are mapped to the real category
            automatically, so use the customer's own word rather than
            guessing the exact stored category name. Optionally filter by
            start_date/end_date (YYYY-MM-DD).
            """
        ),
        StructuredTool.from_function(
            func=lambda loan_type="": tool_get_loan_product_info(loan_type, trace_id),
            name="get_loan_product_info",
            description="""
            Get the bank's published loan terms for a loan type — interest
            rate range, min/max borrowing amount, tenure range, and
            processing fee. Use this for any question about loan interest
            rates, fees, borrowing limits, or repayment tenure (e.g.
            "what's the interest rate on a personal loan", "how much can I
            borrow for a car", "what's the processing fee").

            loan_type is one of: personal, home, vehicle, education
            (or a close synonym like "mortgage", "car", "student"). Leave
            it empty to get the full rate card for all loan types.

            This is the bank's general rate card, not a personal decision —
            never state that a specific customer is approved for, or will
            receive, a particular rate or amount; only report what this
            tool returns.
            """
        ),
        StructuredTool.from_function(
            func=lambda query="": tool_search_bank_documents(query, trace_id),
            name="search_bank_documents",
            description="""
            Search the bank's own indexed documents for informational/policy
            questions — required documents for a loan type, KYC document
            rules, why a cheque might be rejected, or general "how does X
            work" questions. Use this instead of guessing whenever the
            customer asks what's needed for something, rather than asking
            to actually start it. Returns nothing if there is no indexed
            answer — in that case, say so plainly rather than inventing one.
            The indexed documents are written in English, and the search is
            plain keyword matching with no translation step, so ALWAYS pass
            the `query` argument in English regardless of what language the
            customer actually asked in — translate the question yourself
            before calling this tool (e.g. a customer asking in pure Telugu
            with no English words at all should still produce an English
            query like "how does KYC work"), or a real match will be missed.
            """,
        ),
        StructuredTool.from_function(
            func=lambda: tool_start_cheque_workflow(phone_number),
            name="start_cheque_workflow",
            description="""
            Start a cheque deposit workflow.

            Use this tool when the customer wants to:
            - deposit a cheque
            - cash a cheque
            - submit a cheque

            Do not use this tool for balance enquiries or transaction history.
            """),
        StructuredTool.from_function(
            func=lambda request_id="": tool_check_cheque_status(request_id, phone_number, trace_id),
            name="check_cheque_status",
            description="""
            Check cheque requests for the current customer. Do not ask for an
            ID when none was provided: call this with an empty request_id and
            it will list all cheques linked to the customer's phone number.
            Use an ID only when the customer explicitly provides one.
            """
        ),
        StructuredTool.from_function(
            func=lambda request_id="": tool_check_loan_status(request_id, phone_number, trace_id),
            name="check_loan_status",
            description="Check loan applications for the current customer. If no ID is provided, call with an empty request_id to list all applications linked to the customer's phone number. Use an ID only when explicitly provided.",
        ),
        StructuredTool.from_function(
            func=lambda request_id="": tool_check_transfer_status(request_id, phone_number, trace_id),
            name="check_transfer_status",
            description="""
            Check money transfers for the current customer — status
            INITIATED/COMPLETED/FAILED. Use this for "what was my last
            transfer", "check transfer status", or a transaction ID
            (format TRF-XXXXXXXX). If no ID is provided, call with an
            empty request_id to list all transfers linked to the
            customer's phone number, most recent first. This is a
            separate money-transfer record, not a regular account
            transaction — use get_last_transactions instead for the
            account's general transaction/statement history.
            """,
        ),
        StructuredTool.from_function(
            func=lambda: (
                "Beneficiaries listed — the customer sees a tappable list; do not repeat the names/accounts in your own reply, just briefly acknowledge.",
                tool_list_beneficiaries(phone_number, trace_id),
            ),
            name="list_beneficiaries",
            description="""
            List the customer's saved transfer beneficiaries (name and
            masked account number). Use this for "list my beneficiaries",
            "who do I have saved", "show my saved payees", or similar.
            Never invent a beneficiary list from memory or from a previous
            transfer's details — always call this tool.
            """,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=lambda request_id="": tool_check_kyc_status(request_id, phone_number, trace_id),
            name="check_kyc_status",
            description="""
            Check the status of the customer's KYC update requests. If no
            ID is provided, call with an empty request_id to list all KYC
            requests linked to the customer's phone number, most recent
            first. Use this for "check my KYC status", "is my KYC
            complete/incomplete", or similar — never guess whether KYC is
            complete or incomplete from memory or from context.
            """,
        ),
        StructuredTool.from_function(
            func=lambda beneficiary_name="", amount="": tool_start_transfer_workflow(
                phone_number, beneficiary_name, amount, trace_id
            ),
            name="start_transfer_workflow",
            description="""
            Start a money-transfer workflow, optionally pre-filled with a
            beneficiary name and/or amount already stated by the customer.

            IMPORTANT: if the request has a condition (e.g. "if my balance
            is above X", "if I have enough"), you MUST call
            get_account_balance first and confirm the condition actually
            holds against the real returned balance before calling this
            tool. If the condition does not hold, do not call this tool —
            explain the balance and that no transfer was started instead.

            This tool only starts the guided transfer flow — it does not
            move money. The customer still confirms the transfer at the
            end of it, exactly as if they had typed "transfer money"
            themselves.
            """,
        ),
        StructuredTool.from_function(
            func=lambda loan_type="", requested_amount="": tool_start_loan_workflow(
                phone_number, loan_type, requested_amount, trace_id
            ),
            name="start_loan_workflow",
            description="""
            Start a loan application workflow, optionally pre-filled with
            the loan type (personal, home, vehicle, education) and/or the
            amount the customer wants to borrow.

            IMPORTANT: if the request has a condition (e.g. "apply for a
            loan if the interest rate is below X%"), you MUST call
            get_loan_product_info first and confirm the condition actually
            holds against the real returned rate before calling this tool.
            If it does not hold, do not call this tool — explain the rate
            and that no application was started instead.

            This tool does not approve or submit a loan — the workflow
            still requires the customer to upload supporting documents and
            confirm before anything is actually submitted.
            """,
        ),
        StructuredTool.from_function(
            func=lambda: tool_start_kyc_workflow(phone_number, trace_id),
            name="start_kyc_workflow",
            description="""
            Start a KYC update workflow — use this when the customer wants
            to update/complete their KYC, including after
            check_kyc_status shows it is incomplete or missing and the
            customer wants (or was asked and agreed) to fix it. Do not use
            this just to check status — use check_kyc_status for that.
            """,
        ),
    ]
    for t in tools:
        t.func = _timed_tool(t.name, t.func, trace_id)
    return tools


def build_agent(trace_id: str,phone_number: str,) -> Any:
    tools = make_tools(trace_id,phone_number,)
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    active_workflow = get_workflow(phone_number)
    workflow_context = "No active workflow."
    if active_workflow:
        workflow_context = (
            f"Active workflow: {active_workflow.get('type')} at step {active_workflow.get('step')}. "
            f"Known workflow data: {json.dumps(active_workflow.get('data', {}), default=str)[:1200]}"
        )

    async def agent_node(state: AgentState) -> dict:  # type: ignore
        start = time.time()
        system_message = SystemMessage(content=f"""You are a helpful Finacle Banking assistant on WhatsApp.
You help customers check their account balance, view transactions and spend summaries,
deposit cheques, check the status of a cheque deposit request, and start money transfers.

Today's date is {date.today().isoformat()}. Use this — never guess or invent a date — to resolve
any relative date/time reference ("this month", "last week", "today", "this year") into the actual
start_date/end_date (YYYY-MM-DD) you pass to get_last_transactions/get_spend_summary. Do not assume
any other date.

You are friendly and conversational, not rigid — this customer may not be comfortable with
banking apps or with English. Understand English, English mixed with a native language,
pure native-language text, romanized native languages (e.g. Hindi written as "mera balance
kya hai"), and native scripts equally well; never ask the customer to rephrase in English or
in a different form just because their message mixed languages or used a native script.
If a request is ambiguous or only partly clear, don't just decline it — ask a warm, specific
follow-up that helps them get to what they need, the way a helpful bank staff member would,
rather than a flat "I can't help with that."

Only answer questions related to banking and the services this app supports (accounts,
balances, transactions, transfers, cheques, loans, KYC). Do not answer unrelated
general-knowledge questions — if asked something outside banking, warmly redirect the
customer to what you can help with instead, in a way that invites them to ask something you
can actually help with rather than just shutting the request down.

GROUNDING RULE — the most important rule in this prompt: for any claim about the customer's
real data — balance, transactions, beneficiaries, cheque/loan/KYC/transfer status, spend, or
a condition compared against any of these — you must call the matching tool and use only what
it returns. Never answer from memory, from an earlier turn in this conversation, or from
general reasoning, even if you believe you already know the value — balances and statuses can
change between turns, so always call the tool again for this turn. If no tool covers what is
being asked, say so plainly instead of guessing.

CHECK-THEN-ACT RULE: when a request states a condition ("if my balance is more than X",
"if the rate is below X%", "if my KYC is incomplete") before asking you to do something, you
must: (1) call the read tool(s) needed to get the real fact for THIS turn — never reuse a
number mentioned earlier in the conversation, (2) evaluate the condition yourself against that
real, current tool result, (3) only if the condition holds, call the matching start_*_workflow
tool (start_transfer_workflow, start_loan_workflow, start_kyc_workflow) with whatever details
(beneficiary, amount, loan type) you already have from the sentence, (4) if the condition does
not hold, explain the real value you checked and that you have not started anything — do not
call a start_*_workflow tool in that case. If the condition DOES hold, you must actually call the
start_*_workflow tool in this same turn — do not just report that the condition is satisfied and
stop there; reporting the fact without also calling the tool leaves the customer's original request
undone. If the customer's condition was an absence of something ("if I haven't paid my landlord
this month"), a tool result that found no matching transaction/record means the condition holds —
call the start_*_workflow tool. Always state the fact you checked in your reply
(e.g. "Your balance is INR X, which is above INR 50,000, so I've started the transfer") so the
customer can see what the decision was based on. A start_*_workflow tool only begins the
guided flow — it never moves money, submits KYC, or submits a loan application by itself; the
existing confirmation step inside that workflow still has to happen before anything is final.
For loan interest rates, fees, borrowing limits, or tenure, call get_loan_product_info and
state exactly what it returns — never invent a rate/fee/limit yourself, and never state one
from memory. That tool gives the bank's general published terms, not a personal decision:
do not say a specific customer will get a particular rate or amount, and do not say someone
is "eligible" or "approved" — eligibility depends on factors like income, existing
obligations, and credit profile, decided during the actual application. Offer to help them
check requirements or start an application instead of guessing.
For questions about what documents are needed for a loan/KYC/cheque, how KYC works, why a
cheque might be rejected, or other policy/how-it-works questions, call search_bank_documents
and answer only from what it returns. If it finds nothing relevant, say plainly that you don't
have that information rather than guessing. Do not invent any other bank policy or eligibility
rule beyond what a tool result states.
Never claim that a transaction, transfer, application, or update was completed unless a
tool result or the active workflow actually confirms it.

Only state account numbers, transaction details, amounts, statuses, or dates that a tool
result actually returned — never invent or guess one. If a tool result doesn't include
something the customer asked about, say plainly that it isn't available rather than filling
it in. Never say or repeat an Aadhaar number, PAN number, OTP, PIN, CVV, or password, even
if one appears in a tool result or the conversation.

Write in simple, everyday English a customer who isn't comfortable with banking apps can
follow easily — short sentences, no technical or internal terms (never say "intent",
"classifier", "router", "workflow", "system error", or similar). Keep messages short and
use at most one or two emojis, only when they genuinely help.

{workflow_context}
If the customer asks a side question while a workflow is active, answer it naturally and
keep the workflow state unchanged. Do not ask them to restart the workflow. For questions
about Goa, travel, trips, or expenses, use transaction history or spend summaries and
interpret matching descriptions/categories; do not invent transactions.

When a registered customer asks about their balance, transactions, or spend summary,
do not ask for an account number. Use the account linked to their registered phone;
if there are multiple accounts, list them clearly and let the customer choose.
When a customer wants to deposit/cash/submit a cheque, use the start_cheque_workflow tool.
When a customer asks about a cheque, cheque status, cheque details, or their cheques,
use check_cheque_status. If they do not provide an ID, call it with an empty request_id;
the tool lists all cheques linked to their registered phone number. Never ask for an ID first.
When a customer asks about a loan, loan status, application details, or their applications,
use check_loan_status. If they do not provide an ID, call it with an empty request_id;
the tool lists all loan applications linked to their registered phone number. Never ask for an ID first.
When a customer asks to list/see their saved beneficiaries or payees (e.g. "list my
beneficiaries", "who do I have saved", "show my saved payees", "who are my beneficiaries"),
use list_beneficiaries. Never answer this from memory or from a previous transfer's details —
always call the tool, even if you believe you already know who is saved.
When a customer wants to transfer money with no condition attached, the deterministic transfer
workflow usually handles it before you are called. If you are called anyway (e.g. the request
was compound or had a condition, such as "check my balance and transfer to my landlord if I
have enough"), use start_transfer_workflow yourself, following the CHECK-THEN-ACT RULE above.
When a customer asks about a transfer, transaction ID (format TRF-XXXXXXXX), or "my last
transfer"/"last transaction I made" in the context of money sent to someone, use
check_transfer_status. If they do not provide an ID, call it with an empty request_id; the
tool lists all transfers linked to their registered phone number, most recent first. Never
ask for an ID first. Use get_last_transactions instead for general account/statement history.
When a customer asks about their KYC status, use check_kyc_status the same way (empty
request_id lists all their KYC requests). If they then want to fix an incomplete/missing KYC,
use start_kyc_workflow. When a customer wants to apply for a loan with a condition attached
(e.g. "apply if the rate is below 12%"), use start_loan_workflow following the CHECK-THEN-ACT
RULE above, instead of just describing the rate and stopping.
"Needs attention" (e.g. "does my KYC or loan application need attention") means a status of
PENDING, INCOMPLETE, or REJECTED — not COMPLETED/ACTIVE/APPROVED. Only use these exact
status values as returned by the tools, never invent your own status wording.
Always be polite, concise, and professional.
Format currency amounts clearly with the currency symbol.
For balances: "Your current balance is INR 1,234.56"
For transactions: List them clearly with date, description, and amount.
For spend summaries: List each category with its total.

Important: Keep responses short and suitable for WhatsApp messages.""")

        messages = [system_message] + state["messages"]

        # Sarvam's models occasionally emit a malformed pseudo-tool-call
        # (e.g. `<function=...>` text instead of a proper tool call), which
        # the API rejects with a 400 tool_use_failed error. That's a
        # generation glitch, not a real failure — a fresh attempt at the
        # same turn usually succeeds, so retry a couple of times before
        # giving up and failing the whole conversation turn.
        # Confirmed live: some turns still return empty content on all 3
        # attempts (e.g. a plain "transfer to my landlord" trigger, no
        # deeper reasoning than a simple lookup+tool-call). One extra
        # attempt lowers — doesn't eliminate — how often that happens.
        attempts = 4
        for attempt in range(1, attempts + 1):
            try:
                response = await llm_with_tools.ainvoke(messages)
                duration = (time.time() - start) * 1000
                # A reasoning model can occasionally spend its entire token
                # budget on internal reasoning_content and come back with
                # neither a tool call nor any real answer text — confirmed
                # live on compound/conditional turns. That's the same kind
                # of generation glitch as a malformed tool call, not a real
                # failure, so retry it the same way rather than returning
                # an empty reply to the customer.
                has_tool_calls = bool(getattr(response, "tool_calls", None))
                has_content = bool(str(getattr(response, "content", "") or "").strip())
                if not has_tool_calls and not has_content and attempt < attempts:
                    logger.warning(
                        f"[{state['trace_id']}] LLM returned empty content with no tool calls, retrying "
                        f"| attempt={attempt}/{attempts}"
                    )
                    continue
                logger.info(f"[{state['trace_id']}] LLM call successful | duration={duration:.2f}ms")
                return {"messages": [response]}
            except Exception as e:
                error_text = str(e)
                is_malformed_tool_call = "tool_use_failed" in error_text
                # Sarvam's shared model capacity occasionally returns a 503
                # "model_overloaded" — confirmed live, on both a text and a
                # voice turn. That's a momentary, provider-wide condition,
                # not something specific to this request, so a fresh
                # attempt a moment later is worth trying the same way a
                # malformed tool call already is, rather than failing the
                # whole turn on the first hit.
                is_overloaded = "model_overloaded" in error_text or "503" in error_text
                if (is_malformed_tool_call or is_overloaded) and attempt < attempts:
                    logger.warning(
                        f"[{state['trace_id']}] LLM call failed "
                        f"({'model overloaded' if is_overloaded else 'malformed tool call'}), retrying "
                        f"| attempt={attempt}/{attempts} | error={e}"
                    )
                    continue
                duration = (time.time() - start) * 1000
                logger.error(f"[{state['trace_id']}] LLM call failed | error={e} | duration={duration:.2f}ms")
                raise

    def should_continue(state: AgentState) -> str:  # type: ignore
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)  # type: ignore
    graph.add_node("tools", ToolNode(make_tools(trace_id,phone_number,)))  # type: ignore
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=None)


def _beneficiary_list_response(messages: list, intro_text: str) -> StructuredResponse | None:
    """Turn a successful list_beneficiaries tool call into a real tappable
    WhatsApp list instead of the LLM's own prose description — the
    deferred "agent-built dynamic list" work, now unblocked by a safe row
    id scheme.

    Row id = "Transfer to {name}" (e.g. "Transfer to Priya"), not a bare
    digit. This is deliberate: this listing can be shown with NO active
    workflow, and message_handler.py sends back exactly the row's id as
    the customer's next message (see get_interactive_reply's docstring
    in app/services/whatsapp.py) — a bare "1" would be misread by
    WorkflowManager.start_requested's own main-menu digit map (e.g. "1"
    = "Transfer money" from scratch) instead of meaning the tapped
    beneficiary. "Transfer to {name}" has no such collision: it's already
    exactly the free-text phrasing start_transfer_from_text (used by both
    the deterministic keyword path and the LLM-intent path — see its own
    docstring) already parses via _BENEFICIARY_INTENT_RE to pre-fill a
    transfer with that beneficiary. Tapping a row triggers the EXACT same
    already-tested code path a typed "transfer to Priya" would — no new
    routing/handler code needed, no new failure mode introduced.

    Returns None when the tool wasn't called, found nothing, or found no
    beneficiaries — callers fall back to the LLM's own text response."""
    for message in messages:
        if getattr(message, "name", None) != "list_beneficiaries":
            continue
        artifact = getattr(message, "artifact", None)
        if not isinstance(artifact, dict) or not artifact.get("found"):
            return None
        beneficiaries = artifact.get("beneficiaries") or []
        # WhatsApp caps a list at 10 rows — beyond that, fall back to the
        # LLM's own prose (which has no such limit) rather than silently
        # hiding some beneficiaries from a tappable list, matching the
        # same threshold transfer.py's own _beneficiary_prompt uses.
        if not beneficiaries or len(beneficiaries) > 10:
            return None
        rows = [
            InteractiveListRow(
                id=f"Transfer to {b['name']}"[:200],
                title=str(b["name"])[:24],
                description=str(b.get("account_number_masked", ""))[:72],
            )
            for b in beneficiaries
        ]
        return StructuredResponse.list_of(
            intro_text or "Here are your saved beneficiaries:",
            "Beneficiaries",
            [InteractiveListSection(title="Saved beneficiaries", rows=rows)],
        )
    return None


async def _run_llm_agent(
    query: str,
    phone_number: str,
    trace_id: str,
    parsed_document: dict | None = None,
) -> ResponseLike:
    """The LLM+tools branch — unchanged from before Phase 5, only
    extracted into its own function so ConversationManager can call it as
    an injected `llm_fallback` without app/agent/agent.py importing (and
    creating a cycle with) app/conversation/manager.py. Does not touch
    session history or ConversationContext on the way out — the caller
    (ConversationManager._finish) owns persisting the turn."""
    start = time.time()

    history = await asyncio.to_thread(get_session_history, phone_number)

    past_messages = []
    for msg in history[-6:]:
        if msg["role"] == "user":
            past_messages.append(HumanMessage(content=msg["content"][:300]))
        elif msg["role"] == "assistant":
            past_messages.append(AIMessage(content=msg["content"][:300]))

    agent = await asyncio.to_thread(build_agent, trace_id, phone_number)

    initial_state: AgentState = {
        "messages": past_messages + [HumanMessage(content=query)],
        "phone_number": phone_number,
        "trace_id": trace_id,
    }

    result = await agent.ainvoke(  # type: ignore
        initial_state,
        config={"recursion_limit": 10}
    )

    final_message = result["messages"][-1]
    tools_called = [
        m.name for m in result["messages"]
        if hasattr(m, "name") and m.name is not None
    ]

    response_content = str(
        final_message.content if hasattr(final_message, "content")
        else final_message
    )

    # Strip XML tags some models add
    response_content = re.sub(r'<[^>]+>', '', response_content).strip()

    duration = (time.time() - start) * 1000
    logger.info(f"[{trace_id}] Agent completed | duration={duration:.2f}ms | tools={tools_called}")

    beneficiary_list = _beneficiary_list_response(result["messages"], response_content)
    if beneficiary_list is not None:
        return beneficiary_list

    if not response_content:
        # The retry loop in agent_node() already tries again on empty
        # content — this is what's left after every attempt still came
        # back empty (confirmed live: a reasoning model can occasionally
        # exhaust its whole token budget on internal reasoning across
        # every retry with nothing left to say). Without this, the
        # customer would receive a genuinely blank WhatsApp message,
        # which reads as the bot being broken — an honest apology is a
        # real answer, an empty bubble is not.
        logger.warning(f"[{trace_id}] Agent returned empty content after all retries — using fallback text")
        # A dead-end reply is exactly when a tap-to-reply escape hatch
        # matters most — give the customer Back/Cancel/Main Menu instead of
        # only a plain-text apology with no obvious next step.
        return with_nav_buttons("Sorry, I wasn't able to work that out just now. Could you try asking again?")

    # A voice-in customer who asked an informational question that has a
    # matching plain-text menu already in code (e.g. "what loan types do
    # you have") still can't see it from spoken audio alone — flag it here
    # so app/services/message_handler.py::send_voice_reply can send that
    # existing menu as its own follow-up message. Text-in customers are
    # unaffected — send_voice_reply is only called for voice turns.
    if "get_loan_product_info" in tools_called:
        return StructuredResponse(text=response_content, voice_menu="loan_type")

    return response_content


async def run_agent(
    query: str,
    phone_number: str,
    trace_id: str,
    parsed_document: dict | None = None,
    detected_language: str | None = None,
    is_voice: bool = False,
):
    """Thin entry point — see app/conversation/manager.py::ConversationManager
    for the actual turn orchestration (Phase 5). Kept here as a stable,
    unchanged external signature for message_handler.py/main.py, and as a
    last-resort safety net in case ConversationManager itself fails to
    even log (it already catches everything internally and returns a
    user-safe response, so this is only a belt-and-suspenders guard).

    `detected_language` is an optional ISO 639-1 hint from voice
    transcription (Whisper already detected the spoken language — see
    app/services/transcription.py) so ConversationManager doesn't need a
    second, redundant text-based detection call for voice turns.
    `is_voice` tells ConversationManager which of its two independent
    sticky languages (voice vs text) this turn belongs to — see
    ConversationManager._update_language; it must be passed explicitly
    rather than inferred from `detected_language` being set, since a short
    voice utterance can arrive with `detected_language=None` too (Sarvam's
    own per-utterance tag gets dropped as unreliable below a length
    threshold) and must still update the voice channel, not the text one."""
    logger.info(f"[{trace_id}] Agent started | phone={phone_number[-4:]} | query={query[:50]}")
    try:
        return await conversation_manager.handle_message(
            phone_number=phone_number,
            message=query,
            trace_id=trace_id,
            llm_fallback=_run_llm_agent,
            parsed_document=parsed_document,
            detected_language=detected_language,
            is_voice=is_voice,
        )
    except Exception as e:
        logger.error(f"[{trace_id}] Agent failed | error={e}")
        return render_agent_error()
