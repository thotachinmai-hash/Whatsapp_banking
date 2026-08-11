import os
import re
import time
import json
from typing import Annotated, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
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
    tool_list_beneficiaries,
    tool_get_spend_summary,
    tool_get_loan_product_info,
    tool_search_bank_documents,
)
from app.memory import get_session_history
from app.logger import get_logger
from app.workflows.manager import WorkflowManager
from app.workflows.memory import get_workflow
from app.conversation.context_store import ConversationContextStore
from app.conversation.manager import ConversationManager
from app.conversation.responses.errors import render_agent_error
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


def get_llm() -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model,
        api_key=convert_to_secret_str(os.getenv("GROQ_API_KEY", "")),
        temperature=0
    )


def make_tools(trace_id: str,phone_number: str,) -> list:
    return [
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
            func=lambda: tool_list_beneficiaries(phone_number, trace_id),
            name="list_beneficiaries",
            description="""
            List the customer's saved transfer beneficiaries (name and
            masked account number). Use this for "list my beneficiaries",
            "who do I have saved", "show my saved payees", or similar.
            Never invent a beneficiary list from memory or from a previous
            transfer's details — always call this tool.
            """,
        ),
        ]


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

    def agent_node(state: AgentState) -> dict:  # type: ignore
        start = time.time()
        system_message = SystemMessage(content=f"""You are a helpful Finacle Banking assistant on WhatsApp.
You help customers check their account balance, view transactions and spend summaries,
deposit cheques, check the status of a cheque deposit request, and start money transfers.

Only answer questions related to banking and the services this app supports (accounts,
balances, transactions, transfers, cheques, loans, KYC). Do not answer unrelated
general-knowledge questions — if asked something outside banking, politely redirect the
customer to what you can help with instead.
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
When a customer wants to transfer money, the deterministic transfer workflow handles it before you are called.
When a customer asks about a transfer, transaction ID (format TRF-XXXXXXXX), or "my last
transfer"/"last transaction I made" in the context of money sent to someone, use
check_transfer_status. If they do not provide an ID, call it with an empty request_id; the
tool lists all transfers linked to their registered phone number, most recent first. Never
ask for an ID first. Use get_last_transactions instead for general account/statement history.
Always be polite, concise, and professional.
Format currency amounts clearly with the currency symbol.
For balances: "Your current balance is £1,234.56"
For transactions: List them clearly with date, description, and amount.
For spend summaries: List each category with its total.

Important: Keep responses short and suitable for WhatsApp messages.""")

        messages = [system_message] + state["messages"]

        # Groq's Llama models occasionally emit a malformed pseudo-tool-call
        # (e.g. `<function=...>` text instead of a proper tool call), which
        # the API rejects with a 400 tool_use_failed error. That's a
        # generation glitch, not a real failure — a fresh attempt at the
        # same turn usually succeeds, so retry a couple of times before
        # giving up and failing the whole conversation turn.
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                response = llm_with_tools.invoke(messages)
                duration = (time.time() - start) * 1000
                logger.info(f"[{state['trace_id']}] LLM call successful | duration={duration:.2f}ms")
                return {"messages": [response]}
            except Exception as e:
                is_malformed_tool_call = "tool_use_failed" in str(e)
                if is_malformed_tool_call and attempt < attempts:
                    logger.warning(
                        f"[{state['trace_id']}] LLM emitted a malformed tool call, retrying "
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


async def _run_llm_agent(
    query: str,
    phone_number: str,
    trace_id: str,
    parsed_document: dict | None = None,
) -> str:
    """The LLM+tools branch — unchanged from before Phase 5, only
    extracted into its own function so ConversationManager can call it as
    an injected `llm_fallback` without app/agent/agent.py importing (and
    creating a cycle with) app/conversation/manager.py. Does not touch
    session history or ConversationContext on the way out — the caller
    (ConversationManager._finish) owns persisting the turn."""
    start = time.time()

    history = get_session_history(phone_number)

    past_messages = []
    for msg in history[-6:]:
        if msg["role"] == "user":
            past_messages.append(HumanMessage(content=msg["content"][:300]))
        elif msg["role"] == "assistant":
            past_messages.append(AIMessage(content=msg["content"][:300]))

    agent = build_agent(trace_id, phone_number)

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

    return response_content


async def run_agent(
    query: str,
    phone_number: str,
    trace_id: str,
    parsed_document: dict | None = None,
    detected_language: str | None = None,
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
    second, redundant text-based detection call for voice turns."""
    logger.info(f"[{trace_id}] Agent started | phone={phone_number[-4:]} | query={query[:50]}")
    try:
        return await conversation_manager.handle_message(
            phone_number=phone_number,
            message=query,
            trace_id=trace_id,
            llm_fallback=_run_llm_agent,
            parsed_document=parsed_document,
            detected_language=detected_language,
        )
    except Exception as e:
        logger.error(f"[{trace_id}] Agent failed | error={e}")
        return render_agent_error()
