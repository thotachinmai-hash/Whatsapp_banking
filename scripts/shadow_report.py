"""Controlled shadow-data evaluation for the LLM-first routing migration
(the phase before considering classify_workflow_request for replacement).

Combines two sources, exactly as requested:
  - REAL traffic: scripts/_real_log_cases.json, extracted from a real test
    session's logs/logs.txt (trace_id-correlated query text + the actual
    rule classification that ran in production for it).
  - Synthetic, targeted coverage: scripts/shadow_eval_corpus.py's 101-case
    matrix, for the categories real traffic didn't happen to exercise.

For each case, computes and prints EVERY field requested: current rule
intent, LLM intent, current workflow, LLM action, target workflow,
confidence, language, whether the decision would change the user-visible
result, whether the case is safety-sensitive, and whether a disagreement is
a rule error or an LLM error. Never authoritative -- this script only
observes and reports; SHADOW_LLM_ROUTING_ENABLED / LLM_FALLBACK_ENABLED are
not touched by anything here.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from app.conversation.context import ConversationContext  # noqa: E402
from app.conversation.intent import classify_intent  # noqa: E402
from app.conversation.intent.llm_routing import classify_and_route_llm  # noqa: E402
from app.conversation.intent.models import CONFIDENCE_HIGH, WORKFLOW_EXECUTING_INTENTS  # noqa: E402
from app.conversation.router import get_workflow_for_intent, route_intent  # noqa: E402
from app.workflows.manager import _is_cancel_command, _is_current_workflow_input  # noqa: E402
from scripts.shadow_eval_corpus import CORPUS as SYNTHETIC_CORPUS  # noqa: E402

_REAL_CASES_PATH = os.path.join(os.path.dirname(__file__), "_real_log_cases.json")

# Actions that, if wrong, could visibly move the customer into (or out of)
# a workflow -- the closest this taxonomy gets to "safety-sensitive" at the
# ROUTING layer. None of these authorize a financial action by themselves
# (every workflow still requires its own STEP_CONFIRM_* gate) -- this flag
# means "worth a human's attention if wrong," not "could move money."
_SAFETY_SENSITIVE_ACTIONS = {"START_WORKFLOW", "SWITCH", "CANCEL"}


@dataclass
class ShadowCase:
    source: str  # "real" | "synthetic"
    case_id: str
    message: str
    current_workflow: Optional[str]
    current_step: Optional[str]
    category: str = ""

    rule_intent: Optional[str] = None
    rule_confidence: float = 0.0
    rule_action: Optional[str] = None
    rule_workflow: Optional[str] = None

    llm_intent: Optional[str] = None
    llm_action: Optional[str] = None
    llm_certainty: Optional[str] = None
    llm_workflow: Optional[str] = None
    llm_language: Optional[str] = None
    llm_error: Optional[str] = None

    @property
    def agree(self) -> Optional[bool]:
        if self.llm_error:
            return None
        return self.rule_action == self.llm_action and self.rule_workflow == self.llm_workflow

    @property
    def would_change_result(self) -> Optional[bool]:
        """Would swapping to the LLM's decision change what the customer
        sees this turn -- a stricter question than "labels differ", since
        e.g. rule=BANKING_LLM/llm=TOOL both reach the same agent call."""
        if self.llm_error:
            return None
        if self.rule_action == "BANKING_LLM" and self.llm_action in ("TOOL", "RAG"):
            return False
        return not self.agree

    @property
    def safety_sensitive(self) -> bool:
        return self.rule_action in _SAFETY_SENSITIVE_ACTIONS or self.llm_action in _SAFETY_SENSITIVE_ACTIONS

    @property
    def is_multilingual(self) -> bool:
        return any(ord(ch) > 127 for ch in self.message) or "native" in self.category or "romanized" in self.category or "mixed" in self.category

    @property
    def is_create_account_vs_onboarding(self) -> bool:
        return "add_account" in {self.rule_intent, self.llm_intent} or "add_account" in {self.rule_workflow, self.llm_workflow, self.current_workflow}

    @property
    def is_cheque_dep_vs_status(self) -> bool:
        relevant = {"cheque_deposit_request", "cheque_status_request"}
        return bool(relevant & {self.rule_intent, self.llm_intent})

    @property
    def is_workflow_switch(self) -> bool:
        return "SWITCH" in (self.rule_action, self.llm_action)

    @property
    def is_cancel_vs_switch(self) -> bool:
        return "CANCEL" in (self.rule_action, self.llm_action) and self.current_workflow is not None

    @property
    def is_short_reply(self) -> bool:
        return len(self.message.strip()) <= 4

    @property
    def error_attribution(self) -> str:
        """Best-effort call on WHOSE output is more defensible when they
        disagree -- 'rule', 'llm', 'both_plausible', or 'agree'. This is a
        judgment call recorded for human review, not a ground-truth label."""
        if self.agree is None:
            return "llm_error(call_failed)"
        if self.agree:
            return "agree"
        # Rule fell back to CLARIFY/OUT_OF_SCOPE/SAFE_FALLBACK purely
        # because it has no opinion (low confidence / no keyword match) --
        # the LLM actually attempting an answer is very likely the better
        # outcome for the customer, even if occasionally over-eager.
        if self.rule_action in ("CLARIFY", "OUT_OF_SCOPE", "SAFE_FALLBACK") and self.llm_action not in (
            "CLARIFY", "OUT_OF_SCOPE",
        ):
            return "rule_error(no_opinion)"
        # LLM claims OUT_OF_SCOPE/CLARIFY on a message the rule confidently
        # and correctly handled -- LLM under-performing.
        if self.llm_action in ("CLARIFY", "OUT_OF_SCOPE") and self.rule_action not in (
            "CLARIFY", "OUT_OF_SCOPE", "SAFE_FALLBACK",
        ):
            return "llm_error(under_confident)"
        return "both_plausible(needs_human_review)"


def _rule_effective_decision(rule_intent_result, rule_routing, current_workflow, current_step, message):
    if current_workflow:
        if _is_cancel_command(message):
            return "CANCEL", None
        fake_workflow = {"type": current_workflow, "step": current_step, "data": {}}
        if (
            rule_intent_result.intent in WORKFLOW_EXECUTING_INTENTS
            and rule_intent_result.confidence >= CONFIDENCE_HIGH
            and not _is_current_workflow_input(fake_workflow, message)
        ):
            target = get_workflow_for_intent(rule_intent_result.intent)
            if target and target != current_workflow:
                return "SWITCH", target
        if rule_intent_result.intent == "workflow_correction":
            return "CORRECT", current_workflow
        return "CONTINUE", current_workflow

    mapping = {
        "START_WORKFLOW": "START_WORKFLOW", "OUT_OF_SCOPE": "OUT_OF_SCOPE",
        "CLARIFICATION_REQUIRED": "CLARIFY", "BANKING_LLM": "BANKING_LLM", "SAFE_FALLBACK": "SAFE_FALLBACK",
    }
    return mapping.get(rule_routing.action, rule_routing.action), rule_routing.workflow


async def _run_case(case: ShadowCase) -> None:
    context = ConversationContext(phone_number="441111111111")
    context.current_workflow = case.current_workflow
    context.current_step = case.current_step

    rule_intent_result = await classify_intent(
        case.message, context=context, trace_id=f"shadow_report:{case.case_id}", llm_classify=None
    )
    rule_routing = route_intent(rule_intent_result, context=context)
    action, workflow = _rule_effective_decision(
        rule_intent_result, rule_routing, case.current_workflow, case.current_step, case.message
    )
    case.rule_intent = rule_intent_result.intent
    case.rule_confidence = rule_intent_result.confidence
    case.rule_action = action
    case.rule_workflow = workflow

    try:
        decision = await classify_and_route_llm(case.message, context=context, trace_id=f"shadow_report:{case.case_id}")
    except Exception as e:
        case.llm_error = str(e)
        return
    if decision is None:
        case.llm_error = "no response / unparsable JSON"
        return
    case.llm_intent = decision.intent
    case.llm_action = decision.action
    case.llm_certainty = decision.certainty
    case.llm_workflow = decision.resolved_target_workflow()
    case.llm_language = decision.language


def _load_real_cases() -> list[ShadowCase]:
    with open(_REAL_CASES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for r in raw:
        if not r["query"].strip():
            continue
        wf = None if r["workflow"] == "none" else r["workflow"]
        step = None if r["step"] == "none" else r["step"]
        cases.append(ShadowCase(
            source="real", case_id=r["trace_id"], message=r["query"],
            current_workflow=wf, current_step=step, category="real_traffic",
        ))
    return cases


def _load_synthetic_cases() -> list[ShadowCase]:
    return [
        ShadowCase(
            source="synthetic", case_id=c["id"], message=c["message"],
            current_workflow=c.get("current_workflow"), current_step=c.get("current_step"),
            category=c["category"],
        )
        for c in SYNTHETIC_CORPUS
    ]


async def _main() -> None:
    if not os.getenv("SARVAM_API_KEY"):
        print("SARVAM_API_KEY not set -- cannot run the LLM side of this report.")
        return

    real_cases = _load_real_cases()
    synthetic_cases = _load_synthetic_cases()
    all_cases = real_cases + synthetic_cases

    retry_path = os.path.join(os.path.dirname(__file__), "_failed_ids.json")
    only_ids = None
    if "--retry-failed" in sys.argv and os.path.exists(retry_path):
        with open(retry_path, encoding="utf-8") as f:
            only_ids = set(json.load(f))
        all_cases = [c for c in all_cases if c.case_id in only_ids]
        print(f"Retrying {len(all_cases)} previously-failed cases", file=sys.stderr)

    for i, case in enumerate(all_cases):
        await _run_case(case)
        await asyncio.sleep(0.4)  # gentle pacing to avoid rate limits on a long batch
        if (i + 1) % 20 == 0:
            print(f"...{i + 1}/{len(all_cases)} done", file=sys.stderr)

    out_path = os.path.join(
        os.path.dirname(__file__), "_shadow_report_retry.json" if only_ids else "_shadow_report_data.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{
            "source": c.source, "case_id": c.case_id, "message": c.message,
            "current_workflow": c.current_workflow, "current_step": c.current_step, "category": c.category,
            "rule_intent": c.rule_intent, "rule_confidence": c.rule_confidence,
            "rule_action": c.rule_action, "rule_workflow": c.rule_workflow,
            "llm_intent": c.llm_intent, "llm_action": c.llm_action, "llm_certainty": c.llm_certainty,
            "llm_workflow": c.llm_workflow, "llm_language": c.llm_language, "llm_error": c.llm_error,
            "agree": c.agree, "would_change_result": c.would_change_result, "safety_sensitive": c.safety_sensitive,
            "is_multilingual": c.is_multilingual, "is_create_account_vs_onboarding": c.is_create_account_vs_onboarding,
            "is_cheque_dep_vs_status": c.is_cheque_dep_vs_status, "is_workflow_switch": c.is_workflow_switch,
            "is_cancel_vs_switch": c.is_cancel_vs_switch, "is_short_reply": c.is_short_reply,
            "error_attribution": c.error_attribution,
        } for c in all_cases], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_cases)} cases to {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
