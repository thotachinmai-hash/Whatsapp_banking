"""Step 3 of the LLM-first routing migration: run the corpus in
scripts/shadow_eval_corpus.py through BOTH the existing rule pipeline and
the shadow LLM router, and report where they agree, where they disagree,
and where the LLM itself gets an unambiguous case wrong.

Usage:
    python scripts/shadow_eval.py                 # full run (needs SARVAM_API_KEY)
    python scripts/shadow_eval.py --rules-only     # rule side only, no API calls
    python scripts/shadow_eval.py --id transfer_normal_en   # one case, verbose

This is a standalone diagnostic script, not part of the app or the test
suite -- it makes real network calls when SARVAM_API_KEY is set, so it is
never imported by anything else and never run by pytest.
"""

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Native-script corpus entries (Tamil/Telugu/Hindi) need UTF-8 stdout even
# on a Windows console defaulting to cp1252.
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
from scripts.shadow_eval_corpus import CORPUS  # noqa: E402


# route_intent()'s RoutingDecision vocabulary doesn't distinguish TOOL from
# RAG (that split is an agent-internal tool-choice decision even in today's
# rule pipeline -- see the migration analysis), so a rule-side "BANKING_LLM"
# is treated as satisfying either expected value below.
_BANKING_LLM_SATISFIES = {"TOOL", "RAG"}


def _rule_effective_decision(rule_intent_result, rule_routing, case: dict) -> tuple[str, Optional[str]]:
    """Re-expresses the rule pipeline's real decision in the same
    LLM_ROUTING_ACTIONS vocabulary the corpus's expected_action uses, so the
    two sides can be compared like-for-like.

    route_intent() alone doesn't know about WorkflowManager's own
    switch/cancel detection (app/workflows/manager.py), which runs BEFORE
    the router in the real pipeline and is what actually decides a switch
    in production. Reusing those exact (pure, side-effect-free) predicate
    functions here gives a fair comparison baseline instead of unfairly
    penalizing the LLM for "disagreeing" with an incomplete rule view."""
    workflow_type = case.get("current_workflow")
    query = case["message"]

    if workflow_type:
        if _is_cancel_command(query):
            return "CANCEL", None

        fake_workflow = {"type": workflow_type, "step": case.get("current_step"), "data": {}}
        if (
            rule_intent_result.intent in WORKFLOW_EXECUTING_INTENTS
            and rule_intent_result.confidence >= CONFIDENCE_HIGH
            and not _is_current_workflow_input(fake_workflow, query)
        ):
            target = get_workflow_for_intent(rule_intent_result.intent)
            if target and target != workflow_type:
                return "SWITCH", target

        # CONTINUE/CORRECT resolve to the active workflow, matching
        # route_intent()'s real behavior (RoutingDecision.workflow =
        # context.current_workflow whenever action == "WORKFLOW").
        if rule_intent_result.intent == "workflow_correction":
            return "CORRECT", workflow_type
        return "CONTINUE", workflow_type

    mapping = {
        "START_WORKFLOW": "START_WORKFLOW",
        "OUT_OF_SCOPE": "OUT_OF_SCOPE",
        "CLARIFICATION_REQUIRED": "CLARIFY",
        "BANKING_LLM": "BANKING_LLM",  # ambiguous TOOL/RAG, handled by _actions_agree()
        "SAFE_FALLBACK": "SAFE_FALLBACK",
    }
    return mapping.get(rule_routing.action, rule_routing.action), rule_routing.workflow


def _actions_agree(rule_action: str, llm_action: str) -> bool:
    if rule_action == "BANKING_LLM":
        return llm_action in _BANKING_LLM_SATISFIES
    return rule_action == llm_action


@dataclass
class CaseResult:
    case: dict
    rule_intent: Optional[str] = None
    rule_confidence: float = 0.0
    rule_action: Optional[str] = None
    rule_workflow: Optional[str] = None
    llm_intent: Optional[str] = None
    llm_action: Optional[str] = None
    llm_certainty: Optional[str] = None
    llm_workflow: Optional[str] = None
    llm_language: Optional[str] = None
    llm_skipped: bool = False
    llm_error: Optional[str] = None
    notes: list = field(default_factory=list)

    @property
    def agree(self) -> Optional[bool]:
        if self.llm_skipped or self.llm_error:
            return None
        return _actions_agree(self.rule_action, self.llm_action) and self.rule_workflow == self.llm_workflow

    @property
    def llm_matches_expected(self) -> Optional[bool]:
        expected_action = self.case.get("expected_action")
        if expected_action is None or self.llm_skipped or self.llm_error:
            return None
        if expected_action == "CANCEL":
            # to_routing_decision() deliberately ignores target_workflow for
            # CANCEL (cancellation stays handled upstream regardless of what
            # the LLM fills in there), so comparing it here would flag a
            # behaviorally-irrelevant field as a false "LLM wrong" result.
            return self.llm_action == expected_action
        expected_workflow = self.case.get("expected_target_workflow")
        return self.llm_action == expected_action and self.llm_workflow == expected_workflow

    @property
    def rule_matches_expected(self) -> Optional[bool]:
        expected_action = self.case.get("expected_action")
        if expected_action is None:
            return None
        expected_workflow = self.case.get("expected_target_workflow")
        return _actions_agree(self.rule_action, expected_action) and self.rule_workflow == expected_workflow


async def _run_case(case: dict, use_llm: bool) -> CaseResult:
    context = ConversationContext(phone_number="441111111111")
    context.current_workflow = case.get("current_workflow")
    context.current_step = case.get("current_step")

    result = CaseResult(case=case)

    rule_intent_result = await classify_intent(
        case["message"], context=context, trace_id=f"shadow_eval:{case['id']}", llm_classify=None
    )
    rule_routing = route_intent(rule_intent_result, context=context)
    effective_action, effective_workflow = _rule_effective_decision(rule_intent_result, rule_routing, case)
    result.rule_intent = rule_intent_result.intent
    result.rule_confidence = rule_intent_result.confidence
    result.rule_action = effective_action
    result.rule_workflow = effective_workflow

    if not use_llm:
        result.llm_skipped = True
        return result

    try:
        llm_decision = await classify_and_route_llm(
            case["message"], context=context, trace_id=f"shadow_eval:{case['id']}"
        )
    except Exception as e:  # pragma: no cover - diagnostic script
        result.llm_error = str(e)
        return result

    if llm_decision is None:
        result.llm_error = "no response / unparsable JSON"
        return result

    # Compared directly in LLMRoutingDecision's own action vocabulary (not
    # projected through to_routing_decision()) since that's the vocabulary
    # _rule_effective_decision() and the corpus's expected_action also use.
    result.llm_intent = llm_decision.intent
    result.llm_action = llm_decision.action
    result.llm_certainty = llm_decision.certainty
    result.llm_workflow = llm_decision.resolved_target_workflow()
    result.llm_language = llm_decision.language
    return result


def _print_case(r: CaseResult) -> None:
    c = r.case
    tag = "SKIP" if r.llm_skipped else ("ERR " if r.llm_error else ("OK  " if r.agree else "DIFF"))
    print(f"[{tag}] {c['id']:38s} ({c['category']:22s} / {c['operation']})")
    print(f"       message: {c['message']!r}")
    print(f"       rule: intent={r.rule_intent} confidence={r.rule_confidence:.2f} "
          f"-> action={r.rule_action} workflow={r.rule_workflow}")
    if r.llm_skipped:
        print("       llm:  skipped (--rules-only)")
    elif r.llm_error:
        print(f"       llm:  ERROR — {r.llm_error}")
    else:
        print(f"       llm:  intent={r.llm_intent} certainty={r.llm_certainty} lang={r.llm_language} "
              f"-> action={r.llm_action} workflow={r.llm_workflow}")
        if r.llm_matches_expected is False:
            print(f"       ** LLM disagrees with the expected (unambiguous) answer: "
                  f"expected action={c.get('expected_action')} workflow={c.get('expected_target_workflow')}")
        if r.rule_matches_expected is False:
            print(f"       ** RULE ALSO disagrees with the expected answer (pre-existing rule weakness)")
    print()


def _print_summary(results: list[CaseResult]) -> None:
    total = len(results)
    scored = [r for r in results if r.agree is not None]
    agreements = sum(1 for r in scored if r.agree)
    errors = sum(1 for r in results if r.llm_error)
    skipped = sum(1 for r in results if r.llm_skipped)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Total cases:        {total}")
    print(f"LLM skipped:        {skipped}")
    print(f"LLM call errors:    {errors}")
    if scored:
        print(f"Rule/LLM agreement: {agreements}/{len(scored)} ({100 * agreements / len(scored):.0f}%)")

    by_category = defaultdict(lambda: [0, 0])  # [agreements, scored]
    for r in scored:
        by_category[r.case["category"]][1] += 1
        if r.agree:
            by_category[r.case["category"]][0] += 1
    if by_category:
        print("\nBy category:")
        for cat, (agree_n, scored_n) in sorted(by_category.items()):
            print(f"  {cat:24s} {agree_n}/{scored_n}")

    llm_wrong = [r for r in results if r.llm_matches_expected is False]
    rule_wrong = [r for r in results if r.rule_matches_expected is False]
    if llm_wrong:
        print(f"\nLLM wrong on an unambiguous case ({len(llm_wrong)}):")
        for r in llm_wrong:
            print(f"  - {r.case['id']}: expected action={r.case.get('expected_action')} "
                  f"workflow={r.case.get('expected_target_workflow')}, got action={r.llm_action} "
                  f"workflow={r.llm_workflow}")
    if rule_wrong:
        print(f"\nRule ALSO wrong on an unambiguous case ({len(rule_wrong)}) -- pre-existing, not a new regression:")
        for r in rule_wrong:
            print(f"  - {r.case['id']}: expected action={r.case.get('expected_action')} "
                  f"workflow={r.case.get('expected_target_workflow')}, got action={r.rule_action} "
                  f"workflow={r.rule_workflow}")

    disagreements = [r for r in scored if not r.agree]
    if disagreements:
        print(f"\nRule/LLM disagreements to review ({len(disagreements)}):")
        for r in disagreements:
            print(f"  - {r.case['id']}: rule={r.rule_action}/{r.rule_workflow}  "
                  f"llm={r.llm_action}/{r.llm_workflow}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-only", action="store_true", help="Skip all LLM calls (no API key needed).")
    parser.add_argument("--id", help="Run only the case with this id.")
    args = parser.parse_args()

    cases = CORPUS if not args.id else [c for c in CORPUS if c["id"] == args.id]
    if args.id and not cases:
        print(f"No case with id={args.id!r}")
        return

    use_llm = not args.rules_only
    if use_llm and not os.getenv("SARVAM_API_KEY"):
        print("SARVAM_API_KEY not set -- running rules-only. Pass --rules-only to silence this notice,")
        print("or set SARVAM_API_KEY to get a real rule-vs-LLM comparison.\n")
        use_llm = False

    results = []
    for case in cases:
        r = await _run_case(case, use_llm)
        results.append(r)
        _print_case(r)

    _print_summary(results)


if __name__ == "__main__":
    asyncio.run(_main())
