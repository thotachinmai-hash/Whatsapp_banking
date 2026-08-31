"""Real Sarvam API validation for the LLM-first routing migration.

Runs the 101-case corpus in scripts/shadow_eval_corpus.py (all 8 banking
operations x normal/phrasing-variant/continuation/correction/cancellation/
switch/ambiguous/rag/out-of-scope/multilingual-native/romanized/mixed/
voice-transcribed) plus a small GREETING supplement against the actual,
live Sarvam API — not a mocked response — via
app/conversation/intent/llm_routing.py::classify_and_route_llm(), the
exact function app/conversation/manager.py calls in production.

This is a standalone diagnostic script, not part of the app or the pytest
suite (it makes real, billed network calls) — it is never imported by
anything else and never collected by pytest.

Usage:
    python scripts/real_sarvam_validation.py                  # full run
    python scripts/real_sarvam_validation.py --id xfer_normal  # one case
    python scripts/real_sarvam_validation.py --json out.json  # machine-readable report
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from app.conversation.context import ConversationContext  # noqa: E402
from app.conversation.intent.llm_routing import classify_and_route_llm  # noqa: E402
from scripts.shadow_eval_corpus import CORPUS  # noqa: E402

# Small supplement: the corpus predates the GREETING action (added when the
# LLM router became the sole source of greeting understanding — see
# app/conversation/intent/rules.py's module docstring). Covers English,
# native script, romanized, and "greeting + real request" (must NOT be
# GREETING) across the languages the task requires.
GREETING_SUPPLEMENT = [
    {"id": "greet_en", "category": "greeting", "operation": "none",
     "message": "Hi", "current_workflow": None, "current_step": None,
     "expected_action": "GREETING", "expected_target_workflow": None},
    {"id": "greet_hi_native", "category": "multilingual_native", "operation": "none",
     "message": "नमस्ते", "current_workflow": None, "current_step": None,
     "expected_action": "GREETING", "expected_target_workflow": None},
    {"id": "greet_ta_native", "category": "multilingual_native", "operation": "none",
     "message": "வணக்கம்", "current_workflow": None, "current_step": None,
     "expected_action": "GREETING", "expected_target_workflow": None},
    {"id": "greet_te_romanized", "category": "multilingual_romanized", "operation": "none",
     "message": "Baagunnara", "current_workflow": None, "current_step": None,
     "expected_action": "GREETING", "expected_target_workflow": None},
    {"id": "greet_plus_request_not_greeting", "category": "greeting", "operation": "CHECK_BALANCE",
     "message": "Hi, what's my balance?", "current_workflow": None, "current_step": None,
     "expected_action": "TOOL", "expected_target_workflow": None},
    {"id": "side_question_while_loan_active", "category": "switch", "operation": "CHECK_BALANCE",
     # Requirement's own worked example: a side balance question during an
     # active loan workflow must answer it (TOOL) and PRESERVE the loan
     # workflow (not CANCEL/SWITCH).
     "message": "నా ఖాతాలో ఎంత డబ్బు ఉంది?", "current_workflow": "loan", "current_step": "SELECT_LOAN_TYPE",
     "expected_action": "TOOL", "expected_target_workflow": None},
    {"id": "loan_to_create_account_switch", "category": "switch", "operation": "CREATE_ACCOUNT",
     # Requirement's own worked example: mid-loan, a clear request for a
     # different operation must pause the loan and switch.
     "message": "I want to create another account", "current_workflow": "loan", "current_step": "SELECT_LOAN_TYPE",
     "expected_action": "SWITCH", "expected_target_workflow": "add_account"},
]

FULL_CORPUS = CORPUS + GREETING_SUPPLEMENT


@dataclass
class CaseResult:
    case: dict
    llm_intent: Optional[str] = None
    llm_action: Optional[str] = None
    llm_certainty: Optional[str] = None
    llm_workflow: Optional[str] = None
    llm_language: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    entities: dict = field(default_factory=dict)

    @property
    def matches_expected(self) -> Optional[bool]:
        expected_action = self.case.get("expected_action")
        if expected_action is None or self.error:
            return None
        if expected_action in ("CANCEL", "CONTINUE", "CORRECT"):
            # The prompt deliberately tells the model to prefer leaving
            # target_workflow null for these (the active workflow is
            # already known from context, not something the LLM needs to
            # restate) -- comparing it here would flag correct behavior as
            # wrong.
            return self.llm_action == expected_action
        expected_workflow = self.case.get("expected_target_workflow")
        return self.llm_action == expected_action and self.llm_workflow == expected_workflow


async def _run_case(case: dict) -> CaseResult:
    context = ConversationContext(phone_number="441111111111")
    context.current_workflow = case.get("current_workflow")
    context.current_step = case.get("current_step")
    # Every CREATE_ACCOUNT corpus case (and any case with an active
    # workflow, which structurally implies an already-registered customer)
    # is authored assuming a REGISTERED customer -- registration_gate.py
    # already intercepts a genuinely unregistered customer before routing
    # ever sees their message, so that scenario isn't this corpus's job to
    # cover. Without this, every CREATE_ACCOUNT case's real is_registered
    # signal would silently default to False and the model would
    # (correctly, per its own prompt) sometimes prefer registration_request.
    if case.get("current_workflow") or case.get("operation") == "CREATE_ACCOUNT":
        context.is_registered = True

    result = CaseResult(case=case)
    start = time.time()
    try:
        decision = await classify_and_route_llm(
            case["message"], context=context, trace_id=f"real_sarvam_validation:{case['id']}"
        )
    except Exception as e:  # pragma: no cover - diagnostic script
        result.error = str(e)
        result.duration_ms = (time.time() - start) * 1000
        return result

    result.duration_ms = (time.time() - start) * 1000
    if decision is None:
        result.error = "no response / unparsable JSON"
        return result

    result.llm_intent = decision.intent
    result.llm_action = decision.action
    result.llm_certainty = decision.certainty
    result.llm_workflow = decision.resolved_target_workflow()
    result.llm_language = decision.language
    result.entities = decision.entities
    return result


def _print_case(r: CaseResult) -> None:
    c = r.case
    tag = "ERR " if r.error else ("PASS" if r.matches_expected else ("N/A " if r.matches_expected is None else "FAIL"))
    print(f"[{tag}] {c['id']:34s} ({c['category']:22s} / {c['operation']:20s}) {r.duration_ms:7.0f}ms")
    print(f"       message: {c['message']!r}")
    if r.error:
        print(f"       ERROR: {r.error}")
    else:
        print(f"       -> intent={r.llm_intent} action={r.llm_action} certainty={r.llm_certainty} "
              f"workflow={r.llm_workflow} language={r.llm_language} entities={r.entities}")
        if r.matches_expected is False:
            print(f"       ** expected action={c.get('expected_action')} workflow={c.get('expected_target_workflow')}")
    print()


def _summary(results: list[CaseResult]) -> dict:
    total = len(results)
    errors = [r for r in results if r.error]
    scored = [r for r in results if r.matches_expected is not None]
    passed = [r for r in scored if r.matches_expected]
    failed = [r for r in scored if r.matches_expected is False]
    durations = [r.duration_ms for r in results if not r.error]

    by_category = defaultdict(lambda: [0, 0])
    for r in scored:
        by_category[r.case["category"]][1] += 1
        if r.matches_expected:
            by_category[r.case["category"]][0] += 1

    by_operation = defaultdict(lambda: [0, 0])
    for r in scored:
        by_operation[r.case["operation"]][1] += 1
        if r.matches_expected:
            by_operation[r.case["operation"]][0] += 1

    return {
        "total_cases": total,
        "scored_cases": len(scored),
        "unscored_cases": total - len(scored) - len(errors),
        "errors": len(errors),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate_pct": round(100 * len(passed) / len(scored), 1) if scored else None,
        "avg_latency_ms": round(sum(durations) / len(durations), 1) if durations else None,
        "p95_latency_ms": round(sorted(durations)[int(len(durations) * 0.95) - 1], 1) if durations else None,
        "by_category": {k: {"passed": v[0], "scored": v[1]} for k, v in sorted(by_category.items())},
        "by_operation": {k: {"passed": v[0], "scored": v[1]} for k, v in sorted(by_operation.items())},
        "failures": [
            {
                "id": r.case["id"], "message": r.case["message"], "category": r.case["category"],
                "operation": r.case["operation"],
                "expected_action": r.case.get("expected_action"), "expected_workflow": r.case.get("expected_target_workflow"),
                "got_action": r.llm_action, "got_workflow": r.llm_workflow,
            }
            for r in failed
        ],
        "error_details": [{"id": r.case["id"], "message": r.case["message"], "error": r.error} for r in errors],
    }


def _print_summary(summary: dict) -> None:
    print("=" * 78)
    print("SUMMARY — Real Sarvam API validation")
    print("=" * 78)
    print(f"Total cases:        {summary['total_cases']}")
    print(f"Scored (has an expected answer): {summary['scored_cases']}")
    print(f"Errors (call failed):            {summary['errors']}")
    print(f"Passed:             {summary['passed']}")
    print(f"Failed:             {summary['failed']}")
    if summary["pass_rate_pct"] is not None:
        print(f"Pass rate:          {summary['pass_rate_pct']}%")
    if summary["avg_latency_ms"] is not None:
        print(f"Avg latency:        {summary['avg_latency_ms']}ms  (p95: {summary['p95_latency_ms']}ms)")

    print("\nBy category:")
    for cat, v in summary["by_category"].items():
        print(f"  {cat:24s} {v['passed']}/{v['scored']}")

    print("\nBy operation:")
    for op, v in summary["by_operation"].items():
        print(f"  {op:24s} {v['passed']}/{v['scored']}")

    if summary["failures"]:
        print(f"\nFailures ({len(summary['failures'])}):")
        for f in summary["failures"]:
            print(f"  - {f['id']}: {f['message']!r}")
            print(f"      expected action={f['expected_action']} workflow={f['expected_workflow']}, "
                  f"got action={f['got_action']} workflow={f['got_workflow']}")

    if summary["error_details"]:
        print(f"\nCall errors ({len(summary['error_details'])}):")
        for e in summary["error_details"]:
            print(f"  - {e['id']}: {e['error']}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="Run only the case with this id.")
    parser.add_argument("--json", help="Also write a machine-readable summary to this path.")
    args = parser.parse_args()

    if not os.getenv("SARVAM_API_KEY"):
        print("SARVAM_API_KEY is not set — this script requires a real key. Aborting (no results fabricated).")
        sys.exit(1)

    cases = FULL_CORPUS if not args.id else [c for c in FULL_CORPUS if c["id"] == args.id]
    if args.id and not cases:
        print(f"No case with id={args.id!r}")
        return

    results = []
    for case in cases:
        r = await _run_case(case)
        results.append(r)
        _print_case(r)

    summary = _summary(results)
    _print_summary(summary)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote machine-readable summary to {args.json}")


if __name__ == "__main__":
    asyncio.run(_main())
