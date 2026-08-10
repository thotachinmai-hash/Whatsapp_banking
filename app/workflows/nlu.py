"""Shared free-form confirmation interpretation for workflow steps.

Replaces the exact-string "yes"/"no" checks duplicated across cheque,
loan, KYC, onboarding, and transfer processors (each previously had its
own `answer in {"yes", "y", "confirm"}` / `{"no", "n", "cancel"}` check)
so natural phrasing ("yeah go ahead," "nah, that's wrong, cancel it")
works the same everywhere, not just wherever someone happened to widen it.

Rule-based only for now — the phrase/pattern lists below are wide enough
to cover natural confirmation/denial phrasing without adding per-message
LLM latency or cost. If a future case needs it, an optional LLM fallback
can be added the same way app/conversation/intent/classifier.py's
`llm_classify` parameter works: called only when the rules return None.
"""

import re

# Idioms that contain a denial-looking word ("no") but function as an
# affirmative/neutral-positive reply — checked before the deny pattern so
# "no problem, go ahead" isn't misread as a cancellation.
_AFFIRM_IDIOMS = ("no problem", "no worries", "not a problem")

_AFFIRM_PATTERNS = (
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\byup\b", r"\bsure\b",
    r"\bconfirm(ed)?\b", r"\bcorrect\b", r"\bgo ?ahead\b", r"\bproceed\b",
    r"\bcontinue\b", r"\bkeep going\b", r"\bresume\b", r"\bcarry on\b",
    r"\bsounds good\b", r"\ball good\b", r"\blooks (good|right)\b",
    r"\bthat'?s (right|correct)\b", r"^y$", r"^ok(ay)?$",
)
_DENY_PATTERNS = (
    r"\bno\b", r"\bnah\b", r"\bnope\b", r"\bcancel\b", r"\bstop\b",
    r"\bwrong\b", r"\bincorrect\b", r"\bnot (right|correct)\b",
    r"\bdon'?t\b", r"\bdo not\b", r"\bnever ?mind\b", r"^n$",
)

_AFFIRM_RE = re.compile("|".join(_AFFIRM_PATTERNS), re.I)
_DENY_RE = re.compile("|".join(_DENY_PATTERNS), re.I)


def interpret_confirmation(text: str) -> str | None:
    """Return "yes", "no", or None (genuinely unclear) for a free-form
    reply to a confirmation prompt.

    Deliberately returns None rather than guessing when both an affirm and
    a deny signal are present ("yeah no don't") — a banking confirmation
    step should re-ask rather than act on an ambiguous reply.
    """
    if not text or not text.strip():
        return None

    normalized = re.sub(r"[^a-z' ]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    if any(normalized.startswith(idiom) for idiom in _AFFIRM_IDIOMS):
        return "yes"

    deny = bool(_DENY_RE.search(normalized))
    affirm = bool(_AFFIRM_RE.search(normalized))

    if deny and not affirm:
        return "no"
    if affirm and not deny:
        return "yes"
    return None
