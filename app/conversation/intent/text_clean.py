"""Noisy-input cleanup — runs before intent classification and workflow
keyword matching so filler/laughter, stutter-repeats, and stray punctuation
don't stop the underlying banking request from being recognized (e.g.
"check my balance ha ha ha he he he" -> "check my balance").

Deliberately conservative: this never rewrites words, never corrects
spelling, and never touches digits/currency symbols — it only strips noise
that carries no banking meaning. Anything it can't confidently identify as
noise is left alone, so a genuine (if ungrammatical) banking request still
reaches the rule layers unchanged.
"""

import re

# Standalone laughter/filler tokens — matched as whole words only, so real
# words that happen to contain these letters ("hello", "heat") are untouched.
_LAUGHTER_WORDS = {
    "ha", "haha", "hah", "hahaha", "he", "hehe", "heh", "hehehe",
    "lol", "lmao", "rofl", "lolz", "xd",
}
_FILLER_WORDS = {"um", "umm", "uh", "uhh", "erm", "err"}

_NOISE_WORDS = _LAUGHTER_WORDS | _FILLER_WORDS


def clean_noisy_text(text: str) -> str:
    """Strip laughter/filler tokens and collapse noisy punctuation/whitespace.

    Never raises and never returns something less useful than the input —
    if cleaning would remove every word, the original text is returned
    instead so a real (if entirely filler-looking) message isn't silently
    emptied out.
    """
    if not text:
        return text

    original = text

    # Collapse 3+ repeated punctuation marks down to one ("!!!" -> "!").
    cleaned = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # Drop standalone laughter/filler words (word-boundary matched).
    tokens = cleaned.split()
    kept = [t for t in tokens if re.sub(r"[^a-z]", "", t.lower()) not in _NOISE_WORDS]
    cleaned = " ".join(kept)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned if cleaned else original.strip()
