import re

DEFAULT_SENTENCE_SPLIT: str = r"(?<=[.!?])\s+(?=[A-Z])"
DEFAULT_SENTENCE_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "viz.",
    "al.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Ave.",
    "Blvd.",
    "Dept.",
    "Est.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Oct.",
    "Nov.",
    "Dec.",
    "approx.",
    "dept.",
    "ed.",
    "esp.",
    "ex.",
    "govt.",
    "no.",
    "vol.",
    "p.",
    "pp.",
)
DEFAULT_PARAGRAPH_SPLIT: str = r"\n{2,}"

def split_sentences(
    text: str,
    *,
    sentence_split: str = DEFAULT_SENTENCE_SPLIT,
    abbreviations: tuple[str, ...] = DEFAULT_SENTENCE_ABBREVIATIONS,
) -> list[str]:
    """Split text on sentence boundaries while preserving known abbreviations."""
    if not text.strip():
        return []

    sentences = re.split(sentence_split, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    merged: list[str] = []
    abbrev_set = {a.rstrip(".").lower() for a in abbreviations}

    for s in sentences:
        last_word = (
            merged[-1].split()[-1] if merged and merged[-1].split() else ""
        )
        if (
            merged
            and not last_word
            and s.lstrip()
            and s.lstrip()[0].isupper()
        ):
            merged[-1] = f"{merged[-1]} {s}"
            continue
        if (
            merged
            and last_word.lower() in abbrev_set
            and s.lstrip()
            and s.lstrip()[0].isupper()
        ):
            merged[-1] = f"{merged[-1]} {s}"
            continue
        merged.append(s)

    return [m.strip() for m in merged if m.strip()]
