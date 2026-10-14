"""Beat-containment gate tests (ARC_CORPUS_SPEC.md §6). No network, no writer.

Regression provenance: arc_0261 and arc_0408 burned 5 writer attempts each on
session 1 because their plan quotes narrate the client in the third person
("claims she 'never leaves the house at night'") while the client speaks in
the first person ("I never leave the house at night"). The writer planted the
beat naturally in nearly every attempt; the gate rejected it on verb
morphology alone. These tests pin both the extraction path and the
first-person tolerance.
"""

from ai.training.generate_arc_corpus import (
    _BEAT_QUOTE_RE,
    _beat_phrase_is_present,
    _containment_norm,
    _de_inflect_variants,
)


def _client_lines(*lines: str) -> str:
    """Client text as the gate sees it: containment-normalized client lines."""
    return _containment_norm("\n".join(f"[C] {line}" for line in lines))


def test_quote_extraction_from_third_person_plan_narration() -> None:
    original = "claims she 'never leaves the house at night, not once' since her mother's fall"
    assert _BEAT_QUOTE_RE.findall(original) == [
        "never leaves the house at night, not once",
    ]


def test_exact_echo_still_matches() -> None:
    # arc_0408 attempt 5 landed the beat as an exact third-person echo.
    text = _client_lines(
        "I don't text between sessions, by the way — never texts between sessions.",
    )
    assert _beat_phrase_is_present("never texts between sessions", text)


def test_first_person_plant_matches_de_inflected_verb() -> None:
    # The natural spoken form the writer produced across arc_0261 attempts.
    text = _client_lines(
        "I'm tired. That's it.",
        "I never leave the house at night, not once. Not since she fell.",
    )
    assert _beat_phrase_is_present("never leaves the house at night, not once", text)


def test_first_person_texts_beat_matches() -> None:
    text = _client_lines(
        "You know what's funny? I never text between sessions. Not once.",
    )
    assert _beat_phrase_is_present("never texts between sessions", text)


def test_es_and_ies_verbs_de_inflect() -> None:
    text = _client_lines("I never touch my phone, obviously.")
    assert _beat_phrase_is_present("never touches my phone, obviously", text)

    text = _client_lines("I deny it ever happened.")
    assert _beat_phrase_is_present("denies it ever happened", text)


def test_past_tense_is_not_a_match() -> None:
    text = _client_lines("I never left the house at night.")
    assert not _beat_phrase_is_present("never leaves the house at night, not once", text)


def test_unrelated_client_text_is_not_a_match() -> None:
    # The writer habit arc: writing unsent texts is not the planned beat.
    text = _client_lines(
        "Sometimes I write texts I don't send. To people. It's basically journaling.",
    )
    assert not _beat_phrase_is_present("never texts between sessions", text)


def test_sleep_hours_fallback_still_holds() -> None:
    # arc_0224's two-mismatch case predates the de-inflection rule and keeps
    # its word-set fallback ("sleeps ... always has" vs "sleep ... always have").
    text = _client_lines("I sleep a solid six, maybe seven hours. Always have.")
    assert _beat_phrase_is_present("sleeps a solid six, maybe seven hours, always has", text)


def test_de_inflect_variants_keep_word_boundaries() -> None:
    normalized = _containment_norm("never leaves the house at night, not once")
    assert _de_inflect_variants(normalized) == [
        "never leave the house at night, not once",
    ]
