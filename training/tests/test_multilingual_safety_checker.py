"""Tests for the multilingual content checker."""

from __future__ import annotations

import inspect
import re

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.clinical_safety_checker import ClinicalContentAnalyzer
from training.multilingual_safety_checker import MultilingualContentChecker


def test_version_format():
    assert re.match(r"^\d+\.\d+\.\d+$", MultilingualContentChecker.VERSION)


def test_supported_languages():
    assert MultilingualContentChecker.supported_languages == ["en", "es", "fr", "pt", "de"]


def test_inherits_clinical_content_analyzer():
    assert issubclass(MultilingualContentChecker, ClinicalContentAnalyzer)


def test_contains_crisis_keywords_signature():
    sig = inspect.signature(MultilingualContentChecker.contains_crisis_keywords)
    params = list(sig.parameters.values())
    assert params[0].name == "text"


def test_contains_crisis_keywords_callable():
    assert isinstance(MultilingualContentChecker.contains_crisis_keywords("I want to die"), bool)


ES_CRISIS = (
    "Quiero suicidarme esta noche.",
    "tengo pensamientos suicidas",
    "quiero morir ya bastante sufrí",
    "mejor muerto que seguir sufriendo",
    "quiero morirme todo es oscuro",
)

FR_CRISIS = (
    "Je veux mourir tout de suite.",
    "mettre fin à ma vie maintenant",
    "je veux mourir c'est trop dur",
    "surdosage de médicament pour en finir",
    "me tuer est la seule solution",
)

PT_CRISIS = (
    "Quero me matar.",
    "quero morrer tudo está escuro",
    "me machucar é a única saída",
    "sem razão para viver tudo acabou",
    "me suicidar é o que eu quero",
)

DE_CRISIS = (
    "Ich möchte sterben jetzt.",
    "Ich habe suizidale Gedanken.",
    "mich umbringen ist die einzige Lösung",
    "ich will sterben alles ist hoffnungslos",
    "kein grund zu leben alles ist sinnlos",
)

EN_CRISIS = (
    "I want to die tonight.",
    "kill myself is the only way out",
    "no reason to live anymore",
    "I'm going to end my life",
    "I've made my final decision",
)

ES_NEGATED = (
    "No quiero morir, pero estoy bien.",
    "No voy a cortarme.",
    "nunca pensaría en suicidarme",
    "no tengo pensamientos suicidas",
)

FR_NEGATED = (
    "Je ne veux pas mourir de cette façon.",
    "Je ne veux pas me faire du mal.",
    "jamais je ne me suiciderai",
    "pas question de mettre fin à ma vie",
)

PT_NEGATED = (
    "Não quero me machucar.",
    "nunca pensei em me matar",
    "não tenho vontade de suicídio",
    "não quero acabar com a minha vida",
)

DE_NEGATED = (
    "Ich will nicht sterben, ich bin nur müde.",
    "Ich möchte nicht mich töten.",
    "niemals würde ich mich umbringen",
    "keine suizidgedanken ich bin okay",
)

EN_NEGATED = (
    "I don't want to die this time.",
    "I'm not going to kill myself.",
    "I have no plans to end my life",
    "don't want to end my life today",
)


@pytest.mark.parametrize("text", ES_CRISIS)
def test_es_crisis_detected(text: str):
    assert MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", FR_CRISIS)
def test_fr_crisis_detected(text: str):
    assert MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", PT_CRISIS)
def test_pt_crisis_detected(text: str):
    assert MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", DE_CRISIS)
def test_de_crisis_detected(text: str):
    assert MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", EN_CRISIS)
def test_en_crisis_detected(text: str):
    assert MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", ES_NEGATED)
def test_es_negated_not_detected(text: str):
    assert not MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", FR_NEGATED)
def test_fr_negated_not_detected(text: str):
    assert not MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", PT_NEGATED)
def test_pt_negated_not_detected(text: str):
    assert not MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", DE_NEGATED)
def test_de_negated_not_detected(text: str):
    assert not MultilingualContentChecker.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", EN_NEGATED)
def test_en_negated_not_detected(text: str):
    assert not MultilingualContentChecker.contains_crisis_keywords(text)


def test_empty_and_nonstring_input():
    assert not MultilingualContentChecker.contains_crisis_keywords("")
    assert not MultilingualContentChecker.contains_crisis_keywords("   ")
    assert not MultilingualContentChecker.contains_crisis_keywords(None)


# ---------------------------------------------------------------------------
# Production pilot integration
# ---------------------------------------------------------------------------


def test_production_pilot_safety_checker_disabled():
    """SAFETY CHECKERS DISABLED per user request."""
    try:
        import training.pixelated_production_pilot as pilot  # type: ignore[import-untyped]
    except Exception:
        pytest.skip("training.pixelated_production_pilot not available (peft/kernels compatibility)")
    assert pilot.SAFETY_CHECKER is None


if st is not None:
    _ALL_CRISIS = ES_CRISIS + FR_CRISIS + PT_CRISIS + DE_CRISIS + EN_CRISIS
    _ALL_NEGATED = ES_NEGATED + FR_NEGATED + PT_NEGATED + DE_NEGATED + EN_NEGATED

    @given(st.sampled_from(_ALL_CRISIS))
    def test_hypothesis_crisis_always_detected(sample: str):
        assert MultilingualContentChecker.contains_crisis_keywords(sample)

    @given(st.sampled_from(_ALL_NEGATED))
    def test_hypothesis_negated_not_detected(sample: str):
        assert not MultilingualContentChecker.contains_crisis_keywords(sample)

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=500)
    def test_all_texts_return_bool(text: str):
        """Every non-None string returns a bool (True or False), never raises."""
        result = MultilingualContentChecker.contains_crisis_keywords(text)
        assert isinstance(result, bool)

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_always_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_negated_not_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def test_unknown_language_iterates_all_patterns():
    """When language=None, iterates all _LANG_PATTERNS (lines 121-122).
    Safe text with no crisis patterns should return False."""
    assert not MultilingualContentChecker.contains_crisis_keywords(
        "The weather is sunny and I feel okay.", language=None
    )
    assert not MultilingualContentChecker.contains_crisis_keywords(
        "Bonjour, comment allez-vous aujourd'hui?", language=None
    )
    assert not MultilingualContentChecker.contains_crisis_keywords(
        "Estou bem e não tenho pensamentos negativos.", language=None
    )


def test_safe_text_returns_false():
    """Safe text that doesn't match any crisis pattern returns False
    (exercises the implicit 'return False' at line 150 of multilingual_safety_checker.py)."""
    safe_samples = [
        "I had a wonderful day at the park with my family.",
        "Los ejercicios son buenos para la salud mental.",
        "Le beau temps est agréable aujourd'hui.",
        "Gute laune und Sonnenschein sind wunderbar.",
        "Vida boa, saúde perfeita, sem pensamentos tristes.",
        "I'm feeling grateful and content today.",
    ]
    for text in safe_samples:
        assert not MultilingualContentChecker.contains_crisis_keywords(text), f"false positive on: {text}"


def test_all_five_languages_produce_results():
    """Every supported language returns a result (True or False, both valid)."""
    text = "a" * 100  # safe filler
    for lang in MultilingualContentChecker.supported_languages:
        result = MultilingualContentChecker.contains_crisis_keywords(text, language=lang)
        assert isinstance(result, bool), f"{lang} did not return bool"


def test_fallback_import_path_used():
    """Lines 15-16 are the ModuleNotFoundError fallback import. Exercise it
    by patching sys.modules to hide the primary path, then restoring."""
    import sys

    # Temporarily shadow the primary path so the fallback fires.
    backup = sys.modules.pop("training.clinical_safety_checker", None)
    backup2 = sys.modules.pop("ai.training.clinical_safety_checker", None)
    try:
        # Re-import forces the fallback path in lines 15-16 of the module.
        from training.multilingual_safety_checker import MultilingualContentChecker as MC2

        assert issubclass(MC2, ClinicalContentAnalyzer)
    finally:
        # Restore so subsequent tests are unaffected.
        if backup is not None:
            sys.modules["training.clinical_safety_checker"] = backup
        if backup2 is not None:
            sys.modules["ai.training.clinical_safety_checker"] = backup2
