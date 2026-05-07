"""Tests for the multilingual safety checker."""

from __future__ import annotations

import inspect
import re

import pytest

try:
    from hypothesis import given, strategies as st
except ImportError:  # pragma: no cover
    given = None
    st = None

from training.clinical_safety_checker import ClinicalSafetyChecker
from training.multilingual_safety_checker import MultilingualSafetyChecker

# ---------------------------------------------------------------------------
# Interface & metadata tests
# ---------------------------------------------------------------------------

def test_version_format():
    """VERSION follows MAJOR.MINOR.PATCH."""
    assert re.match(r"^\d+\.\d+\.\d+$", MultilingualSafetyChecker.VERSION)


def test_supported_languages():
    """supported_languages contains exactly the five expected codes."""
    assert MultilingualSafetyChecker.supported_languages == ["en", "es", "fr", "pt", "de"]


def test_inherits_clinical_safety_checker():
    """MultilingualSafetyChecker extends ClinicalSafetyChecker."""
    assert issubclass(MultilingualSafetyChecker, ClinicalSafetyChecker)


def test_is_unsafe_signature_compatible():
    """is_unsafe accepts (text, language=None) — backwards compatible with parent."""
    sig = inspect.signature(MultilingualSafetyChecker.is_unsafe)
    params = list(sig.parameters.values())
    assert params[0].name == "text"
    assert params[1].name == "language"
    assert params[1].default is None


def test_is_unsafe_callable_with_single_arg():
    """Calling is_unsafe with just text works (backwards compat)."""
    assert isinstance(MultilingualSafetyChecker.is_unsafe("I want to die"), bool)


# ---------------------------------------------------------------------------
# Per-language crisis expressions (3+ per language = 15 minimum)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Per-language negated expressions (2+ per language = 10 minimum)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parametrized unit tests — crisis detection per language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ES_CRISIS)
def test_es_crisis_detected(text: str):
    assert MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", FR_CRISIS)
def test_fr_crisis_detected(text: str):
    assert MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", PT_CRISIS)
def test_pt_crisis_detected(text: str):
    assert MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", DE_CRISIS)
def test_de_crisis_detected(text: str):
    assert MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", EN_CRISIS)
def test_en_crisis_detected(text: str):
    assert MultilingualSafetyChecker.is_unsafe(text)


# ---------------------------------------------------------------------------
# Parametrized unit tests — negation suppression per language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ES_NEGATED)
def test_es_negated_not_unsafe(text: str):
    assert not MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", FR_NEGATED)
def test_fr_negated_not_unsafe(text: str):
    assert not MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", PT_NEGATED)
def test_pt_negated_not_unsafe(text: str):
    assert not MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", DE_NEGATED)
def test_de_negated_not_unsafe(text: str):
    assert not MultilingualSafetyChecker.is_unsafe(text)


@pytest.mark.parametrize("text", EN_NEGATED)
def test_en_negated_not_unsafe(text: str):
    assert not MultilingualSafetyChecker.is_unsafe(text)


# ---------------------------------------------------------------------------
# Language-scoped is_unsafe tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", MultilingualSafetyChecker.supported_languages)
def test_language_param_crisis(lang: str):
    """Passing language explicitly still detects crisis in that language."""
    crisis_map = {
        "en": "I want to die",
        "es": "quiero suicidarme",
        "fr": "je veux mourir",
        "pt": "quero morrer",
        "de": "ich will sterben",
    }
    assert MultilingualSafetyChecker.is_unsafe(crisis_map[lang], language=lang)


def test_unknown_language_falls_back_to_all():
    """Unknown language code checks all pattern sets."""
    assert MultilingualSafetyChecker.is_unsafe("I want to die", language="xx")


def test_empty_and_nonstring_input():
    assert not MultilingualSafetyChecker.is_unsafe("")
    assert not MultilingualSafetyChecker.is_unsafe("   ")
    assert not MultilingualSafetyChecker.is_unsafe(None)


# ---------------------------------------------------------------------------
# Production pilot integration
# ---------------------------------------------------------------------------

def test_production_pilot_uses_multilingual_safety_checker():
    pilot = pytest.importorskip("training.pixelated_production_pilot")
    assert pilot.SAFETY_CHECKER.__name__ == "MultilingualSafetyChecker"
    assert hasattr(pilot.SAFETY_CHECKER, "VERSION")


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    _ALL_CRISIS = ES_CRISIS + FR_CRISIS + PT_CRISIS + DE_CRISIS + EN_CRISIS
    _ALL_NEGATED = ES_NEGATED + FR_NEGATED + PT_NEGATED + DE_NEGATED + EN_NEGATED

    @given(st.sampled_from(_ALL_CRISIS))
    def test_hypothesis_crisis_always_unsafe(sample: str):
        assert MultilingualSafetyChecker.is_unsafe(sample)

    @given(st.sampled_from(_ALL_NEGATED))
    def test_hypothesis_negated_never_unsafe(sample: str):
        assert not MultilingualSafetyChecker.is_unsafe(sample)

    _LANG_CRISIS_MAP = {
        "en": EN_CRISIS,
        "es": ES_CRISIS,
        "fr": FR_CRISIS,
        "pt": PT_CRISIS,
        "de": DE_CRISIS,
    }

    @given(
        lang_crisis=st.sampled_from(
            [(lang, text) for lang, texts in _LANG_CRISIS_MAP.items() for text in texts]
        ),
    )
    def test_hypothesis_crisis_detected_with_explicit_lang(lang_crisis: tuple[str, str]):
        lang, crisis = lang_crisis
        assert MultilingualSafetyChecker.is_unsafe(crisis, language=lang)

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_always_unsafe():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_negated_never_unsafe():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_detected_with_explicit_lang():
        raise AssertionError("Skipped when hypothesis is unavailable")
