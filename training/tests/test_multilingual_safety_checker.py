"""Tests for the multilingual content checker."""

from __future__ import annotations

import inspect
import re

import pytest

try:
    from hypothesis import given, strategies as st
except ImportError:
    given = None
    st = None

from training.clinical_safety_checker import ClinicalContentAnalyzer
from training.multilingual_safety_checker import MultilingualContentChecker

# ---------------------------------------------------------------------------
# Interface & metadata tests
# ---------------------------------------------------------------------------

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
def test_production_pilot_safety_checker_disabled():
    pilot = pytest.importorskip("training.pixelated_production_pilot")
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

else:
    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_always_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_negated_not_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")
