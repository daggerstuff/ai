#!/usr/bin/env python3
"""Test suite for PATIENT-Ψ 8-component CCD data model."""

import unittest

import pytest
from pydantic import ValidationError

from ai.pkg_mera.platform.patient_psi.schema import (
    BehavioralResponse,
    CognitiveTriad,
    CompensatoryStrategy,
    CopingStrategy,
    CoreBelief,
    EmotionalResponse,
    IntermediateBelief,
    PatientCCD,
    SituationInterpretation,
)


class TestCognitiveTriad(unittest.TestCase):
    """Test CognitiveTriad model."""

    def test_creation(self):
        triad = CognitiveTriad(self_views=0.8, world_views=0.3, future_views=0.5)
        assert triad.self_views == 0.8
        assert triad.world_views == 0.3
        assert triad.future_views == 0.5

    def test_out_of_range_above(self):
        with pytest.raises(ValidationError):
            CognitiveTriad(self_views=1.5, world_views=0.5, future_views=0.5)

    def test_out_of_range_below(self):
        with pytest.raises(ValidationError):
            CognitiveTriad(self_views=-0.1, world_views=0.5, future_views=0.5)

    def test_boundary_values(self):
        triad = CognitiveTriad(self_views=0.0, world_views=1.0, future_views=0.5)
        assert triad.self_views == 0.0
        assert triad.world_views == 1.0
        assert triad.future_views == 0.5


class TestCoreBelief(unittest.TestCase):
    """Test CoreBelief model."""

    def test_creation(self):
        belief = CoreBelief(
            content="I am worthless",
            domain="self",
            conviction=0.9,
        )
        assert belief.content == "I am worthless"
        assert belief.domain == "self"
        assert belief.conviction == 0.9

    def test_all_domains(self):
        for domain in ("self", "others", "world", "future"):
            belief = CoreBelief(content=f"test {domain}", domain=domain, conviction=0.5)
            assert belief.domain == domain

    def test_invalid_domain(self):
        with pytest.raises(ValidationError):
            CoreBelief(content="test", domain="invalid", conviction=0.5)

    def test_invalid_conviction(self):
        with pytest.raises(ValidationError):
            CoreBelief(content="test", domain="self", conviction=1.5)


class TestIntermediateBelief(unittest.TestCase):
    """Test IntermediateBelief model."""

    def test_creation(self):
        belief = IntermediateBelief(
            content="I must be perfect to be accepted",
            rule_type="rule",
            conviction=0.8,
        )
        assert belief.content == "I must be perfect to be accepted"
        assert belief.rule_type == "rule"
        assert belief.conviction == 0.8

    def test_all_rule_types(self):
        for rtype in ("rule", "attitude", "assumption"):
            belief = IntermediateBelief(content=f"test {rtype}", rule_type=rtype, conviction=0.5)
            assert belief.rule_type == rtype

    def test_invalid_rule_type(self):
        with pytest.raises(ValidationError):
            IntermediateBelief(content="test", rule_type="belief", conviction=0.5)

    def test_invalid_conviction(self):
        with pytest.raises(ValidationError):
            IntermediateBelief(content="test", rule_type="rule", conviction=-0.1)


class TestCopingStrategy(unittest.TestCase):
    """Test CopingStrategy model."""

    def test_creation(self):
        strategy = CopingStrategy(
            content="Avoids social gatherings",
            strategy_type="avoidance",
            effectiveness=0.3,
        )
        assert strategy.content == "Avoids social gatherings"
        assert strategy.strategy_type == "avoidance"
        assert strategy.effectiveness == 0.3

    def test_all_strategy_types(self):
        for stype in ("avoidance", "compensation", "overcompensation"):
            strategy = CopingStrategy(content=f"test {stype}", strategy_type=stype, effectiveness=0.5)
            assert strategy.strategy_type == stype

    def test_invalid_strategy_type(self):
        with pytest.raises(ValidationError):
            CopingStrategy(content="test", strategy_type="denial", effectiveness=0.5)

    def test_invalid_effectiveness(self):
        with pytest.raises(ValidationError):
            CopingStrategy(content="test", strategy_type="avoidance", effectiveness=2.0)


class TestCompensatoryStrategy(unittest.TestCase):
    """Test CompensatoryStrategy model."""

    def test_creation(self):
        strategy = CompensatoryStrategy(
            content="Excessive reassurance seeking",
            behavior="asks others for validation",
            overcompensation_for="self_doubt",
        )
        assert strategy.content == "Excessive reassurance seeking"
        assert strategy.behavior == "asks others for validation"
        assert strategy.overcompensation_for == "self_doubt"

    def test_optional_overcompensation(self):
        strategy = CompensatoryStrategy(
            content="Overworking",
            behavior="works 12-hour days",
        )
        assert strategy.overcompensation_for is None


class TestSituationInterpretation(unittest.TestCase):
    """Test SituationInterpretation model."""

    def test_creation(self):
        si = SituationInterpretation(
            situation="Receiving criticism",
            interpretation="I have failed completely",
            distortion_type="all-or-nothing",
        )
        assert si.situation == "Receiving criticism"
        assert si.interpretation == "I have failed completely"
        assert si.distortion_type == "all-or-nothing"

    def test_optional_distortion(self):
        si = SituationInterpretation(
            situation="Getting a compliment",
            interpretation="They are just being nice",
        )
        assert si.distortion_type is None


class TestEmotionalResponse(unittest.TestCase):
    """Test EmotionalResponse model."""

    def test_creation(self):
        er = EmotionalResponse(emotion="sadness", intensity=0.8, valence="negative")
        assert er.emotion == "sadness"
        assert er.intensity == 0.8
        assert er.valence == "negative"

    def test_all_valences(self):
        for valence in ("positive", "negative", "mixed"):
            er = EmotionalResponse(emotion="test", intensity=0.5, valence=valence)
            assert er.valence == valence

    def test_invalid_valence(self):
        with pytest.raises(ValidationError):
            EmotionalResponse(emotion="test", intensity=0.5, valence="neutral")

    def test_invalid_intensity(self):
        with pytest.raises(ValidationError):
            EmotionalResponse(emotion="test", intensity=1.1, valence="negative")


class TestBehavioralResponse(unittest.TestCase):
    """Test BehavioralResponse model."""

    def test_creation(self):
        br = BehavioralResponse(
            behavior="Social withdrawal",
            triggered_by="feeling inadequate",
            consequence="Increased isolation",
        )
        assert br.behavior == "Social withdrawal"
        assert br.triggered_by == "feeling inadequate"
        assert br.consequence == "Increased isolation"

    def test_optional_consequence(self):
        br = BehavioralResponse(
            behavior="Crying",
            triggered_by="overwhelming sadness",
        )
        assert br.consequence is None


class TestPatientCCD(unittest.TestCase):
    """Test PatientCCD composite model."""

    def test_empty_creation(self):
        ccd = PatientCCD(client_id="test_001")
        assert ccd.client_id == "test_001"
        assert ccd.core_beliefs == []
        assert ccd.intermediate_beliefs == []
        assert ccd.coping_strategies == []
        assert ccd.compensatory_strategies == []
        assert ccd.situation_interpretations == []
        assert ccd.emotional_responses == []
        assert ccd.behavioral_responses == []
        assert ccd.triads is None

    def test_fully_populated(self):
        ccd = PatientCCD(
            client_id="populated_001",
            triads=CognitiveTriad(self_views=0.9, world_views=0.8, future_views=0.85),
            core_beliefs=[
                CoreBelief(content="I am capable", domain="self", conviction=0.8),
                CoreBelief(content="People are trustworthy", domain="others", conviction=0.7),
            ],
            intermediate_beliefs=[
                IntermediateBelief(content="Effort leads to success", rule_type="attitude", conviction=0.9),
            ],
            coping_strategies=[
                CopingStrategy(content="Exercise", strategy_type="compensation", effectiveness=0.7),
            ],
            compensatory_strategies=[
                CompensatoryStrategy(content="Over-prepare", behavior="excessive planning"),
            ],
            situation_interpretations=[
                SituationInterpretation(
                    situation="Public speaking",
                    interpretation="I will do well",
                    distortion_type=None,
                ),
            ],
            emotional_responses=[
                EmotionalResponse(emotion="joy", intensity=0.7, valence="positive"),
            ],
            behavioral_responses=[
                BehavioralResponse(behavior="Smile", triggered_by="feeling happy"),
            ],
        )
        assert len(ccd.core_beliefs) == 2
        assert len(ccd.intermediate_beliefs) == 1
        assert len(ccd.coping_strategies) == 1
        assert len(ccd.compensatory_strategies) == 1
        assert len(ccd.situation_interpretations) == 1
        assert len(ccd.emotional_responses) == 1
        assert len(ccd.behavioral_responses) == 1

    def test_add_core_belief(self):
        ccd = PatientCCD(client_id="test")
        belief = ccd.add_core_belief("I am worthless", "self", conviction=0.9)
        assert isinstance(belief, CoreBelief)
        assert belief.content == "I am worthless"
        assert len(ccd.core_beliefs) == 1

    def test_add_emotional_response(self):
        ccd = PatientCCD(client_id="test")
        er = ccd.add_emotional_response("anxiety", 0.8, "negative")
        assert isinstance(er, EmotionalResponse)
        assert er.emotion == "anxiety"
        assert len(ccd.emotional_responses) == 1

    def test_add_behavioral_response(self):
        ccd = PatientCCD(client_id="test")
        br = ccd.add_behavioral_response("Avoidance", "anxiety", consequence="relief")
        assert isinstance(br, BehavioralResponse)
        assert br.behavior == "Avoidance"
        assert br.consequence == "relief"
        assert len(ccd.behavioral_responses) == 1

    def test_add_coping_strategy(self):
        ccd = PatientCCD(client_id="test")
        cs = ccd.add_coping_strategy("Seek reassurance", "compensation", effectiveness=0.4)
        assert isinstance(cs, CopingStrategy)
        assert cs.content == "Seek reassurance"
        assert cs.strategy_type == "compensation"
        assert len(ccd.coping_strategies) == 1

    def test_get_negative_triad_score_no_triad(self):
        ccd = PatientCCD(client_id="test")
        assert ccd.get_negative_triad_score() == 0.5

    def test_get_negative_triad_score_with_triad(self):
        ccd = PatientCCD(
            client_id="test",
            triads=CognitiveTriad(self_views=0.1, world_views=0.2, future_views=0.3),
        )
        # (0.9 + 0.8 + 0.7) / 3 = 0.8
        expected = (0.9 + 0.8 + 0.7) / 3.0
        assert abs(ccd.get_negative_triad_score() - expected) < 0.001

    def test_to_dict_round_trip(self):
        ccd = PatientCCD(
            client_id="rt",
            triads=CognitiveTriad(self_views=0.5, world_views=0.5, future_views=0.5),
            core_beliefs=[CoreBelief(content="test", domain="self", conviction=0.5)],
        )
        d = ccd.to_dict()
        assert d["client_id"] == "rt"
        assert d["triads"]["self_views"] == 0.5
        assert len(d["core_beliefs"]) == 1

    def test_model_validate_round_trip(self):
        ccd = PatientCCD(client_id="vt", triads=CognitiveTriad(self_views=0.3, world_views=0.4, future_views=0.5))
        d = ccd.to_dict()
        restored = PatientCCD.model_validate(d)
        assert restored.client_id == "vt"
        assert restored.triads is not None
        assert restored.triads.self_views == 0.3

    def test_get_high_conviction_beliefs(self):
        ccd = PatientCCD(client_id="test")
        ccd.add_core_belief("Low conviction", "self", conviction=0.3)
        ccd.add_core_belief("High conviction", "world", conviction=0.9)
        ccd.add_core_belief("Medium conviction", "others", conviction=0.7)
        high = ccd.get_high_conviction_beliefs(threshold=0.7)
        assert len(high) == 2
        assert high[0].content == "High conviction"
        assert high[1].content == "Medium conviction"


if __name__ == "__main__":
    unittest.main()
