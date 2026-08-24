import json
import unittest

from ai.tools.utilities.core.pipelines.acquisition_rubric import (
    APPROVED_LICENSES,
    AcquisitionRubric,
    AcquisitionScore,
    CurationExitReport,
    GateDecision,
    PilotReport,
    PriorityTier,
    SourceIntake,
    calculate_overall_score,
    score_from_evaluation,
)


class TestAcquisitionScore(unittest.TestCase):
    def test_calculates_weighted_score(self):
        score = calculate_overall_score(
            therapeutic_relevance=8,
            data_structure_quality=7,
            training_integration=6,
            ethical_accessibility=9,
        )
        expected = round(8 * 0.35 + 7 * 0.25 + 6 * 0.20 + 9 * 0.20, 2)
        assert score.overall_score == expected

    def test_clamps_values_to_1_10(self):
        score = calculate_overall_score(
            therapeutic_relevance=0,
            data_structure_quality=15,
            training_integration=5,
            ethical_accessibility=5,
        )
        assert score.therapeutic_relevance == 1
        assert score.data_structure_quality == 10

    def test_priority_tier_high(self):
        score = calculate_overall_score(10, 10, 10, 10)
        assert score.priority_tier == PriorityTier.HIGH
        assert score.passes_score_floor

    def test_priority_tier_medium(self):
        score = calculate_overall_score(6, 6, 6, 6)
        assert score.priority_tier == PriorityTier.MEDIUM
        assert score.passes_score_floor

    def test_priority_tier_low(self):
        score = calculate_overall_score(3, 3, 3, 3)
        assert score.priority_tier == PriorityTier.LOW
        assert not score.passes_score_floor


class TestGate0Intake(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_passes_clean_source(self):
        source = SourceIntake(
            source_id="PMC-OA-2026-05",
            name="PubMed Central Open Access",
            category="academic",
            license_id="cc-by-4.0",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-08",
        )
        decision = self.rubric.evaluate_intake(source)
        assert decision.passed
        assert len(decision.blocking) == 0

    def test_blocks_unlicensed_source(self):
        source = SourceIntake(
            source_id="UNKNOWN-001",
            name="Unknown Scrape",
            category="social_media",
            license_id="unknown",
            pii_class="high",
            provenance="unknown",
            reproducible=False,
        )
        decision = self.rubric.evaluate_intake(source)
        assert not decision.passed
        assert len(decision.blocking) > 0

    def test_exception_eligible_license(self):
        source = SourceIntake(
            source_id="NC-001",
            name="Non-Commercial Dataset",
            category="academic",
            license_id="cc-by-nc-4.0",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        decision = self.rubric.evaluate_intake(source)
        assert not decision.passed
        assert any("exception" in b for b in decision.blocking)

        granted = self.rubric.grant_intake_exception(source, "approved for research use")
        assert granted.passed
        assert granted.exception_granted

    def test_approved_licenses_contains_expected(self):
        assert "cc0-1.0" in APPROVED_LICENSES
        assert "mit" in APPROVED_LICENSES
        assert "cc-by-4.0" in APPROVED_LICENSES

    def test_blocks_unreviewed_source(self):
        source = SourceIntake(
            source_id="NO-REVIEW",
            name="No Reviewer",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
        )
        decision = self.rubric.evaluate_intake(source)
        assert not decision.passed


class TestGate1Pilot(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_passes_good_pilot(self):
        report = PilotReport(
            source_id="PMC-OA-2026-05",
            sample_size=100,
            population_size=5000,
            schema_coverage_pct=98.0,
            dedup_rate=12.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=7.5,
        )
        decision = self.rubric.evaluate_pilot(report)
        assert decision.passed

    def test_blocks_low_score_floor(self):
        report = PilotReport(
            source_id="LOW-SCORE",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=96.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=7,
            overall_pilot_score=5.5,
        )
        decision = self.rubric.evaluate_pilot(report)
        assert not decision.passed
        assert "floor is non-exceptable" in decision.gates[0].details

    def test_blocks_low_relevance(self):
        report = PilotReport(
            source_id="LOW-REL",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=96.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=4,
            overall_pilot_score=6.5,
        )
        decision = self.rubric.evaluate_pilot(report)
        assert not decision.passed
        relevance_gate = next(g for g in decision.gates if g.gate == "therapeutic_relevance")
        assert relevance_gate.decision == GateDecision.BLOCK

    def test_blocks_low_schema_coverage(self):
        report = PilotReport(
            source_id="LOW-SCHEMA",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=80.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=7,
            overall_pilot_score=7.0,
        )
        decision = self.rubric.evaluate_pilot(report)
        assert not decision.passed

    def test_blocks_high_dedup(self):
        report = PilotReport(
            source_id="HIGH-DEDUP",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=97.0,
            dedup_rate=65.0,
            therapeutic_relevance_score=7,
            overall_pilot_score=7.0,
        )
        decision = self.rubric.evaluate_pilot(report)
        assert not decision.passed


class TestGate2CurationExit(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_passes_good_exit(self):
        report = CurationExitReport(
            source_id="PMC-OA-2026-05",
            net_retention_pct=52.0,
            schema_validation_pct=99.5,
            manifest_signed=True,
            records_passed=2600,
            records_rejected=2400,
        )
        decision = self.rubric.evaluate_curation_exit(report)
        assert decision.passed

    def test_blocks_low_retention(self):
        report = CurationExitReport(
            source_id="LOW-RET",
            net_retention_pct=15.0,
            schema_validation_pct=99.5,
            manifest_signed=True,
            records_passed=150,
            records_rejected=850,
        )
        decision = self.rubric.evaluate_curation_exit(report)
        assert not decision.passed

    def test_blocks_low_schema_validation(self):
        report = CurationExitReport(
            source_id="LOW-SCHEMA",
            net_retention_pct=50.0,
            schema_validation_pct=85.0,
            manifest_signed=True,
            records_passed=500,
            records_rejected=500,
        )
        decision = self.rubric.evaluate_curation_exit(report)
        assert not decision.passed

    def test_blocks_unsigned_manifest(self):
        report = CurationExitReport(
            source_id="NO-SIGN",
            net_retention_pct=50.0,
            schema_validation_pct=99.5,
            manifest_signed=False,
            records_passed=500,
            records_rejected=500,
        )
        decision = self.rubric.evaluate_curation_exit(report)
        assert not decision.passed


class TestPromote(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_promote_full_pipeline(self):
        intake = SourceIntake(
            source_id="FULL-PIPE",
            name="Full Pipeline",
            category="academic",
            license_id="cc-by-4.0",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        pilot = PilotReport(
            source_id="FULL-PIPE",
            sample_size=100,
            population_size=5000,
            schema_coverage_pct=98.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=7.5,
        )
        exit_report = CurationExitReport(
            source_id="FULL-PIPE",
            net_retention_pct=45.0,
            schema_validation_pct=99.5,
            manifest_signed=True,
            records_passed=2250,
            records_rejected=2750,
        )
        gates = self.rubric.promote(intake, pilot, exit_report)
        assert len(gates) == 3
        assert all(g.decision == GateDecision.PASS for g in gates)

    def test_promote_stops_at_intake_failure(self):
        intake = SourceIntake(
            source_id="BAD-INTAKE",
            name="Bad Intake",
            category="social_media",
            license_id="unknown",
            pii_class="high",
            provenance="unknown",
            reproducible=False,
        )
        gates = self.rubric.promote(intake)
        assert len(gates) == 1
        assert gates[0].decision == GateDecision.BLOCK

    def test_promote_stops_at_pilot_failure(self):
        intake = SourceIntake(
            source_id="BAD-PILOT",
            name="Bad Pilot",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        pilot = PilotReport(
            source_id="BAD-PILOT",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=96.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=4,
            overall_pilot_score=6.5,
        )
        gates = self.rubric.promote(intake, pilot)
        assert len(gates) == 2
        assert gates[0].decision == GateDecision.PASS
        assert gates[1].decision == GateDecision.BLOCK

    def test_promote_pilot_only(self):
        intake = SourceIntake(
            source_id="PILOT-ONLY",
            name="Pilot Only",
            category="academic",
            license_id="apache-2.0",
            pii_class="low",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        pilot = PilotReport(
            source_id="PILOT-ONLY",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=96.0,
            dedup_rate=20.0,
            therapeutic_relevance_score=7,
            overall_pilot_score=7.0,
        )
        gates = self.rubric.promote(intake, pilot)
        assert len(gates) == 2
        assert all(g.decision == GateDecision.PASS for g in gates)

    def test_promote_blocks_pilot_source_id_mismatch(self):
        intake = SourceIntake(
            source_id="INTAKE-001",
            name="Mismatch Pilot",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        pilot = PilotReport(
            source_id="WRONG-ID",
            sample_size=100,
            population_size=5000,
            schema_coverage_pct=98.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=7.5,
        )
        gates = self.rubric.promote(intake, pilot)
        assert len(gates) == 2
        assert gates[0].decision == GateDecision.PASS
        assert gates[1].decision == GateDecision.BLOCK
        assert "mismatch" in gates[1].details

    def test_promote_blocks_curation_exit_source_id_mismatch(self):
        intake = SourceIntake(
            source_id="INTAKE-002",
            name="Mismatch Exit",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        pilot = PilotReport(
            source_id="INTAKE-002",
            sample_size=100,
            population_size=5000,
            schema_coverage_pct=98.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=7.5,
        )
        exit_report = CurationExitReport(
            source_id="WRONG-EXIT",
            net_retention_pct=45.0,
            schema_validation_pct=99.5,
            manifest_signed=True,
            records_passed=2250,
            records_rejected=2750,
        )
        gates = self.rubric.promote(intake, pilot, exit_report)
        assert len(gates) == 3
        assert gates[0].decision == GateDecision.PASS
        assert gates[1].decision == GateDecision.PASS
        assert gates[2].decision == GateDecision.BLOCK
        assert "mismatch" in gates[2].details


class TestExceptionProcess(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_grant_exception_on_eligible_license(self):
        source = SourceIntake(
            source_id="EX-NC",
            name="Exception Test NC",
            category="academic",
            license_id="cc-by-nc-sa-4.0",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        decision = self.rubric.grant_intake_exception(source, "approved via exception process")
        assert decision.passed
        assert decision.exception_granted

    def test_exception_still_blocks_non_license_issue(self):
        source = SourceIntake(
            source_id="EX-MULTI",
            name="Multiple Issues",
            category="academic",
            license_id="cc-by-nc-4.0",
            pii_class="high",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        decision = self.rubric.grant_intake_exception(source, "trying exception")
        assert not decision.passed
        assert any("pii" in b.lower() for b in decision.blocking)

    def test_exception_noop_when_already_passed(self):
        source = SourceIntake(
            source_id="EX-NOOP",
            name="Already Passed",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        decision = self.rubric.grant_intake_exception(source, "would not matter")
        assert decision.passed
        assert not decision.exception_granted


class TestScoreFromEvaluation(unittest.TestCase):
    def test_matches_calculate_overall_score(self):
        direct = calculate_overall_score(8, 7, 6, 9)
        factory = score_from_evaluation(8, 7, 6, 9)
        assert direct.overall_score == factory.overall_score
        assert direct.priority_tier == factory.priority_tier


class TestToDict(unittest.TestCase):
    def test_acquisition_score_to_dict(self):
        score = AcquisitionScore(
            therapeutic_relevance=8,
            data_structure_quality=7,
            training_integration=6,
            ethical_accessibility=9,
            overall_score=7.35,
        )
        d = score.to_dict()
        assert d["overall_score"] == 7.35
        assert d["priority_tier"] == "medium"
        assert d["passes_score_floor"]

    def test_intake_decision_to_dict(self):
        src = SourceIntake(
            source_id="DICT-1",
            name="Dict Test",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        dec = self.rubric.evaluate_intake(src)
        d = dec.to_dict()
        assert d["source_id"] == "DICT-1"
        assert d["passed"]

    def setUp(self):
        self.rubric = AcquisitionRubric()


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.rubric = AcquisitionRubric()

    def test_end_to_end_accept_path(self):
        score = calculate_overall_score(8, 7, 6, 9)
        assert score.passes_score_floor
        assert score.priority_tier == PriorityTier.MEDIUM

        source = SourceIntake(
            source_id="INT-ACCEPT",
            name="Integration Accept",
            category="academic",
            license_id="cc-by-4.0",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        intake = self.rubric.evaluate_intake(source)
        assert intake.passed

        pilot = PilotReport(
            source_id="INT-ACCEPT",
            sample_size=100,
            population_size=5000,
            schema_coverage_pct=97.0,
            dedup_rate=15.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=score.overall_score,
        )
        pilot_dec = self.rubric.evaluate_pilot(pilot)
        assert pilot_dec.passed

        exit_rpt = CurationExitReport(
            source_id="INT-ACCEPT",
            net_retention_pct=42.0,
            schema_validation_pct=99.5,
            manifest_signed=True,
            records_passed=2100,
            records_rejected=2900,
        )
        exit_dec = self.rubric.evaluate_curation_exit(exit_rpt)
        assert exit_dec.passed

        promote_gates = self.rubric.promote(source, pilot, exit_rpt)
        assert len(promote_gates) == 3
        assert all(g.decision == GateDecision.PASS for g in promote_gates)

    def test_end_to_end_exception_path(self):
        score = calculate_overall_score(7, 8, 7, 8)
        assert score.passes_score_floor

        source = SourceIntake(
            source_id="INT-EXCEPTION",
            name="Integration Exception",
            category="academic",
            license_id="cc-by-nc-4.0",
            pii_class="low",
            provenance="verified",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        intake = self.rubric.grant_intake_exception(source, "non-commercial license approved")
        assert intake.passed
        assert intake.exception_granted

        pilot = PilotReport(
            source_id="INT-EXCEPTION",
            sample_size=80,
            population_size=4000,
            schema_coverage_pct=96.0,
            dedup_rate=22.0,
            therapeutic_relevance_score=7,
            overall_pilot_score=score.overall_score,
        )
        assert self.rubric.evaluate_pilot(pilot).passed

    def test_end_to_end_reject_path(self):
        score = calculate_overall_score(3, 4, 3, 5)
        assert not score.passes_score_floor

        source = SourceIntake(
            source_id="INT-REJECT",
            name="Integration Reject",
            category="social_media",
            license_id="unknown",
            pii_class="critical",
            provenance="unknown",
            reproducible=False,
        )
        intake = self.rubric.evaluate_intake(source)
        assert not intake.passed
        assert len(intake.blocking) > 0

        promote_gates = self.rubric.promote(source)
        assert promote_gates[0].decision == GateDecision.BLOCK

    def test_dict_serialization_roundtrip(self):
        source = SourceIntake(
            source_id="SER-TEST",
            name="Serialization Test",
            category="academic",
            license_id="mit",
            pii_class="none",
            provenance="trusted",
            reproducible=True,
            reviewer="chad",
            review_date="2026-05-09",
        )
        intake = self.rubric.evaluate_intake(source)
        serialized = json.dumps(intake.to_dict())
        restored = json.loads(serialized)
        assert restored["source_id"] == "SER-TEST"
        assert restored["passed"]

        pilot = PilotReport(
            source_id="SER-TEST",
            sample_size=50,
            population_size=1000,
            schema_coverage_pct=97.0,
            dedup_rate=10.0,
            therapeutic_relevance_score=8,
            overall_pilot_score=7.5,
        )
        pilot_dec = self.rubric.evaluate_pilot(pilot)
        serialized_pilot = json.dumps(pilot_dec.to_dict())
        restored_pilot = json.loads(serialized_pilot)
        assert restored_pilot["source_id"] == "SER-TEST"
        assert restored_pilot["passed"]


if __name__ == "__main__":
    unittest.main()
