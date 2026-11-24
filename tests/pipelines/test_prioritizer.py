"""Tests for the PIX-4747 evidence-based prioritization stack.

Covers the spec's verification and testing checklist:

* Mathematical verification: higher-impact performance anomalies rank
  above lower-impact ones across a 5,000-item population.
* Performance: O(N log N) rank-ordering handles 5,000 items in < 100ms.
* Robustness: randomised weight perturbations preserve rank-ordering
  constraints (no top-quartile item drops below the quartile cut).
* Consistency audit: calculated priorities match recorded ticket fields.
* Priority Sync Bridge: tier -> Highest/High/Medium/Low/Lowest mapping.
* Evidence Collector: dict/JSON ingest of the three evidence streams.
"""

import random
import time
import unittest
from unittest import mock

import numpy as np

from ai.tools.utilities.pipelines.evidence_collector import EvidenceCollector, PostgresEvidenceCollector
from ai.tools.utilities.pipelines.priority_bridge import (
    HIGH,
    HIGHEST,
    LOW,
    LOWEST,
    MEDIUM,
    TIER_TO_LABEL,
    ConsistencyAuditReport,
    PriorityScale,
    PrioritySyncBridge,
)
from ai.tools.utilities.pipelines.reprioritization_engine import PriorityTier
from ai.tools.utilities.pipelines.scoring import (
    CRITERIA_COLUMNS,
    MultiCriteriaScorer,
    PriorityWeights,
    TicketSignal,
)


def _make_ticket(item_id: str, **kwargs) -> TicketSignal:
    defaults = {
        "title": f"Ticket {item_id}",
        "severity": "medium",
        "frequency_score": 0.5,
        "performance_gap": 0.5,
        "user_facing_impact": 0.5,
    }
    defaults.update(kwargs)
    return TicketSignal(item_id=item_id, **defaults)


def _random_population(n: int, seed: int) -> list[TicketSignal]:
    rng = random.Random(seed)
    severities = ("low", "medium", "high", "critical")
    return [
        _make_ticket(
            f"item-{i:05d}",
            severity=rng.choice(severities),
            frequency_score=rng.random(),
            performance_gap=rng.random(),
            user_facing_impact=rng.random(),
        )
        for i in range(n)
    ]


class TestScoringMathematics(unittest.TestCase):
    """Higher-impact performance anomalies must rank above lower-impact ones."""

    def test_higher_impact_anomaly_ranks_above_lower_impact(self):
        rng = random.Random(42)
        population = [_make_ticket(f"low-impact-{i:03d}", performance_gap=rng.uniform(0.05, 0.2)) for i in range(100)]
        anomaly = _make_ticket(
            "high-impact-anomaly",
            severity="critical",
            performance_gap=0.98,
            user_facing_impact=0.95,
            frequency_score=0.9,
        )
        tickets = population + [anomaly]
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)

        self.assertEqual(ranked[0].ticket.item_id, "high-impact-anomaly")
        low_impact_ids = {t.ticket.item_id for t in ranked[1:]}
        self.assertNotIn("high-impact-anomaly", low_impact_ids)
        # The anomaly strictly outranks every low-impact ticket.
        anomaly_rank = next(r.rank for r in ranked if r.ticket.item_id == "high-impact-anomaly")
        self.assertEqual(anomaly_rank, 1)

    def test_scale_invariance(self):
        """Z-scoring keeps criterion scales commensurable."""
        base = [_make_ticket(f"t{i}", performance_gap=g) for i, g in enumerate([0.1, 0.3, 0.9])]
        scaled = [_make_ticket(f"t{i}", performance_gap=g * 10) for i, g in enumerate([0.1, 0.3, 0.9])]
        scorer = MultiCriteriaScorer()
        base_scores = scorer.score(base)
        scaled_scores = scorer.score(scaled)
        np.testing.assert_allclose(base_scores, scaled_scores, atol=1e-9)

    def test_scores_are_min_max_normalised(self):
        tickets = _random_population(50, seed=7)
        scores = MultiCriteriaScorer().score(tickets)
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)
        self.assertAlmostEqual(float(scores.min()), 0.0)
        self.assertAlmostEqual(float(scores.max()), 1.0)

    def test_single_ticket_is_well_defined(self):
        scores = MultiCriteriaScorer().score([_make_ticket("only")])
        self.assertEqual(scores.tolist(), [0.5])

    def test_empty_population(self):
        scorer = MultiCriteriaScorer()
        self.assertEqual(scorer.score([]).size, 0)
        self.assertEqual(scorer.assign_tiers(np.empty(0)), [])
        self.assertEqual(scorer.rank_order_tickets([]), [])

    def test_rank_order_is_deterministic_on_ties(self):
        # Three identical tickets: ties broken by item_id ascending.
        tickets = [_make_ticket(f"id-{c}") for c in ("c", "a", "b")]
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        ids = [r.ticket.item_id for r in ranked]
        self.assertEqual(ids, sorted(ids))

    def test_weights_must_be_positive(self):
        with self.assertRaises(ValueError):
            MultiCriteriaScorer(weights=PriorityWeights(performance_gap=0)).rank_order_tickets([_make_ticket("a")])

    def test_criteria_columns_match_weights(self):
        self.assertEqual(tuple(CRITERIA_COLUMNS), tuple(PriorityWeights.__dataclass_fields__))


class TestScoringPerformance(unittest.TestCase):
    """Rank-ordering 5,000 items must complete in under 100ms (O(N log N))."""

    N = 5_000
    BUDGET_SECONDS = 0.100

    def test_rank_order_5000_items_under_100ms(self):
        tickets = _random_population(self.N, seed=123)
        scorer = MultiCriteriaScorer()
        start = time.perf_counter()
        ranked = scorer.rank_order_tickets(tickets)
        elapsed = time.perf_counter() - start
        self.assertEqual(len(ranked), self.N)
        self.assertEqual([r.rank for r in ranked], list(range(1, self.N + 1)))
        self.assertLess(
            elapsed,
            self.BUDGET_SECONDS,
            f"rank_order_tickets({self.N}) took {elapsed * 1000:.1f}ms; budget {self.BUDGET_SECONDS * 1000:.0f}ms",
        )

    def test_ranks_are_pareto_complete(self):
        tickets = _random_population(200, seed=5)
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        self.assertEqual({r.rank for r in ranked}, set(range(1, 201)))


class TestWeightPerturbation(unittest.TestCase):
    """Randomised weight perturbations must preserve rank-order constraints."""

    def _perturbed_weights(self, base: PriorityWeights, rng: random.Random) -> PriorityWeights:
        def jitter(v: float) -> float:
            return max(0.05, v * rng.uniform(0.8, 1.2))

        return PriorityWeights(
            severity_weight=jitter(base.severity_weight),
            frequency_score=jitter(base.frequency_score),
            performance_gap=jitter(base.performance_gap),
            user_facing_impact=jitter(base.user_facing_impact),
        )

    def test_top_quartile_constraint_preserved_under_perturbation(self):
        # Two well-separated strata: the top quartile is unambiguous and must
        # stay so under any +-20% weight perturbation.
        rng = random.Random(99)
        population: list[TicketSignal] = []
        for i in range(30):
            population.append(
                _make_ticket(
                    f"strong-{i:03d}",
                    severity=rng.choice(("high", "critical")),
                    frequency_score=rng.uniform(0.85, 1.0),
                    performance_gap=rng.uniform(0.85, 1.0),
                    user_facing_impact=rng.uniform(0.85, 1.0),
                )
            )
        for i in range(90):
            population.append(
                _make_ticket(
                    f"weak-{i:03d}",
                    severity=rng.choice(("low", "medium")),
                    frequency_score=rng.uniform(0.0, 0.2),
                    performance_gap=rng.uniform(0.0, 0.2),
                    user_facing_impact=rng.uniform(0.0, 0.2),
                )
            )
        scorer = MultiCriteriaScorer()
        baseline = scorer.rank_order_tickets(population)
        quartile_cut = max(1, len(baseline) // 4)
        strong_ids = {f"strong-{i:03d}" for i in range(30)}

        for trial in range(50):
            perturbed = scorer.rank_order_tickets(population, weights=self._perturbed_weights(scorer.weights, rng))
            audit = scorer.rank_shift_audit(baseline, perturbed)
            self.assertEqual(
                audit["anomalies"],
                [],
                f"trial {trial}: top-quartile item fell out of top quartile",
            )
            # Rank constraint: the top quartile must remain exactly the strong stratum.
            top_quartile = {r.ticket.item_id for r in perturbed if r.rank <= quartile_cut}
            self.assertEqual(top_quartile, strong_ids, f"trial {trial}: top quartile changed composition")

    def test_rank_shift_audit_reports_shifts(self):
        population = _random_population(40, seed=3)
        scorer = MultiCriteriaScorer()
        baseline = scorer.rank_order_tickets(population)
        extreme = PriorityWeights(
            severity_weight=0.9,
            frequency_score=0.01,
            performance_gap=0.05,
            user_facing_impact=0.04,
        )
        shifted = scorer.rank_order_tickets(population, weights=extreme)
        audit = scorer.rank_shift_audit(baseline, shifted)
        self.assertGreater(audit["items_with_rank_shift"], 0)
        self.assertGreaterEqual(audit["max_abs_shift"], 1)
        self.assertEqual(audit["total_items"], 40)

    def test_rank_shift_audit_rejects_length_mismatch(self):
        tickets = _random_population(10, seed=1)
        scorer = MultiCriteriaScorer()
        baseline = scorer.rank_order_tickets(tickets)
        perturbed = scorer.rank_order_tickets(tickets[:-1])
        with self.assertRaises(ValueError):
            scorer.rank_shift_audit(baseline, perturbed)

    def test_identical_weights_produce_zero_shifts(self):
        tickets = _random_population(30, seed=11)
        scorer = MultiCriteriaScorer()
        baseline = scorer.rank_order_tickets(tickets)
        same = scorer.rank_order_tickets(tickets)
        audit = scorer.rank_shift_audit(baseline, same)
        self.assertEqual(audit["items_with_rank_shift"], 0)
        self.assertEqual(audit["max_abs_shift"], 0)


class TestPrioritySyncBridge(unittest.TestCase):
    """Tier -> Atlassian/Linear label mapping (Highest/High/Medium/Low/Lowest)."""

    def setUp(self):
        self.bridge = PrioritySyncBridge(scale=PriorityScale.LINEAR)

    def test_full_mapping(self):
        expected = {
            PriorityTier.URGENT: HIGHEST,
            PriorityTier.HIGH: HIGH,
            PriorityTier.MEDIUM: MEDIUM,
            PriorityTier.LOW: LOW,
            PriorityTier.BACKLOG: LOWEST,
        }
        for tier, label in expected.items():
            self.assertEqual(self.bridge.tier_to_label(tier), label)
            self.assertIs(self.bridge.label_to_tier(label), tier)
            self.assertEqual(TIER_TO_LABEL[tier], label)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            self.bridge.label_to_tier("Blocker")

    def test_build_mapping_flags_updates(self):
        tickets = _random_population(20, seed=17)
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        current = {
            t.ticket.item_id: self.bridge.tier_to_label(
                {
                    "urgent": PriorityTier.URGENT,
                    "high": PriorityTier.HIGH,
                    "medium": PriorityTier.MEDIUM,
                    "low": PriorityTier.LOW,
                    "backlog": PriorityTier.BACKLOG,
                }[t.tier]
            )
            for t in ranked
        }
        current[ranked[0].ticket.item_id] = LOW  # stale recorded label
        mappings = self.bridge.build_mapping(ranked, current_labels=current)
        needs_update = [m for m in mappings if m.needs_update]
        self.assertEqual(len(needs_update), 1)
        self.assertEqual(needs_update[0].item_id, ranked[0].ticket.item_id)

    def test_sync_payload_separates_updates_from_unchanged(self):
        tickets = _random_population(10, seed=21)
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        payload = self.bridge.sync_payload(self.bridge.build_mapping(ranked))
        self.assertEqual(len(payload["updates"]), 10)  # all unrecorded -> all updates
        self.assertEqual(payload["unchanged"], [])

    def test_consistency_audit_detects_mismatches(self):
        tickets = _random_population(20, seed=31)
        scorer = MultiCriteriaScorer()
        ranked = scorer.rank_order_tickets(tickets)
        tier_of = {"urgent": "Highest", "high": "High", "medium": "Medium", "low": "Low", "backlog": "Lowest"}
        recorded = {t.ticket.item_id: tier_of[t.tier] for t in ranked}
        # Sabotage half of the recorded labels.
        for i, t in enumerate(ranked):
            if i % 2 == 0:
                recorded[t.ticket.item_id] = LOWEST if t.tier != "backlog" else MEDIUM
        report = self.bridge.audit_consistency(ranked, recorded)
        self.assertIsInstance(report, ConsistencyAuditReport)
        self.assertFalse(report.is_consistent)
        self.assertEqual(len(report.mismatches), 10)
        self.assertEqual(report.matched + len(report.mismatches), 20)
        for mismatch in report.mismatches:
            self.assertNotEqual(mismatch["recorded_label"], mismatch["calculated_label"])

    def test_consistency_audit_flags_unrecorded(self):
        tickets = _random_population(5, seed=41)
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        reported = {t.ticket.item_id: "Medium" for t in ranked[:2]}
        report = self.bridge.audit_consistency(ranked, reported)
        self.assertEqual(len(report.unrecorded), 3)

    def test_scale_roundtrip(self):
        atlassian = PrioritySyncBridge(scale=PriorityScale.ATLASSIAN)
        linear = PrioritySyncBridge(scale=PriorityScale.LINEAR)
        tickets = _random_population(5, seed=2)
        ranked = MultiCriteriaScorer().rank_order_tickets(tickets)
        self.assertTrue(all(m.scale == PriorityScale.ATLASSIAN for m in atlassian.build_mapping(ranked)))
        self.assertTrue(all(m.scale == PriorityScale.LINEAR for m in linear.build_mapping(ranked)))
        # Payload carries the scale through to the transport layer.
        self.assertEqual(atlassian.sync_payload(atlassian.build_mapping(ranked))["scale"], "atlassian")
        self.assertEqual(linear.sync_payload(linear.build_mapping(ranked))["scale"], "linear")


class TestEvidenceCollector(unittest.TestCase):
    """Dict/JSON ingest of the three evidence streams."""

    def setUp(self):
        self.collector = EvidenceCollector()

    def _payload(self):
        return {
            "performance_gaps": [
                {
                    "item_id": "T-1",
                    "title": "Inference latency regression",
                    "severity": "high",
                    "performance_gap": 0.8,
                    "user_facing_impact": 0.6,
                    "source": "performance_gaps",
                },
            ],
            "safety_alerts": [
                {
                    "item_id": "T-2",
                    "title": "Crisis escalation gap",
                    "severity": "critical",
                    "frequency_score": 0.7,
                    "user_facing_impact": 0.9,
                    "source": "safety_alerts",
                },
                {
                    "item_id": "T-1",
                    "title": "Crisis escalation gap",
                    "severity": "critical",
                    "frequency_score": 0.4,
                    "user_facing_impact": 0.9,
                    "source": "safety_alerts",
                },
            ],
            "interactions": [
                {"item_id": "T-3", "title": "Session length outlier", "frequency_score": 0.3, "source": "interactions"},
            ],
        }

    def test_ingest_dicts_by_key(self):
        rows = self.collector.ingest_dicts(self._payload())
        self.assertEqual(len(rows), 4)
        severities = {r.item_id: r.severity for r in rows if r.item_id in ("T-1", "T-2")}
        self.assertEqual(severities["T-2"], "critical")

    def test_ingest_flat_list(self):
        rows = self.collector.ingest_dicts(self._payload()["safety_alerts"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.source == "safety_alerts" for r in rows))

    def test_ingest_json(self):
        import json
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "evidence.json"
            path.write_text(json.dumps(self._payload()))
            rows = self.collector.ingest_json(path)
        self.assertEqual(len(rows), 4)

    def test_to_signals_aggregates_worst_severity(self):
        rows = self.collector.ingest_dicts(self._payload())
        signals = {s.item_id: s for s in self.collector.to_signals(rows)}
        self.assertEqual(set(signals), {"T-1", "T-2", "T-3"})
        # T-1 has a high perf gap + critical safety alert -> worst severity wins.
        self.assertEqual(signals["T-1"].severity, "critical")
        self.assertAlmostEqual(signals["T-1"].performance_gap, 0.8)
        self.assertAlmostEqual(signals["T-1"].frequency_score, 0.4)
        # T-2 takes the max frequency across its two alerts.
        self.assertAlmostEqual(signals["T-2"].frequency_score, 0.7)
        # Interaction-only ticket keeps default severity.
        self.assertEqual(signals["T-3"].severity, "medium")
        self.assertAlmostEqual(signals["T-3"].frequency_score, 0.3)

    def test_end_to_end_collect_score_rank(self):
        # All three criteria point the same way (T-2 > T-1 > T-3), so the
        # rank order is mathematically guaranteed by the scoring model.
        payload = {
            "performance_gaps": [
                {
                    "item_id": "T-1",
                    "title": "Inference latency regression",
                    "severity": "high",
                    "performance_gap": 0.6,
                    "user_facing_impact": 0.5,
                },
            ],
            "safety_alerts": [
                {
                    "item_id": "T-1",
                    "title": "Crisis escalation gap",
                    "severity": "critical",
                    "frequency_score": 0.4,
                    "user_facing_impact": 0.6,
                },
                {
                    "item_id": "T-2",
                    "title": "Crisis escalation gap",
                    "severity": "critical",
                    "frequency_score": 0.8,
                    "user_facing_impact": 0.9,
                },
            ],
            "interactions": [
                {
                    "item_id": "T-3",
                    "title": "Session length outlier",
                    "frequency_score": 0.1,
                    "user_facing_impact": 0.2,
                },
            ],
        }
        rows = self.collector.ingest_dicts(payload)
        signals = self.collector.to_signals(rows)
        ranked = MultiCriteriaScorer().rank_order_tickets(signals)
        # Critical safety alert with the strongest frequency/impact ranks first.
        self.assertEqual([r.ticket.item_id for r in ranked], ["T-2", "T-1", "T-3"])
        self.assertEqual(ranked[0].tier, "urgent")

    def test_postgres_adapter_requires_psycopg2(self):
        with mock.patch.dict("sys.modules", {"psycopg2": None}):
            with self.assertRaises(RuntimeError) as ctx:
                PostgresEvidenceCollector("postgres://localhost/test")
        self.assertIn("psycopg2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
