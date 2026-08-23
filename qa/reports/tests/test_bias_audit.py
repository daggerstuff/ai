"""Tests for bias audit suite."""

from ai.monitoring.bias_audit import BiasAuditor, BiasCategory


def test_auditor_initialization():
    auditor = BiasAuditor(model_name="test-model")
    assert auditor.model_name == "test-model"
    assert auditor.threshold == 0.05


def test_demographic_bias_detection():
    examples = [
        {"input": "text", "age_group": "young", "gender": "male", "score": 0.9},
        {"input": "text", "age_group": "old", "gender": "female", "score": 0.5},
    ]

    def inference_fn(ex):
        return ex["score"]

    auditor = BiasAuditor(model_name="test-model")
    report = auditor.audit(examples, inference_fn)
    assert report.total_samples == 2
    assert any(d.category == BiasCategory.DEMOGRAPHIC for d in report.demographic_disparities)


def test_no_bias_when_equal():
    examples = [
        {"input": "text", "age_group": "young", "score": 0.7},
        {"input": "text", "age_group": "old", "score": 0.7},
    ]

    def inference_fn(ex):
        return ex["score"]

    auditor = BiasAuditor(model_name="test-model")
    report = auditor.audit(examples, inference_fn)
    disp = report.demographic_disparities[0]
    assert disp.max_disparity == 0.0


def test_report_summary():
    examples = [
        {"input": "text", "age_group": "young", "score": 0.7},
    ]

    def inference_fn(ex):
        return ex["score"]

    auditor = BiasAuditor(model_name="test-model")
    report = auditor.audit(examples, inference_fn)
    summary = report.summary()
    assert "total_samples" in summary
    assert "max_disparity" in summary
