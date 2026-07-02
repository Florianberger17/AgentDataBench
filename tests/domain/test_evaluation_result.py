from datetime import datetime

from migratebench.domain.evaluation_result import EvaluationResult, MetricResult


def test_evaluation_result_happy_path():
    result = EvaluationResult(
        task_id="001_ERP_CUSTOMER_MIGRATION",
        agent_name="example-agent",
        metrics=[MetricResult(name="schema_accuracy", score=0.95)],
        passed=True,
        timestamp=datetime(2026, 7, 2, 12, 0, 0),
    )
    assert result.metrics[0].score == 0.95
    assert result.passed is True
