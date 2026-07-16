"""Tests for ReportGenerator: aggregates EvaluationResults into a per-agent
EvaluationReport, plus the evaluation_result.json loader and markdown
rendering.
"""

import json
from datetime import datetime, timezone

from agentdatabench.domain.evaluation_result import EvaluationResult, MetricResult
from agentdatabench.evaluation.report_generator import (
    ReportGenerator,
    load_evaluation_results,
    render_markdown,
)


def _result(
    agent_name,
    task_id,
    passed,
    metrics=None,
    error=None,
    reproducibility=None,
    duration_seconds=1.0,
    metadata=None,
):
    return EvaluationResult(
        task_id=task_id,
        agent_name=agent_name,
        metrics=metrics or [],
        passed=passed,
        timestamp=datetime(2026, 7, 3, tzinfo=timezone.utc),
        error=error,
        reproducibility=reproducibility,
        duration_seconds=duration_seconds,
        metadata=metadata or {},
    )


def test_generate_summarizes_pass_rate_and_average_scores_per_agent():
    results = [
        _result(
            "agent-a",
            "task-1",
            passed=True,
            metrics=[MetricResult(name="row_accuracy", score=1.0)],
        ),
        _result(
            "agent-a",
            "task-2",
            passed=False,
            metrics=[MetricResult(name="row_accuracy", score=0.5)],
        ),
        _result("agent-b", "task-1", passed=True, metrics=[MetricResult(name="row_accuracy", score=1.0)]),
    ]

    report = ReportGenerator().generate(results)

    assert len(report.results) == 3
    summaries = {s.agent_name: s for s in report.summaries}

    agent_a = summaries["agent-a"]
    assert agent_a.total_tasks == 2
    assert agent_a.passed_tasks == 1
    assert agent_a.failed_tasks == 1
    assert agent_a.pass_rate == 0.5
    assert agent_a.average_metric_scores["row_accuracy"] == 0.75

    agent_b = summaries["agent-b"]
    assert agent_b.total_tasks == 1
    assert agent_b.pass_rate == 1.0


def test_generate_computes_success_rate_separately_from_pass_rate():
    # Success (agent produced output, no crash/timeout) and passed (that
    # output was exactly correct) are different questions - a run can
    # succeed without passing, but never pass without succeeding.
    results = [
        _result("agent-a", "task-1", passed=True),
        _result("agent-a", "task-2", passed=False),  # succeeded, but wrong
        _result("agent-a", "task-3", passed=False, error="agent crashed"),
    ]

    report = ReportGenerator().generate(results)
    summary = report.summaries[0]

    assert summary.pass_rate == 1 / 3
    assert summary.success_rate == 2 / 3


def test_generate_averages_duration_and_metadata():
    results = [
        _result(
            "agent-a",
            "task-1",
            passed=True,
            duration_seconds=10.0,
            metadata={"steps": 3, "prompt_tokens": 100},
        ),
        _result(
            "agent-a",
            "task-2",
            passed=True,
            duration_seconds=20.0,
            metadata={"steps": 5},
        ),
    ]

    report = ReportGenerator().generate(results)
    summary = report.summaries[0]

    assert summary.average_duration_seconds == 15.0
    assert summary.average_metadata["steps"] == 4.0
    # prompt_tokens only appeared in one of the two results - must average
    # over the results that reported it, not silently treat the other as 0.
    assert summary.average_metadata["prompt_tokens"] == 100.0


def test_generate_folds_reproducibility_into_average_scores():
    results = [
        _result(
            "agent-a",
            "task-1",
            passed=True,
            reproducibility=MetricResult(name="reproducibility", score=1.0),
        ),
        _result(
            "agent-a",
            "task-2",
            passed=True,
            reproducibility=MetricResult(name="reproducibility", score=0.0),
        ),
        # Failed runs have no reproducibility result - must not count as 0.
        _result("agent-a", "task-3", passed=False, error="agent crashed"),
    ]

    report = ReportGenerator().generate(results)
    summary = report.summaries[0]

    assert summary.average_metric_scores["reproducibility"] == 0.5


def test_generate_excludes_failed_runs_from_metric_averages_but_counts_them():
    # A run that failed before any metric could be computed (empty metrics,
    # error set) must still count toward pass_rate/failed_tasks, but must not
    # silently drag the metric averages down as if it scored 0.
    results = [
        _result(
            "agent-a",
            "task-1",
            passed=True,
            metrics=[MetricResult(name="row_accuracy", score=1.0)],
        ),
        _result("agent-a", "task-2", passed=False, error="agent crashed"),
    ]

    report = ReportGenerator().generate(results)
    summary = report.summaries[0]

    assert summary.total_tasks == 2
    assert summary.failed_tasks == 1
    assert summary.average_metric_scores["row_accuracy"] == 1.0


def test_load_evaluation_results_reads_all_json_files_under_root(tmp_path):
    for i, agent_name in enumerate(["agent-a", "agent-b"]):
        run_dir = tmp_path / f"run_{i}"
        run_dir.mkdir()
        result = _result(agent_name, f"task-{i}", passed=True)
        (run_dir / "evaluation_result.json").write_text(result.model_dump_json())

    loaded = load_evaluation_results(tmp_path)

    assert len(loaded) == 2
    assert {r.agent_name for r in loaded} == {"agent-a", "agent-b"}


def test_render_markdown_includes_agent_sections_and_scores():
    results = [
        _result(
            "data-interpreter",
            "task-1",
            passed=True,
            metrics=[MetricResult(name="schema_accuracy", score=1.0)],
            duration_seconds=42.0,
            metadata={"steps": 3},
        )
    ]
    report = ReportGenerator().generate(results)

    markdown = render_markdown(report)

    assert "# Evaluation Report" in markdown
    assert "## data-interpreter" in markdown
    assert "1 passed, 0 failed" in markdown
    assert "success rate: 100.0%" in markdown
    assert "schema_accuracy: 1.000" in markdown
    assert "Average duration: 42.0s" in markdown
    assert "steps: 3.00" in markdown
