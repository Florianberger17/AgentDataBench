"""ReportGenerator: the last step of the Evaluation Framework pipeline.
Aggregates EvaluationResults (one or more BenchmarkPackage/AgentAdapter runs,
potentially spanning several agents) into an EvaluationReport with a
per-agent summary, so different agents can be compared on the same tasks.

CSV/LaTeX export and per-task comparison tables are listed as long-term
extensions in the project's own roadmap, not built here yet - with only one
agent (Data Interpreter) evaluated so far, there is nothing to meaningfully
compare tables across yet. `EvaluationReport` and `render_markdown` below are
deliberately the minimal core: aggregate + a human-readable summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agentdatabench.domain.evaluation_report import AgentSummary, EvaluationReport
from agentdatabench.domain.evaluation_result import EvaluationResult


class ReportGenerator:
    def generate(self, results: list[EvaluationResult]) -> EvaluationReport:
        results_by_agent: dict[str, list[EvaluationResult]] = defaultdict(list)
        for result in results:
            results_by_agent[result.agent_name].append(result)

        summaries = [
            self._summarize(agent_name, agent_results)
            for agent_name, agent_results in results_by_agent.items()
        ]

        return EvaluationReport(
            generated_at=datetime.now(timezone.utc),
            results=results,
            summaries=summaries,
        )

    def _summarize(self, agent_name: str, results: list[EvaluationResult]) -> AgentSummary:
        total = len(results)
        passed = sum(1 for result in results if result.passed)

        scores_by_metric: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for metric in result.metrics:
                scores_by_metric[metric.name].append(metric.score)
        average_metric_scores = {
            name: sum(scores) / len(scores) for name, scores in scores_by_metric.items()
        }

        return AgentSummary(
            agent_name=agent_name,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=total - passed,
            pass_rate=passed / total if total else 0.0,
            average_metric_scores=average_metric_scores,
        )


def load_evaluation_results(
    root: Path, pattern: str = "**/evaluation_result.json"
) -> list[EvaluationResult]:
    """Loads every evaluation_result.json found under `root` (e.g. the
    manual_runs/ directory used by scripts/di_smoke_test/), sorted by path
    for a deterministic order."""
    results = []
    for path in sorted(root.glob(pattern)):
        results.append(EvaluationResult(**json.loads(path.read_text())))
    return results


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Total results: {len(report.results)}",
        "",
    ]
    for summary in report.summaries:
        lines.append(f"## {summary.agent_name}")
        lines.append(
            f"- Tasks: {summary.total_tasks} "
            f"({summary.passed_tasks} passed, {summary.failed_tasks} failed) - "
            f"pass rate: {summary.pass_rate:.1%}"
        )
        if summary.average_metric_scores:
            lines.append("- Average scores:")
            for metric_name, score in summary.average_metric_scores.items():
                lines.append(f"  - {metric_name}: {score:.3f}")
        lines.append("")
    return "\n".join(lines)
