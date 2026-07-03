"""EvaluationRunner: the Evaluation Framework pipeline. Loads a
BenchmarkPackage, hands it to an AgentAdapter, receives the agent's result,
computes metrics against ground_truth.csv, and produces an EvaluationResult.

Agent-level failure (timeout/crash/no output - see AgentAdapter) short-
circuits before any metric runs: `metrics` stays empty and `error` carries
the reason, kept distinct from a metric legitimately scoring an incorrect
but produced output.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.evaluation_result import EvaluationResult
from agentdatabench.evaluation.agent_adapter import AgentAdapter
from agentdatabench.evaluation.metrics import DEFAULT_METRICS, Metric


class EvaluationRunner:
    def __init__(self, metrics: list[Metric] | None = None) -> None:
        self._metrics = metrics or DEFAULT_METRICS

    async def run(
        self,
        package: BenchmarkPackage,
        adapter: AgentAdapter,
        *,
        timeout: float | None = None,
    ) -> EvaluationResult:
        agent_result = await adapter.run(package, timeout=timeout)

        if not agent_result.success:
            return EvaluationResult(
                task_id=package.task.task_id,
                agent_name=agent_result.agent_name,
                metrics=[],
                passed=False,
                error=agent_result.error,
                timestamp=datetime.now(timezone.utc),
            )

        output_df = Dataset(agent_result.output_dataset_path).df
        ground_truth_df = package.ground_truth.df

        metric_results = [
            metric.compute(output_df, ground_truth_df, package) for metric in self._metrics
        ]
        passed = all(metric_result.score == 1.0 for metric_result in metric_results)

        return EvaluationResult(
            task_id=package.task.task_id,
            agent_name=agent_result.agent_name,
            metrics=metric_results,
            passed=passed,
            timestamp=datetime.now(timezone.utc),
        )
