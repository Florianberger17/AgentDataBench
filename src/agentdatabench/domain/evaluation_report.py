"""EvaluationReport domain object: an aggregation of EvaluationResults across
one or more (BenchmarkPackage, AgentAdapter) runs, summarized per agent so
different agents can be compared on the same set of tasks.
"""

from __future__ import annotations

from datetime import datetime

from agentdatabench.domain.common import StrictBaseModel
from agentdatabench.domain.evaluation_result import EvaluationResult


class AgentSummary(StrictBaseModel):
    agent_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    # Fraction of runs where the agent produced *some* output without
    # crashing/timing out (EvaluationResult.error is None) - distinct from
    # pass_rate, which additionally requires every metric to score 1.0. An
    # agent can be reliable (high success_rate) but rarely exactly correct
    # (low pass_rate), or vice versa never crash-free but always precise
    # when it does complete - the two numbers answer different questions.
    success_rate: float
    average_duration_seconds: float
    # Mean score per metric, averaged only over the results that actually
    # computed that metric - a run that failed before any metric could run
    # (empty EvaluationResult.metrics) contributes to failed_tasks/pass_rate
    # but not to these averages.
    average_metric_scores: dict[str, float]
    # Mean value per metadata key (steps, token counts, ...), averaged only
    # over the results that actually reported that key - keys are entirely
    # framework-specific (see AgentAdapter._invoke's docstring), so this is
    # necessarily an open dict rather than fixed fields.
    average_metadata: dict[str, float]


class EvaluationReport(StrictBaseModel):
    generated_at: datetime
    results: list[EvaluationResult]
    summaries: list[AgentSummary]
