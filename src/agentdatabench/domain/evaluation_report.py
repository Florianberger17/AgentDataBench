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
    # Mean score per metric, averaged only over the results that actually
    # computed that metric - a run that failed before any metric could run
    # (empty EvaluationResult.metrics) contributes to failed_tasks/pass_rate
    # but not to these averages.
    average_metric_scores: dict[str, float]


class EvaluationReport(StrictBaseModel):
    generated_at: datetime
    results: list[EvaluationResult]
    summaries: list[AgentSummary]
