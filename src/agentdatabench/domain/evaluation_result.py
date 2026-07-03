"""EvaluationResult domain object: the outcome of running EvaluationRunner
for one (BenchmarkPackage, AgentAdapter) pair."""

from __future__ import annotations

from datetime import datetime

from agentdatabench.domain.common import StrictBaseModel


class MetricResult(StrictBaseModel):
    name: str
    score: float
    details: dict | None = None


class EvaluationResult(StrictBaseModel):
    task_id: str
    agent_name: str
    metrics: list[MetricResult]
    passed: bool
    timestamp: datetime
    # Set when the agent itself failed (timeout/crash/no output) before any
    # metric could be computed - distinct from a metric legitimately scoring
    # low on a produced-but-incorrect output. `metrics` is empty in that case.
    error: str | None = None
