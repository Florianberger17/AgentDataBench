"""EvaluationResult domain object: the outcome of running EvaluationRunner
for one (BenchmarkPackage, AgentAdapter) pair."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

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
    # Whether re-executing the agent's own solution.py reproduces its own
    # solution.csv - a signal of hallucination/context loss, not task
    # correctness, so kept separate from `metrics`/`passed`. None only when
    # the agent itself failed (mirrors `metrics` staying empty in that case).
    reproducibility: MetricResult | None = None
    # How long the agent run took, regardless of outcome - copied from
    # AgentRunResult.duration_seconds.
    duration_seconds: float
    # Whatever run metadata the agent framework exposed (steps, token
    # counts, ...) - copied from AgentRunResult.metadata. Empty when the
    # agent failed before producing any.
    metadata: dict = Field(default_factory=dict)
