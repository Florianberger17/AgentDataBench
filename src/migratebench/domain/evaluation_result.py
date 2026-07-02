"""EvaluationResult domain object.

Provisional/skeletal: this shape exists so the domain model is complete per the
overall architecture, but it will be extended once the Evaluation Framework
(metrics computation, report generation) is actually implemented.
"""

from __future__ import annotations

from datetime import datetime

from migratebench.domain.common import StrictBaseModel


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
