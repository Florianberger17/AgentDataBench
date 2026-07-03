"""Evaluation Framework subsystem: runs benchmark tasks against agents and
scores their results."""

from agentdatabench.evaluation.agent_adapter import AgentAdapter
from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter
from agentdatabench.evaluation.metrics import (
    DEFAULT_METRICS,
    ErrorCorrectionAccuracyMetric,
    FieldMappingAccuracyMetric,
    FilteringAccuracyMetric,
    Metric,
    RecordAccuracyMetric,
    RowAccuracyMetric,
    SchemaAccuracyMetric,
    TransformationAccuracyMetric,
)
from agentdatabench.evaluation.report_generator import (
    ReportGenerator,
    load_evaluation_results,
    render_markdown,
)
from agentdatabench.evaluation.runner import EvaluationRunner

__all__ = [
    "AgentAdapter",
    "DataInterpreterAdapter",
    "DEFAULT_METRICS",
    "Metric",
    "SchemaAccuracyMetric",
    "RowAccuracyMetric",
    "FilteringAccuracyMetric",
    "FieldMappingAccuracyMetric",
    "TransformationAccuracyMetric",
    "RecordAccuracyMetric",
    "ErrorCorrectionAccuracyMetric",
    "ReportGenerator",
    "load_evaluation_results",
    "render_markdown",
    "EvaluationRunner",
]
