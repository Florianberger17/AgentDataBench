"""Evaluation Framework subsystem: runs benchmark tasks against agents and
scores their results."""

from agentdatabench.evaluation.agent_adapter import AgentAdapter
from agentdatabench.evaluation.ag2_adapter import AG2Adapter
from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter
from agentdatabench.evaluation.direct_llm_adapter import DirectLLMAdapter
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
from agentdatabench.evaluation.reproducibility import ReproducibilityCheck
from agentdatabench.evaluation.runner import EvaluationRunner

__all__ = [
    "AgentAdapter",
    "AG2Adapter",
    "DataInterpreterAdapter",
    "DirectLLMAdapter",
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
    "ReproducibilityCheck",
    "EvaluationRunner",
]
