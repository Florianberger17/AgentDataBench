"""Domain model for migratebench: Scenario, Task, Schema, Dataset, BenchmarkPackage,
Metadata, EvaluationResult and their supporting types."""

from migratebench.domain.benchmark_package import BenchmarkPackage
from migratebench.domain.dataset import Dataset
from migratebench.domain.evaluation_result import EvaluationResult, MetricResult
from migratebench.domain.metadata import Metadata
from migratebench.domain.noise_configuration import NoiseConfiguration, NoiseTypeConfig
from migratebench.domain.scenario import Scenario, SystemInfo, Systems
from migratebench.domain.synthesis_configuration import (
    ColumnSynthesisConfig,
    SynthesisConfiguration,
)
from migratebench.domain.schema import Attribute, Schema, SchemaConstraint
from migratebench.domain.task import (
    BusinessRules,
    FilterRule,
    FilteringRules,
    MappingRule,
    Task,
    TaskInput,
    TaskOutput,
    TransformationSpec,
)

__all__ = [
    "BenchmarkPackage",
    "Dataset",
    "EvaluationResult",
    "MetricResult",
    "Metadata",
    "NoiseConfiguration",
    "NoiseTypeConfig",
    "ColumnSynthesisConfig",
    "SynthesisConfiguration",
    "Scenario",
    "SystemInfo",
    "Systems",
    "Attribute",
    "Schema",
    "SchemaConstraint",
    "BusinessRules",
    "FilterRule",
    "FilteringRules",
    "MappingRule",
    "Task",
    "TaskInput",
    "TaskOutput",
    "TransformationSpec",
]
