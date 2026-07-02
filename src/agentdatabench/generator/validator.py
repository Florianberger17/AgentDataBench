"""Validator: the last step of the Benchmark Generator pipeline.

Checks a built BenchmarkPackage for completeness, schema conformance, CSV
structure, metadata consistency and reproducibility, and returns a
ValidationResult listing every issue found (rather than raising on the
first problem) so a package author gets a full report in one pass.

Only the artifacts that make up a *published* BenchmarkPackage are in scope
(scenario/task/schemas/metadata, dataset.csv, ground_truth/*.csv, optional
noise_configuration.yaml). source_data/ and synthesis_configuration.yaml are
Benchmark-Generator-internal inputs - source_data/ is deliberately purged
before publishing (see purge_source_data) - and are out of scope here.

Each check is a small, independently testable class; new checks can be added
without changing Validator itself by passing a custom `package_checks` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.domain.task import Task
from agentdatabench.domain.validation_result import ValidationIssue, ValidationResult
from agentdatabench.generator.ground_truth_creator import GroundTruthCreator
from agentdatabench.generator.noise_engine import NoiseEngine

REQUIRED_TOP_LEVEL_FILES = ["scenario.yaml", "task.yaml", "metadata.yaml"]


class PackageCheck(Protocol):
    def check(self, package: BenchmarkPackage) -> list[ValidationIssue]: ...


class CompletenessCheck:
    """Enumerates every missing required file instead of failing on the first
    one, unlike BenchmarkPackage.load()."""

    def check(self, root: Path) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for filename in REQUIRED_TOP_LEVEL_FILES:
            if not (root / filename).is_file():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_file",
                        message=f"Required file is missing: {filename}",
                    )
                )

        if not (root / "task.yaml").is_file():
            return issues

        try:
            task = Task(**load_yaml(root / "task.yaml"))
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_task",
                    message=f"task.yaml is invalid: {exc}",
                )
            )
            return issues

        referenced_files = {
            "task.input.source_dataset": task.input.source_dataset,
            "task.input.source_schema": task.input.source_schema,
            "task.input.target_schema": task.input.target_schema,
            "ground_truth clean dataset": "ground_truth/clean_dataset.csv",
            "ground truth": "ground_truth/ground_truth.csv",
        }
        for label, relative_path in referenced_files.items():
            if not (root / relative_path).is_file():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_file",
                        message=f"{label} is missing: {relative_path}",
                    )
                )

        return issues


class SchemaConformanceCheck:
    """Target-schema/ground-truth mismatches are errors, since GroundTruthCreator
    guarantees column order/required-field presence by construction. Source
    schema/clean-dataset mismatches are only warnings: BenchmarkPackage.load()
    already documents real packages whose clean dataset legitimately isn't
    shaped like the source schema."""

    def check(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        return self._check_target_schema(package) + self._check_source_schema(package)

    def _check_target_schema(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        df = package.ground_truth.df
        attributes = package.target_schema.attributes
        attribute_names = [a.name for a in attributes]

        extra_columns = [c for c in df.columns if c not in attribute_names]
        if extra_columns:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unexpected_column",
                    message=f"ground_truth.csv has columns not in target schema: {extra_columns}",
                )
            )

        for attribute in attributes:
            if attribute.required and attribute.name not in df.columns:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_required_column",
                        message=(
                            f"ground_truth.csv is missing required target "
                            f"column '{attribute.name}'"
                        ),
                    )
                )

        expected_order = [a.name for a in attributes if a.name in df.columns]
        if list(df.columns) != expected_order:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="column_order_mismatch",
                    message=(
                        f"ground_truth.csv column order {list(df.columns)} does not "
                        f"match target schema order {expected_order}"
                    ),
                )
            )

        for attribute in attributes:
            if attribute.name not in df.columns:
                continue
            if attribute.required and df[attribute.name].isna().any():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="required_value_missing",
                        message=(
                            f"ground_truth.csv column '{attribute.name}' has missing "
                            "values but is required"
                        ),
                    )
                )
            if attribute.unique and df[attribute.name].duplicated().any():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="uniqueness_violation",
                        message=(
                            f"ground_truth.csv column '{attribute.name}' is declared "
                            "unique but has duplicate values"
                        ),
                    )
                )

        return issues

    def _check_source_schema(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        df = package.clean_dataset.df
        attribute_names = [a.name for a in package.source_schema.attributes]

        missing = [name for name in attribute_names if name not in df.columns]
        extra = [c for c in df.columns if c not in attribute_names]
        if not missing and not extra:
            return []

        return [
            ValidationIssue(
                severity="warning",
                code="clean_dataset_schema_mismatch",
                message=(
                    f"clean_dataset.csv columns {list(df.columns)} do not match "
                    f"source schema attributes {attribute_names} "
                    f"(missing={missing}, extra={extra})"
                ),
            )
        ]


class CsvStructureCheck:
    def check(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        datasets = {
            "dataset.csv": package.dataset,
            "clean_dataset.csv": package.clean_dataset,
            "ground_truth.csv": package.ground_truth,
        }
        for label, dataset in datasets.items():
            df = dataset.df
            if df.shape[1] == 0:
                issues.append(
                    ValidationIssue(
                        severity="error", code="no_columns", message=f"{label} has no columns"
                    )
                )
            if df.empty:
                issues.append(
                    ValidationIssue(
                        severity="error", code="empty_csv", message=f"{label} has no rows"
                    )
                )
        return issues


class MetadataConsistencyCheck:
    def check(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if package.metadata.scenario_id != package.scenario.scenario_id:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="scenario_id_mismatch",
                    message=(
                        f"metadata.scenario_id '{package.metadata.scenario_id}' != "
                        f"scenario.scenario_id '{package.scenario.scenario_id}'"
                    ),
                )
            )

        if package.metadata.task_id != package.task.task_id:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="task_id_mismatch",
                    message=(
                        f"metadata.task_id '{package.metadata.task_id}' != "
                        f"task.task_id '{package.task.task_id}'"
                    ),
                )
            )

        noise_config_path = package.root / "noise_configuration.yaml"
        if package.metadata.seed is not None and noise_config_path.is_file():
            noise_config = NoiseConfiguration(**load_yaml(noise_config_path))
            if package.metadata.seed != noise_config.seed:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="seed_mismatch",
                        message=(
                            f"metadata.seed {package.metadata.seed} != "
                            f"noise_configuration.seed {noise_config.seed}"
                        ),
                    )
                )

        return issues


class ReproducibilityCheck:
    """Re-derives dataset.csv and ground_truth.csv from clean_dataset.csv and
    compares them to what is on disk. Deliberately does not touch source_data/
    or synthesis_configuration.yaml (out of scope, see module docstring)."""

    def __init__(
        self,
        noise_engine: NoiseEngine | None = None,
        ground_truth_creator: GroundTruthCreator | None = None,
    ) -> None:
        self._noise_engine = noise_engine or NoiseEngine()
        self._ground_truth_creator = ground_truth_creator or GroundTruthCreator()

    def check(self, package: BenchmarkPackage) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        noise_config_path = package.root / "noise_configuration.yaml"
        if noise_config_path.is_file():
            noise_config = NoiseConfiguration(**load_yaml(noise_config_path))
            expected_dataset = self._noise_engine.apply_noise(package.clean_dataset.df, noise_config)
        else:
            expected_dataset = package.clean_dataset.df

        if not expected_dataset.reset_index(drop=True).equals(
            package.dataset.df.reset_index(drop=True)
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="dataset_not_reproducible",
                    message=(
                        "dataset.csv does not match re-running NoiseEngine "
                        "(or a no-noise pass-through) on clean_dataset.csv"
                    ),
                )
            )

        expected_ground_truth = self._ground_truth_creator.create_ground_truth(
            package.clean_dataset.df, package.task, package.target_schema
        )
        if not expected_ground_truth.reset_index(drop=True).astype(str).equals(
            package.ground_truth.df.reset_index(drop=True).astype(str)
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="ground_truth_not_reproducible",
                    message=(
                        "ground_truth.csv does not match re-running GroundTruthCreator "
                        "on clean_dataset.csv"
                    ),
                )
            )

        return issues


DEFAULT_PACKAGE_CHECKS: list[PackageCheck] = [
    SchemaConformanceCheck(),
    CsvStructureCheck(),
    MetadataConsistencyCheck(),
    ReproducibilityCheck(),
]


class Validator:
    def __init__(
        self,
        completeness_check: CompletenessCheck | None = None,
        package_checks: list[PackageCheck] | None = None,
    ) -> None:
        self._completeness_check = completeness_check or CompletenessCheck()
        self._package_checks = package_checks or DEFAULT_PACKAGE_CHECKS

    def validate(self, root: Path) -> ValidationResult:
        root = Path(root)
        issues = list(self._completeness_check.check(root))
        if any(issue.severity == "error" for issue in issues):
            return ValidationResult(issues=issues)

        try:
            package = BenchmarkPackage.load(root)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="package_load_failed",
                    message=f"BenchmarkPackage.load() failed: {exc}",
                )
            )
            return ValidationResult(issues=issues)

        for check in self._package_checks:
            issues.extend(check.check(package))

        return ValidationResult(issues=issues)
