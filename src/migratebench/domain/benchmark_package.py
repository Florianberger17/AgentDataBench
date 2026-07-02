"""BenchmarkPackage domain object: the aggregate of all artifacts for one benchmark task.

Not a pydantic model: it is a loader/aggregate over already-validated domain
objects plus file-backed ``Dataset`` instances, not itself a piece of validated
scalar data. ``load`` only checks that referenced files exist; it deliberately
does not cross-validate CSV columns against schemas, since real example packages
are not fully consistent in that respect (e.g. a package's "clean" dataset can
already be in target-schema shape).
"""

from __future__ import annotations

from pathlib import Path

from migratebench.domain.common import load_yaml
from migratebench.domain.dataset import Dataset
from migratebench.domain.metadata import Metadata
from migratebench.domain.scenario import Scenario
from migratebench.domain.schema import Schema
from migratebench.domain.task import Task


class BenchmarkPackage:
    def __init__(
        self,
        root: Path,
        scenario: Scenario,
        task: Task,
        metadata: Metadata,
        source_schema: Schema,
        target_schema: Schema,
        dataset: Dataset,
        clean_dataset: Dataset,
        ground_truth: Dataset,
    ) -> None:
        self.root = root
        self.scenario = scenario
        self.task = task
        self.metadata = metadata
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.dataset = dataset
        self.clean_dataset = clean_dataset
        self.ground_truth = ground_truth

    @classmethod
    def load(cls, root: Path) -> "BenchmarkPackage":
        root = Path(root)

        scenario = Scenario(**load_yaml(root / "scenario.yaml"))
        task = Task(**load_yaml(root / "task.yaml"))
        metadata = Metadata(**load_yaml(root / "metadata.yaml"))

        source_schema = Schema(**load_yaml(root / task.input.source_schema))
        target_schema = Schema(**load_yaml(root / task.input.target_schema))

        dataset = Dataset(root / task.input.source_dataset)
        clean_dataset = Dataset(root / "ground_truth" / "clean_dataset.csv")
        ground_truth = Dataset(root / "ground_truth" / "ground_truth.csv")

        return cls(
            root=root,
            scenario=scenario,
            task=task,
            metadata=metadata,
            source_schema=source_schema,
            target_schema=target_schema,
            dataset=dataset,
            clean_dataset=clean_dataset,
            ground_truth=ground_truth,
        )

    def __repr__(self) -> str:
        return f"BenchmarkPackage(root={self.root!r}, task_id={self.task.task_id!r})"
