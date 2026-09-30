"""Standalone ground-truth step: run GroundTruthCreator for one benchmark package.

Derives the expected solution from a package's task.yaml and target schema and
writes ground_truth/. This is only the ground-truth stage of the Benchmark
Generator - it does NOT inject noise or write data/dataset.csv (use
PackageBuilder for the full pipeline). Handy while authoring a task.yaml,
and usable before a package has a scenario.yaml/metadata.yaml, which
PackageBuilder requires but this step does not.

The CleanDataset is re-synthesized from source_data/ rather than read back
from disk, so the ground truth always matches the current
synthesis_configuration.yaml; synthesis is seeded and therefore reproducible.
Like PackageBuilder, the ground truth is derived from the *clean* data, never
from a noised dataset.

Usage (from the project root):
    python scripts/build_ground_truth.py <package_dir>

Example:
    python scripts/build_ground_truth.py \
        artifacts/benchmark_package/001_customer_migration_basic_explicit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.dataset import CSV_ENCODING, Dataset
from agentdatabench.domain.schema import Schema
from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.domain.task import Task
from agentdatabench.generator.dataset_creator import DatasetCreator
from agentdatabench.generator.ground_truth_creator import (
    GroundTruthCreator,
    load_reference_data,
)


def build_ground_truth(package_dir: Path) -> Path:
    package_dir = Path(package_dir)

    task_path = package_dir / "task.yaml"
    if not task_path.is_file():
        raise FileNotFoundError(f"Missing task.yaml in {package_dir}")
    task = Task(**load_yaml(task_path))

    # An implicit package deliberately hides its target schema from the
    # agent (TaskInput.target_example), but authoring the ground truth still
    # needs one, so fall back to the schema kept on disk for internal tooling.
    schema_reference = task.input.target_schema or "schemas/target_schema.yaml"
    schema_path = package_dir / schema_reference
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Missing target schema '{schema_reference}' in {package_dir}"
        )
    target_schema = Schema(**load_yaml(schema_path))

    config_path = package_dir / "synthesis_configuration.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing synthesis_configuration.yaml in {package_dir}")

    source_path = package_dir / "source_data" / "source_data.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source_data/source_data.csv in {package_dir}")

    config = SynthesisConfiguration(**load_yaml(config_path))
    clean_df = DatasetCreator().create_clean_dataset(Dataset(source_path).df, config)
    ground_truth_df = GroundTruthCreator().create_ground_truth(
        clean_df, task, target_schema, load_reference_data(package_dir, task)
    )

    out_dir = package_dir / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean_dataset.csv"
    ground_truth_path = out_dir / "ground_truth.csv"
    clean_df.to_csv(clean_path, index=False, encoding=CSV_ENCODING)
    ground_truth_df.to_csv(ground_truth_path, index=False, encoding=CSV_ENCODING)

    print(
        f"Clean dataset:  {clean_df.shape[0]} rows x {clean_df.shape[1]} columns "
        f"-> {clean_path}"
    )
    print(
        f"Ground truth:   {ground_truth_df.shape[0]} rows x "
        f"{ground_truth_df.shape[1]} columns -> {ground_truth_path}"
    )
    if len(ground_truth_df) != len(clean_df):
        print(
            f"Filtering removed {len(clean_df) - len(ground_truth_df)} of "
            f"{len(clean_df)} rows"
        )
    return ground_truth_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path, help="Benchmark package directory")
    args = parser.parse_args()
    build_ground_truth(args.package_dir)


if __name__ == "__main__":
    main()
