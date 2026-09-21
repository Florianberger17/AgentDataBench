"""Standalone synthesis step: run DatasetCreator for one benchmark package.

Turns a package's real source_data/source_data.csv into a synthetic CleanDataset,
using that package's synthesis_configuration.yaml. This is only the synthesis
stage of the Benchmark Generator - it does NOT inject noise or build ground
truth (use PackageBuilder for the full pipeline). Handy for iterating on a
synthesis_configuration.yaml until the synthetic data looks right.

Usage (from the project root):
    python scripts/synthesize_dataset.py <package_dir> [--out PATH]

Example:
    python scripts/synthesize_dataset.py \\
        artifacts/benchmark_package/003a_material_migration_basic

By default the result is written to <package_dir>/ground_truth/clean_dataset.csv,
the same location and filename PackageBuilder uses. Override with --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.dataset import CSV_ENCODING, Dataset
from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.generator.dataset_creator import DatasetCreator


def synthesize(package_dir: Path, out: Path | None = None) -> Path:
    package_dir = Path(package_dir)

    config_path = package_dir / "synthesis_configuration.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing synthesis_configuration.yaml in {package_dir}")

    source_path = package_dir / "source_data" / "source_data.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source_data/source_data.csv in {package_dir}")

    config = SynthesisConfiguration(**load_yaml(config_path))
    source_df = Dataset(source_path).df
    clean_df = DatasetCreator().create_clean_dataset(source_df, config)

    out = out or package_dir / "ground_truth" / "clean_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out, index=False, encoding=CSV_ENCODING)

    print(f"Synthesized {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
    print(f"Written to: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path, help="Benchmark package directory")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output CSV path (default: ground_truth/clean_dataset.csv)"
    )
    args = parser.parse_args()
    synthesize(args.package_dir, args.out)


if __name__ == "__main__":
    main()
