"""PackageBuilder: orchestrates the Benchmark Generator pipeline end-to-end.

Turns a package input directory (scenario/task/schemas/metadata plus
source_data/, a synthesis_configuration.yaml and an optional
noise_configuration.yaml) into a written BenchmarkPackage.

synthesis_configuration.yaml is mandatory whenever source_data/ is present:
it is the only step that keeps real/raw company data out of the published
CleanDataset, so the build is refused rather than silently publishing
source_data.csv unredacted. noise_configuration.yaml stays optional per the
BenchmarkPackage spec; when absent, the BenchmarkDataset handed to an agent
is simply the CleanDataset with no injected errors.
"""

from __future__ import annotations

from pathlib import Path

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.domain.schema import Schema
from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.domain.task import Task
from agentdatabench.generator.dataset_creator import DatasetCreator
from agentdatabench.generator.ground_truth_creator import GroundTruthCreator
from agentdatabench.generator.noise_engine import NoiseEngine
from agentdatabench.generator.source_data import purge_source_data


class PackageBuilder:
    def __init__(
        self,
        dataset_creator: DatasetCreator | None = None,
        noise_engine: NoiseEngine | None = None,
        ground_truth_creator: GroundTruthCreator | None = None,
    ) -> None:
        self._dataset_creator = dataset_creator or DatasetCreator()
        self._noise_engine = noise_engine or NoiseEngine()
        self._ground_truth_creator = ground_truth_creator or GroundTruthCreator()

    def build(self, root: Path, *, purge_source: bool = False) -> BenchmarkPackage:
        root = Path(root)

        task = Task(**load_yaml(root / "task.yaml"))
        target_schema = Schema(**load_yaml(root / task.input.target_schema))

        synthesis_config_path = root / "synthesis_configuration.yaml"
        if not synthesis_config_path.is_file():
            raise FileNotFoundError(
                f"{root}: synthesis_configuration.yaml is required to build a "
                "package from source_data/ (real/raw data must never enter a "
                "published CleanDataset unsynthesized)."
            )
        synthesis_config = SynthesisConfiguration(**load_yaml(synthesis_config_path))

        source_df = Dataset(root / "source_data" / "source_data.csv").df
        clean_df = self._dataset_creator.create_clean_dataset(source_df, synthesis_config)

        noise_config_path = root / "noise_configuration.yaml"
        if noise_config_path.is_file():
            noise_config = NoiseConfiguration(**load_yaml(noise_config_path))
            dataset_df = self._noise_engine.apply_noise(clean_df, noise_config)
        else:
            dataset_df = clean_df

        ground_truth_df = self._ground_truth_creator.create_ground_truth(
            clean_df, task, target_schema
        )

        dataset_path = root / task.input.source_dataset
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(dataset_path, index=False)

        ground_truth_dir = root / "ground_truth"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(ground_truth_dir / "clean_dataset.csv", index=False)
        ground_truth_df.to_csv(ground_truth_dir / "ground_truth.csv", index=False)

        if purge_source:
            purge_source_data(root)

        return BenchmarkPackage.load(root)
