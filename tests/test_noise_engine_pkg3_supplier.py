"""End-to-end test for package 003 (supplier migration): runs the real
noise_configuration.yaml against the real clean_dataset.csv and writes the
result to data/dataset.csv — the file this package has been missing since
no NoiseEngine existed yet to produce it.

There's no pre-existing "golden" noisy file to match (that's the point of
this component), so this asserts structural invariants instead.
"""

import pandas as pd

from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.generator.noise_engine import NoiseEngine


def test_noise_engine_reproduces_deterministically_on_pkg3(pkg3_root):
    config = NoiseConfiguration(**load_yaml(pkg3_root / "noise_configuration.yaml"))
    clean_df = Dataset(pkg3_root / "ground_truth" / "clean_dataset.csv").df

    result_a = NoiseEngine().apply_noise(clean_df, config)
    result_b = NoiseEngine().apply_noise(clean_df, config)

    pd.testing.assert_frame_equal(result_a, result_b)
    assert 99 <= len(result_a) <= 198


def test_noise_engine_preserves_excluded_columns_on_pkg3(pkg3_root):
    config = NoiseConfiguration(**load_yaml(pkg3_root / "noise_configuration.yaml"))
    clean_df = Dataset(pkg3_root / "ground_truth" / "clean_dataset.csv").df

    result = NoiseEngine().apply_noise(clean_df, config)

    assert set(result["supplier no"]).issubset(set(clean_df["supplier no"]))
    assert set(result["name 1"]).issubset(set(clean_df["name 1"]))


def test_write_pkg3_benchmark_dataset(pkg3_root):
    config = NoiseConfiguration(**load_yaml(pkg3_root / "noise_configuration.yaml"))
    clean_df = Dataset(pkg3_root / "ground_truth" / "clean_dataset.csv").df

    result = NoiseEngine().apply_noise(clean_df, config)
    output_path = pkg3_root / "data" / "dataset.csv"
    result.to_csv(output_path, index=False)

    written = Dataset(output_path).df
    assert len(written) == len(result)
    assert list(written.columns) == list(clean_df.columns)
