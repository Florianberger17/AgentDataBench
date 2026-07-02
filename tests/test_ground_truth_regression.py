"""Regression test: GroundTruthCreator must reproduce the real ground_truth.csv
files exactly, starting from each package's clean_dataset.csv and task.yaml.
"""

import pandas as pd
import pytest

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.generator.ground_truth_creator import GroundTruthCreator

PACKAGE_DIRS = ["001_customer_migration", "002_material_master_migration"]


@pytest.mark.parametrize("package_dir", PACKAGE_DIRS)
def test_ground_truth_creator_reproduces_real_ground_truth(package_dir, request):
    artifacts_root = request.getfixturevalue("pkg1_root").parent
    pkg = BenchmarkPackage.load(artifacts_root / package_dir)

    actual = GroundTruthCreator().create_ground_truth(
        pkg.clean_dataset.df, pkg.task, pkg.target_schema
    )
    expected = pkg.ground_truth.df

    # Both sides are cast to str before comparing: pandas' type inference on
    # pd.read_csv assigns different dtypes to values that are semantically
    # identical (e.g. generated "30000000000" str vs int64 read from CSV).
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True).astype(str),
        expected.reset_index(drop=True).astype(str),
        check_dtype=False,
    )
