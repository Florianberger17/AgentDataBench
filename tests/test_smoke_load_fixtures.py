"""End-to-end smoke test: load both real example benchmark packages from disk.

This closes the loop between the domain model and the actual example artifacts
under artifacts/benchmark_package/, rather than relying only on hand-written
fixture dicts in the unit tests.
"""

import pytest

from migratebench.domain.benchmark_package import BenchmarkPackage

PACKAGE_DIRS = ["001_customer_migration", "002_material_master_migration"]


@pytest.mark.parametrize("package_dir", PACKAGE_DIRS)
def test_load_real_benchmark_package(package_dir, request):
    artifacts_root = request.getfixturevalue("pkg1_root").parent
    root = artifacts_root / package_dir

    pkg = BenchmarkPackage.load(root)

    assert pkg.scenario is not None
    assert pkg.task is not None
    assert pkg.metadata is not None
    assert pkg.source_schema.attributes
    assert pkg.target_schema.attributes

    assert not pkg.dataset.df.empty
    assert not pkg.clean_dataset.df.empty
    assert not pkg.ground_truth.df.empty
