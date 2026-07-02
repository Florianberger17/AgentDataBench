"""Tests for PackageBuilder, the orchestrator that turns a package input
directory (scenario/task/schemas/source_data + configs) into a written,
loadable BenchmarkPackage.
"""

import shutil

import pandas as pd
import pytest

from agentdatabench.domain.dataset import Dataset
from agentdatabench.generator.package_builder import PackageBuilder


def _copy_package_inputs(src_root, dst_root):
    shutil.copytree(src_root, dst_root)
    shutil.rmtree(dst_root / "data", ignore_errors=True)
    shutil.rmtree(dst_root / "ground_truth", ignore_errors=True)
    return dst_root


def test_build_without_synthesis_configuration_raises(pkg3_root, tmp_path):
    work_root = _copy_package_inputs(pkg3_root, tmp_path / "003_supplier_migration")
    (work_root / "synthesis_configuration.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="synthesis_configuration.yaml"):
        PackageBuilder().build(work_root)


def test_build_reproduces_pkg3_end_to_end(pkg3_root, tmp_path):
    source_path = pkg3_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    work_root = _copy_package_inputs(pkg3_root, tmp_path / "003_supplier_migration")

    pkg = PackageBuilder().build(work_root)

    expected_clean = Dataset(pkg3_root / "ground_truth" / "clean_dataset.csv").df
    expected_ground_truth = Dataset(pkg3_root / "ground_truth" / "ground_truth.csv").df

    pd.testing.assert_frame_equal(pkg.clean_dataset.df, expected_clean)
    pd.testing.assert_frame_equal(pkg.ground_truth.df, expected_ground_truth)

    assert (work_root / "data" / "dataset.csv").is_file()
    assert list(pkg.dataset.df.columns) == list(expected_clean.columns)
    assert (work_root / "source_data" / "source_data.csv").is_file()


def test_build_purge_source_removes_source_data(pkg3_root, tmp_path):
    source_path = pkg3_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    work_root = _copy_package_inputs(pkg3_root, tmp_path / "003_supplier_migration")

    PackageBuilder().build(work_root, purge_source=True)

    assert not (work_root / "source_data").exists()


def test_build_pkg4_matches_pkg3_clean_and_ground_truth(pkg3_root, pkg4_root, tmp_path):
    # pkg4 shares source_data.csv, synthesis_configuration.yaml (same seed)
    # and business_rules with pkg3, so its CleanDataset and ground truth must
    # come out byte-identical even though its noise_configuration.yaml (and
    # therefore its noisy dataset.csv) differs.
    source_path = pkg4_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    work_root = _copy_package_inputs(pkg4_root, tmp_path / "004_supplier_migration")

    pkg = PackageBuilder().build(work_root)

    expected_clean = Dataset(pkg3_root / "ground_truth" / "clean_dataset.csv").df
    expected_ground_truth = Dataset(pkg3_root / "ground_truth" / "ground_truth.csv").df

    pd.testing.assert_frame_equal(pkg.clean_dataset.df, expected_clean)
    pd.testing.assert_frame_equal(pkg.ground_truth.df, expected_ground_truth)


def test_build_pkg4_is_deterministic_and_does_not_leak_source_values(pkg4_root, tmp_path):
    source_path = pkg4_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    work_root = _copy_package_inputs(pkg4_root, tmp_path / "004_supplier_migration")
    source_df = Dataset(work_root / "source_data" / "source_data.csv").df

    first = PackageBuilder().build(work_root)
    first_dataset, first_clean = first.dataset.df.copy(), first.clean_dataset.df.copy()

    second = PackageBuilder().build(work_root)

    pd.testing.assert_frame_equal(first_dataset, second.dataset.df)
    pd.testing.assert_frame_equal(first_clean, second.clean_dataset.df)

    for column in ["name 1", "name 2", "street", "supplier no"]:
        assert set(first_clean[column]).isdisjoint(set(source_df[column])), column
    assert first_clean["supplier no"].nunique() == len(first_clean)


def test_build_pkg5_without_noise_configuration_induces_no_noise(pkg5_root, tmp_path):
    source_path = pkg5_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    work_root = _copy_package_inputs(pkg5_root, tmp_path / "005_material_master_migration")
    assert not (work_root / "noise_configuration.yaml").exists()
    source_df = Dataset(work_root / "source_data" / "source_data.csv").df

    pkg = PackageBuilder().build(work_root)

    pd.testing.assert_frame_equal(pkg.dataset.df, pkg.clean_dataset.df)

    resynthesized_columns = ["part no.", "name", "creator"]
    for column in resynthesized_columns:
        assert set(pkg.clean_dataset.df[column]).isdisjoint(set(source_df[column])), column
    assert pkg.clean_dataset.df["part no."].nunique() == len(pkg.clean_dataset.df)

    untouched_columns = [c for c in source_df.columns if c not in resynthesized_columns]
    for column in untouched_columns:
        pd.testing.assert_series_equal(
            pkg.clean_dataset.df[column].reset_index(drop=True),
            source_df[column].reset_index(drop=True),
            check_names=False,
        )
