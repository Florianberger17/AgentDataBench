"""Tests for Validator: completeness, schema conformance, CSV structure,
metadata consistency and reproducibility checks over a built BenchmarkPackage.
"""

import shutil

import pandas as pd
import pytest
import yaml

from agentdatabench.generator.validator import Validator


def _copy(src_root, dst_root):
    shutil.copytree(src_root, dst_root)
    return dst_root


@pytest.mark.parametrize(
    "package_dir_fixture",
    ["pkg1_root", "pkg2_root", "pkg3_root", "pkg4_root", "pkg5_root"],
)
def test_validate_real_packages_have_no_issues(package_dir_fixture, request):
    root = request.getfixturevalue(package_dir_fixture)
    if not (root / "data" / "dataset.csv").is_file():
        pytest.skip(f"{root} has no built dataset.csv yet")

    result = Validator().validate(root)

    assert result.is_valid, [i.message for i in result.issues]
    assert result.issues == []


def test_validate_detects_missing_additional_document(pkg1_root, tmp_path):
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    task_path = work_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    task_data["input"]["additional_documents"] = ["order_confirmation.pdf"]
    task_path.write_text(yaml.safe_dump(task_data))

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(
        i.code == "missing_file" and "order_confirmation.pdf" in i.message
        for i in result.issues
    )


def test_validate_reports_all_missing_files_at_once(pkg3_root, tmp_path):
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    (work_root / "scenario.yaml").unlink()
    (work_root / "metadata.yaml").unlink()

    result = Validator().validate(work_root)

    assert not result.is_valid
    codes = [i.code for i in result.issues]
    assert codes.count("missing_file") == 2


def test_validate_detects_metadata_task_id_mismatch(pkg3_root, tmp_path):
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    meta_path = work_root / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["task_id"] = "WRONG_ID"
    meta_path.write_text(yaml.safe_dump(meta))

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "task_id_mismatch" for i in result.issues)


def test_validate_detects_data_quality_declared_clean_but_noise_config_present(pkg3_root, tmp_path):
    # pkg3 genuinely has a noise_configuration.yaml - declaring "clean" is a
    # real authoring inconsistency.
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    meta_path = work_root / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["data_quality"] = "clean"
    meta_path.write_text(yaml.safe_dump(meta))

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "data_quality_mismatch" for i in result.issues)


def test_validate_detects_data_quality_declared_noisy_but_no_noise_config(pkg1_root, tmp_path):
    # pkg1 genuinely has no noise_configuration.yaml - declaring "noisy" is a
    # real authoring inconsistency.
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    meta_path = work_root / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["data_quality"] = "noisy"
    meta_path.write_text(yaml.safe_dump(meta))

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "data_quality_mismatch" for i in result.issues)


def test_validate_detects_non_reproducible_dataset(pkg3_root, tmp_path):
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    dataset_path = work_root / "data" / "dataset.csv"
    df = pd.read_csv(dataset_path, dtype=str)
    df.iloc[0, 0] = "TAMPERED"
    df.to_csv(dataset_path, index=False)

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "dataset_not_reproducible" for i in result.issues)


def test_validate_detects_uniqueness_violation_and_non_reproducible_ground_truth(
    pkg3_root, tmp_path
):
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    ground_truth_path = work_root / "ground_truth" / "ground_truth.csv"
    df = pd.read_csv(ground_truth_path, dtype=str)
    df.iloc[1, 0] = df.iloc[0, 0]
    df.to_csv(ground_truth_path, index=False)

    result = Validator().validate(work_root)

    assert not result.is_valid
    codes = {i.code for i in result.issues}
    assert "uniqueness_violation" in codes
    assert "ground_truth_not_reproducible" in codes


def test_validate_detects_column_order_mismatch(pkg3_root, tmp_path):
    work_root = _copy(pkg3_root, tmp_path / "pkg")
    ground_truth_path = work_root / "ground_truth" / "ground_truth.csv"
    df = pd.read_csv(ground_truth_path, dtype=str)
    columns = list(df.columns)
    columns[0], columns[1] = columns[1], columns[0]
    df[columns].to_csv(ground_truth_path, index=False)

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "column_order_mismatch" for i in result.issues)


def test_validate_pkg5_no_noise_configuration_is_reproducible(pkg5_root):
    if not (pkg5_root / "data" / "dataset.csv").is_file():
        pytest.skip("pkg5 has no built dataset.csv yet")
    assert not (pkg5_root / "noise_configuration.yaml").exists()

    result = Validator().validate(pkg5_root)

    assert result.is_valid


def _underspecify(work_root):
    """Turns a copied real package into an underspecified one: drops schema
    references from task.yaml in favor of a small target_example.csv."""
    ground_truth_lines = (work_root / "ground_truth" / "ground_truth.csv").read_text().splitlines()
    (work_root / "data" / "target_example.csv").write_text(
        "\n".join(ground_truth_lines[:3]) + "\n"
    )

    task_path = work_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    del task_data["input"]["source_schema"]
    del task_data["input"]["target_schema"]
    task_data["input"]["target_example"] = "data/target_example.csv"
    task_path.write_text(yaml.safe_dump(task_data))

    meta_path = work_root / "metadata.yaml"
    meta_data = yaml.safe_load(meta_path.read_text())
    meta_data["specification_completeness"] = "underspecified"
    meta_path.write_text(yaml.safe_dump(meta_data))


def test_validate_underspecified_package_skips_schema_dependent_checks(pkg1_root, tmp_path):
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    _underspecify(work_root)

    result = Validator().validate(work_root)

    # No schema/ground_truth-reproducibility issues - those checks are
    # no-ops without a formal target/source schema. dataset.csv
    # reproducibility (independent of schemas) still applies and passes.
    codes = {i.code for i in result.issues}
    assert "column_order_mismatch" not in codes
    assert "missing_required_column" not in codes
    assert "unexpected_column" not in codes
    assert "clean_dataset_schema_mismatch" not in codes
    assert "ground_truth_not_reproducible" not in codes
    assert result.is_valid, [i.message for i in result.issues]


def test_validate_underspecified_package_detects_missing_target_example(pkg1_root, tmp_path):
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    _underspecify(work_root)
    (work_root / "data" / "target_example.csv").unlink()

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(
        i.code == "missing_file" and "target_example" in i.message for i in result.issues
    )
