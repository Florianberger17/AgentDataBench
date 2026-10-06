"""Tests for Validator: completeness, schema conformance, CSV structure,
metadata consistency and the two reproducibility checks over a built
BenchmarkPackage - the one starting at clean_dataset.csv and the one starting
at source_data/.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.generator.validator import (
    SynthesisReproducibilityCheck,
    Validator,
)


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
    """Turns a copied real package into an implicit one: drops schema
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
    meta_data["specification_style"] = "implicit"
    meta_path.write_text(yaml.safe_dump(meta_data))


def test_validate_implicit_package_skips_schema_dependent_checks(pkg1_root, tmp_path):
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


def test_validate_implicit_package_detects_missing_target_example(pkg1_root, tmp_path):
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    _underspecify(work_root)
    (work_root / "data" / "target_example.csv").unlink()

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(
        i.code == "missing_file" and "target_example" in i.message for i in result.issues
    )


def test_validate_detects_scientific_notation_from_a_spreadsheet_round_trip(pkg1_root, tmp_path):
    """Excel rewrites long numbers as "2,54991E+12", keeping six significant
    digits - distinct identifiers then collapse onto one value, irreversibly."""
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    ground_truth = work_root / "ground_truth" / "ground_truth.csv"
    df = pd.read_csv(ground_truth, dtype=str, encoding="utf-8-sig")
    df.loc[0, df.columns[0]] = "2,54991E+12"
    df.to_csv(ground_truth, index=False, encoding="utf-8-sig")

    result = Validator().validate(work_root)

    codes = [issue.code for issue in result.issues]
    assert "scientific_notation" in codes
    assert not result.is_valid


def test_validate_accepts_plain_long_numbers_and_decimal_commas(pkg1_root, tmp_path):
    """A 13-digit number and a comma decimal are normal data, not damage."""
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    ground_truth = work_root / "ground_truth" / "ground_truth.csv"
    df = pd.read_csv(ground_truth, dtype=str, encoding="utf-8-sig")
    df.loc[0, df.columns[0]] = "2549910000000"
    df.loc[1, df.columns[0]] = "12,50"
    df.to_csv(ground_truth, index=False, encoding="utf-8-sig")

    codes = [issue.code for issue in Validator().validate(work_root).issues]
    assert "scientific_notation" not in codes


def _requires_source_data(root):
    """source_data/ holds raw input, is gitignored and is purged before a
    package is published, so it may legitimately be absent."""
    if not (root / "source_data" / "source_data.csv").is_file():
        pytest.skip(f"{root} has no source_data/ to re-synthesize from")


def test_validate_detects_a_clean_dataset_the_synthesis_no_longer_produces(
    pkg1_root, tmp_path
):
    """The gap ReproducibilityCheck leaves: it starts at clean_dataset.csv and
    so cannot tell whether synthesis_configuration.yaml still produces it."""
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    clean_path = work_root / "ground_truth" / "clean_dataset.csv"
    df = pd.read_csv(clean_path, dtype=str)
    df.iloc[0, 1] = "TAMPERED"
    df.to_csv(clean_path, index=False)

    result = Validator().validate(work_root)

    assert not result.is_valid
    assert any(i.code == "clean_dataset_not_reproducible" for i in result.issues)


def test_synthesis_check_names_the_column_that_diverges(pkg1_root, tmp_path):
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    clean_path = work_root / "ground_truth" / "clean_dataset.csv"
    df = pd.read_csv(clean_path, dtype=str)
    column = df.columns[1]
    df[column] = "TAMPERED"
    df.to_csv(clean_path, index=False)

    issues = SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root))

    assert len(issues) == 1
    assert column in issues[0].message


def test_synthesis_check_notices_a_differing_row_count(pkg1_root, tmp_path):
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    clean_path = work_root / "ground_truth" / "clean_dataset.csv"
    df = pd.read_csv(clean_path, dtype=str)
    df.iloc[:-1].to_csv(clean_path, index=False)

    issues = SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root))

    assert len(issues) == 1
    assert "rows re-synthesized" in issues[0].message


def test_synthesis_check_is_silent_once_source_data_is_purged(pkg1_root, tmp_path):
    """A published package has no source_data/ - that absence is the normal
    state and must not read as a finding."""
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    clean_path = work_root / "ground_truth" / "clean_dataset.csv"
    df = pd.read_csv(clean_path, dtype=str)
    df.iloc[0, 1] = "TAMPERED"
    df.to_csv(clean_path, index=False)
    shutil.rmtree(work_root / "source_data")

    assert SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root)) == []


def test_synthesis_check_is_silent_without_a_synthesis_configuration(
    pkg1_root, tmp_path
):
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    (work_root / "synthesis_configuration.yaml").unlink()

    assert SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root)) == []


def test_synthesis_check_reports_an_invalid_synthesis_configuration(
    pkg1_root, tmp_path
):
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    (work_root / "synthesis_configuration.yaml").write_text(
        "columns: 42\n", encoding="utf-8"
    )

    issues = SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root))

    assert [i.code for i in issues] == ["invalid_synthesis_configuration"]


def test_synthesis_check_reports_a_strategy_that_does_not_exist(pkg1_root, tmp_path):
    _requires_source_data(pkg1_root)
    work_root = _copy(pkg1_root, tmp_path / "pkg")
    column = pd.read_csv(
        work_root / "ground_truth" / "clean_dataset.csv", dtype=str
    ).columns[0]
    (work_root / "synthesis_configuration.yaml").write_text(
        f"seed: 1\ncolumns:\n  - column: {column}\n    strategy: no_such_strategy\n",
        encoding="utf-8",
    )

    issues = SynthesisReproducibilityCheck().check(BenchmarkPackage.load(work_root))

    assert [i.code for i in issues] == ["synthesis_failed"]


def test_synthesis_check_passes_on_every_real_package_that_still_has_source_data():
    """The regression guard for the defect this check was built for: a
    synthesis_configuration.yaml in the repository that no longer reproduces
    its own clean_dataset.csv."""
    roots = sorted(
        path
        for path in (Path("artifacts") / "benchmark_package").glob("*")
        if (path / "source_data" / "source_data.csv").is_file()
        and (path / "synthesis_configuration.yaml").is_file()
        and (path / "ground_truth" / "clean_dataset.csv").is_file()
        and (path / "metadata.yaml").is_file()
    )
    if not roots:
        pytest.skip("no package in the working tree still has source_data/")

    check = SynthesisReproducibilityCheck()
    offenders = {}
    for root in roots:
        try:
            package = BenchmarkPackage.load(root)
        except Exception:
            continue  # Covered by CompletenessCheck, not by this check.
        issues = check.check(package)
        if issues:
            offenders[root.name] = issues[0].message

    assert not offenders, offenders
