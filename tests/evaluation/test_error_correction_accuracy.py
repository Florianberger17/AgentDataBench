"""Tests for ErrorCorrectionAccuracyMetric against real packages - it needs
a full BenchmarkPackage (noise_configuration.yaml, clean_dataset.csv,
business_rules.mappings), so synthetic package=None inputs (used for the
other metrics) don't apply here.
"""

import pytest

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.metrics import ErrorCorrectionAccuracyMetric


def test_not_applicable_without_noise_configuration(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    result = ErrorCorrectionAccuracyMetric().compute(
        package.ground_truth.df, package.ground_truth.df, package
    )
    assert result.score == 1.0
    assert result.details["applicable"] is False


def test_perfect_output_scores_one_and_is_applicable(pkg3_root):
    # pkg3 has noise_configuration.yaml, so this exercises the real
    # clean-vs-noisy diff and join-key logic, not just the early-exit path.
    package = BenchmarkPackage.load(pkg3_root)
    result = ErrorCorrectionAccuracyMetric().compute(
        package.ground_truth.df, package.ground_truth.df, package
    )
    assert result.score == 1.0
    assert result.details["applicable"] is True
    assert result.details["checked"] > 0
    assert result.details["checked"] == result.details["correct"]


def test_detects_uncorrected_noise_in_a_mapped_field(pkg3_root):
    # Find a row/column pair NoiseEngine actually corrupted, then build an
    # output that reproduces ground_truth.csv everywhere *except* it leaves
    # that one field as the raw noisy value instead of the clean one -
    # mirrors exactly what we saw a real Data Interpreter run do (a typo'd
    # country code left uncorrected).
    package = BenchmarkPackage.load(pkg3_root)
    clean_df = package.clean_dataset.df
    noisy_df = package.dataset.df

    join_key = "supplier no"
    clean_indexed = clean_df.set_index(join_key)
    noisy_indexed = noisy_df.drop_duplicates(subset=[join_key]).set_index(join_key)
    common_ids = clean_indexed.index.intersection(noisy_indexed.index)

    corrupted_row_id, corrupted_column, noisy_value = None, None, None
    for row_id in common_ids:
        for column in [c for c in clean_df.columns if c != join_key]:
            if str(clean_indexed.loc[row_id, column]) != str(noisy_indexed.loc[row_id, column]):
                corrupted_row_id = row_id
                corrupted_column = column
                noisy_value = str(noisy_indexed.loc[row_id, column])
                break
        if corrupted_row_id is not None:
            break
    if corrupted_row_id is None:
        pytest.skip("no noise-corrupted cell found in this package's fixed seed output")

    # Map the corrupted source column to its target field via task.yaml.
    target_field = None
    for mapping in package.task.business_rules.mappings:
        source = mapping.source_field
        if source == corrupted_column:
            target_field = mapping.target_field
            break
    if target_field is None:
        pytest.skip(f"corrupted column '{corrupted_column}' is not mapped to any target field")

    ground_truth_df = package.ground_truth.df
    # additional_info carries the original supplier no (a "copy" mapping),
    # so this is how we find the corresponding ground_truth row.
    target_row_mask = ground_truth_df["additional_info"].astype(str) == str(corrupted_row_id)
    if not target_row_mask.any():
        pytest.skip("corrupted row was filtered out of ground_truth - not observable in output")

    bad_output = ground_truth_df.copy()
    bad_output.loc[target_row_mask, target_field] = noisy_value

    result = ErrorCorrectionAccuracyMetric().compute(bad_output, ground_truth_df, package)

    assert result.details["applicable"] is True
    assert result.details["correct"] < result.details["checked"]
    assert result.score < 1.0
