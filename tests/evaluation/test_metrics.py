"""Unit tests for the Metric implementations, isolated from BenchmarkPackage/
AgentAdapter (package=None is fine - most metrics don't touch it;
ErrorCorrectionAccuracyMetric is the exception, see
test_error_correction_accuracy.py)."""

import pandas as pd
import pytest

from agentdatabench.evaluation.metrics import (
    FieldMappingAccuracyMetric,
    FilteringAccuracyMetric,
    RecordAccuracyMetric,
    RowAccuracyMetric,
    SchemaAccuracyMetric,
    TransformationAccuracyMetric,
)


def test_row_accuracy_perfect_match():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    result = RowAccuracyMetric().compute(df, df, package=None)
    assert result.score == 1.0


def test_row_accuracy_ignores_row_order():
    ground_truth = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    output = pd.DataFrame({"a": ["2", "1"], "b": ["y", "x"]})
    result = RowAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 1.0


def test_row_accuracy_partial_match():
    ground_truth = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    output = pd.DataFrame({"a": ["1", "2", "9"], "b": ["x", "y", "z"]})
    result = RowAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(2 / 3)


def test_row_accuracy_column_mismatch_scores_zero_instead_of_raising():
    ground_truth = pd.DataFrame({"a": ["1"], "b": ["x"]})
    output = pd.DataFrame({"a": ["1"], "c": ["x"]})
    result = RowAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_row_accuracy_empty_ground_truth_scores_one():
    ground_truth = pd.DataFrame({"a": [], "b": []})
    result = RowAccuracyMetric().compute(ground_truth, ground_truth, package=None)
    assert result.score == 1.0


def test_schema_accuracy_perfect_match():
    df = pd.DataFrame({"a": [], "b": []})
    result = SchemaAccuracyMetric().compute(df, df, package=None)
    assert result.score == 1.0


def test_schema_accuracy_wrong_order_scores_zero():
    ground_truth = pd.DataFrame({"a": [], "b": []})
    output = pd.DataFrame({"b": [], "a": []})
    result = SchemaAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_schema_accuracy_missing_column():
    ground_truth = pd.DataFrame({"a": [], "b": [], "c": []})
    output = pd.DataFrame({"a": [], "b": []})
    result = SchemaAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(2 / 3)


def test_filtering_accuracy_exact_row_count_match():
    ground_truth = pd.DataFrame({"a": ["1", "2", "3"]})
    output = pd.DataFrame({"a": ["1", "2", "3"]})
    result = FilteringAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 1.0


def test_filtering_accuracy_penalizes_row_count_deviation():
    ground_truth = pd.DataFrame({"a": ["1", "2", "3", "4"]})
    output = pd.DataFrame({"a": ["1", "2"]})
    result = FilteringAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(0.5)
    assert result.details == {"expected_rows": 4, "actual_rows": 2}


def test_filtering_accuracy_never_goes_negative_on_large_overshoot():
    ground_truth = pd.DataFrame({"a": ["1"]})
    output = pd.DataFrame({"a": ["1", "2", "3", "4", "5"]})
    result = FilteringAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_filtering_accuracy_empty_ground_truth():
    ground_truth = pd.DataFrame({"a": []})
    result = FilteringAccuracyMetric().compute(ground_truth, ground_truth, package=None)
    assert result.score == 1.0


def test_field_mapping_accuracy_all_columns_populated():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    result = FieldMappingAccuracyMetric().compute(df, df, package=None)
    assert result.score == 1.0


def test_field_mapping_accuracy_partially_empty_column():
    ground_truth = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    output = pd.DataFrame({"a": ["1", ""], "b": ["x", "y"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(0.75)
    assert result.details["populated_fraction_by_column"]["a"] == 0.5
    assert result.details["populated_fraction_by_column"]["b"] == 1.0


def test_field_mapping_accuracy_missing_column_scores_zero_for_that_column():
    ground_truth = pd.DataFrame({"a": ["1"], "b": ["x"]})
    output = pd.DataFrame({"a": ["1"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(0.5)


def test_field_mapping_accuracy_treats_concatenated_nan_as_unpopulated():
    # Real AG2 output: f"{row['StreetAddress']} {row['HouseNo']}" with both
    # source fields empty produces the literal, non-empty string "nan nan" -
    # must not be counted as a populated field.
    ground_truth = pd.DataFrame({"street": ["Holstenplatz 189"]})
    output = pd.DataFrame({"street": ["nan nan"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_field_mapping_accuracy_treats_bare_missing_tokens_as_unpopulated():
    ground_truth = pd.DataFrame({"a": ["1"], "b": ["2"], "c": ["3"]})
    output = pd.DataFrame({"a": ["nan"], "b": ["None"], "c": ["<NA>"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_field_mapping_accuracy_handles_raw_nan_values_without_crashing():
    # Dataset (dtype=str) still yields real float NaN for genuinely empty
    # CSV cells - pandas' own astype(str) does not reliably turn every one
    # of those into the string "nan" before Series.map() sees it, so this
    # must not raise (regression: 'float' object has no attribute 'strip').
    ground_truth = pd.DataFrame({"a": ["1"], "b": ["2"]})
    output = pd.DataFrame({"a": [float("nan")], "b": ["2"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(0.5)


def test_field_mapping_accuracy_does_not_flag_real_values_containing_nan_substring():
    # "Nano" starts with "nan" but is not the token "nan" - must not be
    # confused with a stringified missing value.
    ground_truth = pd.DataFrame({"name": ["Nano Systems"]})
    output = pd.DataFrame({"name": ["Nano Systems"]})
    result = FieldMappingAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 1.0


def test_transformation_accuracy_perfect_match():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    result = TransformationAccuracyMetric().compute(df, df, package=None)
    assert result.score == 1.0


def test_transformation_accuracy_gives_partial_credit_for_near_miss_rows():
    # Row Accuracy would score this 0.0 (neither row is a full exact match);
    # Transformation Accuracy should reward the mostly-correct cells.
    ground_truth = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    output = pd.DataFrame({"a": ["1", "2"], "b": ["WRONG", "y"]})
    result = TransformationAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(3 / 4)


def test_transformation_accuracy_column_mismatch_scores_zero():
    ground_truth = pd.DataFrame({"a": ["1"]})
    output = pd.DataFrame({"b": ["1"]})
    result = TransformationAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == 0.0


def test_record_accuracy_counts_near_matches_above_threshold():
    # Row 0: 3/4 cells correct (0.75, just under the 0.8 default threshold).
    # Row 1: 4/4 cells correct.
    ground_truth = pd.DataFrame(
        {"a": ["1", "5"], "b": ["2", "6"], "c": ["3", "7"], "d": ["4", "8"]}
    )
    output = pd.DataFrame(
        {"a": ["1", "5"], "b": ["2", "6"], "c": ["3", "7"], "d": ["WRONG", "8"]}
    )
    result = RecordAccuracyMetric().compute(output, ground_truth, package=None)
    assert result.score == pytest.approx(0.5)
    assert result.details["near_matches"] == 1


def test_record_accuracy_all_above_threshold():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    result = RecordAccuracyMetric().compute(df, df, package=None)
    assert result.score == 1.0
