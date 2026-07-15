"""Metrics for EvaluationRunner, registered in DEFAULT_METRICS. Adding a new
metric only requires adding a new class and registering it here;
EvaluationRunner itself never needs to change.

All seven metrics planned for the project are implemented. Schema/Row
Accuracy are strict and structural (exact column layout, exact full-row
match). The other five give partial credit or isolate one specific pipeline
stage, motivated directly by real Data Interpreter runs where a
schema-correct output scored row_accuracy=0.0 with no visibility into *how*
wrong it was:
  - Filtering Accuracy: did the agent keep roughly the right *number* of rows?
  - Field Mapping Accuracy: did each target column get populated at all?
  - Transformation Accuracy: cell-level correctness, continuous (0..1).
  - Record Accuracy: fraction of records that are "basically right" (a
    lenient threshold between Row Accuracy's 100%-exact and Transformation
    Accuracy's per-cell average).
  - Error Correction Accuracy: for noisy packages only, did noise-corrupted
    source values get cleaned before being mapped into the output?
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Protocol

import pandas as pd

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.evaluation_result import MetricResult
from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.generator.filtering import apply_filtering

# String forms pandas itself produces when stringifying a missing value
# (e.g. str(float("nan")) == "nan", str(None) == "None", str(pd.NA) ==
# "<NA>"). Matched case-insensitively, one token at a time - see
# _is_meaningfully_populated.
_MISSING_VALUE_TOKENS = {"nan", "none", "null", "nat", "<na>"}
_TOKEN_SPLIT_PATTERN = re.compile(r"[\s\-_/,.]+")


def _is_meaningfully_populated(value: object) -> bool:
    """False for a raw NaN/None, an empty string, and also for a string that
    consists *entirely* of missing-value tokens (e.g. "nan nan") - the real
    result of a live AG2 run whose generated code string-concatenated two
    empty source fields (f"{row['StreetAddress']} {row['HouseNo']}" with
    both NaN). That output is non-empty text, so a plain non-null/non-empty
    check silently counts it as "populated" even though it carries no real
    information. A field containing "nan" only as part of a longer real
    value (e.g. "Nano Systems") is unaffected: every token must match.

    Takes the raw cell value (not a pre-stringified one): pandas' own
    `Series.astype(str)` does not reliably turn every NaN into the string
    "nan" before `Series.map()` sees it, so converting here - after an
    explicit `pd.isna` check - is the robust order of operations.
    """
    if pd.isna(value):
        return False
    stripped = str(value).strip()
    if not stripped:
        return False
    tokens = [token for token in _TOKEN_SPLIT_PATTERN.split(stripped.lower()) if token]
    if not tokens:
        return False
    return not all(token in _MISSING_VALUE_TOKENS for token in tokens)


class Metric(Protocol):
    name: str

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult: ...


class SchemaAccuracyMetric:
    """Fraction of ground_truth.csv's columns that appear in the output at
    the same position with the same name. Compares against the *actual*
    ground_truth columns (not target_schema.attributes directly), so an
    optional target attribute that GroundTruthCreator legitimately dropped
    (no mapping rule, e.g. an unmapped Email field) is correctly not
    penalized - same convention as Validator's SchemaConformanceCheck."""

    name = "schema_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        expected_columns = list(ground_truth_df.columns)
        actual_columns = list(output_df.columns)

        matches = sum(
            1
            for i, column in enumerate(expected_columns)
            if i < len(actual_columns) and actual_columns[i] == column
        )
        score = matches / len(expected_columns) if expected_columns else 1.0

        return MetricResult(
            name=self.name,
            score=score,
            details={"expected_columns": expected_columns, "actual_columns": actual_columns},
        )


class RowAccuracyMetric:
    """Fraction of ground_truth.csv rows that appear verbatim (all columns,
    as strings) somewhere in the output, ignoring row order and treating
    duplicate rows as a multiset. Scores 0.0 on a column mismatch rather than
    raising, since a wrong-shaped output is exactly the kind of result this
    metric must be able to score."""

    name = "row_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        if list(output_df.columns) != list(ground_truth_df.columns):
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "reason": "column mismatch",
                    "expected_columns": list(ground_truth_df.columns),
                    "actual_columns": list(output_df.columns),
                },
            )

        expected_rows = Counter(
            ground_truth_df.astype(str).itertuples(index=False, name=None)
        )
        actual_rows = Counter(output_df.astype(str).itertuples(index=False, name=None))

        total = sum(expected_rows.values())
        matched = sum((expected_rows & actual_rows).values())
        score = matched / total if total else 1.0

        return MetricResult(
            name=self.name,
            score=score,
            details={"matched_rows": matched, "expected_rows": total},
        )


class FilteringAccuracyMetric:
    """How closely the number of rows the agent kept matches the number
    ground_truth.csv kept - a coarse proxy for filtering-rule correctness,
    independent of whether the retained rows' *content* is right (that's
    Row/Transformation Accuracy's job). Row-count-based rather than
    identity-based: several tasks regenerate primary keys (e.g. a
    sequential_number ERP id), so there is no schema-independent way to
    verify *which* rows were kept, only *how many*."""

    name = "filtering_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        expected = len(ground_truth_df)
        actual = len(output_df)
        if expected == 0:
            score = 1.0 if actual == 0 else 0.0
        else:
            score = max(0.0, 1 - abs(actual - expected) / expected)

        return MetricResult(
            name=self.name,
            score=score,
            details={"expected_rows": expected, "actual_rows": actual},
        )


class FieldMappingAccuracyMetric:
    """Fraction of expected target columns that are populated with a
    meaningful (non-null, non-empty, not-just-missing-value-tokens) value in
    the output - measures whether each field was mapped from *something* at
    all, independent of whether the mapped value is exactly correct (that's
    Transformation Accuracy's job). Motivated by two real failure modes seen
    in live agent runs: (1) Data Interpreter left a target column blank
    after an unresolved value-mapping lookup instead of raising or falling
    back; (2) AG2's own generated code string-concatenated two empty source
    fields into the literal text "nan nan" - non-empty, so it would pass a
    naive blank check, but just as uninformative as an empty cell. See
    _is_meaningfully_populated."""

    name = "field_mapping_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        expected_columns = list(ground_truth_df.columns)
        if not expected_columns:
            return MetricResult(name=self.name, score=1.0, details={})

        populated_fraction_by_column: dict[str, float] = {}
        for column in expected_columns:
            if column not in output_df.columns or output_df.empty:
                populated_fraction_by_column[column] = 0.0
                continue
            populated = output_df[column].map(_is_meaningfully_populated)
            populated_fraction_by_column[column] = float(populated.mean())

        score = sum(populated_fraction_by_column.values()) / len(expected_columns)
        return MetricResult(
            name=self.name,
            score=score,
            details={"populated_fraction_by_column": populated_fraction_by_column},
        )


def _best_row_matches(
    output_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> list[tuple[int | None, int, int]] | None:
    """For each ground_truth row (in order), finds the positional index into
    output_df of the row with the most matching cells (string comparison),
    without assuming any row order or identity correspondence between the
    two frames. Returns a list of (best_output_position, matched_cells,
    total_columns) tuples, or None if the column sets don't even match.

    O(n_ground_truth * n_output * n_columns) - fine at benchmark scale
    (~100 rows); would need a smarter index for much larger datasets.
    """
    if list(output_df.columns) != list(ground_truth_df.columns):
        return None

    columns = list(ground_truth_df.columns)
    output_rows = output_df.astype(str).reset_index(drop=True).to_dict("records")
    gt_rows = ground_truth_df.astype(str).reset_index(drop=True).to_dict("records")

    matches: list[tuple[int | None, int, int]] = []
    for gt_row in gt_rows:
        best_index: int | None = None
        best_count = -1
        for i, out_row in enumerate(output_rows):
            count = sum(1 for c in columns if gt_row[c] == out_row[c])
            if count > best_count:
                best_index, best_count = i, count
        matches.append((best_index, max(best_count, 0), len(columns)))
    return matches


class TransformationAccuracyMetric:
    """Cell-level accuracy: each ground_truth row is greedily paired with its
    closest output row (most matching cells - see _best_row_matches), then
    the fraction of matching cells is averaged across all cells. Unlike Row
    Accuracy (all-or-nothing per row), this gives partial credit - a row
    that's correct except for one mistyped field scores close to 1.0 here
    instead of 0.0."""

    name = "transformation_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        matches = _best_row_matches(output_df, ground_truth_df)
        if matches is None:
            return MetricResult(
                name=self.name, score=0.0, details={"reason": "column mismatch"}
            )

        total_matched = sum(matched for _, matched, _ in matches)
        total_cells = sum(total for _, _, total in matches)
        score = total_matched / total_cells if total_cells else 1.0

        return MetricResult(
            name=self.name,
            score=score,
            details={"matched_cells": total_matched, "total_cells": total_cells},
        )


class RecordAccuracyMetric:
    """Fraction of ground_truth records whose closest output row (see
    _best_row_matches) matches on at least `threshold` of its cells - a
    lenient middle ground between Row Accuracy (100% of cells must match)
    and Transformation Accuracy (continuous average with no per-record
    cutoff): it answers "how many records did the agent basically get
    right", not "how many were perfect" or "how correct were cells on
    average"."""

    name = "record_accuracy"
    THRESHOLD = 0.8

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        matches = _best_row_matches(output_df, ground_truth_df)
        if matches is None:
            return MetricResult(
                name=self.name, score=0.0, details={"reason": "column mismatch"}
            )
        if not matches:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"near_matches": 0, "total_records": 0, "threshold": self.THRESHOLD},
            )

        near_matches = sum(
            1 for _, matched, total in matches if total and matched / total >= self.THRESHOLD
        )
        score = near_matches / len(matches)

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "near_matches": near_matches,
                "total_records": len(matches),
                "threshold": self.THRESHOLD,
            },
        )


class ErrorCorrectionAccuracyMetric:
    """For packages with injected noise only: did noise-corrupted source
    values get cleaned before being mapped into the output, rather than
    passed through verbatim? Returns a vacuous 1.0 ("nothing to correct")
    for packages without a noise_configuration.yaml, so it stays safe to
    include in DEFAULT_METRICS for every package.

    Tracing a noisy source cell to its output value needs two things:
    1. Which source rows/columns NoiseEngine actually changed - found by
       diffing clean_dataset.csv against dataset.csv, joined on the first
       excluded_columns entry (NoiseEngine never touches excluded columns,
       so it is a stable id even across duplicate-row noise).
    2. Which ground_truth.csv row a given clean row became - re-derived by
       re-running apply_filtering() on clean_dataset.csv (the same call
       GroundTruthCreator itself makes), which preserves row order/count,
       rather than requiring an id-preserving "copy" mapping in the target
       schema (not every task has one, e.g. once an id is replaced by a
       sequential_number).
    Output rows are then matched back to ground_truth rows via the same
    greedy _best_row_matches used by Transformation/Record Accuracy.

    If the join key is missing, or the filtered clean dataset's row count
    doesn't match ground_truth.csv (e.g. a badly hand-authored package),
    reports "applicable": False rather than guessing.
    """

    name = "error_correction_accuracy"

    def compute(
        self,
        output_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        package: BenchmarkPackage,
    ) -> MetricResult:
        noise_config_path = package.root / "noise_configuration.yaml"
        if not noise_config_path.is_file():
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"applicable": False, "reason": "no noise_configuration.yaml"},
            )

        noise_config = NoiseConfiguration(**load_yaml(noise_config_path))
        if not noise_config.excluded_columns:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"applicable": False, "reason": "no excluded_columns to use as a join key"},
            )

        join_key = noise_config.excluded_columns[0]
        clean_df = package.clean_dataset.df
        noisy_df = package.dataset.df
        if join_key not in clean_df.columns or join_key not in noisy_df.columns:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"applicable": False, "reason": f"join key '{join_key}' not found"},
            )

        filtered_clean = apply_filtering(clean_df, package.task.business_rules.filtering)
        if len(filtered_clean) != len(ground_truth_df):
            return MetricResult(
                name=self.name,
                score=1.0,
                details={
                    "applicable": False,
                    "reason": "filtered clean dataset size does not match ground_truth.csv",
                },
            )

        clean_indexed = clean_df.set_index(join_key)
        noisy_indexed = noisy_df.drop_duplicates(subset=[join_key]).set_index(join_key)
        common_ids = clean_indexed.index.intersection(noisy_indexed.index)

        noisy_columns_by_id: dict[object, list[str]] = {}
        for row_id in common_ids:
            changed = [
                column
                for column in clean_df.columns
                if column != join_key
                and str(clean_indexed.loc[row_id, column]) != str(noisy_indexed.loc[row_id, column])
            ]
            if changed:
                noisy_columns_by_id[row_id] = changed

        target_fields_by_source: dict[str, list[str]] = defaultdict(list)
        for mapping in package.task.business_rules.mappings or []:
            sources = [mapping.source_field] if mapping.source_field else (mapping.source_fields or [])
            for source in sources:
                target_fields_by_source[source].append(mapping.target_field)

        matches = _best_row_matches(output_df, ground_truth_df)
        if matches is None:
            return MetricResult(
                name=self.name, score=0.0, details={"applicable": True, "reason": "column mismatch"}
            )

        checked = 0
        correct = 0
        for gt_position, (best_index, _, _) in enumerate(matches):
            row_id = filtered_clean.iloc[gt_position][join_key]
            noisy_columns = noisy_columns_by_id.get(row_id)
            if not noisy_columns:
                continue
            for source_column in noisy_columns:
                for target_field in target_fields_by_source.get(source_column, []):
                    if target_field not in ground_truth_df.columns:
                        continue
                    checked += 1
                    if best_index is None or target_field not in output_df.columns:
                        continue
                    expected = str(ground_truth_df.iloc[gt_position][target_field])
                    actual = str(output_df.iloc[best_index][target_field])
                    if actual == expected:
                        correct += 1

        if checked == 0:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"applicable": True, "reason": "no noisy field is mapped into the output"},
            )

        return MetricResult(
            name=self.name,
            score=correct / checked,
            details={"applicable": True, "checked": checked, "correct": correct},
        )


DEFAULT_METRICS: list[Metric] = [
    SchemaAccuracyMetric(),
    RowAccuracyMetric(),
    FilteringAccuracyMetric(),
    FieldMappingAccuracyMetric(),
    TransformationAccuracyMetric(),
    RecordAccuracyMetric(),
    ErrorCorrectionAccuracyMetric(),
]
