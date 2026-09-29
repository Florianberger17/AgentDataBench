"""Applies FilteringRules to a pandas DataFrame.

A membership rule can test a combination of columns: a customer is identified
by customer number *and* address number, so both have to be looked up together.
`field` and `key_field` are then lists that pair up positionally.

Dataset columns are always strings (see domain.dataset.Dataset). A rule
comparing against a numeric value needs its column coerced to numeric first,
including the decimal comma a German-locale export writes.
A rule comparing against a string value is normally an ISO date, which
compares correctly as a plain string (ISO 8601 sorts lexicographically in
date order) — unless the rule sets ``field_format``, meaning the *source*
column isn't stored in ISO format (e.g. "DD.MM.YYYY"); in that case both
sides are parsed into real dates before comparing, since naive string
comparison across differing date formats gives meaningless results.

Three operator families are supported: the plain comparisons in ``OPERATORS``,
the emptiness tests, and membership in a reference table. The latter two do
not compare against ``rule.value`` at all, so they bypass ``_column_and_value``.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from agentdatabench.domain.task import FilterRule, FilteringRules
from agentdatabench.generator.date_formats import translate_date_format

OPERATORS: dict[str, Callable[..., pd.Series]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

# Operators that test the field itself or a reference table rather than
# comparing it against `rule.value`, so they are applied separately.
EMPTINESS_OPERATORS = ("is empty", "is not empty")
REFERENCE_OPERATORS = ("in", "not in")

_ISO_DATE_FORMAT = "%Y-%m-%d"


# Joins the parts of a composite key. The ASCII unit separator cannot occur in
# CSV text, so two different combinations can never collide on one joined key.
# Deliberately not NUL - see the same constant in transformations.py.
_KEY_JOIN = chr(31)


def _is_empty(column: pd.Series) -> pd.Series:
    return column.isna() | (column.astype(str).str.strip() == "")


def _joined(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    stripped = df[fields].astype(str).apply(lambda column: column.str.strip())
    return stripped.agg(_KEY_JOIN.join, axis=1)


def _reference_mask(
    df: pd.DataFrame,
    rule: FilterRule,
    reference_data: dict[str, pd.DataFrame] | None,
) -> pd.Series:
    """Membership in a reference table's key column - e.g. keeping only
    movements whose part number exists in the target system's material
    master."""
    name = rule.reference
    reference = (reference_data or {}).get(name)
    if reference is None:
        raise ValueError(
            f"Filter rule on '{rule.field}' refers to reference data {name!r}, "
            f"which is not available - declare it under task.input.reference_data"
        )

    fields = rule.field if isinstance(rule.field, list) else [rule.field]
    key_fields = rule.key_field if isinstance(rule.key_field, list) else [rule.key_field]
    if len(fields) != len(key_fields):
        raise ValueError(
            f"Filter rule on {rule.field!r} pairs {len(fields)} source field(s) "
            f"with {len(key_fields)} key field(s) - they have to match up"
        )
    for column in key_fields:
        if column not in reference.columns:
            raise ValueError(
                f"Reference table {name!r} has no column {column!r} "
                f"(available: {list(reference.columns)})"
            )

    keys = set(_joined(reference.dropna(subset=key_fields), key_fields))
    contained = _joined(df, fields).isin(keys)
    return contained if rule.operator == "in" else ~contained


def _coerce_column(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return series
    # Numeric columns arrive as strings in whatever decimal convention the
    # source export wrote; "1234,56" from a German-locale system is a number,
    # not a parse error.
    text = series.astype(str).str.strip()
    if text.str.contains(",", regex=False).any():
        text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text)


def _column_and_value(df: pd.DataFrame, rule: FilterRule) -> tuple[pd.Series, Any]:
    if rule.field_format is not None:
        source_format = translate_date_format(rule.field_format)
        column = df[rule.field].map(lambda v: datetime.strptime(str(v), source_format))
        value = datetime.strptime(rule.value, _ISO_DATE_FORMAT)
        return column, value

    return _coerce_column(df[rule.field], rule.value), rule.value


def apply_filtering(
    df: pd.DataFrame,
    filtering: FilteringRules | None,
    reference_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if filtering is None:
        return df

    mask = pd.Series(True, index=df.index)
    for rule in filtering.rules:
        if rule.operator in EMPTINESS_OPERATORS:
            empty = _is_empty(df[rule.field])
            mask &= empty if rule.operator == "is empty" else ~empty
            continue
        if rule.operator in REFERENCE_OPERATORS:
            mask &= _reference_mask(df, rule, reference_data)
            continue
        if rule.operator not in OPERATORS:
            raise ValueError(
                f"Unknown filter operator {rule.operator!r} (supported: "
                f"{sorted([*OPERATORS, *EMPTINESS_OPERATORS, *REFERENCE_OPERATORS])})"
            )
        column, value = _column_and_value(df, rule)
        mask &= OPERATORS[rule.operator](column, value)

    return df[mask].reset_index(drop=True)
