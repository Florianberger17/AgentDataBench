"""Applies FilteringRules to a pandas DataFrame.

Dataset columns are always strings (see domain.dataset.Dataset). A rule
comparing against a numeric value needs its column coerced to numeric first.
A rule comparing against a string value is normally an ISO date, which
compares correctly as a plain string (ISO 8601 sorts lexicographically in
date order) — unless the rule sets ``field_format``, meaning the *source*
column isn't stored in ISO format (e.g. "DD.MM.YYYY"); in that case both
sides are parsed into real dates before comparing, since naive string
comparison across differing date formats gives meaningless results.
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

_ISO_DATE_FORMAT = "%Y-%m-%d"


def _coerce_column(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return series
    return pd.to_numeric(series)


def _column_and_value(df: pd.DataFrame, rule: FilterRule) -> tuple[pd.Series, Any]:
    if rule.field_format is not None:
        source_format = translate_date_format(rule.field_format)
        column = df[rule.field].map(lambda v: datetime.strptime(str(v), source_format))
        value = datetime.strptime(rule.value, _ISO_DATE_FORMAT)
        return column, value

    return _coerce_column(df[rule.field], rule.value), rule.value


def apply_filtering(df: pd.DataFrame, filtering: FilteringRules | None) -> pd.DataFrame:
    if filtering is None:
        return df

    mask = pd.Series(True, index=df.index)
    for rule in filtering.rules:
        column, value = _column_and_value(df, rule)
        mask &= OPERATORS[rule.operator](column, value)

    return df[mask].reset_index(drop=True)
