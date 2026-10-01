"""Applies an Aggregation to a pandas DataFrame: many records become one.

This is the one step of the pipeline that changes the number of records for a
reason other than dropping them. ``business_rules.grouping`` only names which
records belong together while each still becomes its own target row; here the
records of a group collapse into a single one, because the target system keeps
one record per group and not per source record - one credit master record per
customer, derived from every order that customer ever placed.

A collapsed record carries three kinds of column:

* the key columns, by definition identical within the group,
* the declared aggregates, each written to a new column named by
  ``Aggregate.name`` so that a mapping rule reads it like any other field. An
  aggregate may restrict itself to part of the group with ``where``, which is
  what lets two figures of the same group rest on different records - the
  actual costs of a project on its cost postings and its planned revenue on
  its order-value postings, from one export that holds both,
* every remaining column, taken from the group's first record. Those are
  either constant within the group anyway (a customer's name) or not read by
  any mapping rule; keeping them means a rule may still reference them.

Groups appear in the order their first record does, so the result does not
depend on how pandas happens to sort the keys - ``business_rules.record_order``
is what puts them in a defined order.
"""

from __future__ import annotations

import pandas as pd

from agentdatabench.domain.task import Aggregate, Aggregation
from agentdatabench.generator.filtering import filter_mask
from agentdatabench.generator.transformations import _format_numeric, _to_numeric

# Joins the parts of a composite group key. The ASCII unit separator cannot
# occur in CSV text, so two different combinations never collide on one key -
# the same constant and the same reason as in filtering.py.
_KEY_JOIN = chr(31)

# Functions that need the column as numbers; the rest read it as text.
_NUMERIC_FUNCTIONS = ("sum", "min", "max")

# Functions for which "no record qualified" is a figure, not a blank: a
# project with no cost posting has costs of zero, not unknown costs.
_ZERO_ON_EMPTY = ("sum", "count")


def _joined_key(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    stripped = df[fields].astype(str).apply(lambda column: column.str.strip())
    return stripped.agg(_KEY_JOIN.join, axis=1)


def _aggregate_values(
    df: pd.DataFrame,
    keys: pd.Series,
    aggregate: Aggregate,
    reference_data: dict[str, pd.DataFrame] | None,
) -> tuple[pd.Series, str]:
    """One value per group, plus the decimal separator it should be rendered
    with (irrelevant for the non-numeric functions, which return text).

    Only the records ``aggregate.where`` admits are accumulated. A group none
    of whose records qualify is simply absent from the result; the caller
    decides what that means per function.
    """
    admitted = filter_mask(df, aggregate.where, reference_data)
    group = keys[admitted]

    if aggregate.function == "count":
        return group.groupby(group, sort=False).size(), ""

    if aggregate.function in _NUMERIC_FUNCTIONS:
        numbers, separator = _to_numeric(df[aggregate.source_field])
        values = numbers[admitted].groupby(group, sort=False).agg(aggregate.function)
        if aggregate.factor is not None:
            values = values * aggregate.factor
        return values, separator

    # first/last: the column as it stands, since the value does not have to be
    # a number - the name of the customer a group belongs to, say.
    text = df[aggregate.source_field].astype(str)[admitted]
    position = 0 if aggregate.function == "first" else -1
    return text.groupby(group, sort=False).agg(lambda rows: rows.iloc[position]), ""


def apply_aggregation(
    df: pd.DataFrame,
    aggregation: Aggregation | None,
    reference_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if aggregation is None:
        return df

    missing = [name for name in aggregation.key_fields if name not in df.columns]
    if missing:
        raise ValueError(
            f"aggregation.key_fields refers to unknown source field(s) {missing} "
            f"(available: {list(df.columns)})"
        )

    for aggregate in aggregation.aggregates:
        if aggregate.name in aggregation.key_fields:
            raise ValueError(
                f"Aggregate '{aggregate.name}' would overwrite the group key of "
                f"the same name - give it a name of its own"
            )
        if aggregate.factor is not None and aggregate.function not in (
            *_NUMERIC_FUNCTIONS,
            "count",
        ):
            raise ValueError(
                f"Aggregate '{aggregate.name}' sets a factor on function "
                f"'{aggregate.function}', whose result is not a number"
            )
        if (
            aggregate.function != "count"
            and aggregate.source_field not in df.columns
        ):
            raise ValueError(
                f"Aggregate '{aggregate.name}' reads unknown source field "
                f"'{aggregate.source_field}' (available: {list(df.columns)})"
            )

    if df.empty:
        collapsed = df.copy()
        for aggregate in aggregation.aggregates:
            collapsed[aggregate.name] = pd.Series(dtype="object")
        return collapsed

    keys = _joined_key(df, aggregation.key_fields)
    # The first record of each group, in the order the groups first appear.
    leading = keys.groupby(keys, sort=False).cumcount() == 0
    collapsed = df[leading].reset_index(drop=True)
    collapsed_keys = keys[leading].reset_index(drop=True)

    for aggregate in aggregation.aggregates:
        values, separator = _aggregate_values(df, keys, aggregate, reference_data)
        per_row = collapsed_keys.map(values)
        if aggregate.function in _ZERO_ON_EMPTY:
            per_row = per_row.fillna(0)
        if aggregate.function in _NUMERIC_FUNCTIONS or aggregate.function == "count":
            collapsed[aggregate.name] = _format_numeric(
                per_row, separator, aggregate.decimals
            )
        else:
            collapsed[aggregate.name] = per_row.fillna("").astype(str)

    return collapsed
