"""Turns one source record into the several target records a reorganisation
splits it into.

This is the inverse of `aggregation`, and the two are the only steps that
change how many records the frame holds. An expansion joins the dataset onto a
reference table whose key deliberately repeats - a cross reference listing, per
legacy object, every object that succeeded it - and emits one record per match.

The join is inner and its result deterministic: records keep the order of the
dataset, and the successors of one record the order of the reference table. A
key absent from the reference table has no successor, so the record is dropped
rather than carried through with empty successor columns.

Reference values are written onto the record as ordinary source columns. A
mapping rule then reads them like any field of the dataset, which is why the
carried names must not collide with columns the dataset already has - the
collision would silently replace source data.
"""

from __future__ import annotations

import pandas as pd

from agentdatabench.domain.task import Expansion

# Same convention as filtering and aggregation: the ASCII unit separator cannot
# occur in CSV text, so two different combinations never collide on one key.
_KEY_JOIN = chr(31)


def _as_list(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def _joined(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    stripped = frame[fields].astype(str).apply(lambda column: column.str.strip())
    return stripped.agg(_KEY_JOIN.join, axis=1)


def _validated_fields(
    df: pd.DataFrame,
    expansion: Expansion,
    reference: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    source_fields = _as_list(expansion.source_field)
    key_fields = _as_list(expansion.key_field)
    if len(source_fields) != len(key_fields):
        raise ValueError(
            f"Expansion pairs {len(source_fields)} source field(s) with "
            f"{len(key_fields)} key field(s) - they have to match up"
        )
    for column in source_fields:
        if column not in df.columns:
            raise ValueError(
                f"Expansion reads unknown source field {column!r} "
                f"(available: {list(df.columns)})"
            )
    for column in [*key_fields, *expansion.carry_fields]:
        if column not in reference.columns:
            raise ValueError(
                f"Reference table {expansion.reference!r} has no column "
                f"{column!r} (available: {list(reference.columns)})"
            )
    for column in expansion.carry_fields:
        if column in df.columns:
            raise ValueError(
                f"Expansion carries reference column {column!r}, which the "
                f"dataset already has - carrying it would replace source data"
            )
    return source_fields, key_fields


def apply_expansion(
    df: pd.DataFrame,
    expansion: Expansion | None,
    reference_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if expansion is None:
        return df

    reference = (reference_data or {}).get(expansion.reference)
    if reference is None:
        raise ValueError(
            f"Expansion refers to reference data {expansion.reference!r}, which "
            f"is not available - declare it under task.input.reference_data"
        )

    source_fields, key_fields = _validated_fields(df, expansion, reference)

    usable = reference.dropna(subset=key_fields)
    successors: dict[str, list[int]] = {}
    for position, key in enumerate(_joined(usable, key_fields)):
        successors.setdefault(key, []).append(position)

    pairs = [
        (record, successor)
        for record, key in enumerate(_joined(df, source_fields))
        for successor in successors.get(key, [])
    ]

    expanded = df.iloc[[record for record, _ in pairs]].reset_index(drop=True)
    carried = usable.iloc[[successor for _, successor in pairs]].reset_index(drop=True)
    for column in expansion.carry_fields:
        expanded[column] = carried[column].fillna("").astype(str)
    return expanded
