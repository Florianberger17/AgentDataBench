"""GroundTruthCreator: derives the expected solution dataset from a CleanDataset.

Column order of the result always follows target_schema.attributes, not the
order of business_rules.mappings in task.yaml — this is also what makes an
optional, unmapped target attribute (e.g. an "Email" field with no available
source data) correctly disappear from the output instead of erroring.

Rows are dropped in two distinct steps, in this order: ``business_rules.filtering``
removes records the migration is not supposed to carry over at all, then any
handler exposing ``excluded_rows`` removes records that cannot be mapped (a
``lookup`` with ``on_missing: exclude_record`` whose key is absent from the
reference table). Both run before the first column is built, so every column
is derived from the same set of surviving rows.

Target columns are then built in dependency order rather than schema order:
a field derived from other target fields (a valuation price from a converted
quantity) is deferred until those exist - see _build_columns.

Two steps then change how many records the frame holds, both after the dropping
steps - so that a record the migration does not carry over neither contributes
to a sum nor acquires successors - and both before record_order, which has to
sort the records that actually end up in the output.
``business_rules.expansion`` runs first and multiplies records, where one
source object became several target objects (a legacy cost center re-cut across
two units of the target operating model). ``business_rules.aggregation`` then
collapses them, where the target system keeps one record per group rather than
per source record (one credit master record per customer, from every order that
customer placed). Expansion before aggregation because it refines the
granularity of the frame while aggregation coarsens it.

``business_rules.record_order`` sorts what survives. It has to run after the
dropping and collapsing steps and before any column is built, because a
``sequential_number`` target field numbers the rows in exactly this order - a
row removed later would leave a gap, and numbering an unsorted frame would
make the expected result depend on the source file's arbitrary row order.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.schema import Schema
from agentdatabench.domain.task import MappingRule, RecordOrder, Task
from agentdatabench.generator.aggregation import apply_aggregation
from agentdatabench.generator.expansion import apply_expansion
from agentdatabench.generator.filtering import apply_filtering
from agentdatabench.generator.transformations import (
    DEFAULT_TRANSFORMATION_HANDLERS,
    MissingTargetColumn,
    TransformationContext,
    TransformationHandler,
)


def load_reference_data(root: Path, task: Task) -> dict[str, pd.DataFrame]:
    """Loads every table declared under ``task.input.reference_data``, keyed by
    its ``name`` - the key a ``lookup`` transformation refers to."""
    return {
        reference.name: Dataset(root / reference.file).df
        for reference in (task.input.reference_data or [])
    }


class GroundTruthCreator:
    def __init__(self, handlers: dict[str, TransformationHandler] | None = None) -> None:
        self._handlers = handlers or DEFAULT_TRANSFORMATION_HANDLERS

    def create_ground_truth(
        self,
        clean_df: pd.DataFrame,
        task: Task,
        target_schema: Schema,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        grouping = task.business_rules.grouping
        context = TransformationContext(
            reference_data=reference_data or {},
            group_keys=grouping.key_fields if grouping else [],
        )
        filtered = apply_filtering(
            clean_df, task.business_rules.filtering, context.reference_data
        )

        mappings_by_target = {
            mapping.target_field: mapping
            for mapping in (task.business_rules.mappings or [])
        }

        target_field_names = {attribute.name for attribute in target_schema.attributes}
        for target_field in mappings_by_target:
            if target_field not in target_field_names:
                raise ValueError(
                    f"Mapping rule targets unknown field '{target_field}' "
                    f"(not present in target schema '{target_schema.table}')"
                )

        filtered = self._drop_unmappable_records(
            filtered, list(mappings_by_target.values()), context
        )
        filtered = apply_expansion(
            filtered, task.business_rules.expansion, context.reference_data
        )
        filtered = apply_aggregation(
            filtered, task.business_rules.aggregation, context.reference_data
        )
        filtered = self._apply_record_order(filtered, task.business_rules.record_order)

        pending = []
        for attribute in target_schema.attributes:
            mapping = mappings_by_target.get(attribute.name)
            if mapping is None:
                if attribute.required:
                    raise ValueError(
                        f"No mapping rule for required target field '{attribute.name}'"
                    )
                continue
            if mapping.transformation.type not in self._handlers:
                raise ValueError(
                    f"No transformation handler registered for type "
                    f"'{mapping.transformation.type}'"
                )
            pending.append((attribute.name, mapping))

        columns = self._build_columns(filtered, pending, context)
        # Column order follows the target schema, not the order the fields
        # happened to become resolvable in.
        return pd.DataFrame({name: columns[name] for name, _ in pending})

    def _build_columns(
        self,
        df: pd.DataFrame,
        pending: list[tuple[str, MappingRule]],
        context: TransformationContext,
    ) -> dict[str, pd.Series]:
        """Builds the target columns, deferring any field that turns out to
        need another target field not built yet (MissingTargetColumn) and
        retrying it once that one exists.

        A field can depend on one that comes *later* in the target schema - a
        valuation price on a price unit, say - so a single pass in schema
        order is not enough, and requiring the author to order task.yaml by
        dependency would be a trap.
        """
        columns: dict[str, pd.Series] = {}
        remaining = list(pending)
        while remaining:
            deferred: list[tuple[str, MappingRule]] = []
            blocked: dict[str, str] = {}
            for name, mapping in remaining:
                handler = self._handlers[mapping.transformation.type]
                try:
                    columns[name] = handler.apply(
                        df, mapping, replace(context, target_columns=columns)
                    )
                except MissingTargetColumn as missing:
                    deferred.append((name, mapping))
                    blocked[name] = str(missing)

            if len(deferred) == len(remaining):
                raise ValueError(
                    f"Target field(s) {sorted(blocked)} cannot be resolved - each "
                    f"waits for a field that is never built: {blocked}"
                )
            remaining = deferred
        return columns

    def _apply_record_order(
        self, df: pd.DataFrame, record_order: RecordOrder | None
    ) -> pd.DataFrame:
        if record_order is None:
            return df

        missing = [name for name in record_order.fields if name not in df.columns]
        if missing:
            raise ValueError(
                f"record_order refers to unknown source field(s) {missing} "
                f"(available: {list(df.columns)})"
            )

        # Every column is text (see domain.dataset.Dataset), so a numeric
        # column would sort lexicographically - "10" before "9". Sort on a
        # numeric key wherever the whole column parses as one.
        keys = pd.DataFrame(index=df.index)
        for name in record_order.fields:
            numeric = pd.to_numeric(df[name], errors="coerce")
            keys[name] = df[name] if numeric.isna().any() else numeric

        order = keys.sort_values(
            by=record_order.fields, ascending=record_order.direction == "ascending"
        ).index
        return df.loc[order].reset_index(drop=True)

    def _drop_unmappable_records(
        self,
        df: pd.DataFrame,
        mappings: list[MappingRule],
        context: TransformationContext,
    ) -> pd.DataFrame:
        excluded = pd.Series(False, index=df.index)
        for mapping in mappings:
            handler = self._handlers.get(mapping.transformation.type)
            report = getattr(handler, "excluded_rows", None)
            if report is not None:
                excluded |= report(df, mapping, context)

        if not excluded.any():
            return df
        return df[~excluded].reset_index(drop=True)
