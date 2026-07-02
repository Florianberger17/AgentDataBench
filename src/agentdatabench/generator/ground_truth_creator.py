"""GroundTruthCreator: derives the expected solution dataset from a CleanDataset.

Column order of the result always follows target_schema.attributes, not the
order of business_rules.mappings in task.yaml — this is also what makes an
optional, unmapped target attribute (e.g. an "Email" field with no available
source data) correctly disappear from the output instead of erroring.
"""

from __future__ import annotations

import pandas as pd

from agentdatabench.domain.schema import Schema
from agentdatabench.domain.task import Task
from agentdatabench.generator.filtering import apply_filtering
from agentdatabench.generator.transformations import (
    DEFAULT_TRANSFORMATION_HANDLERS,
    TransformationHandler,
)


class GroundTruthCreator:
    def __init__(self, handlers: dict[str, TransformationHandler] | None = None) -> None:
        self._handlers = handlers or DEFAULT_TRANSFORMATION_HANDLERS

    def create_ground_truth(
        self, clean_df: pd.DataFrame, task: Task, target_schema: Schema
    ) -> pd.DataFrame:
        filtered = apply_filtering(clean_df, task.business_rules.filtering)

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

        columns: dict[str, pd.Series] = {}
        for attribute in target_schema.attributes:
            mapping = mappings_by_target.get(attribute.name)

            if mapping is None:
                if attribute.required:
                    raise ValueError(
                        f"No mapping rule for required target field '{attribute.name}'"
                    )
                continue

            handler = self._handlers.get(mapping.transformation.type)
            if handler is None:
                raise ValueError(
                    f"No transformation handler registered for type "
                    f"'{mapping.transformation.type}'"
                )
            columns[attribute.name] = handler.apply(filtered, mapping)

        return pd.DataFrame(columns)
