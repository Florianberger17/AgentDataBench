"""Transformation handlers for MappingRule.transformation.type.

Handlers are registered in DEFAULT_TRANSFORMATION_HANDLERS by their ``type``
string. Adding a new transformation type only requires adding a new handler
and registering it here; GroundTruthCreator itself never needs to change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from migratebench.domain.task import MappingRule
from migratebench.generator.date_formats import translate_date_format


class TransformationHandler(Protocol):
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series: ...


class CopyHandler:
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        return df[mapping.source_field]


class ConcatenateHandler:
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        separator = getattr(mapping.transformation, "separator", "")
        columns = [df[field].astype(str) for field in mapping.source_fields]
        result = columns[0]
        for column in columns[1:]:
            result = result.str.cat(column, sep=separator)
        return result


class ValueMappingHandler:
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        value_map = mapping.transformation.mapping

        def lookup(value: str) -> str:
            if value not in value_map:
                raise ValueError(
                    f"No value_mapping entry for value {value!r} in field "
                    f"'{mapping.source_field}'"
                )
            return value_map[value]

        return df[mapping.source_field].astype(str).map(lookup)


class DateFormatHandler:
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        input_format = translate_date_format(mapping.transformation.input_format)
        output_format = translate_date_format(mapping.transformation.output_format)

        def convert(value: str) -> str:
            return datetime.strptime(str(value), input_format).strftime(output_format)

        return df[mapping.source_field].map(convert)


class SequentialNumberHandler:
    def apply(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        start = mapping.transformation.start
        increment = getattr(mapping.transformation, "increment", 1)
        digits = getattr(mapping.transformation, "digits", None)

        numbers = [start + i * increment for i in range(len(df))]
        if digits is not None:
            values = [str(n).zfill(digits) for n in numbers]
        else:
            values = [str(n) for n in numbers]
        return pd.Series(values, index=df.index)


DEFAULT_TRANSFORMATION_HANDLERS: dict[str, TransformationHandler] = {
    "copy": CopyHandler(),
    "concatenate": ConcatenateHandler(),
    "value_mapping": ValueMappingHandler(),
    "date_format": DateFormatHandler(),
    "sequential_number": SequentialNumberHandler(),
}
