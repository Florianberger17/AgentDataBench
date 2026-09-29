"""Transformation handlers for MappingRule.transformation.type.

Handlers are registered in DEFAULT_TRANSFORMATION_HANDLERS by their ``type``
string. Adding a new transformation type only requires adding a new handler
and registering it here; GroundTruthCreator itself never needs to change.

Two handlers need more than the row values themselves:

* ``lookup`` resolves against a table outside the dataset, handed to it via
  ``TransformationContext.reference_data`` (see ``domain.task.ReferenceData``).
* A lookup whose key is missing can drop the whole record rather than produce
  an empty cell (``on_missing: exclude_record``). Since that affects every
  other column too, it cannot be expressed by a handler returning one Series.
  A handler opts in by implementing ``excluded_rows``, which GroundTruthCreator
  applies to the frame before any column is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd

from agentdatabench.domain.task import MappingRule
from agentdatabench.generator.date_formats import translate_date_format


@dataclass(frozen=True)
class TransformationContext:
    """Inputs a handler may need beyond the dataset row it transforms.

    ``reference_data`` maps a ``ReferenceData.name`` to the loaded table.
    Handlers default the context to an empty one, so the majority that only
    read row values stay callable with two arguments.
    """

    reference_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    # business_rules.grouping.key_fields - the columns identifying one target
    # document, for a field numbered `assigned_per: group`.
    group_keys: list[str] = field(default_factory=list)


EMPTY_CONTEXT = TransformationContext()


class TransformationHandler(Protocol):
    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = ...,
    ) -> pd.Series: ...


class CopyHandler:
    """Passes the source value through unchanged, optionally substituting
    ``default_if_empty`` where the target schema forbids the empty value the
    source permits. Blank-but-present strings count as empty too: a CSV
    round-trip turns a missing cell into ``""`` as readily as into NaN.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        column = df[mapping.source_field]
        default = getattr(mapping.transformation, "default_if_empty", None)
        if default is None:
            return column
        return column.mask(_is_empty(column), str(default))


class ConcatenateHandler:
    """Joins ``source_fields`` with a single ``separator``, or renders a
    free-form ``pattern`` such as ``"{OrderType} {Transaction}-{Name}"`` when
    the parts are separated by more than one distinct character.

    ``pattern`` placeholders are substituted literally rather than through
    ``str.format``, so source fields whose name contains a space (e.g.
    ``"Order Currency"``) can be referenced like any other.

    ``skip_empty`` leaves out parts carrying no value instead of joining an
    empty string, which would otherwise leave a dangling separator on every
    row whose optional second designation is blank. ``post_processing`` with
    ``type: truncate`` cuts the joined result to the target field's length.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        pattern = getattr(mapping.transformation, "pattern", None)
        if pattern is not None:
            return self._render_pattern(df, mapping.source_fields or [], pattern)

        separator = getattr(mapping.transformation, "separator", "")
        if getattr(mapping.transformation, "skip_empty", False):
            result = self._join_non_empty(df, mapping.source_fields or [], separator)
        else:
            columns = [df[field].astype(str) for field in mapping.source_fields]
            result = columns[0]
            for column in columns[1:]:
                result = result.str.cat(column, sep=separator)
        return self._post_process(result, mapping)

    def _join_non_empty(
        self, df: pd.DataFrame, source_fields: list[str], separator: str
    ) -> pd.Series:
        def join(row: pd.Series) -> str:
            return separator.join(
                str(value).strip()
                for value in row
                if not (pd.isna(value) or str(value).strip() == "")
            )

        return df[source_fields].apply(join, axis=1)

    def _post_process(self, result: pd.Series, mapping: MappingRule) -> pd.Series:
        post_processing = getattr(mapping.transformation, "post_processing", None)
        if post_processing and post_processing.get("type") == "truncate":
            return result.str.slice(0, post_processing["length"])
        return result

    def _render_pattern(
        self, df: pd.DataFrame, source_fields: list[str], pattern: str
    ) -> pd.Series:
        def render(row: pd.Series) -> str:
            text = pattern
            for field in source_fields:
                text = text.replace("{" + field + "}", str(row[field]))
            return text

        if df.empty:
            return pd.Series([], index=df.index, dtype="object")
        return df.apply(render, axis=1)


class ValueMappingHandler:
    """Translates source values via a fixed table.

    Without ``default``, a value missing from the table is an authoring error
    and is rejected - the common case, where the table is meant to be
    exhaustive. ``default: copy`` instead adopts unmapped values unchanged
    (a table that only renames the units it has to rename), and any other
    ``default`` is used as a literal fallback.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        value_map = {str(key): value for key, value in mapping.transformation.mapping.items()}
        default = getattr(mapping.transformation, "default", None)

        def lookup(value: str) -> str:
            if value in value_map:
                return value_map[value]
            if default is None:
                raise ValueError(
                    f"No value_mapping entry for value {value!r} in field "
                    f"'{mapping.source_field}' (and no 'default' configured)"
                )
            return value if default == "copy" else default

        return df[mapping.source_field].astype(str).map(lookup)


class DateFormatHandler:
    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        input_format = translate_date_format(mapping.transformation.input_format)
        output_format = translate_date_format(mapping.transformation.output_format)

        def convert(value: str) -> str:
            return datetime.strptime(str(value), input_format).strftime(output_format)

        return df[mapping.source_field].map(convert)


def _joined_key(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    # fillna() has to run before astype(str): pandas' default string dtype
    # (future.infer_string, default since pandas 3.0) keeps a missing cell as
    # a bare float NaN through astype(str) instead of stringifying it to
    # "nan", which str.join() below then rejects as a non-str sequence item.
    stripped = df[fields].fillna("nan").astype(str).apply(lambda column: column.str.strip())
    return stripped.agg(_KEY_JOIN.join, axis=1)


def _running_numbers(count: int, start: int, increment: int, digits: int | None) -> list[str]:
    numbers = [start + i * increment for i in range(count)]
    if digits is not None:
        return [str(n).zfill(digits) for n in numbers]
    return [str(n) for n in numbers]


class SequentialNumberHandler:
    """A running number, by default one per row.

    ``assigned_per: group`` numbers the *documents* instead: every row of one
    group (``business_rules.grouping.key_fields``) carries the same number, so
    all items of one sales order end up on one document. Groups are numbered
    in order of first appearance, which is the record order the frame was
    already sorted into.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        start = mapping.transformation.start
        increment = getattr(mapping.transformation, "increment", 1)
        digits = getattr(mapping.transformation, "digits", None)

        if getattr(mapping.transformation, "assigned_per", "record") != "group":
            values = _running_numbers(len(df), start, increment, digits)
            return pd.Series(values, index=df.index)

        if not context.group_keys:
            raise ValueError(
                f"Target field '{mapping.target_field}' is numbered per group "
                f"but no grouping is declared - add business_rules.grouping"
            )
        missing = [name for name in context.group_keys if name not in df.columns]
        if missing:
            raise ValueError(
                f"Grouping refers to unknown source field(s) {missing} "
                f"(available: {list(df.columns)})"
            )

        group = _joined_key(df, context.group_keys)
        numbers = _running_numbers(group.nunique(), start, increment, digits)
        return group.map(dict(zip(group.drop_duplicates(), numbers)))


class SequenceHandler:
    """A running number wrapped in a fixed ``prefix``/``suffix`` (e.g.
    ``P-310000``), unlike ``sequential_number``, which yields the bare digits.

    Source-independent: the rule carries no ``source_field``, the numbers come
    from the row count alone. ``start_value`` is the authored spelling;
    ``start`` is accepted as the ``sequential_number`` spelling of the same
    thing.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        transformation = mapping.transformation
        start = getattr(transformation, "start_value", None)
        if start is None:
            start = getattr(transformation, "start", 0)
        increment = getattr(transformation, "increment", 1)
        digits = getattr(transformation, "digits", None)
        prefix = getattr(transformation, "prefix", "")
        suffix = getattr(transformation, "suffix", "")

        numbers = _running_numbers(len(df), start, increment, digits)
        return pd.Series([f"{prefix}{n}{suffix}" for n in numbers], index=df.index)


class ConstantHandler:
    """The same literal in every row (e.g. a company code every migrated
    record is assigned to). Source-independent, like ``SequenceHandler``.

    The value is stringified so the column is homogeneously typed: an
    unquoted YAML integer (``value: 1``) and a quoted code that must keep its
    leading zero (``value: "0898"``) both survive the CSV round-trip, which
    reads every column back as a string.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        value = str(mapping.transformation.value)
        return pd.Series(value, index=df.index, dtype="object")


# Joins the parts of a composite key. The ASCII unit separator cannot occur in
# CSV text, so two different combinations can never collide on one joined key.
# Deliberately not NUL: pandas hashes object strings as C strings, where a NUL
# terminates the value, so "a<NUL>2" and "a<NUL>4" count as one value in
# nunique()/groupby() - which silently reported unambiguous lookup tables as
# ambiguous.
_KEY_JOIN = chr(31)


def _is_empty(column: pd.Series) -> pd.Series:
    return column.isna() | (column.astype(str).str.strip() == "")


def _pad(values: pd.Series, pad_to: int, pad_character: str) -> pd.Series:
    return values.astype(str).str.rjust(pad_to, pad_character)


class PrefixAndPadHandler:
    """Right-aligns the source value to ``pad_to`` characters and prepends a
    fixed ``prefix`` - e.g. a legacy document number 103 padded to 5 and
    prefixed with "97000" becomes "9700000103"."""

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        transformation = mapping.transformation
        pad_to = getattr(transformation, "pad_to", 0)
        pad_character = getattr(transformation, "pad_character", "0")
        prefix = getattr(transformation, "prefix", "")

        padded = _pad(df[mapping.source_field], pad_to, pad_character)
        return prefix + padded


class UppercaseHandler:
    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        return df[mapping.source_field].astype(str).str.upper()


class LookupHandler:
    """Resolves the source value against a reference table named by
    ``reference`` (see ``domain.task.ReferenceData``).

    An ambiguous reference table - the same key carrying more than one
    distinct value - is rejected rather than silently resolved to whichever
    row happens to come first, since that would make the ground truth depend
    on the row order of a file nobody treats as ordered.

    An *empty* source value is never a lookup failure: there is nothing to
    resolve, so the target cell stays empty (the target attribute is optional
    in that case). ``on_missing`` therefore only governs values that are
    present but absent from the table: ``exclude_record`` drops the whole
    record (``excluded_rows`` reports it to GroundTruthCreator, which removes
    it before any column is built), ``keep_empty`` leaves the cell empty, and
    the default rejects it as an authoring error.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        lookup = self._lookup_table(mapping, context)
        column = self._key_column(df, mapping)
        blank = self._blank_rows(df, mapping)
        resolved = column.map(lookup)

        unresolved = resolved.isna() & ~blank
        if unresolved.any() and not self._tolerates_missing(mapping):
            missing = sorted(set(column[unresolved]))
            raise ValueError(
                f"No entry in reference table "
                f"'{mapping.transformation.reference}' for value(s) "
                f"{missing[:5]} of field '{mapping.source_field}'"
            )
        resolved = resolved.fillna("")

        post_processing = getattr(mapping.transformation, "post_processing", None)
        if post_processing and post_processing.get("type") == "pad":
            padded = _pad(
                resolved,
                post_processing.get("pad_to", 0),
                post_processing.get("pad_character", "0"),
            )
            # Padding an empty cell would invent a value where the lookup
            # deliberately produced none.
            resolved = padded.mask(resolved == "", "")
        return resolved

    def _tolerates_missing(self, mapping: MappingRule) -> bool:
        return getattr(mapping.transformation, "on_missing", None) in (
            "exclude_record",
            "keep_empty",
        )

    def excluded_rows(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        if getattr(mapping.transformation, "on_missing", None) != "exclude_record":
            return pd.Series(False, index=df.index)
        lookup = self._lookup_table(mapping, context)
        unknown = ~self._key_column(df, mapping).isin(lookup.index)
        return unknown & ~self._blank_rows(df, mapping)

    def _key_column(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        """Each row's lookup key as one string. A composite key pairs
        ``source_fields`` with ``lookup_key`` positionally."""
        return _joined_key(df, self._source_fields(mapping))

    def _blank_rows(self, df: pd.DataFrame, mapping: MappingRule) -> pd.Series:
        """Rows with nothing to look up. Emptiness is read off the source
        columns, not the joined key, which has already stringified NaN.
        A composite key needs every part, so one empty part makes the row
        blank."""
        fields = self._source_fields(mapping)
        blank = _is_empty(df[fields[0]])
        for name in fields[1:]:
            blank |= _is_empty(df[name])
        return blank

    @staticmethod
    def _source_fields(mapping: MappingRule) -> list[str]:
        return mapping.source_fields or [mapping.source_field]

    def _lookup_table(
        self, mapping: MappingRule, context: TransformationContext
    ) -> pd.Series:
        transformation = mapping.transformation
        name = transformation.reference
        reference = context.reference_data.get(name)
        if reference is None:
            raise ValueError(
                f"Reference data '{name}' is not available - declare it under "
                f"task.input.reference_data"
            )

        key_fields = transformation.lookup_key
        if isinstance(key_fields, str):
            key_fields = [key_fields]
        return_field = transformation.return_field
        for column in (*key_fields, return_field):
            if column not in reference.columns:
                raise ValueError(
                    f"Reference table '{name}' has no column '{column}' "
                    f"(available: {list(reference.columns)})"
                )

        table = reference[[*key_fields, return_field]].dropna().copy()
        table["_key"] = _joined_key(table, key_fields)
        table[return_field] = table[return_field].astype(str).str.strip()

        conflicting = table.groupby("_key")[return_field].nunique()
        conflicting = conflicting[conflicting > 1]
        if not conflicting.empty:
            raise ValueError(
                f"Reference table '{name}' maps {len(conflicting)} key(s) to more "
                f"than one distinct '{return_field}', e.g. "
                f"{sorted(conflicting.index)[:5]} - the lookup is ambiguous"
            )

        return table.drop_duplicates(subset="_key").set_index("_key")[return_field]


# Numeric columns arrive as strings in whatever decimal convention the source
# export wrote ("1234.5" or "1234,5"). Arithmetic transformations detect the
# separator per column and re-emit the result in the same convention, so a
# calculated column stays consistent with the columns it was derived from.
def _separator(mapping: MappingRule, detected: str) -> str:
    """The configured `decimal_separator`, or the one detected in the source
    column. A source column holding only whole numbers carries no separator to
    detect, so a target field that needs decimals has to state it - otherwise
    a rounded quantity comes out dot-separated next to a comma-separated
    amount in the same file."""
    return getattr(mapping.transformation, "decimal_separator", None) or detected


def _to_numeric(column: pd.Series) -> tuple[pd.Series, str]:
    text = column.astype(str).str.strip()
    separator = "," if text.str.contains(",", regex=False).any() else "."
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce"), separator


def _format_numeric(values: pd.Series, separator: str, decimals: int | None) -> pd.Series:
    def render(value: float) -> str:
        if pd.isna(value):
            return ""
        if decimals is not None:
            return f"{value:.{decimals}f}"
        # Not "%g": a document number like 9700021910 would come out as
        # "9.70002e+09". Whole numbers are rendered as integers, everything
        # else with its trailing zeros trimmed.
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.10f}".rstrip("0").rstrip(".")

    rendered = values.map(render)
    if separator == ",":
        rendered = rendered.str.replace(".", ",", regex=False)
    return rendered


class NumericOffsetHandler:
    """Adds a fixed ``offset`` to the source value - a number range shift such
    as legacy transaction number 21910 becoming 9700021910."""

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        offset = mapping.transformation.offset
        numbers, separator = _to_numeric(df[mapping.source_field])
        return _format_numeric(
            numbers + offset, _separator(mapping, separator), _decimals(mapping)
        )


class NumericFactorHandler:
    """Multiplies the source value by a fixed ``factor`` - e.g. item numbers
    renumbered in steps of ten."""

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        factor = mapping.transformation.factor
        numbers, separator = _to_numeric(df[mapping.source_field])
        return _format_numeric(
            numbers * factor, _separator(mapping, separator), _decimals(mapping)
        )


class CalculationHandler:
    """Evaluates ``formula`` over the ``source_fields`` of the same row (e.g.
    ``"OrderPrice * OpenQuantity"``).

    Only the named source columns are in scope and only arithmetic on them is
    possible: the formula is evaluated by pandas' own expression engine, not
    by ``eval``, so a task.yaml cannot reach anything else from here.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        formula = mapping.transformation.formula
        source_fields = mapping.source_fields or []

        operands = {}
        separator = "."
        for field_name in source_fields:
            numbers, field_separator = _to_numeric(df[field_name])
            operands[field_name] = numbers
            if field_separator == ",":
                separator = ","

        result = pd.DataFrame(operands).eval(formula)
        return _format_numeric(
            result, _separator(mapping, separator), _decimals(mapping)
        )


def _decimals(mapping: MappingRule) -> int | None:
    return getattr(mapping.transformation, "decimals", None)


class ConditionalValueHandler:
    """Picks a value per row from the first matching condition.

    Conditions are structured rather than prose, so they are actually
    evaluated instead of only read by a human::

        conditions:
          - when: {field: CostCenter, is: not_empty}
            value: "K"
          - when: {field: MaterialNo, is: empty}
            from_field: Description1
          - value: ""                      # no `when`: the fallback

    ``value`` is a literal, ``from_field`` takes that row's value from another
    column - the distinction prose like ``value: "Description1"`` could not
    make. Rows matching no condition become empty.

    Supported ``is`` tests: ``empty``, ``not_empty``, ``equals``/``not_equals``
    (with ``to``), ``in``/``not_in`` (with ``to`` as a list), and
    ``in_reference``/``not_in_reference``, which test membership in a
    reference table's key column::

        - when:
            field: component no.
            is: in_reference
            reference: material_cross_reference
            key_field: "part no."
          value: "L"

    ``key_field`` is named at the condition rather than taken from the
    ReferenceData declaration, so a condition reads on its own and a table can
    be tested against more than one of its columns.
    """

    _TESTS = {
        "empty": lambda column, _: _is_empty(column),
        "not_empty": lambda column, _: ~_is_empty(column),
        "equals": lambda column, other: column.astype(str).str.strip() == str(other),
        "not_equals": lambda column, other: column.astype(str).str.strip() != str(other),
        "in": lambda column, other: column.astype(str).str.strip().isin(
            [str(value) for value in other or []]
        ),
        "not_in": lambda column, other: ~column.astype(str).str.strip().isin(
            [str(value) for value in other or []]
        ),
    }
    # Tested against a reference table rather than a literal, so they need the
    # context and are handled separately from _TESTS.
    _REFERENCE_TESTS = ("in_reference", "not_in_reference")

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        result = pd.Series("", index=df.index, dtype="object")
        unassigned = pd.Series(True, index=df.index)

        for condition in mapping.transformation.conditions:
            matches = self._matches(df, condition, context) & unassigned
            if not matches.any():
                continue
            result = result.mask(matches, self._values(df, condition))
            unassigned &= ~matches

        return result

    def _matches(
        self, df: pd.DataFrame, condition: dict, context: TransformationContext
    ) -> pd.Series:
        when = condition.get("when")
        if when is None:
            return pd.Series(True, index=df.index)

        field_name = when["field"]
        if field_name not in df.columns:
            raise ValueError(
                f"Condition refers to unknown source field '{field_name}' "
                f"(available: {list(df.columns)})"
            )

        test = when.get("is", "not_empty")
        if test in self._REFERENCE_TESTS:
            contained = self._in_reference(df[field_name], when, context)
            return contained if test == "in_reference" else ~contained

        if test not in self._TESTS:
            raise ValueError(
                f"Unknown condition test '{test}' (supported: "
                f"{sorted([*self._TESTS, *self._REFERENCE_TESTS])})"
            )
        return self._TESTS[test](df[field_name], when.get("to"))

    def _in_reference(
        self, column: pd.Series, when: dict, context: TransformationContext
    ) -> pd.Series:
        name = when["reference"]
        reference = context.reference_data.get(name)
        if reference is None:
            raise ValueError(
                f"Reference data '{name}' is not available - declare it under "
                f"task.input.reference_data"
            )

        key_field = when["key_field"]
        if key_field not in reference.columns:
            raise ValueError(
                f"Reference table '{name}' has no column '{key_field}' "
                f"(available: {list(reference.columns)})"
            )

        keys = reference[key_field].dropna().astype(str).str.strip()
        return column.astype(str).str.strip().isin(set(keys))

    def _values(self, df: pd.DataFrame, condition: dict) -> pd.Series:
        from_field = condition.get("from_field")
        if from_field is not None:
            if from_field not in df.columns:
                raise ValueError(
                    f"Condition refers to unknown source field '{from_field}' "
                    f"(available: {list(df.columns)})"
                )
            return df[from_field].fillna("").astype(str)

        value = condition.get("value")
        return pd.Series("" if value is None else str(value), index=df.index, dtype="object")


class TruncateHandler:
    """Cuts the source value to the target field's maximum length. Values that
    already fit pass through untouched."""

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        length = mapping.transformation.length
        return df[mapping.source_field].astype(str).str.slice(0, length)


class RoundHandler:
    """Rounds to ``decimals`` places, keeping the column's decimal convention
    (see _to_numeric)."""

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        decimals = getattr(mapping.transformation, "decimals", 2)
        numbers, separator = _to_numeric(df[mapping.source_field])
        return _format_numeric(numbers.round(decimals), _separator(mapping, separator), decimals)


class SerialDateConversionHandler:
    """Turns a spreadsheet serial day number into a calendar date.

    Excel counts days from an epoch of 1899-12-30, so 37681 is 2003-03-01.
    The epoch is configurable because other systems count from a different
    day. ``default_if_empty`` covers rows carrying no serial at all - the
    target field is usually mandatory and needs a fallback such as a cutover
    date.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        transformation = mapping.transformation
        epoch = datetime.strptime(getattr(transformation, "epoch", "1899-12-30"), "%Y-%m-%d")
        output_format = translate_date_format(transformation.output_format)
        default = getattr(transformation, "default_if_empty", None)

        column = df[mapping.source_field]
        blank = _is_empty(column)

        def convert(value: object) -> str:
            return (epoch + timedelta(days=int(float(str(value).replace(",", "."))))).strftime(
                output_format
            )

        result = column.mask(blank, None).map(lambda v: "" if v is None else convert(v))
        if default is None:
            return result
        return result.mask(blank, str(default))


DEFAULT_TRANSFORMATION_HANDLERS: dict[str, TransformationHandler] = {
    "copy": CopyHandler(),
    "concatenate": ConcatenateHandler(),
    "value_mapping": ValueMappingHandler(),
    "date_format": DateFormatHandler(),
    "sequential_number": SequentialNumberHandler(),
    "sequence": SequenceHandler(),
    "constant": ConstantHandler(),
    "prefix_and_pad": PrefixAndPadHandler(),
    # The same handler: `prefix` defaults to "", so padding alone is a
    # prefix_and_pad without one.
    "pad": PrefixAndPadHandler(),
    "uppercase": UppercaseHandler(),
    "lookup": LookupHandler(),
    "numeric_offset": NumericOffsetHandler(),
    "numeric_factor": NumericFactorHandler(),
    "calculation": CalculationHandler(),
    "conditional_value": ConditionalValueHandler(),
    "truncate": TruncateHandler(),
    "round": RoundHandler(),
    "serial_date_conversion": SerialDateConversionHandler(),
}
