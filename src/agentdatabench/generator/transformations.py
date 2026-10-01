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

import math
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
    # Target columns built so far, for a field derived from other target
    # fields rather than from the source (see MissingTargetColumn).
    target_columns: dict[str, pd.Series] = field(default_factory=dict)


EMPTY_CONTEXT = TransformationContext()


class MissingTargetColumn(LookupError):
    """A handler needs a target column that has not been built yet.

    Raised rather than failing, so GroundTruthCreator can defer the field and
    retry once its dependencies exist. That keeps the dependency order out of
    task.yaml: a valuation price derived from a converted quantity works no
    matter where either field sits in the target schema.
    """


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
    """The column as numbers, plus the decimal separator it was written with.

    A comma in the column means the German convention, where the dot is a
    thousands separator and has to go before parsing - "4.028,000" is four
    thousand and twenty-eight, not a parse error. Without a comma the dot is
    the decimal separator and stays.
    """
    text = column.astype(str).str.strip()
    if text.str.contains(",", regex=False).any():
        return (
            pd.to_numeric(
                text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
            ",",
        )
    return pd.to_numeric(text, errors="coerce"), "."


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


def evaluate_formula(formula: str, operands: dict[str, pd.Series]) -> pd.Series:
    """Evaluates an arithmetic formula over named operand columns.

    Operand names are substituted by safe identifiers first, so a formula can
    name a column the way task.yaml spells it - `"stock value / LBKUM"` -
    without the backticks pandas would otherwise require. Longest names are
    replaced first, so one name cannot eat a prefix of another. `^` is
    accepted for exponentiation and `pi` is available as a constant.

    Evaluation goes through pandas' own expression engine, never `eval`, so a
    task.yaml cannot reach anything beyond these operands.
    """
    frame = {}
    expression = formula.replace("^", "**")
    for index, name in enumerate(sorted(operands, key=len, reverse=True)):
        alias = f"_op{index}"
        frame[alias] = operands[name]
        expression = expression.replace(name, alias)
    frame["pi"] = pd.Series(math.pi, index=next(iter(operands.values())).index)
    return pd.DataFrame(frame).eval(expression)


def _resolve_column(
    df: pd.DataFrame, name: str, context: TransformationContext
) -> pd.Series:
    """The named column, from the source row or - for a field derived from
    another target field - from the target columns built so far.

    Raises MissingTargetColumn for a name that is neither, so
    GroundTruthCreator can defer the field and retry it once the one it waits
    for exists.
    """
    if name in df.columns:
        return df[name]
    if name in context.target_columns:
        return context.target_columns[name]
    raise MissingTargetColumn(name)


def _numeric_operands(
    df: pd.DataFrame, fields: list[str], context: TransformationContext
) -> tuple[dict[str, pd.Series], str]:
    """The named fields as numbers, taken from the source row or - for a field
    derived from another target field - from the target columns built so far."""
    operands: dict[str, pd.Series] = {}
    separator = "."
    for name in fields:
        column = _resolve_column(df, name, context)
        numbers, field_separator = _to_numeric(column)
        operands[name] = numbers
        if field_separator == ",":
            separator = ","
    return operands, separator


class CalculationHandler:
    """Evaluates ``formula`` over the ``source_fields`` of the same row (e.g.
    ``"OrderPrice * OpenQuantity"``).

    A name that is not a source column is looked up among the target columns
    already built, so a field can be derived from other target fields - a
    valuation price from a converted quantity and a price unit. Those are
    declared in ``source_fields`` alongside the real source columns.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        formula = mapping.transformation.formula
        operands, separator = _numeric_operands(df, mapping.source_fields or [], context)
        result = evaluate_formula(formula, operands)
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

    A ``when`` may also resolve its field *through* a reference table before
    testing it, by naming a ``return_field`` beside the ``reference``. The
    test then reads the resolved value, which is what lets a rule branch on an
    attribute the dataset does not carry at all::

        - when:
            field: customer no.
            reference: customer_cross_reference
            key_field: "customer no. legacy"
            return_field: "payment behaviour"
            is: equals
            to: "good payer"
          value: "A"

    ``all_of`` holds several such clauses that all have to hold, for a value
    that follows from a combination rather than from one field::

        - all_of:
            - {field: payment behaviour, is: equals, to: "good payer"}
            - {field: CREDIT_LIMIT, is: in, to: ["50.000", "100.000"]}
          value: "AA"

    Both ``field`` and ``from_field`` may name a *target* field instead of a
    source column, so a field can be derived from another field of the output
    (a risk class from the credit limit above it). The handler raises
    MissingTargetColumn in that case and GroundTruthCreator retries it once
    the field it waits for has been built.
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
            result = result.mask(matches, self._values(df, condition, context))
            unassigned &= ~matches

        return result

    def _matches(
        self, df: pd.DataFrame, condition: dict, context: TransformationContext
    ) -> pd.Series:
        clauses = condition.get("all_of")
        if clauses is not None:
            matches = pd.Series(True, index=df.index)
            for clause in clauses:
                matches &= self._clause_matches(df, clause, context)
            return matches

        when = condition.get("when")
        if when is None:
            return pd.Series(True, index=df.index)
        return self._clause_matches(df, when, context)

    def _clause_matches(
        self, df: pd.DataFrame, when: dict, context: TransformationContext
    ) -> pd.Series:
        """One ``when`` clause. Raises MissingTargetColumn where it tests a
        target field that is not built yet, so the whole rule is deferred."""
        column = self._tested_column(df, when, context)

        test = when.get("is", "not_empty")
        if test in self._REFERENCE_TESTS:
            contained = self._in_reference(column, when, context)
            return contained if test == "in_reference" else ~contained

        if test not in self._TESTS:
            raise ValueError(
                f"Unknown condition test '{test}' (supported: "
                f"{sorted([*self._TESTS, *self._REFERENCE_TESTS])})"
            )
        return self._TESTS[test](column, when.get("to"))

    def _tested_column(
        self, df: pd.DataFrame, when: dict, context: TransformationContext
    ) -> pd.Series:
        """The column the clause tests - the named field, or the value that
        field resolves to in a reference table when the clause names a
        ``return_field``."""
        column = _resolve_column(df, when["field"], context)
        if when.get("return_field") is None:
            return column
        return self._resolved_through_reference(column, when, context)

    def _resolved_through_reference(
        self, column: pd.Series, when: dict, context: TransformationContext
    ) -> pd.Series:
        """Each row's value replaced by what it maps to in a reference table.
        A value absent from the table resolves to the empty string, so a
        clause can test for that as well."""
        reference = self._reference_table(when, context)
        key_field = when["key_field"]
        return_field = when["return_field"]
        for name in (key_field, return_field):
            if name not in reference.columns:
                raise ValueError(
                    f"Reference table '{when['reference']}' has no column "
                    f"'{name}' (available: {list(reference.columns)})"
                )

        table = reference[[key_field, return_field]].dropna(subset=[key_field])
        resolved = (
            table.assign(
                _key=table[key_field].astype(str).str.strip(),
                _value=table[return_field].astype(str).str.strip(),
            )
            .drop_duplicates(subset="_key")
            .set_index("_key")["_value"]
        )
        return column.astype(str).str.strip().map(resolved).fillna("")

    def _reference_table(
        self, when: dict, context: TransformationContext
    ) -> pd.DataFrame:
        reference = context.reference_data.get(when["reference"])
        if reference is None:
            raise ValueError(
                f"Reference data '{when['reference']}' is not available - "
                f"declare it under task.input.reference_data"
            )
        return reference

    def _in_reference(
        self, column: pd.Series, when: dict, context: TransformationContext
    ) -> pd.Series:
        name = when["reference"]
        reference = self._reference_table(when, context)

        key_field = when["key_field"]
        if key_field not in reference.columns:
            raise ValueError(
                f"Reference table '{name}' has no column '{key_field}' "
                f"(available: {list(reference.columns)})"
            )

        keys = reference[key_field].dropna().astype(str).str.strip()
        return column.astype(str).str.strip().isin(set(keys))

    def _values(
        self,
        df: pd.DataFrame,
        condition: dict,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        from_field = condition.get("from_field")
        if from_field is not None:
            return _resolve_column(df, from_field, context).fillna("").astype(str)

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


class DateDifferenceHandler:
    """The distance between two dates, optionally translated into a label.

    A legacy system often records only the due date, while the target system
    expects the payment term that produced it. The difference in days is the
    bridge; ``value_mapping`` then turns it into the term the target system
    knows, which is why an unmapped difference is rejected rather than
    emitted as a bare number.

    Config keys: ``from_field``, ``to_field``, ``input_format``, and
    optionally ``unit`` (only ``days``) and ``value_mapping``.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        transformation = mapping.transformation
        source_format = translate_date_format(transformation.input_format)

        def parse(column: pd.Series) -> pd.Series:
            return column.map(lambda value: datetime.strptime(str(value).strip(), source_format))

        start = parse(df[transformation.from_field])
        end = parse(df[transformation.to_field])
        days = (end - start).map(lambda delta: delta.days)

        value_map = getattr(transformation, "value_mapping", None)
        if value_map is None:
            return days.astype(str)

        lookup = {str(key): value for key, value in value_map.items()}
        unmapped = sorted({str(d) for d in days if str(d) not in lookup})
        if unmapped:
            raise ValueError(
                f"No value_mapping entry for a difference of {unmapped} day(s) "
                f"in target field '{mapping.target_field}'"
            )
        return days.astype(str).map(lookup)


class UnitConversionHandler:
    """Converts a quantity from the source unit into the target unit.

    Which conversion applies is decided per row by the pair of units: the
    source unit comes from ``source_unit_field``, the target unit from
    ``target_unit_field``, which may name a target column already resolved by
    a lookup. ``conversions`` lists one ``rule`` per ``from``/``to`` pair; a
    pair the list does not cover is an authoring error, not a silent pass
    through.

    A rule may use ``cross_section``, the cross sectional area of a bar, which
    is itself conditional on the profile - a round bar is computed from its
    diameter, a square bar from width and depth. It may also use any column of
    the reference table (``density``), looked up with ``lookup_key`` against
    ``lookup_source_field``, which is how a stock held in pieces becomes a
    weight.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        transformation = mapping.transformation
        source_units = df[transformation.source_unit_field].astype(str).str.strip()
        target_units = self._target_units(df, transformation, context)

        operands = self._operands(df, mapping, transformation, context)
        operands["cross_section"] = self._cross_section(df, transformation)

        result = pd.Series(float("nan"), index=df.index, dtype="float64")
        covered = pd.Series(False, index=df.index)
        for conversion in transformation.conversions:
            applies = (source_units == conversion["from"]) & (target_units == conversion["to"])
            if not applies.any():
                continue
            result = result.mask(applies, evaluate_formula(conversion["rule"], operands))
            covered |= applies

        if not covered.all():
            pairs = sorted(set(zip(source_units[~covered], target_units[~covered])))
            raise ValueError(
                f"No conversion rule for unit pair(s) {pairs} in target field "
                f"'{mapping.target_field}'"
            )

        decimals = _decimals(mapping)
        post_processing = getattr(transformation, "post_processing", None)
        if post_processing and post_processing.get("type") == "round":
            decimals = post_processing.get("decimals", decimals)
            result = result.round(decimals)
        return _format_numeric(result, _separator(mapping, ","), decimals)

    def _target_units(
        self, df: pd.DataFrame, transformation, context: TransformationContext
    ) -> pd.Series:
        name = transformation.target_unit_field
        if name in df.columns:
            return df[name].astype(str).str.strip()
        if name in context.target_columns:
            return context.target_columns[name].astype(str).str.strip()
        raise MissingTargetColumn(name)

    def _operands(
        self, df: pd.DataFrame, mapping: MappingRule, transformation, context
    ) -> dict[str, pd.Series]:
        """The numeric source columns the rules may use, plus every reference
        table column named in ``reference_fields`` (e.g. the density)."""
        numeric_fields = [
            name
            for name in (mapping.source_fields or [])
            if name in df.columns and name != transformation.source_unit_field
        ]
        operands = {}
        for name in numeric_fields:
            numbers, _ = _to_numeric(df[name])
            operands[name] = numbers

        for name in getattr(transformation, "reference_fields", None) or []:
            operands[name] = self._reference_column(df, transformation, context, name)
        return operands

    def _reference_column(
        self, df: pd.DataFrame, transformation, context, column: str
    ) -> pd.Series:
        reference = context.reference_data.get(transformation.reference)
        if reference is None:
            raise ValueError(
                f"Reference data '{transformation.reference}' is not available - "
                f"declare it under task.input.reference_data"
            )
        table = reference[[transformation.lookup_key, column]].dropna().copy()
        table[transformation.lookup_key] = (
            table[transformation.lookup_key].astype(str).str.strip()
        )
        lookup = table.drop_duplicates(subset=transformation.lookup_key).set_index(
            transformation.lookup_key
        )[column]
        keys = df[transformation.lookup_source_field].astype(str).str.strip()
        numbers, _ = _to_numeric(keys.map(lookup))
        return numbers

    def _cross_section(self, df: pd.DataFrame, transformation) -> pd.Series:
        """Cross sectional area in square millimetres, by profile. Rows whose
        profile matches no rule keep NaN - harmless unless a conversion rule
        actually uses the value for that row."""
        spec = getattr(transformation, "cross_section", None)
        if not spec:
            return pd.Series(float("nan"), index=df.index, dtype="float64")

        field_name = spec.get("field", "profile")
        profiles = df[field_name].fillna("").astype(str).str.strip()
        operands = {}
        for name in spec.get("fields", []):
            numbers, _ = _to_numeric(df[name])
            operands[name] = numbers

        area = pd.Series(float("nan"), index=df.index, dtype="float64")
        for rule in spec.get("rules", []):
            applies = profiles == rule["profile"]
            if applies.any():
                area = area.mask(applies, evaluate_formula(rule["formula"], operands))
        return area


class RangeMappingHandler:
    """Maps a numeric source value to the value of the bracket it falls into.

    The brackets are half-open - ``from`` inclusive, ``to`` exclusive - so
    adjacent ones can share a boundary without overlapping, and the first
    matching one wins. An open end is expressed by leaving the bound out::

        ranges:
          - to: 5000
            value: "1"
          - from: 5000
            to: 10000
            value: "3.000"
          - from: 100000
            value: "100.000"

    This is what ``value_mapping`` cannot do: a credit limit class follows
    from a revenue no two customers share, so there is no finite table of
    values to enumerate. ``source_field`` may name a target field as well, so
    a bracket can be applied to a figure the output itself carries.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        numbers, _ = _to_numeric(
            _resolve_column(df, mapping.source_field, context)
        )
        ranges = mapping.transformation.ranges
        if not ranges:
            raise ValueError(
                f"range_mapping for '{mapping.target_field}' declares no ranges"
            )

        result = pd.Series("", index=df.index, dtype="object")
        unassigned = numbers.notna()
        for bracket in ranges:
            if "value" not in bracket:
                raise ValueError(
                    f"Range {bracket} of '{mapping.target_field}' has no 'value'"
                )
            lower, upper = bracket.get("from"), bracket.get("to")
            if lower is None and upper is None:
                raise ValueError(
                    f"Range {bracket} of '{mapping.target_field}' bounds nothing - "
                    f"give it a 'from', a 'to', or both"
                )
            matches = unassigned.copy()
            if lower is not None:
                matches &= numbers >= lower
            if upper is not None:
                matches &= numbers < upper
            if not matches.any():
                continue
            result = result.mask(matches, str(bracket["value"]))
            unassigned &= ~matches

        if unassigned.any():
            uncovered = sorted(set(numbers[unassigned]))
            raise ValueError(
                f"range_mapping for '{mapping.target_field}' covers no bracket for "
                f"value(s) {uncovered[:5]} of '{mapping.source_field}'"
            )
        return result


class BoundedValueHandler:
    """Keeps a numeric value inside a bound the business rules impose.

    Each bound is either a literal (``minimum``/``maximum``) or another
    column's value per row (``minimum_field``/``maximum_field``), optionally
    scaled by ``minimum_factor``/``maximum_factor`` for a field carried with
    the opposite sign. A bound may be left out; at least one has to be given,
    or there is nothing to bound::

        - source_field: actual costs
          target_field: WIP
          transformation:
            type: bounded_value
            maximum_field: planned revenue
            maximum_factor: -1          # posted as a credit
            minimum: 0
            decimals: 2

    The maximum is applied before the minimum, so the floor wins where the
    two cross: work in progress capped at a planned revenue of zero is zero,
    never a negative figure.

    The value and either bound may name a *target* field as well, so a figure
    can be bounded by another field of the output.
    """

    def apply(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext = EMPTY_CONTEXT,
    ) -> pd.Series:
        values, separator = _to_numeric(
            _resolve_column(df, mapping.source_field, context)
        )

        upper = self._bound(df, mapping, context, "maximum")
        lower = self._bound(df, mapping, context, "minimum")
        if upper is None and lower is None:
            raise ValueError(
                f"bounded_value for '{mapping.target_field}' declares neither a "
                f"minimum nor a maximum - there is nothing to bound"
            )

        if upper is not None:
            values = values.clip(upper=upper)
        if lower is not None:
            values = values.clip(lower=lower)
        return _format_numeric(
            values, _separator(mapping, separator), _decimals(mapping)
        )

    def _bound(
        self,
        df: pd.DataFrame,
        mapping: MappingRule,
        context: TransformationContext,
        edge: str,
    ) -> pd.Series | float | None:
        transformation = mapping.transformation
        literal = getattr(transformation, edge, None)
        field_name = getattr(transformation, f"{edge}_field", None)
        if literal is not None and field_name is not None:
            raise ValueError(
                f"bounded_value for '{mapping.target_field}' gives both a "
                f"'{edge}' and a '{edge}_field' - it reads one of them"
            )

        factor = getattr(transformation, f"{edge}_factor", None)
        if field_name is None:
            if literal is None:
                return None
            return literal if factor is None else literal * factor

        bound, _ = _to_numeric(_resolve_column(df, field_name, context))
        return bound if factor is None else bound * factor


DEFAULT_TRANSFORMATION_HANDLERS: dict[str, TransformationHandler] = {
    "copy": CopyHandler(),
    "concatenate": ConcatenateHandler(),
    "value_mapping": ValueMappingHandler(),
    "range_mapping": RangeMappingHandler(),
    "bounded_value": BoundedValueHandler(),
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
    "unit_conversion": UnitConversionHandler(),
    "date_difference": DateDifferenceHandler(),
}
