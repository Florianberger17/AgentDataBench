"""Synthesis strategies for DatasetCreator, registered in
DEFAULT_SYNTHESIS_STRATEGIES by their `strategy` string. Adding a new
strategy only requires adding a new class and registering it here;
DatasetCreator itself never needs to change.

Aside from `identity`, none of these strategies ever emit a real observed
value for numeric/date columns: they fit a simple range from the real data
and draw fresh values from it, so no original measurement survives into the
synthetic output. `identity` is an intentional, explicit opt-out of that
guarantee for columns a benchmark package author has judged non-identifying
(e.g. business/technical codes) - it must be chosen per column, never a
default. Iteration over distinct categorical labels uses pandas' `.unique()`
(first-occurrence order), never a Python `set`, for the same determinism
reasons documented in noise_models.py.

Most strategies see only their own column. A strategy that has to stay
consistent with another one (`date_offset`, which keeps a delivery date at or
after its order date) reads the already-synthesized columns via `synthesized`,
which DatasetCreator fills as it walks the columns in source order. A
dependent column therefore has to come *after* the column it references.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
from faker import Faker

from agentdatabench.domain.synthesis_configuration import ColumnSynthesisConfig
from agentdatabench.generator.date_formats import translate_date_format
from agentdatabench.generator.transformations import evaluate_formula


@dataclass(frozen=True)
class SynthesisContext:
    """What a strategy needs beyond its own column.

    `source_df` is the real frame the column was sliced out of - the only
    place two real columns are still aligned row by row, which a strategy
    fitting the *relationship* between columns needs. `synthesized` holds the
    columns already produced in this run, in source-column order.
    """

    source_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    synthesized: dict[str, pd.Series] = field(default_factory=dict)


EMPTY_SYNTHESIS_CONTEXT = SynthesisContext()


class SynthesisStrategy(Protocol):
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = ...,
    ) -> pd.Series: ...


class FakerStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        provider = getattr(faker, config.provider)
        return pd.Series([provider() for _ in range(n)])


# Numeric columns reach the strategies as strings (Dataset reads every column
# as text), so the decimal separator is whatever the source export wrote:
# "12.5" from an English-locale system, "12,5" from a German one. The
# separator is detected per column and reproduced in the output, so the
# synthetic data keeps the locale convention an agent has to cope with.
_DECIMAL_PATTERN = re.compile(r"^-?\d+([.,])(\d+)$")
_INTEGER_PATTERN = re.compile(r"^-?\d+$")


class NumericDistributionStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        raw_values = real_series.dropna().astype(str).str.strip()
        raw_values = raw_values[raw_values != ""]

        separators: set[str] = set()
        decimal_lengths: list[int] = []
        for value in raw_values:
            if match := _DECIMAL_PATTERN.match(value):
                separators.add(match.group(1))
                decimal_lengths.append(len(match.group(2)))
            elif not _INTEGER_PATTERN.match(value):
                raise ValueError(
                    f"NumericDistributionStrategy: column '{config.column}' "
                    f"contains the non-numeric value '{value}'. This strategy "
                    f"only accepts plain integers and decimals ('1234', "
                    f"'12.5', '12,5') - grouped numbers ('1.234,56'), units "
                    f"and currency symbols must be cleaned from the source "
                    f"data or handled by a different strategy."
                )

        # Both separators in one column is genuinely ambiguous ('1.234' could
        # be a decimal or a grouped integer), so it is refused rather than
        # guessed at.
        if len(separators) > 1:
            raise ValueError(
                f"NumericDistributionStrategy: column '{config.column}' mixes "
                f"'.' and ',' as decimal separators. Normalise the source "
                f"column to one separator."
            )

        decimal_separator = separators.pop() if separators else "."
        precision = max(decimal_lengths, default=0)

        numeric_values = [
            float(value.replace(decimal_separator, ".")) for value in raw_values
        ]
        low, high = min(numeric_values), max(numeric_values)

        if precision > 0:
            return pd.Series(
                [
                    f"{rng.uniform(low, high):.{precision}f}".replace(
                        ".", decimal_separator
                    )
                    for _ in range(n)
                ]
            )
        return pd.Series([str(rng.randint(int(low), int(high))) for _ in range(n)])


class DateDistributionStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        date_format = translate_date_format(config.field_format)
        real_dates = [
            datetime.strptime(str(value), date_format)
            for value in real_series.dropna()
        ]
        low, high = min(real_dates), max(real_dates)
        span_seconds = (high - low).total_seconds()

        result = []
        for _ in range(n):
            offset = timedelta(seconds=rng.random() * span_seconds)
            result.append((low + offset).strftime(date_format))
        return pd.Series(result)


class UniqueSequenceStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        start = getattr(config, "start", 0)
        value_format = config.format
        return pd.Series([value_format.format(start + i) for i in range(n)])


class CategoricalResampleStrategy:
    """Redraws the column from its own real labels, weighted by how often each
    one occurs.

    A label occurring once in a hundred rows is a coin flip to survive that
    redraw, which silently kills any business rule written against it (a
    filter on a rare status value ends up matching nothing). Set
    `ensure_all_values: true` to guarantee every real label appears at least
    once: labels missing after the weighted draw replace occurrences of
    labels that still have one to spare, so the frequency profile stays as
    close as the guarantee allows.

    Config keys: `max_cardinality` (default 20), `ensure_all_values`
    (default false).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        max_cardinality = getattr(config, "max_cardinality", 20)
        values = real_series.dropna().astype(str)

        distinct = values.unique().tolist()
        if len(distinct) > max_cardinality:
            raise ValueError(
                f"CategoricalResampleStrategy: column '{config.column}' has "
                f"{len(distinct)} distinct values, exceeds max_cardinality="
                f"{max_cardinality}. This strategy is only for low-cardinality, "
                f"non-identifying code fields."
            )

        value_counts = values.value_counts()
        weights = [value_counts[label] for label in distinct]
        drawn = rng.choices(distinct, weights=weights, k=n)

        if getattr(config, "ensure_all_values", False):
            drawn = self._ensure_all_values(drawn, distinct, config, rng, n)
        return pd.Series(drawn)

    def _ensure_all_values(
        self,
        drawn: list[str],
        distinct: list[str],
        config: ColumnSynthesisConfig,
        rng: random.Random,
        n: int,
    ) -> list[str]:
        if len(distinct) > n:
            raise ValueError(
                f"CategoricalResampleStrategy: column '{config.column}' has "
                f"{len(distinct)} distinct values but only {n} rows to place "
                f"them in - ensure_all_values cannot be satisfied."
            )

        counts = Counter(drawn)
        for label in distinct:
            if counts[label]:
                continue
            # Only overwrite a label that still has another occurrence left,
            # so satisfying one missing label never drops a different one.
            positions = [index for index, value in enumerate(drawn) if counts[value] > 1]
            position = rng.choice(positions)
            counts[drawn[position]] -= 1
            drawn[position] = label
            counts[label] += 1
        return drawn


class IdentityStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        return real_series.reset_index(drop=True)


# Curated library of realistic, non-identifying part names shipped with the
# repo. Resolved relative to this file so the default works from any CWD.
DEFAULT_PART_NAME_LIBRARY_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "part_name_library"
    / "part_name_library.csv"
)


def _load_part_names(config: ColumnSynthesisConfig) -> tuple[list[str], Path]:
    """The distinct part names of the configured CSV library, in
    first-occurrence order (see module docstring) so the same seed always
    draws the same sample. The first column of the CSV is used; an optional
    `library_path` on the column config overrides
    DEFAULT_PART_NAME_LIBRARY_PATH."""
    library_path = Path(getattr(config, "library_path", DEFAULT_PART_NAME_LIBRARY_PATH))
    if not library_path.is_file():
        raise FileNotFoundError(
            f"Part name library not found at '{library_path}' "
            f"(column '{config.column}')"
        )

    # utf-8-sig transparently strips the BOM Excel writes into CSV exports.
    library = pd.read_csv(library_path, encoding="utf-8-sig", dtype=str)
    if library.shape[1] == 0:
        raise ValueError(f"Part name library '{library_path}' has no columns")

    names = library.iloc[:, 0].dropna().str.strip()
    names = names[names != ""]
    return names.unique().tolist(), library_path


def _draw_distinct_faker_values(provider, count: int, config: ColumnSynthesisConfig) -> list[str]:
    """`count` distinct values from a faker provider.

    Faker repeats itself, so drawing `count` times would yield duplicates and
    silently collapse two real values onto one synthetic value - exactly what
    the mapping strategies promise not to do. Draws until enough distinct
    values exist instead, and gives up rather than looping forever on a
    provider too low-cardinality for the column.
    """
    values: list[str] = []
    seen: set[str] = set()
    for _ in range(count * 100):
        if len(values) == count:
            return values
        value = str(provider())
        if value not in seen:
            seen.add(value)
            values.append(value)
    raise ValueError(
        f"Column '{config.column}' needs {count} distinct values from faker "
        f"provider '{config.provider}', but only {len(values)} could be drawn - "
        f"the provider is too low-cardinality for this column."
    )


def _draw_names(
    available: list[str],
    count: int,
    config: ColumnSynthesisConfig,
    rng: random.Random,
) -> list[str]:
    """`count` distinct library entries, honouring an optional guarantee that
    some of them exceed `ensure_longer_than` characters.

    Without it, a library of mostly short entries never produces a value long
    enough to exercise a truncation rule: the target field's length limit is
    then never reached and the rule is dead. `ensure_count` (default 1) says
    how many drawn entries must exceed the threshold.
    """
    threshold = getattr(config, "ensure_longer_than", None)
    if threshold is None:
        return rng.sample(available, count)

    required = getattr(config, "ensure_count", 1)
    longer = [name for name in available if len(name) > threshold]
    if len(longer) < required:
        raise ValueError(
            f"Column '{config.column}' requires {required} library entries "
            f"longer than {threshold} characters, but only {len(longer)} are "
            f"available. Extend the part name library or lower ensure_count."
        )
    if required > count:
        raise ValueError(
            f"Column '{config.column}' requires {required} entries longer than "
            f"{threshold} characters but only draws {count} values in total."
        )

    guaranteed = rng.sample(longer, required)
    remaining = [name for name in available if name not in set(guaranteed)]
    names = guaranteed + rng.sample(remaining, count - required)
    rng.shuffle(names)
    return names


class PartNameReplacementStrategy:
    """Replaces every value with a part name drawn from a CSV library.

    Names are sampled *without replacement*, so within one synthesize() call
    each library entry is used at most once and the output contains no
    duplicates. The library must therefore hold at least `n` distinct
    entries. Use `part_name_mapping` instead when a repeated real value has
    to stay a repeated synthetic one.
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        distinct, library_path = _load_part_names(config)

        if n > len(distinct):
            raise ValueError(
                f"PartNameReplacementStrategy: column '{config.column}' needs "
                f"{n} part names but library '{library_path}' only provides "
                f"{len(distinct)} distinct entries. Extend the library or use "
                f"fewer source rows - entries are never reused."
            )

        return pd.Series(_draw_names(distinct, n, config, rng))


class PartNameMappingStrategy:
    """One part name per *distinct* real value, substituted consistently.

    `part_name_replacement` draws a fresh name per row, which destroys any
    repetition the column carries: a header designation occurring 19 times
    becomes 19 different names, and its 1:1 pairing with the part number
    beside it is lost. This strategy builds a substitution table instead -
    every occurrence of the same real value becomes the same synthetic name -
    so the repetition pattern and any pairing with another column survive,
    while no real designation does.

    Names are drawn without replacement, so two distinct real values never
    collapse onto one synthetic name. The library therefore only has to cover
    the *distinct* values, not the rows. Empty values stay empty: there is
    nothing to substitute.

    The mapping is per column. Two columns using this strategy get
    independent tables, so a value occurring in both would be replaced
    differently in each.
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        available, library_path = _load_part_names(config)

        values = real_series.head(n)
        present = values[~_is_blank(values)].astype(str).str.strip()
        distinct = present.unique().tolist()

        if len(distinct) > len(available):
            raise ValueError(
                f"PartNameMappingStrategy: column '{config.column}' has "
                f"{len(distinct)} distinct values but library '{library_path}' "
                f"only provides {len(available)} distinct entries. Extend the "
                f"library - two real values must not share one synthetic name."
            )

        substitution = dict(
            zip(distinct, _draw_names(available, len(distinct), config, rng))
        )
        return values.map(
            lambda value: value
            if _is_blank_value(value)
            else substitution[str(value).strip()]
        ).reset_index(drop=True)


class NumericMappingStrategy:
    """One random number per *distinct* real value, substituted consistently
    and rendered through a configurable format string.

    Numbers are drawn without replacement from `[min, max]`, so two distinct
    real values never collapse onto the same synthetic one - the same
    guarantee `part_name_mapping` gives for text. Use this instead when the
    column already *is* a formatted number (e.g. a material number like
    "3001-6994") rather than a name drawn from a library.

    Config keys: `format` (a Python str.format() template applied to the
    drawn integer, e.g. `"00-{:06d}"`), `min` and `max` (inclusive bounds of
    the integer drawn before formatting). Empty values stay empty.
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        value_format = config.format
        minimum = config.min
        maximum = config.max
        if minimum > maximum:
            raise ValueError(
                f"NumericMappingStrategy: column '{config.column}' has min "
                f"({minimum}) greater than max ({maximum})."
            )

        values = real_series.head(n)
        present = values[~_is_blank(values)].astype(str).str.strip()
        distinct = present.unique().tolist()

        capacity = maximum - minimum + 1
        if len(distinct) > capacity:
            raise ValueError(
                f"NumericMappingStrategy: column '{config.column}' has "
                f"{len(distinct)} distinct values but the range [{minimum}, "
                f"{maximum}] only provides {capacity} distinct numbers. "
                f"Widen the range - two real values must not share one "
                f"synthetic number."
            )

        drawn = rng.sample(range(minimum, maximum + 1), len(distinct))
        substitution = {
            value: value_format.format(number)
            for value, number in zip(distinct, drawn)
        }
        return values.map(
            lambda value: value
            if _is_blank_value(value)
            else substitution[str(value).strip()]
        ).reset_index(drop=True)


def _is_blank_value(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _is_blank(column: pd.Series) -> pd.Series:
    return column.isna() | (column.astype(str).str.strip() == "")


class FakerMappingStrategy:
    """One faker value per *distinct* real value, substituted consistently.

    The Faker-backed sibling of `part_name_mapping`: where that one draws
    replacements from a CSV library, this draws them from a faker provider,
    which suits columns holding names faker actually models - companies,
    cities, people - rather than part designations.

    Every occurrence of the same real value becomes the same faker value, so
    the repetition pattern and any pairing with another column survive while
    no real value does. Values are drawn distinct, so two different real
    values never collapse onto one.

    Config keys: `provider` (required). Empty values stay empty.

    Unlike `faker_categorical`, the substitution is row-faithful rather than
    resampled: `faker_categorical` redraws the column from the real frequency
    distribution, so row *i* has no relation to the real value in row *i*,
    which breaks any pairing with a neighbouring column. Use that one to
    preserve a frequency profile, this one to preserve the rows themselves.
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        provider = getattr(faker, config.provider)

        values = real_series.head(n)
        present = values[~_is_blank(values)].astype(str).str.strip()
        distinct = present.unique().tolist()

        drawn = _draw_distinct_faker_values(provider, len(distinct), config)
        substitution = dict(zip(distinct, drawn))
        return values.map(
            lambda value: value
            if _is_blank_value(value)
            else substitution[str(value).strip()]
        ).reset_index(drop=True)


class DateOffsetStrategy:
    """A date derived from another synthesized date column, drawn from the
    real offsets observed between the two (e.g. a confirmed delivery date as
    its purchase order date plus a real lead time).

    `date_distribution` on both columns independently would produce delivery
    dates before their own order date - in 13% of rows, measured on the
    package this strategy was written for - which no "clean" dataset should
    contain. Offsets are resampled from the real pairs rather than drawn from
    a fitted range, so the synthetic lead times keep the shape of the real
    ones; only pairs whose real offset satisfies `min_offset_days` contribute,
    so a contradiction in the source data is not carried over.

    Config keys: `reference_column` (the already-synthesized column to offset
    from), `field_format`, and optionally `reference_format` (defaults to
    `field_format`) and `min_offset_days` (defaults to 0).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        reference_column = config.reference_column
        reference = context.synthesized.get(reference_column)
        if reference is None:
            raise ValueError(
                f"Column '{config.column}' offsets from '{reference_column}', "
                f"which has not been synthesized yet - move it above "
                f"'{config.column}' in synthesis_configuration.yaml"
            )

        date_format = translate_date_format(config.field_format)
        reference_format = translate_date_format(
            getattr(config, "reference_format", None) or config.field_format
        )
        minimum = getattr(config, "min_offset_days", 0)

        offsets = self._real_offsets(
            real_series, context.source_df, reference_column, reference_format, minimum
        )
        if not offsets:
            raise ValueError(
                f"No real offset of at least {minimum} day(s) between "
                f"'{reference_column}' and '{config.column}' to resample from"
            )

        result = []
        for value in reference.head(n):
            base = datetime.strptime(str(value), reference_format)
            result.append((base + timedelta(days=rng.choice(offsets))).strftime(date_format))
        return pd.Series(result)

    def _real_offsets(
        self,
        real_series: pd.Series,
        source_df: pd.DataFrame,
        reference_column: str,
        reference_format: str,
        minimum: int,
    ) -> list[int]:
        """The real day offsets between the two real columns, which are only
        aligned row by row in the source frame."""
        if reference_column not in source_df.columns:
            raise ValueError(
                f"Real values of '{reference_column}' are not available to fit "
                f"offsets from"
            )
        real_reference = source_df[reference_column]

        offsets = []
        for own, other in zip(real_series, real_reference):
            if pd.isna(own) or pd.isna(other):
                continue
            days = (
                datetime.strptime(str(own), reference_format)
                - datetime.strptime(str(other), reference_format)
            ).days
            if days >= minimum:
                offsets.append(days)
        return offsets


class FakerCategoricalStrategy:
    """One synthetic label per distinct real value, then resampled with the
    real frequencies - `faker` and `categorical_resample` combined.

    `categorical_resample` keeps the real labels, which is fine for codes but
    not for person names; plain `faker` protects them but draws a fresh value
    per row, destroying the low cardinality a column like "requisitioner" has
    (5 buyers over 100 rows) and with it any mapping keyed on those labels.
    This strategy keeps the shape - the same number of distinct values, drawn
    with the real frequencies as weights - while no real label survives. The
    realized counts follow the real proportions in distribution, not row for
    row, exactly as `categorical_resample` does.

    The label assignment is by descending real frequency, so the most common
    real value maps to the first synthetic one: a stable, seed-reproducible
    assignment that a task.yaml value_mapping can be written against.

    Config keys: `provider` (required), `max_cardinality` (default 20).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        max_cardinality = getattr(config, "max_cardinality", 20)
        values = real_series.dropna().astype(str)
        value_counts = values.value_counts()

        if len(value_counts) > max_cardinality:
            raise ValueError(
                f"FakerCategoricalStrategy: column '{config.column}' has "
                f"{len(value_counts)} distinct values, exceeds max_cardinality="
                f"{max_cardinality}. This strategy is only for low-cardinality "
                f"columns whose labels must not survive."
            )

        provider = getattr(faker, config.provider)
        labels = _draw_distinct_faker_values(provider, len(value_counts), config)
        weights = value_counts.tolist()
        return pd.Series(rng.choices(labels, weights=weights, k=n))


class ConstantStrategy:
    """The same fixed value in every row.

    `identity` would carry the real value through, which is wrong for a code
    a benchmark package wants to standardise - a plant or a fiscal year that
    should read the same everywhere regardless of what the export happened to
    contain. Config keys: `value` (required).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        return pd.Series(str(config.value), index=range(n), dtype="object")


class ComputedStrategy:
    """Derives the column from other, already-synthesized columns.

    Some columns are not independent facts but consequences: a stock value is
    quantity times price divided by price unit, and the source schema states
    exactly that as a constraint. Resynthesizing the operands while keeping
    the result would break it in every row - the same way drawing two date
    columns independently breaks their order.

    The referenced columns have to be configured above this one, since they
    must already be synthesized. Config keys: `formula` (required), `decimals`
    (optional, the places the result is rounded and rendered to).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        formula = config.formula
        decimals = getattr(config, "decimals", None)

        operands: dict[str, pd.Series] = {}
        separator = "."
        for name in _formula_operands(formula, context.synthesized):
            numbers, found = _as_numeric(context.synthesized[name].head(n))
            operands[name] = numbers
            if found == ",":
                separator = found

        if not operands:
            raise ValueError(
                f"Column '{config.column}' computes {formula!r} but none of its "
                f"operands has been synthesized yet - move them above "
                f"'{config.column}' in synthesis_configuration.yaml"
            )

        result = evaluate_formula(formula, operands)
        if decimals is not None:
            result = result.round(decimals)
        return _render_numeric(result, separator, decimals)


def _formula_operands(formula: str, available: dict[str, pd.Series]) -> list[str]:
    """The synthesized column names the formula mentions, longest first so a
    name cannot be mistaken for a prefix of another."""
    return [name for name in sorted(available, key=len, reverse=True) if name in formula]


def _as_numeric(column: pd.Series) -> tuple[pd.Series, str]:
    text = column.astype(str).str.strip()
    separator = "," if text.str.contains(",", regex=False).any() else "."
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce"), separator


def _render_numeric(values: pd.Series, separator: str, decimals: int | None) -> pd.Series:
    def render(value: float) -> str:
        if pd.isna(value):
            return ""
        if decimals is not None:
            return f"{value:.{decimals}f}"
        return str(int(value)) if float(value).is_integer() else repr(float(value))

    rendered = values.map(render)
    return rendered.str.replace(".", ",", regex=False) if separator == "," else rendered


class ConditionalResampleStrategy:
    """Resamples this column *within* the groups another column defines, so a
    pairing between the two survives.

    `categorical_resample` draws each column independently, which silently
    breaks any agreement between them: a validity indicator saying "valid
    from" ends up on rows carrying no date, and rows that are generally valid
    end up carrying one - 13 of 100 rows, measured on the package this
    strategy was written for, in source data where that never happens.

    For every row the value is drawn from the real values observed *for that
    row's reference value*. A combination absent from the real data can
    therefore never appear in the synthetic data, empty values included: if a
    reference group never carries a value, its synthetic rows stay empty too.

    The reference column has to be configured above this one, since it must
    already be synthesized. Config keys: `reference_column` (required).
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
        context: SynthesisContext = EMPTY_SYNTHESIS_CONTEXT,
    ) -> pd.Series:
        reference_column = config.reference_column
        reference = context.synthesized.get(reference_column)
        if reference is None:
            raise ValueError(
                f"Column '{config.column}' resamples within '{reference_column}', "
                f"which has not been synthesized yet - move it above "
                f"'{config.column}' in synthesis_configuration.yaml"
            )
        if reference_column not in context.source_df.columns:
            raise ValueError(
                f"Real values of '{reference_column}' are not available to group by"
            )

        pools = self._pools(real_series, context.source_df[reference_column])

        result = []
        for group in reference.head(n).astype(str):
            pool = pools.get(group)
            if not pool:
                raise ValueError(
                    f"No real values of '{config.column}' observed for "
                    f"'{reference_column}' == {group!r} - cannot resample within "
                    f"a group the real data does not have"
                )
            result.append(rng.choice(pool))
        return pd.Series(result)

    def _pools(
        self, real_series: pd.Series, real_reference: pd.Series
    ) -> dict[str, list[object]]:
        """Real values per reference group, NaN included - an empty cell is an
        observation about that group, not a missing one."""
        pools: dict[str, list[object]] = {}
        for group, value in zip(real_reference.astype(str), real_series):
            pools.setdefault(group, []).append(value)
        return pools


DEFAULT_SYNTHESIS_STRATEGIES: dict[str, SynthesisStrategy] = {
    "faker": FakerStrategy(),
    "numeric_distribution": NumericDistributionStrategy(),
    "date_distribution": DateDistributionStrategy(),
    "unique_sequence": UniqueSequenceStrategy(),
    "categorical_resample": CategoricalResampleStrategy(),
    "identity": IdentityStrategy(),
    "part_name_replacement": PartNameReplacementStrategy(),
    "date_offset": DateOffsetStrategy(),
    "faker_categorical": FakerCategoricalStrategy(),
    "part_name_mapping": PartNameMappingStrategy(),
    "conditional_resample": ConditionalResampleStrategy(),
    "numeric_mapping": NumericMappingStrategy(),
    "faker_mapping": FakerMappingStrategy(),
    "computed": ComputedStrategy(),
    "constant": ConstantStrategy(),
}
