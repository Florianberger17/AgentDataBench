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
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
from faker import Faker

from agentdatabench.domain.synthesis_configuration import ColumnSynthesisConfig
from agentdatabench.generator.date_formats import translate_date_format


class SynthesisStrategy(Protocol):
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
    ) -> pd.Series: ...


class FakerStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
    ) -> pd.Series:
        provider = getattr(faker, config.provider)
        return pd.Series([provider() for _ in range(n)])


_DECIMAL_PATTERN = re.compile(r"^-?\d+\.(\d+)$")


class NumericDistributionStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
    ) -> pd.Series:
        raw_values = real_series.dropna().astype(str)
        decimal_lengths = [
            len(match.group(1))
            for value in raw_values
            if (match := _DECIMAL_PATTERN.match(value))
        ]
        is_float = len(decimal_lengths) > 0
        precision = max(decimal_lengths, default=0)

        numeric_values = raw_values.astype(float)
        low, high = numeric_values.min(), numeric_values.max()

        if is_float:
            return pd.Series(
                [f"{rng.uniform(low, high):.{precision}f}" for _ in range(n)]
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
    ) -> pd.Series:
        start = getattr(config, "start", 0)
        value_format = config.format
        return pd.Series([value_format.format(start + i) for i in range(n)])


class CategoricalResampleStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
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
        return pd.Series(rng.choices(distinct, weights=weights, k=n))


class IdentityStrategy:
    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
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


class PartNameReplacementStrategy:
    """Replaces every value with a part name drawn from a CSV library.

    Names are sampled *without replacement*, so within one synthesize() call
    each library entry is used at most once and the output contains no
    duplicates. The library must therefore hold at least `n` distinct
    entries. The first column of the CSV is used; an optional `library_path`
    on the column config overrides DEFAULT_PART_NAME_LIBRARY_PATH.
    """

    def synthesize(
        self,
        real_series: pd.Series,
        config: ColumnSynthesisConfig,
        rng: random.Random,
        faker: Faker,
        n: int,
    ) -> pd.Series:
        library_path = Path(
            getattr(config, "library_path", DEFAULT_PART_NAME_LIBRARY_PATH)
        )
        if not library_path.is_file():
            raise FileNotFoundError(
                f"PartNameReplacementStrategy: part name library not found at "
                f"'{library_path}' (column '{config.column}')"
            )

        # utf-8-sig transparently strips the BOM Excel writes into CSV exports.
        library = pd.read_csv(library_path, encoding="utf-8-sig", dtype=str)
        if library.shape[1] == 0:
            raise ValueError(
                f"PartNameReplacementStrategy: library '{library_path}' has no columns"
            )
        names = library.iloc[:, 0].dropna().str.strip()
        names = names[names != ""]
        # .unique() keeps first-occurrence order (see module docstring), so
        # the same seed always maps to the same sample.
        distinct = names.unique().tolist()

        if n > len(distinct):
            raise ValueError(
                f"PartNameReplacementStrategy: column '{config.column}' needs "
                f"{n} part names but library '{library_path}' only provides "
                f"{len(distinct)} distinct entries. Extend the library or use "
                f"fewer source rows - entries are never reused."
            )

        return pd.Series(rng.sample(distinct, n))


DEFAULT_SYNTHESIS_STRATEGIES: dict[str, SynthesisStrategy] = {
    "faker": FakerStrategy(),
    "numeric_distribution": NumericDistributionStrategy(),
    "date_distribution": DateDistributionStrategy(),
    "unique_sequence": UniqueSequenceStrategy(),
    "categorical_resample": CategoricalResampleStrategy(),
    "identity": IdentityStrategy(),
    "part_name_replacement": PartNameReplacementStrategy(),
}
