"""Noise models for NoiseEngine, registered in DEFAULT_NOISE_MODELS by their
``type`` string. Adding a new noise type only requires adding a new handler
and registering it here; NoiseEngine itself never needs to change.

Invariant every model must follow: iteration order over columns/rows must be
fully deterministic given a seed. Iterating a Python ``set`` of column names
is *not* deterministic across process runs (PYTHONHASHSEED randomization) —
always iterate ``df.columns`` (a list) filtered by an excluded-columns set
used only for O(1) membership checks, and rows via ``df.index``.
"""

from __future__ import annotations

import random
from typing import Protocol

import pandas as pd

from agentdatabench.domain.noise_configuration import NoiseTypeConfig


class NoiseModel(Protocol):
    def apply(
        self,
        df: pd.DataFrame,
        config: NoiseTypeConfig,
        rng: random.Random,
        excluded_columns: set[str],
    ) -> pd.DataFrame: ...


class TypoNoiseModel:
    def apply(
        self,
        df: pd.DataFrame,
        config: NoiseTypeConfig,
        rng: random.Random,
        excluded_columns: set[str],
    ) -> pd.DataFrame:
        df = df.copy()
        columns = [c for c in df.columns if c not in excluded_columns]

        for row in df.index:
            for column in columns:
                value = df.at[row, column]
                if pd.isna(value) or len(value) < 2:
                    continue
                if rng.random() < config.probability:
                    i = rng.randrange(len(value) - 1)
                    chars = list(value)
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    df.at[row, column] = "".join(chars)

        return df


class MissingValueNoiseModel:
    def apply(
        self,
        df: pd.DataFrame,
        config: NoiseTypeConfig,
        rng: random.Random,
        excluded_columns: set[str],
    ) -> pd.DataFrame:
        df = df.copy()
        columns = [c for c in df.columns if c not in excluded_columns]

        for row in df.index:
            for column in columns:
                if rng.random() < config.probability:
                    df.at[row, column] = pd.NA

        return df


class DuplicateNoiseModel:
    def apply(
        self,
        df: pd.DataFrame,
        config: NoiseTypeConfig,
        rng: random.Random,
        excluded_columns: set[str],
    ) -> pd.DataFrame:
        rows = []
        for row in df.index:
            rows.append(df.loc[row])
            if rng.random() < config.probability:
                rows.append(df.loc[row])

        return pd.DataFrame(rows).reset_index(drop=True)


DEFAULT_NOISE_MODELS: dict[str, NoiseModel] = {
    "typo": TypoNoiseModel(),
    "missing_value": MissingValueNoiseModel(),
    "duplicate": DuplicateNoiseModel(),
}
