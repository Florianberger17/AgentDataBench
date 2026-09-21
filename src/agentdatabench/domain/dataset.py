"""Dataset domain object: a CSV-backed dataset referenced by a benchmark package.

Kept as a plain class (not a pydantic model) since it wraps a ``pandas.DataFrame``
and lazy file IO rather than validated scalar data. Delimiters are sniffed per file
because real benchmark packages have been observed to mix comma- and
semicolon-delimited CSVs within the same package.

All columns are read as strings (``dtype=str``): benchmark data is full of
codes that look numeric but must not be interpreted as such (postal codes,
zero-padded dates/IDs). pandas' default type inference silently strips
leading zeros from such columns, which corrupts them irrecoverably. Code that
needs numeric comparisons (e.g. filtering) must coerce explicitly.

CSV files are read and written as ``utf-8-sig``: Excel on Windows opens a
BOM-less UTF-8 CSV as cp1252 and renders umlauts as mojibake ("DÃ¶rr"), and
its own CSV exports carry a BOM. ``utf-8-sig`` writes that BOM and strips it
transparently on read, while plain UTF-8 input reads unchanged.
"""

from __future__ import annotations

import csv
from functools import cached_property
from pathlib import Path

import pandas as pd

CSV_ENCODING = "utf-8-sig"


class Dataset:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path

    def _sniff_delimiter(self) -> str:
        with self.path.open("r", encoding=CSV_ENCODING) as f:
            sample = f.read(4096)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ","

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path, delimiter=self._sniff_delimiter(), dtype=str, encoding=CSV_ENCODING
        )

    def __repr__(self) -> str:
        return f"Dataset(path={self.path!r})"
