"""Shared translation of the benchmark packages' custom date-token format
(e.g. "DD.MM.YYYY") into Python's strftime/strptime directives."""

from __future__ import annotations

_DATE_TOKENS = [("YYYY", "%Y"), ("YY", "%y"), ("MM", "%m"), ("DD", "%d")]


def translate_date_format(fmt: str) -> str:
    for token, directive in _DATE_TOKENS:
        fmt = fmt.replace(token, directive)
    return fmt
