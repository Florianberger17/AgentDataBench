"""ValidationResult domain object: the outcome of running a Validator over a
BenchmarkPackage. Mirrors EvaluationResult's shape (a flat list of typed
findings plus an aggregate pass/fail flag) for consistency across the two
report-producing subsystems.
"""

from __future__ import annotations

from typing import Literal

from agentdatabench.domain.common import StrictBaseModel


class ValidationIssue(StrictBaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str


class ValidationResult(StrictBaseModel):
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)
