"""AgentAdapter: the extensibility point for the Evaluation Framework
(Subsystem B). Every agent under test (Data Interpreter, PandasAI, a
hand-written baseline, ...) is wrapped by one AgentAdapter subclass, so the
rest of the Evaluation Framework never depends on a specific agent's SDK.

Contract, informed by surveying several real agent implementations (Data
Interpreter, TaskWeaver, PandasAI, DA-Code, Harmonia, ...): none of them
accept our domain objects directly. They are driven by a natural-language
instruction string plus a filesystem workspace containing input files, and
they write their result back into that workspace as a file. `run()`
therefore follows the Template Method pattern: it owns workspace lifecycle,
prompt rendering and output collection, and delegates only the actual agent
invocation to `_invoke`, the one method a concrete adapter must implement.

Two extension points (`_prepare_workspace`, `_build_prompt`) are plain
overridable methods rather than a swappable-strategy registry (unlike
NoiseModel/SynthesisStrategy/TransformationHandler): there is, at this
point, no menu of interchangeable prompt-building algorithms to select
between - just adapter-specific needs (e.g. a DataFrame-only agent that
cannot read a YAML schema file and needs the schema inlined as text
instead). Revisit as a registry only if a second concrete need for that
emerges.

Failures are returned as a failed AgentRunResult, never raised: these agents
execute arbitrary LLM-generated code, so hanging, crashing, or producing no
output file is the normal case, not the exception.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp

from agentdatabench.domain.agent_run_result import AgentRunResult
from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.task import BusinessRules

OUTPUT_FILENAME = "solution.csv"


class AgentAdapter(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    async def run(
        self, package: BenchmarkPackage, *, timeout: float | None = None
    ) -> AgentRunResult:
        """Runs the wrapped agent on `package` in a fresh, isolated workspace
        directory and returns an AgentRunResult - on success or on failure.
        The workspace is left on disk (not cleaned up) so a failed run can be
        inspected; callers that care about disk usage are responsible for
        removing `result.workspace` themselves.
        """
        workspace = Path(mkdtemp(prefix=f"agentdatabench_{self.name}_"))
        started = time.monotonic()

        try:
            self._prepare_workspace(package, workspace)
            prompt = self._build_prompt(package, workspace)
            await asyncio.wait_for(self._invoke(prompt, workspace), timeout=timeout)
        except TimeoutError:
            return self._failure(
                package, workspace, started, f"Agent did not finish within {timeout}s"
            )
        except Exception as exc:
            return self._failure(package, workspace, started, f"{type(exc).__name__}: {exc}")

        output_path = workspace / OUTPUT_FILENAME
        if not output_path.is_file():
            return self._failure(
                package,
                workspace,
                started,
                f"Agent did not produce {OUTPUT_FILENAME} in the workspace",
            )

        return AgentRunResult(
            agent_name=self.name,
            task_id=package.task.task_id,
            success=True,
            duration_seconds=time.monotonic() - started,
            workspace=workspace,
            output_dataset_path=output_path,
        )

    def _failure(
        self, package: BenchmarkPackage, workspace: Path, started: float, error: str
    ) -> AgentRunResult:
        return AgentRunResult(
            agent_name=self.name,
            task_id=package.task.task_id,
            success=False,
            duration_seconds=time.monotonic() - started,
            workspace=workspace,
            error=error,
        )

    def _prepare_workspace(self, package: BenchmarkPackage, workspace: Path) -> None:
        """Materializes the task's input files into `workspace`. Copies
        whatever files the task references rather than assuming CSV-only, so
        future attachments (e.g. PDFs an agent must search) need no change
        here beyond referencing them in `_build_prompt`."""
        shutil.copy(package.dataset.path, workspace / "dataset.csv")
        shutil.copy(
            package.root / package.task.input.source_schema, workspace / "source_schema.yaml"
        )
        shutil.copy(
            package.root / package.task.input.target_schema, workspace / "target_schema.yaml"
        )

    def _build_prompt(self, package: BenchmarkPackage, workspace: Path) -> str:
        task = package.task
        lines = [
            f"Objective: {task.objective.strip()}",
            "",
            f"Required operations: {', '.join(task.required_operations)}",
            *self._render_data_quality_note(task.required_operations),
            "",
            f"Input dataset: {workspace / 'dataset.csv'}",
            f"Source schema: {workspace / 'source_schema.yaml'}",
            f"Target schema: {workspace / 'target_schema.yaml'}",
            "",
            *self._render_business_rules(task.business_rules),
            "Constraints:",
            *(f"- {constraint}" for constraint in task.constraints),
            "",
            f"Write the final result as a CSV file to exactly this path: "
            f"{workspace / OUTPUT_FILENAME}",
        ]
        return "\n".join(lines)

    def _render_data_quality_note(self, required_operations: list[str]) -> list[str]:
        """Warns the agent that the input may be noisy, without revealing
        what's wrong or how to fix it. Without this, a task requiring "data
        cleaning" is unwinnable: ground_truth.csv is derived from a clean
        dataset the agent never sees, so a value that doesn't match any
        mapping key (e.g. a typo'd country code) looks like a dead end
        instead of a cleaning opportunity - confirmed by a smoke test where
        DI left such fields blank instead of attempting a correction."""
        if "data cleaning" not in required_operations:
            return []
        return [
            "",
            "Note: the input dataset may contain data quality issues (typos, "
            "missing values, duplicate rows). Clean the data before applying "
            "the mappings below - a value that doesn't exactly match an "
            "expected mapping key may be a typo of a valid value rather than "
            "a genuinely different value.",
        ]

    def _render_business_rules(self, business_rules: BusinessRules) -> list[str]:
        """Renders task.business_rules into natural language. Without this,
        an agent only sees the schemas and has to guess exact filter cutoffs
        and value-mapping tables (e.g. country code DE -> DEU) that cannot be
        inferred from the schema alone - confirmed by an early smoke test
        where DI produced a schema-correct but content-wrong output because
        the prompt didn't carry this section yet."""
        lines: list[str] = []

        if business_rules.filtering:
            lines.append("Filtering rules:")
            if business_rules.filtering.description:
                lines.append(f"  {business_rules.filtering.description.strip()}")
            for rule in business_rules.filtering.rules:
                condition = f"  - Keep rows where {rule.field} {rule.operator} {rule.value!r}"
                if rule.field_format:
                    condition += f" (source field format: {rule.field_format})"
                lines.append(condition)
            lines.append("")

        if business_rules.mappings:
            lines.append("Field mappings (apply exactly as specified, do not invent your own):")
            for mapping in business_rules.mappings:
                source = mapping.source_field or " + ".join(mapping.source_fields or [])
                transformation = mapping.transformation.model_dump()
                lines.append(f"  - {source} -> {mapping.target_field}: {transformation}")
                if mapping.description:
                    lines.append(f"    ({mapping.description.strip()})")
            lines.append("")

        return lines

    @abstractmethod
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        """Runs the wrapped agent on `prompt` inside `workspace`. Must write
        the result to `workspace / OUTPUT_FILENAME` before returning. Raise
        on unrecoverable failure - `run()` turns that into a failed
        AgentRunResult rather than propagating it."""
        ...
