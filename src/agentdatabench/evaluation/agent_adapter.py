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
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from agentdatabench.domain.agent_run_result import AgentRunResult
from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.task import BusinessRules

OUTPUT_FILENAME = "solution.csv"
# The agent's own documentation of how it produced OUTPUT_FILENAME - re-run
# by ReproducibilityCheck to verify the agent's result is actually
# reproducible, not a one-off. See reproducibility.py.
SOLUTION_SCRIPT_FILENAME = "solution.py"


class AgentAdapter(ABC):
    def __init__(self, name: str, default_workspace_root: Path | None = None) -> None:
        self.name = name
        # A concrete adapter can set its own default (e.g. DataInterpreterAdapter
        # points this at venv-di/metagpt-runtime/manual_runs by default) so
        # callers get durable, easy-to-find workspaces without having to pass
        # workspace_root on every single run() call.
        self._default_workspace_root = default_workspace_root

    @property
    def execution_python(self) -> Path:
        """The interpreter that can run this agent's generated code (e.g.
        SOLUTION_SCRIPT_FILENAME) - not necessarily the interpreter running
        this process. Defaults to this process's own interpreter; overridden
        by adapters that run the agent in a separate environment (e.g.
        DataInterpreterAdapter's venv-di), since re-executing generated code
        with the wrong interpreter would report false irreproducibility from
        a missing dependency, not a genuine agent failure."""
        return Path(sys.executable)

    async def run(
        self,
        package: BenchmarkPackage,
        *,
        timeout: float | None = None,
        workspace_root: Path | None = None,
    ) -> AgentRunResult:
        """Runs the wrapped agent on `package` in a fresh, isolated workspace
        directory and returns an AgentRunResult - on success or on failure.
        The workspace is left on disk (not cleaned up) so a failed run can be
        inspected; callers that care about disk usage are responsible for
        removing `result.workspace` themselves.

        `workspace_root` (falling back to `self._default_workspace_root`, and
        from there to the system temp directory if neither is set) is where
        the workspace is created. A system-temp workspace is fine for quick/
        throwaway calls, but can be cleaned up by the OS at any time and
        isn't a reliable place to keep results. Pass an explicit
        `workspace_root` (e.g. a project-relative directory) for runs whose
        output you want to keep; the workspace is then named
        `<agent_name>_<timestamp>` under it, e.g.
        `data-interpreter_20260715_185700`.
        """
        workspace = self._make_workspace(workspace_root or self._default_workspace_root)
        started = time.monotonic()

        try:
            self._prepare_workspace(package, workspace)
            prompt = self._build_prompt(package, workspace)
            metadata = await asyncio.wait_for(self._invoke(prompt, workspace), timeout=timeout)
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
                metadata=metadata,
            )

        return AgentRunResult(
            agent_name=self.name,
            task_id=package.task.task_id,
            success=True,
            duration_seconds=time.monotonic() - started,
            workspace=workspace,
            output_dataset_path=output_path,
            metadata=metadata or {},
        )

    def _make_workspace(self, workspace_root: Path | None) -> Path:
        if workspace_root is None:
            return Path(mkdtemp(prefix=f"agentdatabench_{self.name}_"))

        workspace_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = workspace_root / f"{self.name}_{timestamp}"
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            # Two runs started within the same second - fall back to a
            # unique suffix rather than silently reusing the other run's
            # workspace.
            return Path(mkdtemp(prefix=f"{self.name}_{timestamp}_", dir=workspace_root))

    def _failure(
        self,
        package: BenchmarkPackage,
        workspace: Path,
        started: float,
        error: str,
        metadata: dict | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            agent_name=self.name,
            task_id=package.task.task_id,
            success=False,
            duration_seconds=time.monotonic() - started,
            workspace=workspace,
            error=error,
            metadata=metadata or {},
        )

    def _prepare_workspace(self, package: BenchmarkPackage, workspace: Path) -> None:
        """Materializes the task's input files into `workspace`. Copies
        whatever files the task references rather than assuming CSV-only, so
        additional attachments (e.g. a PDF an agent must search for
        information missing from the CSV) need no change here beyond the
        `additional_documents` loop below and being mentioned in
        `_build_prompt`."""
        shutil.copy(package.dataset.path, workspace / "dataset.csv")
        shutil.copy(
            package.root / package.task.input.source_schema, workspace / "source_schema.yaml"
        )
        shutil.copy(
            package.root / package.task.input.target_schema, workspace / "target_schema.yaml"
        )
        for document in package.task.input.additional_documents or []:
            source_path = package.root / document
            shutil.copy(source_path, workspace / source_path.name)

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
            *self._render_additional_documents(task.input.additional_documents, workspace),
            *self._render_business_rules(task.business_rules),
            "Constraints:",
            *(f"- {constraint}" for constraint in task.constraints),
            "",
            f"Write the final result as a CSV file to exactly this path: "
            f"{workspace / OUTPUT_FILENAME}",
            "",
            f"Additionally, produce a self-contained, commented Python script "
            f"at exactly this path: {workspace / SOLUTION_SCRIPT_FILENAME}. "
            f"Running that script on its own, with no other prior state, "
            f"must read the input dataset from {workspace / 'dataset.csv'} "
            f"and deterministically reproduce the exact same result at "
            f"{workspace / OUTPUT_FILENAME} - this script is how the "
            f"reproducibility of your solution will be verified.",
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

    def _render_additional_documents(
        self, additional_documents: list[str] | None, workspace: Path
    ) -> list[str]:
        """Points the agent at supplementary files (e.g. a PDF order
        confirmation) and hints that they may hold information missing from
        the dataset - without saying which record or field, so the agent
        still has to notice the gap and go looking, not just copy a value
        it's told about upfront."""
        if not additional_documents:
            return []
        lines = ["Additional documents:"]
        for document in additional_documents:
            lines.append(f"  - {workspace / Path(document).name}")
        lines.append(
            "Note: some information required for the mappings below may be "
            "missing from the input dataset but available in the additional "
            "documents above - consult them when a required field is empty."
        )
        lines.append("")
        return lines

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
    async def _invoke(self, prompt: str, workspace: Path) -> dict | None:
        """Runs the wrapped agent on `prompt` inside `workspace`. Must write
        the result to `workspace / OUTPUT_FILENAME` before returning. Raise
        on unrecoverable failure - `run()` turns that into a failed
        AgentRunResult rather than propagating it.

        May return a dict of whatever run metadata the underlying agent
        framework happens to expose (e.g. {"steps": 3, "prompt_tokens": 407,
        "completion_tokens": 37}) - merged verbatim into AgentRunResult.metadata.
        Left as an open dict rather than fixed fields since what's available
        differs entirely per framework (e.g. Direct-LLM has no step concept
        at all); return None/{} if nothing is available."""
        ...
