"""Tests for AgentAdapter, the Template-Method base class every concrete
agent wrapper subclasses. Exercised via small fake adapters instead of a
real agent SDK, since the base class's job is workspace/prompt/timeout/
failure handling around whatever `_invoke` does.
"""

import asyncio
import shutil
import sys
import tempfile
import unittest.mock
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import (
    OUTPUT_FILENAME,
    SOLUTION_SCRIPT_FILENAME,
    AgentAdapter,
)


class _EchoingFakeAdapter(AgentAdapter):
    """Writes the (unmodified) input dataset back as the solution - exercises
    the success path without a real LLM-backed agent."""

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)


class _NoOutputFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        pass


class _NoOutputButReportsMetadataFakeAdapter(AgentAdapter):
    """Reproduces a real Data Interpreter failure mode seen live against
    package 004: the agent runs its full plan (spending real steps/tokens)
    but never actually writes solution.csv - a genuine failure, but one
    where _invoke() still returns metadata that must not be silently
    dropped just because the run overall failed."""

    async def _invoke(self, prompt: str, workspace: Path) -> dict:
        return {"steps": 3, "prompt_tokens": 5322, "completion_tokens": 486}


class _MetadataReportingFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> dict:
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)
        return {"steps": 2, "prompt_tokens": 99}


class _SlowFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        await asyncio.sleep(10)


class _RaisingFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        raise RuntimeError("agent blew up")


def test_run_success_collects_output_dataset(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _EchoingFakeAdapter(name="echo")

    result = asyncio.run(adapter.run(package))

    assert result.success
    assert result.agent_name == "echo"
    assert result.task_id == package.task.task_id
    assert result.error is None
    assert result.output_dataset_path is not None
    assert result.output_dataset_path.is_file()
    assert result.duration_seconds >= 0
    assert result.metadata == {}


def test_run_captures_metadata_returned_by_invoke(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _MetadataReportingFakeAdapter(name="metadata-reporting")

    result = asyncio.run(adapter.run(package))

    assert result.success
    assert result.metadata == {"steps": 2, "prompt_tokens": 99}


def test_run_prepares_workspace_with_input_files(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _EchoingFakeAdapter(name="echo")

    result = asyncio.run(adapter.run(package))

    assert (result.workspace / "dataset.csv").is_file()
    assert (result.workspace / "source_schema.yaml").is_file()
    assert (result.workspace / "target_schema.yaml").is_file()


def test_run_uses_system_temp_when_no_workspace_root_given(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _EchoingFakeAdapter(name="echo")

    result = asyncio.run(adapter.run(package))

    assert str(result.workspace).startswith(tempfile.gettempdir())


def test_run_names_workspace_by_agent_and_timestamp_under_workspace_root(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _EchoingFakeAdapter(name="echo")
    workspace_root = tmp_path / "manual_runs"

    before = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = asyncio.run(adapter.run(package, workspace_root=workspace_root))
    after = datetime.now().strftime("%Y%m%d_%H%M%S")

    assert result.workspace.parent == workspace_root
    assert result.workspace.name.startswith("echo_")
    timestamp = result.workspace.name.removeprefix("echo_")
    assert before <= timestamp <= after


def test_run_falls_back_to_unique_suffix_on_workspace_name_collision(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _EchoingFakeAdapter(name="echo")
    workspace_root = tmp_path / "manual_runs"

    fixed_timestamp = "20260715_185700"
    workspace_root.mkdir(parents=True)
    (workspace_root / f"echo_{fixed_timestamp}").mkdir()

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.strptime(fixed_timestamp, "%Y%m%d_%H%M%S")

    with unittest.mock.patch("agentdatabench.evaluation.agent_adapter.datetime", _FixedDatetime):
        result = asyncio.run(adapter.run(package, workspace_root=workspace_root))

    assert result.workspace.parent == workspace_root
    assert result.workspace.name != f"echo_{fixed_timestamp}"
    assert result.workspace.name.startswith(f"echo_{fixed_timestamp}_")


def test_run_fails_when_no_output_produced(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _NoOutputFakeAdapter(name="no-output")

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert result.output_dataset_path is None
    assert "did not produce" in result.error


def test_run_preserves_metadata_even_when_no_output_produced(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _NoOutputButReportsMetadataFakeAdapter(name="no-output-with-metadata")

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert result.metadata == {"steps": 3, "prompt_tokens": 5322, "completion_tokens": 486}


def test_run_fails_on_exception_instead_of_raising(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _RaisingFakeAdapter(name="raises")

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "agent blew up" in result.error


def test_run_fails_on_timeout(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _SlowFakeAdapter(name="slow")

    result = asyncio.run(adapter.run(package, timeout=0.05))

    assert not result.success
    assert "did not finish" in result.error


class _PromptCapturingFakeAdapter(AgentAdapter):
    def __init__(self, name: str = "prompt-capture") -> None:
        super().__init__(name)
        self.captured_prompt: str | None = None

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        self.captured_prompt = prompt


def test_execution_python_defaults_to_this_process_interpreter():
    adapter = _EchoingFakeAdapter(name="echo")
    assert adapter.execution_python == Path(sys.executable)


def test_prompt_requires_solution_script(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    prompt = adapter.captured_prompt
    assert SOLUTION_SCRIPT_FILENAME in prompt
    assert "reproduc" in prompt.lower()


def test_prompt_includes_filtering_and_mapping_business_rules(pkg1_root):
    # Package 001's task.yaml has a filtering rule and mapping rules with
    # exact value tables (e.g. country code DE -> DEU) that cannot be
    # inferred from the schemas alone - without this in the prompt, no agent
    # could reproduce ground_truth.csv even in principle.
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    prompt = adapter.captured_prompt
    assert "LastOrderDate >= '2023-01-01'" in prompt
    assert "CountryCode -> Country" in prompt
    assert "'DE': 'DEU'" in prompt
    assert "do not invent your own" in prompt


def test_prompt_omits_data_quality_note_when_no_cleaning_required(pkg1_root):
    # Package 001 has no "data cleaning" in required_operations (its
    # dataset.csv isn't noise-injected) - the note would be misleading here.
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    assert "data quality issues" not in adapter.captured_prompt


def test_prompt_includes_data_quality_note_when_cleaning_required(pkg3_root):
    # Package 003 has "data cleaning" in required_operations and a noise-
    # injected dataset.csv - without a heads-up, an agent has no way to know
    # a mapping-table miss (e.g. a typo'd country code) is a cleaning
    # opportunity rather than a dead end.
    package = BenchmarkPackage.load(pkg3_root)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    prompt = adapter.captured_prompt
    assert "data cleaning" in prompt
    assert "data quality issues" in prompt


def _package_with_additional_document(pkg1_root, tmp_path, document_name="extra.pdf"):
    package_root = tmp_path / "pkg"
    shutil.copytree(pkg1_root, package_root)
    (package_root / document_name).write_bytes(b"%PDF-1.4 fake pdf content")

    task_path = package_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    task_data["input"]["additional_documents"] = [document_name]
    task_path.write_text(yaml.safe_dump(task_data))

    return BenchmarkPackage.load(package_root)


def test_prepare_workspace_copies_additional_documents(pkg1_root, tmp_path):
    package = _package_with_additional_document(pkg1_root, tmp_path)
    adapter = _EchoingFakeAdapter(name="echo")

    result = asyncio.run(adapter.run(package))

    copied = result.workspace / "extra.pdf"
    assert copied.is_file()
    assert copied.read_bytes() == b"%PDF-1.4 fake pdf content"


def test_prompt_mentions_additional_documents_when_present(pkg1_root, tmp_path):
    package = _package_with_additional_document(pkg1_root, tmp_path)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    prompt = adapter.captured_prompt
    assert "Additional documents:" in prompt
    assert "extra.pdf" in prompt
    assert "consult them when a required field is empty" in prompt


def test_prompt_omits_additional_documents_section_when_absent(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PromptCapturingFakeAdapter()

    asyncio.run(adapter.run(package))

    assert "Additional documents" not in adapter.captured_prompt
