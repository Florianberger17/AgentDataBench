"""Tests for DirectLLMAdapter. Injects a fake generate_code (and, where
relevant, a fake extract_pdf_text) so these run without a real LLM call or a
genuinely parseable PDF fixture. Code *execution* itself is not mocked -
it's just `sys.executable script.py` in this project's own venv, safe and
deterministic to run for real in tests.
"""

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.direct_llm_adapter import (
    DirectLLMAdapter,
    GeneratedCode,
    _extract_code,
)

_ECHO_SCRIPT = """
```python
import pandas as pd
df = pd.read_csv(r"{dataset_path}", dtype=str)
df.to_csv(r"{output_path}", index=False)
```
"""


def _echo_code(dataset_path: Path, output_path: Path):
    def generate_code(prompt: str, model: str) -> GeneratedCode:
        return GeneratedCode(_ECHO_SCRIPT.format(dataset_path=dataset_path, output_path=output_path))

    return generate_code


def test_extract_code_parses_fenced_python_block():
    response = "Some preamble\n```python\nprint('hi')\n```\ntrailing text"
    assert _extract_code(response) == "print('hi')"


def test_extract_code_raises_when_no_code_block_present():
    with pytest.raises(ValueError, match="No fenced Python code block"):
        _extract_code("just plain text, no code block")


def test_build_prompt_inlines_dataset_and_schema_content(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = DirectLLMAdapter(default_workspace_root=tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter._prepare_workspace(package, workspace)

    prompt = adapter._build_prompt(package, workspace)

    assert "--- Input dataset content (dataset.csv) ---" in prompt
    assert "CustNo,CompanyName" in prompt  # actual CSV header, not just a path
    assert "--- Source schema content ---" in prompt
    assert "--- Target schema content ---" in prompt


def test_build_prompt_inlines_target_example_instead_of_schemas(pkg1_root, tmp_path):
    package_root = tmp_path / "pkg"
    shutil.copytree(pkg1_root, package_root)

    ground_truth_lines = (package_root / "ground_truth" / "ground_truth.csv").read_text().splitlines()
    (package_root / "data" / "target_example.csv").write_text(
        "\n".join(ground_truth_lines[:3]) + "\n"
    )
    task_path = package_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    del task_data["input"]["source_schema"]
    del task_data["input"]["target_schema"]
    task_data["input"]["target_example"] = "data/target_example.csv"
    task_path.write_text(yaml.safe_dump(task_data))
    package = BenchmarkPackage.load(package_root)

    adapter = DirectLLMAdapter(default_workspace_root=tmp_path / "runs")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter._prepare_workspace(package, workspace)

    prompt = adapter._build_prompt(package, workspace)

    assert "--- Target example content (target_example.csv) ---" in prompt
    assert ground_truth_lines[0] in prompt  # the example CSV's header row
    assert "--- Source schema content ---" not in prompt
    assert "--- Target schema content ---" not in prompt


def test_build_prompt_inlines_pdf_text_via_injected_extractor(pkg1_root, tmp_path):
    package_root = tmp_path / "pkg"
    shutil.copytree(pkg1_root, package_root)
    (package_root / "order.pdf").write_bytes(b"%PDF-1.4 fake")
    task_path = package_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    task_data["input"]["additional_documents"] = ["order.pdf"]
    task_path.write_text(yaml.safe_dump(task_data))
    package = BenchmarkPackage.load(package_root)

    adapter = DirectLLMAdapter(
        default_workspace_root=tmp_path / "runs",
        extract_pdf_text=lambda path: f"FAKE EXTRACTED TEXT FROM {path.name}",
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter._prepare_workspace(package, workspace)

    prompt = adapter._build_prompt(package, workspace)

    assert "--- Extracted text content of order.pdf ---" in prompt
    assert "FAKE EXTRACTED TEXT FROM order.pdf" in prompt


def test_run_executes_generated_code_and_collects_output(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    workspace_root = tmp_path / "runs"

    # generate_code doesn't know the real workspace path in advance, so
    # build it lazily from the prompt the adapter passes to _invoke.
    def generate_code(prompt: str, model: str) -> GeneratedCode:
        dataset_line = next(
            line for line in prompt.splitlines() if line.startswith("Input dataset:")
        )
        dataset_path = Path(dataset_line.split(": ", 1)[1])
        output_path = dataset_path.parent / "solution.csv"
        return GeneratedCode(
            text=_ECHO_SCRIPT.format(dataset_path=dataset_path, output_path=output_path),
            usage={"prompt_tokens": 111, "completion_tokens": 22, "total_tokens": 133},
        )

    adapter = DirectLLMAdapter(
        default_workspace_root=workspace_root, generate_code=generate_code, model="test-model"
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert result.output_dataset_path.is_file()
    assert (result.workspace / "prompt.txt").is_file()
    assert (result.workspace / "response.txt").is_file()
    assert (result.workspace / "solution.py").is_file()
    assert (result.workspace / "run.log").is_file()
    assert result.metadata == {
        "prompt_tokens": 111,
        "completion_tokens": 22,
        "total_tokens": 133,
        "model": "test-model",
    }


def test_run_metadata_has_only_model_when_generate_code_reports_no_usage(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    def generate_code(prompt: str, model: str) -> GeneratedCode:
        dataset_line = next(
            line for line in prompt.splitlines() if line.startswith("Input dataset:")
        )
        dataset_path = Path(dataset_line.split(": ", 1)[1])
        output_path = dataset_path.parent / "solution.csv"
        return GeneratedCode(
            text=_ECHO_SCRIPT.format(dataset_path=dataset_path, output_path=output_path)
        )

    adapter = DirectLLMAdapter(
        default_workspace_root=tmp_path, generate_code=generate_code, model="test-model"
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert result.metadata == {"model": "test-model"}


def test_run_fails_cleanly_when_generate_code_raises(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    def failing_generate_code(prompt: str, model: str) -> GeneratedCode:
        raise RuntimeError("LLM call blew up")

    adapter = DirectLLMAdapter(default_workspace_root=tmp_path, generate_code=failing_generate_code)

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "LLM call blew up" in result.error


def test_run_fails_cleanly_when_response_has_no_code_block(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = DirectLLMAdapter(
        default_workspace_root=tmp_path,
        generate_code=lambda prompt, model: GeneratedCode("I refuse to write code."),
    )

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "No fenced Python code block" in result.error


def test_run_fails_cleanly_when_generated_script_errors(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = DirectLLMAdapter(
        default_workspace_root=tmp_path,
        generate_code=lambda prompt, model: GeneratedCode(
            "```python\nraise RuntimeError('bad script')\n```"
        ),
    )

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "exited with code" in result.error
    assert b"bad script" in (result.workspace / "run.log").read_bytes()
