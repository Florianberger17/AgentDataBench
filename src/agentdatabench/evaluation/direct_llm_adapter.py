"""DirectLLMAdapter: a baseline that skips agent scaffolding entirely. One
LLM call generates a self-contained Python/pandas script (no planning loop,
no tool use, no execution-observe-retry cycle); we execute that script
ourselves. Isolates how much of an agent framework's performance comes from
the underlying model's raw capability vs. the framework's orchestration
around it - Data Interpreter and AG2 both fundamentally "write and run
pandas code" too, just wrapped in a plan/execute/observe loop this adapter
deliberately omits.

Unlike DataInterpreterAdapter/AG2Adapter, there is no code-execution loop
for the model to inspect files itself here, so `_build_prompt` inlines the
actual dataset/schema/document content into the message (the base class's
file-path-only prompt assumes the wrapped agent can open files on its own -
not true for a single text-completion call).

`generate_code` and `extract_pdf_text` are injectable (default to a real
OpenAI call / pypdf extraction) so tests can run without hitting a real API
or needing an actually-parseable PDF fixture. Code execution itself is not
mocked in tests - it's just `sys.executable script.py` in this project's own
venv, no special environment needed (unlike metagpt/ag2's heavier deps).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Callable

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME, AgentAdapter

CodeGenerator = Callable[[str, str], str]
PdfTextExtractor = Callable[[Path], str]

_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(response_text: str) -> str:
    match = _CODE_BLOCK_PATTERN.search(response_text)
    if not match:
        raise ValueError("No fenced Python code block found in the model's response")
    return match.group(1).strip()


def _default_generate_code(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data engineer. Respond with exactly one fenced "
                    "Python code block and nothing else - no explanation before "
                    "or after it."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _default_extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class DirectLLMAdapter(AgentAdapter):
    def __init__(
        self,
        name: str = "direct-llm",
        model: str | None = None,
        generate_code: CodeGenerator | None = None,
        extract_pdf_text: PdfTextExtractor | None = None,
        default_workspace_root: Path | None = None,
    ) -> None:
        super().__init__(
            name,
            default_workspace_root=(
                default_workspace_root or Path("direct_llm_runtime/manual_runs")
            ).resolve(),
        )
        self._model = model or os.environ.get("DIRECT_LLM_MODEL", "gpt-4o-mini")
        self._generate_code = generate_code or _default_generate_code
        self._extract_pdf_text = extract_pdf_text or _default_extract_pdf_text

    def _build_prompt(self, package: BenchmarkPackage, workspace: Path) -> str:
        base_prompt = super()._build_prompt(package, workspace)

        lines = [
            base_prompt,
            "",
            "--- Input dataset content (dataset.csv) ---",
            (workspace / "dataset.csv").read_text(),
            "",
            "--- Source schema content ---",
            (workspace / "source_schema.yaml").read_text(),
            "",
            "--- Target schema content ---",
            (workspace / "target_schema.yaml").read_text(),
        ]

        for document in package.task.input.additional_documents or []:
            doc_path = workspace / Path(document).name
            if doc_path.suffix.lower() == ".pdf":
                lines += [
                    "",
                    f"--- Extracted text content of {doc_path.name} ---",
                    self._extract_pdf_text(doc_path),
                ]

        lines += [
            "",
            "Respond with exactly one fenced Python code block implementing "
            "this transformation and nothing else (no explanation before or "
            "after). The script will be executed as-is, with no prior state: "
            f"read the input dataset from {workspace / 'dataset.csv'} and "
            f"write the result to exactly {workspace / OUTPUT_FILENAME}.",
        ]
        return "\n".join(lines)

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        (workspace / "prompt.txt").write_text(prompt)
        response_text = await asyncio.to_thread(self._generate_code, prompt, self._model)
        (workspace / "response.txt").write_text(response_text)

        code = _extract_code(response_text)
        script_path = workspace / "solution.py"
        script_path.write_text(code)

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        (workspace / "run.log").write_bytes(stdout)

        if process.returncode != 0:
            raise RuntimeError(
                f"Generated script exited with code {process.returncode} "
                f"- see {workspace / 'run.log'}"
            )
