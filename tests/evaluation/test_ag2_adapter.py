"""Tests for AG2Adapter. Injects a fake run_chat so these run without a real
LLM call / ag2 conversation while still covering the adapter's own wiring:
prompt/workspace handoff, PDF discovery, log capture, and the
RetrieveUserProxyAgent-vs-plain-UserProxyAgent branch (driven by whether any
PDFs are present in the workspace).
"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.ag2_adapter import AG2Adapter
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME


def test_default_workspace_root_is_manual_runs(tmp_path):
    adapter = AG2Adapter(default_workspace_root=tmp_path / "ag2_runtime" / "manual_runs")

    assert adapter._default_workspace_root == tmp_path / "ag2_runtime" / "manual_runs"


def test_invoke_calls_run_chat_with_prompt_and_writes_output(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    calls = []

    def fake_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        calls.append(
            {
                "prompt": prompt,
                "workspace": workspace,
                "docs_paths": docs_paths,
                "llm_config": llm_config,
            }
        )
        print("fake chat ran")
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)

    adapter = AG2Adapter(
        default_workspace_root=tmp_path,
        run_chat=fake_run_chat,
        llm_config={"config_list": [{"model": "fake-model", "api_key": "fake"}]},
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert len(calls) == 1
    assert package.task.objective.strip() in calls[0]["prompt"]
    assert calls[0]["docs_paths"] == []
    assert calls[0]["llm_config"]["config_list"][0]["model"] == "fake-model"
    assert (result.workspace / "run.log").read_text() == "fake chat ran\n"


class _FakeChatResult:
    """Duck-typed stand-in for autogen's real ChatResult - _extract_metadata
    only reads .chat_history and .cost, so a real ag2 install isn't needed
    to exercise that extraction logic."""

    def __init__(self, chat_history, cost):
        self.chat_history = chat_history
        self.cost = cost


def test_invoke_extracts_steps_and_token_usage_from_chat_result(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    def fake_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)
        return _FakeChatResult(
            # Mirrors user_proxy's real chat_messages shape: its own
            # messages (the initial prompt, relaying execution results) are
            # tagged role="assistant" from its own bookkeeping perspective
            # (see the module docstring) - only "name" reliably identifies
            # who actually sent each message. Two of these three came from
            # "assistant" - a real LLM call each.
            chat_history=[
                {"content": "here is the task", "role": "assistant", "name": "user_proxy"},
                {"content": "```python\n...\n```", "role": "user", "name": "assistant"},
                {"content": "execution result: ...", "role": "assistant", "name": "user_proxy"},
                {"content": "```python\n...\n```", "role": "user", "name": "assistant"},
            ],
            cost={
                "usage_excluding_cached_inference": {
                    "total_cost": 0.0012,
                    "gpt-4o-mini": {
                        "cost": 0.0012,
                        "prompt_tokens": 500,
                        "completion_tokens": 80,
                        "total_tokens": 580,
                    },
                },
            },
        )

    adapter = AG2Adapter(
        default_workspace_root=tmp_path,
        run_chat=fake_run_chat,
        llm_config={"config_list": [{"model": "test-model", "api_key": "fake"}]},
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert result.metadata == {
        "steps": 4,
        "llm_calls": 2,
        "total_cost": 0.0012,
        "prompt_tokens": 500,
        "completion_tokens": 80,
        "total_tokens": 580,
        "model": "test-model",
    }


def test_invoke_metadata_has_only_model_when_run_chat_returns_none(pkg1_root, tmp_path):
    # The default plain-UserProxyAgent/AssistantAgent branch used to return
    # None implicitly before _run_chat started returning initiate_chat's
    # result - a custom injected run_chat may still do this.
    package = BenchmarkPackage.load(pkg1_root)

    def fake_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)
        return None

    adapter = AG2Adapter(
        default_workspace_root=tmp_path,
        run_chat=fake_run_chat,
        llm_config={"config_list": [{"model": "test-model", "api_key": "fake"}]},
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert result.metadata == {"model": "test-model"}


def test_invoke_passes_pdf_paths_from_workspace(pkg1_root, tmp_path):
    # Package 001 itself has no additional_documents, so this exercises the
    # PDF-discovery path directly by dropping a fake PDF into the workspace
    # via a wrapping run_chat, then re-invoking against a package that does
    # reference it - simplest is to build a package with additional_documents
    # like test_agent_adapter.py's helper does.
    import shutil

    import yaml

    package_root = tmp_path / "pkg"
    shutil.copytree(pkg1_root, package_root)
    (package_root / "order.pdf").write_bytes(b"%PDF-1.4 fake")
    task_path = package_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    task_data["input"]["additional_documents"] = ["order.pdf"]
    task_path.write_text(yaml.safe_dump(task_data))
    package = BenchmarkPackage.load(package_root)

    calls = []

    def fake_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        calls.append(docs_paths)
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)

    adapter = AG2Adapter(default_workspace_root=tmp_path / "runs", run_chat=fake_run_chat)

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert len(calls) == 1
    assert [p.name for p in calls[0]] == ["order.pdf"]


def test_run_fails_cleanly_when_run_chat_raises(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    def failing_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        raise RuntimeError("chat blew up")

    adapter = AG2Adapter(default_workspace_root=tmp_path, run_chat=failing_run_chat)

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "chat blew up" in result.error


def test_run_fails_when_agent_never_writes_output(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    def noop_run_chat(prompt, workspace, docs_paths, llm_config, retrieve_config_extra):
        pass

    adapter = AG2Adapter(default_workspace_root=tmp_path, run_chat=noop_run_chat)

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "did not produce" in result.error


def test_default_run_chat_requires_ag2_installed_or_fails_cleanly(pkg1_root, tmp_path, monkeypatch):
    # Doesn't assert ag2 is absent (it's installed in this project's own
    # venv, unlike metagpt) - just confirms that if the real _run_chat blows
    # up for any reason (e.g. no OPENAI_API_KEY), AgentAdapter.run() still
    # returns a clean failed result instead of raising.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    package = BenchmarkPackage.load(pkg1_root)
    adapter = AG2Adapter(default_workspace_root=tmp_path)

    result = asyncio.run(adapter.run(package, timeout=5))

    assert not result.success
    assert result.error is not None
