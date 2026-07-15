"""AG2Adapter: wraps AG2 (the community continuation of AutoGen 0.2, see
pyproject.toml's `ag2` extra for the version-pinning rationale) behind the
AgentAdapter interface.

Unlike DataInterpreterAdapter, `ag2[retrievechat]` installs cleanly into this
project's own >=3.12 environment - no separate venv/subprocess needed.
`_invoke` therefore runs the conversation in-process, in a worker thread
(`asyncio.to_thread`) since AG2's `initiate_chat` is a blocking call, not a
coroutine.

Uses `RetrieveUserProxyAgent` when the task has additional_documents (PDFs
etc. - it both retrieves relevant chunks via a Chroma vector index *and*
executes code, being a UserProxyAgent subclass), falling back to a plain
UserProxyAgent + AssistantAgent conversation when there are none, since
RetrieveUserProxyAgent expects a non-empty document corpus.

Known limitation: because the chat runs in a worker thread rather than a
subprocess, a timeout-triggered cancellation (see AgentAdapter.run) cannot
forcibly stop it the way DataInterpreterAdapter kills its subprocess - the
thread keeps running in the background even after run() has already
returned a timed-out AgentRunResult. Acceptable for now given this project
runs one evaluation at a time; would need revisiting for concurrent runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
from pathlib import Path
from typing import Any, Callable

from agentdatabench.evaluation.agent_adapter import AgentAdapter

ChatRunner = Callable[[str, Path, list[Path], dict, dict], None]


def _default_llm_config() -> dict:
    return {
        "config_list": [
            {
                "model": os.environ.get("AG2_MODEL", "gpt-4o-mini"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        ]
    }


def _run_chat(
    prompt: str,
    workspace: Path,
    docs_paths: list[Path],
    llm_config: dict,
    retrieve_config_extra: dict,
) -> None:
    from autogen import AssistantAgent, UserProxyAgent

    assistant = AssistantAgent(name="assistant", llm_config=llm_config)
    code_execution_config = {"work_dir": str(workspace), "use_docker": False}

    if docs_paths:
        from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

        retrieve_config = {
            "task": "code",
            "docs_path": [str(path) for path in docs_paths],
            "chunk_token_size": 2000,
            "model": llm_config["config_list"][0]["model"],
            "vector_db": "chroma",
            "overwrite": True,
            "get_or_create": True,
        }
        retrieve_config.update(retrieve_config_extra)

        user_proxy = RetrieveUserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            retrieve_config=retrieve_config,
            code_execution_config=code_execution_config,
        )
        user_proxy.initiate_chat(assistant, message=user_proxy.message_generator, problem=prompt)
    else:
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=code_execution_config,
        )
        user_proxy.initiate_chat(assistant, message=prompt)


class AG2Adapter(AgentAdapter):
    def __init__(
        self,
        name: str = "ag2",
        llm_config: dict | None = None,
        run_chat: ChatRunner | None = None,
        default_workspace_root: Path | None = None,
        **retrieve_config_extra: Any,
    ) -> None:
        super().__init__(
            name,
            default_workspace_root=(
                default_workspace_root or Path("ag2_runtime/manual_runs")
            ).resolve(),
        )
        self._llm_config = llm_config or _default_llm_config()
        self._run_chat = run_chat or _run_chat
        self._retrieve_config_extra = retrieve_config_extra

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        docs_paths = sorted(p for p in workspace.iterdir() if p.suffix.lower() == ".pdf")

        log_buffer = io.StringIO()

        def run_and_capture() -> None:
            with contextlib.redirect_stdout(log_buffer):
                self._run_chat(
                    prompt, workspace, docs_paths, self._llm_config, self._retrieve_config_extra
                )

        try:
            await asyncio.to_thread(run_and_capture)
        finally:
            (workspace / "run.log").write_text(log_buffer.getvalue())
