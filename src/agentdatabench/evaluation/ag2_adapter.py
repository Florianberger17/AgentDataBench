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

`_run_chat` returns `initiate_chat`'s ChatResult (previously discarded) so
`_invoke` can pull step count (`len(chat_history)`) and token usage
(`.cost["usage_excluding_cached_inference"]`) into AgentRunResult.metadata -
see _extract_metadata.

`chat_history` is `user_proxy`'s own message log (`initiate_chat` is called
on it), and ConversableAgent._append_oai_message tags a message "assistant"
in *whoever's* log it's appended to whenever *that* agent sent it - so in
user_proxy's own history, user_proxy's own messages (relaying code-execution
results, not LLM calls) are the ones labelled role="assistant", and the
genuine LLM completions from `assistant` show up as role="user". Counting
real LLM calls (`llm_calls` in _extract_metadata) therefore filters by the
`name` field instead (which _append_oai_message always sets to the actual
sender's name, unaffected by this role-perspective inversion) - confirmed
against conversable_agent.py's source, not assumed.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
from pathlib import Path
from typing import Any, Callable

from agentdatabench.evaluation.agent_adapter import AgentAdapter

ChatRunner = Callable[[str, Path, list[Path], dict, dict], Any]

# The assistant's fixed name, used both to construct it in _run_chat and to
# identify its messages (vs. user_proxy's own) in _extract_metadata.
_ASSISTANT_NAME = "assistant"


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
) -> Any:
    from autogen import AssistantAgent, UserProxyAgent

    assistant = AssistantAgent(name=_ASSISTANT_NAME, llm_config=llm_config)
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
        return user_proxy.initiate_chat(
            assistant, message=user_proxy.message_generator, problem=prompt
        )
    else:
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=code_execution_config,
        )
        return user_proxy.initiate_chat(assistant, message=prompt)


def _extract_metadata(chat_result: Any) -> dict:
    """Best-effort extraction from AG2's ChatResult - a real object with a
    documented shape (see autogen.agentchat.utils.gather_usage_summary), but
    still an "(Experimental)"-flagged one per its own docstring, so failures
    here are swallowed rather than turning an otherwise-successful run into
    a failure."""
    metadata: dict = {}
    if chat_result is None:
        return metadata
    try:
        metadata["steps"] = len(chat_result.chat_history)
    except Exception:
        pass
    try:
        metadata["llm_calls"] = sum(
            1 for msg in chat_result.chat_history if msg.get("name") == _ASSISTANT_NAME
        )
    except Exception:
        pass
    try:
        usage = chat_result.cost["usage_excluding_cached_inference"]
        metadata["total_cost"] = usage.get("total_cost", 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            metadata[key] = sum(
                model_usage.get(key, 0)
                for model, model_usage in usage.items()
                if model != "total_cost"
            )
    except Exception:
        pass
    return metadata


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

    async def _invoke(self, prompt: str, workspace: Path) -> dict | None:
        docs_paths = sorted(p for p in workspace.iterdir() if p.suffix.lower() == ".pdf")

        log_buffer = io.StringIO()
        chat_result = None

        def run_and_capture() -> None:
            nonlocal chat_result
            with contextlib.redirect_stdout(log_buffer):
                chat_result = self._run_chat(
                    prompt, workspace, docs_paths, self._llm_config, self._retrieve_config_extra
                )

        try:
            await asyncio.to_thread(run_and_capture)
        finally:
            (workspace / "run.log").write_text(log_buffer.getvalue())

        return _extract_metadata(chat_result)
