"""Runs a prompt through MetaGPT's DataInterpreter. Executed as a subprocess
by DataInterpreterAdapter in a separate Python <3.12 virtualenv with metagpt
installed, since metagpt is incompatible with this project's own Python
version - see data_interpreter_adapter.py. No agentdatabench import here on
purpose: this script has to run standalone in that other environment.

Usage: python _data_interpreter_runner.py <workspace_path> [role_kwargs_json]
Reads <workspace>/prompt.txt (written by DataInterpreterAdapter) and lets the
agent write <workspace>/solution.csv itself, per the prompt's instructions.

Also writes <workspace>/di_metadata.json with whatever run metadata the role
exposes after finishing (plan step count, token usage from its own
CostManager, real LLM call count) - DataInterpreterAdapter reads this back
into AgentRunResult.metadata. Best-effort: metagpt's internal attributes
aren't a stable public API, so extraction failures are swallowed rather than
turning an otherwise-successful run into a failure.

`steps` (plan task count) is a much coarser number than the real LLM call
count - a single plan task commonly costs several completions (write code,
inspect data, retry on error). `llm_calls` counts actual completions by
wrapping CostManager.update_cost, which base_llm.py calls exactly once per
completion that returns usage - confirmed against a real run where the plan
had 3 tasks but the log showed 6 update_cost lines.

The wrapper patches CostManager at the *class* level, not on the specific
`di.llm.cost_manager` instance: Role._process_role_extra and Role.set_env
both do `self.llm.cost_manager = self.context.cost_manager`, and the latter
runs again during DataInterpreter's own run() - an instance-level patch
applied right after construction gets silently orphaned by that
reassignment (confirmed live: token totals came back correct, but the
instance-patched counter stayed 0). A class-level patch survives regardless
of which CostManager instance ends up attached to `di.llm`.
"""

import asyncio
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
prompt = (workspace / "prompt.txt").read_text()
role_kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}


def _extract_metadata(di, llm_calls: int) -> dict:
    metadata: dict = {"llm_calls": llm_calls}
    try:
        metadata["steps"] = len(di.planner.plan.tasks)
    except Exception:
        pass
    try:
        cost_manager = di.llm.cost_manager
        metadata["prompt_tokens"] = cost_manager.total_prompt_tokens
        metadata["completion_tokens"] = cost_manager.total_completion_tokens
        metadata["total_cost"] = cost_manager.total_cost
    except Exception:
        pass
    try:
        # di.llm.model is documented "deprecated" in metagpt's own source in
        # favor of di.llm.config.model, but every built-in provider (openai,
        # anthropic, azure, dashscope, ...) still assigns self.model =
        # self.config.model itself - config.model is the authoritative one.
        metadata["model"] = di.llm.config.model
    except Exception:
        pass
    return metadata


async def main() -> None:
    from metagpt.roles.di.data_interpreter import DataInterpreter
    from metagpt.utils.cost_manager import CostManager

    di = DataInterpreter(**role_kwargs)

    llm_call_count = 0
    original_update_cost = CostManager.update_cost

    def _counting_update_cost(self, *args, **kwargs):
        nonlocal llm_call_count
        llm_call_count += 1
        return original_update_cost(self, *args, **kwargs)

    CostManager.update_cost = _counting_update_cost
    try:
        await di.run(prompt)
    finally:
        CostManager.update_cost = original_update_cost

    metadata = _extract_metadata(di, llm_call_count)
    (workspace / "di_metadata.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    asyncio.run(main())
