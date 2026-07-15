"""DataInterpreterAdapter: wraps MetaGPT's DataInterpreter role behind the
AgentAdapter interface (see agent_adapter.py for the contract).

`metagpt` only supports Python <3.12 as of this writing, incompatible with
this project's own >=3.12 requirement, so it cannot be installed into this
project's virtualenv at all. `_invoke` therefore runs the agent in a
*separate* virtualenv (by default `venv-di/`, Python <3.12, metagpt
installed - see scripts/di_smoke_test/README.md for how to set that one up)
as a subprocess running `_data_interpreter_runner.py`, rather than importing
metagpt directly in this process. This is the same two-venv split the manual
scripts/di_smoke_test/ scripts already used by hand; the adapter now does it
itself so it works through the normal AgentAdapter.run()/EvaluationRunner
path instead of needing three separate manual script invocations.

`subprocess_launcher` is injectable (defaults to
asyncio.create_subprocess_exec) so tests can fake the subprocess without a
real venv-di + metagpt installation.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agentdatabench.evaluation.agent_adapter import AgentAdapter

_RUNNER_SCRIPT = Path(__file__).parent / "_data_interpreter_runner.py"


class _Process(Protocol):
    returncode: int | None

    async def communicate(self) -> tuple[bytes, bytes]: ...
    def kill(self) -> None: ...
    async def wait(self) -> int: ...


SubprocessLauncher = Callable[..., Awaitable[_Process]]


class DataInterpreterAdapter(AgentAdapter):
    def __init__(
        self,
        name: str = "data-interpreter",
        venv_python: Path | None = None,
        metagpt_project_root: Path | None = None,
        default_workspace_root: Path | None = None,
        subprocess_launcher: SubprocessLauncher | None = None,
        **role_kwargs: Any,
    ) -> None:
        # Runs land in metagpt_project_root/manual_runs by default (the same
        # place scripts/di_smoke_test/ has always used) rather than the
        # system temp directory AgentAdapter otherwise defaults to, so real
        # runs stay easy to find and don't get cleaned up by the OS. Pass
        # workspace_root=... on an individual run() call to override this
        # per-call, or default_workspace_root=... here to change it for
        # every run from this adapter instance.
        metagpt_project_root = (metagpt_project_root or Path("venv-di/metagpt-runtime")).resolve()
        super().__init__(
            name,
            default_workspace_root=default_workspace_root
            or (metagpt_project_root / "manual_runs"),
        )
        # .absolute(), never .resolve(): venv-di/bin/python is a symlink to a
        # shared base interpreter (e.g. installed by uv). Resolving it away
        # makes Python invoke the base interpreter directly, which then
        # fails to detect it's running inside the venv and loses access to
        # its site-packages (where metagpt is installed) entirely.
        self._venv_python = (venv_python or Path("venv-di/bin/python")).absolute()
        self._metagpt_project_root = metagpt_project_root
        self._launch_subprocess = subprocess_launcher or asyncio.create_subprocess_exec
        self._role_kwargs = role_kwargs

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        (workspace / "prompt.txt").write_text(prompt)

        env = os.environ.copy()
        env["METAGPT_PROJECT_ROOT"] = str(self._metagpt_project_root)

        process = await self._launch_subprocess(
            str(self._venv_python),
            str(_RUNNER_SCRIPT),
            str(workspace),
            json.dumps(self._role_kwargs),
            cwd=str(workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await process.communicate()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        (workspace / "run.log").write_bytes(stdout)

        if process.returncode != 0:
            raise RuntimeError(
                f"Data Interpreter subprocess exited with code {process.returncode} "
                f"- see {workspace / 'run.log'}"
            )
