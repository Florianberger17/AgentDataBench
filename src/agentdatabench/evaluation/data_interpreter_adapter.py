"""DataInterpreterAdapter: wraps MetaGPT's DataInterpreter role behind the
AgentAdapter interface (see agent_adapter.py for the contract).

`metagpt` is an optional dependency (see pyproject.toml's `data-interpreter`
extra) and, as of this writing, only supports Python <3.12 - incompatible
with this project's >=3.12 requirement. Importing this module therefore
never imports metagpt; only actually running a DataInterpreterAdapter built
with the default role factory does. Tests inject a fake role factory instead,
so the adapter's own logic (prompt wiring, invocation) is fully covered
without metagpt installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from agentdatabench.evaluation.agent_adapter import AgentAdapter


class _InterpreterRole(Protocol):
    async def run(self, requirement: str) -> Any: ...


def _default_role_factory(**role_kwargs: Any) -> _InterpreterRole:
    from metagpt.roles.di.data_interpreter import DataInterpreter

    return DataInterpreter(**role_kwargs)


class DataInterpreterAdapter(AgentAdapter):
    def __init__(
        self,
        name: str = "data-interpreter",
        role_factory: Callable[..., _InterpreterRole] | None = None,
        **role_kwargs: Any,
    ) -> None:
        super().__init__(name)
        self._role_factory = role_factory or _default_role_factory
        self._role_kwargs = role_kwargs

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        role = self._role_factory(**self._role_kwargs)
        await role.run(prompt)
