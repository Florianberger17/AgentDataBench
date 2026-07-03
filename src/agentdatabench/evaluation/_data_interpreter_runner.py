"""Runs a prompt through MetaGPT's DataInterpreter. Executed as a subprocess
by DataInterpreterAdapter in a separate Python <3.12 virtualenv with metagpt
installed, since metagpt is incompatible with this project's own Python
version - see data_interpreter_adapter.py. No agentdatabench import here on
purpose: this script has to run standalone in that other environment.

Usage: python _data_interpreter_runner.py <workspace_path> [role_kwargs_json]
Reads <workspace>/prompt.txt (written by DataInterpreterAdapter) and lets the
agent write <workspace>/solution.csv itself, per the prompt's instructions.
"""

import asyncio
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
prompt = (workspace / "prompt.txt").read_text()
role_kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}


async def main() -> None:
    from metagpt.roles.di.data_interpreter import DataInterpreter

    di = DataInterpreter(**role_kwargs)
    await di.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
