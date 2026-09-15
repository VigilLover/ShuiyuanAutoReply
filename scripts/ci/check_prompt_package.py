"""Exercise prompt migration from the built wheel, outside the source checkout."""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check_wheel(wheel: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        env = dict(os.environ, PYTHONPATH=directory)
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
from pathlib import Path
import shuiyuan_auto_reply
from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts.profiles import (
    normalize_profile, render_profile,
)
from shuiyuan_auto_reply.infrastructure.prompts.file_repository import FilePromptRepository

assert Path(shuiyuan_auto_reply.__file__).resolve().is_relative_to(Path.cwd())
repo = FilePromptRepository()
for scope in PromptScope:
    custom = normalize_profile({'system_prompt': 'custom legacy prompt'}, scope.value)
    assert custom['prompt_mode'] == 'legacy'
    assert render_profile(custom, scope.value) == 'custom legacy prompt'
    for persona in repo._personas:
        for capabilities in (set(), {'multimodal'}):
            prompt = repo.load(persona, capabilities, scope).system_prompt
            migrated = normalize_profile({'system_prompt': prompt}, scope.value)
            assert migrated['prompt_mode'] == 'managed'
            assert render_profile(migrated, scope.value)
print('Packaged prompt migration and rendering passed')
""",
            ],
            cwd=directory,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_prompt_package.py path/to/package.whl")
    check_wheel(Path(sys.argv[1]).resolve())
