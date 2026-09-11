"""Run with isolated Python against a non-editable install, never a live job.

uv run --isolated --locked --no-editable python -I tests/check_installed.py
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from typer.main import get_command

from cyber_post_train import cli
from evals.exploitgym import output_lifecycle
from evals.fleet import evaluate
from training import miles_training, skyrl_training

source = Path(__file__).resolve().parents[1]
installed = Path(cli.__file__).resolve().parents[1]
assert sys.flags.isolated, "use python -I to exclude checkout/PYTHONPATH imports"
assert not installed.is_relative_to(source), "the source checkout is not an installed wheel"

checked = 0
for package in ("cyber_post_train", "training", "evals"):
    module = importlib.import_module(package)
    assert Path(module.__file__).resolve().parent == installed / package
    for path in (source / package).rglob("*.py"):
        relative = path.relative_to(source)
        payload = (installed / relative).read_bytes()
        assert payload == path.read_bytes(), f"missing or stale installed module: {relative}"
        checked += 1

# These builders must find their runtime dependencies in the wheel itself.
for backend in (miles_training, skyrl_training):
    for name, content in backend._runtime().items():
        assert content == (source / name).read_text(), f"stale runtime bundle member: {name}"
evaluate.runtime_identity()
benchmark_root = Path(output_lifecycle.__file__).parent
for relative in [
    *output_lifecycle.BOOTSTRAP_SOURCE_PATHS.values(),
    output_lifecycle.JOB_MANIFEST_RELATIVE,
]:
    assert (benchmark_root / relative).read_bytes() == (
        source / "evals/exploitgym" / relative
    ).read_bytes(), f"missing or stale historical runtime resource: {relative}"

commands = [([], get_command(cli.app))]
count = 0
with tempfile.TemporaryDirectory(prefix="cpt-installed-help-") as directory:
    env = {
        "PATH": os.defpath,
        "HOME": directory,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    while commands:
        args, command = commands.pop()
        result = subprocess.run(
            [str(Path(sys.executable).with_name("cyber-post-train")), *args, "--help"],
            cwd=directory,
            env=env,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"installed command help failed: {args}"
        count += 1
        commands.extend(
            (args + [name], child) for name, child in getattr(command, "commands", {}).items()
        )

print(
    json.dumps(
        {
            "installed_modules": checked,
            "command_help_checks": count,
            "source_tree_imported": False,
            "jobs_submitted": 0,
        }
    )
)
