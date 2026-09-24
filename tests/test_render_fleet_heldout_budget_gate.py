"""The heldout budget gate binds only a clean exact checkout."""

from __future__ import annotations

import gzip
import json
import subprocess

import pytest

from scripts import render_fleet_heldout_budget_gate as gate

COMMIT = "a" * 40


@pytest.mark.parametrize("dirty", [" M tracked.py\n", "M  tracked.py\n"])
def test_gate_rejects_dirty_checkout(monkeypatch: pytest.MonkeyPatch, dirty: str) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = f"{COMMIT}\n" if command[1:3] == ["rev-parse", "HEAD"] else dirty
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", run)

    with pytest.raises(ValueError, match="not clean"):
        gate._assert_commit(COMMIT)  # noqa: SLF001
    assert calls == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    ]


def test_gate_rejects_noncanonical_commit() -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        gate._assert_commit("HEAD")  # noqa: SLF001


def test_gate_runtime_stages_canonical_run_script() -> None:
    files = json.loads(gzip.decompress(gate._runtime({})))  # noqa: SLF001
    relative = str(gate.heldout_launch.CANONICAL_RUN_SCRIPT.relative_to(gate.ROOT))

    assert files[relative] == gate.heldout_launch.CANONICAL_RUN_SCRIPT.read_text(encoding="utf-8")
