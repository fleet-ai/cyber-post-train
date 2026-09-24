"""The heldout budget gate binds only a clean exact checkout."""

from __future__ import annotations

import subprocess

import pytest

from scripts import render_fleet_heldout_budget_gate as gate

COMMIT = "a" * 40


def test_gate_rejects_dirty_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stdout = f"{COMMIT}\n" if command[1:3] == ["rev-parse", "HEAD"] else " M tracked.py\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", run)

    with pytest.raises(ValueError, match="not clean"):
        gate._assert_commit(COMMIT)  # noqa: SLF001


def test_gate_rejects_noncanonical_commit() -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        gate._assert_commit("HEAD")  # noqa: SLF001
