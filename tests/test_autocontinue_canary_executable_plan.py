from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from evals.fleet import autocontinue_canary_controller as canary

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "evals/fleet/scripts/run_opencode_autocontinue_canary.sh"
Q_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"


def run_renderer_gate(plan: dict, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    test_root = tmp_path / "workspace"
    config_dir = test_root / "evals/fleet/configs"
    config_dir.mkdir(parents=True)
    (config_dir / "plan.json").write_text(json.dumps(plan))
    environment = dict(os.environ)
    environment.update(
        {
            "CYBER_ROOT": str(test_root),
            "HOSTED_PLAN_FILE": "plan.json",
            "CANARY_RENDERER_GATE_ONLY": "1",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def corrected(plan: dict) -> dict:
    value = copy.deepcopy(plan)
    value["harness"]["compaction_headroom_tokens"] = 20000
    return value


@pytest.mark.parametrize("path", [Q_PLAN, G_PLAN])
def test_historical_v2_plan_parses_but_executable_renderer_fails(
    path: Path, tmp_path: Path
) -> None:
    plan = canary.load_object(path)
    canary.validate_plan(plan)
    result = run_renderer_gate(plan, tmp_path)
    assert result.returncode != 0
    assert "compaction headroom is absent or invalid" in result.stderr
    assert "HOSTED_JOB_NAME is required" not in result.stderr


@pytest.mark.parametrize("path", [Q_PLAN, G_PLAN])
def test_corrected_v2_plan_copy_passes_executable_renderer(path: Path, tmp_path: Path) -> None:
    result = run_renderer_gate(corrected(canary.load_object(path)), tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("headroom", [None, 0, -1, True, "20000", 10000])
def test_executable_renderer_rejects_invalid_headroom(
    headroom: object, tmp_path: Path
) -> None:
    plan = corrected(canary.load_object(Q_PLAN))
    plan["harness"]["compaction_headroom_tokens"] = headroom
    result = run_renderer_gate(plan, tmp_path)
    assert result.returncode != 0
    assert "compaction headroom is absent or invalid" in result.stderr


@pytest.mark.parametrize("field", ["settings_canonical_sha256", "settings_file_sha256"])
def test_executable_renderer_rejects_mismatched_settings_hash(
    field: str, tmp_path: Path
) -> None:
    plan = corrected(canary.load_object(Q_PLAN))
    plan["harness"][field] = "sha256:" + "0" * 64
    result = run_renderer_gate(plan, tmp_path)
    assert result.returncode != 0
    assert "renderer settings digest drifted" in result.stderr


def test_executable_renderer_gate_precedes_all_canary_side_effects() -> None:
    source = SCRIPT.read_text()
    gate = source.index('validate_executable_renderer "$PLAN_PATH"')
    for later in (
        "JOB_NAME=${HOSTED_JOB_NAME",
        "docker info",
        "docker build",
        "docker pull",
        "run-hosted",
    ):
        assert gate < source.index(later)
