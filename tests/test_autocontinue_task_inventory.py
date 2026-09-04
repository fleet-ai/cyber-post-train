from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evals/fleet/autocontinue_task_inventory.py"
SPEC = importlib.util.spec_from_file_location("autocontinue_task_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def test_prepare_exact_campaign_inventory() -> None:
    expected = inventory.prepare_expected(ROOT)
    assert expected["selection_counts"] == {"qwen3.8-27b": 50, "glm-5.3": 100, "total": 150}
    assert expected["cell_counts"] == {"qwen3.8-27b": 200, "glm-5.3": 400, "total": 600}
    assert expected["split_counts"] == {"train": 150, "dev": 0, "test": 0}
    assert expected["environment_version_id_fresh_api_observable"] is False
    assert len({(row["model"], row["task_version_id"]) for row in expected["tasks"]}) == 150
    assert expected["r114_execution_binding"] == {
        "plan_path": str(inventory.R114_PLAN_PATH),
        "plan_sha256": inventory.R114_PLAN_SHA256,
        "hydration_path": str(inventory.R114_HYDRATION_PATH),
        "hydration_receipt_sha256": inventory.R114_HYDRATION_SHA256,
        "attempts": 4,
        "launch_authorized": False,
    }
    sealed = json.loads(
        (
            ROOT / "evals/fleet/configs/q38-glm53-opencode-autocontinue-task-inventory-v1.json"
        ).read_text()
    )
    assert sealed == expected


def test_r114_plan_tamper_fails(tmp_path: Path) -> None:
    plan = inventory.load_json(ROOT / inventory.R114_PLAN_PATH)
    plan["attempts"][0]["attempt"] = 4
    path = tmp_path / inventory.R114_PLAN_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    hydration = tmp_path / inventory.R114_HYDRATION_PATH
    hydration.parent.mkdir(parents=True)
    hydration.write_bytes((ROOT / inventory.R114_HYDRATION_PATH).read_bytes())
    with pytest.raises(inventory.GateError):
        inventory._validate_r114(tmp_path)


def test_live_comparable_rejects_empty_runtime_seed() -> None:
    expected = inventory.prepare_expected(ROOT)["tasks"][0]
    with pytest.raises(inventory.GateError, match="runtime_seed_manifest_incomplete"):
        inventory._live_comparable({"metadata": {"runtime_seed_manifest": {"files": []}}}, expected)


def test_observer_manifest_is_non_scored_high_priority() -> None:
    manifest = yaml.safe_load(
        (ROOT / "evals/fleet/cluster/opencode-autocontinue-task-inventory-v1.yaml").read_text()
    )
    assert manifest["kind"] == "Job"
    assert manifest["spec"]["backoffLimit"] == 0
    pod = manifest["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-train-high"
    assert pod["restartPolicy"] == "Never"
    container = pod["containers"][0]
    assert container["env"][0]["valueFrom"]["secretKeyRef"] == {
        "name": inventory.SECRET_NAME,
        "key": "FLEET_API_KEY",
    }
    text = MODULE_PATH.read_text()
    assert 'method="GET"' in text
    assert 'method="POST"' not in text
    assert "/v1/sessions" not in text
    assert "/v1/rollout-rewards" not in text


def test_expected_digest_tamper_rejected() -> None:
    expected = inventory.prepare_expected(ROOT)
    tampered = copy.deepcopy(expected)
    tampered["split_counts"]["train"] = 149
    with pytest.raises(inventory.GateError, match="expected_inventory_digest_mismatch"):
        inventory.observe(tampered, "unused", job_uid="job", pod_uid="pod")


def test_projected_configmap_symlink_is_narrowly_accepted(tmp_path: Path) -> None:
    generation = tmp_path / "..2026_09_04"
    generation.mkdir()
    target = generation / "expected.json"
    target.write_text('{"safe":true}')
    projected = tmp_path / "expected.json"
    projected.symlink_to(Path(generation.name) / target.name)
    assert inventory.load_configmap_json(projected) == {"safe": True}
    outside = tmp_path.parent / "outside-inventory.json"
    outside.write_text('{"safe":false}')
    projected.unlink()
    projected.symlink_to(outside)
    with pytest.raises(inventory.GateError, match="expected_file_absent_or_unsafe"):
        inventory.load_configmap_json(projected)
