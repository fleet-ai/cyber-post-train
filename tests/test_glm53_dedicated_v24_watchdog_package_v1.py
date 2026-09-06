import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as server
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict:
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": "ft-run-v24exact",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": "http://ft-run-v24exact-head-svc.fleet-train-jobs.svc:8000",
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def _priority_classes() -> list[dict]:
    return [
        {
            "metadata": {"name": "fleet-infra-quiet"},
            "value": -1000,
            "preemptionPolicy": "Never",
        },
        {
            "metadata": {"name": "fleet-serve-low"},
            "value": 100,
            "preemptionPolicy": "Never",
        },
    ]


def test_v24_binding_is_exact_and_rejects_v23_identity() -> None:
    server.validate_binding(_binding())
    wrong = dict(_binding())
    wrong["server_title"] = "chris-cyber-evalserve-glm53-tp8-a-v23"
    with pytest.raises(server.ServerPlanError, match="binding"):
        server.validate_binding(wrong)


def test_v24_watcher_package_is_uid_bound_create_once_and_score_free() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    rendered = package.render(
        ROOT,
        commit,
        _binding(),
        ready_at_epoch=123.0,
        priority_classes=_priority_classes(),
    )
    configmap, job = rendered["objects"]["items"]
    manifest = json.loads(configmap["data"]["package.json"])
    assert configmap["metadata"]["name"] == package.JOB_NAME + "-package"
    assert configmap["immutable"] is True
    assert manifest["job_name"] == package.JOB_NAME
    assert manifest["result_root"] == package.RESULT_ROOT
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row for row in container["env"]}
    assert env["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['batch.kubernetes.io/controller-uid']"
    )
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    assert rendered["server_binding"] == _binding()
    assert rendered["watchdog_launch_authorized"] is False
    assert rendered["server_launch_authorized"] is False
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


def test_tracked_watcher_package_receipt_is_self_digested_and_held() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v24-watchdog-package-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "READY_HELD_FOR_EXACT_LIVE_BINDING"
    assert value["watchdog_launch_authorized"] is False
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["rendered_sentinel"]["uses_non_live_sentinel_uids"] is True
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
