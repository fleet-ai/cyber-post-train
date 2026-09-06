import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v32_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v32_create_v1 as server
from evals.fleet import glm53_dedicated_v32_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v32_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v32_watchdog_package_v1 as watchdog

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f370770520a3d872de6e04929ceb862f9504e382"


def create_authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": server.SCHEMA,
        "status": "PASSED_LIVE_CREATE_GATES",
        "observed_at_epoch": time.time(),
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "request_sha256": server.request_sha256(),
        "preview_http_status": 200,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_or_remnant_matches": 0,
        "sfs_run_dir_absent": True,
        "control_result_absent": True,
        "active_dedicated_nodes": 0,
        "active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "coexisting_qwen_server": None,
        "priority_class": v24.PRIORITY_CLASS,
        "preemption_policy": v24.PREEMPTION_POLICY,
        "server_launch_authorized": True,
        "watchdog_handoff_required_immediately": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def binding() -> dict[str, object]:
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": "ft-run-1234abcd",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": (
            "http://ft-run-1234abcd-abcde-head-svc.fleet-train-jobs.svc:8000"
        ),
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def parity_authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": parity.SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_INCLUSTER_PARITY",
        "observed_at_epoch": time.time(),
        "server_binding": binding(),
        "server_ready_receipt_sha256": "sha256:" + "1" * 64,
        "watchdog_active_receipt_sha256": "sha256:" + "2" * 64,
        "job_identity_absent": True,
        "result_root_absent": True,
        "cpu_priority_class": parity.CPU_PRIORITY_CLASS,
        "cpu_preemption_policy": "Never",
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v32_create_gate_requires_fresh_zero_state() -> None:
    value = create_authorization()
    server.validate_authorization(value)
    changed = copy.deepcopy(value)
    changed.update(
        {
            "active_dedicated_nodes": 1,
            "active_dedicated_gpus": 6,
            "planned_nodes_after_create": 2,
            "planned_gpus_after_create": 14,
        }
    )
    changed["receipt_sha256"] = crypto.digest_without(changed, "receipt_sha256")
    changed["coexisting_qwen_server"] = server.EXACT_QWEN_COEXISTENCE
    changed["receipt_sha256"] = crypto.digest_without(changed, "receipt_sha256")
    server.validate_authorization(changed)
    drifted = copy.deepcopy(changed)
    drifted["coexisting_qwen_server"]["head_pod_restarts"] = 1
    drifted["receipt_sha256"] = crypto.digest_without(drifted, "receipt_sha256")
    with pytest.raises(server.CreateError, match="authorization_invalid"):
        server.validate_authorization(drifted)
    changed["coexisting_qwen_server"] = None
    changed["receipt_sha256"] = crypto.digest_without(changed, "receipt_sha256")
    with pytest.raises(server.CreateError, match="authorization_invalid"):
        server.validate_authorization(changed)
    assert server.TITLE.endswith("-v32")
    assert server.RUN_DIR.endswith("-v32")
    assert server.payload()["gpus_per_worker"] == 8
    assert server.payload()["priority_class"] == "fleet-infra-quiet"


def test_v32_parity_renderer_rejects_non_v30_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = {
        "objects": {
            "items": [
                {
                    "data": {
                        "run.sh": (
                            "python -m evals.fleet."
                            "glm53_dedicated_v31_incluster_parity_v1 run"
                        ),
                        "package.json": json.dumps({}),
                    }
                }
            ]
        }
    }
    monkeypatch.setattr(parity.engine, "render", lambda *_args: rendered)
    with pytest.raises(
        parity.ParityPackageError, match="entrypoint_template_invalid"
    ):
        parity.render(ROOT, COMMIT, parity_authorization())


def test_v32_materialized_parity_entrypoint_and_schema_are_generation_exact(
    tmp_path: Path,
) -> None:
    rendered = parity.render(ROOT, COMMIT, parity_authorization())
    source, auth, job = rendered["objects"]["items"]
    run = source["data"]["run.sh"]
    manifest = json.loads(source["data"]["package.json"])
    mounted_auth = json.loads(auth["data"]["authorization.json"])
    assert (
        "python -m evals.fleet.glm53_dedicated_v32_incluster_parity_v1 run"
        in run
    )
    assert "glm53_dedicated_v30_incluster_parity_v1 run" not in run
    assert "glm53_dedicated_v31_incluster_parity_v1 run" not in run
    assert manifest["run_sha256"] == crypto.sha256(run.encode())
    assert manifest["schema_version"] == parity.PACKAGE_SCHEMA
    assert manifest["server_title"] == server.TITLE
    assert manifest["server_run_dir"] == server.RUN_DIR
    assert mounted_auth["schema_version"] == parity.SCHEMA
    assert mounted_auth["receipt_sha256"] == crypto.digest_without(
        mounted_auth, "receipt_sha256"
    )
    assert job["metadata"]["name"] == parity.JOB_NAME
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    for relative in parity.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(auth["data"]["authorization.json"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.fleet.glm53_dedicated_v32_incluster_parity_v1",
            "plan",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    held = json.loads(result.stdout)
    assert held["entrypoint_module"].endswith("v32_incluster_parity_v1")
    assert held["authorization_schema"] == parity.SCHEMA
    assert held["scored_launch_authorized"] is False


def test_v32_controller_package_has_only_fresh_executable_identities(
    tmp_path: Path,
) -> None:
    rendered = package.render(ROOT, COMMIT, create_authorization())
    source, auth, job = rendered["objects"]["items"]
    manifest = json.loads(source["data"]["package.json"])
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "glm53_dedicated_v32_controller_v1" in command
    assert "glm53_dedicated_v30_controller_v1" not in command
    assert "glm53_dedicated_v31_controller_v1" not in command
    assert job["metadata"]["name"] == package.JOB_NAME
    assert manifest["schema_version"] == package.PACKAGE_SCHEMA
    assert manifest["server_title"] == server.TITLE
    assert manifest["server_run_dir"] == server.RUN_DIR
    assert auth["immutable"] is True
    assert len(json.dumps(source)) < 1_000_000
    for relative in package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v32_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v32_held_contract_is_score_free_and_idle_bounded() -> None:
    create = server.build_held()
    live = adapter.build_held(COMMIT)
    score_free = parity.build_held()
    assert watchdog.JOB_NAME.endswith("v32-request-watchdog-v1")
    assert live["idle_release_seconds"] == 600
    assert live["exactly_one_live_observation"] is True
    assert score_free["entrypoint_module"].endswith("v32_incluster_parity_v1")
    for receipt in (create, live, score_free):
        if "server_launch_authorized" in receipt:
            assert receipt["server_launch_authorized"] is False
        assert receipt["scored_launch_authorized"] is False
        assert receipt["protected_content_included"] is False
        assert receipt["receipt_sha256"] == crypto.digest_without(
            receipt, "receipt_sha256"
        )


def test_v32_held_and_v31_terminal_receipts_are_digest_valid() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v32-zero-state-lifecycle-held-v1.json"
        ).read_text()
    )
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v31-parity-package-failure-release-v1.json"
        ).read_text()
    )
    assert held == package.build_held(ROOT, COMMIT)
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")
    assert terminal["receipt_sha256"] == crypto.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["status"] == "RELEASED_ZERO_GPU_REMNANTS"
    assert terminal["retry_same_identity"] is False
    assert terminal["parity_result_present"] is False
    assert terminal["fleet_task_instance_calls"] == 0
    assert terminal["fleet_session_calls"] == 0
    assert terminal["verifier_calls"] == 0
    assert terminal["scoring_calls"] == 0
