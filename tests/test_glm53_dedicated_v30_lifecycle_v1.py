import copy
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v30_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v30_controller_v1 as controller
from evals.fleet import glm53_dedicated_v30_create_v1 as server
from evals.fleet import glm53_dedicated_v30_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v30_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v30_watchdog_package_v1 as watchdog

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "652ce10aeec4cb613a3062a301f3236a3b51c255"


def authorization() -> dict[str, object]:
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


def watchdog_launch() -> dict[str, object]:
    return {
        "receipt_sha256": "sha256:" + "1" * 64,
        "server_binding": binding(),
        "application_ready_receipt_sha256": "sha256:" + "3" * 64,
        "runtime": {
            "watchdog_job_uid": "55555555-5555-4555-8555-555555555555",
            "watchdog_pod_uid": "66666666-6666-4666-8666-666666666666",
            "active_receipt_sha256": "sha256:" + "2" * 64,
        },
    }


def live_state() -> dict[str, object]:
    return {"application_ready_receipt_sha256": "sha256:" + "3" * 64}


def parity_result() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": parity.RESULT_SCHEMA,
        "status": "PASSED_NON_SCORED_INCLUSTER_PARITY",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding())),
        "authorization_receipt_sha256": "sha256:" + "4" * 64,
        "parity_receipt_sha256": "sha256:" + "5" * 64,
        "job_uid": "77777777-7777-4777-8777-777777777777",
        "pod_uid": "88888888-8888-4888-8888-888888888888",
        "nested_container_network": "host",
        "local_proxy_bind_address": "127.0.0.1",
        "endpoint_origin": binding()["service_origin"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v30_create_gate_requires_exact_zero_to_one_node_eight_gpus() -> None:
    value = authorization()
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
    with pytest.raises(server.CreateError, match="requires_zero_project_server"):
        server.validate_authorization(changed)
    held = server.build_held()
    assert held["required_active_dedicated_nodes"] == 0
    assert held["planned_gpus_after_create"] == 8
    assert held["priority_class"] == "fleet-infra-quiet"
    assert held["preemption_policy"] == "Never"
    assert held["server_launch_authorized"] is False


def test_parity_authorization_binds_ready_and_active_watchdog() -> None:
    value = controller.build_parity_authorization(
        binding(), live_state(), watchdog_launch(), observed_at_epoch=time.time()
    )
    parity.validate_authorization(value)
    assert value["server_binding"] == binding()
    assert value["watchdog_active_receipt_sha256"] == "sha256:" + "2" * 64
    assert value["fleet_task_instance_calls"] == 0
    assert value["scoring_calls"] == 0
    assert value["scored_launch_authorized"] is False


def test_watchdog_adapter_returns_the_exact_observed_binding_without_reobserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = watchdog_launch()["runtime"]
    engine_receipt: dict[str, object] = {
        "schema_version": adapter.LAUNCH_SCHEMA,
        "status": "WATCHDOG_CREATE_REQUEST_ACCEPTED",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding())),
        "runtime": runtime,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    engine_receipt["receipt_sha256"] = crypto.digest_without(
        engine_receipt, "receipt_sha256"
    )
    monkeypatch.setattr(
        adapter.engine,
        "observe_live",
        lambda _run: (binding(), live_state()),
    )
    monkeypatch.setattr(adapter.engine, "launch", lambda *_args, **_kwargs: engine_receipt)
    observed = adapter.launch(ROOT, COMMIT, "ft-run-1234abcd", priority_classes=[])
    assert observed["schema_version"] == adapter.ADAPTER_LAUNCH_SCHEMA
    assert observed["server_binding"] == binding()
    assert observed["application_ready_receipt_sha256"] == "sha256:" + "3" * 64
    assert observed["runtime"] == runtime
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )


def test_controller_orders_watchdog_before_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("FLEET_API_KEY", "not-serialized")
    monkeypatch.setattr(controller, "_load_authorization", lambda _path: authorization())
    monkeypatch.setattr(
        controller.server,
        "create_once",
        lambda *_args, **_kwargs: {"api_run_id": "ft-run-1234abcd"},
    )
    monkeypatch.setattr(controller, "_wait_ready", lambda: events.append("ready"))
    monkeypatch.setattr(
        controller.live_engine,
        "_kubectl_json",
        lambda *_args: {"items": []},
    )
    monkeypatch.setattr(controller, "materialized_source", lambda _root: nullcontext())

    def launch(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("watchdog")
        return watchdog_launch()

    monkeypatch.setattr(controller.adapter, "launch", launch)
    monkeypatch.setattr(controller, "_assert_parity_absent", lambda: events.append("absent"))
    monkeypatch.setattr(
        controller.parity,
        "render",
        lambda *_args: {
            "objects": {
                "items": [
                    {"kind": "ConfigMap", "metadata": {"name": "package"}},
                    {"kind": "ConfigMap", "metadata": {"name": "authorization"}},
                    {"kind": "Job", "metadata": {"name": "parity"}},
                ]
            }
        },
    )

    def create(item: dict[str, object]) -> str:
        events.append("create:" + str(item["kind"]))
        return "CREATED"

    monkeypatch.setattr(controller.live_engine, "_create_or_verify_exact", create)
    monkeypatch.setattr(
        controller,
        "_wait_parity",
        lambda _binding: events.append("parity-passed") or parity_result(),
    )
    result = controller.run(ROOT, COMMIT, Path("unused"))
    assert events == [
        "ready",
        "watchdog",
        "absent",
        "create:ConfigMap",
        "create:ConfigMap",
        "create:Job",
        "parity-passed",
    ]
    assert result["status"] == "WATCHDOG_ACTIVE_INCLUSTER_PARITY_PASSED_NON_SCORED"
    assert result["scoring_calls"] == 0
    assert result["scored_launch_authorized"] is False


def test_v30_package_is_generation_exact_isolated_and_held(tmp_path: Path) -> None:
    rendered = package.render(ROOT, COMMIT, authorization())
    source, auth, job = rendered["objects"]["items"]
    manifest = json.loads(source["data"]["package.json"])
    assert manifest["schema_version"] == package.PACKAGE_SCHEMA
    assert manifest["package_commit"] == COMMIT
    assert set(manifest["files"]) == set(package.FILES)
    assert "evals/fleet/glm53_dedicated_v30_incluster_parity_v1.py" in package.FILES
    assert len(json.dumps(source)) < 1_000_000
    assert auth["immutable"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    command = pod["containers"][0]["command"][-1]
    assert "glm53_dedicated_v30_controller_v1" in command
    assert "glm53_dedicated_v29_controller_v1" not in command
    for relative in package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v30_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert rendered["server_launch_authorized"] is False
    assert rendered["qualification_launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False


def test_v30_watchdog_and_parity_contracts_are_600s_score_free() -> None:
    assert adapter.engine.runtime.IDLE_RELEASE_SECONDS == 600
    assert watchdog.JOB_NAME == "chris-glm53-dedicated-v30-request-watchdog-v1"
    assert parity.SERVER_TITLE == server.TITLE
    assert parity.SERVER_RUN_DIR == server.RUN_DIR
    held = adapter.build_held(COMMIT)
    assert held["idle_release_seconds"] == 600
    assert held["server_launch_authorized"] is False
    assert held["watchdog_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False


def test_v30_composite_held_receipt_is_digest_valid_and_no_launch() -> None:
    expected = package.build_held(COMMIT)
    path = (
        ROOT
        / "docs/evidence/glm53-study"
        / "2026-09-06-glm53-dedicated-v30-zero-state-lifecycle-held-v1.json"
    )
    observed = json.loads(path.read_text())
    assert observed == expected
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )
    assert observed["server_launch_authorized"] is False
    assert observed["qualification_launch_authorized"] is False
    assert observed["scored_launch_authorized"] is False
    assert observed["api_mutation_calls"] == 0
