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
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as core
from evals.fleet import glm53_dedicated_v31_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v31_controller_v1 as controller
from evals.fleet import glm53_dedicated_v31_create_v1 as server
from evals.fleet import glm53_dedicated_v31_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v31_watchdog_live_release_v1 as adapter

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "65a3949229dd4360b66fc2fc1b9b1d46802f9a86"


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


def live_state() -> dict[str, object]:
    return {
        "head_pod_name": "ft-run-1234abcd-abcde-head-12345",
        "application_ready_receipt_sha256": "sha256:" + "3" * 64,
    }


def engine_receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": adapter.LAUNCH_SCHEMA,
        "status": "WATCHDOG_CREATE_REQUEST_ACCEPTED",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding())),
        "runtime": {
            "watchdog_job_uid": "55555555-5555-4555-8555-555555555555",
            "watchdog_pod_uid": "66666666-6666-4666-8666-666666666666",
            "active_receipt_sha256": "sha256:" + "2" * 64,
        },
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v31_create_gate_is_fresh_zero_to_one_node_eight_gpus() -> None:
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
    assert server.TITLE.endswith("-v31")
    assert server.payload()["priority_class"] == "fleet-infra-quiet"
    assert server.payload()["workers"] == 1
    assert server.payload()["gpus_per_worker"] == 8


def test_core_launch_observes_exactly_once_and_passes_same_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    observed_binding = binding()
    observed_live = live_state()
    monkeypatch.setattr(
        core,
        "observe_live",
        lambda run_id: calls.append(("observe", run_id))
        or (observed_binding, observed_live),
    )

    def launch_from(
        _root: Path,
        _commit: str,
        got_binding: dict[str, object],
        got_live: dict[str, object],
        *,
        priority_classes: list[dict[str, object]],
    ) -> dict[str, object]:
        calls.append(("launch", got_binding is observed_binding, got_live is observed_live))
        assert priority_classes == []
        return {"ok": True}

    monkeypatch.setattr(core, "launch_from_observation", launch_from)
    assert core.launch(ROOT, COMMIT, "ft-run-1234abcd", priority_classes=[]) == {
        "ok": True
    }
    assert calls == [
        ("observe", "ft-run-1234abcd"),
        ("launch", True, True),
    ]


def test_adapter_creates_o_excl_prevalidation_once_and_never_reobserves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "WATCHDOG-LIVE-PREVALIDATION.json"
    calls = {"observe": 0, "launch_from": 0}
    observed_binding = binding()
    observed_live = live_state()

    def observe(_run_id: str) -> tuple[dict[str, object], dict[str, object]]:
        calls["observe"] += 1
        with marker.open("x") as handle:
            json.dump({"binding": observed_binding}, handle)
        return observed_binding, observed_live

    def launch_from(
        _root: Path,
        _commit: str,
        got_binding: dict[str, object],
        got_live: dict[str, object],
        *,
        priority_classes: list[dict[str, object]],
    ) -> dict[str, object]:
        calls["launch_from"] += 1
        assert marker.is_file()
        assert got_binding is observed_binding
        assert got_live is observed_live
        assert priority_classes == []
        return engine_receipt()

    monkeypatch.setattr(adapter.engine, "observe_live", observe)
    monkeypatch.setattr(adapter.engine, "launch_from_observation", launch_from)
    monkeypatch.setattr(
        adapter.engine,
        "launch",
        lambda *_args, **_kwargs: pytest.fail("second observation path invoked"),
    )
    result = adapter.launch(ROOT, COMMIT, "ft-run-1234abcd", priority_classes=[])
    assert calls == {"observe": 1, "launch_from": 1}
    assert result["status"] == "WATCHDOG_ACTIVE_UID_BOUND_SINGLE_OBSERVATION"
    assert result["server_binding"] == observed_binding
    assert result["receipt_sha256"] == crypto.digest_without(
        result, "receipt_sha256"
    )


def test_launch_from_observation_rolls_back_and_releases_on_create_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rendered = {
        "objects": {
            "items": [{"kind": "ConfigMap", "metadata": {"name": "watchdog"}}]
        }
    }
    monkeypatch.setattr(core, "validate_live_state", lambda *_args: None)
    monkeypatch.setattr(core, "render", lambda *_args, **_kwargs: rendered)
    monkeypatch.setattr(
        core, "_reconcile_existing_watcher_objects", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        core,
        "_create_or_verify_exact",
        lambda _item: (_ for _ in ()).throw(core.LiveReleaseError("create_failed")),
    )
    monkeypatch.setattr(
        core, "_rollback_watcher_objects", lambda _value: events.append("rollback")
    )
    monkeypatch.setattr(
        core,
        "_release_on_handoff_failure",
        lambda _binding, _pod: events.append("release"),
    )
    with pytest.raises(core.LiveReleaseError, match="create_failed"):
        core.launch_from_observation(
            ROOT, COMMIT, binding(), live_state(), priority_classes=[]
        )
    assert events == ["rollback", "release"]


def test_v31_controller_rehashes_score_free_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value: dict[str, object] = {
        "schema_version": "fleet-glm53-dedicated-v30-controller-result-v1",
        "status": "WATCHDOG_ACTIVE_INCLUSTER_PARITY_PASSED_NON_SCORED",
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    monkeypatch.setattr(controller.engine, "run", lambda *_args: value.copy())
    result = controller.run(ROOT, COMMIT, Path("unused"))
    assert result["schema_version"] == controller.RESULT_SCHEMA
    assert result["scoring_calls"] == 0
    assert result["receipt_sha256"] == crypto.digest_without(
        result, "receipt_sha256"
    )


def test_v31_package_is_generation_exact_isolated_and_held(tmp_path: Path) -> None:
    rendered = package.render(ROOT, COMMIT, authorization())
    source, auth, job = rendered["objects"]["items"]
    manifest = json.loads(source["data"]["package.json"])
    assert manifest["schema_version"] == package.PACKAGE_SCHEMA
    assert manifest["package_commit"] == COMMIT
    assert set(manifest["files"]) == set(package.FILES)
    assert "evals/fleet/glm53_dedicated_v31_controller_v1.py" in package.FILES
    assert len(json.dumps(source)) < 1_000_000
    assert auth["immutable"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    command = pod["containers"][0]["command"][-1]
    assert "glm53_dedicated_v31_controller_v1" in command
    assert "glm53_dedicated_v30_controller_v1" not in command
    for relative in package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v31_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    held = package.build_held(ROOT, COMMIT)
    assert held["exactly_one_live_observation"] is True
    assert held["launch_from_immutable_observation"] is True
    assert held["idle_release_seconds"] == 600
    assert held["scoring_calls"] == 0
    assert held["server_launch_authorized"] is False
    assert held["qualification_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")


def test_v31_parity_and_watchdog_are_held_score_free() -> None:
    assert parity.build_held()["scored_launch_authorized"] is False
    held = adapter.build_held(COMMIT)
    assert held["idle_release_seconds"] == 600
    assert held["exactly_one_live_observation"] is True
    assert held["api_mutation_calls"] == 0
    assert held["scored_launch_authorized"] is False


def test_v31_held_and_v30_tombstone_receipts_are_digest_valid() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    v31 = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v31-zero-state-lifecycle-held-v1.json"
        ).read_text()
    )
    v30 = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v30-double-observation-release-v1.json"
        ).read_text()
    )
    assert v31 == package.build_held(ROOT, COMMIT)
    assert v31["receipt_sha256"] == crypto.digest_without(v31, "receipt_sha256")
    assert v30["receipt_sha256"] == crypto.digest_without(v30, "receipt_sha256")
    assert v30["status"] == "RELEASED_ZERO_GPU_REMNANTS"
    assert v30["retry_same_identity"] is False
    assert v30["fleet_task_instance_calls"] == 0
    assert v30["fleet_session_calls"] == 0
    assert v30["verifier_calls"] == 0
    assert v30["scoring_calls"] == 0
