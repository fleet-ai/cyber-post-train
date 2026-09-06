from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_f_scorefree_v1 as consumed
from evals.fleet import qwen38_dp6_g_live_v1 as live
from evals.fleet import qwen38_dp6_g_scorefree_v1 as held
from evals.fleet import qwen38_dp6_qualification_guard_v2 as lifecycle_guard
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, status_code: int, value: dict) -> None:
        self.status_code = status_code
        self.value = value

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict:
        return self.value


class Client:
    def __init__(self) -> None:
        self.posts: list[str] = []

    def post(self, path: str, *, json: dict) -> Response:
        self.posts.append(path)
        if path.endswith("preview"):
            return Response(200, {"manifest_yaml": "preview"})
        return Response(202, {"name": "ft-run-dp6g"})


def _resign(value: dict[str, object]) -> None:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")


def _project_shape() -> dict[str, object]:
    return {
        "current_gpu_nodes": 0,
        "current_gpus": 0,
        "projected_gpu_nodes": 1,
        "projected_gpus": 6,
        "maximum_gpu_nodes": 2,
        "maximum_gpus": 16,
        "active_project_rayjobs": 0,
        "active_project_gpu_pods": 0,
        "capacity": {
            "eligible_six_gpu_node_count": 1,
            "eligible_node_uids": ["11111111-1111-4111-8111-111111111111"],
            "b300_nominal_gpu_quota": 128,
            "b300_used_gpu_quota": 64,
            "b300_gpu_quota_headroom": 64,
            "local_queue_uid": "22222222-2222-4222-8222-222222222222",
            "cluster_queue_uid": "33333333-3333-4333-8333-333333333333",
            "priority_class": {
                "name": "fleet-infra-quiet",
                "value": -1000,
                "preemption_policy": "Never",
            },
            "peer_preemption_required": False,
        },
    }


def test_held_packet_is_append_only_score_free_and_exact() -> None:
    config, plan, preview, inventory, release = held.load_all(ROOT)
    assert held.TITLE == "chris-cyber-evalserve-q38-dp6-g-v1"
    assert held.RUN_DIR == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-g-v1"
    assert held.QUALIFIER_JOB == "chris-cyber-q38-dp6-g-qualifier-v9"
    assert held.TITLE != consumed.TITLE
    assert held.RUN_DIR != consumed.RUN_DIR
    assert held.SERVING_BLOCK != consumed.SERVING_BLOCK
    assert held.QUALIFIER_JOB != consumed.QUALIFIER_JOB
    assert held.QUALIFIER_OUTPUT_ROOT != consumed.QUALIFIER_OUTPUT_ROOT
    assert all(value["launch_authorized"] is False for value in (config, plan, release))
    assert plan["scored_calls"] == release["scored_calls"] == 0
    assert preview["http_status"] == 200 and preview["api_mutations"] == 0
    assert inventory["status"] == "BLOCKED_ACTIVE_PROJECT_SERVING"
    assert inventory["active_project_serving_runs"] == 1
    assert release["status"] == "HELD_BLOCKED_ACTIVE_PROJECT_SERVING"
    assert inventory["capacity"]["schedulable_six_gpu_fit_nodes"] >= 1
    assert plan["superseded_plan"]["rewritten"] is False
    assert plan["successor_authority"]["receipt_sha256"].startswith("sha256:")
    assert plan["qualification"]["server_pre_ready_timeout_seconds"] == 600
    assert plan["qualification"]["counter_producer_startup_grace_seconds"] == 120
    assert plan["qualification"]["observer_sfs_mount_path_must_be_derived_from_pod_spec"]


def test_payload_installs_metric_observer_v4_and_uses_600_second_rail() -> None:
    payload = held.jobs_payload(ROOT)
    assert payload["priority_class"] == "fleet-infra-quiet"
    assert payload["gpus_per_worker"] == 6
    assert payload["workers"] == 1
    assert "qwen38_dp6_metric_observer_v4.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v3.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v2.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v1.py" in payload["command"]
    assert "IDLE_SECONDS=600" in payload["command"]
    assert "PRE_READY_TIMEOUT_SECONDS=600" in payload["command"]
    assert payload["env"]["QWEN38_DP6_OBSERVER_SCRIPT"].endswith("observer_v4.py")
    assert payload["env"]["PYTHONPATH"] == "/tmp"
    lifecycle_guard.validate_observer_environment(
        payload["env"], held.RUNTIME_DEPENDENCY_DIR
    )
    priority = held.config(ROOT)["priority_contract"]
    assert priority == {
        "class": "fleet-infra-quiet",
        "value": -1000,
        "preemption_policy": "Never",
        "jobs_api_allowed_classes": ["fleet-train-high", "fleet-infra-quiet"],
        "fleet_serve_low_preview_http_status": 422,
        "fleet_train_high_disallowed_as_preempting": True,
        "highest_jobs_api_admitted_nonpreempting": True,
        "fresh_jobs_api_preview_acceptance_required": True,
    }


@pytest.mark.parametrize(
    ("artifact", "field", "replacement", "validator"),
    [
        (held.PLAN_PATH, "launch_authorized", True, held.validate_plan),
        (held.PREVIEW_PATH, "http_status", 422, held.validate_preview),
        (held.INVENTORY_PATH, "active_project_serving_runs", 0, held.validate_inventory),
        (held.RELEASE_PATH, "launch_authorized", True, held.validate_held_release),
    ],
)
def test_artifacts_fail_closed(
    artifact: Path, field: str, replacement: object, validator: object
) -> None:
    value = copy.deepcopy(held._load(ROOT / artifact))  # noqa: SLF001
    value[field] = replacement
    _resign(value)
    if validator in {held.validate_plan, held.validate_preview, held.validate_held_release}:
        with pytest.raises(ValueError):
            validator(value, ROOT)
    else:
        with pytest.raises(ValueError):
            validator(value)


def test_release_first_shape_rejects_any_active_or_excess_project_state() -> None:
    shape = _project_shape()
    assert live._shape_safe(shape)  # noqa: SLF001
    for field, value in (("current_gpus", 1), ("projected_gpus", 17)):
        changed = copy.deepcopy(shape)
        changed[field] = value
        assert not live._shape_safe(changed)  # noqa: SLF001


def test_create_once_repeats_complete_release_first_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = held.jobs_payload(ROOT)
    release = {"receipt_sha256": "sha256:" + "a" * 64}
    source_commit = "b" * 40
    shape = _project_shape()
    gate = {
        "schema_version": live.LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "active_project_serving_runs": 0,
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": {
            "observer_pod_name": "observer",
            "observer_pod_uid": "44444444-4444-4444-8444-444444444444",
            "observer_sfs_mount_path": "/shared",
            "run_dir_exists": False,
        },
        "project_resource_shape": shape,
        "rendered": {},
        "api_mutations": 0,
        "scored_calls": 0,
    }
    _resign(gate)
    monkeypatch.setattr(live, "_active_project_runs", lambda _client: [])
    monkeypatch.setattr(live, "_kubernetes_gate", lambda _active: shape)
    monkeypatch.setattr(live, "_observer_pod", lambda: ("observer", "uid", "/shared"))
    monkeypatch.setattr(
        live,
        "_sfs_absence",
        lambda _observer: copy.deepcopy(gate["sfs_observation"]),
    )
    monkeypatch.setattr(held, "preview_identity", lambda *_args: {})
    client = Client()
    assert (
        live.submit_create_once(client, payload, gate, release, source_commit, ROOT)
        == "ft-run-dp6g"
    )
    assert client.posts == ["/v1/runs/preview", "/v1/runs"]
    monkeypatch.setattr(
        live,
        "_active_project_runs",
        lambda _client: [{"api_run_id": "late", "run_dir": "late", "status": "RUNNING"}],
    )
    with pytest.raises(RuntimeError, match="release-first"):
        live.submit_create_once(Client(), payload, gate, release, source_commit, ROOT)


def test_inventory_validator_rejects_low_capacity_and_duplicate_identity() -> None:
    value = copy.deepcopy(held._load(ROOT / held.INVENTORY_PATH))  # noqa: SLF001
    value["capacity"]["nominal_quota_headroom_gpus"] = 5
    _resign(value)
    with pytest.raises(ValueError):
        held.validate_inventory(value)


def test_live_sfs_absence_uses_selected_observer_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pod = {
        "kind": "Pod",
        "metadata": {"name": "allie-dev", "uid": "observer-uid"},
        "spec": {
            "volumes": [
                {
                    "name": "shared",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                }
            ],
            "containers": [
                {
                    "name": "dev",
                    "volumeMounts": [{"name": "shared", "mountPath": "/shared"}],
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": True, "restartCount": 0}],
        },
    }
    monkeypatch.setattr(
        live,
        "_global",
        lambda *_args: {"items": [copy.deepcopy(pod)]}
        if "pods" in _args
        else copy.deepcopy(pod),
    )
    observed: list[str] = []

    def run(args, **_kwargs):
        observed.extend(args)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(live.subprocess, "run", run)
    observer = live._observer_pod()  # noqa: SLF001
    assert observer == ("allie-dev", "observer-uid", "/shared")
    receipt = live._sfs_absence(observer)  # noqa: SLF001
    assert receipt["observer_sfs_mount_path"] == "/shared"
    assert "/shared/jobs/chris-cyber-evalserve-q38-dp6-g-v1" in observed
    assert "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-g-v1" not in observed
    value = copy.deepcopy(held._load(ROOT / held.INVENTORY_PATH))  # noqa: SLF001
    value["target_identity_matches"]["sfs"] = 1
    _resign(value)
    with pytest.raises(ValueError):
        held.validate_inventory(value)


def test_submission_validator_binds_complete_nested_gate_and_rejects_drift() -> None:
    config = held.config(ROOT)
    source_commit = "b" * 40
    release = {"receipt_sha256": "sha256:" + "a" * 64}
    shape = _project_shape()
    gate = {
        "schema_version": live.LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": "2026-09-06T10:30:00Z",
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": config["config_sha256"],
        "request_sha256": config["request_sha256"],
        "active_project_serving_runs": 0,
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": {
            "observer_pod_name": "observer",
            "observer_pod_uid": "44444444-4444-4444-8444-444444444444",
            "observer_sfs_mount_path": "/shared",
            "run_dir_exists": False,
        },
        "rendered": held._load(ROOT / held.PREVIEW_PATH)["rendered"],  # noqa: SLF001
        "project_resource_shape": shape,
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _resign(gate)
    receipt = {
        "schema_version": live.SUBMISSION_SCHEMA,
        "status": "SUBMITTED_SCORE_FREE_DP6_G_SERVER",
        "api_run_id": "ft-run-dp6g",
        "source_commit": source_commit,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "server_release_receipt_sha256": release["receipt_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "live_gate": gate,
        "request_sha256": config["request_sha256"],
        "project_resource_shape": shape,
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _resign(receipt)
    live.validate_submission(receipt, release, ROOT, source_commit)
    tampered = copy.deepcopy(receipt)
    tampered["live_gate"]["project_resource_shape"]["projected_gpus"] = 7
    _resign(tampered["live_gate"])
    tampered["live_gate_receipt_sha256"] = tampered["live_gate"]["receipt_sha256"]
    tampered["project_resource_shape"] = tampered["live_gate"]["project_resource_shape"]
    _resign(tampered)
    with pytest.raises(ValueError):
        live.validate_submission(tampered, release, ROOT, source_commit)
    extra = copy.deepcopy(receipt)
    extra["live_gate"]["ignored_future_field"] = "unsafe"
    _resign(extra["live_gate"])
    extra["live_gate_receipt_sha256"] = extra["live_gate"]["receipt_sha256"]
    _resign(extra)
    with pytest.raises(ValueError):
        live.validate_submission(extra, release, ROOT, source_commit)


def test_release_first_plan_has_no_scored_cell_or_active_peer_exception() -> None:
    text = json.dumps(held.load_all(ROOT)[1], sort_keys=True)
    assert 'statistical_cells_selected": 0' in text
    assert "allowed_existing_serving_block" not in text
    assert "tp1-j" not in text
    assert held.QUALIFIER_PRIORITY_CLASS == "fleet-serve-low"
    assert held.QUALIFIER_PRIORITY_VALUE == 100
