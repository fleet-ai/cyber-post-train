from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_i_scorefree_v1 as consumed
from evals.fleet import qwen38_dp6_j_live_v1 as live
from evals.fleet import qwen38_dp6_j_qualifier_package_v1 as qualifier_package
from evals.fleet import qwen38_dp6_j_scorefree_v1 as held
from evals.fleet import qwen38_dp6_qualification_guard_v3 as lifecycle_guard
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


class InventoryClient:
    def __init__(self, values: dict[str, dict]) -> None:
        self.values = values

    def get(self, path: str) -> Response:
        return Response(200, self.values[path.rsplit("/", 1)[-1]])


def _resign(value: dict[str, object]) -> None:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")


def _project_shape() -> dict[str, object]:
    return {
        "current_gpu_nodes": 1,
        "current_gpus": 8,
        "projected_gpu_nodes": 2,
        "projected_gpus": 14,
        "maximum_gpu_nodes": 2,
        "maximum_gpus": 16,
        "active_project_rayjobs": 1,
        "active_project_gpu_pods": 1,
        "coexisting_glm_v29": copy.deepcopy(held.COEXISTING_GLM_V29),
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


def _glm_inventory() -> list[dict]:
    glm = held.COEXISTING_GLM_V29
    return [
        {
            "kind": "RayJob",
            "metadata": {"name": glm["rayjob_name"], "uid": glm["rayjob_uid"]},
            "spec": {"runtimeEnvYAML": glm["run_dir"]},
            "status": {"jobStatus": "RUNNING"},
        },
        {
            "kind": "Workload",
            "metadata": {
                "name": glm["workload_name"],
                "uid": glm["workload_uid"],
                "ownerReferences": [
                    {"kind": "RayJob", "name": glm["rayjob_name"], "uid": glm["rayjob_uid"]}
                ],
            },
            "status": {
                "conditions": [
                    {"type": "QuotaReserved", "status": "True"},
                    {"type": "Admitted", "status": "True"},
                ]
            },
        },
        {
            "kind": "RayCluster",
            "metadata": {
                "name": glm["raycluster_name"],
                "uid": glm["raycluster_uid"],
                "ownerReferences": [
                    {"kind": "RayJob", "name": glm["rayjob_name"], "uid": glm["rayjob_uid"]}
                ],
            },
        },
        {
            "kind": "Pod",
            "metadata": {
                "name": glm["head_pod_name"],
                "uid": glm["head_pod_uid"],
                "ownerReferences": [
                    {
                        "kind": "RayCluster",
                        "name": glm["raycluster_name"],
                        "uid": glm["raycluster_uid"],
                    }
                ],
            },
            "spec": {
                "nodeName": glm["head_node_name"],
                "containers": [
                    {
                        "env": [
                            {"name": "RUN_DIR", "value": glm["run_dir"]},
                            {"name": "GLM53_RUN_DIR", "value": glm["run_dir"]},
                        ],
                        "resources": {"requests": {"nvidia.com/gpu": "8"}},
                    }
                ],
            },
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"restartCount": 0}],
            },
        },
        {
            "kind": "Service",
            "metadata": {
                "name": glm["service_name"],
                "uid": glm["service_uid"],
                "ownerReferences": [
                    {
                        "kind": "RayCluster",
                        "name": glm["raycluster_name"],
                        "uid": glm["raycluster_uid"],
                    }
                ],
            },
            "spec": {
                "selector": {"ray.io/cluster": glm["raycluster_name"]},
                "ports": [{"name": "serve", "port": 8000, "targetPort": 8000}],
            },
        },
        {
            "kind": "Job",
            "metadata": {"name": glm["watchdog_job_name"], "uid": glm["watchdog_job_uid"]},
            "status": {"active": 1},
        },
        {
            "kind": "Pod",
            "metadata": {
                "name": glm["watchdog_pod_name"],
                "uid": glm["watchdog_pod_uid"],
                "ownerReferences": [
                    {
                        "kind": "Job",
                        "name": glm["watchdog_job_name"],
                        "uid": glm["watchdog_job_uid"],
                    }
                ],
            },
            "spec": {"containers": [{}]},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"restartCount": 0}],
            },
        },
    ]


def test_held_packet_is_append_only_score_free_and_exact() -> None:
    config, plan, preview, inventory, release = held.load_all(ROOT)
    assert held.TITLE == "chris-cyber-evalserve-q38-dp6-j-v1"
    assert held.RUN_DIR == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-j-v1"
    assert held.QUALIFIER_JOB == "chris-cyber-q38-dp6-j-qualifier-v12"
    assert held.TITLE != consumed.TITLE
    assert held.RUN_DIR != consumed.RUN_DIR
    assert held.SERVING_BLOCK != consumed.SERVING_BLOCK
    assert held.QUALIFIER_JOB != consumed.QUALIFIER_JOB
    assert held.QUALIFIER_OUTPUT_ROOT != consumed.QUALIFIER_OUTPUT_ROOT
    assert all(value["launch_authorized"] is False for value in (config, plan, release))
    assert plan["scored_calls"] == release["scored_calls"] == 0
    assert preview["http_status"] == 200 and preview["api_mutations"] == 0
    assert inventory["status"] == "CLEAR_EXACT_GLM_V29_COEXISTENCE_FOR_REVIEW_ONLY"
    assert inventory["active_project_serving_runs"] == 1
    assert inventory["active_project_gpu_nodes"] == 1
    assert inventory["active_project_gpus"] == 8
    assert release["status"] == "HELD_FOR_INDEPENDENT_REVIEW"
    assert inventory["allowed_existing_serving_binding"] == held.COEXISTING_GLM_V29
    assert plan["coexistence_gate"]["projected_gpu_nodes"] == 2
    assert plan["coexistence_gate"]["projected_gpus"] == 14
    assert inventory["capacity"]["schedulable_six_gpu_fit_nodes"] >= 1
    assert plan["superseded_plan"]["rewritten"] is False
    assert plan["successor_authority"]["receipt_sha256"].startswith("sha256:")
    assert plan["qualification"]["server_pre_ready_timeout_seconds"] == 600
    assert plan["qualification"]["counter_producer_startup_grace_seconds"] == 120
    assert plan["qualification"]["controller_resource_sample_wait_seconds"] == 30
    assert plan["qualification"]["controller_resource_sample_ready_before_probe"] is True
    assert plan["qualification"]["observer_sfs_mount_path_must_be_derived_from_pod_spec"]
    assert plan["qualification"]["server_binding_projected_create_once_before_counter_producer"]
    assert plan["qualification"]["server_binding_projection_exact_readback_required"]
    assert plan["qualification"]["fresh_sanitized_failed_observer_status_releases_immediately"]


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


def test_qualifier_package_closure_contains_fresh_h_guard() -> None:
    paths = qualifier_package.source_paths(ROOT)
    assert Path("evals/fleet/qwen38_dp6_qualification_guard_v3.py") in paths
    assert Path("evals/fleet/qwen38_dp6_j_qualifier_runtime_v1.py") in paths


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


def test_coexistence_shape_rejects_identity_or_excess_project_state() -> None:
    shape = _project_shape()
    assert live._shape_safe(shape)  # noqa: SLF001
    for field, value in (("current_gpus", 0), ("projected_gpus", 17)):
        changed = copy.deepcopy(shape)
        changed[field] = value
        assert not live._shape_safe(changed)  # noqa: SLF001


def test_kubernetes_gate_accepts_only_exact_healthy_glm_v29(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _glm_inventory()
    node = {
        "metadata": {
            "name": held.COEXISTING_GLM_V29["head_node_name"],
            "uid": held.COEXISTING_GLM_V29["head_node_uid"],
        },
        "spec": {"unschedulable": False},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    monkeypatch.setattr(
        live,
        "_global",
        lambda *args: copy.deepcopy(node) if "node" in args else {"items": copy.deepcopy(items)},
    )
    monkeypatch.setattr(live, "_capacity_gate", lambda: copy.deepcopy(_project_shape()["capacity"]))
    shape = live._kubernetes_gate([copy.deepcopy(live.EXPECTED_ACTIVE_API)])  # noqa: SLF001
    assert live._shape_safe(shape)  # noqa: SLF001


@pytest.mark.parametrize(
    ("kind", "name_key", "mutate"),
    [
        ("RayJob", "rayjob_name", lambda row: row["metadata"].update(uid="bad")),
        (
            "Workload",
            "workload_name",
            lambda row: row["status"]["conditions"][1].update(status="False"),
        ),
        (
            "Pod",
            "head_pod_name",
            lambda row: row["status"]["containerStatuses"][0].update(restartCount=1),
        ),
        ("Service", "service_name", lambda row: row["metadata"].update(uid="bad")),
        ("Job", "watchdog_job_name", lambda row: row["status"].update(active=0)),
    ],
)
def test_kubernetes_gate_rejects_any_exact_glm_v29_drift(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    name_key: str,
    mutate: Callable[[dict], None],
) -> None:
    items = _glm_inventory()
    row = next(
        item
        for item in items
        if item["kind"] == kind
        and item["metadata"]["name"] == held.COEXISTING_GLM_V29[name_key]
    )
    mutate(row)
    node = {
        "metadata": {"uid": held.COEXISTING_GLM_V29["head_node_uid"]},
        "spec": {"unschedulable": False},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    monkeypatch.setattr(
        live,
        "_global",
        lambda *args: copy.deepcopy(node) if "node" in args else {"items": copy.deepcopy(items)},
    )
    monkeypatch.setattr(live, "_capacity_gate", lambda: copy.deepcopy(_project_shape()["capacity"]))
    with pytest.raises(RuntimeError, match="GLM v29"):
        live._kubernetes_gate([copy.deepcopy(live.EXPECTED_ACTIVE_API)])  # noqa: SLF001


def test_kubernetes_gate_rejects_an_extra_active_project_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _glm_inventory()
    extra = copy.deepcopy(items[0])
    extra["metadata"] = {"name": "extra", "uid": "extra"}
    items.append(extra)
    node = {
        "metadata": {"uid": held.COEXISTING_GLM_V29["head_node_uid"]},
        "spec": {"unschedulable": False},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    monkeypatch.setattr(
        live,
        "_global",
        lambda *args: copy.deepcopy(node) if "node" in args else {"items": copy.deepcopy(items)},
    )
    monkeypatch.setattr(live, "_capacity_gate", lambda: copy.deepcopy(_project_shape()["capacity"]))
    with pytest.raises(RuntimeError, match="extra or drifted active project RayJob"):
        live._kubernetes_gate([copy.deepcopy(live.EXPECTED_ACTIVE_API)])  # noqa: SLF001


def test_jobs_api_gate_accepts_only_exact_glm_v29_and_rejects_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    glm = held.COEXISTING_GLM_V29
    row = {"name": glm["api_run_id"], "run_dir": glm["run_dir"]}
    values = {
        glm["api_run_id"]: {
            "name": glm["api_run_id"],
            "run_dir": glm["run_dir"],
            "status": "RUNNING",
        }
    }
    monkeypatch.setattr(live.shared, "_runs", lambda _client: [row])
    assert live._active_project_runs(InventoryClient(values)) == [live.EXPECTED_ACTIVE_API]
    extra = {"name": "ft-run-extra", "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-extra"}
    values["ft-run-extra"] = {**extra, "status": "RUNNING"}
    monkeypatch.setattr(live.shared, "_runs", lambda _client: [row, extra])
    with pytest.raises(RuntimeError, match="not exact healthy GLM v29"):
        live._active_project_runs(InventoryClient(values))


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
        "active_project_serving_runs": 1,
        "allowed_existing_serving_binding": copy.deepcopy(held.COEXISTING_GLM_V29),
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
    monkeypatch.setattr(live, "_active_project_runs", lambda _client: [live.EXPECTED_ACTIVE_API])
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
    def reject_drift(_active: object) -> dict:
        raise RuntimeError("coexistence drift")

    monkeypatch.setattr(live, "_kubernetes_gate", reject_drift)
    with pytest.raises(RuntimeError, match="coexistence drift"):
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
    assert "/shared/jobs/chris-cyber-evalserve-q38-dp6-j-v1" in observed
    assert "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-j-v1" not in observed
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
        "active_project_serving_runs": 1,
        "allowed_existing_serving_binding": copy.deepcopy(held.COEXISTING_GLM_V29),
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
        "status": "SUBMITTED_SCORE_FREE_DP6_J_SERVER",
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


def test_coexistence_plan_has_no_scored_cell_or_unbounded_peer_exception() -> None:
    text = json.dumps(held.load_all(ROOT)[1], sort_keys=True)
    assert 'statistical_cells_selected": 0' in text
    assert held.COEXISTING_GLM_V29["api_run_id"] in text
    assert '"allowed_active_project_serving_runs": 1' in text
    assert held.QUALIFIER_PRIORITY_CLASS == "fleet-serve-low"
    assert held.QUALIFIER_PRIORITY_VALUE == 100
