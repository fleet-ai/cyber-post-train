from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_early_live_v1 as live
from evals.fleet import qwen38_dp8_early_qualification_v1 as early
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, status_code: int, value: dict) -> None:
        self.status_code = status_code
        self._value = value

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict:
        return self._value


class Client:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post(self, path: str, *, json: dict) -> Response:
        self.posts.append((path, json))
        return Response(202, {"name": "ft-run-12345678"})

    def get(self, path: str) -> Response:
        return Response(
            200,
            {
                "name": path.rsplit("/", 1)[-1],
                "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-unknown-v1",
                "status": "RUNNING",
            },
        )


def _gate(payload: dict, release: dict, source_commit: str) -> dict:
    value = {
        "schema_version": live.LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": "2026-09-06T00:00:00Z",
        "freshness_seconds_at_submit": 0,
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": "sha256:" + "a" * 64,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "allowed_active_peer": {},
        "project_resource_shape": {
            "current_gpu_nodes": 1,
            "current_gpus": 1,
            "projected_gpu_nodes": 2,
            "projected_gpus": 9,
            "maximum_gpu_nodes": 2,
            "maximum_gpus": 16,
            "project_object_inventory": {
                "orphan_project_rayjobs": 0,
                "orphan_project_gpu_pods": 0,
            },
        },
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_matches": 0,
        "sfs_run_dir_exists": False,
        "rendered": {},
        "api_mutations": 0,
        "scoring_authorized": False,
        "statistical_cells_selected": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_submitter_posts_exactly_once_only_after_digest_valid_live_gate() -> None:
    payload = {"title": early.TITLE, "run_dir": early.RUN_DIR}
    release = {"receipt_sha256": "sha256:" + "b" * 64}
    source_commit = "c" * 40
    gate = _gate(payload, release, source_commit)
    client = Client()
    assert live.submit_create_once(client, payload, gate, release, source_commit) == (
        "ft-run-12345678"
    )
    assert client.posts == [("/v1/runs", payload)]

    changed = copy.deepcopy(gate)
    changed["jobs_api_title_matches"] = 1
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="not clear"):
        live.submit_create_once(client, payload, changed, release, source_commit)
    assert len(client.posts) == 1


def test_kubernetes_gate_requires_exact_tp1_peer_and_one_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = json.loads((ROOT / live.TP1_BINDING_PATH).read_text())
    pod = {
        "kind": "Pod",
        "metadata": {"name": "tp1-head", "uid": binding["head_pod_uid"]},
        "spec": {
            "nodeName": "gpu-node-a",
            "containers": [
                {
                    "env": [
                        {
                            "name": "RUN_DIR",
                            "value": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
                        }
                    ],
                    "resources": {"requests": {"nvidia.com/gpu": "1"}},
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": True, "restartCount": 0}],
        },
    }
    inventory = {
        "items": [
            {
                "kind": "RayJob",
                "metadata": {"name": binding["api_run_id"], "uid": binding["rayjob_uid"]},
                "status": {"jobStatus": "RUNNING"},
            },
            {"kind": "Workload", "metadata": {"uid": binding["workload_uid"]}},
            pod,
            {"kind": "Service", "metadata": {"uid": binding["service_uid"]}},
            {
                "kind": "Pod",
                "metadata": {"name": "unrelated-peer", "uid": "peer-uid"},
                "spec": {
                    "nodeName": "other-gpu-node",
                    "containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}],
                },
                "status": {"phase": "Running"},
            },
        ]
    }
    monkeypatch.setattr(live.shared, "_kubectl", lambda *_args: json.dumps(inventory))
    active = [
        {
            "api_run_id": binding["api_run_id"],
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
            "status": "RUNNING",
        }
    ]
    shape, observer = live._kubernetes_gate(binding, active)  # noqa: SLF001
    assert observer == "tp1-head"
    assert shape["projected_gpu_nodes"] == 2
    assert shape["projected_gpus"] == 9
    assert shape["maximum_gpu_nodes"] == 2
    assert shape["maximum_gpus"] == 16
    assert shape["scope"] == "project_chris_cyber_evalserve_runs_only"
    assert shape["unrelated_namespace_gpu_pods_counted"] is False
    assert shape["project_object_inventory"]["gpu_requests"] == 1
    assert shape["project_object_inventory"]["orphan_project_gpu_pods"] == 0

    pod["status"]["containerStatuses"][0]["restartCount"] = 1
    with pytest.raises(RuntimeError, match="not Running/Ready/restart0"):
        live._kubernetes_gate(binding, active)  # noqa: SLF001


def test_kubernetes_gate_rejects_orphan_project_gpu_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = json.loads((ROOT / live.TP1_BINDING_PATH).read_text())
    pod = {
        "kind": "Pod",
        "metadata": {"name": "tp1", "uid": binding["head_pod_uid"]},
        "spec": {
            "nodeName": "node-a",
            "containers": [
                {
                    "env": [
                        {
                            "name": "RUN_DIR",
                            "value": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
                        }
                    ],
                    "resources": {"requests": {"nvidia.com/gpu": "1"}},
                }
            ],
        },
        "status": {"phase": "Running", "containerStatuses": [{"ready": True, "restartCount": 0}]},
    }
    orphan = copy.deepcopy(pod)
    orphan["metadata"] = {"name": "orphan", "uid": "orphan-uid"}
    orphan["spec"]["containers"][0]["env"][0]["value"] = (
        "/mnt/sfs/jobs/chris-cyber-evalserve-orphan-v1"
    )
    inventory = {
        "items": [
            {
                "kind": "RayJob",
                "metadata": {
                    "name": binding["api_run_id"],
                    "uid": binding["rayjob_uid"],
                },
                "status": {"jobStatus": "RUNNING"},
            },
            {"kind": "Workload", "metadata": {"uid": binding["workload_uid"]}},
            pod,
            orphan,
            {"kind": "Service", "metadata": {"uid": binding["service_uid"]}},
        ]
    }
    monkeypatch.setattr(live.shared, "_kubectl", lambda *_args: json.dumps(inventory))
    active = [
        {
            "api_run_id": binding["api_run_id"],
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
            "status": "RUNNING",
        }
    ]
    with pytest.raises(RuntimeError, match="orphan project serving"):
        live._kubernetes_gate(binding, active)  # noqa: SLF001


def test_unknown_active_chris_serving_run_is_not_an_allowed_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live.shared,
        "_runs",
        lambda _client: [
            {
                "name": "ft-run-unknown",
                "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-unknown-v1",
            }
        ],
    )
    rows = live._active_serving_runs(Client())  # noqa: SLF001
    assert rows == [
        {
            "api_run_id": "ft-run-unknown",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-unknown-v1",
            "status": "RUNNING",
        }
    ]
    assert rows != [
        {
            "api_run_id": "ft-run-e87e2bd4",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1",
            "status": "RUNNING",
        }
    ]
    binding = json.loads((ROOT / live.TP1_BINDING_PATH).read_text())
    with pytest.raises(RuntimeError, match="not exactly productive TP1-j"):
        live._validate_active_peer(rows, binding)  # noqa: SLF001


def test_server_release_stays_score_free_and_commit_bound() -> None:
    config, plan, preview, inventory, held = early.load_all(ROOT)
    binding = json.loads((ROOT / live.TP1_BINDING_PATH).read_text())
    source_commit = "d" * 40
    value = {
        "schema_version": live.SERVER_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_NON_SCORED_DP8_SERVER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "title": early.TITLE,
        "run_dir": early.RUN_DIR,
        "serving_block": early.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "held_release_receipt_sha256": held["receipt_sha256"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "preview_receipt_sha256": preview["receipt_sha256"],
        "review_inventory_receipt_sha256": inventory["receipt_sha256"],
        "allowed_peer_binding_receipt_sha256": binding["receipt_sha256"],
        "server_create_limit": 1,
        "statistical_cells_selected": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    live.validate_server_release(value, ROOT, source_commit)
    value["scoring_authorized"] = True
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError, match="not executable"):
        live.validate_server_release(value, ROOT, source_commit)


def test_submission_embeds_and_revalidates_complete_sanitized_live_gate() -> None:
    config = early.load_all(ROOT)[0]
    payload = early.jobs_payload(ROOT)
    release = {"receipt_sha256": "sha256:" + "b" * 64}
    source_commit = "c" * 40
    gate = _gate(payload, release, source_commit)
    gate["config_sha256"] = config["config_sha256"]
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    receipt = {
        "schema_version": live.SUBMISSION_SCHEMA,
        "status": "SUBMITTED_NON_SCORED_SERVER",
        "submitted_at_utc": "2026-09-06T00:00:00Z",
        "api_run_id": "ft-run-12345678",
        "title": early.TITLE,
        "run_dir": early.RUN_DIR,
        "serving_block": early.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "server_release_receipt_sha256": release["receipt_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "live_gate": gate,
        "source_commit": source_commit,
        "project_resource_shape": gate["project_resource_shape"],
        "route": "POST /v1/runs",
        "http_status": 202,
        "server_instances_created": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    live.validate_submission(receipt, payload, release, source_commit, ROOT)

    drifted = copy.deepcopy(receipt)
    drifted["live_gate"]["jobs_api_title_matches"] = 1
    drifted["receipt_sha256"] = self_hosted.digest_without(drifted, "receipt_sha256")
    with pytest.raises(ValueError, match="receipt drifted"):
        live.validate_submission(drifted, payload, release, source_commit, ROOT)
