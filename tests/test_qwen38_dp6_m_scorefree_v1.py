from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals.fleet import qwen38_dp6_l_scorefree_v1 as consumed
from evals.fleet import qwen38_dp6_m_live_v1 as live
from evals.fleet import qwen38_dp6_m_qualifier_package_v1 as qualifier_package
from evals.fleet import qwen38_dp6_m_scorefree_v1 as held
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
        return Response(202, {"name": "q38-dp6m-v1-deadbeef"})


def _resign(value: dict[str, object]) -> None:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")


def test_scorefree_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"safe","status":"SECRET_PROTECTED_VALUE"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        held._load(path)  # noqa: SLF001


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


def _server_release(source_commit: str) -> dict[str, object]:
    config, plan, preview, inventory, held_release = held.load_all(ROOT)
    value: dict[str, object] = {
        "schema_version": live.SERVER_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_SCORE_FREE_DP6_M_SERVER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "title": held.TITLE,
        "name": held.API_NAME,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "preview_receipt_sha256": preview["receipt_sha256"],
        "review_inventory_receipt_sha256": inventory["receipt_sha256"],
        "held_release_receipt_sha256": held_release["receipt_sha256"],
        "release_first_reconciled": True,
        "server_create_limit": 1,
        "qualifier_create_limit": 0,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _resign(value)
    return value


def _preview_manifest(*, name: object = "q38-dp6m-v1-0123abcd") -> str:
    payload = held.jobs_payload(ROOT)
    return yaml.safe_dump(
        {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                "name": name,
                "labels": {"kueue.x-k8s.io/queue-name": "training-lq"},
            },
            "spec": {
                "suspend": True,
                "entrypoint": payload["command"],
                "rayClusterSpec": {
                    "headGroupSpec": {
                        "template": {
                            "spec": {
                                "priorityClassName": held.SERVER_PRIORITY_CLASS,
                                "containers": [
                                    {
                                        "image": payload["image"],
                                        "env": [{"name": "RUN_DIR", "value": held.RUN_DIR}],
                                        "resources": {
                                            "requests": {"nvidia.com/gpu": "6"},
                                            "limits": {"nvidia.com/gpu": "6"},
                                        },
                                    }
                                ],
                            }
                        }
                    }
                },
            },
        }
    )


def test_held_packet_is_append_only_score_free_and_exact() -> None:
    config, plan, preview, inventory, release = held.load_all(ROOT)
    assert held.TITLE == "chris-cyber-evalserve-q38-dp6-m-v1"
    assert held.API_NAME == "q38-dp6m-v1"
    assert held.RUN_DIR == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-m-v1"
    assert held.QUALIFIER_JOB == "chris-cyber-q38-dp6-m-qualifier-v13"
    assert held.TITLE != consumed.TITLE
    assert held.RUN_DIR != consumed.RUN_DIR
    assert held.SERVING_BLOCK != consumed.SERVING_BLOCK
    assert held.QUALIFIER_JOB != consumed.QUALIFIER_JOB
    assert held.QUALIFIER_OUTPUT_ROOT != consumed.QUALIFIER_OUTPUT_ROOT
    assert all(value["launch_authorized"] is False for value in (config, plan, release))
    assert plan["scored_calls"] == release["scored_calls"] == 0
    assert preview["status"] == "HELD_NOT_RUN"
    assert preview["http_status"] is None and preview["api_mutations"] == 0
    assert inventory["status"] == "HELD_NOT_QUERIED"
    assert inventory["active_project_serving_runs"] is None
    assert inventory["active_project_gpu_nodes"] is None
    assert inventory["active_project_gpus"] is None
    assert release["status"] == "HELD_CODE_REVIEW_ONLY"
    assert config["operator_reservation"] == {
        "path": "/tmp/fleet-qwen38-dp6m-v1-create.lock",
        "scope": "local_host_all_operators",
        "exclusive": True,
        "held_from": "final_preflight",
        "held_through": "submission_receipt_publication",
    }
    assert plan["server"]["operator_reservation"] == config["operator_reservation"]
    assert release["fresh_preview_and_inventory_required_before_release"] is True
    assert plan["superseded_plan"]["rewritten"] is False
    assert plan["successor_authority"]["receipt_sha256"].startswith("sha256:")
    assert plan["qualification"]["server_pre_ready_timeout_seconds"] == 600
    assert plan["qualification"]["counter_producer_startup_grace_seconds"] == 120
    assert plan["qualification"]["controller_resource_sample_wait_seconds"] == 30
    assert plan["qualification"]["controller_resource_sample_ready_before_probe"] is True
    assert plan["qualification"]["controller_resource_cgroup_membership_source"] == (
        "/proc/self/cgroup"
    )
    assert plan["qualification"]["controller_resource_exact_cgroup_v2_leaf_required"] is True
    assert plan["qualification"]["controller_resource_shared_ready_marker_required"] is True
    assert plan["qualification"]["controller_resource_sample_max_age_seconds"] == 5
    assert plan["qualification"]["controller_resource_sample_advance_wait_seconds"] == 5
    assert plan["qualification"]["controller_resource_producer_consumer_end_to_end_tested"] is True
    assert plan["qualification"]["http_failure_method_path_status_required"] is True
    assert plan["qualification"]["http_failure_structured_code_fixed_allowlist_only"] is True
    assert plan["qualification"]["http_failure_response_shape_and_class_hash_required"] is True
    assert plan["qualification"]["http_failure_free_text_forbidden"] is True
    assert plan["qualification"]["observer_sfs_mount_path_must_be_derived_from_pod_spec"]
    assert plan["qualification"]["server_binding_projected_create_once_before_counter_producer"]
    assert plan["qualification"]["server_binding_projection_exact_readback_required"]
    assert plan["qualification"]["fresh_sanitized_failed_observer_status_releases_immediately"]


def test_live_loader_and_server_release_reject_ambiguous_or_extra_inputs(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"safe","status":"SECRET_PROTECTED_VALUE"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        live._load(duplicate)  # noqa: SLF001

    source_commit = "a" * 40
    release = _server_release(source_commit)
    release["protected_score"] = "SECRET_MARKER"
    _resign(release)
    with pytest.raises(ValueError, match="not executable"):
        live.validate_server_release(release, ROOT, source_commit)


def test_submission_round_trip_binds_fresh_rendered_preview() -> None:
    source_commit = "a" * 40
    release = _server_release(source_commit)
    config = held.config(ROOT)
    gate: dict[str, object] = {
        "schema_version": live.LIVE_GATE_SCHEMA,
        "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
        "observed_at_utc": "2026-09-06T10:30:00Z",
        "source_commit": source_commit,
        "server_release_receipt_sha256": release["receipt_sha256"],
        "config_sha256": config["config_sha256"],
        "request_sha256": config["request_sha256"],
        "active_project_serving_runs": 0,
        "project_resource_shape": _project_shape(),
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": {
            "observer_pod_name": "observer",
            "observer_pod_uid": "44444444-4444-4444-8444-444444444444",
            "observer_sfs_mount_path": "/shared",
            "run_dir_exists": False,
        },
        "rendered": live._expected_rendered(ROOT),  # noqa: SLF001
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _resign(gate)
    receipt = live.submission_receipt("q38-dp6m-v1-0123abcd", gate, release, source_commit, ROOT)
    live.validate_submission(receipt, release, ROOT, source_commit)

    tampered = copy.deepcopy(receipt)
    tampered["live_gate"]["rendered"]["unexpected"] = "SECRET_MARKER"
    _resign(tampered["live_gate"])
    tampered["live_gate_receipt_sha256"] = tampered["live_gate"]["receipt_sha256"]
    _resign(tampered)
    with pytest.raises(ValueError, match="not executable evidence"):
        live.validate_submission(tampered, release, ROOT, source_commit)


def test_payload_installs_metric_observer_v5_and_uses_600_second_rail() -> None:
    payload = held.jobs_payload(ROOT)
    assert payload["name"] == held.API_NAME
    assert set(payload) == held.jobs_api.EXPECTED_API_FIELDS | {"name"}
    assert payload["priority_class"] == "fleet-infra-quiet"
    assert payload["gpus_per_worker"] == 6
    assert payload["workers"] == 1
    assert "qwen38_dp6_metric_observer_v5.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v3.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v2.py" in payload["command"]
    assert "qwen38_dp6_metric_observer_v1.py" in payload["command"]
    assert "IDLE_SECONDS=600" in payload["command"]
    assert "PRE_READY_TIMEOUT_SECONDS=600" in payload["command"]
    assert payload["env"]["QWEN38_DP6_OBSERVER_SCRIPT"].endswith("observer_v5.py")
    assert payload["env"]["PYTHONPATH"] == "/tmp"
    lifecycle_guard.validate_observer_environment(payload["env"], held.RUNTIME_DEPENDENCY_DIR)
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


@pytest.mark.parametrize("attack", ["missing", "wrong", "extra"])
def test_payload_rejects_missing_wrong_or_extra_name_binding(attack: str) -> None:
    payload = held.jobs_payload(ROOT)
    if attack == "missing":
        payload.pop("name")
    elif attack == "wrong":
        payload["name"] = "q38-dp6l-v1"
    else:
        payload["run_name"] = payload["name"]
    with pytest.raises(AssertionError, match="payload"):
        held.validate_jobs_payload(payload, ROOT)


def test_preview_identity_binds_generated_name_and_rejects_aliases() -> None:
    rendered = held.preview_identity(_preview_manifest(), ROOT)
    assert rendered["api_name"] == held.API_NAME
    assert rendered["api_run_id_pattern"] == held.API_RUN_ID_RE.pattern
    for attacked in (None, held.API_NAME, "q38-dp6l-v1-0123abcd", "q38-dp6m-v1-XYZ12345"):
        with pytest.raises(ValueError, match="run name"):
            held.preview_identity(_preview_manifest(name=attacked), ROOT)


def test_qualifier_package_closure_contains_fresh_k_guard() -> None:
    paths = qualifier_package.source_paths(ROOT)
    assert Path("evals/fleet/qwen38_dp6_qualification_guard_v3.py") in paths
    assert Path("evals/fleet/qwen38_dp6_m_qualifier_runtime_v1.py") in paths


@pytest.mark.parametrize(
    ("artifact", "field", "replacement", "validator"),
    [
        (held.PLAN_PATH, "launch_authorized", True, held.validate_plan),
        (held.PREVIEW_PATH, "http_status", 422, held.validate_preview),
        (held.INVENTORY_PATH, "active_project_serving_runs", 1, held.validate_inventory),
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


@pytest.mark.parametrize(
    ("artifact", "validator"),
    [
        (held.PREVIEW_PATH, held.validate_preview),
        (held.INVENTORY_PATH, held.validate_inventory),
    ],
)
def test_archived_receipts_reject_rehashed_protected_fields(
    artifact: Path, validator: object
) -> None:
    value = copy.deepcopy(held._load(ROOT / artifact))  # noqa: SLF001
    value["protected_score"] = "SECRET_MARKER"
    _resign(value)
    with pytest.raises(ValueError):
        if validator is held.validate_preview:
            validator(value, ROOT)
        else:
            validator(value)


def test_release_first_shape_rejects_any_active_or_excess_project_state() -> None:
    shape = _project_shape()
    assert live._shape_safe(shape)  # noqa: SLF001
    for field, value in (("current_gpus", 1), ("projected_gpus", 17)):
        changed = copy.deepcopy(shape)
        changed[field] = value
        assert not live._shape_safe(changed)  # noqa: SLF001


def test_generated_name_collision_blocks_even_without_run_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"name": "q38-dp6m-v1-deadbeef", "run_dir": None, "title": None}]
    monkeypatch.setattr(live.shared, "_runs", lambda _client: rows)
    with pytest.raises(RuntimeError, match="identity already exists"):
        live._active_project_runs(object())  # type: ignore[arg-type]  # noqa: SLF001


def test_operator_reservation_excludes_a_concurrent_local_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "create.lock"
    script = """
import sys
from pathlib import Path
from evals.fleet.qwen38_dp6_m_live_v1 import operator_reservation
with operator_reservation(Path(sys.argv[1])):
    print("LOCKED", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
        with (
            pytest.raises(RuntimeError, match="already held"),
            live.operator_reservation(lock_path),
        ):
            pass
        assert process.stdin is not None
        process.stdin.write("release\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
        with live.operator_reservation(lock_path):
            pass
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_operator_reservation_rejects_symlink_or_permissive_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("")
    target.chmod(0o600)
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="unsafe"), live.operator_reservation(symlink):
        pass

    permissive = tmp_path / "permissive.lock"
    permissive.write_text("")
    permissive.chmod(0o644)
    with pytest.raises(RuntimeError, match="unsafe"), live.operator_reservation(permissive):
        pass


def test_create_reservation_spans_preflight_post_and_receipt_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "create.lock"
    output = tmp_path / "submission.json"
    stages: list[str] = []

    def assert_locked(stage: str) -> None:
        with (
            pytest.raises(RuntimeError, match="already held"),
            live.operator_reservation(lock_path),
        ):
            pass
        stages.append(stage)

    def fake_gate(*_args):
        assert_locked("final_preflight")
        return {"name": held.API_NAME}, {"gate": True}

    def fake_submit(*_args):
        assert_locked("post")
        return "q38-dp6m-v1-deadbeef"

    def fake_receipt(*_args):
        assert_locked("receipt")
        return {"receipt_sha256": "sha256:" + "a" * 64}

    def fake_write(path: Path, value: dict[str, object]) -> None:
        assert path == output
        assert value["receipt_sha256"] == "sha256:" + "a" * 64
        assert_locked("publication")

    monkeypatch.setattr(live, "live_gate", fake_gate)
    monkeypatch.setattr(live, "submit_create_once", fake_submit)
    monkeypatch.setattr(live, "submission_receipt", fake_receipt)
    monkeypatch.setattr(live.shared, "_write_once", fake_write)
    receipt = live.submit_and_publish_create_once(
        object(),
        {},
        ROOT,
        "a" * 40,
        output,
        lock_path=lock_path,  # type: ignore[arg-type]
    )
    assert receipt["receipt_sha256"] == "sha256:" + "a" * 64
    assert stages == ["final_preflight", "post", "receipt", "publication"]
    with live.operator_reservation(lock_path):
        pass


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
        == "q38-dp6m-v1-deadbeef"
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
    value["target_identity_matches"] = {"jobs_api": 0, "kubernetes": 0, "sfs": 1}
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
        lambda *_args: {"items": [copy.deepcopy(pod)]} if "pods" in _args else copy.deepcopy(pod),
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
    assert "/shared/jobs/chris-cyber-evalserve-q38-dp6-m-v1" in observed
    assert "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-m-v1" not in observed
    value = copy.deepcopy(held._load(ROOT / held.INVENTORY_PATH))  # noqa: SLF001
    value["target_identity_matches"] = {"jobs_api": 0, "kubernetes": 0, "sfs": 1}
    _resign(value)
    with pytest.raises(ValueError):
        held.validate_inventory(value)


def test_release_first_plan_has_no_scored_cell_or_active_peer_exception() -> None:
    text = json.dumps(held.load_all(ROOT)[1], sort_keys=True)
    assert 'statistical_cells_selected": 0' in text
    assert "allowed_existing_serving_block" not in text
    assert "tp1-j" not in text
    assert held.QUALIFIER_PRIORITY_CLASS == "fleet-serve-low"
    assert held.QUALIFIER_PRIORITY_VALUE == 100
