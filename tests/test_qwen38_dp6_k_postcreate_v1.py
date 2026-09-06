from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_k_postcreate_v1 as postcreate
from evals.fleet import qwen38_dp6_k_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_k_scorefree_v1 as held
from evals.fleet import qwen38_dp6_metric_observer_v4 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
HELD_PATH = ROOT / "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-k-postcreate-held-v1.json"
INCIDENT_PATH = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-i-c1-terminal-v1.json"
)


def _receipt(value: dict) -> dict:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _submission() -> dict:
    return _receipt(
        {
            "api_run_id": "ft-run-example",
            "receipt_sha256": "",
        }
    )


def _objects() -> list[dict]:
    rayjob_uid = "11111111-1111-4111-8111-111111111111"
    cluster_uid = "22222222-2222-4222-8222-222222222222"
    return [
        {
            "kind": "RayJob",
            "metadata": {"name": "ft-run-example", "uid": rayjob_uid},
            "status": {"jobStatus": "RUNNING", "rayClusterName": "ft-run-example-abcd1"},
        },
        {
            "kind": "Workload",
            "metadata": {
                "name": "workload",
                "uid": "33333333-3333-4333-8333-333333333333",
                "ownerReferences": [{"uid": rayjob_uid}],
            },
            "status": {
                "conditions": [
                    {"type": "Admitted", "status": "True"},
                    {"type": "QuotaReserved", "status": "True"},
                ]
            },
        },
        {
            "kind": "RayCluster",
            "metadata": {
                "name": "ft-run-example-abcd1",
                "uid": cluster_uid,
                "ownerReferences": [{"uid": rayjob_uid}],
            },
        },
        {
            "kind": "Service",
            "metadata": {
                "name": "ft-run-example-abcd1-head-svc",
                "uid": "44444444-4444-4444-8444-444444444444",
                "ownerReferences": [{"uid": cluster_uid}],
            },
        },
        {
            "kind": "Pod",
            "metadata": {
                "name": "ft-run-example-abcd1-head-abcde",
                "uid": "55555555-5555-4555-8555-555555555555",
                "ownerReferences": [{"uid": cluster_uid}],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 0}],
            },
        },
    ]


def test_postcreate_authority_is_held_and_source_bound() -> None:
    value = json.loads(HELD_PATH.read_text())
    source = ROOT / value["source_path"]
    guard = ROOT / value["guard_source_path"]
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["source_sha256"] == self_hosted.sha256(source.read_bytes())
    assert value["guard_source_sha256"] == self_hosted.sha256(guard.read_bytes())
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["api_mutations"] == value["scored_calls"] == 0
    assert value["terminal_identity_reuse_authorized"] is False


def test_terminal_incident_is_digest_valid_score_free_and_nonreusable() -> None:
    value = json.loads(INCIDENT_PATH.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "TERMINAL_SCORE_FREE_QUALIFICATION_FAILED_NO_IDENTITY_REUSE"
    assert value["terminal_identity_reuse_authorized"] is False
    assert value["fresh_successor_required"] is True
    assert value["scored_calls"] == 0
    assert (
        value["deterministic_source_diagnosis"]["failure_boundary"]
        == "pre_request_controller_resource_sample"
    )
    assert value["deterministic_source_diagnosis"]["producer_failure"] == (
        "hard_coded_cgroup_root_memory_events_absent"
    )
    assert value["qualifier_result"]["observed_model_request_count"] == 0
    assert value["qualifier_result"]["highest_passing_concurrency"] == 0
    assert value["terminal_reconciliation"]["kubernetes_owner_closure_count"] == 0
    assert value["terminal_reconciliation"]["qualifier_job_and_configmap_absent"] is True
    assert value["prompts_traces_flags_or_scores_included"] is False


def test_fresh_payload_declares_validated_observer_pythonpath() -> None:
    payload = held.jobs_payload(ROOT)
    postcreate.lifecycle_guard.validate_observer_environment(
        payload["env"], held.RUNTIME_DEPENDENCY_DIR
    )
    assert payload["env"]["PYTHONPATH"] == "/tmp"


def test_build_binding_follows_exact_uid_owner_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    submission = _submission()
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    value = postcreate.build_binding(
        {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
        submission,
        _objects(),
        ROOT,
    )
    assert value["rayjob_uid"] == "11111111-1111-4111-8111-111111111111"
    assert value["ray_cluster_name"] == "ft-run-example-abcd1"
    assert value["ray_cluster_uid"] == "22222222-2222-4222-8222-222222222222"
    assert value["service_name"] == "ft-run-example-abcd1-head-svc"
    assert value["head_pod_name"] == "ft-run-example-abcd1-head-abcde"
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")


@pytest.mark.parametrize("kind", ["Workload", "RayCluster", "Service", "Pod"])
def test_build_binding_rejects_missing_or_ambiguous_uid_chain(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    items = _objects()
    missing = [row for row in items if row["kind"] != kind]
    with pytest.raises(RuntimeError):
        postcreate.build_binding(
            {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
            _submission(),
            missing,
            ROOT,
        )
    duplicate = copy.deepcopy(items)
    duplicate.append(copy.deepcopy(next(row for row in items if row["kind"] == kind)))
    with pytest.raises(RuntimeError):
        postcreate.build_binding(
            {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
            _submission(),
            duplicate,
            ROOT,
        )


def test_postcreate_release_is_score_free_gap_bound_and_exact() -> None:
    server_release = {"receipt_sha256": "sha256:" + "a" * 64}
    source_commit = "b" * 40
    value = {
        "schema_version": postcreate.POSTCREATE_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_GAP_FREE_SCORE_FREE_DP6_K_SEQUENCE",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "server_release_receipt_sha256": server_release["receipt_sha256"],
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "qualifier_job": postcreate.qualifier.JOB_NAME,
        "qualifier_output_root": postcreate.qualifier.OUTPUT_ROOT,
        "server_create_limit": 1,
        "qualifier_create_limit": 1,
        "idle_release_seconds": 600,
        "gap_free_monitor_required": True,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _receipt(value)
    postcreate.validate_release(value, server_release, source_commit)
    for field, replacement in (
        ("scoring_authorized", True),
        ("idle_release_seconds", 601),
        ("gap_free_monitor_required", False),
    ):
        changed = {**value, field: replacement}
        _receipt(changed)
        with pytest.raises(ValueError):
            postcreate.validate_release(changed, server_release, source_commit)


def test_monitor_releases_immediately_at_terminal_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postcreate,
        "_kubectl_json",
        lambda *_args: {
            "metadata": {"uid": "66666666-6666-4666-8666-666666666666"},
            "status": {"succeeded": 1},
        },
    )
    monkeypatch.setattr(
        postcreate,
        "_release_server",
        lambda *_args: {
            "get_before": 200,
            "delete": 204,
            "get_after": 404,
            "rayjob_workload_raycluster_pod_service_absent": True,
        },
    )
    monkeypatch.setattr(
        postcreate,
        "_stop_qualifier",
        lambda *_args: {"job_known": True, "delete": "terminal_no_action"},
    )
    monkeypatch.setattr(
        postcreate,
        "_sfs_json",
        lambda _path, _observer=None: {
            "status": "PASSED",
            "receipt_sha256": "sha256:" + "a" * 64,
        },
    )
    monkeypatch.setattr(
        postcreate, "_stable_sfs_observer", lambda _uid: ("observer", "uid", "/shared")
    )
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    monkeypatch.setattr(runtime, "qualification_plan", lambda *_args: {})
    monkeypatch.setattr(runtime, "validate_result", lambda *_args: 6)
    binding = {
        "rayjob_uid": "1",
        "workload_uid": "2",
        "ray_cluster_uid": "3",
        "head_pod_uid": "4",
        "service_uid": "5",
        "head_pod_name": "pod",
        "service_origin": "http://service:8000",
    }
    receipt = postcreate.monitor_and_release(
        object(),
        "ft-run-example",
        {},
        binding,
        "77777777-7777-4777-8777-777777777777",
        "66666666-6666-4666-8666-666666666666",
        ROOT,
    )
    assert receipt["qualifier_phase"] == "Succeeded"
    assert receipt["jobs_api_release"]["delete"] == 204
    assert receipt["jobs_api_release"]["rayjob_workload_raycluster_pod_service_absent"] is True
    assert receipt["scored_calls"] == 0
    assert receipt["terminal_sfs_observer"] == {
        "pod_name": "observer",
        "pod_uid": "uid",
        "sfs_mount_path": "/shared",
        "excluded_target_head_pod_uid": "4",
    }


def test_counter_producer_gate_binds_ready_receipts_before_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {
        "api_run_id": "ft-run-example",
        "head_pod_name": "server-head",
        "head_pod_uid": "11111111-1111-4111-8111-111111111111",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "receipt_sha256": "sha256:" + "a" * 64,
    }
    baseline = observer.baseline_observation(
        7,
        server_run_dir=held.RUN_DIR,
        pod_name=binding["head_pod_name"],
        pod_uid=binding["head_pod_uid"],
        api_run_id=binding["api_run_id"],
        service_uid=binding["service_uid"],
        server_binding_receipt_sha256=binding["receipt_sha256"],
        observed_at_epoch=1,
    )
    values = {
        "REQUEST-COUNTER-BASELINE.json": baseline,
        ".request-counters.json": {observer.STATE_KEY: 7},
        "OBSERVER-STATUS.json": observer.observer_status(
            "STABLE_BOUND_COUNTER_BASELINE", "stable_baseline", observed_at_epoch=1
        ),
    }
    monkeypatch.setattr(
        postcreate,
        "_stable_sfs_observer",
        lambda _uid: ("observer", "observer-uid", "/shared"),
    )
    monkeypatch.setattr(
        postcreate,
        "_bound_sfs_json",
        lambda _observer, path: copy.deepcopy(values[path.rsplit("/", 1)[-1]]),
    )
    projection = {"projected_at_epoch": 0, "pre_projection_observer_status_sha256": None}
    value = postcreate._counter_producer_gate(binding, projection)  # noqa: SLF001
    assert value["status"] == "READY_BOUND_STABLE_COUNTER_PRODUCER"
    assert value["observer_pod_name"] == "observer"
    assert value["observer_pod_uid"] == "observer-uid"
    assert value["observer_sfs_mount_path"] == "/shared"
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")


def test_binding_projection_uses_uid_bound_materialized_path_and_exact_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _receipt(
        {
            "api_run_id": "ft-run-example",
            "head_pod_uid": "11111111-1111-4111-8111-111111111111",
        }
    )
    observer_tuple = (
        "allie-dev",
        "22222222-2222-4222-8222-222222222222",
        "/shared",
    )
    monkeypatch.setattr(postcreate, "_stable_sfs_observer", lambda _uid: observer_tuple)
    monkeypatch.setattr(postcreate, "_validated_observer", lambda value: value)
    prior = observer.observer_status(
        "FAILED", "binding", observed_at_epoch=1, error_type="FileNotFoundError"
    )
    monkeypatch.setattr(postcreate, "_bound_sfs_json", lambda *_args: prior)
    payload = self_hosted.canonical_json(binding) + b"\n"
    monkeypatch.setattr(postcreate, "_bound_sfs_bytes", lambda *_args: payload)
    calls: list[list[str]] = []

    def run(args, **_kwargs):
        calls.append(args)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(postcreate.subprocess, "run", run)
    value = postcreate._project_binding_once(binding)  # noqa: SLF001
    assert len(calls) == 1
    assert (
        "/shared/jobs/chris-cyber-evalserve-q38-dp6-k-v1/lifecycle/SERVER-BINDING.json"
        in calls[0]
    )
    assert "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-k-v1" not in calls[0]
    encoded = calls[0][-1]
    assert postcreate.base64.b64decode(encoded) == self_hosted.canonical_json(binding) + b"\n"
    assert value["status"] == "PROJECTED_EXACT_SERVER_BINDING_CREATE_ONCE"
    assert value["observer_pod_uid"] == observer_tuple[1]
    assert value["pre_projection_observer_status_sha256"] == prior["receipt_sha256"]
    assert value["payload_sha256"] == self_hosted.sha256(payload)
    assert value["readback_sha256"] == self_hosted.sha256(payload)
    assert value["readback_exact"] is True


def test_binding_projection_collision_fails_before_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {"head_pod_uid": "11111111-1111-4111-8111-111111111111"}
    monkeypatch.setattr(
        postcreate,
        "_stable_sfs_observer",
        lambda _uid: ("observer", "22222222-2222-4222-8222-222222222222", "/shared"),
    )
    monkeypatch.setattr(postcreate, "_validated_observer", lambda value: value)
    monkeypatch.setattr(
        postcreate,
        "_bound_sfs_json",
        lambda *_args: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cat")),
    )
    monkeypatch.setattr(
        postcreate.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "project")
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        postcreate._project_binding_once(binding)  # noqa: SLF001


def test_binding_projection_rejects_nonidentical_byte_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _receipt(
        {
            "api_run_id": "ft-run-example",
            "head_pod_uid": "11111111-1111-4111-8111-111111111111",
        }
    )
    monkeypatch.setattr(
        postcreate,
        "_stable_sfs_observer",
        lambda _uid: ("observer", "22222222-2222-4222-8222-222222222222", "/shared"),
    )
    monkeypatch.setattr(postcreate, "_validated_observer", lambda value: value)
    monkeypatch.setattr(
        postcreate,
        "_bound_sfs_json",
        lambda *_args: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cat")),
    )
    monkeypatch.setattr(
        postcreate,
        "_bound_sfs_bytes",
        lambda *_args: self_hosted.canonical_json(binding),
    )
    monkeypatch.setattr(
        postcreate.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    with pytest.raises(RuntimeError, match="byte readback drifted"):
        postcreate._project_binding_once(binding)  # noqa: SLF001


def test_counter_producer_fresh_failed_status_aborts_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = observer.observer_status(
        "FAILED", "metric_schema_warmup", observed_at_epoch=11, error_type="RuntimeError"
    )
    monkeypatch.setattr(
        postcreate,
        "_stable_sfs_observer",
        lambda _uid: ("observer", "observer-uid", "/shared"),
    )
    monkeypatch.setattr(postcreate, "_bound_sfs_json", lambda *_args: copy.deepcopy(failed))
    monkeypatch.setattr(
        postcreate.time,
        "sleep",
        lambda *_args: pytest.fail("fresh FAILED status must not poll"),
    )
    with pytest.raises(RuntimeError, match="producer failed: RuntimeError"):
        postcreate._counter_producer_gate(  # noqa: SLF001
            {"head_pod_uid": "server-uid"},
            {
                "projected_at_epoch": 10,
                "pre_projection_observer_status_sha256": "sha256:" + "f" * 64,
            },
        )


def test_counter_producer_deadline_covers_server_pre_ready_and_stabilization() -> None:
    assert held.SERVER_PRE_READY_TIMEOUT_SECONDS == 600
    assert held.COUNTER_PRODUCER_STARTUP_GRACE_SECONDS == 120
    assert postcreate.COUNTER_PRODUCER_TIMEOUT_SECONDS == 720
    assert postcreate.COUNTER_PRODUCER_TIMEOUT_SECONDS > held.SERVER_PRE_READY_TIMEOUT_SECONDS


def test_bound_sfs_reader_translates_canonical_path_to_observer_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        postcreate,
        "_kubectl_json",
        lambda *_args: {
            "metadata": {"name": "observer", "uid": "observer-uid"},
            "kind": "Pod",
            "spec": {
                "volumes": [
                    {
                        "name": "shared",
                        "persistentVolumeClaim": {"claimName": "sfs-shared"},
                    }
                ],
                "containers": [
                    {
                        "name": "observer",
                        "volumeMounts": [{"name": "shared", "mountPath": "/shared"}],
                    }
                ],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 0}],
            },
        },
    )

    def run(args, **_kwargs):
        observed.extend(args)
        return type("Completed", (), {"stdout": b"{}"})()

    monkeypatch.setattr(postcreate.subprocess, "run", run)
    value = postcreate._bound_sfs_json(  # noqa: SLF001
        ("observer", "observer-uid", "/shared"),
        "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-k-v1/lifecycle/STATUS.json",
    )
    assert value == {}
    assert "/shared/jobs/chris-cyber-evalserve-q38-dp6-k-v1/lifecycle/STATUS.json" in observed
    assert (
        "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-k-v1/lifecycle/STATUS.json"
        not in observed
    )


def test_bound_sfs_reader_rejects_observer_restart_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postcreate,
        "_kubectl_json",
        lambda *_args: {
            "metadata": {"name": "observer", "uid": "observer-uid"},
            "kind": "Pod",
            "spec": {
                "volumes": [
                    {
                        "name": "shared",
                        "persistentVolumeClaim": {"claimName": "sfs-shared"},
                    }
                ],
                "containers": [
                    {
                        "name": "observer",
                        "volumeMounts": [{"name": "shared", "mountPath": "/shared"}],
                    }
                ],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 1}],
            },
        },
    )
    monkeypatch.setattr(
        postcreate.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("drifted observer must not be read"),
    )
    with pytest.raises(RuntimeError, match="stable non-target SFS observer"):
        postcreate._bound_sfs_json(  # noqa: SLF001
            ("observer", "observer-uid", "/shared"),
            "/mnt/sfs/jobs/example/receipt.json",
        )


def test_qualifier_stop_is_uid_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"metadata":{"uid":"different"}}'

    monkeypatch.setattr(postcreate.subprocess, "run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(RuntimeError, match="failed cleanup"):
        postcreate._stop_qualifier(  # noqa: SLF001
            "66666666-6666-4666-8666-666666666666", None
        )


def test_server_release_requires_api404_and_bound_uid_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class Client:
        def __init__(self) -> None:
            self.gets = 0

        def get(self, _path: str) -> Response:
            self.gets += 1
            return Response(200 if self.gets == 1 else 404)

        def delete(self, _path: str) -> Response:
            return Response(204)

    monkeypatch.setattr(postcreate, "_inventory", lambda: [])
    binding = {
        "rayjob_uid": "1",
        "workload_uid": "2",
        "ray_cluster_uid": "3",
        "head_pod_uid": "4",
        "service_uid": "5",
    }
    value = postcreate._release_server(Client(), "ft-run-example", binding)  # noqa: SLF001
    assert value == {
        "get_before": 200,
        "delete": 204,
        "get_after": 404,
        "rayjob_workload_raycluster_pod_service_absent": True,
    }


def test_server_release_rejects_suffixed_object_without_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class Client:
        def __init__(self) -> None:
            self.gets = 0

        def get(self, _path: str) -> Response:
            self.gets += 1
            return Response(200 if self.gets == 1 else 404)

        def delete(self, _path: str) -> Response:
            return Response(204)

    suffix = {
        "kind": "RayCluster",
        "metadata": {
            "name": "ft-run-example-abcd1",
            "uid": "22222222-2222-4222-8222-222222222222",
        },
    }
    monkeypatch.setattr(postcreate, "_inventory", lambda: [suffix])
    moments = iter([0.0, 0.0, 999.0])
    monkeypatch.setattr(postcreate.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(postcreate.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="remained after release"):
        postcreate._release_server(Client(), "ft-run-example", None)  # noqa: SLF001


def test_uncertain_create_recovers_exact_target_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postcreate.shared,
        "_runs",
        lambda _client: [
            {
                "name": "ft-run-recovered",
                "title": held.TITLE,
                "run_dir": held.RUN_DIR,
            }
        ],
    )
    assert postcreate._recover_created_api_run(object()) == "ft-run-recovered"  # noqa: SLF001


def test_qualifier_cleanup_attempts_configmap_after_job_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def remove(kind: str, _name: str, _uid: str | None) -> dict:
        called.append(kind)
        if kind == "job":
            raise RuntimeError("job cleanup failed")
        return {"known": True, "delete": "completed"}

    monkeypatch.setattr(postcreate, "_delete_qualifier_object", remove)
    with pytest.raises(RuntimeError, match="failed cleanup"):
        postcreate._stop_qualifier(None, None)  # noqa: SLF001
    assert called == ["job", "configmap"]


def test_server_cleanup_error_does_not_skip_qualifier_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def release(*_args):
        called.append("server")
        raise RuntimeError("server cleanup failed")

    def stop(*_args):
        called.append("qualifier")
        return {"job_and_configmap_absent": True}

    monkeypatch.setattr(postcreate, "_release_server", release)
    monkeypatch.setattr(postcreate, "_stop_qualifier", stop)
    api_run_id, _released, stopped, errors = postcreate._failure_cleanup(  # noqa: SLF001
        object(),
        "ft-run-example",
        None,
        {},
        server_create_started=True,
        qualifier_create_started=True,
    )
    assert api_run_id == "ft-run-example"
    assert called == ["server", "qualifier"]
    assert stopped == {"job_and_configmap_absent": True}
    assert errors == {
        "create_reconciliation": None,
        "server": "RuntimeError",
        "qualifier": None,
    }


def test_partial_qualifier_create_retains_configmap_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = {
        "configmap": {"kind": "ConfigMap", "metadata": {"name": "cm"}},
        "job": {"kind": "Job", "metadata": {"name": "job"}},
    }
    monkeypatch.setattr(postcreate.qualifier, "render", lambda *_args: rendered)
    calls = 0

    def run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return type(
                "Completed",
                (),
                {
                    "stdout": json.dumps(
                        {
                            "kind": "ConfigMap",
                            "metadata": {
                                "uid": "77777777-7777-4777-8777-777777777777"
                            },
                        }
                    )
                },
            )()
        raise RuntimeError("Job create failed")

    monkeypatch.setattr(postcreate.subprocess, "run", run)
    created: dict[str, str] = {}
    with pytest.raises(RuntimeError, match="Job create failed"):
        postcreate.create_qualifier({}, {}, {}, ROOT, created)
    assert created == {"ConfigMap": "77777777-7777-4777-8777-777777777777"}
