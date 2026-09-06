import copy
import datetime
import inspect
import json
import subprocess
import urllib.error
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as server
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_release
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0
PACKAGE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    text=True,
    capture_output=True,
).stdout.strip()


def _binding() -> dict:
    api_run_id = "ft-run-deadbeef"
    service_name = "ft-run-deadbeef-abcde-head-svc"
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": api_run_id,
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": f"http://{service_name}.fleet-train-jobs.svc:8000",
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


def _application_ready() -> dict:
    ready = NOW - 15
    value = {
        "schema_version": "fleet-glm53-dedicated-v24-application-ready-v1",
        "status": "APPLICATION_HEALTH_HTTP_200",
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
        "ready_at_epoch": ready,
        "ready_at_utc": datetime.datetime.fromtimestamp(
            ready, datetime.UTC
        ).isoformat().replace("+00:00", "Z"),
        "health_http_status": 200,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _live(binding: dict | None = None) -> dict:
    binding = binding or _binding()
    value = {
        "schema_version": live_release.LIVE_STATE_SCHEMA,
        "status": "READY_POST_CREATE_UID_BOUND",
        "observed_at_epoch": NOW - 5,
        "ready_at_epoch": NOW - 15,
        "server_binding": binding,
        "request_sha256": crypto.sha256(crypto.canonical_json(server.payload())),
        "api_get_http_status": 200,
        "api_run_id_match_count": 1,
        "api_run_state": "RUNNING",
        "api_title_match_count": 1,
        "api_run_dir_match_count": 1,
        "rayjob_running": True,
        "rayjob_name": binding["api_run_id"],
        "rayjob_uid_match_count": 1,
        "raycluster_name": "ft-run-deadbeef-abcde",
        "raycluster_uid": "66666666-6666-4666-8666-666666666666",
        "workload_admitted": True,
        "workload_name": "rayjob-workload-deadbeef",
        "workload_finished": False,
        "workload_preemption_events": 0,
        "workload_uid_match_count": 1,
        "head_pod_phase": "Running",
        "head_pod_name": "ft-run-deadbeef-abcde-head-xyz12",
        "head_pod_ready": True,
        "head_pod_restarts": 0,
        "head_pod_uid_match_count": 1,
        "service_present": True,
        "service_name": "ft-run-deadbeef-abcde-head-svc",
        "service_uid_match_count": 1,
        "sfs_pvc_name": "sfs-claim",
        "sfs_pvc_uid": "77777777-7777-4777-8777-777777777777",
        "head_pod_sfs_mount_path": "/mnt/sfs",
        "metrics_http_status": 200,
        "activity_metric_families": list(runtime.ACTIVITY_METRICS),
        "jobs_api_credential_secret_name": "ft-run-deadbeef-fleet-key",
        "jobs_api_credential_secret_uid": "55555555-5555-4555-8555-555555555555",
        "jobs_api_credential_owner_rayjob_uid": binding["rayjob_uid"],
        "jobs_api_credential_key": "FLEET_API_KEY",
        "jobs_api_credential_probe_http_status": 200,
        "watchdog_job_match_count": 0,
        "watchdog_configmap_match_count": 0,
        "server_run_dir_exists": True,
        "watchdog_result_root_absent": True,
        "application_ready_receipt_sha256": _application_ready()["receipt_sha256"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _rehash(value: dict) -> dict:
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _package_identity(binding: dict | None = None) -> tuple[str, str]:
    binding = binding or _binding()
    held = package.render(
        ROOT,
        PACKAGE_COMMIT,
        binding,
        ready_at_epoch=NOW - 15,
        priority_classes=_priority_classes(),
    )
    package_json = json.loads(held["objects"]["items"][0]["data"]["package.json"])
    implementation = package_json["files"][
        "evals/fleet/glm53_dedicated_v23_request_counter_watchdog_v1.py"
    ]
    return package_json["package_sha256"], implementation


def _release(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    binding = _binding()
    live = _live(binding)
    package_sha, implementation_sha = _package_identity(binding)
    release = live_release.build_release(
        live,
        binding,
        watchdog_package_commit=PACKAGE_COMMIT,
        watchdog_package_sha256=package_sha,
        watchdog_implementation_sha256=implementation_sha,
    )
    return binding, live, release


def test_public_renderer_and_validators_have_no_caller_controlled_clock() -> None:
    for function in (
        live_release.validate_live_state,
        live_release.build_release,
        live_release.validate_release,
        live_release.render,
    ):
        assert "now_epoch" not in inspect.signature(function).parameters


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("status", "HELD"),
        ("request_sha256", "sha256:" + "0" * 64),
        ("head_pod_sfs_mount_path", "/tmp"),
        ("api_get_http_status", 503),
        ("api_run_id_match_count", 0),
        ("api_run_state", "FAILED"),
        ("api_title_match_count", True),
        ("api_title_match_count", 2),
        ("api_run_dir_match_count", 0),
        ("rayjob_running", False),
        ("rayjob_name", "wrong"),
        ("rayjob_uid_match_count", 0),
        ("raycluster_name", ""),
        ("raycluster_uid", "not-a-uid"),
        ("workload_admitted", False),
        ("workload_finished", True),
        ("workload_preemption_events", 1),
        ("workload_uid_match_count", 2),
        ("head_pod_phase", "Pending"),
        ("head_pod_ready", False),
        ("head_pod_restarts", 1),
        ("head_pod_uid_match_count", 0),
        ("service_present", False),
        ("service_uid_match_count", 2),
        ("sfs_pvc_name", ""),
        ("sfs_pvc_uid", "not-a-uid"),
        ("head_pod_sfs_mount_path", "/tmp"),
        ("metrics_http_status", 500),
        ("activity_metric_families", ["sglang:num_requests_total"]),
        ("jobs_api_credential_secret_name", "github-token"),
        ("jobs_api_credential_secret_uid", "not-a-uid"),
        ("jobs_api_credential_key", "GH_TOKEN"),
        ("jobs_api_credential_probe_http_status", 403),
        ("watchdog_job_match_count", 2),
        ("watchdog_configmap_match_count", 3),
        ("server_run_dir_exists", False),
        ("watchdog_result_root_absent", "false"),
        ("fleet_task_instance_calls", 1),
        ("fleet_session_calls", 1),
        ("verifier_calls", 1),
        ("scoring_calls", 1),
        ("protected_content_included", True),
    ],
)
def test_live_state_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object
) -> None:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    value = _live()
    value[field] = bad
    _rehash(value)
    with pytest.raises(live_release.LiveReleaseError, match="live_state"):
        live_release.validate_live_state(value, _binding())


def test_live_state_rejects_extra_missing_stale_future_old_ready_and_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    values = []
    extra = _live()
    extra["unexpected"] = False
    values.append(_rehash(extra))
    missing = _live()
    del missing["service_present"]
    values.append(_rehash(missing))
    for observed, ready in (
        (NOW - 61, NOW - 62),
        (NOW + 1, NOW - 15),
        (NOW - 5, NOW - 126),
        (float("nan"), NOW - 15),
    ):
        value = _live()
        value["observed_at_epoch"] = observed
        value["ready_at_epoch"] = ready
        values.append(_rehash(value))
    for value in values:
        with pytest.raises(live_release.LiveReleaseError, match="live_state"):
            live_release.validate_live_state(value, _binding())


def test_live_renderer_mounts_exact_authorization_and_fleet_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    binding = _binding()
    rendered = live_release.render(
        ROOT,
        PACKAGE_COMMIT,
        binding,
        _live(binding),
        priority_classes=_priority_classes(),
    )
    configmap, authorization, job = rendered["objects"]["items"]
    release = rendered["live_release"]
    assert configmap["metadata"]["name"] == package.JOB_NAME + "-package"
    assert authorization["metadata"]["name"] == live_release.AUTHORIZATION_CONFIGMAP_NAME
    assert authorization["immutable"] is True
    assert json.loads(authorization["data"]["LIVE_RELEASE.json"]) == release
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row for row in container["env"]}
    assert "GH_TOKEN" not in env
    assert env["FLEET_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "ft-run-deadbeef-fleet-key",
        "key": "FLEET_API_KEY",
    }
    assert env["WATCHDOG_API_RUN_ID"]["value"] == "ft-run-deadbeef"
    assert "/authorization/LIVE_RELEASE.json" in container["command"][-1]
    assert {row["name"] for row in container["volumeMounts"]} >= {"authorization"}
    runtime.validate_launch_authorization(
        release,
        binding,
        ready_at_epoch=release["ready_at_epoch"],
        package_commit=env["WATCHDOG_PACKAGE_COMMIT"]["value"],
        package_sha256=env["WATCHDOG_PACKAGE_SHA256"]["value"],
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("status", "HELD"),
        ("server_binding_sha256", "sha256:" + "0" * 64),
        ("live_state_receipt_sha256", "sha256:" + "0" * 64),
        ("request_sha256", "sha256:" + "0" * 64),
        ("raycluster_name", "wrong"),
        ("raycluster_uid", "66666666-6666-4666-8666-000000000000"),
        ("sfs_pvc_name", "wrong"),
        ("sfs_pvc_uid", "77777777-7777-4777-8777-000000000000"),
        ("head_pod_sfs_mount_path", "/tmp"),
        ("application_ready_receipt_sha256", "sha256:" + "0" * 64),
        ("watchdog_job_name", "wrong"),
        ("watchdog_result_root", "/wrong"),
        ("watchdog_package_commit", "0" * 40),
        ("watchdog_package_sha256", "sha256:" + "0" * 64),
        ("watchdog_implementation_sha256", "sha256:" + "0" * 64),
        ("ready_at_epoch", NOW),
        ("idle_release_seconds", 601),
        ("create_once", False),
        ("server_launch_authorized", True),
        ("watchdog_launch_authorized", False),
        ("qualification_launch_authorized", True),
        ("scored_launch_authorized", True),
        ("protected_content_included", True),
    ],
)
def test_rehashed_live_release_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object
) -> None:
    binding, live, release = _release(monkeypatch)
    package_sha, implementation_sha = _package_identity(binding)
    mutated = copy.deepcopy(release)
    mutated[field] = bad
    _rehash(mutated)
    with pytest.raises(live_release.LiveReleaseError, match="live_release"):
        live_release.validate_release(
            mutated,
            live,
            binding,
            watchdog_package_commit=PACKAGE_COMMIT,
            watchdog_package_sha256=package_sha,
            watchdog_implementation_sha256=implementation_sha,
        )


def test_runtime_rejects_rehashed_wrong_authorization_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, _, release = _release(monkeypatch)
    package_sha, _ = _package_identity(binding)
    for field, bad in (
        ("watchdog_package_sha256", "sha256:" + "0" * 64),
        ("watchdog_implementation_sha256", "sha256:" + "0" * 64),
        ("fleet_session_calls", 1),
        ("ready_at_epoch", float("nan")),
    ):
        mutated = copy.deepcopy(release)
        mutated[field] = bad
        _rehash(mutated)
        with pytest.raises(runtime.WatchdogError, match="authorization"):
            runtime.validate_launch_authorization(
                mutated,
                binding,
                ready_at_epoch=release["ready_at_epoch"],
                package_commit=PACKAGE_COMMIT,
                package_sha256=package_sha,
            )


def _kubernetes_fixture() -> dict[tuple[str, str | None], dict]:
    binding = _binding()
    rayjob_uid = binding["rayjob_uid"]
    cluster_uid = "66666666-6666-4666-8666-666666666666"
    cluster_name = "ft-run-deadbeef-abcde"
    pod_name = "ft-run-deadbeef-abcde-head-xyz12"
    service_name = "ft-run-deadbeef-abcde-head-svc"
    return {
        ("rayjobs.ray.io", "ft-run-deadbeef"): {
            "metadata": {"name": "ft-run-deadbeef", "uid": rayjob_uid},
            "status": {
                "jobDeploymentStatus": "Running",
                "rayClusterName": cluster_name,
                "rayClusterStatus": {
                    "head": {"podName": pod_name, "serviceName": service_name}
                },
            },
        },
        ("workloads.kueue.x-k8s.io", None): {
            "items": [
                {
                    "metadata": {
                        "name": "rayjob-workload-deadbeef",
                        "uid": binding["workload_uid"],
                        "ownerReferences": [{"kind": "RayJob", "uid": rayjob_uid}],
                    },
                    "status": {"conditions": [{"type": "Admitted", "status": "True"}]},
                }
            ]
        },
        ("rayclusters.ray.io", cluster_name): {
            "metadata": {
                "name": cluster_name,
                "uid": cluster_uid,
                "ownerReferences": [{"kind": "RayJob", "uid": rayjob_uid}],
            }
        },
        ("pods", pod_name): {
            "metadata": {
                "name": pod_name,
                "uid": binding["head_pod_uid"],
                "labels": {"ray.io/node-type": "head"},
                "ownerReferences": [{"kind": "RayCluster", "uid": cluster_uid}],
            },
            "spec": {
                "containers": [
                    {
                        "name": "ray-head",
                        "volumeMounts": [{"name": "shared-sfs", "mountPath": "/mnt/sfs"}],
                    }
                ],
                "volumes": [
                    {
                        "name": "shared-sfs",
                        "persistentVolumeClaim": {"claimName": "sfs-claim"},
                    }
                ],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [
                    {"name": "ray-head", "ready": True, "restartCount": 0}
                ],
            },
        },
        ("services", service_name): {
            "metadata": {
                "uid": binding["service_uid"],
                "ownerReferences": [{"kind": "RayCluster", "uid": cluster_uid}],
            },
            "spec": {"selector": {"ray.io/node-type": "head"}},
        },
        ("persistentvolumeclaims", "sfs-claim"): {
            "metadata": {
                "name": "sfs-claim",
                "uid": "77777777-7777-4777-8777-777777777777",
            }
        },
    }


def test_secret_observer_requests_only_metadata_and_key_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "55555555-5555-4555-8555-555555555555\n"
            "RayJob\t11111111-1111-4111-8111-111111111111\n"
            "FLEET_API_KEY\n",
            "",
        )

    monkeypatch.setattr(live_release.subprocess, "run", run)
    value = live_release._kubectl_secret_metadata("ft-run-deadbeef-fleet-key")
    assert value["keys"] == ["FLEET_API_KEY"]
    assert commands[0][-1].startswith("go-template=")
    assert ".data}}{{$key}}" in commands[0][-1]
    assert "-o" in commands[0]
    assert "json" not in commands[0]


def test_observer_binds_randomized_object_names_without_secret_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = _kubernetes_fixture()
    seen_sources: list[str] = []
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    monkeypatch.setattr(
        live_release,
        "_kubectl_json",
        lambda kind, name=None: copy.deepcopy(objects[(kind, name)]),
    )
    monkeypatch.setattr(live_release, "_kubectl_optional", lambda kind, name: None)
    monkeypatch.setattr(
        live_release,
        "_kubectl_secret_metadata",
        lambda _name: {
            "uid": "55555555-5555-4555-8555-555555555555",
            "ownerReferences": [
                {"kind": "RayJob", "uid": _binding()["rayjob_uid"]}
            ],
            "keys": ["FLEET_API_KEY"],
        },
    )

    def pod_python(_pod: str, source: str) -> dict:
        seen_sources.append(source)
        if "/v1/runs/" in source:
            return {
                "http_status": 200,
                "api_run_id": "ft-run-deadbeef",
                "title": server.TITLE,
                "run_dir": server.RUN_DIR,
                "state": "RUNNING",
            }
        if "/metrics" in source:
            return {"http_status": 200, "families": list(runtime.ACTIVITY_METRICS)}
        return {
            "server_run_dir_exists": True,
            "watchdog_result_root_absent": True,
            "application_ready": _application_ready(),
        }

    monkeypatch.setattr(live_release, "_pod_python", pod_python)
    binding, live = live_release.observe_live("ft-run-deadbeef")
    assert binding == _binding()
    assert live["jobs_api_credential_secret_name"] == "ft-run-deadbeef-fleet-key"
    assert all("FLEET_API_KEY" not in source or "os.environ" in source for source in seen_sources)


def test_application_ready_requires_matching_real_timestamp_and_digest() -> None:
    live_release._validate_application_ready(_application_ready())
    for field, bad in (
        ("ready_at_utc", "2020-01-01T00:00:00Z"),
        ("health_http_status", 503),
        ("prompts_traces_flags_scores_or_model_outputs_included", True),
    ):
        value = _application_ready()
        value[field] = bad
        _rehash(value)
        with pytest.raises(live_release.LiveReleaseError, match="application_ready"):
            live_release._validate_application_ready(value)


def test_observation_failure_releases_the_exact_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_release,
        "_observe_live",
        lambda _run: (_ for _ in ()).throw(live_release.LiveReleaseError("bad")),
    )
    monkeypatch.setattr(
        live_release,
        "_kubectl_optional",
        lambda kind, _name: (
            {
                "status": {
                    "rayClusterStatus": {"head": {"podName": "exact-head-pod"}}
                }
            }
            if kind == "rayjobs.ray.io"
            else None
        ),
    )
    released: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        live_release,
        "_release_on_handoff_failure",
        lambda binding, pod: released.append((binding, pod)),
    )
    with pytest.raises(live_release.LiveReleaseError, match="bad"):
        live_release.observe_live("ft-run-deadbeef")
    assert released == [({"api_run_id": "ft-run-deadbeef"}, "exact-head-pod")]


def test_exact_partial_objects_are_adopted_without_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    binding = _binding()
    rendered = live_release.render(
        ROOT,
        PACKAGE_COMMIT,
        binding,
        _live(binding),
        priority_classes=_priority_classes(),
    )
    first = copy.deepcopy(rendered["objects"]["items"][0])
    monkeypatch.setattr(
        live_release,
        "_kubectl_optional",
        lambda kind, name: (
            copy.deepcopy(first)
            if (kind, name) == (first["kind"], first["metadata"]["name"])
            else None
        ),
    )
    monkeypatch.setattr(
        live_release,
        "_delete_exact",
        lambda _value: (_ for _ in ()).throw(AssertionError("must not delete")),
    )
    outcomes = live_release._reconcile_existing_watcher_objects(
        rendered,
        binding,
        _live(binding),
        head_pod_name="head",
    )
    assert outcomes[0]["outcome"] == "ADOPTED_EXACT_PARTIAL_OR_ACTIVE"


def test_inert_different_run_partial_configmap_is_cleaned_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_release.time, "time", lambda: NOW)
    binding = _binding()
    rendered = live_release.render(
        ROOT,
        PACKAGE_COMMIT,
        binding,
        _live(binding),
        priority_classes=_priority_classes(),
    )
    stale = copy.deepcopy(rendered["objects"]["items"][0])
    prior = copy.deepcopy(binding)
    prior["api_run_id"] = "ft-run-cafebabe"
    prior["service_origin"] = (
        "http://ft-run-cafebabe-abcde-head-svc.fleet-train-jobs.svc:8000"
    )
    stale["data"]["binding.json"] = json.dumps(
        prior, sort_keys=True, separators=(",", ":")
    )
    present = {(stale["kind"], stale["metadata"]["name"]): stale}

    def optional(kind: str, name: str) -> dict | None:
        return copy.deepcopy(present.get((kind, name)))

    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(live_release, "_kubectl_optional", optional)
    monkeypatch.setattr(
        live_release,
        "_pod_python",
        lambda _pod, _source: {"http_status": 404},
    )

    def delete(value: dict) -> str:
        identity = (value["kind"], value["metadata"]["name"])
        deleted.append(identity)
        present.pop(identity)
        return "DELETED_EXACT"

    monkeypatch.setattr(live_release, "_delete_exact", delete)
    outcomes = live_release._reconcile_existing_watcher_objects(
        rendered,
        binding,
        _live(binding),
        head_pod_name="head",
    )
    assert deleted == [(stale["kind"], stale["metadata"]["name"])]
    assert outcomes[0]["outcome"] == "DELETED_EXACT"


def test_wait_requires_uid_bound_active_and_runtime_authorization_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, _, release = _release(monkeypatch)
    job_uid = "88888888-8888-4888-8888-888888888888"
    pod_uid = "99999999-9999-4999-8999-999999999999"
    job = {
        "metadata": {
            "uid": job_uid,
            "annotations": {
                "cyber-post-train.fleet.ai/live-release-receipt-sha256": release[
                    "receipt_sha256"
                ]
            },
        },
        "status": {"active": 1},
    }
    pod = {
        "metadata": {
            "uid": pod_uid,
            "ownerReferences": [{"kind": "Job", "uid": job_uid}],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": True, "restartCount": 0}],
        },
    }
    active = runtime.build_active_receipt(
        binding, watcher_job_uid=job_uid, watcher_pod_uid=pod_uid
    )
    active.update(
        {
            "initial_request_counter": 0,
            "initial_running_requests": 0,
            "initial_queued_requests": 0,
            "ready_at_epoch": release["ready_at_epoch"],
            "terminal_receipt_required": True,
        }
    )
    _rehash(active)
    authorization = runtime.build_runtime_authorization_receipt(
        release,
        binding,
        watcher_job_uid=job_uid,
        watcher_pod_uid=pod_uid,
        package_commit=release["watchdog_package_commit"],
        package_sha256=release["watchdog_package_sha256"],
    )
    monkeypatch.setattr(
        live_release,
        "_kubectl_optional",
        lambda kind, _name: copy.deepcopy(job) if kind == "jobs.batch" else None,
    )
    monkeypatch.setattr(
        live_release,
        "_kubectl_json",
        lambda kind, _name=None: {"items": [copy.deepcopy(pod)]}
        if kind == "pods"
        else {},
    )
    monkeypatch.setattr(
        live_release,
        "_pod_python",
        lambda pod_name, _source: (
            {"active": active, "authorization": authorization}
            if pod_name == "head-pod"
            else {}
        ),
    )
    value = live_release._wait_for_watchdog_active(
        binding, release, head_pod_name="head-pod", attempts=1
    )
    assert value["watchdog_job_uid"] == job_uid
    assert value["watchdog_pod_uid"] == pod_uid


def test_create_once_recovers_partial_success_and_rejects_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "x"}}
    existing: dict[str, dict | None] = {"x": copy.deepcopy(value)}
    monkeypatch.setattr(
        live_release,
        "_kubectl_optional",
        lambda _kind, name: copy.deepcopy(existing.get(name)),
    )
    assert live_release._create_or_verify_exact(value) == "PREEXISTING_EXACT"
    existing["x"] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "x"},
        "data": {"bad": "1"},
    }
    with pytest.raises(live_release.LiveReleaseError, match="collision"):
        live_release._create_or_verify_exact({**value, "data": {"wanted": "1"}})


def test_launch_releases_exact_server_if_watcher_handoff_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    live = _live(binding)
    objects = [{"kind": "ConfigMap", "metadata": {"name": "first"}}]
    monkeypatch.setattr(live_release, "observe_live", lambda _id: (binding, live))
    monkeypatch.setattr(
        live_release,
        "render",
        lambda *_args, **_kwargs: {"objects": {"items": objects}},
    )
    monkeypatch.setattr(
        live_release,
        "_create_or_verify_exact",
        lambda _value: (_ for _ in ()).throw(live_release.LiveReleaseError("boom")),
    )
    releases: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        live_release,
        "_release_on_handoff_failure",
        lambda bound, pod: releases.append((bound, pod)),
    )
    with pytest.raises(live_release.LiveReleaseError, match="boom"):
        live_release.launch(
            ROOT,
            PACKAGE_COMMIT,
            "ft-run-deadbeef",
            priority_classes=_priority_classes(),
        )
    assert releases == [(binding, live["head_pod_name"])]


def test_handoff_failure_release_uses_independent_local_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        live_release,
        "_pod_python",
        lambda _pod, _source: (_ for _ in ()).throw(
            ConnectionError("server Pod disappeared during its own deletion")
        ),
    )
    monkeypatch.setattr(live_release, "_release_local", calls.append)
    live_release._release_on_handoff_failure(_binding(), "head-pod")
    assert calls == ["ft-run-deadbeef"]


def test_local_release_confirms_api_and_kubernetes_absence_after_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    methods: list[str] = []

    class Response:
        status = 202

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def open_request(request: object, *, timeout: int) -> Response:
        assert timeout == 30
        method = request.get_method()  # type: ignore[attr-defined]
        methods.append(method)
        if method == "DELETE":
            return Response()
        raise urllib.error.HTTPError(
            request.full_url,  # type: ignore[attr-defined]
            404,
            "gone",
            {},
            None,
        )

    monkeypatch.setenv("FLEET_API_KEY", "not-persisted")
    monkeypatch.setattr(live_release.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(live_release, "_kubectl_optional", lambda _kind, _name: None)
    live_release._release_local("ft-run-deadbeef")
    assert methods == ["DELETE", "GET"]


def test_tracked_live_release_receipt_is_self_digested_and_held() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v24-watchdog-live-release-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "READY_HELD_FOR_EXACT_POST_CREATE_BINDING"
    assert value["watchdog_launch_authorized"] is False
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
