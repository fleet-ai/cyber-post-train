import copy
import inspect
import json
import subprocess
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
        "api_run_state": "RUNNING",
        "api_title_match_count": 1,
        "api_run_dir_match_count": 1,
        "rayjob_running": True,
        "rayjob_name": binding["api_run_id"],
        "rayjob_uid_match_count": 1,
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
        ("api_get_http_status", 503),
        ("api_run_state", "FAILED"),
        ("api_title_match_count", True),
        ("api_title_match_count", 2),
        ("api_run_dir_match_count", 0),
        ("rayjob_running", False),
        ("rayjob_name", "wrong"),
        ("rayjob_uid_match_count", 0),
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
        ("metrics_http_status", 500),
        ("activity_metric_families", ["sglang:num_requests_total"]),
        ("jobs_api_credential_secret_name", "github-token"),
        ("jobs_api_credential_secret_uid", "not-a-uid"),
        ("jobs_api_credential_key", "GH_TOKEN"),
        ("jobs_api_credential_probe_http_status", 403),
        ("watchdog_job_match_count", 1),
        ("watchdog_configmap_match_count", 1),
        ("server_run_dir_exists", False),
        ("watchdog_result_root_absent", False),
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
                "uid": cluster_uid,
                "ownerReferences": [{"kind": "RayJob", "uid": rayjob_uid}],
            }
        },
        ("pods", pod_name): {
            "metadata": {
                "uid": binding["head_pod_uid"],
                "labels": {"ray.io/node-type": "head"},
                "ownerReferences": [{"kind": "RayCluster", "uid": cluster_uid}],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 0}],
            },
        },
        ("services", service_name): {
            "metadata": {
                "uid": binding["service_uid"],
                "ownerReferences": [{"kind": "RayCluster", "uid": cluster_uid}],
            },
            "spec": {"selector": {"ray.io/node-type": "head"}},
        },
        ("secrets", "ft-run-deadbeef-fleet-key"): {
            "metadata": {
                "uid": "55555555-5555-4555-8555-555555555555",
                "ownerReferences": [{"kind": "RayJob", "uid": rayjob_uid}],
            },
            "data": {"FLEET_API_KEY": "redacted-never-read"},
        },
    }


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
        return {"server_run_dir_exists": True, "watchdog_result_root_absent": True}

    monkeypatch.setattr(live_release, "_pod_python", pod_python)
    binding, live = live_release.observe_live("ft-run-deadbeef")
    assert binding == _binding()
    assert live["jobs_api_credential_secret_name"] == "ft-run-deadbeef-fleet-key"
    assert all("redacted-never-read" not in source for source in seen_sources)


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


def test_handoff_failure_release_requires_api_and_kubernetes_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_release,
        "_pod_python",
        lambda _pod, _source: {"delete_status": 202, "api_absent": True},
    )
    monkeypatch.setattr(live_release, "_kubectl_optional", lambda _kind, _name: None)
    live_release._release_on_handoff_failure(_binding(), "head-pod")
    monkeypatch.setattr(
        live_release,
        "_pod_python",
        lambda _pod, _source: {"delete_status": 202, "api_absent": False},
    )
    with pytest.raises(live_release.LiveReleaseError, match="release_unconfirmed"):
        live_release._release_on_handoff_failure(_binding(), "head-pod")


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
