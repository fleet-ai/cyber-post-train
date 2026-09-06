from __future__ import annotations

import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_e_qualifier_package_v1 as package
from evals.fleet import qwen38_dp6_e_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_e_scorefree_v1 as early
from evals.fleet import qwen38_dp6_metric_observer_v4 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _receipt(value: dict[str, object]) -> dict[str, object]:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


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
            "eligible_node_uids": ["55555555-5555-4555-8555-555555555555"],
            "b300_nominal_gpu_quota": 128,
            "b300_used_gpu_quota": 64,
            "b300_gpu_quota_headroom": 64,
            "local_queue_uid": "66666666-6666-4666-8666-666666666666",
            "cluster_queue_uid": "77777777-7777-4777-8777-777777777777",
            "priority_class": {
                "name": "fleet-infra-quiet",
                "value": -1000,
                "preemption_policy": "Never",
            },
            "peer_preemption_required": False,
        },
    }


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = early.load_all(ROOT)[0]
    request_sha256 = self_hosted.sha256(self_hosted.canonical_json(early.jobs_payload(ROOT)))
    server_release_sha256 = "sha256:" + "9" * 64
    source_commit = "8" * 40
    project_shape = _project_shape()
    live_gate = _receipt(
        {
            "schema_version": runtime.LIVE_GATE_SCHEMA,
            "status": "PASSED_IMMEDIATELY_BEFORE_CREATE",
            "observed_at_utc": "2026-09-06T10:30:00Z",
            "source_commit": source_commit,
            "server_release_receipt_sha256": server_release_sha256,
            "config_sha256": config["config_sha256"],
            "request_sha256": request_sha256,
            "active_project_serving_runs": 0,
            "project_resource_shape": project_shape,
            "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
            "sfs_observation": {
                "observer_pod_name": "observer",
                "observer_pod_uid": "88888888-8888-4888-8888-888888888888",
                "run_dir_exists": False,
            },
            "rendered": early._load(ROOT / early.PREVIEW_PATH)["rendered"],  # noqa: SLF001
            "api_mutations": 0,
            "scored_calls": 0,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    submission = _receipt(
        {
            "schema_version": runtime.SUBMISSION_SCHEMA,
            "status": "SUBMITTED_SCORE_FREE_DP6_E_SERVER",
            "api_run_id": "ft-run-example",
            "title": early.TITLE,
            "run_dir": early.RUN_DIR,
            "serving_block": early.SERVING_BLOCK,
            "config_sha256": config["config_sha256"],
            "request_sha256": request_sha256,
            "server_release_receipt_sha256": server_release_sha256,
            "live_gate_receipt_sha256": live_gate["receipt_sha256"],
            "live_gate": live_gate,
            "project_resource_shape": project_shape,
            "source_commit": source_commit,
            "route": "POST /v1/runs",
            "http_status": 202,
            "server_instances_created": 1,
            "scored_calls": 0,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    binding = _receipt(
        {
            "schema_version": runtime.BINDING_SCHEMA,
            "status": "READY_NON_SCORED",
            "submission_receipt_sha256": submission["receipt_sha256"],
            "api_run_id": "ft-run-example",
            "title": early.TITLE,
            "run_dir": early.RUN_DIR,
            "serving_block": early.SERVING_BLOCK,
            "service_name": "ft-run-example-head-svc",
            "service_origin": "http://ft-run-example-head-svc.fleet-train-jobs.svc.cluster.local:8000",
            "rayjob_uid": "11111111-1111-4111-8111-111111111111",
            "workload_uid": "22222222-2222-4222-8222-222222222222",
            "head_pod_name": "ft-run-example-head",
            "head_pod_uid": "33333333-3333-4333-8333-333333333333",
            "service_uid": "44444444-4444-4444-8444-444444444444",
            "image": early.runtime.IMAGE,
            "model_revision": early.runtime.MODEL_REVISION,
            "context_length": 262144,
            "tensor_parallel_size": 1,
            "data_parallel_size": 6,
            "head_pod_running_ready": True,
            "head_pod_restarts": 0,
            "kueue_preempted": False,
            "scoring_authorized": False,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    release = _receipt(
        {
            "schema_version": package.RELEASE_SCHEMA,
            "status": "RELEASED_FOR_ONE_NON_SCORED_QUALIFIER",
            "launch_authorized": True,
            "scoring_authorized": False,
            "job_name": package.JOB_NAME,
            "configmap_name": package.CONFIGMAP_NAME,
            "output_root": package.OUTPUT_ROOT,
            "serving_block": early.SERVING_BLOCK,
            "submission_receipt_sha256": submission["receipt_sha256"],
            "server_binding_receipt_sha256": binding["receipt_sha256"],
            "package_sha256": package.package_sha256(ROOT),
            "harness_runtime_image": package.staged_image.identity(),
            "fresh_job_matches": 0,
            "fresh_configmap_matches": 0,
            "fresh_output_root_exists": False,
            "server_running_ready_restart0": True,
            "api_mutations_before_create": 0,
            "task_instance_session_verifier_scoring_calls": 0,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    return submission, binding, release


def test_archive_is_deterministic_and_contains_exact_runtime_closure(tmp_path: Path) -> None:
    first = package.archive_bytes(ROOT)
    assert first == package.archive_bytes(ROOT)
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert str(early.QUALIFIER_RUNTIME_PATH) in names
    assert str(early.OBSERVER_V1_PATH) in names
    assert str(early.OBSERVER_V2_PATH) in names
    assert str(early.OBSERVER_V3_PATH) in names
    assert str(early.OBSERVER_V4_PATH) in names
    assert str(early.LIFECYCLE_PATH) in names
    assert "evals/fleet/opencode_actual_harness_parity_v1.py" in names
    assert "evals/fleet/configs/blackbox-ctf-tool-catalog-v1.json" in names
    assert "evals/fleet/opencode_staged_image_v1.py" in names
    assert "evals/fleet/Dockerfile.opencode" not in names
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    completed = subprocess.run(
        [
            "python3",
            "-c",
            (
                "from pathlib import Path; "
                "from evals.fleet import qwen38_dp6_e_scorefree_v1 as e; "
                "e.load_all(Path('.')); print('ok')"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_renderer_is_create_once_score_free_and_needs_no_kubectl() -> None:
    submission, binding, release = _inputs()
    value = package.render(ROOT, submission, binding, release)
    configmap, job = value["configmap"], value["job"]
    assert configmap["immutable"] is True
    assert len(json.dumps(configmap).encode()) < 900_000
    assert job["metadata"]["annotations"] == {
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/scoring-authorized": "false",
    }
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["priorityClassName"] == package.QUALIFIER_PRIORITY_CLASS
    assert package.QUALIFIER_PRIORITY_VALUE == 100
    command = pod["containers"][0]["args"][0]
    assert "kubectl" not in command
    assert "qwen38_dp6_e_qualifier_runtime_v1" in command
    assert "FLEET_API_KEY" not in json.dumps(job)
    cli, dind = pod["initContainers"]
    assert cli["name"] == "docker-cli"
    assert cli["image"] == package.DIND_IMAGE
    assert package.DOCKER_CLI_SHA256 in cli["args"][0]
    assert package.DOCKER_BUILDX_SHA256 in cli["args"][0]
    assert str(package.DOCKER_CLI_TOTAL_BYTES) in cli["args"][0]
    assert cli["volumeMounts"] == [{"name": "docker-cli", "mountPath": "/cli"}]
    assert dind["name"] == "dind"
    assert dind["resources"] == {
        "requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
        "limits": {"cpu": "10", "memory": "24Gi", "ephemeral-storage": "80Gi"},
    }
    assert "dind-resource-samples.tsv" in dind["args"][0]
    dind_mounts = {row["name"]: row["mountPath"] for row in dind["volumeMounts"]}
    evaluator = pod["containers"][0]
    assert evaluator["resources"] == {
        "requests": {"cpu": "1", "memory": "2Gi", "ephemeral-storage": "10Gi"},
        "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
    }
    evaluator_mounts = {row["name"]: row["mountPath"] for row in evaluator["volumeMounts"]}
    assert dind_mounts["workspace"] == "/workspace"
    assert evaluator_mounts["workspace"] == "/workspace"
    env = {row["name"]: row["value"] for row in evaluator["env"]}
    assert env["TMPDIR"] == "/workspace/tmp"
    assert env["PATH"].startswith("/docker-cli/bin:")
    assert env["DOCKER_CONFIG"] == "/workspace/docker-config"
    assert evaluator_mounts["docker-cli"] == "/docker-cli"
    assert next(row for row in pod["volumes"] if row["name"] == "docker-cli") == {
        "name": "docker-cli",
        "emptyDir": {"sizeLimit": "256Mi"},
    }
    assert "chmod 0700 /workspace/tmp" in command
    assert 'test "$(command -v docker)" = /docker-cli/bin/docker' in command
    assert package.DOCKER_CLI_SHA256 in command
    assert package.DOCKER_BUILDX_SHA256 in command
    assert str(package.DOCKER_CLI_TOTAL_BYTES) in command
    assert "Docker version 27.5.1, build 9f9e405" in command
    assert "v0.20.1" in command
    assert "docker build" not in command
    assert "opencode_staged_image_v1" in command
    assert (
        "uv run --with httpx --with pyyaml python -m evals.fleet.opencode_staged_image_v1"
    ) in command
    assert "gzip -dc" in command
    assert package.staged_image.ARCHIVE_SHA256.removeprefix("sha256:") not in command
    assert package.staged_image.RUNTIME_IMAGE_ID in command


def test_fresh_qualifier_identity_and_failure_drain_are_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert package.JOB_NAME == "chris-cyber-q38-dp6-e-qualifier-v7"
    assert package.CONFIGMAP_NAME == package.JOB_NAME
    assert package.OUTPUT_ROOT.endswith("/chris-cyber-q38-dp6-e-qualifier-v7")
    drain = tmp_path / "lifecycle" / "DRAIN"
    drain.parent.mkdir()
    monkeypatch.setattr(runtime, "DRAIN_PATH", drain)
    first = runtime._write_drain_once("qualifier_infrastructure_failure")  # noqa: SLF001
    assert json.loads(drain.read_text()) == first
    assert runtime._write_drain_once("qualifier_infrastructure_failure") == first  # noqa: SLF001
    with pytest.raises(RuntimeError, match="different bytes"):
        runtime._write_drain_once("qualification_below_c6")  # noqa: SLF001


def test_lifecycle_releases_pre_ready_idle_and_bound_observer_failure() -> None:
    source = (ROOT / early.LIFECYCLE_PATH).read_text()
    assert "PRE_READY_TIMEOUT_SECONDS=600" in source
    assert "now - load_started_at >= PRE_READY_TIMEOUT_SECONDS" in source
    assert "fleet-qwen38-dedicated-dp6-idle-release-v1" in source
    assert "write_idle_receipt pre_ready" in source
    assert "write_idle_receipt post_ready" in source
    assert "IDLE_SECONDS=600" in source
    assert '--status-path "$OBSERVER_STATUS"' in source
    assert "observer_status != 0" in source
    assert '[[ -f "$SERVER_BINDING" && ! -L "$SERVER_BINDING" ]]' in source
    assert 'stop_server; server_pid=; exit "$observer_status"' in source


def test_release_fails_closed_on_scoring_or_identity_drift() -> None:
    submission, binding, release = _inputs()
    changed = dict(release)
    changed["scoring_authorized"] = True
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    try:
        package.render(ROOT, submission, binding, changed)
    except ValueError as exc:
        assert "not clear" in str(exc)
    else:
        raise AssertionError("scoring-enabled qualifier release was accepted")


def test_runtime_revalidates_mounted_release_and_package(tmp_path: Path) -> None:
    submission, binding, release = _inputs()
    archive = tmp_path / "package.tgz"
    archive.write_bytes(package.archive_bytes(ROOT))
    runtime.validate_runtime_release(release, submission, binding, archive)
    archive.write_bytes(archive.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="mounted early DP6 qualifier release drifted"):
        runtime.validate_runtime_release(release, submission, binding, archive)


def test_binding_requires_exact_fields_and_nonzero_workload_uid() -> None:
    submission, binding, _ = _inputs()
    variants = []
    missing = copy.deepcopy(binding)
    missing.pop("workload_uid")
    variants.append(missing)
    for invalid in ("", "not-a-uuid", "00000000-0000-0000-0000-000000000000"):
        changed = copy.deepcopy(binding)
        changed["workload_uid"] = invalid
        variants.append(changed)
    extra = copy.deepcopy(binding)
    extra["ignored_future_field"] = "unsafe"
    variants.append(extra)
    empty_pod_name = copy.deepcopy(binding)
    empty_pod_name["head_pod_name"] = ""
    variants.append(empty_pod_name)
    for changed in variants:
        _receipt(changed)
        with pytest.raises(ValueError):
            runtime.validate_binding(changed, submission, ROOT)


def test_binding_rejects_resigned_unrelated_service_and_retains_full_authority() -> None:
    submission, binding, _ = _inputs()
    aligned = runtime.validate_binding(binding, submission, ROOT)
    assert aligned["service_name"] == "ft-run-example-head-svc"
    assert aligned["service_origin"] == binding["service_origin"]
    assert aligned["workload_uid"] == binding["workload_uid"]
    assert aligned["server_binding_receipt_sha256"] == binding["receipt_sha256"]
    unrelated = copy.deepcopy(binding)
    unrelated["service_name"] = "unrelated-head-svc"
    unrelated["service_origin"] = (
        "http://unrelated-head-svc.fleet-train-jobs.svc.cluster.local:8000"
    )
    _receipt(unrelated)
    with pytest.raises(ValueError, match="Service identity"):
        runtime.validate_binding(unrelated, submission, ROOT)


def test_runtime_submission_uses_strict_live_gate_validation() -> None:
    submission, _, _ = _inputs()
    runtime.validate_submission(submission, ROOT)
    for mutate in ("project_resource_shape", "rendered"):
        changed = copy.deepcopy(submission)
        changed["live_gate"].pop(mutate)
        _receipt(changed["live_gate"])
        changed["live_gate_receipt_sha256"] = changed["live_gate"]["receipt_sha256"]
        _receipt(changed)
        with pytest.raises(ValueError):
            runtime.validate_submission(changed, ROOT)


def test_runtime_uses_cluster_dind_network_mode() -> None:
    source = (ROOT / "evals/fleet/qwen38_dp6_e_qualifier_runtime_v1.py").read_text()
    assert "cluster_dind=True" in source
    parity_source = (ROOT / "evals/fleet/opencode_actual_harness_parity_v1.py").read_text()
    assert 'bind_host = "0.0.0.0" if cluster_dind else "127.0.0.1"' in parity_source
    assert '"host.docker.internal:host-gateway"' in parity_source


def test_server_local_events_are_uid_bound_and_aggregated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission, binding, _ = _inputs()
    reduced = runtime.validate_binding(binding, submission, ROOT)
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    monkeypatch.setattr(runtime, "EVENT_DIR", event_dir)
    samples = tmp_path / "dind-resource-samples.tsv"
    samples.write_text("1 0 0 0\n")
    monkeypatch.setattr(runtime, "DIND_RESOURCE_SAMPLES_PATH", samples)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({observer.STATE_KEY: 0}))
    baseline_path = tmp_path / "baseline.json"
    baseline = observer.baseline_observation(
        0,
        server_run_dir=early.RUN_DIR,
        pod_name=str(binding["head_pod_name"]),
        pod_uid=str(binding["head_pod_uid"]),
        api_run_id=str(binding["api_run_id"]),
        service_uid=str(binding["service_uid"]),
        server_binding_receipt_sha256=str(binding["receipt_sha256"]),
        observed_at_epoch=1,
    )
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(runtime, "COUNTER_STATE_PATH", state)
    monkeypatch.setattr(runtime, "COUNTER_BASELINE_PATH", baseline_path)

    plan = {
        "receipt_sha256": "sha256:" + "a" * 64,
        "server_binding": reduced,
        "service_origin": binding["service_origin"],
    }

    def execute() -> list[dict[str, object]]:
        event = observer.traffic_observation(
            0,
            6,
            memory_mib=[250_000] * 6,
            utilization_percent=[100] * 6,
            server_run_dir=early.RUN_DIR,
            pod_name=str(binding["head_pod_name"]),
            pod_uid=str(binding["head_pod_uid"]),
            api_run_id=str(binding["api_run_id"]),
            service_uid=str(binding["service_uid"]),
            server_binding_receipt_sha256=str(binding["receipt_sha256"]),
            observed_at_epoch=1,
        )
        assert event is not None
        (event_dir / "1.json").write_text(json.dumps(event))
        samples.write_text("1 0 0 0\n2 0 0 0\n")
        return [{"receipt_sha256": f"sha256:{index:064x}"} for index in range(6)]

    rows, distribution, resource, observed_baseline = runtime.observe_wave(
        6, execute, plan, binding
    )
    assert len(rows) == 6
    assert distribution["global_request_delta"] == 6
    assert distribution["per_rank_request_attribution_claimed"] is False
    assert distribution["gpu_peak_utilization_percent_by_device"] == [100] * 6
    assert distribution["gpu_device_count"] == 6
    assert distribution["server_binding"] == reduced
    assert distribution["prompts_traces_flags_or_scores_included"] is False
    assert resource["status"] == "PASSED_NO_CONTROLLER_RESOURCE_ERROR"
    assert resource["dind_oom_kill_delta"] == 0
    assert resource["dind_nr_throttled_delta"] == 0
    assert resource["dind_throttled_usec_delta"] == 0
    assert observed_baseline == baseline


def test_controller_resource_control_fails_closed_on_oom_or_throttle() -> None:
    before = {
        "observed_at_epoch": 1,
        "oom_kill": 0,
        "nr_throttled": 0,
        "throttled_usec": 0,
    }
    for field in ("oom_kill", "nr_throttled", "throttled_usec"):
        after = {**before, "observed_at_epoch": 2, field: 1}
        receipt = runtime._dind_resource_control(before, after)  # noqa: SLF001
        assert receipt["status"] == "FAILED_CONTROLLER_RESOURCE_ERROR"
        assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")


def test_result_uses_observed_requests_and_records_latency_headroom() -> None:
    submission, binding, _ = _inputs()
    reduced = runtime.validate_binding(binding, submission, ROOT)
    plan = runtime.qualification_plan(reduced, str(binding["service_origin"]), ROOT)
    baseline = observer.baseline_observation(
        4,
        server_run_dir=early.RUN_DIR,
        pod_name=str(binding["head_pod_name"]),
        pod_uid=str(binding["head_pod_uid"]),
        api_run_id=str(binding["api_run_id"]),
        service_uid=str(binding["service_uid"]),
        server_binding_receipt_sha256=str(binding["receipt_sha256"]),
        observed_at_epoch=1,
    )
    resource = _receipt(
        {
            "status": "PASSED_NO_CONTROLLER_RESOURCE_ERROR",
            "dind_oom_kill_delta": 0,
            "dind_nr_throttled_delta": 0,
            "dind_throttled_usec_delta": 0,
            "protocol_errors": 0,
        }
    )
    streams = [{"execution": {"model_requests": 4}}]
    distribution = {"global_request_delta": 4}
    level = {
        "concurrency": 1,
        "status": "PASSED",
        "completed_stream_count": 1,
        "observed_model_request_count": 4,
        "observed_server_request_delta": 4,
        "elapsed_milliseconds": 1_000,
        "timeout_budget_milliseconds": runtime.parity.TIMEOUT_SECONDS * 1_000,
        "latency_headroom_milliseconds": runtime.parity.TIMEOUT_SECONDS * 1_000 - 1_000,
        "required_latency_headroom_milliseconds": max(
            runtime.MIN_LATENCY_HEADROOM_MILLISECONDS,
            int(runtime.parity.TIMEOUT_SECONDS * 1_000 * runtime.MIN_LATENCY_HEADROOM_FRACTION),
        ),
        "minimum_latency_headroom_fraction": runtime.MIN_LATENCY_HEADROOM_FRACTION,
        "error_count": 0,
        "tool_order_exact": True,
        "tool_arguments_exact": True,
        "stream_receipts": streams,
        "distribution_receipt": distribution,
        "controller_resource_receipt": resource,
        "stable_counter_baseline_receipt": baseline,
    }
    result = {
        "schema_version": runtime.RESULT_SCHEMA,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "qualification_plan": plan,
        "levels": [level],
        "highest_passing_concurrency": 1,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    assert runtime.validate_result(result, plan, ROOT) == 1
    result_binding = result["qualification_plan"]["server_binding"]
    assert result_binding["service_origin"] == binding["service_origin"]
    assert result_binding["workload_uid"] == binding["workload_uid"]
    assert result_binding["server_binding_receipt_sha256"] == binding["receipt_sha256"]
    level["observed_model_request_count"] = 2
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    with pytest.raises(ValueError, match="request/latency"):
        runtime.validate_result(result, plan, ROOT)

    level["observed_model_request_count"] = 4
    level["elapsed_milliseconds"] = runtime.parity.TIMEOUT_SECONDS * 1_000 - 1
    level["latency_headroom_milliseconds"] = 1
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    with pytest.raises(ValueError, match="request/latency"):
        runtime.validate_result(result, plan, ROOT)
