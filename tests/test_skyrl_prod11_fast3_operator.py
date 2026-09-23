from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, JobsError, digest
from training import dev_cleanup_observer as cleanup
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_training as training
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"


def _stage() -> dict:
    predecessor = json.loads(PREDECESSOR.read_bytes())
    predecessor["name"] = operator.FAST3_IDENTITY.predecessor_run_name
    predecessor["sha256"] = "sha256:" + digest(
        {key: value for key, value in predecessor.items() if key != "sha256"}
    )
    return training.stage_spec(operator.FAST3_IDENTITY, predecessor)


def _plan(stage: dict) -> dict:
    fast3_training = pytest.importorskip("training.skyrl_fast3_training")
    sft = pytest.importorskip("training.sft")
    run_path = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v11-fast3.json"
    selected = json.loads(run_path.read_bytes())
    data = copy.deepcopy(stage["predecessor_manifest"])
    data["name"] = operator.FAST3_IDENTITY.run_name
    data["sha256"] = "sha256:" + digest(
        {key: value for key, value in data.items() if key != "sha256"}
    )
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(selected["data"]["manifest"]):
            return copy.deepcopy(data)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        return fast3_training.compile_rl(selected, relative_to=run_path.parent)


def _predecessor_evidence() -> dict:
    return operator.fast3_predecessor_evidence(
        source_head="a" * 40,
        failure_diagnostic_file_sha256="sha256:" + "1" * 64,
        failure_diagnostic_self_sha256="sha256:" + "2" * 64,
        fast2_retirement_file_sha256="sha256:" + "3" * 64,
        fast2_retirement_self_sha256="sha256:" + "4" * 64,
        generation_retry_policy_sha256="sha256:" + "5" * 64,
        predecessor_science_sha256="sha256:" + "6" * 64,
    )


def test_fast3_identity_and_operator_names_are_exact() -> None:
    assert operator.operator_names(operator.FAST3_IDENTITY) == operator.FAST3_OPERATOR_NAMES
    changed = historical.RailIdentity(
        **{
            **operator.FAST3_IDENTITY.__dict__,
            "predecessor_run_name": "chris-q38-rlreward-prod11-fast1",
        }
    )
    with pytest.raises(ValueError, match="identity changed"):
        operator.operator_names(changed)


def test_fast3_stage_package_is_fresh_alert_off_c1_q1_zero_gpu() -> None:
    packet = operator_job.stage_packet(identity=operator.FAST3_IDENTITY, stage=_stage())
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    pod_template = package.job["spec"]["template"]

    assert packet["fresh_identity"] is True
    assert "precreate_recovery" not in packet
    assert packet["operator_name"] == operator.FAST3_OPERATOR_NAMES["stage"]
    assert package.job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == "off"
    assert pod_template["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == "off"
    assert proof["priority"] == "c1"
    assert proof["queue_priority"] == "q1"
    assert proof["gpus"] == 0
    assert package.job["metadata"]["labels"]["cyber-post-train.fleet.ai/role"] == (
        "prod11-fast3-bounded-operator"
    )
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)


def test_fast3_manifest_packet_requires_and_preserves_full_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    plan = _plan(stage)
    launch = {"synthetic": "released-stage"}
    monkeypatch.setattr(direct, "_direct_stage_launch", lambda value, *_args, **_kwargs: value)
    packet = operator_job.manifest_packet(
        identity=operator.FAST3_IDENTITY,
        stage=stage,
        stage_launch_result=launch,
        plan=plan,
    )
    package = operator_job.build_operator_package(packet)

    assert packet["fresh_identity"] is True
    assert packet["plan"] == plan
    assert "preflight_v1_failure" not in packet
    assert package.job["metadata"]["name"] == operator.FAST3_OPERATOR_NAMES["manifest"]
    with pytest.raises(ValueError, match="full training plan"):
        operator_job.manifest_packet(
            identity=operator.FAST3_IDENTITY,
            stage=stage,
            stage_launch_result=launch,
        )
    partial = {"schema": training.SCHEMA, "run_name": operator.FAST3_IDENTITY.run_name}
    with pytest.raises(ValueError, match="full training plan"):
        operator_job.manifest_packet(
            identity=operator.FAST3_IDENTITY,
            stage=stage,
            stage_launch_result=launch,
            plan=partial,
        )


def test_fast3_plan_dispatch_uses_append_only_request_and_zero_gpu_preflight() -> None:
    fast3_training = pytest.importorskip("training.skyrl_fast3_training")
    plan = _plan(_stage())
    request = fast3_training.job_request(plan)
    manifest = launch_direct.preflight_job_manifest(plan, identity=operator.FAST3_IDENTITY)
    operation_root = launch_direct.training_operation_root(plan, identity=operator.FAST3_IDENTITY)

    assert launch_direct.plan_identity(plan, operator.FAST3_IDENTITY) == operator.FAST3_IDENTITY
    assert launch_direct.job_request(plan, identity=operator.FAST3_IDENTITY) == request
    assert direct.training is training
    assert operation_root.parent == operator.hardening.CREATE_ONCE_ROOT
    assert operation_root.name.startswith("training-")
    assert manifest["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == "off"
    assert manifest["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(manifest, sort_keys=True)


def test_fast3_plan_dispatch_validates_fast3_preflight_receipt() -> None:
    fast3_training = pytest.importorskip("training.skyrl_fast3_training")
    plan = _plan(_stage())
    request = fast3_training.job_request(plan)
    raw = {
        "schema": "cyber_skyrl_fast3_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "prod9_runtime": training._binding(),
        "fast3_runtime": fast3_training._binding(),
        "generation_retry_policy_sha256": plan["qualification"]["fast3"]["generation_retry_policy"][
            "sha256"
        ],
        "native_parser_checked": True,
        "ordered_multi_tool_parser_checked": True,
        "chunk_continuation_checked": True,
        "compaction_checked": True,
        "stepwise_prompt_checked": True,
        "ordered_multi_tool_execution_checked": True,
        "output_limit_gradeable_checked": True,
        "output_limit_partial_tool_blocked_checked": True,
        "fresh_recorder_checked": True,
        "tool_result_token_safe": True,
        "recorder_implementation": "training.skyrl_prod9_hardening.Recorder",
        "counts": {"train": 1, "dev": 1},
        "planned_steps": 1,
        "output_absent": True,
        "wandb_create_once": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": operator.FAST3_IDENTITY.wandb_run_id,
            "resume": "never",
        },
    }
    receipt = launch_direct._seal_fresh_preflight_receipt(raw)
    assert (
        launch_direct.preflight_receipt(plan, request, receipt, identity=operator.FAST3_IDENTITY)
        == receipt
    )
    changed_body = {**raw, "generation_retry_policy_sha256": "sha256:" + "0" * 64}
    changed = launch_direct._seal_fresh_preflight_receipt(changed_body)
    with pytest.raises(JobsError, match="Fast3 CPU preflight receipt"):
        launch_direct.preflight_receipt(plan, request, changed, identity=operator.FAST3_IDENTITY)


def test_fast3_launch_packet_has_no_prod10_incident_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"schema": training.SCHEMA}
    request = {"workers": 1, "gpus_per_worker": 8}
    monkeypatch.setattr(launch_direct, "plan_identity", lambda _plan, identity: identity)
    monkeypatch.setattr(launch_direct, "job_request", lambda _plan, *, identity: request)
    monkeypatch.setattr(launch_direct, "_preflight_launch", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(direct, "_source", lambda value: value)
    preflight = direct._seal({"schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA})
    preview = direct._seal(
        {
            "schema": direct.PREVIEW_SCHEMA,
            "context": direct.DEV_CONTEXT,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    provenance = direct._seal(
        {
            "schema": launch_direct.SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA,
            "status": "fresh_sealed_external_dev_server_preview_validated",
            "context": direct.DEV_CONTEXT,
            "sealed_dev_server_preview_sha256": preview["sha256"],
            "checked_at": preview["checked_at"],
        }
    )
    host = direct._seal(
        {
            "schema": launch_direct.FAST3_HOST_IDENTITY_SCHEMA,
            "status": "kubernetes_and_jobs_identity_absent",
            "identity_sha256": operator.FAST3_IDENTITY.sealed_mapping()["sha256"],
            "run_name": operator.FAST3_IDENTITY.run_name,
            "kubernetes_inventories_checked": 10,
            "jobs_api_rows_checked": 0,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    capacity = {
        "schema": "cyber_project_gpu_capacity_census_v1",
        "limits": {"nodes": 10, "gpus": 80},
        "planned": {"nodes": 1, "gpus": 8},
    }
    capacity["sha256"] = digest(capacity)
    packet = operator_job.launch_packet(
        identity=operator.FAST3_IDENTITY,
        plan=plan,
        request=request,
        preflight_launch_result=preflight,
        source_preview={"manifest_yaml": "{}"},
        manifest_sha256="sha256:" + "7" * 64,
        dev_preview=preview,
        dev_preview_provenance=provenance,
        duplicate_proof=host,
        capacity_census=capacity,
        predecessor_evidence=_predecessor_evidence(),
    )
    package = operator_job.build_operator_package(packet)
    historical_keys = {
        "launch_v1_failure",
        "launch_v2_failure",
        "probe_v6_success",
        "probe_v7_failure",
        "probe_v8_failure",
        "probe_v9_success",
        "launch_v3_recovery",
        "inspect_v4_success",
        "launch_v4_failure",
        "inspect_v5_success",
        "launch_v5_failure",
        "inspect_v6_success",
        "launch_v6_failure",
        "launch_v7_failure",
    }

    assert packet["fresh_identity"] is True
    assert historical_keys.isdisjoint(packet)
    assert packet["predecessor_evidence"]["exact_runtime_parity_claimed"] is False
    assert package.job["metadata"]["name"] == operator.FAST3_OPERATOR_NAMES["launch"]
    assert operator_job._validate_packet_semantics(packet) == packet


def test_fast3_host_proof_is_truthful_and_jit_checks_sfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"items":[]}', stderr="")

    class FakeJobs:
        def __init__(self, _token: str, *, base_url: str):
            self.base_url = base_url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def all_runs(self):
            return []

    host = launch_direct.host_identity_proof(
        operator.FAST3_IDENTITY,
        token="token",
        runner=runner,
        jobs_factory=FakeJobs,
    )
    assert "output_root" not in host
    assert "output_absent" not in host
    output_checks: list[dict] = []
    monkeypatch.setattr(launch_direct, "require_output_absent", output_checks.append)
    proof = launch_direct.jit_duplicate_proof(
        operator.FAST3_IDENTITY,
        host,
        token="token",
        runner=runner,
        jobs_factory=FakeJobs,
    )
    assert output_checks == [{"run_dir": operator.FAST3_IDENTITY.output_root}]
    assert proof["runtime_prod_kubernetes_inventories_checked"] == 5
    assert proof["output_absent"] is True
    assert len(commands) == 15


def test_fast3_rejects_legacy_host_output_absence_claim() -> None:
    identity = operator.FAST3_IDENTITY
    legacy = direct._seal(
        {
            "schema": launch_direct.DUPLICATE_SCHEMA,
            "status": "identity_and_output_absent",
            "identity_sha256": identity.sealed_mapping()["sha256"],
            "run_name": identity.run_name,
            "output_root": identity.output_root,
            "kubernetes_inventories_checked": 10,
            "jobs_api_rows_checked": 0,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    with pytest.raises(JobsError, match="proof schema changed"):
        launch_direct._duplicate(legacy, identity, fresh=False)


def test_fast3_marker_gate_rejects_every_present_kind(tmp_path: Path) -> None:
    root = tmp_path / "operation"
    root.mkdir()
    proof = operator._fast3_marker_absence(root)
    assert proof["status"] == "all_create_once_markers_absent"
    marker = root / "PROD10_DIRECT_V3_CREATE.jsonl"
    marker.symlink_to(root / "missing")
    with pytest.raises(operator.OperatorFailure, match="marker_absence"):
        operator._fast3_marker_absence(root)


def test_fast3_live_submitter_normalization_rejects_all_other_drift() -> None:
    expected = {
        "metadata": {
            "annotations": {
                "fleet.ai/submitted-by": "reviewer@example.com",
                "fleet.ai/submitted-by-profile": "00000000-0000-4000-8000-000000000001",
                "fleet.ai/failure-alerts": "off",
            }
        }
    }
    live = copy.deepcopy(expected)
    live["metadata"]["annotations"]["fleet.ai/submitted-by"] = "runtime@example.com"
    live["metadata"]["annotations"]["fleet.ai/submitted-by-profile"] = (
        "00000000-0000-4000-8000-000000000002"
    )
    proof = launch_direct.live_submitter_normalization(expected, live)
    assert proof["status"] == "two_server_owned_annotations_normalized"
    changed = copy.deepcopy(live)
    changed["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(JobsError, match="manifest changed"):
        launch_direct.live_submitter_normalization(expected, changed)


def test_fast3_launch_skips_historical_archive_and_passes_live_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "operation"
    root.mkdir()
    identity = operator.FAST3_IDENTITY
    plan = {"schema": training.SCHEMA, "arguments": {}}
    request = {
        "name": identity.run_name,
        "run_dir": identity.output_root,
        "image": "image@sha256:" + "1" * 64,
        "workers": 1,
        "gpus_per_worker": 8,
    }
    source = {"sealed": True}
    live_source = {"live": True}
    expected = {"kind": "RayJob"}
    preview = {"sha256": "sha256:" + "2" * 64, "checked_at": "now"}
    events: list[str] = []
    monkeypatch.setenv("FLEET_API_KEY", "token")
    monkeypatch.setenv("WANDB_API_KEY", "token")
    monkeypatch.setattr(
        operator,
        "_pre_guard_launch",
        lambda *_args: {
            "identity": identity,
            "plan": plan,
            "request": request,
            "preflight": {"sha256": "sha256:" + "3" * 64},
            "revalidation": {"sha256": "sha256:" + "4" * 64},
            "image_identity": {},
            "source_preview": source,
            "expected": expected,
            "operation_root": root,
        },
    )
    marker_proofs = iter(
        [
            {"sha256": "sha256:" + "5" * 64},
            {"sha256": "sha256:" + "6" * 64},
        ]
    )
    monkeypatch.setattr(
        operator,
        "_fast3_marker_absence",
        lambda *_args: events.append("marker_absence") or next(marker_proofs),
    )
    monkeypatch.setattr(direct, "server_dry_run", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(launch_direct, "validate_gpu_preview", lambda *_args, **_kwargs: preview)
    monkeypatch.setattr(
        launch_direct,
        "_sealed_dev_preview_provenance",
        lambda *_args, **_kwargs: preview,
    )
    jit = iter(
        [
            {"sha256": "sha256:" + "7" * 64},
            {"sha256": "sha256:" + "8" * 64},
        ]
    )
    monkeypatch.setattr(
        launch_direct,
        "jit_duplicate_proof",
        lambda *_args, **_kwargs: events.append("jit") or next(jit),
    )

    class FakeJobs:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def preview(self, _request):
            events.append("live_preview")
            return live_source

    monkeypatch.setattr(operator, "Jobs", FakeJobs)
    monkeypatch.setattr(launch_direct, "gpu_manifest", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        launch_direct,
        "live_submitter_normalization",
        lambda *_args: {"sha256": "sha256:" + "9" * 64},
    )
    monkeypatch.setattr(operator, "_fresh_capacity_census", lambda *_args: {})
    monkeypatch.setattr(launch_direct, "capacity_gate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(direct, "_fresh_at", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        operator,
        "_archive_launch_v3_guard",
        lambda *_args, **_kwargs: pytest.fail("Fast3 reused the historical archive"),
    )

    class FakeGuard:
        def __init__(self, **_kwargs):
            events.append("guard_construct")

        def arm(self):
            events.append("guard_arm")
            return {}

    monkeypatch.setattr(cleanup, "JobsApiPrefixGuard", FakeGuard)
    monkeypatch.setattr(
        operator,
        "_fast3_guard_armed",
        lambda *_args: {"sha256": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        launch_direct,
        "authorize",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "e" * 64},
    )

    def create_once(*_args, **kwargs):
        assert kwargs["live_source_preview"] == live_source
        events.append("create_once")
        return {"capacity_gate_sha256": "sha256:" + "b" * 64, "gpus": 8}

    monkeypatch.setattr(launch_direct, "create_once", create_once)
    monkeypatch.setattr(operator, "_observe_created_run", lambda *_args, **_kwargs: {})
    packet = {
        "sha256": "sha256:" + "c" * 64,
        "dev_preview": {},
        "sealed_dev_preview_provenance": {},
        "duplicate_proof": {},
        "capacity_census": {"sha256": "sha256:" + "d" * 64},
        "predecessor_evidence": _predecessor_evidence(),
    }
    result = operator.run_launch(packet, runner=object())

    assert result["status"] == "gpu_run_succeeded_and_released"
    assert events == [
        "marker_absence",
        "jit",
        "live_preview",
        "jit",
        "marker_absence",
        "guard_construct",
        "guard_arm",
        "create_once",
    ]
