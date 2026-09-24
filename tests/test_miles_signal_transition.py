from __future__ import annotations

import datetime as dt
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import dev_cleanup_observer as cleanup
from training import miles96_exact_image_preflight as image_preflight
from training import miles96_mechanics_canary as mechanics
from training import miles96_mechanics_launch as launch
from training import miles96_signal_qualification as signal
from training import miles_signal_transition as transition
from training import miles_signal_wave

NOW = dt.datetime(2026, 9, 24, 9, 0, tzinfo=dt.UTC)


class FakeClient:
    def __init__(self, wave):
        self.wave = wave

    def _get(self, path, params=None):
        if path == "/v1/account":
            return {"team_id": transition.fleet.FLEET_TEAM_ID, "team_name": "fleet"}
        row = next(
            item
            for item in self.wave["candidates"]
            if item["task"]["version_id"] == params["version_id"]
        )
        return {
            "key": row["task"]["key"],
            "eval_task_version_id": row["task"]["version_id"],
            "task_lifecycle_status": "production",
            "environment_version_id": row["environment"]["version_id"],
            "metadata": {
                "cyber_contract": {
                    "evidence_schema": "1.0.0",
                    "submission_protocol": "2.0.0",
                    "verifier_contract": "3.0.0",
                }
            },
        }


def _safe_bind(monkeypatch, wave):
    by_version = {row["task"]["version_id"]: row for row in wave["candidates"]}

    def bind(_response, selected):
        row = by_version[selected["task_version_id"]]
        return (
            {
                "key": row["task"]["key"],
                "version_id": row["task"]["version_id"],
                "prompt_sha256": row["task"]["prompt_sha256"],
                "env_variables_sha256": row["task"]["env_variables_sha256"],
                "output_json_schema_sha256": row["task"]["output_json_schema_sha256"],
            },
            {**row["environment"], "ttl_seconds": 32400},
            {**row["verifier"], "function_name": "verify"},
        )

    monkeypatch.setattr(transition.fleet, "bind_task", bind)


@pytest.fixture
def live_receipt(monkeypatch):
    wave = miles_signal_wave.load()
    _safe_bind(monkeypatch, wave)
    return transition.collect_live_task_receipt(FakeClient(wave), observed_at=NOW)


def _wrap(body, label):
    return {
        "file_sha256": "sha256:" + digest(label),
        "body_sha256": "sha256:" + digest(body),
        "body": body,
    }


def _static_evidence():
    wave = miles_signal_wave.load()
    lanes = transition._lane_receipts()
    adapter_commit = "1" * 40
    operator_commit = "2" * 40
    packet, config_map, job = image_preflight.build_packet(operator_commit)
    previewed_at = transition._stamp(NOW - dt.timedelta(seconds=2))
    adapter_tests = transition._sealed(
        {
            "schema": transition.TEST_RECEIPT_SCHEMA,
            "subject": "adapter",
            "commit": adapter_commit,
            "status": "passed",
            "commands": ["pytest focused-adapter", "ruff check", "ruff format --check"],
            "passed": 213,
            "failed": 0,
            "ruff_check": True,
            "ruff_format_check": True,
            "git_diff_check": True,
            "observed_at": transition._stamp(NOW),
        }
    )
    image = transition._sealed(
        {
            "schema": transition.IMAGE_PREFLIGHT_SCHEMA,
            "status": "passed",
            "runtime_image": mechanics.IMAGE,
            "image_digest": "sha256:" + mechanics.IMAGE.rsplit("@sha256:", 1)[1],
            "source_closure_sha256": "sha256:" + digest(signal.runtime_source_manifest()),
            "driver_sha256": packet["driver_sha256"],
            "runtime_bundle_sha256": packet["runtime_bundle_sha256"],
            "plan_sha256": packet["plan_sha256"],
            "raw_tool_catalog_sha256": mechanics.RAW_TOOL_CATALOG_SHA256,
            "openai_tool_catalog_sha256": mechanics.OPENAI_TOOL_CATALOG_SHA256,
            "tool_transform_source_sha256": "sha256:" + mechanics.FTI_V1_SHA256,
            "source_hashes": {
                name: "sha256:" + value
                for name, value in image_preflight.EXPECTED_SOURCE_HASHES.items()
            },
            "checks": {
                "image_digest_exact": True,
                "pinned_fti_imports": True,
                "pinned_miles_sources": True,
                "zero_update_entrypoint": True,
                "sealed_raw_tool_catalog_exact": True,
                "pinned_openai_projection_exact": True,
                "session_open_exact_catalog_passed": True,
                "session_open_visible_drift_rejected_and_closed": True,
                "session_open_plan_drift_rejected_and_closed": True,
                "session_open_plan_load_error_rejected_and_closed": True,
            },
            "observed_at": transition._stamp(NOW),
        }
    )
    image_execution = transition._sealed(
        {
            "schema": transition.IMAGE_PREFLIGHT_EXECUTION_SCHEMA,
            "status": "passed",
            "job_name": packet["name"],
            "config_map_name": packet["config_map"],
            "namespace": mechanics.NAMESPACE,
            "kubernetes_context": mechanics.PROD_CONTEXT,
            "job_uid": "00000000-0000-4000-8000-000000000001",
            "pod_uid": "00000000-0000-4000-8000-000000000002",
            "pod_owner_job_uid": "00000000-0000-4000-8000-000000000001",
            "config_map_uid": "00000000-0000-4000-8000-000000000003",
            "workload_name": packet["name"] + "-00000001",
            "workload_uid": "00000000-0000-4000-8000-000000000004",
            "workload_owner_job_uid": "00000000-0000-4000-8000-000000000001",
            "runtime_image": mechanics.IMAGE,
            "observed_image_id": mechanics.IMAGE,
            "priority_class": "c1",
            "queue_priority": "q1",
            "root_failure_alerts": "off",
            "backoff_limit": 0,
            "restart_policy": "Never",
            "requested_gpus": 0,
            "pod_restarts": 0,
            "exit_code": 0,
            "job_server_previews": [
                image_preflight.preview_receipt(job, job, observed_at=previewed_at),
                image_preflight.preview_receipt(job, job, observed_at=previewed_at),
            ],
            "config_map_server_previews": [
                image_preflight.preview_receipt(config_map, config_map, observed_at=previewed_at),
                image_preflight.preview_receipt(config_map, config_map, observed_at=previewed_at),
            ],
            "precreate_absence_receipt": image_preflight.absence_receipt(
                job_absent=True,
                config_map_absent=True,
                pod_prefix_collision_count=0,
                workload_prefix_collision_count=0,
                observed_at=transition._stamp(NOW - dt.timedelta(seconds=1)),
            ),
            "created_at": transition._stamp(NOW),
            "job_manifest_sha256": packet["job_manifest_sha256"],
            "config_map_manifest_sha256": packet["config_map_manifest_sha256"],
            "driver_sha256": packet["driver_sha256"],
            "runtime_bundle_sha256": packet["runtime_bundle_sha256"],
            "plan_sha256": packet["plan_sha256"],
            "preflight_receipt_sha256": image["sha256"],
            "precreate_job_absent": True,
            "precreate_config_map_absent": True,
            "absence_observed_at": transition._stamp(NOW - dt.timedelta(seconds=1)),
            "cleanup_job_uid": "00000000-0000-4000-8000-000000000001",
            "cleanup_config_map_uid": "00000000-0000-4000-8000-000000000003",
            "job_delete_uid_precondition": "00000000-0000-4000-8000-000000000001",
            "config_map_delete_uid_precondition": "00000000-0000-4000-8000-000000000003",
            "job_delete_accepted": True,
            "config_map_delete_accepted": True,
            "workload_cleanup_mode": "job_foreground_cascade",
            "job_absent_after_cleanup": True,
            "pod_absent_after_cleanup": True,
            "config_map_absent_after_cleanup": True,
            "workload_absent_after_cleanup": True,
            "resources_released": True,
            "observed_at": transition._stamp(NOW),
            "released_at": transition._stamp(NOW + dt.timedelta(seconds=1)),
        }
    )
    operator_tests = transition._sealed(
        {
            "schema": transition.TEST_RECEIPT_SCHEMA,
            "subject": "operator",
            "commit": operator_commit,
            "status": "passed",
            "commands": ["pytest focused-operator", "ruff check", "ruff format --check"],
            "passed": 50,
            "failed": 0,
            "ruff_check": True,
            "ruff_format_check": True,
            "git_diff_check": True,
            "observed_at": transition._stamp(NOW),
        }
    )
    excluded = {
        "optimizer_steps",
        "checkpoint",
        "sample_indexes",
        "max_concurrent_episodes",
        "runtime_source_manifest_sha256",
    }
    adapter = {
        "schema": transition.ADAPTER_FREEZE_SCHEMA,
        "branch": "codex/test",
        "commit": adapter_commit,
        "clean": True,
        "authority_sha256": wave["sha256"],
        "runtime_image": mechanics.IMAGE,
        "source_file_count": len(signal.runtime_source_manifest()),
        "source_closure_sha256": "sha256:" + digest(signal.runtime_source_manifest()),
        "tests_receipt_sha256": adapter_tests["sha256"],
        "exact_image_preflight_receipt_sha256": image["sha256"],
        "exact_image_preflight_execution_sha256": image_execution["sha256"],
        "tests": {
            "passed": 213,
            "failed": 0,
            "ruff_check": True,
            "ruff_format_check": True,
            "git_diff_check": True,
        },
        "lanes": [
            {key: value for key, value in lane.items() if key not in excluded} for lane in lanes
        ],
    }
    manifest = transition._operator_source_manifest()
    operator = {
        "schema": transition.OPERATOR_FREEZE_SCHEMA,
        "commit": operator_commit,
        "clean": True,
        "source_manifest": manifest,
        "source_closure_sha256": "sha256:" + digest(manifest),
        "tests_receipt_sha256": operator_tests["sha256"],
        "tests": {"passed": 50, "failed": 0},
    }
    adapter_wrapper = _wrap(adapter, "adapter-file")
    operator_wrapper = _wrap(operator, "operator-file")
    adapter_tests_wrapper = _wrap(adapter_tests, "adapter-tests-file")
    image_wrapper = _wrap(image, "image-file")
    image_execution_wrapper = _wrap(image_execution, "image-execution-file")
    operator_tests_wrapper = _wrap(operator_tests, "operator-tests-file")
    return {
        "pins": {
            "adapter_commit": adapter["commit"],
            "adapter_freeze_file_sha256": adapter_wrapper["file_sha256"],
            "adapter_test_receipt_file_sha256": adapter_tests_wrapper["file_sha256"],
            "exact_image_preflight_receipt_file_sha256": image_wrapper["file_sha256"],
            "exact_image_preflight_execution_receipt_file_sha256": image_execution_wrapper[
                "file_sha256"
            ],
            "operator_commit": operator["commit"],
            "operator_freeze_file_sha256": operator_wrapper["file_sha256"],
            "operator_test_receipt_file_sha256": operator_tests_wrapper["file_sha256"],
        },
        "adapter_freeze": adapter_wrapper,
        "adapter_test_receipt": adapter_tests_wrapper,
        "exact_image_preflight_receipt": image_wrapper,
        "exact_image_preflight_execution_receipt": image_execution_wrapper,
        "operator_freeze": operator_wrapper,
        "operator_test_receipt": operator_tests_wrapper,
    }


def _lane():
    row, plan, request = transition._lane_objects()[0]
    return row, plan, request


def _parent_review(candidate):
    return transition._sealed(
        {
            "schema": transition.REVIEW_SCHEMA,
            "candidate_sha256": candidate["sha256"],
            "approved": True,
            "reviewer": "root",
            "reviewed_at": transition._stamp(NOW),
            "evidence_file_sha256s": {
                key: value
                for key, value in candidate["future_bindings"]["pins"].items()
                if key.endswith("_file_sha256")
            },
        }
    )


def _reseal_wrapper(wrapper):
    body = transition._sealed(
        {key: value for key, value in wrapper["body"].items() if key != "sha256"}
    )
    wrapper["body"] = body
    wrapper["body_sha256"] = "sha256:" + digest(body)
    return wrapper


def test_exact_image_packet_is_deterministic_and_safe():
    commit = "2" * 40
    packet, config_map, job = image_preflight.build_packet(commit)
    assert (packet, config_map, job) == image_preflight.build_packet(commit)
    assert packet["external_post_count"] == 0
    assert packet["requested_gpus"] == 0
    assert packet["job_manifest_sha256"] == "sha256:" + digest(job)
    assert packet["config_map_manifest_sha256"] == "sha256:" + digest(config_map)
    assert config_map["immutable"] is True
    assert config_map["data"]["driver.py"] == Path(image_preflight.__file__).read_text()
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["restartPolicy"] == "Never"
    assert pod["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(pod["containers"][0]["resources"])


def test_server_preview_receipts_bind_full_normalized_bodies():
    _packet, config_map, job = image_preflight.build_packet("2" * 40)
    rendered = deepcopy(job)
    rendered["metadata"]["uid"] = "00000000-0000-4000-8000-000000000001"
    rendered["status"] = {"active": 0}
    rendered["spec"]["selector"] = {"matchLabels": {"controller-uid": "volatile"}}
    rendered["spec"]["template"]["metadata"]["labels"]["controller-uid"] = "volatile"
    receipt = image_preflight.preview_receipt(rendered, job, observed_at=transition._stamp(NOW))
    image_preflight.validate_preview_receipt(receipt, job)
    assert "status" not in receipt["normalized_manifest"]
    assert "selector" not in receipt["normalized_manifest"]["spec"]

    config_receipt = image_preflight.preview_receipt(
        config_map, config_map, observed_at=transition._stamp(NOW)
    )
    image_preflight.validate_preview_receipt(config_receipt, config_map)


def test_server_preview_rejects_alert_and_gpu_drift():
    _packet, _config_map, job = image_preflight.build_packet("2" * 40)
    alert = deepcopy(job)
    alert["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(ValueError, match="differs"):
        image_preflight.preview_receipt(alert, job, observed_at=transition._stamp(NOW))
    gpu = deepcopy(job)
    gpu["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "1"
    with pytest.raises(ValueError, match="zero-GPU"):
        image_preflight.validate_server_manifest(gpu, gpu)


def test_observed_image_id_extracts_exact_digest_from_cri_forms():
    image_digest = mechanics.IMAGE.rsplit("@", 1)[1]
    assert transition._image_id_digest(mechanics.IMAGE) == image_digest
    assert transition._image_id_digest("containerd://" + image_digest) == image_digest
    assert transition._image_id_digest("sha256:" + "0" * 64) != image_digest


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "failed"),
        ("job_name", "other-preflight"),
        ("config_map_name", "other-config"),
        ("namespace", "default"),
        ("kubernetes_context", "dev-context"),
        ("runtime_image", "registry.invalid/example@sha256:" + "0" * 64),
        ("observed_image_id", "registry.invalid/example@sha256:" + "0" * 64),
        ("job_uid", "not-a-uuid"),
        ("pod_uid", "not-a-uuid"),
        ("pod_owner_job_uid", "00000000-0000-4000-8000-000000000002"),
        ("config_map_uid", "not-a-uuid"),
        ("workload_name", "unowned-workload"),
        ("workload_uid", "not-a-uuid"),
        ("workload_owner_job_uid", "00000000-0000-4000-8000-000000000002"),
        ("priority_class", "c0"),
        ("queue_priority", "q0"),
        ("root_failure_alerts", "on"),
        ("backoff_limit", 1),
        ("restart_policy", "OnFailure"),
        ("requested_gpus", 1),
        ("pod_restarts", 1),
        ("exit_code", 1),
        ("job_server_previews", []),
        ("config_map_server_previews", []),
        ("precreate_absence_receipt", {}),
        ("job_manifest_sha256", "sha256:" + "0" * 64),
        ("config_map_manifest_sha256", "sha256:" + "0" * 64),
        ("driver_sha256", "sha256:" + "0" * 64),
        ("runtime_bundle_sha256", "sha256:" + "0" * 64),
        ("plan_sha256", "sha256:" + "0" * 64),
        ("preflight_receipt_sha256", "sha256:" + "0" * 64),
        ("precreate_job_absent", False),
        ("precreate_config_map_absent", False),
        ("cleanup_job_uid", "00000000-0000-4000-8000-000000000003"),
        ("cleanup_config_map_uid", "00000000-0000-4000-8000-000000000001"),
        ("job_delete_uid_precondition", "00000000-0000-4000-8000-000000000002"),
        (
            "config_map_delete_uid_precondition",
            "00000000-0000-4000-8000-000000000001",
        ),
        ("job_delete_accepted", False),
        ("config_map_delete_accepted", False),
        ("workload_cleanup_mode", "name_only"),
        ("job_absent_after_cleanup", False),
        ("pod_absent_after_cleanup", False),
        ("config_map_absent_after_cleanup", False),
        ("workload_absent_after_cleanup", False),
        ("resources_released", False),
        ("released_at", "2026-09-24T08:59:59Z"),
    ],
)
def test_exact_image_execution_receipt_tamper_fails_closed(field, bad_value):
    evidence = _static_evidence()
    image = evidence["exact_image_preflight_receipt"]["body"]
    wrapper = evidence["exact_image_preflight_execution_receipt"]
    wrapper["body"][field] = bad_value
    _reseal_wrapper(wrapper)
    packet, config_map, job = image_preflight.build_packet("2" * 40)
    with pytest.raises(ValueError, match="execution receipt is incomplete"):
        transition._validate_image_preflight_execution(
            wrapper,
            expected_file_sha256=wrapper["file_sha256"],
            expected_runtime_image=mechanics.IMAGE,
            expected_preflight_sha256=image["sha256"],
            expected_packet=packet,
            expected_config_map=config_map,
            expected_job=job,
        )


def test_exact_image_execution_rejects_preview_body_and_absence_drift():
    packet, config_map, job = image_preflight.build_packet("2" * 40)
    for section in ("job_server_previews", "precreate_absence_receipt"):
        evidence = _static_evidence()
        image = evidence["exact_image_preflight_receipt"]["body"]
        wrapper = evidence["exact_image_preflight_execution_receipt"]
        if section == "job_server_previews":
            preview = wrapper["body"][section][0]
            preview["normalized_manifest"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] = (
                "on"
            )
            preview["normalized_manifest_sha256"] = "sha256:" + image_preflight.canonical_digest(
                preview["normalized_manifest"]
            )
            wrapper["body"][section][0] = image_preflight.sealed(
                {key: value for key, value in preview.items() if key != "sha256"}
            )
        else:
            absence = wrapper["body"][section]
            absence["job_absent"] = False
            wrapper["body"][section] = image_preflight.sealed(
                {key: value for key, value in absence.items() if key != "sha256"}
            )
        _reseal_wrapper(wrapper)
        with pytest.raises(ValueError, match="execution receipt is incomplete"):
            transition._validate_image_preflight_execution(
                wrapper,
                expected_file_sha256=wrapper["file_sha256"],
                expected_runtime_image=mechanics.IMAGE,
                expected_preflight_sha256=image["sha256"],
                expected_packet=packet,
                expected_config_map=config_map,
                expected_job=job,
            )


def test_static_gate_requires_exact_image_source_hashes(live_receipt):
    evidence = _static_evidence()
    wrapper = evidence["exact_image_preflight_receipt"]
    wrapper["body"]["source_hashes"]["fti.fleet.v1"] = "sha256:" + "0" * 64
    _reseal_wrapper(wrapper)
    candidate = transition.build_review_candidate(live_receipt, evidence, observed_at=NOW)
    assert candidate["gates"][0]["passed"] is False


def test_static_gate_requires_execution_body_and_adapter_cross_link(live_receipt):
    missing = _static_evidence()
    missing.pop("exact_image_preflight_execution_receipt")
    candidate = transition.build_review_candidate(live_receipt, missing, observed_at=NOW)
    assert candidate["gates"][0]["passed"] is False

    mismatch = _static_evidence()
    mismatch["adapter_freeze"]["body"]["exact_image_preflight_execution_sha256"] = (
        "sha256:" + "0" * 64
    )
    _reseal_wrapper(mismatch["adapter_freeze"])
    candidate = transition.build_review_candidate(live_receipt, mismatch, observed_at=NOW)
    assert candidate["gates"][0]["passed"] is False


def _post_bundle(plan, request):
    now = NOW.timestamp()
    preview = transition._sealed(
        {
            "schema": "cyber_miles96_live_server_preview_v1",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest("manifest"),
            "root_failure_alerts": "off",
            "backoff_limit": 0,
            "shutdown_after_job_finishes": True,
            "nodes": 1,
            "gpus": 8,
            "observed_at_unix": now,
            "preview_count": 2,
            "priority_class": "c1",
            "queue_priority": "q1",
            "requeue_if_preempted": False,
        }
    )
    observer = transition._sealed(
        {
            "schema": cleanup.JOBS_API_PREFIX_GUARD_SCHEMA,
            "status": "armed_non_destructive_prefix_guard",
            "run_name_prefix": request["name"],
            "run_dir": request["run_dir"],
            "image": request["image"],
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": preview["manifest_sha256"],
            "maximum_seconds": launch._maximum_seconds(plan, request),
            "expected_gpus": 8,
            "prefix_collision_count_before_post": 0,
            "observer_pid": os.getpid(),
            "armed_at": transition._stamp(NOW),
        }
    )
    sfs_body = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": int(now),
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "run_name": request["name"],
        "run_dir": request["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    sfs = {**sfs_body, "sha256": digest(sfs_body)}
    absence = transition._sealed(
        {
            "schema": "cyber_miles96_fresh_absence_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "observer_armed_sha256": observer["sha256"],
            "sfs_output_absent": True,
            "sfs_output_absence_receipt_sha256": sfs["sha256"],
            "observed_at_unix": now,
        }
    )
    final = transition._sealed(
        {
            "schema": "cyber_miles96_final_prepost_gate_v1",
            "status": "passed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "observer_armed_sha256": observer["sha256"],
            "observed_at_unix": now,
        }
    )
    capacity = transition._sealed(
        {
            "schema": launch.CAPACITY_GATE_SCHEMA,
            "status": "passed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "planned": {"nodes": 1, "gpus": 8},
            "observed_at": transition._stamp(NOW),
            "capacity_census": {
                "qualified": True,
                "limits": {"nodes": 10, "gpus": 80},
                "projected": {"nodes": 4, "gpus": 32},
            },
        }
    )
    return transition._sealed(
        {
            "schema": transition.POST_BUNDLE_SCHEMA,
            "name": request["name"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "server_preview": preview,
            "observer_armed": observer,
            "fresh_absence": absence,
            "final_prepost_gate": final,
            "sfs_output_absence": sfs,
            "capacity_gate": capacity,
        }
    )


def test_live_receipt_is_sanitized_and_exact(live_receipt):
    transition.validate_live_task_receipt(live_receipt, now=NOW, require_fresh=True)
    encoded = json.dumps(live_receipt)
    for forbidden in ('"prompt"', '"code"', '"env_variables"', '"credentials"'):
        assert forbidden not in encoded


def test_live_receipt_rejects_environment_version_response_drift(monkeypatch):
    wave = miles_signal_wave.load()
    _safe_bind(monkeypatch, wave)
    client = FakeClient(wave)
    original = client._get

    def drift(path, params=None):
        value = original(path, params)
        if path != "/v1/account":
            value["environment_version_id"] = "00000000-0000-4000-8000-000000000000"
        return value

    client._get = drift
    with pytest.raises(ValueError, match="environment-version"):
        transition.collect_live_task_receipt(client, observed_at=NOW)


def test_live_task_receipt_freshness_boundary_is_900_seconds(live_receipt):
    transition.validate_live_task_receipt(
        live_receipt,
        now=NOW + dt.timedelta(seconds=900),
        require_fresh=True,
    )
    with pytest.raises(ValueError, match="stale"):
        transition.validate_live_task_receipt(
            live_receipt,
            now=NOW + dt.timedelta(seconds=901),
            require_fresh=True,
        )


def test_static_candidate_rebuilds_all_four_lanes_and_stays_unlaunchable(live_receipt):
    candidate = transition.build_review_candidate(live_receipt, _static_evidence(), observed_at=NOW)
    transition.validate_review_candidate(candidate, live_receipt)
    assert candidate["launchable"] is False
    assert [gate["passed"] for gate in candidate["gates"]] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert len(candidate["future_bindings"]["lanes"]) == 4


def test_sha_shaped_claims_cannot_replace_receipt_bodies(live_receipt):
    evidence = _static_evidence()
    evidence["adapter_freeze"]["body"]["lanes"][0]["request_sha256"] = "sha256:" + "0" * 64
    evidence["adapter_freeze"]["body_sha256"] = "sha256:" + digest(
        evidence["adapter_freeze"]["body"]
    )
    candidate = transition.build_review_candidate(live_receipt, evidence, observed_at=NOW)
    assert candidate["gates"][0]["passed"] is False


def test_parent_review_digest_is_an_independent_required_pin(live_receipt):
    candidate = transition.build_review_candidate(live_receipt, _static_evidence(), observed_at=NOW)
    review = _parent_review(candidate)
    with pytest.raises(ValueError, match="independently pinned"):
        transition.validate_parent_review(
            candidate,
            review,
            expected_parent_review_sha256="sha256:" + "0" * 64,
        )

    review = _parent_review(candidate)
    review["evidence_file_sha256s"]["adapter_freeze_file_sha256"] = "sha256:" + "0" * 64
    review = transition._sealed({key: value for key, value in review.items() if key != "sha256"})
    with pytest.raises(ValueError, match="independently pinned"):
        transition.validate_parent_review(
            candidate,
            review,
            expected_parent_review_sha256=review["sha256"],
        )


def test_lane_successor_binds_exact_post_time_receipts(live_receipt):
    _, plan, request = _lane()
    candidate = transition.build_review_candidate(live_receipt, _static_evidence(), observed_at=NOW)
    review = _parent_review(candidate)
    bundle = _post_bundle(plan, request)
    approved = transition.approve_review_candidate(
        candidate,
        review,
        live_receipt,
        bundle,
        plan,
        request,
        expected_parent_review_sha256=review["sha256"],
        reviewed_at=NOW,
    )
    assert approved["lane"] == request["name"]
    assert approved["post_receipt_bundle_sha256"] == bundle["sha256"]
    assert approved["launchable"] is True


def test_post_time_receipt_body_drift_fails_closed():
    _, plan, request = _lane()
    bundle = _post_bundle(plan, request)
    bundle["server_preview"]["root_failure_alerts"] = "on"
    bundle["server_preview"] = transition._sealed(
        {k: v for k, v in bundle["server_preview"].items() if k != "sha256"}
    )
    bundle = transition._sealed({k: v for k, v in bundle.items() if k != "sha256"})
    with pytest.raises(ValueError, match="unsafe"):
        transition.validate_post_receipt_bundle(bundle, plan, request, now=NOW)


def test_committed_default_transition_remains_unlaunchable():
    live = json.loads(
        Path(
            "configs/qualification/qwen38-miles-signal-live-task-receipt-20260924.json"
        ).read_text()
    )
    candidate = json.loads(
        Path(
            "configs/qualification/qwen38-miles-signal-transition-candidate-20260924.json"
        ).read_text()
    )
    transition.validate_live_task_receipt(live)
    transition.validate_review_candidate(candidate)
    assert candidate["launchable"] is False
