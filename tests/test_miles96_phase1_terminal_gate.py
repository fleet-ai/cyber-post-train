from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from cyber_post_train.jobs import JobsError, digest
from training import miles96_mechanics_canary as mechanics
from training import miles96_mechanics_launch as launch
from training import miles96_phase1_terminal_gate as gate
from training import miles96_signal_qualification as signal
from training import miles_signal_wave


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _plan() -> dict:
    wave = miles_signal_wave.load()
    candidate = wave["candidates"][0]
    task_binding = {
        **miles_signal_wave.task_binding(wave, candidate),
        "authority_receipt_sha256": candidate["authority_receipt_sha256"],
    }
    phase1 = signal.build_plan(
        name=candidate["identity"]["name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
        task_binding=task_binding,
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=candidate["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )
    phase1_request = signal.job_request(phase1)
    body = {
        "schema": mechanics.TASK_SIGNAL_EVIDENCE_SCHEMA,
        "phase1_plan_sha256": "sha256:" + mechanics.digest(phase1),
        "phase1_run_name": candidate["identity"]["name"],
        "model_revision": signal.HF_MODEL_REVISION,
        "model_binding_sha256": signal.HF_MODEL_BINDING_SHA256,
        **phase1["selection_authority"],
        "planned_slot_count": 8,
        "terminal_slot_count": 8,
        "excluded_slot_count": 0,
        "outer_replacement_count": 0,
        **miles_signal_wave.task_binding(wave, candidate),
        "raw_tool_catalog_sha256": phase1["tool_contract"]["raw_tool_catalog_sha256"],
        "openai_tool_catalog_sha256": phase1["tool_contract"]["openai_tool_catalog_sha256"],
        "tool_transform_source_sha256": phase1["tool_contract"]["transform_source_sha256"],
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "episode_timeout_s": 2400,
        "completed_episode_count": 8,
        "finite_rewards": True,
        "reward_variation": True,
        "all_instances_released": True,
        "source_receipt_sha256": _sha("7"),
        "native_terminal_sha256": _sha("9"),
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "runtime_source_manifest_sha256": "sha256:" + mechanics.digest(phase1["runtime_sources"]),
        "runtime_bundle_sha256": signal._runtime_bundle_sha256(phase1_request),
        "request_binding_sha256": signal._request_binding_sha256(phase1_request),
        "runtime_preflight_sha256": _sha("5"),
    }
    return mechanics.build_plan(
        name="chris-q38-m96-term-a1",
        reload_name="chris-q38-m96-term-r1",
        model_root="/mnt/sfs/jobs/q38-prepared-model-v1",
        model_binding_sha256=_sha("a"),
        task_binding=task_binding,
        task_signal_evidence={**body, "sha256": "sha256:" + mechanics.digest(body)},
    )


def _source(plan: dict) -> dict:
    phase1 = gate._phase1_plan(plan)
    evidence = plan["task_signal_evidence"]
    files = {
        signal.EVIDENCE_FILE: _sha("1"),
        signal.NATIVE_TERMINAL_FILE: _sha("2"),
        signal.RUNTIME_PREFLIGHT_FILE: _sha("3"),
        f"{mechanics.PRIVATE_EVIDENCE_DIR}/{signal.PRIVATE_GROUP_FILE}": _sha("4"),
        f"{mechanics.PRIVATE_EVIDENCE_DIR}/episode-1.json": _sha("6"),
    }
    return {
        **gate._phase1_hashes(phase1),
        "public_evidence_sha256": evidence["sha256"],
        "private_group_sha256": evidence["source_receipt_sha256"],
        "native_terminal_sha256": evidence["native_terminal_sha256"],
        "runtime_preflight_sha256": evidence["runtime_preflight_sha256"],
        "receipt_files": files,
        "receipt_file_manifest_sha256": "sha256:" + digest(files),
        "receipt_file_count": len(files),
        "completed_episode_count": 8,
        "distinct_finite_reward_count": 2,
        "unique_verifier_execution_count": 8,
        "terminal_slot_count": 8,
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "actual_sfs_revalidation": True,
    }


def _terminal(plan: dict, observed: datetime) -> dict:
    phase1 = gate._phase1_plan(plan)
    request = signal.job_request(phase1)
    run_name = request["name"] + "-1234abcd"
    rayjob_uid = str(uuid.UUID("11111111-1111-4111-8111-111111111111"))
    raycluster_uid = str(uuid.UUID("22222222-2222-4222-8222-222222222222"))
    return gate._seal(
        {
            "schema": gate.TERMINAL_SCHEMA,
            "observed_at": observed.isoformat().replace("+00:00", "Z"),
            **gate._phase1_hashes(phase1),
            "jobs_api": {
                "run_id": str(uuid.UUID("33333333-3333-4333-8333-333333333333")),
                "name": run_name,
                "status": "SUCCEEDED",
                "run_dir": request["run_dir"],
                "requested_image": request["image"],
            },
            "rayjob": {
                "name": run_name,
                "uid": rayjob_uid,
                "namespace": mechanics.NAMESPACE,
                "job_status": "SUCCEEDED",
                "deployment_status": "Complete",
                "requested_image": request["image"],
                "failure_alerts": "off",
            },
            "raycluster_uid": raycluster_uid,
            "workload_uid": str(uuid.UUID("44444444-4444-4444-8444-444444444444")),
            "pod": {
                "name": run_name + "-head-abcde",
                "uid": str(uuid.UUID("55555555-5555-4555-8555-555555555555")),
                "namespace": mechanics.NAMESPACE,
                "phase": "Succeeded",
                "restart_count": 0,
                "exit_code": 0,
                "termination_reason": "Completed",
                "requested_image": request["image"],
                "runtime_image_id": "docker-pullable://" + request["image"],
                "owner_raycluster_uid": raycluster_uid,
                "root_rayjob_uid": rayjob_uid,
            },
        }
    )


def _census(terminal: dict, observed: datetime, suffix: int) -> dict:
    owner_uids = sorted(
        {
            terminal["rayjob"]["uid"],
            terminal["raycluster_uid"],
            terminal["workload_uid"],
            terminal["pod"]["uid"],
        }
    )
    return gate._seal(
        {
            "schema": gate.CENSUS_SCHEMA,
            "observation_id": str(uuid.UUID(f"00000000-0000-4000-8000-{suffix:012d}")),
            "observed_at": observed.isoformat().replace("+00:00", "Z"),
            "context": mechanics.PROD_CONTEXT,
            "namespace": mechanics.NAMESPACE,
            "jobs_api_run_id": terminal["jobs_api"]["run_id"],
            "run_name": terminal["jobs_api"]["name"],
            "rayjob_uid": terminal["rayjob"]["uid"],
            "raycluster_uid": terminal["raycluster_uid"],
            "workload_uid": terminal["workload_uid"],
            "pod_uid": terminal["pod"]["uid"],
            "root_query": {
                "selector": f"exact-root-name={terminal['jobs_api']['name']}",
                "resource_version": str(100 + suffix),
                "items": [],
            },
            "workload_query": {
                "selector": f"owner-uid={terminal['rayjob']['uid']}",
                "resource_version": str(200 + suffix),
                "items": [],
            },
            "owner_ref_query": {
                "selector": "owner-uids=" + ",".join(owner_uids),
                "resource_version": str(300 + suffix),
                "items": [],
            },
        }
    )


def _authority(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict, datetime]:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    plan = _plan()
    request = mechanics.job_request(plan)
    source = _source(plan)
    terminal = _terminal(plan, now - timedelta(seconds=12))
    censuses = [
        _census(terminal, now - timedelta(seconds=8), 1),
        _census(terminal, now - timedelta(seconds=4), 2),
    ]
    monkeypatch.setattr(gate, "collect_actual_source", lambda _plan: copy.deepcopy(source))
    authority = gate.build_authority(plan, request, terminal, censuses, now=now)
    return plan, request, authority, now


def _reseal(value: dict) -> dict:
    return gate._seal({key: item for key, item in value.items() if key != "sha256"})


def test_exact_actual_terminal_authority_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    assert (
        gate.validate_authority(
            plan,
            request,
            authority,
            expected_sha256=authority["sha256"],
            now=now,
        )
        == authority
    )


def test_self_sealed_public_signal_evidence_is_not_launch_authority() -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    evidence = plan["task_signal_evidence"]
    with pytest.raises(JobsError, match=gate.SCHEMA):
        gate.validate_authority(
            plan,
            request,
            evidence,
            expected_sha256=evidence["sha256"],
        )


def test_review_artifact_cannot_replace_missing_actual_sfs_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    monkeypatch.setattr(
        gate,
        "collect_actual_source",
        lambda _plan: (_ for _ in ()).throw(JobsError("actual receipts absent")),
    )
    with pytest.raises(JobsError, match="actual receipts absent"):
        gate.validate_authority(
            plan,
            request,
            authority,
            expected_sha256=authority["sha256"],
            now=now,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("terminal_observation", "pod", "runtime_image_id"), "wrong@sha256:" + "0" * 64),
        (("terminal_observation", "pod", "restart_count"), 1),
        (("terminal_observation", "pod", "exit_code"), 1),
        (("terminal_observation", "rayjob", "failure_alerts"), "on"),
        (("phase1_source", "optimizer_steps"), 1),
        (("phase1_source", "checkpoint_artifacts_absent"), False),
    ],
)
def test_terminal_or_zero_update_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    changed = copy.deepcopy(authority)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "terminal_observation":
        changed["terminal_observation"] = _reseal(changed["terminal_observation"])
    changed = _reseal(changed)
    with pytest.raises(JobsError):
        gate.validate_authority(
            plan,
            request,
            changed,
            expected_sha256=changed["sha256"],
            now=now,
        )


def test_nonempty_resource_census_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    changed = copy.deepcopy(authority)
    changed["resource_censuses"][1]["owner_ref_query"]["items"] = [{"uid": "still-live"}]
    changed["resource_censuses"][1] = _reseal(changed["resource_censuses"][1])
    changed = _reseal(changed)
    with pytest.raises(JobsError, match="empty resources"):
        gate.validate_authority(
            plan,
            request,
            changed,
            expected_sha256=changed["sha256"],
            now=now,
        )


@pytest.mark.parametrize("defect", ["duplicate", "stale"])
def test_two_resource_censuses_must_be_fresh_and_independent(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    changed = copy.deepcopy(authority)
    if defect == "duplicate":
        changed["resource_censuses"][1]["observation_id"] = changed["resource_censuses"][0][
            "observation_id"
        ]
    else:
        changed["resource_censuses"][0]["observed_at"] = (
            (now - timedelta(seconds=gate.MAX_CENSUS_AGE_SECONDS + 1))
            .isoformat()
            .replace("+00:00", "Z")
        )
    changed["resource_censuses"][0] = _reseal(changed["resource_censuses"][0])
    changed["resource_censuses"][1] = _reseal(changed["resource_censuses"][1])
    changed = _reseal(changed)
    with pytest.raises(JobsError):
        gate.validate_authority(
            plan,
            request,
            changed,
            expected_sha256=changed["sha256"],
            now=now,
        )


def test_reviewed_source_must_still_match_actual_receipt_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    actual = copy.deepcopy(authority["phase1_source"])
    actual["receipt_files"][signal.RUNTIME_PREFLIGHT_FILE] = _sha("f")
    actual["receipt_file_manifest_sha256"] = "sha256:" + digest(actual["receipt_files"])
    monkeypatch.setattr(gate, "collect_actual_source", lambda _plan: actual)
    with pytest.raises(JobsError, match="actual SFS receipts"):
        gate.validate_authority(
            plan,
            request,
            authority,
            expected_sha256=authority["sha256"],
            now=now,
        )


def test_authority_requires_an_independent_digest_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, request, authority, now = _authority(monkeypatch)
    with pytest.raises(JobsError, match="independent digest pin"):
        gate.validate_authority(
            plan,
            request,
            authority,
            expected_sha256=_sha("0"),
            now=now,
        )


def test_mechanics_learning_fails_before_preview_without_terminal_authority(tmp_path) -> None:
    plan = _plan()
    request = mechanics.job_request(plan)

    class Client:
        def preview(self, _request):
            raise AssertionError("ungated learning request reached preview")

    with pytest.raises(JobsError, match="requires exact phase-1 terminal authority"):
        launch.submit_once(plan, request, Client(), tmp_path / "launch")
    assert not (tmp_path / "launch").exists()
