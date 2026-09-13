"""Adversarial, value-free tests for the prepared HF export/reload rails."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from training import miles, miles_event_evidence
from training import miles_hf_export as hf
from training import miles_hf_export_job as job


def _active_binding_reference() -> dict[str, str]:
    return {
        "path": "/exact/ACTIVE_CANARY_BINDING.json",
        "file_sha256": "9" * 64,
        "receipt_sha256": "a" * 64,
    }


def _prediction_probe() -> dict:
    return {
        "schema": hf.PREDICTION_SCHEMA,
        "probe_id": hf.PREDICTION_PROBE_ID,
        "input_ids_sha256": "sha256:" + digest(list(hf.PREDICTION_INPUT_IDS)),
        "sequence_length": len(hf.PREDICTION_INPUT_IDS),
        "top_k": hf.PREDICTION_TOP_K,
        "selection_margin_threshold": hf.PREDICTION_MARGIN,
        "selection_margin_satisfied": True,
        "prediction_sha256": "sha256:" + "a" * 64,
        "logits_included": False,
        "task_content_included": False,
        "benchmark_content_included": False,
    }


@pytest.fixture(autouse=True)
def _local_paths(monkeypatch) -> None:
    monkeypatch.setattr(job, "_sfs", lambda value, _label: str(value))
    monkeypatch.setattr("cyber_post_train.jobs.validate_request", lambda _request: None)


def _plan(tmp_path: Path, stage: str) -> dict:
    root = tmp_path / stage
    source = (
        {
            "checkpoint": {
                "path": "/exact/checkpoint.json",
                "file_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
            },
            "terminal_acceptance": {
                "path": "/exact/terminal.json",
                "file_sha256": "3" * 64,
                "receipt_sha256": "4" * 64,
            },
            "native_reload_acceptance": {
                "path": "/exact/reload.json",
                "file_sha256": "5" * 64,
                "receipt_sha256": "6" * 64,
            },
            "source_plan_sha256": "b" * 64,
            "active_canary_binding": _active_binding_reference(),
            "prediction_probe": _prediction_probe(),
        }
        if stage == "export"
        else {
            "export_acceptance": {
                "path": "/exact/HF_EXPORT_ACCEPTED.json",
                "file_sha256": "7" * 64,
                "receipt_sha256": "8" * 64,
            },
            "export": {
                "path": "/exact/EXPORT.json",
                "file_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
                "tensor_inventory_sha256": "3" * 64,
                "active_canary_binding": _active_binding_reference(),
            },
        }
    )
    return {
        "schema": job.PLAN_SCHEMA,
        "stage": stage,
        "run_name": "chris-q38-miles-hf-" + stage,
        "output_root": str(root),
        "artifact_path": str(root / ("model" if stage == "export" else "HF_RELOAD_VALIDATED.json")),
        "source": source,
        "runtime_sha256": digest(job._runtime()),
        "deadline_seconds": job.DEADLINES[stage],
        "execution": job._execution(stage),
        "privacy": {
            "secrets": [],
            "network_downloads": False,
            "reward_values": False,
            "task_content": False,
            "benchmark_feedback": False,
        },
    }


def _submission(plan: dict) -> dict:
    request = job.job_request(plan)
    if plan["stage"] == "export":
        request_env = {
            row["name"]: row["value"]
            for row in request["spec"]["template"]["spec"]["containers"][0]["env"]
        }
    else:
        request_env = request["env"]
    ordinary = {
        key: value
        for key, value in request_env.items()
        if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    run_id = "11111111-1111-4111-8111-111111111111"
    value = {
        "schema": job.SUBMISSION_SCHEMA,
        "source_commit": "a" * 40,
        "source_plan_path": "/exact/plan.json",
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_plan_file_sha256": "sha256:" + "4" * 64,
        "source_request_path": "/exact/request.json",
        "source_request_sha256": "sha256:" + digest(request),
        "source_request_file_sha256": "sha256:" + "5" * 64,
        "submission_journal_path": "/exact/SUBMISSION.jsonl",
        "submission_journal_file_sha256": "sha256:" + "6" * 64,
        "runtime_bundle_sha256": "sha256:" + job._transport(request)[1],
        "api": {
            "base_url": (
                job.DEV_KUBERNETES_API if plan["stage"] == "export" else job.API_URLS["dev"]
            ),
            "run_id": run_id,
            "run_name": (
                plan["run_name"]
                if plan["stage"] == "export"
                else plan["run_name"] + "-" + run_id[:8]
            ),
        },
        "request": job._request_summary(request, ordinary, job._transport(request)[0]),
        "dispatch": plan["execution"]["dispatch"],
        "create_call_count": 1,
        "submitted_at": "2026-09-12T12:00:00Z",
        "secret_values_included": False,
        "reward_values_included": False,
        "task_content_included": False,
        "benchmark_feedback_included": False,
    }
    value["sha256"] = digest(value)
    return value


def test_request_is_plan_bound_secretless_and_allocates_zero_export_gpus(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path, "export")
    request = job.job_request(plan)

    assert request["apiVersion"] == "batch/v1" and request["kind"] == "Job"
    spec = request["spec"]
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    env = {row["name"]: row["value"] for row in container["env"]}
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert pod["priorityClassName"] == "c2"
    assert request["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q2"
    assert pod["nodeSelector"] == {"workload": job.DEV_GPU_NODE_POOL}
    assert spec["suspend"] is True and spec["ttlSecondsAfterFinished"] == 0
    assert spec["activeDeadlineSeconds"] == job.DEADLINES["export"]
    payload, bundle_sha256 = job._transport(request)
    assert payload["argv"][-1] == digest(plan)
    assert "training/miles_promotion.py" in payload["files"]
    assert "training/miles_policy_observer.py" not in payload["files"]
    assert bundle_sha256 in container["command"][2]
    assert (
        request["metadata"]["annotations"][
            "cyber-post-train.fleet.ai/runtime-bundle-sha256"
        ]
        == bundle_sha256
    )
    runtime = job._runtime()
    monkeypatch.setattr(
        job.subprocess,
        "run",
        lambda argv, **_kwargs: SimpleNamespace(
            returncode=0, stdout=runtime[argv[-1].split(":", 1)[1]].encode()
        ),
    )
    projection, observed_bundle = job._request_projection(
        plan, request, source_commit="a" * 40, repo_root=tmp_path
    )
    assert projection["gpus"] == 0 and observed_bundle == bundle_sha256

    changed = copy.deepcopy(request)
    changed["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"][
        "memory"
    ] = "64Gi"
    with pytest.raises(ValueError, match="request or source commit"):
        job._request_projection(plan, changed, source_commit="a" * 40, repo_root=tmp_path)

    leaked = copy.deepcopy(plan)
    leaked["benchmark_payload"] = "must never enter the runtime bundle"
    with pytest.raises(ValueError, match="prepared plan drift"):
        job.job_request(leaked)

    expanded = copy.deepcopy(plan)
    expanded["source"]["unbound_input"] = {"path": "/tmp/private", "file_sha256": "0" * 64}
    with pytest.raises(ValueError, match="source fields"):
        job.job_request(expanded)


def test_dev4_templates_are_inert_until_terminal_digests_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    export_template = json.loads(
        (root / "configs/qualification/qwen38-miles-hf-export-dev4-v1.template.json").read_text()
    )
    reload_template = json.loads(
        (root / "configs/qualification/qwen38-miles-hf-reload-dev4-v1.template.json").read_text()
    )

    assert export_template["schema"] == job.CONFIG_SCHEMA
    assert export_template["name"] == "q38-miles-hf-export-dev4-v1"
    assert set(export_template["source"]) == {
        "active_canary_binding",
        "checkpoint",
        "terminal_acceptance",
        "native_reload_acceptance",
    }
    assert all(ref["file_sha256"] is None for ref in export_template["source"].values())
    assert reload_template["schema"] == job.CONFIG_SCHEMA
    assert reload_template["source"]["export_acceptance"]["file_sha256"] is None
    with pytest.raises(ValueError, match="digest-bound"):
        job.compile_job(export_template)
    with pytest.raises(ValueError, match="digest-bound"):
        job.compile_job(reload_template)


def test_plan_rejects_null_active_canary_digest(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "export")
    plan["source"]["active_canary_binding"]["file_sha256"] = None
    with pytest.raises(ValueError, match="prepared reference"):
        job.validate_plan(plan, check_files=False)


def test_runtime_bundle_imports_in_an_isolated_checkout(tmp_path: Path) -> None:
    files = {
        **job._runtime(),
        "training/__init__.py": "",
        "cyber_post_train/__init__.py": "",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(tmp_path)!r});"
                "import training.miles_hf_export,training.miles_hf_export_job"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_submission_rejects_api_or_bundle_rebinding(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, "reload")
    submission = _submission(plan)
    job.validate_submission_binding(submission, plan, check_files=False)

    for mutation in ("api", "request", "bundle", "projection"):
        changed = copy.deepcopy(submission)
        if mutation == "api":
            changed["api"]["run_id"] = "22222222-2222-4222-8222-222222222222"
        elif mutation == "request":
            changed["source_request_sha256"] = "sha256:" + "0" * 64
        else:
            if mutation == "bundle":
                changed["runtime_bundle_sha256"] = "sha256:" + "0" * 64
            else:
                changed["request"]["environment_names"].append("FLEET_API_KEY")
        changed.pop("sha256")
        changed["sha256"] = digest(changed)
        with pytest.raises(ValueError, match="submission binding"):
            job.validate_submission_binding(changed, plan, check_files=False)


def test_submission_journal_is_the_api_identity_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "reload")
    request = job.job_request(plan)
    run_id = "11111111-1111-4111-8111-111111111111"
    rows = [
        {
            "state": "POST_INTENT_DO_NOT_RETRY",
            "api_base_url": job.API_URLS["dev"],
            "request_sha256": digest(request),
            "manifest_sha256": "1" * 64,
            "nodes": 1,
            "gpus": 1,
            "image": miles.IMAGE,
        },
        {
            "state": "POST_RESPONSE",
            "name": plan["run_name"] + "-" + run_id[:8],
            "job_id": run_id,
            "run_dir": plan["output_root"],
            "status": "queued",
            "created_at": "2026-09-12T12:00:00Z",
            "finished_at": None,
        },
    ]
    journal = tmp_path / "SUBMISSION.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reopened, file_sha256 = job._submission_journal(journal, plan, request)
    assert reopened["response"]["job_id"] == run_id
    assert file_sha256 == hf._hash(journal)

    rows[1]["job_id"] = "22222222-2222-4222-8222-222222222222"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="exact dev API POST"):
        job._submission_journal(journal, plan, request)


def test_direct_job_journal_binds_server_dry_run_and_created_uid(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "export")
    request = job.job_request(plan)
    run_id = "11111111-1111-4111-8111-111111111111"
    rows = [
        {
            "state": "KUBERNETES_POST_INTENT_DO_NOT_RETRY",
            "api_base_url": job.DEV_KUBERNETES_API,
            "kube_context": job.DEV_KUBE_CONTEXT,
            "namespace": job.DEV_NAMESPACE,
            "namespace_uid": job.DEV_NAMESPACE_UID,
            "request_sha256": digest(request),
            "server_dry_run_sha256": "1" * 64,
            "name": plan["run_name"],
            "image": miles.IMAGE,
            "gpus": 0,
        },
        {
            "state": "KUBERNETES_POST_RESPONSE",
            "api_version": "batch/v1",
            "kind": "Job",
            "name": plan["run_name"],
            "namespace": job.DEV_NAMESPACE,
            "uid": run_id,
            "resource_version": "12345",
            "creation_timestamp": "2026-09-12T12:00:00Z",
        },
    ]
    journal = tmp_path / "SUBMISSION.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    observed, _ = job._submission_journal(journal, plan, request)
    assert observed["intent"]["gpus"] == 0
    assert observed["response"]["uid"] == run_id

    rows[0]["gpus"] = 1
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="exact dev API POST"):
        job._submission_journal(journal, plan, request)


def test_preflight_receipt_cannot_self_declare_smaller_storage_or_memory(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "reload")
    request = job.job_request(plan)
    required = 8 * 1024**3
    proof = {
        "schema": job.PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "stage": "reload",
        "source_reopened": True,
        "runtime_bundle_verified": True,
        "required_free_bytes": required,
        "observed_free_bytes": required,
        "memory_request_bytes": int(job.quantity(job.RESOURCES["reload"]["memory_request"])),
        "memory_limit_bytes": int(job.quantity(job.RESOURCES["reload"]["memory_limit"])),
        "submitted": False,
    }
    proof["sha256"] = digest(proof)
    job.validate_preflight(plan, request, proof)

    for key in ("required_free_bytes", "memory_request_bytes"):
        changed = copy.deepcopy(proof)
        changed[key] = 0
        changed.pop("sha256")
        changed["sha256"] = digest(changed)
        with pytest.raises(ValueError, match="preflight"):
            job.validate_preflight(plan, request, changed)


def test_wall_clock_deadline_raises_a_truthful_bounded_failure() -> None:
    with pytest.raises(TimeoutError, match="prepared wall-clock deadline"):
        job._deadline_expired(0, None)


@pytest.mark.parametrize(
    ("stage", "schema"),
    (
        ("export", job.EXPORT_CONTROLLER_SCHEMA),
        ("reload", hf.RELOAD_CONTROLLER_SCHEMA),
    ),
)
def test_release_reopens_the_controller_with_its_stage_schema(
    stage: str, schema: str, tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path, stage)
    submission = _submission(plan)
    controller_path = tmp_path / "CONTROLLER.json"
    controller_path.write_text("{}")
    seen: list[str] = []

    def read(_path: Path, selected_schema: str) -> tuple[dict, str]:
        seen.append(selected_schema)
        return {"observed_at": 1.0}, "a" * 64

    monkeypatch.setattr(miles_event_evidence, "_read", read)
    monkeypatch.setattr(job, "validate_event_controller", lambda *_args: {})
    monkeypatch.setattr(miles_event_evidence, "validate_hf_release_journal", lambda *_args: None)
    monkeypatch.setattr(
        job,
        "release_from_event_journal",
        lambda **_kwargs: {"schema": "synthetic", "status": "released"},
    )
    monkeypatch.setattr(miles_event_evidence, "_write", lambda _path, value: value)

    absence = (
        {
            "job_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
        }
        if stage == "export"
        else {
            "raycluster_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
        }
    )
    released = miles_event_evidence.compile_release(
        plan,
        submission,
        controller_path=controller_path,
        absence=absence,
        observed_at=1.0,
        output=tmp_path / "RELEASE.json",
    )

    assert released["status"] == "released"
    assert seen == [schema]


def test_export_acceptance_is_create_once_and_reload_consumes_only_it(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path, "export")
    root = Path(plan["output_root"])
    model = root / "model"
    model.mkdir(parents=True)
    artifact = {
        "source": {
            "source_plan_sha256": "b" * 64,
            "active_canary_binding": _active_binding_reference(),
        },
        "tensor_inventory_sha256": "1" * 64,
        "sha256": "sha256:" + "2" * 64,
    }
    artifact_path = model / "EXPORT.json"
    artifact_path.write_text(json.dumps(artifact))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    submission = _submission(plan)
    submission_path = tmp_path / "SUBMITTED.json"
    submission_path.write_text(json.dumps(submission))
    controller = {"artifact_path": str(artifact_path), "sha256": "3" * 64}
    controller_path = root / "HF_EXPORT_CONTROLLER_TERMINAL.json"
    controller_path.write_text(json.dumps(controller))
    release = {"sha256": "4" * 64}
    release_path = root / "HF_EXPORT_RELEASE.json"
    release_path.write_text(json.dumps(release))

    validate_plan = job.validate_plan
    monkeypatch.setattr(job, "validate_plan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(job, "validate_submission_binding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(job, "_validate_export_controller", lambda *_args: artifact)
    monkeypatch.setattr(job, "_validate_export_release", lambda *_args: None)
    monkeypatch.setattr(
        job, "_inspect_plan_bound_export", lambda *_args, **_kwargs: (artifact, [])
    )
    monkeypatch.setattr(hf, "inspect_export", lambda *_args, **_kwargs: (artifact, []))
    output = root / "HF_EXPORT_ACCEPTED.json"
    accepted = job.accept_export_job(
        plan_path=plan_path,
        submission_path=submission_path,
        controller_path=controller_path,
        release_path=release_path,
        output=output,
    )
    assert job.validate_export_accepted(accepted)["export"]["path"] == str(artifact_path)
    with pytest.raises(FileExistsError, match="already exists"):
        job.accept_export_job(
            plan_path=plan_path,
            submission_path=submission_path,
            controller_path=controller_path,
            release_path=release_path,
            output=output,
        )

    monkeypatch.setattr(job, "validate_plan", validate_plan)
    reload_plan = _plan(tmp_path, "reload")
    reload_plan["source"]["export_acceptance"] = {
        "path": str(output),
        "file_sha256": hf._hash(output),
        "receipt_sha256": accepted["sha256"].removeprefix("sha256:"),
    }
    reload_plan["source"]["export"] = job.validate_export_accepted(accepted, check_files=False)[
        "export"
    ]
    job.validate_plan(reload_plan, check_files=True, validate_historical=False)

    direct = copy.deepcopy(reload_plan)
    direct["source"].pop("export_acceptance")
    with pytest.raises(ValueError, match="source fields"):
        job.validate_plan(direct, check_files=False)


def _owner(kind: str, name: str, uid: str) -> list[dict]:
    return [{"kind": kind, "name": name, "uid": uid, "controller": True}]


def _event(
    kind: str,
    name: str,
    uid: str,
    *,
    owner: list[dict] | None = None,
    spec: dict | None = None,
    status: dict | None = None,
    version: int,
    event_type: str = "MODIFIED",
) -> dict:
    return {
        "type": event_type,
        "object": {
            "kind": kind,
            "metadata": {
                "namespace": job.DEV_NAMESPACE,
                "name": name,
                "uid": uid,
                "resourceVersion": str(version),
                "creationTimestamp": "2026-09-12T12:00:01Z",
                "ownerReferences": owner or [],
            },
            "spec": spec or {},
            "status": status or {},
        },
    }


def _reload_result(plan: dict) -> dict:
    value = {
        "schema": hf.RELOAD_SCHEMA,
        "status": "reload_validated",
        "run_name": plan["run_name"],
        "image": miles.IMAGE,
        "export_path": "/mnt/sfs/jobs/exact/model/EXPORT.json",
        "export_file_sha256": "1" * 64,
        "export_receipt_sha256": "2" * 64,
        "export_tensor_inventory_sha256": "3" * 64,
        "model_repo": hf.MODEL_REPO,
        "model_revision": hf.MODEL_REVISION,
        "runtime_tensor_count": 1,
        "artifact_only_mtp_tensor_count": 0,
        "runtime_layout_sha256": "4" * 64,
        "runtime_value_inventory_sha256": "5" * 64,
        "all_runtime_weights_loaded": True,
        "all_runtime_values_finite": True,
        "all_runtime_values_match_export": True,
        "source_export_unchanged": True,
        "native_prediction_probe": _prediction_probe(),
        "hf_prediction_probe": _prediction_probe(),
        "semantic_prediction_match": True,
        "optimizer_updates": 0,
        "rollouts": 0,
        "verifier_calls": 0,
        "forwards": 1,
        "backwards": 0,
        "checkpoint_writes": 0,
        "wandb_events": 0,
        "gpus": 1,
        "gpu_name": "synthetic",
        "peak_memory_bytes": 1,
        "external_gpu_release_verified": False,
        "serving_qualified": False,
        "completed_at": 1789214404.0,
    }
    value["sha256"] = digest(value)
    return value


def test_zero_gpu_batch_watch_compiles_exact_job_controller(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path, "export")
    submission = _submission(plan)
    artifact_path = Path(plan["artifact_path"]) / "EXPORT.json"
    artifact_path.parent.mkdir(parents=True)
    artifact = {"completed_at": 1789214404.0, "sha256": "sha256:" + "a" * 64}
    artifact_path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        job, "_inspect_plan_bound_export", lambda *_args, **_kwargs: (artifact, [])
    )
    directory = tmp_path / "watch-export"
    miles_event_evidence.start_capture(
        plan,
        submission,
        namespace_uid=job.DEV_NAMESPACE_UID,
        started_at="2026-09-12T12:00:00Z",
        directory=directory,
        kube_context=job.DEV_KUBE_CONTEXT,
    )
    job_uid = submission["api"]["run_id"]
    job_name = submission["api"]["run_name"]
    workload_uid = "33333333-3333-4333-8333-333333333333"
    pod_uid = "44444444-4444-4444-8444-444444444444"
    direct_job = _event(
        "Job",
        job_name,
        job_uid,
        spec={
            "ttlSecondsAfterFinished": 0,
            "template": {"spec": {"priorityClassName": "c2"}},
        },
        status={
            "conditions": [
                {
                    "type": "Complete",
                    "status": "True",
                    "lastTransitionTime": "2026-09-12T12:00:05Z",
                }
            ]
        },
        version=1,
    )
    workload = _event(
        "Workload",
        "exact-workload",
        workload_uid,
        owner=_owner("Job", job_name, job_uid),
        spec={"priorityClassName": "q2"},
        status={
            "conditions": [
                {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-09-12T12:00:02Z"}
            ]
        },
        version=2,
    )
    pod = _event(
        "Pod",
        "exact-pod",
        pod_uid,
        owner=_owner("Job", job_name, job_uid),
        spec={"containers": [{"name": "export", "resources": {"limits": {"memory": "768Gi"}}}]},
        status={
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "export",
                    "imageID": "containerd://registry/image@" + miles.IMAGE.rsplit("@", 1)[1],
                    "restartCount": 0,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                            "finishedAt": "2026-09-12T12:00:04Z",
                        }
                    },
                }
            ],
        },
        version=3,
    )
    lifecycle = [direct_job, workload, pod]
    lifecycle.extend(
        {
            **copy.deepcopy(raw),
            "type": "DELETED",
        }
        for raw in (direct_job, workload, pod)
    )
    for sequence, raw in enumerate(lifecycle):
        raw["object"]["metadata"]["resourceVersion"] = str(sequence + 1)
        miles_event_evidence.record_event(
            directory,
            raw,
            observed_at=f"2026-09-12T12:00:0{sequence + 1}Z",
            sequence=sequence,
        )
    controller_path = tmp_path / "HF_EXPORT_CONTROLLER_TERMINAL.json"
    controller = miles_event_evidence.compile_controller(
        plan,
        submission,
        directory=directory,
        api_status="SUCCEEDED",
        observed_at="2026-09-12T12:00:09Z",
        output=controller_path,
    )
    assert controller["job_uid"] == job_uid
    assert controller["gpus_per_worker"] == controller["total_gpus"] == 0
    assert controller["pods"][0]["owner_job_uid"] == job_uid
    job.validate_event_controller(controller, plan, submission)


def test_ttl_zero_watch_compiles_bound_hf_controller_and_release(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path, "reload")
    submission = _submission(plan)
    plan_path = Path(plan["artifact_path"])
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps(_reload_result(plan)))
    directory = tmp_path / "watch"
    miles_event_evidence.start_capture(
        plan,
        submission,
        namespace_uid=job.DEV_NAMESPACE_UID,
        started_at="2026-09-12T12:00:00Z",
        directory=directory,
        kube_context=job.DEV_KUBE_CONTEXT,
    )
    api_name = submission["api"]["run_name"]
    run_id = submission["api"]["run_id"]
    ids = {
        "RayJob": "22222222-2222-4222-8222-222222222222",
        "Workload": "33333333-3333-4333-8333-333333333333",
        "RayCluster": "44444444-4444-4444-8444-444444444444",
        "Pod": "55555555-5555-4555-8555-555555555555",
    }
    raycluster = "exact-raycluster"
    rayjob = _event(
        "RayJob",
        api_name,
        ids["RayJob"],
        spec={
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 0,
            "rayClusterSpec": {
                "headGroupSpec": {"template": {"spec": {"priorityClassName": "c1"}}}
            },
        },
        version=1,
        event_type="ADDED",
    )
    rayjob["object"]["metadata"]["labels"] = {"fleet.ai/run-id": run_id}
    workload = _event(
        "Workload",
        "exact-workload",
        ids["Workload"],
        owner=_owner("RayJob", api_name, ids["RayJob"]),
        status={
            "conditions": [
                {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-09-12T12:00:02Z"}
            ]
        },
        version=2,
    )
    cluster = _event(
        "RayCluster",
        raycluster,
        ids["RayCluster"],
        owner=_owner("RayJob", api_name, ids["RayJob"]),
        version=3,
        event_type="ADDED",
    )
    pod = _event(
        "Pod",
        "exact-pod",
        ids["Pod"],
        owner=_owner("RayCluster", raycluster, ids["RayCluster"]),
        spec={
            "containers": [{"name": "ray-head", "resources": {"limits": {"nvidia.com/gpu": "1"}}}]
        },
        status={
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "ray-head",
                    "imageID": "containerd://registry/image@" + miles.IMAGE.rsplit("@", 1)[1],
                    "restartCount": 0,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                            "finishedAt": "2026-09-12T12:00:04Z",
                        }
                    },
                }
            ],
        },
        version=4,
    )
    terminal = copy.deepcopy(rayjob)
    terminal["type"] = "DELETED"
    terminal["object"]["metadata"]["resourceVersion"] = "5"
    terminal["object"]["status"] = {"jobStatus": "SUCCEEDED", "rayClusterName": raycluster}
    deleted_workload = copy.deepcopy(workload)
    deleted_workload["type"] = "DELETED"
    deleted_workload["object"]["metadata"]["resourceVersion"] = "6"
    deleted_cluster = copy.deepcopy(cluster)
    deleted_cluster["type"] = "DELETED"
    deleted_cluster["object"]["metadata"]["resourceVersion"] = "7"
    deleted_pod = copy.deepcopy(pod)
    deleted_pod["type"] = "DELETED"
    deleted_pod["object"]["metadata"]["resourceVersion"] = "8"
    for sequence, event in enumerate(
        (
            rayjob,
            workload,
            cluster,
            pod,
            terminal,
            deleted_workload,
            deleted_cluster,
            deleted_pod,
        )
    ):
        miles_event_evidence.record_event(
            directory,
            event,
            observed_at=f"2026-09-12T12:00:0{sequence + 1}Z",
            sequence=sequence,
        )
    controller_path = tmp_path / "HF_RELOAD_CONTROLLER_TERMINAL.json"
    controller = miles_event_evidence.compile_controller(
        plan,
        submission,
        directory=directory,
        api_status="SUCCEEDED",
        observed_at="2026-09-12T12:00:06Z",
        output=controller_path,
    )
    assert controller["source_plan_sha256"] == "sha256:" + digest(plan)
    assert controller["runtime_bundle_sha256"] == submission["runtime_bundle_sha256"]
    assert controller["rayjob_uid"] == ids["RayJob"]
    release = miles_event_evidence.compile_release(
        plan,
        submission,
        controller_path=controller_path,
        absence={
            "raycluster_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
        },
        observed_at="2026-09-12T12:00:09Z",
        output=tmp_path / "HF_RELOAD_RELEASE.json",
    )
    assert release["active_gpus"] == 0 and release["pod_uids"] == [ids["Pod"]]
    result = _reload_result(plan)
    monkeypatch.setattr(
        hf,
        "inspect_export",
        lambda _path, _sha256, **_kwargs: (
            {
                "sha256": "sha256:" + result["export_receipt_sha256"],
                "tensor_inventory_sha256": result["export_tensor_inventory_sha256"],
                "source": {"prediction_probe": _prediction_probe()},
            },
            [],
        ),
    )
    accepted_export = hf._validate_reload_evidence(
        result=result,
        result_path=Path(plan["artifact_path"]),
        result_file_sha256=hf._hash(Path(plan["artifact_path"])),
        controller=controller,
        controller_path=controller_path,
        controller_file_sha256=hf._hash(controller_path),
        release=release,
        plan=plan,
        submission=submission,
    )
    assert accepted_export["tensor_inventory_sha256"] == result["export_tensor_inventory_sha256"]

    changed = copy.deepcopy(controller)
    changed["source_request_sha256"] = "sha256:" + "0" * 64
    changed.pop("sha256")
    changed["sha256"] = digest(changed)
    with pytest.raises(ValueError, match="controller evidence"):
        job.validate_event_controller(changed, plan, submission)

    event_path = directory / "EVENT-000001.json"
    event = json.loads(event_path.read_text())
    event["observed_at"] = "2026-09-12T12:01:00Z"
    event.pop("sha256")
    event["sha256"] = digest(event)
    event_path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="event-journal file changed"):
        job.validate_event_controller(controller, plan, submission)
