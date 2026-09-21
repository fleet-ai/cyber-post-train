"""Regression tests for the sealed successor direct root-RayJob rail."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cyber_post_train.jobs import JobsError, digest
from evals.fleet import opencode_self_hosted as fleet
from scripts.audit_qwen38_skyrl_launch_readiness import (
    compile_prod4,
    load,
)
from training import dev_cleanup_observer as cleanup
from training import skyrl_reward_rayjob as direct

ROOT = Path(__file__).resolve().parents[1]
CANARY_RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v7.json"
CANARY_MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v7.json"


def successor_metadata(run: dict) -> dict:
    value = load(CANARY_MANIFEST)
    assert value["name"] == run["name"]
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_private_successor_rebinds_only_run_identity(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "successor"
    source.mkdir(mode=0o700)
    prior = "chris-q38-rlreward-prod6"
    config = {
        "run_id": prior,
        "task": {"key": "safe-task", "version_id": "00000000-0000-0000-0000-000000000001"},
        "environment": {"id": "safe-environment"},
        "rl": {"context_tokens": 262144},
    }
    config["config_sha256"] = fleet.digest_without(config, "config_sha256")
    files = {}
    for split in ("train", "dev"):
        row = {
            "split": split,
            "env_class": "safe-environment",
            "prompt": [{"role": "user", "content": "private"}],
            "cyber_config_json": fleet.canonical_json(config).decode(),
        }
        payload = fleet.canonical_json(row) + b"\n"
        (source / f"{split}.jsonl").write_bytes(payload)
        (source / f"{split}.jsonl").chmod(0o600)
        files[split] = {
            "path": f"{split}.jsonl",
            "rows": 1,
            "sha256": fleet.sha256(payload),
            "max_prompt_tokens": 1,
        }
    for name in ("split.json", "task-set.json"):
        (source / name).write_text("{}\n")
        (source / name).chmod(0o600)
    manifest = {
        "schema": "cyber_skyrl_data_v1",
        "name": prior,
        "files": files,
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    (source / "manifest.json").write_bytes(fleet.canonical_json(manifest) + b"\n")
    (source / "manifest.json").chmod(0o600)

    result = direct.rebind_private_source_run_id(
        source,
        destination,
        prior_run_id=prior,
    )

    assert result["name"] == direct.RUN_NAME
    assert result["sha256"] == "sha256:" + digest(
        {key: value for key, value in result.items() if key != "sha256"}
    )
    for split in ("train", "dev"):
        prior_row = json.loads((source / f"{split}.jsonl").read_text())
        successor_row = json.loads((destination / f"{split}.jsonl").read_text())
        prior_config = json.loads(prior_row["cyber_config_json"])
        successor_config = json.loads(successor_row["cyber_config_json"])
        assert successor_config["run_id"] == direct.RUN_NAME
        assert successor_config["config_sha256"] == fleet.digest_without(
            successor_config, "config_sha256"
        )
        assert {
            key for key in successor_config if successor_config.get(key) != prior_config.get(key)
        } == {"run_id", "config_sha256"}
        assert {**successor_row, "cyber_config_json": ""} == {
            **prior_row,
            "cyber_config_json": "",
        }
        assert result["files"][split]["sha256"] == fleet.sha256(
            (destination / f"{split}.jsonl").read_bytes()
        )


def test_private_successor_rejects_stale_manifest_digest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl"):
        (source / name).write_text("{}\n")
    with pytest.raises(JobsError, match="predecessor manifest"):
        direct.rebind_private_source_run_id(
            source,
            tmp_path / "successor",
            prior_run_id="chris-q38-rlreward-prod6",
        )


def test_preflight_rejection_is_sanitized() -> None:
    error = direct.PreflightGateError("scientific_preflight", ValueError("private detail"))
    assert error.phase == "scientific_preflight"
    assert error.error_class == "ValueError"
    assert error.message_sha256 == hashlib.sha256(b"private detail").hexdigest()
    assert "private detail" not in vars(error).values()


@pytest.fixture(scope="module")
def plan_request() -> tuple[dict, dict]:
    run = load(CANARY_RUN)
    return compile_prod4(run, successor_metadata(run))


def _source_preview(plan: dict, request: dict) -> dict:
    placeholder = direct.RUN_NAME + "-00000000"
    resources = request["resources"]
    output_init = f"mkdir -p {plan['output_root']} && chown 1000:100 {plan['output_root']}"
    environment = [
        {"name": name, "value": value}
        for name, value in sorted(
            {
                **request["env"],
                "RUN_DIR": request["run_dir"],
                "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
                "FLEET_RUN_NAME": placeholder,
            }.items()
        )
    ]
    value = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": direct.NAMESPACE,
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": direct.RUN_NAME,
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/job-image": request["image"],
                "fleet.ai/run-dir": plan["output_root"],
            },
        },
        "spec": {
            "entrypoint": request["command"],
            "submissionMode": "HTTPMode",
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {},
                        "spec": {
                            "priorityClassName": "c1",
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": request["image"],
                                    "env": environment,
                                    "envFrom": [
                                        {"secretRef": {"name": "fleet-api"}},
                                        {"secretRef": {"name": "wandb-api"}},
                                        {"secretRef": {"name": placeholder + "-fleet-key"}},
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": resources["cpu_request"],
                                            "memory": resources["memory_request"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                        "limits": {
                                            "cpu": resources["cpu_limit"],
                                            "memory": resources["memory_limit"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                    },
                                }
                            ],
                            "initContainers": [
                                {
                                    "name": "sfs-init",
                                    "command": [
                                        "sh",
                                        "-c",
                                        output_init,
                                    ],
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    import yaml

    return {"name": placeholder, "warnings": [], "manifest_yaml": yaml.safe_dump(value)}


def _direct_render(plan: dict, request: dict, preview: dict) -> dict:
    value = copy.deepcopy(direct.manifest(plan, request, preview))
    value["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    value["spec"]["ttlSecondsAfterFinished"] = 0
    value["spec"]["rayClusterSpec"]["headGroupSpec"].update({"numOfHosts": 1, "scaleStrategy": {}})
    return value


def _cpu_render(expected: dict) -> dict:
    value = copy.deepcopy(expected)
    uid = "00000000-0000-0000-0000-000000000002"
    labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": expected["metadata"]["name"],
        "controller-uid": uid,
        "job-name": expected["metadata"]["name"],
    }
    value["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "generation": 1,
            "uid": uid,
            "labels": labels,
        }
    )
    value["status"] = {}
    value["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
    )
    value["spec"]["template"]["metadata"]["labels"] = labels
    pod = value["spec"]["template"]["spec"]
    pod.update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    pod["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    return value


def _synthetic_stage(plan: dict, tmp_path: Path) -> tuple[dict, Path]:
    staged_plan = copy.deepcopy(plan)
    source = tmp_path / "source"
    source.mkdir(parents=True)
    payloads = {"train.jsonl": b'{"row":"train"}\n', "dev.jsonl": b'{"row":"dev"}\n'}
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
    for split in ("train", "dev"):
        staged_plan["data"]["files"][split]["sha256"] = (
            "sha256:" + hashlib.sha256(payloads[split + ".jsonl"]).hexdigest()
        )
    for name, key, schema in (
        ("task-set.json", "selection_sha256", "cyber_rl_task_set_v1"),
        ("split.json", "split_sha256", "cyber_task_split_v1"),
    ):
        value = direct._seal({"schema": schema, "test_only": True})
        staged_plan["data"][key] = value["sha256"]
        (source / name).write_text(json.dumps(value))
    data_body = {key: value for key, value in staged_plan["data"].items() if key != "sha256"}
    staged_plan["data"] = {**data_body, "sha256": "sha256:" + digest(data_body)}
    destination = tmp_path / "published"
    staged_plan["arguments"]["data_manifest"] = str(destination / "manifest.json")
    (source / "manifest.json").write_text(json.dumps(staged_plan["data"]))
    archive = tmp_path / "payload.tar.gz"
    return direct.stage_plan(staged_plan, source, archive), archive


def test_direct_manifest_preserves_science_and_adds_only_transport_guards(plan_request) -> None:
    plan, request = plan_request
    preview = _source_preview(plan, request)
    source = direct._source(preview)
    value = direct.manifest(plan, request, preview)
    assert value["spec"]["entrypoint"] == source["spec"]["entrypoint"] == request["command"]
    assert value["spec"]["submissionMode"] == "HTTPMode"
    assert value["spec"]["suspend"] is value["spec"]["shutdownAfterJobFinishes"] is True
    assert value["spec"]["backoffLimit"] == 0
    assert value["spec"]["activeDeadlineSeconds"] == direct.MAXIMUM_SECONDS
    assert value["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert value["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    head = value["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    assert head["priorityClassName"] == "c1"
    assert head["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert head["containers"][0]["securityContext"]["runAsUser"] == 1000
    assert head["containers"][0]["securityContext"]["runAsGroup"] == 100
    assert [row["secretRef"]["name"] for row in head["containers"][0]["envFrom"]] == [
        "fleet-api",
        "wandb-api",
    ]
    assert head["initContainers"][0]["command"][1] == "-ec"
    assert head["initContainers"][0]["command"][2].startswith("mkdir ")


def test_direct_previews_reject_alert_priority_and_science_drift(plan_request) -> None:
    plan, request = plan_request
    preview = _source_preview(plan, request)
    expected = direct.manifest(plan, request, preview)
    proof = direct.validate_preview(
        plan,
        request,
        preview,
        expected,
        _direct_render(plan, request, preview),
        context=direct.PROD_CONTEXT,
    )
    assert proof["gpus"] == 8 and proof["failure_alerts"] == "off"
    for mutate in (
        lambda item: item["metadata"]["annotations"].pop("fleet.ai/failure-alerts"),
        lambda item: item["metadata"]["labels"].update({"kueue.x-k8s.io/priority-class": "q0"}),
        lambda item: item["spec"].update({"entrypoint": "false"}),
        lambda item: item["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
            "containers"
        ][0]["resources"]["limits"].update({"nvidia.com/gpu": 7}),
    ):
        changed = _direct_render(plan, request, preview)
        mutate(changed)
        with pytest.raises(JobsError):
            direct.validate_preview(
                plan, request, preview, expected, changed, context=direct.PROD_CONTEXT
            )


def test_private_stage_archive_is_bound_atomic_and_create_once(
    plan_request, tmp_path, monkeypatch
) -> None:
    plan, _ = plan_request
    staged, archive = _synthetic_stage(plan, tmp_path)
    second_source = tmp_path / "source-copy"
    second_source.mkdir()
    for item in (tmp_path / "source").iterdir():
        (second_source / item.name).write_bytes(item.read_bytes())
    second = tmp_path / "payload-2.tar.gz"
    assert direct.build_stage_archive(second_source, second) == staged["archive"]
    monkeypatch.setattr(direct, "UPLOAD", archive)
    monkeypatch.setattr(direct.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(direct.os, "getegid", lambda: 100)
    receipt = direct.stage_runtime(staged)
    assert receipt["status"] == "published" and receipt["gpus"] == 0
    assert receipt["archive"] == staged["archive"]
    assert Path(staged["destination"]).exists()
    with pytest.raises(FileExistsError):
        direct.stage_runtime(staged)


def test_zero_gpu_jobs_and_server_previews_are_alert_safe(plan_request, tmp_path) -> None:
    plan, _ = plan_request
    staged, _ = _synthetic_stage(plan, tmp_path)
    for purpose, job in (
        ("data_stage", direct.stage_job_manifest(staged)),
        ("preflight", direct.preflight_job_manifest(plan)),
    ):
        raw = json.dumps(job, sort_keys=True)
        assert "nvidia.com/gpu" not in raw
        assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        proof = direct.validate_cpu_preview(
            job, _cpu_render(job), context=direct.PROD_CONTEXT, purpose=purpose
        )
        assert proof["gpus"] == 0 and proof["failure_alerts"] == "off"
        changed = _cpu_render(job)
        changed["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
        with pytest.raises(JobsError):
            direct.validate_cpu_preview(job, changed, context=direct.PROD_CONTEXT, purpose=purpose)


def test_production_cleanup_observer_is_root_rayjob_only(plan_request, tmp_path) -> None:
    plan, request = plan_request
    preview = _source_preview(plan, request)
    expected = direct.manifest(plan, request, preview)
    observer = cleanup.Observer(
        context=direct.PROD_CONTEXT,
        namespace=direct.NAMESPACE,
        kind="rayjob",
        name=direct.RUN_NAME,
        maximum_seconds=direct.MAXIMUM_SECONDS,
        expected_gpus=8,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected),
        armed_path=tmp_path / "armed.json",
        result_path=tmp_path / "result.json",
        profile="production-direct",
    )
    assert observer.armed_schema == "cyber_direct_cleanup_observer_armed_v1"
    cpu = cleanup.Observer(
        context=direct.PROD_CONTEXT,
        namespace=direct.NAMESPACE,
        kind="job",
        name=direct.PREFLIGHT_NAME,
        maximum_seconds=1800,
        expected_gpus=0,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(direct.preflight_job_manifest(plan)),
        armed_path=tmp_path / "cpu-armed.json",
        result_path=tmp_path / "cpu-result.json",
        profile="production-cpu",
    )
    assert cpu.result_schema == "cyber_direct_cleanup_observer_result_v1"
    with pytest.raises(cleanup.ObserverError, match="root RayJob"):
        cleanup.Observer(
            context=direct.PROD_CONTEXT,
            namespace=direct.NAMESPACE,
            kind="fleetjob",
            name=direct.RUN_NAME,
            maximum_seconds=direct.MAXIMUM_SECONDS,
            expected_gpus=8,
            plan_sha256="sha256:" + digest(plan),
            manifest_sha256="sha256:" + digest(expected),
            armed_path=tmp_path / "bad-armed.json",
            result_path=tmp_path / "bad-result.json",
            profile="production-direct",
        )


class EmptyJobs:
    def __init__(self, _token: str, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def all_runs(self) -> list[dict]:
        return []


def _launch_evidence(plan: dict, request: dict, preview: dict, staged: dict) -> tuple:
    expected = direct.manifest(plan, request, preview)
    direct_previews = [
        direct.validate_preview(
            plan,
            request,
            preview,
            expected,
            _direct_render(plan, request, preview),
            context=context,
        )
        for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT)
    ]
    stage_receipt = direct._seal(
        {
            "schema": direct.STAGE_RECEIPT_SCHEMA,
            "status": "published",
            "stage_plan_sha256": digest(staged),
            "stage_plan": staged,
            "training_plan_sha256": digest(plan),
            "destination": staged["destination"],
            "files": staged["files"],
            "archive": staged["archive"],
            "runtime_user": {"uid": 1000, "gid": 100},
            **staged["scientific_work"],
        }
    )
    cpu_previews = []
    for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT):
        for purpose, job in (
            ("data_stage", direct.stage_job_manifest(staged)),
            ("preflight", direct.preflight_job_manifest(plan)),
        ):
            cpu_previews.append(
                direct.validate_cpu_preview(job, _cpu_render(job), context=context, purpose=purpose)
            )
    preflight = direct._seal(
        {
            "schema": direct.PREFLIGHT_RECEIPT_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "counts": {"train": 1, "dev": 1},
            "planned_steps": 1,
            "native_parser_checked": True,
            "ordered_multi_tool_parser_checked": True,
            "chunk_continuation_checked": True,
            "compaction_checked": True,
            "stepwise_prompt_checked": True,
            "ordered_multi_tool_execution_checked": True,
            "output_absent": True,
            "wandb_create_once": {
                "entity": "thefleet",
                "project": "cyber-post-train",
                "run_id": direct.RUN_NAME,
                "resume": "never",
            },
            "wandb_remote_lookup": "deferred_to_runtime_start",
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    observer = direct._seal(
        {
            "schema": "cyber_direct_cleanup_observer_armed_v1",
            "status": "armed",
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "kind": "rayjob",
            "name": direct.RUN_NAME,
            "maximum_seconds": direct.MAXIMUM_SECONDS,
            "expected_gpus": 8,
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": "sha256:" + digest(expected),
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer_pid": os.getpid(),
        }
    )
    authorization = direct.authorize(
        plan,
        request,
        preview,
        expected,
        dev_preview=direct_previews[0],
        prod_preview=direct_previews[1],
        cpu_previews=cpu_previews,
        stage_receipt=stage_receipt,
        preflight_receipt=preflight,
        observer=observer,
    )
    return expected, authorization


def test_create_is_journaled_once_and_never_retried(plan_request, tmp_path, monkeypatch) -> None:
    plan, request = plan_request
    preview = _source_preview(plan, request)
    train = plan["data"]["files"]["train"]
    dev = plan["data"]["files"]["dev"]
    staged = direct._seal(
        {
            "schema": direct.STAGE_SCHEMA,
            "name": direct.STAGE_NAME,
            "destination": plan["arguments"]["data_manifest"].removesuffix("/manifest.json"),
            "files": [
                {"path": "dev.jsonl", "bytes": 1, "sha256": dev["sha256"].removeprefix("sha256:")},
                {"path": "manifest.json", "bytes": 1, "sha256": "1" * 64},
                {"path": "split.json", "bytes": 1, "sha256": "2" * 64},
                {"path": "task-set.json", "bytes": 1, "sha256": "3" * 64},
                {
                    "path": "train.jsonl",
                    "bytes": 1,
                    "sha256": train["sha256"].removeprefix("sha256:"),
                },
            ],
            "archive": {"bytes": 5, "sha256": "4" * 64},
            "plan_sha256": digest(plan),
            "image": plan["execution"]["image"],
            "scientific_work": {
                "task_rows_read": 0,
                "rollout_episodes": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
                "gpus": 0,
            },
        }
    )
    expected, authorization = _launch_evidence(plan, request, preview, staged)
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        arguments = argv[5:]
        if arguments[:1] == ["get"]:
            return NS(returncode=0, stdout=json.dumps({"items": []}), stderr="")
        if arguments[:2] == ["create", "--dry-run=server"]:
            return NS(
                returncode=0,
                stdout=json.dumps(_direct_render(plan, request, preview)),
                stderr="",
            )
        if arguments[:1] == ["create"]:
            assert (tmp_path / "DIRECT_RAYJOB_CREATE.jsonl").exists()
            return NS(
                returncode=0,
                stdout=json.dumps(_direct_render(plan, request, preview)),
                stderr="",
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(direct.os, "kill", lambda *_args: None)
    result = direct.create_once(
        tmp_path,
        plan,
        request,
        preview,
        expected,
        authorization,
        token="test-token",
        runner=runner,
        jobs_factory=EmptyJobs,
    )
    assert result["status"] == "created_once"
    mutating = [
        call for call in calls if call[5:6] == ["create"] and "--dry-run=server" not in call
    ]
    assert len(mutating) == 1
    with pytest.raises(JobsError, match="intent exists"):
        direct.create_once(
            tmp_path,
            plan,
            request,
            preview,
            expected,
            authorization,
            token="test-token",
            runner=runner,
            jobs_factory=EmptyJobs,
        )
    assert (
        len([call for call in calls if call[5:6] == ["create"] and "--dry-run=server" not in call])
        == 1
    )
