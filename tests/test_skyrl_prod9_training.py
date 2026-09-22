"""Fresh prod9 runtime-binding tests; synthetic only and no cluster calls."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import JobsError, digest
from scripts import prepare_qwen38_skyrl_prod9_successor as prod9_prepare
from training import skyrl_prod9_direct as prod9_direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod9_reload as prod9_reload
from training import skyrl_prod9_rollout as rollout
from training import skyrl_prod9_training as prod9_training
from training import skyrl_reward_rayjob as historical_direct

pytest_plugins = ("test_skyrl_training",)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_CLOSURE = ROOT / "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v8.json"


def test_prod9_reuses_only_the_proven_miles_shape_and_qwen38_reload_gate() -> None:
    comparison = json.loads(SOURCE_CLOSURE.read_text())["proven_recipe_comparison"]
    prod9 = comparison["prod9_one_update_recipe"]
    miles = comparison["miles_rank_safe_precedent"]
    reload_gate = comparison["qwen38_lr30_reload_precedent"]
    for binding in (
        prod9["source"],
        miles["evidence"],
        reload_gate["source"],
        reload_gate["qualification"],
    ):
        source = (SOURCE_CLOSURE.parent / binding["path"]).resolve()
        assert binding["file_sha256"] == "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    lr30 = json.loads((SOURCE_CLOSURE.parent / reload_gate["qualification"]["path"]).read_text())
    assert reload_gate["qualification"]["self_sha256"] == lr30["sha256"]
    controls = (
        "nodes",
        "gpus_per_node",
        "groups",
        "samples_per_prompt",
        "global_batch_size",
        "optimizer_updates",
        "learning_rate",
        "checkpoint_interval",
    )
    assert {key: prod9[key] for key in controls} == {key: miles[key] for key in controls}
    assert prod9["model"] == "Qwen/Qwen3.8-27B" and prod9["backend"] == "skyrl"
    assert miles["model"] == "Qwen/Qwen3.6-27B" and miles["backend"] == "miles"
    assert miles["reward_variation_observed"] is False
    assert miles["finite_nonzero_parameter_update_observed"] is False
    assert reload_gate["nodes"] == reload_gate["gpus"] == 1
    assert reload_gate["generated_tokens"] == 2
    assert reload_gate["optimizer_updates"] == 0
    assert hardening.verify_source_closure(SOURCE_CLOSURE)["tool_result_token_safe"] is True


@pytest.mark.parametrize(
    "relative",
    ("training/skyrl.py", "training/skyrl_training.py", *prod9_reload.RUNTIME_FILES),
)
def test_prod9_source_closure_rejects_transitive_runtime_byte_drift(
    tmp_path: Path, monkeypatch, relative: str
) -> None:
    """A self-consistent regenerated plan cannot authorize unreviewed runtime bytes."""
    value = json.loads(SOURCE_CLOSURE.read_text())
    review_root = tmp_path / "review-root"
    for binding in value["tool_surface_authority"]["local_code"].values():
        source = (SOURCE_CLOSURE.parent / binding["path"]).resolve()
        destination = review_root / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        binding["file_sha256"] = "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
    evidence = review_root / "configs/data/source-closure.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    value["sha256"] = "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    evidence.write_text(json.dumps(value))
    target = review_root / relative
    target.write_bytes(target.read_bytes() + b"\n# adversarial unreviewed runtime drift\n")
    monkeypatch.setattr(
        hardening,
        "__file__",
        str(review_root / "training/skyrl_prod9_hardening.py"),
    )
    with pytest.raises(ValueError, match="source closure file digest changed"):
        hardening.verify_source_closure(evidence)


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_RUNTIME_BUNDLE")
    if encoded is None:
        parts = [
            (key, value)
            for key, value in request["env"].items()
            if key.startswith("CYBER_RUNTIME_BUNDLE_")
        ]
        encoded = "".join(
            value for _, value in sorted(parts, key=lambda item: int(item[0].rsplit("_", 1)[1]))
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def _assert_hermetic_bundle_imports(request: dict, root: Path) -> None:
    """Import only from the transported closure, never from this checkout."""
    bundle = _bundle(request)
    for relative, source in bundle["files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib,sys;"
                f"sys.path.insert(0,{str(root)!r});"
                f"importlib.import_module({bundle['module']!r})"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _prod9_plan() -> tuple[dict, dict, historical_direct.RailIdentity]:
    identity = historical_direct.load_identity(
        ROOT / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
    )
    manifest = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    manifest["name"] = identity.run_name
    manifest["sha256"] = "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    run = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v9.json").read_text()
    )
    plan, request = prod9_prepare._compile(run, manifest)
    return plan, request, identity


def _source_preview(plan: dict, request: dict) -> dict:
    """A strict synthetic Jobs API preview, with no cluster interaction."""
    placeholder = request["name"] + "-00000000"
    environment = [
        {"name": key, "value": value}
        for key, value in sorted(
            {
                **request["env"],
                "RUN_DIR": request["run_dir"],
                "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
                "FLEET_RUN_NAME": placeholder,
            }.items()
        )
    ]
    resources = request["resources"]
    value = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": prod9_direct.NAMESPACE,
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": request["name"],
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/job-image": request["image"],
                "fleet.ai/run-dir": plan["output_root"],
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "backoffLimit": 0,
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
                                    "securityContext": prod9_direct._runtime_context(),
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
                                        (
                                            f"mkdir -p {plan['output_root']} && chown 1000:100 "
                                            f"{plan['output_root']}"
                                        ),
                                    ],
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    return {"name": placeholder, "warnings": [], "manifest_yaml": yaml.safe_dump(value)}


def _image_identity_receipt(request: dict) -> dict:
    body = {
        "schema": prod9_direct.IMAGE_DEFAULT_IDENTITY_SCHEMA,
        "status": "passed",
        "image": request["image"],
        "uid": 1000,
        "gid": 100,
        "gpus": 0,
    }
    return {**body, "receipt_sha256": digest(body)}


def _direct_render(value: dict) -> dict:
    rendered = copy.deepcopy(value)
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-21T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    rendered["spec"]["ttlSecondsAfterFinished"] = 0
    rendered["spec"]["rayClusterSpec"]["headGroupSpec"].update(
        {"numOfHosts": 1, "scaleStrategy": {}}
    )
    return rendered


def _cpu_render(value: dict) -> dict:
    """Normal server-side defaults for a synthetic zero-GPU Job preview."""
    rendered = copy.deepcopy(value)
    uid = "00000000-0000-0000-0000-000000000002"
    name = value["metadata"]["name"]
    rendered["metadata"].update(
        {
            "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation": 1,
            "uid": uid,
            "labels": {
                "batch.kubernetes.io/controller-uid": uid,
                "batch.kubernetes.io/job-name": name,
                "controller-uid": uid,
                "job-name": name,
            },
        }
    )
    rendered["status"] = {}
    rendered["spec"].update(
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
    rendered["spec"]["template"]["metadata"]["labels"] = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    rendered["spec"]["template"]["spec"].update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    rendered["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    return rendered


def _observer(
    kind: str,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    gpus: int,
    seconds: int,
    creator_binding_path: Path,
):
    return prod9_direct._seal(
        {
            "schema": "cyber_direct_cleanup_observer_armed_v1",
            "status": "armed",
            "context": prod9_direct.PROD_CONTEXT,
            "namespace": prod9_direct.NAMESPACE,
            "kind": kind,
            "name": name,
            "maximum_seconds": seconds,
            "expected_gpus": gpus,
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer_pid": os.getpid(),
            "creator_binding_path": str(creator_binding_path),
        }
    )


def _jobs_api_guard(
    root: Path,
    purpose: str,
    request: dict,
    plan_sha256: str,
    manifest_sha256: str,
    gpus: int,
    seconds: int,
) -> dict:
    value = prod9_direct._seal(
        {
            "schema": "cyber_jobs_api_prefix_guard_armed_v1",
            "status": "armed_non_destructive_prefix_guard",
            "context": prod9_direct.PROD_CONTEXT,
            "namespace": prod9_direct.NAMESPACE,
            "run_name_prefix": request["name"],
            "generated_name_pattern": "^" + re.escape(request["name"]) + r"-[a-f0-9]{8}$",
            "run_dir": request["run_dir"],
            "image": request["image"],
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "maximum_seconds": seconds,
            "expected_gpus": gpus,
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer_pid": os.getpid(),
            "prefix_collision_count_before_post": 0,
        }
    )
    prod9_direct.jobs_api_guard_path(root, purpose).write_text(json.dumps(value))
    return value


def _release(
    *,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    receipt: dict,
    uid: str,
) -> dict:
    return prod9_direct._seal(
        {
            "schema": "cyber_direct_cleanup_observer_result_v1",
            "status": "released",
            "context": prod9_direct.PROD_CONTEXT,
            "namespace": prod9_direct.NAMESPACE,
            "kind": "job",
            "name": name,
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "expected_gpus": 0,
            "peak_gpus": 0,
            "active_gpus": 0,
            "terminal_status": "Succeeded",
            "restarts": 0,
            "observer_error_class": "",
            "exit_codes": [0],
            "receipt": receipt,
            "uid": uid,
            "target_present": False,
            "pods_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "raycluster_present": False,
            "release_observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _stage_receipt(stage: dict, plan: dict, identity: historical_direct.RailIdentity) -> dict:
    files = [
        {"path": name, "bytes": 1, "sha256": "sha256:" + name.encode().hex().ljust(64, "0")}
        for name in ("manifest.json", "split.json", "task-set.json")
    ]
    files.extend(
        {
            "path": plan["data"]["files"][split]["path"],
            "bytes": 1,
            "sha256": plan["data"]["files"][split]["sha256"],
        }
        for split in ("train", "dev")
    )
    body = {
        "schema": prod9_training.STAGE_RECEIPT_SCHEMA,
        "status": "published",
        "stage_spec_sha256": stage["sha256"],
        "identity_sha256": identity.sealed_mapping()["sha256"],
        "source": identity.predecessor_data_root,
        "destination": identity.data_root,
        "predecessor_manifest_sha256": stage["predecessor_manifest_sha256"],
        "successor_manifest": plan["data"],
        "successor_manifest_sha256": plan["data"]["sha256"],
        "files": files,
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "task_rows_read": 0,
        "rollout_episodes": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
    }
    return {**body, "receipt_sha256": digest(body)}


def _preflight_receipt(plan: dict, request: dict, identity: historical_direct.RailIdentity) -> dict:
    body = {
        "schema": "cyber_skyrl_prod9_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "prod9_runtime": prod9_training._binding(),
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
        "planned_steps": plan["arguments"]["steps"],
        "output_absent": True,
        "wandb_create_once": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": identity.wandb_run_id,
            "resume": "never",
        },
    }
    return {**body, "receipt_sha256": digest(body)}


def test_prod9_request_bundles_fresh_entrypoint_and_historical_rail_rejects(
    prepared, tmp_path: Path
):
    plan = prod9_training.compile_rl(prepared.config, relative_to=prepared.state.tmp)
    request = prod9_training.job_request(plan)
    bundle = _bundle(request)

    assert plan["schema"] == prod9_training.SCHEMA
    assert plan["runtime_sha256"] == digest(prod9_training._runtime())
    assert plan["prod9_runtime"] == prod9_training._binding()
    assert bundle["module"] == prod9_training.MODULE
    assert bundle["files"]["training/skyrl_prod9_training.py"] == (
        Path(prod9_training.__file__).read_text()
    )
    assert bundle["files"]["training/skyrl_prod9_rollout.py"] == Path(rollout.__file__).read_text()
    assert "hardening.Recorder(" in bundle["files"]["training/skyrl_prod9_rollout.py"]
    _assert_hermetic_bundle_imports(request, tmp_path / "gpu")
    _assert_hermetic_bundle_imports(prod9_training.preflight_request(plan), tmp_path / "preflight")

    # The old direct rail starts through its historical compiler.  It must be
    # explicitly unavailable instead of rendering a fresh plan as prod8.
    assert prod9_training.reject_historical_direct_rail(plan) == {
        "status": "rejected",
        "reason": "historical_direct_rail_cannot_render_fresh_prod9_runtime",
    }


def test_prod9_stage_bundle_imports_hermetically(tmp_path: Path) -> None:
    _plan, _request, identity = _prod9_plan()
    predecessor = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    stage = prod9_training.stage_spec(identity, predecessor)
    _assert_hermetic_bundle_imports(prod9_training.stage_request(stage), tmp_path / "stage")


def test_prod9_limits_accept_manifest_and_recorder_surfaces() -> None:
    config = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json").read_text()
    )
    limits = config["limits"]
    hardening.validate_episode_limits(limits)
    hardening.validate_exact_episode_limits(limits)
    hardening.validate_episode_limits(
        {key: value for key, value in limits.items() if key != "response_tokens"}
    )
    with pytest.raises(rollout.rl_episode.InvalidEpisode, match="exact_horizon"):
        hardening.validate_exact_episode_limits(
            {key: value for key, value in limits.items() if key != "response_tokens"}
        )
    assert hardening.verify_source_closure(SOURCE_CLOSURE)["response_tokens"] == 4194304


def test_prod9_fresh_direct_renderer_and_cpu_preflight_are_alert_safe() -> None:
    plan, request, identity = _prod9_plan()
    assert request["command"].startswith(
        "mkdir " + plan["output_root"] + "/.prod9-training-create-claim-v1 && exec "
    )
    preview = _source_preview(plan, request)
    rayjob = prod9_direct.manifest(plan, request, preview, identity=identity)
    packet = prod9_direct.packet(plan, request, preview, identity=identity)
    rendered = _direct_render(rayjob)
    proof = prod9_direct.validate_preview(
        plan,
        request,
        preview,
        rayjob,
        rendered,
        context=prod9_direct.PROD_CONTEXT,
        identity=identity,
    )
    cpu_job = prod9_direct.preflight_job_manifest(plan, identity=identity)
    container = cpu_job["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item["value"] for item in container["env"]}

    assert rayjob["metadata"]["annotations"] == {
        "fleet.ai/run-id": rayjob["metadata"]["labels"]["fleet.ai/run-id"],
        "fleet.ai/job-image": request["image"],
        "fleet.ai/run-dir": plan["output_root"],
        "fleet.ai/failure-alerts": "off",
    }
    assert packet["submitted"] is False
    assert packet["failure_alerts"] == proof["failure_alerts"] == "off"
    assert proof["nodes"] == 1 and proof["gpus"] == 8
    assert cpu_job["metadata"]["annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert "nvidia.com/gpu" not in json.dumps(cpu_job, sort_keys=True)
    assert _bundle({"env": environment})["module"] == prod9_training.MODULE
    assert _bundle({"env": environment})["argv"][-3:] == [
        "--preflight",
        "--receipt",
        "/dev/termination-log",
    ]
    assert not hasattr(hardening, "create_once")

    cpu_rendered = _cpu_render(cpu_job)
    cpu_proof = prod9_direct.validate_cpu_preview(
        cpu_job,
        cpu_rendered,
        context=prod9_direct.PROD_CONTEXT,
        purpose="preflight",
    )
    assert cpu_proof["gpus"] == 0 and cpu_proof["failure_alerts"] == "off"
    cpu_rendered["metadata"]["labels"]["job-name"] = "other"
    with pytest.raises(JobsError, match="Kubernetes defaults"):
        prod9_direct.validate_cpu_preview(
            cpu_job,
            cpu_rendered,
            context=prod9_direct.PROD_CONTEXT,
            purpose="preflight",
        )

    missing_alert_preview = copy.deepcopy(preview)
    missing_alert_manifest = yaml.safe_load(missing_alert_preview["manifest_yaml"])
    missing_alert_manifest["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    missing_alert_preview["manifest_yaml"] = yaml.safe_dump(missing_alert_manifest)
    with pytest.raises(JobsError, match="root alert-off"):
        prod9_direct.manifest(plan, request, missing_alert_preview, identity=identity)

    retrying_preview = copy.deepcopy(preview)
    retrying_manifest = yaml.safe_load(retrying_preview["manifest_yaml"])
    retrying_manifest["spec"]["backoffLimit"] = 1
    retrying_preview["manifest_yaml"] = yaml.safe_dump(retrying_manifest)
    with pytest.raises(JobsError, match="execution changed"):
        prod9_direct.manifest(plan, request, retrying_preview, identity=identity)

    default_backoff_preview = copy.deepcopy(preview)
    default_backoff_manifest = yaml.safe_load(default_backoff_preview["manifest_yaml"])
    default_backoff_manifest["spec"].pop("backoffLimit")
    default_backoff_preview["manifest_yaml"] = yaml.safe_dump(default_backoff_manifest)
    accepted_default = prod9_direct.manifest(
        plan, request, default_backoff_preview, identity=identity
    )
    defaulted_render = _direct_render(accepted_default)
    defaulted_render["spec"]["backoffLimit"] = 0
    assert (
        prod9_direct.validate_preview(
            plan,
            request,
            default_backoff_preview,
            accepted_default,
            defaulted_render,
            context=prod9_direct.PROD_CONTEXT,
            identity=identity,
        )["status"]
        == "passed"
    )
    root_preview = copy.deepcopy(preview)
    root_manifest = yaml.safe_load(root_preview["manifest_yaml"])
    root_manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
        "securityContext"
    ] = {
        "privileged": False,
        "runAsUser": 1000,
        "runAsGroup": 100,
        "runAsNonRoot": True,
    }
    root_preview["manifest_yaml"] = yaml.safe_dump(root_manifest)
    with pytest.raises(JobsError, match="runtime security context"):
        prod9_direct.manifest(plan, request, root_preview, identity=identity)

    default_user_preview = copy.deepcopy(preview)
    default_user_manifest = yaml.safe_load(default_user_preview["manifest_yaml"])
    default_user_manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0].pop("securityContext")
    default_user_preview["manifest_yaml"] = yaml.safe_dump(default_user_manifest)
    with pytest.raises(JobsError, match="exact-image runtime-user receipt"):
        prod9_direct.manifest(plan, request, default_user_preview, identity=identity)
    assert (
        prod9_direct.manifest(
            plan,
            request,
            default_user_preview,
            identity=identity,
            image_identity_receipt=_image_identity_receipt(request),
        )
        == default_user_manifest
    )
    wrong_image = _image_identity_receipt(request)
    wrong_image["image"] = "invalid.example/image@sha256:" + "0" * 64
    wrong_image["receipt_sha256"] = digest(
        {key: value for key, value in wrong_image.items() if key != "receipt_sha256"}
    )
    with pytest.raises(JobsError, match="receipt changed"):
        prod9_direct.manifest(
            plan,
            request,
            default_user_preview,
            identity=identity,
            image_identity_receipt=wrong_image,
        )

    census = build_capacity_census(
        {"items": []},
        {"items": []},
        owner_prefixes=hardening.PROJECT_OWNER_PREFIXES,
        max_nodes=hardening.PROJECT_MAX_NODES,
        max_gpus=hardening.PROJECT_MAX_GPUS,
        planned_nodes=1,
        planned_gpus=8,
        observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    capacity = hardening.capacity_gate(
        plan,
        request,
        rayjob,
        identity=identity,
        reader=lambda _context, **_kwargs: census,
    )
    assert capacity["planned"] == {"nodes": 1, "gpus": 8}

    unannotated = copy.deepcopy(rayjob)
    unannotated["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="root failure-alert annotation"):
        hardening.capacity_gate(
            plan,
            request,
            unannotated,
            identity=identity,
            reader=lambda _context, **_kwargs: pytest.fail(
                "missing root annotation must stop before a cluster census"
            ),
        )

    rendered["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="direct RayJob"):
        prod9_direct.validate_preview(
            plan,
            request,
            preview,
            rayjob,
            rendered,
            context=prod9_direct.PROD_CONTEXT,
            identity=identity,
        )


def test_prod9_live_create_requires_the_operational_sfs_root(tmp_path: Path, monkeypatch) -> None:
    assert (
        Path("/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/prod9-create-once-v1")
        == hardening.CREATE_ONCE_ROOT
    )
    root = tmp_path.resolve() / "prod9-create-once-v1"
    owner = tmp_path.stat()
    monkeypatch.setattr(hardening, "CREATE_ONCE_ROOT", root)
    monkeypatch.setattr(prod9_direct, "RUNTIME_UID", owner.st_uid)
    monkeypatch.setattr(prod9_direct, "RUNTIME_GID", owner.st_gid)
    monkeypatch.setattr(prod9_direct.os, "geteuid", lambda: owner.st_uid)
    monkeypatch.setattr(prod9_direct.os, "getegid", lambda: owner.st_gid)

    assert prod9_direct.live_create_is_available() is False

    root.mkdir(mode=0o700)
    assert prod9_direct.live_create_is_available() is True

    root.rmdir()
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    root.symlink_to(target, target_is_directory=True)
    assert prod9_direct.live_create_is_available() is False


def test_prod9_preflight_replaces_the_preexisting_termination_file(
    tmp_path: Path, monkeypatch
) -> None:
    """A Kubernetes termination message is pre-created, not create-once."""
    target = tmp_path / "termination-log"
    target.write_text("stale\n")
    monkeypatch.setattr(prod9_training, "PREFLIGHT_RECEIPT", target)
    value = {"schema": "synthetic", "status": "passed", "gpus": 0}

    prod9_training._write_preflight_receipt(target, value)

    assert json.loads(target.read_text()) == {
        **value,
        "receipt_sha256": digest(value),
    }
    with pytest.raises(ValueError, match="receipt path"):
        prod9_training._write_preflight_receipt(tmp_path / "other", value)


def test_prod9_cpu_preflight_projects_only_the_required_wandb_secret() -> None:
    plan, _request, identity = _prod9_plan()
    predecessor = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )

    preflight = prod9_direct.preflight_job_manifest(plan, identity=identity)
    stage = prod9_direct.stage_job_manifest(
        prod9_training.stage_spec(identity, predecessor),
        identity=identity,
    )

    container = preflight["spec"]["template"]["spec"]["containers"][0]
    stage_container = stage["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "wandb-api"}}]
    assert "envFrom" not in stage_container


def test_prod9_failed_preflight_writes_only_a_sanitized_phase_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "termination-log"
    monkeypatch.setattr(prod9_training, "PREFLIGHT_RECEIPT", target)
    monkeypatch.setattr(prod9_training, "_PREFLIGHT_STAGE", "long_horizon_probe")
    plan = {"schema": "synthetic", "private": "must-not-enter-receipt"}

    prod9_training._write_preflight_failure_receipt(target, plan, AssertionError("secret"))

    value = json.loads(target.read_bytes())
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    assert value["receipt_sha256"] == digest(body)
    assert body == {
        "schema": prod9_training.PREFLIGHT_FAILURE_SCHEMA,
        "status": "failed",
        "stage": "long_horizon_probe",
        "error_class": "AssertionError",
        "plan_sha256": digest(plan),
        "gpus": 0,
        "runtime_user": {"uid": prod9_training.os.geteuid(), "gid": prod9_training.os.getegid()},
    }
    assert "secret" not in target.read_text()


def test_prod9_native_config_diagnostic_identifies_the_exact_failing_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        class SkyRLTrainConfig:
            @staticmethod
            def from_cli_overrides(_values: dict) -> object:
                raise AssertionError("private native detail")

    class Utils:
        @staticmethod
        def validate_cfg(_cfg: object) -> None:
            pytest.fail("validation must not run after parse failure")

    modules = {
        "skyrl.train.config.config": Config,
        "skyrl.train.utils.utils": Utils,
    }
    monkeypatch.setattr(prod9_training.skyrl, "overrides", lambda _args: {"exact": "values"})
    monkeypatch.setattr(
        prod9_training.skyrl_episode,
        "_module",
        lambda name, _sha: modules[name],
    )

    with pytest.raises(AssertionError, match="private native detail"):
        prod9_training._preflight_native_config(object())

    assert prod9_training._PREFLIGHT_STAGE == "native_config_parse"


def test_prod9_one_create_rail_rejects_a_historical_plan_before_any_live_check(
    tmp_path: Path,
) -> None:
    plan, request, identity = _prod9_plan()
    plan["schema"] = "cyber_skyrl_training_v1"
    with pytest.raises(JobsError, match="fresh runtime schema"):
        prod9_direct.create_once(
            tmp_path,
            plan,
            request,
            {},
            {},
            {},
            token="synthetic",
            identity=identity,
        )


def test_prod9_prior_stage_evidence_is_uid_bound_without_global_freshness() -> None:
    plan, _request, identity = _prod9_plan()
    predecessor = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    stage = prod9_training.stage_spec(identity, predecessor)
    stage_job = prod9_direct.stage_job_manifest(stage, identity=identity)
    authorization_sha256 = "sha256:" + "a" * 64
    uid = "00000000-0000-0000-0000-000000000099"
    created = prod9_direct._seal(
        {
            "schema": prod9_direct.CPU_CREATED_SCHEMA,
            "status": "created_once",
            "purpose": "stage",
            "name": identity.stage_name,
            "plan_sha256": stage["sha256"],
            "manifest_sha256": "sha256:" + digest(stage_job),
            "authorization_sha256": authorization_sha256,
            "job_uid": uid,
            "created_at": "2026-09-21T00:00:00Z",
        }
    )
    assert (
        prod9_direct._cpu_created(
            created,
            purpose="stage",
            name=identity.stage_name,
            plan_sha256=stage["sha256"],
            manifest_sha256="sha256:" + digest(stage_job),
            authorization_sha256=authorization_sha256,
        )["job_uid"]
        == uid
    )

    receipt = {"immutable": True}
    release = _release(
        name=identity.stage_name,
        plan_sha256=stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_job),
        receipt=receipt,
        uid=uid,
    )
    release = prod9_direct._seal(
        {
            **{key: value for key, value in release.items() if key != "sha256"},
            "release_observed_at": "2026-09-21T00:00:00Z",
        }
    )
    assert (
        prod9_direct._cpu_release(
            release,
            receipt,
            name=identity.stage_name,
            plan_sha256=stage["sha256"],
            manifest_sha256="sha256:" + digest(stage_job),
            fresh=False,
        )["uid"]
        == uid
    )
    with pytest.raises(JobsError, match="stale"):
        prod9_direct._cpu_release(
            release,
            receipt,
            name=identity.stage_name,
            plan_sha256=stage["sha256"],
            manifest_sha256="sha256:" + digest(stage_job),
            fresh=True,
        )


def test_prod9_stage_and_one_create_rail_are_fresh_and_alert_safe(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the entire rail synthetically; no Kubernetes client is used."""
    plan, request, identity = _prod9_plan()
    global_root = tmp_path / "global-create-once"
    global_root.mkdir()
    monkeypatch.setattr(hardening, "CREATE_ONCE_ROOT", global_root)
    training_root = hardening.training_operation_root(plan)
    training_root.mkdir()
    predecessor = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    stage = prod9_training.stage_spec(identity, predecessor)
    stage_root = hardening.stage_operation_root(stage)
    stage_root.mkdir()
    stage_job = prod9_direct.stage_job_manifest(stage, identity=identity)
    stage_dev = prod9_direct.validate_cpu_preview(
        stage_job, _cpu_render(stage_job), context=prod9_direct.DEV_CONTEXT, purpose="stage"
    )
    stage_prod = prod9_direct.validate_cpu_preview(
        stage_job, _cpu_render(stage_job), context=prod9_direct.PROD_CONTEXT, purpose="stage"
    )
    stage_observer = _observer(
        "job",
        identity.stage_name,
        stage["sha256"],
        "sha256:" + digest(stage_job),
        0,
        prod9_direct.CPU_MAXIMUM_SECONDS,
        hardening.creator_binding_path(stage_root, "stage"),
    )
    stage_auth = prod9_direct.authorize_stage(
        stage,
        stage_job,
        dev_preview=stage_dev,
        prod_preview=stage_prod,
        observer=stage_observer,
        identity=identity,
    )
    alternate_stage = tmp_path / "alternate-stage"
    alternate_stage.mkdir()
    with pytest.raises(JobsError, match="canonical durable directory"):
        prod9_direct.authorize_stage(
            stage,
            stage_job,
            dev_preview=stage_dev,
            prod_preview=stage_prod,
            observer=_observer(
                "job",
                identity.stage_name,
                stage["sha256"],
                "sha256:" + digest(stage_job),
                0,
                prod9_direct.CPU_MAXIMUM_SECONDS,
                alternate_stage / "STAGE_OBSERVER_ARMED.json.created.json",
            ),
            identity=identity,
        )

    created_uid = "00000000-0000-0000-0000-000000000003"

    def cpu_runner(command, **kwargs):
        if "--dry-run=server" in command:
            expected = json.loads(kwargs["input"])
            return NS(returncode=0, stdout=json.dumps(_cpu_render(expected)))
        if "get" in command:
            return NS(returncode=0, stdout=json.dumps({"items": []}))
        if "create" in command:
            expected = json.loads(kwargs["input"])
            return NS(
                returncode=0,
                stdout=json.dumps(
                    {
                        "metadata": {
                            "name": expected["metadata"]["name"],
                            "uid": created_uid,
                            "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                    }
                ),
            )
        pytest.fail(f"unexpected CPU command: {command}")

    stage_created = prod9_direct.create_stage_once(
        stage_root, stage, stage_job, stage_auth, identity=identity, runner=cpu_runner
    )
    staged = _stage_receipt(stage, plan, identity)
    stage_release = _release(
        name=identity.stage_name,
        plan_sha256=stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_job),
        receipt=staged,
        uid=created_uid,
    )

    preflight_job = prod9_direct.preflight_job_manifest(plan, identity=identity)
    preflight_dev = prod9_direct.validate_cpu_preview(
        preflight_job,
        _cpu_render(preflight_job),
        context=prod9_direct.DEV_CONTEXT,
        purpose="preflight",
    )
    preflight_prod = prod9_direct.validate_cpu_preview(
        preflight_job,
        _cpu_render(preflight_job),
        context=prod9_direct.PROD_CONTEXT,
        purpose="preflight",
    )
    preflight_observer = _observer(
        "job",
        identity.preflight_name,
        "sha256:" + digest(plan),
        "sha256:" + digest(preflight_job),
        0,
        prod9_direct.CPU_MAXIMUM_SECONDS,
        hardening.creator_binding_path(training_root, "preflight"),
    )
    preflight_auth = prod9_direct.authorize_preflight(
        plan,
        request,
        stage,
        stage_auth,
        stage_created,
        staged,
        stage_release,
        preflight_job,
        dev_preview=preflight_dev,
        prod_preview=preflight_prod,
        observer=preflight_observer,
        identity=identity,
    )
    alternate_preflight = tmp_path / "alternate-preflight"
    alternate_preflight.mkdir()
    with pytest.raises(JobsError, match="canonical durable directory"):
        prod9_direct.authorize_preflight(
            plan,
            request,
            stage,
            stage_auth,
            stage_created,
            staged,
            stage_release,
            preflight_job,
            dev_preview=preflight_dev,
            prod_preview=preflight_prod,
            observer=_observer(
                "job",
                identity.preflight_name,
                "sha256:" + digest(plan),
                "sha256:" + digest(preflight_job),
                0,
                prod9_direct.CPU_MAXIMUM_SECONDS,
                alternate_preflight / "PREFLIGHT_OBSERVER_ARMED.json.created.json",
            ),
            identity=identity,
        )
    preflight_created = prod9_direct.create_preflight_once(
        training_root,
        plan,
        request,
        stage,
        preflight_auth,
        identity=identity,
        runner=cpu_runner,
    )
    preflight = _preflight_receipt(plan, request, identity)
    preflight_release = _release(
        name=identity.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(preflight_job),
        receipt=preflight,
        uid=created_uid,
    )

    source_preview = _source_preview(plan, request)
    rayjob = prod9_direct.manifest(plan, request, source_preview, identity=identity)
    direct_dev = prod9_direct.validate_preview(
        plan,
        request,
        source_preview,
        rayjob,
        _direct_render(rayjob),
        context=prod9_direct.DEV_CONTEXT,
        identity=identity,
    )
    direct_prod = prod9_direct.validate_preview(
        plan,
        request,
        source_preview,
        rayjob,
        _direct_render(rayjob),
        context=prod9_direct.PROD_CONTEXT,
        identity=identity,
    )
    direct_observer = _jobs_api_guard(
        training_root,
        "training",
        request,
        "sha256:" + digest(plan),
        "sha256:" + digest(rayjob),
        8,
        prod9_direct.MAXIMUM_SECONDS,
    )
    fresh_checks = []
    original_fresh_at = prod9_direct._fresh_at

    def record_fresh(value, **kwargs):
        fresh_checks.append(value)
        return original_fresh_at(value, **kwargs)

    monkeypatch.setattr(prod9_direct, "_fresh_at", record_fresh)
    authorization = prod9_direct.authorize(
        plan,
        request,
        source_preview,
        rayjob,
        stage,
        stage_auth,
        stage_created,
        staged,
        stage_release,
        preflight_auth,
        preflight_created,
        preflight,
        preflight_release,
        dev_preview=direct_dev,
        prod_preview=direct_prod,
        observer=direct_observer,
        identity=identity,
    )
    assert fresh_checks == [
        preflight_release["release_observed_at"],
        direct_dev["checked_at"],
        direct_prod["checked_at"],
        direct_observer["armed_at"],
    ]
    alternate_training = tmp_path / "alternate-training"
    alternate_training.mkdir()
    alternate_guard = _jobs_api_guard(
        alternate_training,
        "training",
        request,
        "sha256:" + digest(plan),
        "sha256:" + digest(rayjob),
        8,
        prod9_direct.MAXIMUM_SECONDS,
    )
    alternate_guard = prod9_direct._seal({**alternate_guard, "observer_pid": os.getpid() + 1})
    with pytest.raises(JobsError, match="binding changed"):
        prod9_direct.authorize(
            plan,
            request,
            source_preview,
            rayjob,
            stage,
            stage_auth,
            stage_created,
            staged,
            stage_release,
            preflight_auth,
            preflight_created,
            preflight,
            preflight_release,
            dev_preview=direct_dev,
            prod_preview=direct_prod,
            observer=alternate_guard,
            identity=identity,
        )
    created_rayjob_uid = "00000000-0000-0000-0000-000000000004"
    jobs_api_run_id = "00000000-0000-0000-0000-000000000005"
    jobs_api_run_name = request["name"] + "-1a2b3c4d"
    mutations = []
    live_order = []

    class JobsClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def all_runs(self):
            live_order.append("jobs")
            return []

        def preview(self, submitted):
            assert submitted == request
            live_order.append("api-preview")
            return source_preview

        def request(self, method, path, **kwargs):
            assert (method, path, kwargs.get("json")) == ("POST", "/v1/runs", request)
            live_order.append("post")
            mutations.append((method, path))
            return {
                "name": jobs_api_run_name,
                "job_id": jobs_api_run_id,
                "run_dir": request["run_dir"],
                "status": "queued",
            }

    census = build_capacity_census(
        {"items": []},
        {"items": []},
        owner_prefixes=hardening.PROJECT_OWNER_PREFIXES,
        max_nodes=hardening.PROJECT_MAX_NODES,
        max_gpus=hardening.PROJECT_MAX_GPUS,
        planned_nodes=1,
        planned_gpus=8,
        observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    def direct_runner(command, **kwargs):
        if "--dry-run=server" in command:
            live_order.append("dry-run")
            return NS(returncode=0, stdout=json.dumps(_direct_render(json.loads(kwargs["input"]))))
        if "get" in command and jobs_api_run_name in command:
            live_order.append("exact-bind")
            created_resource = copy.deepcopy(rayjob)
            created_resource["metadata"].update(
                {
                    "name": jobs_api_run_name,
                    "uid": created_rayjob_uid,
                    "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            return NS(returncode=0, stdout=json.dumps(created_resource))
        if "get" in command:
            live_order.append("inventory")
            return NS(returncode=0, stdout=json.dumps({"items": []}))
        pytest.fail(f"unexpected direct command: {command}")

    created = prod9_direct.create_once(
        training_root,
        plan,
        request,
        source_preview,
        rayjob,
        authorization,
        token="synthetic",
        identity=identity,
        runner=direct_runner,
        jobs_factory=JobsClient,
        wandb_exists=lambda *_args: live_order.append("wandb") or False,
        capacity_reader=lambda _context, **_kwargs: live_order.append("capacity") or census,
    )

    assert created["rayjob_uid"] == created_rayjob_uid
    assert len(mutations) == 1
    assert live_order.index("capacity") < live_order.index("post") < live_order.index("exact-bind")
    journal = [
        json.loads(line)
        for line in (training_root / "PROD9_DIRECT_RAYJOB_CREATE.jsonl").read_text().splitlines()
    ]
    assert [row.get("state") for row in journal[:2]] == [
        "POST_INTENT_DO_NOT_RETRY",
        "POST_RESPONSE",
    ]
    assert journal[2]["status"] == "submitted_once_and_bound_exact_uid"
    assert journal[2]["capacity_gate_sha256"] == journal[0]["capacity_gate"]["sha256"]
    with pytest.raises(JobsError, match="create intent exists"):
        prod9_direct.create_once(
            training_root,
            plan,
            request,
            source_preview,
            rayjob,
            authorization,
            token="synthetic",
            identity=identity,
            runner=direct_runner,
            jobs_factory=JobsClient,
            wandb_exists=lambda *_args: False,
            capacity_reader=lambda _context, **_kwargs: census,
        )
    alternate = tmp_path / "alternate-create"
    alternate.mkdir()
    # The canonical journal wins even when a caller supplies a fresh alternate
    # directory: once intent exists, no alternate path can reach create.
    with pytest.raises(JobsError, match="create intent exists"):
        prod9_direct.create_once(
            alternate,
            plan,
            request,
            source_preview,
            rayjob,
            authorization,
            token="synthetic",
            identity=identity,
            runner=direct_runner,
            jobs_factory=JobsClient,
            wandb_exists=lambda *_args: False,
            capacity_reader=lambda _context, **_kwargs: census,
        )
    assert len(mutations) == 1


@pytest.mark.asyncio
async def test_prod9_generator_constructs_fresh_recorder_and_stamps_receipt(
    tmp_path: Path, monkeypatch
):
    calls = []

    class MarkerRecorder:
        def __init__(self, *args):
            calls.append(args)

    @asynccontextmanager
    async def passthrough_engine(engine, _tokenizer, _timeout):
        yield engine

    async def collect(_config, _directory, recorder, _parse, *, client):
        assert isinstance(recorder, MarkerRecorder)
        assert client is not None
        return [
            NS(
                reward=1.0,
                tokens=[1, 2],
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                metadata={"done_reason": "report_submitted"},
            )
        ]

    monkeypatch.setenv("FLEET_API_KEY", "synthetic")
    monkeypatch.setattr(rollout.hardening, "Recorder", MarkerRecorder)
    monkeypatch.setattr(rollout.skyrl_episode, "single_attempt_engine", passthrough_engine)
    monkeypatch.setattr(rollout.skyrl_episode, "_module", lambda *_args: NS(__file__="/tmp/helper"))
    monkeypatch.setattr(rollout.rl_episode, "collect", collect)
    monkeypatch.setattr(rollout.rl_episode, "validate_samples", lambda _samples: None)

    generator = object.__new__(rollout.Generator)
    generator.busy = generator.failed = False
    generator.root = tmp_path
    generator.manifest = {"sha256": "sha256:synthetic"}
    generator.tokenizer = object()
    generator.engine = object()
    generator.response_tokens = 8
    generator.concurrency = 1
    config = {"run_id": "synthetic", "rl": {"episode_seconds": 1}}
    generator._inputs = lambda _batch: (
        {"phase": "train", "global_step": 0, "trajectory_ids": [("0", 0)]},
        "synthetic-batch",
        [config],
    )
    result = await generator.generate(
        {"sampling_params": {}, "trajectory_ids": [NS(instance_id="0", repetition_id=0)]}
    )

    assert len(calls) == 1
    assert result["rewards"] == [1.0]
    collected = json.loads((tmp_path / "batches/synthetic-batch/COLLECTED.json").read_text())
    assert collected["recorder_implementation"] == rollout.RECORDER_IMPLEMENTATION


def test_prod9_preflight_fails_before_model_or_network_when_not_image_user(prepared, monkeypatch):
    plan = prod9_training.compile_rl(prepared.config, relative_to=prepared.state.tmp)
    monkeypatch.setattr(prod9_training.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prod9_training.os, "getegid", lambda: 0)
    with pytest.raises(ValueError, match="image user"):
        prod9_training.preflight(plan)
