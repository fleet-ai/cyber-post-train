"""Seal the zero-GPU prod10 in-cluster Jobs-preview difference probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch
from cyber_post_train.jobs import digest
from training import skyrl_prod9_direct as direct
from training import skyrl_prod10_operator as operator
from training import skyrl_prod10_preview_diff as preview_diff
from training import skyrl_reward_rayjob as historical

PLAN_SHA256 = "sha256:8f68c9502f394fea1d2ce2339a4908b50a26c3a5a534a4c03944a5aaa7a0d7d1"
REQUEST_SHA256 = "sha256:22965dae5ed43e1c62297522867b53daade798810cf75388b83eb56914eee5f1"
SOURCE_PREVIEW_SHA256 = "sha256:c33f585e0cfb15a90bc0f3034a3de6e5dfb87eeaba390caa71575865259d120d"
MANIFEST_SHA256 = "sha256:151fcb31d5ba37b06320defba764ec964ee34df87bf96b49d9c1f56b7e1cba68"
HOST_RECHECK_FILE_SHA256 = "sha256:82ba08a0b889d83fe8668e2e2419622c0d27539f02a097c9247862c201445112"


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not one object")
    return value


def _write_once(directory: Path, name: str, value: object) -> None:
    path = directory / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args: argparse.Namespace) -> dict:
    root = args.source_root.resolve()
    output = operator_launch._operation_directory(args.operation_directory)
    if any(output.iterdir()):
        raise ValueError("operation directory must be a new empty real directory")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != (
        args.source_head
    ):
        raise ValueError("reviewed source head changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("reviewed source worktree is dirty")

    v9 = args.v9_operation_directory
    plan, request = _load(v9 / "PLAN.json"), _load(v9 / "REQUEST.json")
    source_preview, manifest = _load(v9 / "SOURCE_PREVIEW.json"), _load(v9 / "GPU_MANIFEST.json")
    if (
        "sha256:" + digest(plan) != PLAN_SHA256
        or "sha256:" + digest(request) != REQUEST_SHA256
        or "sha256:" + digest(source_preview) != SOURCE_PREVIEW_SHA256
        or "sha256:" + digest(manifest) != MANIFEST_SHA256
    ):
        raise ValueError("exact v9 plan/request/preview/manifest binding changed")

    observer = _load(v9 / "OPERATOR_OBSERVER_RESULT.json")
    if (
        "sha256:" + hashlib.sha256((v9 / "OPERATOR_OBSERVER_RESULT.json").read_bytes()).hexdigest()
        != operator.launch_v9_failure_binding()["observer_result_file_sha256"]
        or observer.get("sha256") != operator.launch_v9_failure_binding()["observer_result_sha256"]
        or observer.get("receipt", {}).get("sha256")
        != operator.launch_v9_failure_binding()["failure_receipt_sha256"]
    ):
        raise ValueError("exact v9 terminal evidence changed")
    host = _load(v9 / "HOST_PREVIEW_RECHECK.json")
    host_file_sha = (
        "sha256:" + hashlib.sha256((v9 / "HOST_PREVIEW_RECHECK.json").read_bytes()).hexdigest()
    )
    if host_file_sha != HOST_RECHECK_FILE_SHA256:
        raise ValueError("host preview recheck file changed")
    if {**host, "file_sha256": host_file_sha} != operator.host_preview_recheck_binding():
        raise ValueError("host preview recheck binding changed")

    v1 = args.v1_operation_directory
    v1_binding = operator.preview_diff_v1_result_binding()
    v1_armed = _load(v1 / "OPERATOR_OBSERVER_ARMED.json")
    v1_creator = _load(v1 / "OPERATOR_OBSERVER_ARMED.json.created.json")
    v1_observer = _load(v1 / "OPERATOR_OBSERVER_RESULT.json")
    if (
        "sha256:" + hashlib.sha256((v1 / "OPERATOR_CREATE.jsonl").read_bytes()).hexdigest()
        != v1_binding["create_journal_file_sha256"]
        or "sha256:"
        + hashlib.sha256((v1 / "OPERATOR_OBSERVER_ARMED.json").read_bytes()).hexdigest()
        != v1_binding["observer_armed_file_sha256"]
        or v1_armed.get("sha256") != v1_binding["observer_armed_sha256"]
        or "sha256:"
        + hashlib.sha256(
            (v1 / "OPERATOR_OBSERVER_ARMED.json.created.json").read_bytes()
        ).hexdigest()
        != v1_binding["creator_binding_file_sha256"]
        or v1_creator.get("sha256") != v1_binding["creator_binding_sha256"]
        or "sha256:"
        + hashlib.sha256((v1 / "OPERATOR_OBSERVER_RESULT.json").read_bytes()).hexdigest()
        != v1_binding["observer_result_file_sha256"]
        or v1_observer.get("sha256") != v1_binding["observer_result_sha256"]
        or v1_observer.get("uid") != v1_binding["operator_job_uid"]
        or v1_observer.get("name") != v1_binding["operator_name"]
        or v1_observer.get("pod_names") != [v1_binding["operator_pod_name"]]
        or v1_observer.get("pod_uids") != [v1_binding["operator_pod_uid"]]
        or v1_observer.get("workload_name") != v1_binding["operator_workload_name"]
        or v1_observer.get("workload_uid") != v1_binding["operator_workload_uid"]
        or v1_observer.get("receipt") is not None
        or v1_observer.get("status") != "released_without_accepted_execution"
        or v1_observer.get("terminal_status") != "Succeeded"
        or v1_observer.get("exit_codes") != [0]
        or v1_observer.get("restarts") != 0
        or v1_observer.get("peak_gpus") != 0
        or v1_observer.get("release_observed_at") != v1_binding["release_observed_at"]
    ):
        raise ValueError("preview-difference v1 terminal evidence changed")

    identity_path = root / "configs/qualification/qwen38-rl-reward-canary-prod10-identity-v1.json"
    identity = historical.load_identity(identity_path)
    image_body = {
        "schema": direct.IMAGE_DEFAULT_IDENTITY_SCHEMA,
        "status": "passed",
        "image": request["image"],
        "uid": direct.RUNTIME_UID,
        "gid": direct.RUNTIME_GID,
        "gpus": 0,
    }
    image_identity = {**image_body, "receipt_sha256": digest(image_body)}
    pointers = preview_diff.pointer_proof(manifest)
    submitter = preview_diff.submitter_identity_proof(manifest)
    packet = operator_job.preview_difference_packet(
        identity=identity,
        plan=plan,
        request=request,
        source_preview_sha256=SOURCE_PREVIEW_SHA256,
        expected_pointers=pointers,
        expected_submitter_identity=submitter,
        image_identity_receipt=image_identity,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    previews = operator_launch.server_previews(package)
    duplicate = operator_launch.duplicate_proof(package, previews=previews)

    [container] = package.job["spec"]["template"]["spec"]["containers"]
    pod = package.job["spec"]["template"]["spec"]
    encoded_job = json.dumps(package.job, sort_keys=True)
    if (
        package.job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or package.job["spec"]["template"]["metadata"]["annotations"].get("fleet.ai/failure-alerts")
        != "off"
        or proof["priority"] != "c1"
        or proof["queue_priority"] != "q1"
        or proof["gpus"] != 0
        or package.job["spec"]["backoffLimit"] != 0
        or container.get("envFrom") != [{"secretRef": {"name": "fleet-api"}}]
        or "wandb-api" in encoded_job
        or "nvidia.com/gpu" in encoded_job
        or "controls-rw" in encoded_job
        or pod["volumes"][-1].get("persistentVolumeClaim", {}).get("readOnly") is not True
    ):
        raise ValueError("preview-difference alert/resource/secret policy changed")

    artifacts = {
        "PREVIEW_DIFF_PACKET.json": packet,
        "OPERATOR_SOURCE_CONFIG_MAP.json": package.source_config_map,
        "OPERATOR_PACKET_CONFIG_MAP.json": package.packet_config_map,
        "OPERATOR_JOB.json": package.job,
        "OPERATOR_PREVIEWS.json": previews,
        "OPERATOR_DUPLICATE_PROOF.json": duplicate,
    }
    for name, value in artifacts.items():
        _write_once(output, name, value)
    if (
        operator_job.build_operator_package(packet) != package
        or operator_job.validate_operator_package(package) != proof
    ):
        raise ValueError("preview-difference package did not rebuild exactly")
    summary = {
        "schema": "cyber_skyrl_prod10_preview_difference_summary_v1",
        "status": "sealed_for_independent_review_no_create",
        "source_head": args.source_head,
        "operation_directory": str(output),
        "plan_sha256": PLAN_SHA256,
        "request_sha256": REQUEST_SHA256,
        "source_preview_sha256": SOURCE_PREVIEW_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "expected_pointer_proof_sha256": pointers["sha256"],
        "expected_submitter_identity_proof_sha256": submitter["sha256"],
        "launch_v9_failure_sha256": operator.launch_v9_failure_binding()["sha256"],
        "host_preview_recheck_file_sha256": HOST_RECHECK_FILE_SHA256,
        "preview_diff_v1_result_sha256": v1_binding["sha256"],
        "packet_sha256": packet["sha256"],
        "package": proof,
        "preview_sha256": [value["sha256"] for value in previews],
        "duplicate_sha256": duplicate["sha256"],
        "jobs_api_preview_calls_at_runtime": 1,
        "jobs_api_create_calls_at_runtime": 0,
        "kubernetes_api_calls_at_runtime": 0,
        "kubernetes_service_account_token_mounted": False,
        "create_authorized": False,
    }
    _write_once(output, "SUMMARY.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-head", required=True)
    value.add_argument("--v9-operation-directory", type=Path, required=True)
    value.add_argument("--v1-operation-directory", type=Path, required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    return value


def main() -> None:
    print(json.dumps(prepare(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
