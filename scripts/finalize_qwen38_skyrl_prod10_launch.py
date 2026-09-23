"""Seal and immediately submit one reviewed prod10 launch packet.

All slow duplicate work happens before time-bound server previews.  Execution
fails closed unless enough preview lifetime remains for the zero-GPU outer to
finish its own validation.  This script is intentionally specific to prod10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch
from cyber_post_train.jobs import Jobs, digest
from scripts import prepare_qwen38_skyrl_prod9_successor as prepare
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_reward_rayjob as historical


PLAN_SHA256 = "sha256:96cd9fa1343a17389c4cc0e4a6d9e3c89d2e21cc3ab45b5256bc9f0dd578fbb6"
REQUEST_SHA256 = "sha256:9ba0700bca6c88030cd761f7ae2394c7cda3ea58558aea4339fc6bc1a3cee501"
MANIFEST_SHA256 = "sha256:6693f547904794b6edc2b9f677a0baad2b524279959ef831128388f1e6fda329"
MIN_RUNTIME_PROOF_REMAINING_SECONDS = 180
MIN_HOST_CAPACITY_REMAINING_SECONDS = 10


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


def _remaining_seconds(value: object, *, ttl: int, now: datetime) -> float:
    if not isinstance(value, str):
        raise ValueError("time-bound proof timestamp is missing")
    try:
        checked = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as error:
        raise ValueError("time-bound proof timestamp is invalid") from error
    age = (now - checked).total_seconds()
    if age < 0:
        raise ValueError("time-bound proof is future-dated")
    return ttl - age


def validate_remaining_ttl(
    *,
    gpu_previews: list[dict],
    gpu_duplicate: dict,
    operator_previews: list[dict],
    operator_duplicate: dict,
    capacity: dict,
    now: datetime | None = None,
) -> dict:
    """Require a full runtime margin without renewing any timestamp."""
    checked_at = now or datetime.now(UTC)
    runtime = [
        *(value.get("checked_at") for value in gpu_previews),
        gpu_duplicate.get("checked_at"),
        *(value.get("checked_at") for value in operator_previews),
        operator_duplicate.get("checked_at"),
    ]
    runtime_remaining = min(
        _remaining_seconds(value, ttl=direct.EVIDENCE_MAX_AGE_SECONDS, now=checked_at)
        for value in runtime
    )
    capacity_remaining = _remaining_seconds(
        capacity.get("observed_at"),
        ttl=hardening.CAPACITY_MAX_AGE_SECONDS,
        now=checked_at,
    )
    if runtime_remaining < MIN_RUNTIME_PROOF_REMAINING_SECONDS:
        raise ValueError("runtime proof margin is insufficient; reseal, never submit")
    if capacity_remaining < MIN_HOST_CAPACITY_REMAINING_SECONDS:
        raise ValueError("capacity proof margin is insufficient; refresh, never submit")
    return {
        "checked_at": checked_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_proof_remaining_seconds": int(runtime_remaining),
        "capacity_proof_remaining_seconds": int(capacity_remaining),
    }


def _terminal_preflight(path: Path) -> dict:
    rows = [
        row
        for row in (json.loads(line) for line in path.read_text().splitlines())
        if row.get("state") == "TERMINAL_RESULT"
    ]
    if len(rows) != 1:
        raise ValueError("preflight launch terminal result is not unique")
    return {key: value for key, value in rows[0].items() if key != "state"}


def finalize(args: argparse.Namespace) -> dict:
    root = args.source_root.resolve()
    output = operator_launch._operation_directory(args.operation_directory)
    if any(output.iterdir()):
        raise ValueError("operation directory must be a new empty real directory")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_head) is None:
        raise ValueError("source head must be one full commit")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_head:
        raise ValueError("reviewed source head changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("reviewed source worktree is dirty")

    run_path = root / "configs/qualification/qwen38-rl-reward-canary-prod-v10.json"
    identity_path = root / "configs/qualification/qwen38-rl-reward-canary-prod10-identity-v1.json"
    run = _load(run_path)
    identity = historical.load_identity(identity_path)
    manifest = _load(args.manifest_result)["receipt"]["successor_manifest"]
    plan, request = prepare._compile(run, manifest, relative_to=run_path.parent)
    preflight = _terminal_preflight(args.preflight_journal)
    token = os.environ["FLEET_API_KEY"]
    with Jobs(token) as client:
        source_preview = client.preview(request)
    image_body = {
        "schema": direct.IMAGE_DEFAULT_IDENTITY_SCHEMA,
        "status": "passed",
        "image": request["image"],
        "uid": direct.RUNTIME_UID,
        "gid": direct.RUNTIME_GID,
        "gpus": 0,
    }
    image_identity = {**image_body, "receipt_sha256": digest(image_body)}
    expected = direct.manifest(
        plan,
        request,
        source_preview,
        identity=identity,
        image_identity_receipt=image_identity,
    )
    if (
        "sha256:" + digest(plan) != PLAN_SHA256
        or "sha256:" + digest(request) != REQUEST_SHA256
        or "sha256:" + digest(expected) != MANIFEST_SHA256
    ):
        raise ValueError("frozen prod10 science changed")

    # This is the slow host-wide scan.  It must precede all runtime TTL proofs.
    duplicate = launch_direct.duplicate_proof(identity, token=token)
    capacity = _load(args.capacity)
    capacity_body = {key: value for key, value in capacity.items() if key != "sha256"}
    if (
        capacity.get("sha256") != digest(capacity_body)
        or capacity.get("qualified") is not True
        or capacity.get("problems") != []
        or capacity.get("limits") != {"nodes": 10, "gpus": 80}
        or capacity.get("planned") != {"nodes": 1, "gpus": 8}
    ):
        raise ValueError("host capacity proof changed")

    # Generate every time-bound server proof only after slow work is complete.
    gpu_previews = [
        direct.validate_preview(
            plan,
            request,
            source_preview,
            expected,
            direct.server_dry_run(expected, context=context),
            context=context,
            identity=identity,
            image_identity_receipt=image_identity,
        )
        for context in (direct.DEV_CONTEXT, direct.PROD_CONTEXT)
    ]
    dev_preview_provenance = launch_direct.sealed_dev_preview_provenance(
        plan,
        request,
        source_preview,
        expected,
        gpu_previews[0],
        image_identity_receipt=image_identity,
        identity=identity,
    )
    packet = operator_job.launch_packet(
        identity=identity,
        plan=plan,
        request=request,
        preflight_launch_result=preflight,
        source_preview=source_preview,
        manifest_sha256=MANIFEST_SHA256,
        dev_preview=gpu_previews[0],
        dev_preview_provenance=dev_preview_provenance,
        duplicate_proof=duplicate,
        capacity_census=capacity,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    operator_previews = operator_launch.server_previews(package)
    operator_duplicate = operator_launch.duplicate_proof(package, previews=operator_previews)
    margin = validate_remaining_ttl(
        gpu_previews=gpu_previews,
        gpu_duplicate=duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )

    job = package.job
    container = job["spec"]["template"]["spec"]["containers"][0]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    encoded = json.dumps(job, sort_keys=True)
    gpu = expected["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"]
    if (
        job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or job["spec"]["template"]["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or proof["priority"] != "c1"
        or proof["queue_priority"] != "q1"
        or proof["gpus"] != 0
        or job["spec"]["backoffLimit"] != 0
        or "nvidia.com/gpu" in encoded
        or "controls-rw" not in mounts
        or "secretRef" not in encoded
        or expected["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or gpu["requests"].get("nvidia.com/gpu") != 8
        or gpu["limits"].get("nvidia.com/gpu") != 8
    ):
        raise ValueError("launch alert/resource policy changed")

    artifacts = {
        "PLAN.json": plan,
        "REQUEST.json": request,
        "SOURCE_PREVIEW.json": source_preview,
        "GPU_MANIFEST.json": expected,
        "GPU_PREVIEWS.json": gpu_previews,
        "GPU_DUPLICATE_PROOF.json": duplicate,
        "HOST_CAPACITY_CENSUS.json": capacity,
        "LAUNCH_PACKET.json": packet,
        "OPERATOR_SOURCE_CONFIG_MAP.json": package.source_config_map,
        "OPERATOR_PACKET_CONFIG_MAP.json": package.packet_config_map,
        "OPERATOR_JOB.json": package.job,
        "OPERATOR_PREVIEWS.json": operator_previews,
        "OPERATOR_DUPLICATE_PROOF.json": operator_duplicate,
    }
    for name, value in artifacts.items():
        _write_once(output, name, value)
    if (
        operator_job.build_operator_package(packet) != package
        or operator_job.validate_operator_package(package) != proof
    ):
        raise ValueError("sealed package did not rebuild exactly")
    summary = {
        "schema": "cyber_skyrl_prod10_jit_launch_summary_v1",
        "status": "sealed_and_validated",
        "source_head": args.source_head,
        "operation_directory": str(output),
        "plan_sha256": PLAN_SHA256,
        "request_sha256": REQUEST_SHA256,
        "gpu_manifest_sha256": MANIFEST_SHA256,
        "packet_sha256": packet["sha256"],
        "package": proof,
        "ttl_margin": margin,
        "capacity_file_sha256": "sha256:" + hashlib.sha256(args.capacity.read_bytes()).hexdigest(),
        "create_authorized": args.create,
    }
    _write_once(output, "SUMMARY.json", summary)
    if not args.create:
        return summary
    # Recheck immediately before the single mutation.  create_once writes its
    # durable do-not-retry intent before creating the exact root objects.
    validate_remaining_ttl(
        gpu_previews=gpu_previews,
        gpu_duplicate=duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )
    return operator_launch.create_once(
        package,
        previews=operator_previews,
        duplicates=operator_duplicate,
        operation_directory=output,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-head", required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    value.add_argument("--capacity", type=Path, required=True)
    value.add_argument("--manifest-result", type=Path, required=True)
    value.add_argument("--preflight-journal", type=Path, required=True)
    value.add_argument("--create", action="store_true")
    return value


def main() -> None:
    print(json.dumps(finalize(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
