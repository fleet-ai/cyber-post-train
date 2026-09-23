"""Seal and optionally create the exact prod11-fast2 launch once.

Fresh GPU previews and capacity precede the exhaustive host Kubernetes/Jobs
identity scan, which is the last slow proof. The accepted zero-GPU preflight
binds prior SFS absence; the in-cluster JIT rail rechecks SFS immediately before
the sole inner POST. ``--create`` crosses exactly one outer-Job create rail only
while enough host-proof TTL remains for admission plus that preguard.
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
from cyber_post_train.gpu_capacity import live_capacity_census
from cyber_post_train.jobs import Jobs, digest
from scripts import finalize_qwen38_skyrl_prod10_launch as prod10_finalizer
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod9_training as training
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

BASE_SOURCE_HEAD = "3e7958eea4bc3dffe93af0d81fc8ce26f32d9545"
IDENTITY_PATH = Path("configs/qualification/qwen38-rl-reward-canary-prod11-fast2-identity-v1.json")
FAST1_RETIREMENT_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-fast1-proof-expiry-v1.json"
)
FAST1_RETIREMENT_FILE_SHA256 = (
    "sha256:0713e892576427d78e747063531c50fbe70c2521954076e8f922d76c3325573b"
)
FAST1_RETIREMENT_SELF_SHA256 = (
    "sha256:1e759c4a70f53278747075c3990e9bdf41fd4cee7b052ce92df0f9177752992e"
)
RUN_NAME = "chris-q38-rlreward-prod11-fast2"
OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2"
MIN_RUNTIME_REMAINING_SECONDS = 180
MIN_CAPACITY_REMAINING_SECONDS = 30


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not one object")
    return value


def _source_head(root: Path, expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("source head must be one full commit")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != expected:
        raise ValueError("reviewed fast2 source head changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("reviewed fast2 source is dirty")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_HEAD, actual],
        cwd=root,
        check=False,
    ).returncode:
        raise ValueError("fast2 source does not descend from fast1")


def _preflight(path: Path, plan: dict, request: dict) -> dict:
    packet = operator._packet(_load(path / "PREFLIGHT_PACKET.json"), "preflight")
    launch = prod10_finalizer._terminal_preflight(path / "OPERATOR_CREATE.jsonl")
    if launch != _load(path / "PREFLIGHT_LAUNCH_RESULT.json"):
        raise ValueError("fast2 preflight terminal copies differ")
    receipt = launch.get("observer", {}).get("receipt", {})
    if (
        packet.get("plan") != plan
        or packet.get("request") != request
        or launch.get("status") != "operator_succeeded_and_released"
        or receipt.get("status") != "passed"
        or receipt.get("phase") != "preflight"
        or receipt.get("gpus") != 0
        or launch.get("observer", {}).get("terminal_status") != "Succeeded"
        or launch.get("observer", {}).get("exit_codes") != [0]
        or launch.get("observer", {}).get("restarts") != 0
        or launch.get("observer", {}).get("peak_gpus") != 0
    ):
        raise ValueError("accepted fast2 preflight changed")
    return launch


def _fast1_retirement(source_root: Path) -> dict:
    path = source_root / FAST1_RETIREMENT_PATH
    raw = path.read_bytes()
    value = _load(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    outer = value.get("outer", {})
    if (
        "sha256:" + hashlib.sha256(raw).hexdigest() != FAST1_RETIREMENT_FILE_SHA256
        or value.get("sha256") != FAST1_RETIREMENT_SELF_SHA256
        or value.get("sha256") != "sha256:" + digest(body)
        or value.get("schema") != "cyber_skyrl_prod11_fast1_retirement_reconciliation_v1"
        or value.get("status") != "released_without_accepted_execution_identity_and_output_absent"
        or value.get("run_name") != "chris-q38-rlreward-prod11-fast1"
        or value.get("output_root") != "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1"
        or outer.get("job", {}).get("state") != "absent"
        or outer.get("pod", {}).get("state") != "absent"
        or outer.get("workload", {}).get("state") != "absent"
        or outer.get("source_config_map", {}).get("state") != "present"
        or outer.get("packet_config_map", {}).get("state") != "present"
        or value.get("terminal_evidence", {}).get("failure_stage") != "duplicate_before_guard"
        or value.get("terminal_evidence", {}).get("gpus") != 0
        or value.get("inner", {}).get("created") is not False
        or value.get("inner", {}).get("dev_run_name_inventory_matches") != 0
        or value.get("inner", {}).get("prod_run_name_inventory_matches") != 0
        or value.get("inner", {}).get("prod_workload_name_matches") != 0
        or value.get("jobs_api", {}).get("dev_match_count") != 0
        or value.get("jobs_api", {}).get("prod_match_count") != 0
        or value.get("sfs", {}).get("output_root_present") is not False
        or value.get("classification")
        != {
            "duplicate_conflict": False,
            "cause": "host_duplicate_proof_expired_before_runtime_preguard",
            "retry_same_identity": False,
        }
    ):
        raise ValueError("fast1 retirement evidence changed")
    return value


def _capacity(request: dict) -> dict:
    value = live_capacity_census(
        direct.PROD_CONTEXT,
        owner_prefixes=tuple(hardening.PROJECT_OWNER_PREFIXES),
        max_nodes=launch_direct.MAX_NODES,
        max_gpus=launch_direct.MAX_GPUS,
        planned_nodes=request["workers"],
        planned_gpus=request["workers"] * request["gpus_per_worker"],
    )
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("sha256") != digest(body)
        or value.get("qualified") is not True
        or value.get("problems") != []
        or value.get("limits") != {"nodes": launch_direct.MAX_NODES, "gpus": launch_direct.MAX_GPUS}
        or value.get("planned")
        != {"nodes": request["workers"], "gpus": request["workers"] * request["gpus_per_worker"]}
    ):
        raise ValueError("fresh fast2 capacity proof changed")
    return value


def _fresh_gpu_proofs(
    *, plan: dict, request: dict, identity: historical.RailIdentity, token: str
) -> tuple[dict, dict, list[dict], dict]:
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
    previews = [
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
    provenance = launch_direct.sealed_dev_preview_provenance(
        plan,
        request,
        source_preview,
        expected,
        previews[0],
        image_identity_receipt=image_identity,
        identity=identity,
    )
    return source_preview, expected, previews, provenance


def validate_create_margin(
    *,
    gpu_previews: list[dict],
    host_identity: dict,
    operator_previews: list[dict],
    operator_duplicate: dict,
    capacity: dict,
    now: datetime | None = None,
) -> dict:
    checked_at = now or datetime.now(UTC)
    runtime = [
        *(value.get("checked_at") for value in gpu_previews),
        host_identity.get("checked_at"),
        *(value.get("checked_at") for value in operator_previews),
        operator_duplicate.get("checked_at"),
    ]
    runtime_remaining = min(
        prod10_finalizer._remaining_seconds(
            value,
            ttl=direct.EVIDENCE_MAX_AGE_SECONDS,
            now=checked_at,
        )
        for value in runtime
    )
    capacity_remaining = prod10_finalizer._remaining_seconds(
        capacity.get("observed_at"),
        ttl=hardening.CAPACITY_MAX_AGE_SECONDS,
        now=checked_at,
    )
    if runtime_remaining < MIN_RUNTIME_REMAINING_SECONDS:
        raise ValueError("fast2 runtime proof lacks admission and preguard margin")
    if capacity_remaining < MIN_CAPACITY_REMAINING_SECONDS:
        raise ValueError("fast2 capacity proof lacks create margin")
    return {
        "checked_at": checked_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_proof_remaining_seconds": int(runtime_remaining),
        "capacity_proof_remaining_seconds": int(capacity_remaining),
    }


def finalize(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    output = operator_launch._operation_directory(args.operation_directory)
    if any(output.iterdir()):
        raise ValueError("operation directory must be new and empty")
    _source_head(source_root, args.source_head)
    preflight_dir = operator_launch._operation_directory(args.preflight_directory)
    plan = _load(preflight_dir / "PLAN.json")
    request = _load(preflight_dir / "REQUEST.json")
    data_manifest = _load(preflight_dir / "SUCCESSOR_MANIFEST.json")
    identity = historical.load_identity(source_root / IDENTITY_PATH)
    if (
        training.job_request(plan) != request
        or plan.get("run_name") != RUN_NAME
        or plan.get("output_root") != OUTPUT_ROOT
        or identity.run_name != RUN_NAME
        or identity.output_root != OUTPUT_ROOT
        or plan.get("arguments", {}).get("eval_before_train") is not False
        or plan.get("arguments", {}).get("steps") != 1
        or plan.get("arguments", {}).get("groups") != 1
        or plan.get("arguments", {}).get("samples_per_prompt") != 8
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("failureAlerts") is not False
        or data_manifest.get("sha256") != plan.get("data", {}).get("sha256")
    ):
        raise ValueError("reviewed fast2 immutable inputs changed")
    preflight = _preflight(preflight_dir, plan, request)
    fast1_retirement = _fast1_retirement(source_root)
    token = os.environ["FLEET_API_KEY"]

    source_preview, expected, gpu_previews, provenance = _fresh_gpu_proofs(
        plan=plan,
        request=request,
        identity=identity,
        token=token,
    )
    capacity = _capacity(request)
    # Mint the exhaustive host Kubernetes/Jobs identity receipt last. It makes
    # no SFS claim; runtime retains the authoritative fresh SFS pre-POST check.
    duplicate = launch_direct.host_identity_proof(identity, token=token)

    manifest_sha256 = "sha256:" + digest(expected)
    packet = operator_job.launch_packet(
        identity=identity,
        plan=plan,
        request=request,
        preflight_launch_result=preflight,
        source_preview=source_preview,
        manifest_sha256=manifest_sha256,
        dev_preview=gpu_previews[0],
        dev_preview_provenance=provenance,
        duplicate_proof=duplicate,
        capacity_census=capacity,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    operator_previews = operator_launch.server_previews(package)
    operator_duplicate = operator_launch.duplicate_proof(package, previews=operator_previews)
    margin = validate_create_margin(
        gpu_previews=gpu_previews,
        host_identity=duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )
    artifacts = {
        "PLAN.json": plan,
        "REQUEST.json": request,
        "SUCCESSOR_MANIFEST.json": data_manifest,
        "FAST1_RETIREMENT.json": fast1_retirement,
        "SOURCE_PREVIEW.json": source_preview,
        "GPU_MANIFEST.json": expected,
        "GPU_PREVIEWS.json": gpu_previews,
        "HOST_IDENTITY_PROOF.json": duplicate,
        "HOST_CAPACITY_CENSUS.json": capacity,
        "LAUNCH_PACKET.json": packet,
        "OPERATOR_SOURCE_CONFIG_MAP.json": package.source_config_map,
        "OPERATOR_PACKET_CONFIG_MAP.json": package.packet_config_map,
        "OPERATOR_JOB.json": package.job,
        "OPERATOR_PREVIEWS.json": operator_previews,
        "OPERATOR_DUPLICATE_PROOF.json": operator_duplicate,
    }
    for name, value in artifacts.items():
        prod10_finalizer._write_once(output, name, value)
    if operator_job.build_operator_package(packet) != package:
        raise ValueError("sealed fast2 package did not rebuild exactly")
    summary = {
        "schema": "cyber_skyrl_prod11_fast2_jit_launch_summary_v1",
        "status": "sealed_and_validated",
        "source_head": args.source_head,
        "operation_directory": str(output),
        "identity_sha256": identity.sealed_mapping()["sha256"],
        "run_name": identity.run_name,
        "output_root": identity.output_root,
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "data_manifest_sha256": data_manifest["sha256"],
        "gpu_manifest_sha256": manifest_sha256,
        "preflight_launch_sha256": preflight["sha256"],
        "preflight_result_sha256": preflight["observer"]["receipt"]["result_sha256"],
        "fast1_retirement_sha256": fast1_retirement["sha256"],
        "fast1_retirement_file_sha256": FAST1_RETIREMENT_FILE_SHA256,
        "packet_sha256": packet["sha256"],
        "package": proof,
        "gpu_preview_sha256": [value["sha256"] for value in gpu_previews],
        "host_identity_proof_sha256": duplicate["sha256"],
        "host_sfs_absence_claimed": False,
        "accepted_preflight_sfs_absence_receipt_sha256": preflight["observer"]["receipt"]["sha256"],
        "accepted_preflight_sfs_absence_result_sha256": preflight["observer"]["receipt"][
            "result_sha256"
        ],
        "runtime_jit_sfs_absence_required": True,
        "capacity_sha256": "sha256:" + capacity["sha256"],
        "operator_preview_sha256": [value["sha256"] for value in operator_previews],
        "operator_duplicate_sha256": operator_duplicate["sha256"],
        "ttl_margin": margin,
        "create_authorized": args.create,
    }
    prod10_finalizer._write_once(output, "SUMMARY.json", summary)
    if not args.create:
        return summary
    validate_create_margin(
        gpu_previews=gpu_previews,
        host_identity=duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )
    result = operator_launch.create_once(
        package,
        previews=operator_previews,
        duplicates=operator_duplicate,
        operation_directory=output,
    )
    prod10_finalizer._write_once(output, "LAUNCH_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-head", required=True)
    value.add_argument("--preflight-directory", type=Path, required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    value.add_argument("--create", action="store_true")
    return value


def main() -> None:
    print(json.dumps(finalize(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
