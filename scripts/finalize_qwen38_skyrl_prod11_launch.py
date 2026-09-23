"""Refresh, seal, and optionally create the reviewed prod11 launch once.

The slow all-context duplicate scan runs before the short-lived capacity and
server-preview proofs.  With ``--create``, the exact reviewed immutable plan,
request, GPU manifest, operator source closure, and policy are rebuilt and the
outer zero-GPU Job is created immediately in the same invocation.  There is no
inner Jobs API POST in this host process; the bounded outer operator owns that
single create rail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
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

BASE_SOURCE_HEAD = "f7452d7eafb33f2d84b724f07480b58c91826207"
PLAN_SHA256 = "sha256:f86ca0c93754def88d2f9053fb7fa012af3b6d29046846fa5ad229972ce9910f"
REQUEST_SHA256 = "sha256:954650c1a574d18f9eac19d3c8e00c38a24718235bbce074724e1bada7488048"
MANIFEST_SHA256 = "sha256:b414749c64eb86e057f18ac6fe12eded50506ae31a17edfdf2e587558f5673ea"
OPERATOR_SOURCE_SHA256 = "sha256:e5ebe5f3c21421562316503449095cd4081b52f6169038ace32a765ffaa9fb78"
PREFLIGHT_LAUNCH_SHA256 = "sha256:b9baf9d2e0bd925ffeeb28e58e751e6c3c7fdc04d10c77de17ff478ffcd1ad3d"
PREFLIGHT_RESULT_SHA256 = "sha256:d6f2b0cfba710fd9331ce9b053bea725952510fd6bc348a12795536320a21064"
PREFLIGHT_RECEIPT_SHA256 = "sha256:0996997c477a22f2056356737cdc8222f4a89b6c1d345a2e552e90c379215a17"
IDENTITY_PATH = Path("configs/qualification/qwen38-rl-reward-canary-prod11-identity-v1.json")
MIN_CREATE_CAPACITY_REMAINING_SECONDS = 30


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not one object")
    return value


def _source_head(root: Path, expected_head: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise ValueError("source head must be one full commit")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != expected_head:
        raise ValueError("reviewed finalizer head changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("reviewed finalizer worktree is dirty")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_HEAD, actual],
        cwd=root,
        check=False,
    ).returncode:
        raise ValueError("reviewed prod11 source is not an ancestor")


def _reviewed_directory(path: Path) -> Path:
    return operator_launch._operation_directory(path)


def _reviewed_immutables(path: Path) -> tuple[dict, dict, dict, dict]:
    root = _reviewed_directory(path)
    summary = _load(root / "SUMMARY.json")
    plan = _load(root / "PLAN.json")
    request = _load(root / "REQUEST.json")
    manifest = _load(root / "GPU_MANIFEST.json")
    if (
        summary.get("schema") != "cyber_skyrl_prod11_launch_review_packet_v1"
        or summary.get("status") != "sealed_for_independent_review_no_create"
        or summary.get("source_head") != BASE_SOURCE_HEAD
        or summary.get("create_authorized") is not False
        or summary.get("plan_sha256") != PLAN_SHA256
        or summary.get("request_sha256") != REQUEST_SHA256
        or summary.get("gpu_manifest_sha256") != MANIFEST_SHA256
        or summary.get("package", {}).get("source_sha256") != OPERATOR_SOURCE_SHA256
        or "sha256:" + digest(plan) != PLAN_SHA256
        or "sha256:" + digest(request) != REQUEST_SHA256
        or "sha256:" + digest(manifest) != MANIFEST_SHA256
    ):
        raise ValueError("reviewed prod11 immutable packet changed")
    return summary, plan, request, manifest


def _preflight(path: Path, plan: dict, request: dict) -> dict:
    packet = operator._packet(_load(path / "PREFLIGHT_PACKET.json"), "preflight")
    launch = prod10_finalizer._terminal_preflight(path / "OPERATOR_CREATE.jsonl")
    if launch != _load(path / "PREFLIGHT_LAUNCH_RESULT.json"):
        raise ValueError("preflight terminal copies differ")
    receipt = launch.get("observer", {}).get("receipt", {})
    if (
        packet.get("plan") != plan
        or packet.get("request") != request
        or launch.get("sha256") != PREFLIGHT_LAUNCH_SHA256
        or receipt.get("result_sha256") != PREFLIGHT_RESULT_SHA256
        or receipt.get("sha256") != PREFLIGHT_RECEIPT_SHA256
    ):
        raise ValueError("accepted prod11 preflight changed")
    return launch


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
        != {
            "nodes": request["workers"],
            "gpus": request["workers"] * request["gpus_per_worker"],
        }
    ):
        raise ValueError("fresh host capacity proof changed")
    return value


def validate_create_margin(
    *,
    gpu_previews: list[dict],
    gpu_duplicate: dict,
    operator_previews: list[dict],
    operator_duplicate: dict,
    capacity: dict,
    now: datetime | None = None,
) -> dict:
    value = prod10_finalizer.validate_remaining_ttl(
        gpu_previews=gpu_previews,
        gpu_duplicate=gpu_duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
        now=now,
    )
    if value["capacity_proof_remaining_seconds"] < MIN_CREATE_CAPACITY_REMAINING_SECONDS:
        raise ValueError("capacity proof lacks the JIT create margin")
    return value


def finalize(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    output = operator_launch._operation_directory(args.operation_directory)
    if any(output.iterdir()):
        raise ValueError("operation directory must be new and empty")
    _source_head(source_root, args.source_head)
    reviewed, reviewed_plan, reviewed_request, reviewed_manifest = _reviewed_immutables(
        args.reviewed_packet_directory
    )
    plan = _load(args.preflight_directory / "PLAN.json")
    request = _load(args.preflight_directory / "REQUEST.json")
    if (
        plan != reviewed_plan
        or request != reviewed_request
        or training.job_request(plan) != request
    ):
        raise ValueError("accepted plan/request differ from reviewed immutable bytes")
    preflight = _preflight(args.preflight_directory, plan, request)
    identity = historical.load_identity(source_root / IDENTITY_PATH)
    if identity.sealed_mapping()["sha256"] != reviewed.get("identity_sha256"):
        raise ValueError("prod11 identity changed")

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
    if expected != reviewed_manifest or "sha256:" + digest(expected) != MANIFEST_SHA256:
        raise ValueError("live Jobs preview changed the reviewed GPU manifest")

    # Slow, exhaustive history/Kubernetes/output scan first.  Every short-lived
    # proof is generated only after this completes.
    duplicate = launch_direct.duplicate_proof(identity, token=token)
    capacity = _capacity(request)
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
    provenance = launch_direct.sealed_dev_preview_provenance(
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
        dev_preview_provenance=provenance,
        duplicate_proof=duplicate,
        capacity_census=capacity,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    if proof.get("source_sha256") != OPERATOR_SOURCE_SHA256:
        raise ValueError("reviewed operator source closure changed")
    operator_previews = operator_launch.server_previews(package)
    operator_duplicate = operator_launch.duplicate_proof(package, previews=operator_previews)
    margin = validate_create_margin(
        gpu_previews=gpu_previews,
        gpu_duplicate=duplicate,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )

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
        prod10_finalizer._write_once(output, name, value)
    if (
        operator_job.build_operator_package(packet) != package
        or operator_job.validate_operator_package(package) != proof
    ):
        raise ValueError("sealed package did not rebuild exactly")
    summary = {
        "schema": "cyber_skyrl_prod11_jit_launch_summary_v1",
        "status": "sealed_and_validated",
        "finalizer_source_head": args.source_head,
        "base_operator_source_head": BASE_SOURCE_HEAD,
        "reviewed_packet_sha256": reviewed["packet_sha256"],
        "operation_directory": str(output),
        "plan_sha256": PLAN_SHA256,
        "request_sha256": REQUEST_SHA256,
        "gpu_manifest_sha256": MANIFEST_SHA256,
        "packet_sha256": packet["sha256"],
        "package": proof,
        "gpu_preview_sha256": [value["sha256"] for value in gpu_previews],
        "gpu_duplicate_sha256": duplicate["sha256"],
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
    value.add_argument("--preflight-directory", type=Path, required=True)
    value.add_argument("--reviewed-packet-directory", type=Path, required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    value.add_argument("--create", action="store_true")
    return value


def main() -> None:
    print(json.dumps(finalize(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
