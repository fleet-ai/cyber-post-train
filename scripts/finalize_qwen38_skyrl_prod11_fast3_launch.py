"""Seal, preview, and optionally create the exact prod11 Fast3 launch.

Fast3 is a bounded retry-policy successor, not an exact runtime-parity replay.
Immutable source and evidence files are explicit inputs until that source is
reviewed and frozen.  The default path performs previews and writes a sealed
no-create packet; ``--create`` crosses one outer zero-GPU Job create rail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch
from cyber_post_train.gpu_capacity import live_capacity_census
from cyber_post_train.jobs import Jobs, digest
from scripts import finalize_qwen38_skyrl_prod10_launch as prod10_finalizer
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

BASE_SOURCE_HEAD = "cc07933546abb82023cf0413b4538dcf9d992ed5"
SOURCE_MERGE_HEAD = operator.FAST3_SOURCE_MERGE_HEAD
LAUNCHER_MERGE_HEAD = operator.FAST3_LAUNCHER_MERGE_HEAD
LAUNCH_GATE_PATH = Path("configs/qualification/qwen38-rl-reward-canary-port-v10.json")
LAUNCH_GATE_FILE_SHA256 = "sha256:469d6168eb42eeaf1e8acf94790f036708ba9d3cab9f0cfc88d23b442f4a6cd9"
PORT_V9_FILE_SHA256 = "sha256:0a64f23677833b29bb4ddf21c7516352d75040b8b85c885e62490dbb8d791fd2"
PORT_V9_SELF_SHA256 = "sha256:c2c11ee405563bb6025bc6f7309184deb8f0ed9c2eb6213e549ceefb350e609d"
RUNTIME_V9_FILE_SHA256 = "sha256:96c9ab8ff66bfa7f435404d9ff5cd5d690c498ea0798a6fde46798d613a5789a"
RUNTIME_V9_SELF_SHA256 = "sha256:50f8122021a7e96f79fbe9e2c60816394ca5202dc9f38493f7776406a80d3bdd"
SCIENCE_FILE_SHA256 = "sha256:76bbabb6775c4a82d4d11a2a76c3b6c5e4198b7143e05a4a55cf73dad81b6080"
SCIENCE_SELF_SHA256 = operator.FAST3_PREDECESSOR_SCIENCE_SHA256
RETRY_FILE_SHA256 = "sha256:3ead20ef5330303687a377560dff0b097bef6f481194542503978b07788e577f"
RETRY_SELF_SHA256 = operator.FAST3_RETRY_POLICY_SHA256
RELOAD_FILE_SHA256 = "sha256:46c9ea1daeb6965b85f0eed8ab9dd2c50a13ef68916092b0cba306367031ee84"
DIAGNOSTIC_FILE_SHA256 = operator.FAST3_DIAGNOSTIC_FILE_SHA256
DIAGNOSTIC_SELF_SHA256 = operator.FAST3_DIAGNOSTIC_SELF_SHA256
FAST2_RETIREMENT_FILE_SHA256 = operator.FAST3_FAST2_RETIREMENT_FILE_SHA256
FAST2_RETIREMENT_SELF_SHA256 = operator.FAST3_FAST2_RETIREMENT_SELF_SHA256
LAUNCHER_ROOT = Path(__file__).resolve().parents[1]
MIN_RUNTIME_REMAINING_SECONDS = 180
MIN_CAPACITY_REMAINING_SECONDS = 30


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not one object")
    return value


def _source_file(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    try:
        candidate_identity = candidate.lstat()
        path = candidate.resolve(strict=True)
        relative = path.relative_to(root)
    except (ValueError, OSError) as exc:
        raise ValueError("Fast3 evidence file is outside the frozen source") from exc
    if candidate != path or candidate.is_symlink() or not stat.S_ISREG(candidate_identity.st_mode):
        raise ValueError("Fast3 evidence input is not a regular source file")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if tracked.returncode:
        raise ValueError("Fast3 evidence input is not tracked by the frozen source")
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative.as_posix()}"], cwd=root)
    if committed != path.read_bytes():
        raise ValueError("Fast3 evidence input differs from the frozen source")
    return path


def _launcher_file(value: Path) -> Path:
    path = _source_file(LAUNCHER_ROOT, value)
    if path.relative_to(LAUNCHER_ROOT) != LAUNCH_GATE_PATH:
        raise ValueError("Fast3 launch gate is not the exact launcher config")
    if _file_sha256(path) != LAUNCH_GATE_FILE_SHA256:
        raise ValueError("Fast3 launch gate bytes changed")
    return path


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _self_sha256(value: dict[str, Any]) -> str:
    supplied = value.get("sha256")
    expected = digest({key: item for key, item in value.items() if key != "sha256"})
    if supplied not in {expected, "sha256:" + expected}:
        raise ValueError("Fast3 evidence self digest changed")
    return "sha256:" + expected


def _expected_sha256(value: str, *, name: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"expected {name} digest must be one SHA-256")
    return value


def _source_head(root: Path, expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("source head must be one full commit")
    if expected != SOURCE_MERGE_HEAD:
        raise ValueError("source head is not the accepted Fast3 source merge")
    top = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
        ).strip()
    ).resolve()
    if top != root:
        raise ValueError("Fast3 source root is not the frozen worktree root")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != expected:
        raise ValueError("reviewed Fast3 source head changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("reviewed Fast3 source is dirty")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_HEAD, actual],
        cwd=root,
        check=False,
    ).returncode:
        raise ValueError("Fast3 source does not descend from the launcher base")


def _launcher_head(expected: str, *, source_head: str) -> None:
    roots = {
        Path(module.__file__).resolve().parents[1]
        for module in (operator_job, operator_launch, launch_direct, operator)
    }
    if roots != {LAUNCHER_ROOT}:
        raise ValueError("Fast3 launcher imports do not share one source root")
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("launcher head must be one full commit")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LAUNCHER_ROOT, text=True
    ).strip()
    if actual != expected:
        raise ValueError("reviewed Fast3 launcher head changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=LAUNCHER_ROOT, text=True
    ).strip():
        raise ValueError("reviewed Fast3 launcher source is dirty")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_head, actual],
        cwd=LAUNCHER_ROOT,
        check=False,
    ).returncode:
        raise ValueError("Fast3 launcher does not contain the frozen source head")


def _merge_identity(commit: str) -> dict[str, Any]:
    tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", commit], cwd=LAUNCHER_ROOT, text=True
    ).strip()
    parents = subprocess.check_output(
        ["git", "show", "-s", "--format=%P", commit], cwd=LAUNCHER_ROOT, text=True
    ).split()
    return {"commit": commit, "parents": parents, "tree": tree}


def _launch_gate(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the append-only authority that closes only the two merge gates."""
    value = operator.fast3_launch_gate(_load(path))
    parent_path = LAUNCHER_ROOT / "configs/qualification/qwen38-rl-reward-canary-port-v9.json"
    runtime_path = (
        LAUNCHER_ROOT / "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v9.json"
    )
    science_path = (
        LAUNCHER_ROOT
        / "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-predecessor-science-v1.json"
    )
    policy_path = (
        LAUNCHER_ROOT
        / "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-generation-retry-v1.json"
    )
    identity_path = (
        LAUNCHER_ROOT
        / "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-identity-v1.json"
    )
    run_path = LAUNCHER_ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v11-fast3.json"
    reload_path = LAUNCHER_ROOT / "training/skyrl_fast3_reload.py"
    diagnostic_path = (
        LAUNCHER_ROOT
        / "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-generation-failure-diagnostic-v1.json"
    )
    retirement_path = (
        LAUNCHER_ROOT
        / "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-fast2-retirement-v1.json"
    )
    parent = _load(parent_path)
    preserved = value.get("preserved_identity", {})
    sealed_files = (
        ("runtime_evidence", runtime_path, RUNTIME_V9_FILE_SHA256, RUNTIME_V9_SELF_SHA256),
        ("predecessor_science", science_path, SCIENCE_FILE_SHA256, SCIENCE_SELF_SHA256),
        ("generation_retry_policy", policy_path, RETRY_FILE_SHA256, RETRY_SELF_SHA256),
        ("failure_diagnostic", diagnostic_path, DIAGNOSTIC_FILE_SHA256, DIAGNOSTIC_SELF_SHA256),
        (
            "fast2_retirement",
            retirement_path,
            FAST2_RETIREMENT_FILE_SHA256,
            FAST2_RETIREMENT_SELF_SHA256,
        ),
        ("fast3_identity", identity_path, None, None),
    )
    for key, file_path, literal_file, literal_self in sealed_files:
        current = _load(file_path)
        bound = preserved.get(key, {})
        if (
            _file_sha256(file_path) != bound.get("file_sha256")
            or _self_sha256(current) != bound.get("self_sha256")
            or literal_file is not None
            and _file_sha256(file_path) != literal_file
            or literal_self is not None
            and _self_sha256(current) != literal_self
        ):
            raise ValueError("Fast3 append-only launch authorization changed")
    for key, file_path in (("reload_source", reload_path), ("run_profile", run_path)):
        if _file_sha256(file_path) != preserved.get(key, {}).get("file_sha256"):
            raise ValueError("Fast3 append-only launch authorization changed")
    historical_gate = {
        "blockers": [
            "fast3_source_pr_not_merged",
            "fast3_launch_chain_not_separately_bound",
        ],
        "preview_authorized": False,
        "submission_authorized": False,
    }
    qualification = plan.get("qualification", {})
    if not isinstance(qualification, dict):
        raise ValueError("Fast3 append-only launch authorization changed")
    fast3 = qualification.get("fast3", {})
    if (
        {key: _merge_identity(bound["commit"]) for key, bound in value["merge_evidence"].items()}
        != value["merge_evidence"]
        or _file_sha256(parent_path) != PORT_V9_FILE_SHA256
        or _self_sha256(parent) != PORT_V9_SELF_SHA256
        or _file_sha256(reload_path) != RELOAD_FILE_SHA256
        or parent.get("submission_gate") != historical_gate
        or qualification.get("qualification_self_sha256") != parent.get("sha256")
        or qualification.get("submission_gate") != historical_gate
        or fast3.get("qualification") != parent
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_MERGE_HEAD, "HEAD"],
            cwd=LAUNCHER_ROOT,
            check=False,
        ).returncode
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", LAUNCHER_MERGE_HEAD, "HEAD"],
            cwd=LAUNCHER_ROOT,
            check=False,
        ).returncode
    ):
        raise ValueError("Fast3 append-only launch authorization changed")
    return value


def _retry_policy(path: Path) -> tuple[dict[str, Any], str, str]:
    value = _load(path)
    self_sha = _self_sha256(value)
    if (
        value.get("schema") != "cyber_skyrl_generation_http_retry_policy_v1"
        or value.get("max_http_attempts") != 3
        or value.get("retryable_http_statuses")
        != {"exact": [429], "inclusive_ranges": [[500, 599]]}
        or value.get("backoff_seconds") != [1, 2]
        or value.get("transport_retry") is not False
        or value.get("follow_redirects") is not False
        or value.get("whole_episode_retry") is not False
        or value.get("failed_response_body_admitted") is not False
        or value.get("failed_response_tokens_admitted") is not False
        or value.get("admitted_response") != "first_2xx_json_only"
    ):
        raise ValueError("Fast3 bounded generation retry policy changed")
    return value, _file_sha256(path), self_sha


def _failure_diagnostic(path: Path) -> tuple[dict[str, Any], str, str]:
    value = _load(path)
    self_sha = _self_sha256(value)
    source = value.get("source")
    if (
        value.get("schema") != "cyber_rl_prod11_failure_diagnostic_v1"
        or value.get("status") != "completed"
        or value.get("sfs_read_only") is not True
        or not isinstance(source, dict)
        or not isinstance(source.get("plan_sha256"), str)
        or not isinstance(value.get("episode_failure"), dict)
        or not isinstance(value.get("native_failure"), dict)
    ):
        raise ValueError("prod11 source failure diagnostic changed")
    return value, _file_sha256(path), self_sha


def _fast2_retirement(path: Path) -> tuple[dict[str, Any], str, str]:
    value = _load(path)
    self_sha = _self_sha256(value)
    launch = value.get("gpu_launch")
    output = value.get("output_absence")
    if (
        value.get("schema") != "cyber_skyrl_prod11_fast2_retirement_reconciliation_v1"
        or value.get("status") != "retired_without_gpu_launch_identity_and_output_absent"
        or value.get("run_name") != operator.FAST3_IDENTITY.predecessor_run_name
        or value.get("output_root") != "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2"
        or value.get("current_absence", {}).get("all_match_counts_zero") is not True
        or not isinstance(launch, dict)
        or launch.get("outer_created") is not False
        or launch.get("jobs_post_attempted") is not False
        or launch.get("inner_created") is not False
        or launch.get("rayjob_created") is not False
        or launch.get("gpus_requested") != 0
        or launch.get("retry_same_identity") is not False
        or not isinstance(output, dict)
        or output.get("output_absent") is not True
        or output.get("fresh_sfs_lstat_in_this_receipt") is not False
    ):
        raise ValueError("Fast2 retirement evidence changed")
    return value, _file_sha256(path), self_sha


def _predecessor_science(
    path: Path,
    *,
    diagnostic_file_sha256: str,
    diagnostic_self_sha256: str,
    retirement_file_sha256: str,
    retirement_self_sha256: str,
    retry_policy_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    value = _load(path)
    self_sha = _self_sha256(value)
    scientific = value.get("scientific_predecessor", {})
    operational = value.get("operational_predecessor", {})
    recipe = value.get("normalized_science", {}).get("recipe", {})
    diagnostic = scientific.get("failure_diagnostic", {})
    retirement = operational.get("retirement", {})

    def normalized(item: object) -> str:
        return str(item) if str(item).startswith("sha256:") else "sha256:" + str(item)

    if (
        value.get("schema") != "cyber_skyrl_fast3_predecessor_science_v1"
        or scientific.get("run_name") != "chris-q38-rlreward-prod11"
        or operational.get("run_name") != operator.FAST3_IDENTITY.predecessor_run_name
        or operational.get("scientific_parity_claim") is not False
        or recipe.get("eval_before_train") is not True
        or recipe.get("steps") != 1
        or recipe.get("groups") != 1
        or recipe.get("samples_per_prompt") != 8
        or set(value.get("permitted_deltas", []))
        != {"fresh_identity_output_data_root", "port_successor_sha256", "generation_retry_policy"}
        or value.get("retry_evidence")
        != {"historical_http_status_known": False, "retryability_proven": False}
        or value.get("retry_policy_sha256") != retry_policy_sha256
        or normalized(diagnostic.get("file_sha256")) != diagnostic_file_sha256
        or normalized(diagnostic.get("self_sha256")) != diagnostic_self_sha256
        or diagnostic.get("http_status_known") is not False
        or diagnostic.get("retryability_proven") is not False
        or normalized(retirement.get("file_sha256")) != retirement_file_sha256
        or normalized(retirement.get("self_sha256")) != retirement_self_sha256
        or retirement.get("gpu_launch_attempted") is not False
    ):
        raise ValueError("Fast3 predecessor-science receipt changed")
    return value, _file_sha256(path), self_sha


def _preflight(
    path: Path,
    plan: dict[str, Any],
    request: dict[str, Any],
    launch_gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = operator._packet(_load(path / "PREFLIGHT_PACKET.json"), "preflight")
    successor = operator_job.fast3_successor_manifest(
        packet["stage"], packet["stage_launch_result"], launch_gate=launch_gate
    )
    rebuilt_plan = operator_job.fast3_plan_from_successor(successor, launch_gate=launch_gate)
    if (
        packet.get("identity") != operator.FAST3_IDENTITY.sealed_mapping()
        or packet.get("launch_gate") != launch_gate
        or packet.get("plan") != plan
        or packet.get("request") != request
        or rebuilt_plan != plan
        or launch_direct.job_request(rebuilt_plan, identity=operator.FAST3_IDENTITY) != request
        or successor != plan.get("data")
    ):
        raise ValueError("accepted zero-GPU Fast3 preflight changed")
    launch = prod10_finalizer._terminal_preflight(path / "OPERATOR_CREATE.jsonl")
    copied = _load(path / "PREFLIGHT_LAUNCH_RESULT.json")
    observer = launch.get("observer", {})
    receipt = observer.get("receipt", {}) if isinstance(observer, dict) else {}
    if (
        launch != copied
        or launch.get("status") != "operator_succeeded_and_released"
        or launch.get("package", {}).get("name") != operator.FAST3_OPERATOR_NAMES["preflight"]
        or launch.get("package", {}).get("packet_sha256") != packet.get("sha256")
        or receipt.get("status") != "passed"
        or receipt.get("phase") != "preflight"
        or receipt.get("gpus") != 0
        or observer.get("terminal_status") != "Succeeded"
        or observer.get("exit_codes") != [0]
        or observer.get("restarts") != 0
        or observer.get("peak_gpus") != 0
    ):
        raise ValueError("accepted zero-GPU Fast3 preflight changed")
    return launch, successor


def _capacity(request: dict[str, Any]) -> dict[str, Any]:
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
        or value.get("planned") != {"nodes": 1, "gpus": 8}
    ):
        raise ValueError("fresh Fast3 capacity proof changed")
    return value


def _fresh_gpu_proofs(
    *, plan: dict[str, Any], request: dict[str, Any], token: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    identity = operator.FAST3_IDENTITY
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
    expected = launch_direct.gpu_manifest(
        plan,
        request,
        source_preview,
        identity=identity,
        image_identity_receipt=image_identity,
    )
    previews = [
        launch_direct.validate_gpu_preview(
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
    gpu_previews: list[dict[str, Any]],
    host_identity: dict[str, Any],
    operator_previews: list[dict[str, Any]],
    operator_duplicate: dict[str, Any],
    capacity: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    checked_at = now or datetime.now(UTC)
    runtime = [
        *(value.get("checked_at") for value in gpu_previews),
        host_identity.get("checked_at"),
        *(value.get("checked_at") for value in operator_previews),
        operator_duplicate.get("checked_at"),
    ]
    runtime_remaining = min(
        prod10_finalizer._remaining_seconds(
            value, ttl=direct.EVIDENCE_MAX_AGE_SECONDS, now=checked_at
        )
        for value in runtime
    )
    capacity_remaining = prod10_finalizer._remaining_seconds(
        capacity.get("observed_at"),
        ttl=hardening.CAPACITY_MAX_AGE_SECONDS,
        now=checked_at,
    )
    if runtime_remaining < MIN_RUNTIME_REMAINING_SECONDS:
        raise ValueError("Fast3 runtime proof lacks admission and preguard margin")
    if capacity_remaining < MIN_CAPACITY_REMAINING_SECONDS:
        raise ValueError("Fast3 capacity proof lacks create margin")
    return {
        "checked_at": checked_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_proof_remaining_seconds": int(runtime_remaining),
        "capacity_proof_remaining_seconds": int(capacity_remaining),
    }


def _validate_topology(expected: dict[str, Any], package: operator_job.OperatorPackage) -> None:
    cluster = expected.get("spec", {}).get("rayClusterSpec", {})
    pod = cluster.get("headGroupSpec", {}).get("template", {}).get("spec", {})
    containers = pod.get("containers", [])
    resources = containers[0].get("resources", {}) if len(containers) == 1 else {}
    job = package.job
    outer_pod = job.get("spec", {}).get("template", {})
    if (
        expected.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
        or expected.get("metadata", {}).get("labels", {}).get("kueue.x-k8s.io/priority-class")
        != "q1"
        or pod.get("priorityClassName") != "c1"
        or cluster.get("workerGroupSpecs") not in (None, [])
        or resources.get("requests", {}).get("nvidia.com/gpu") != 8
        or resources.get("limits", {}).get("nvidia.com/gpu") != 8
        or job.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
        or outer_pod.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts")
        != "off"
        or outer_pod.get("spec", {}).get("priorityClassName") != "c1"
        or job.get("metadata", {}).get("labels", {}).get("kueue.x-k8s.io/priority-class") != "q1"
        or "nvidia.com/gpu" in json.dumps(job, sort_keys=True)
    ):
        raise ValueError("Fast3 alert, priority, or one-node topology changed")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output = operator_launch._operation_directory(args.operation_directory)
    if any(output.iterdir()):
        raise ValueError("operation directory must be new and empty")
    early_gate_path = (
        args.launch_gate if args.launch_gate.is_absolute() else LAUNCHER_ROOT / args.launch_gate
    )
    early_gate = operator.fast3_launch_gate(_load(early_gate_path))
    if _file_sha256(early_gate_path) != LAUNCH_GATE_FILE_SHA256:
        raise ValueError("Fast3 launch gate bytes changed")
    _source_head(source_root, args.source_head)
    _launcher_head(args.launcher_head, source_head=args.source_head)
    launch_gate_path = _launcher_file(args.launch_gate)
    identity_path = _source_file(source_root, args.identity)
    diagnostic_path = _source_file(source_root, args.failure_diagnostic)
    retirement_path = _source_file(source_root, args.fast2_retirement)
    policy_path = _source_file(source_root, args.generation_retry_policy)
    science_path = _source_file(source_root, args.predecessor_science)
    identity = historical.load_identity(identity_path)
    if identity != operator.FAST3_IDENTITY:
        raise ValueError("reviewed Fast3 identity changed")

    preflight_dir = operator_launch._operation_directory(args.preflight_directory)
    plan = _load(preflight_dir / "PLAN.json")
    request = _load(preflight_dir / "REQUEST.json")
    launch_gate = _launch_gate(launch_gate_path, plan)
    if launch_gate != early_gate:
        raise ValueError("Fast3 append-only launch authorization changed")
    preflight, data_manifest = _preflight(preflight_dir, plan, request, launch_gate)
    expected_plan_sha = _expected_sha256(args.plan_sha256, name="plan")
    expected_request_sha = _expected_sha256(args.request_sha256, name="request")
    expected_data_sha = _expected_sha256(args.data_manifest_sha256, name="data manifest")
    expected_gpu_sha = _expected_sha256(args.gpu_manifest_sha256, name="GPU manifest")
    policy, policy_file_sha, policy_self_sha = _retry_policy(policy_path)
    plan_policy = plan.get("qualification", {}).get("fast3", {}).get("generation_retry_policy")
    data_self_sha = _self_sha256(data_manifest)
    if (
        launch_direct.job_request(plan, identity=identity) != request
        or plan.get("run_name") != identity.run_name
        or plan.get("output_root") != identity.output_root
        or plan.get("native_overrides", {}).get("trainer.eval_before_train") is not True
        or "eval_before_train" in plan.get("arguments", {})
        or plan.get("arguments", {}).get("steps") != 1
        or plan.get("arguments", {}).get("groups") != 1
        or plan.get("arguments", {}).get("samples_per_prompt") != 8
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("failureAlerts") is not False
        or data_manifest != plan.get("data")
        or "sha256:" + digest(plan) != expected_plan_sha
        or "sha256:" + digest(request) != expected_request_sha
        or data_self_sha != expected_data_sha
        or plan_policy != policy
    ):
        raise ValueError("reviewed Fast3 immutable inputs changed")

    diagnostic, diagnostic_file_sha, diagnostic_self_sha = _failure_diagnostic(diagnostic_path)
    retirement, retirement_file_sha, retirement_self_sha = _fast2_retirement(retirement_path)
    science, science_file_sha, science_self_sha = _predecessor_science(
        science_path,
        diagnostic_file_sha256=diagnostic_file_sha,
        diagnostic_self_sha256=diagnostic_self_sha,
        retirement_file_sha256=retirement_file_sha,
        retirement_self_sha256=retirement_self_sha,
        retry_policy_sha256=policy_self_sha,
    )
    evidence = operator.fast3_predecessor_evidence(
        source_head=args.source_head,
        failure_diagnostic_file_sha256=diagnostic_file_sha,
        failure_diagnostic_self_sha256=diagnostic_self_sha,
        fast2_retirement_file_sha256=retirement_file_sha,
        fast2_retirement_self_sha256=retirement_self_sha,
        generation_retry_policy_sha256=policy_self_sha,
        predecessor_science_sha256=science_self_sha,
    )
    token = os.environ["FLEET_API_KEY"]
    source_preview, expected, gpu_previews, provenance = _fresh_gpu_proofs(
        plan=plan, request=request, token=token
    )
    if "sha256:" + digest(expected) != expected_gpu_sha:
        raise ValueError("reviewed Fast3 GPU manifest changed")
    capacity = _capacity(request)
    host_identity = launch_direct.host_identity_proof(
        identity, token=token, launch_gate=launch_gate
    )
    packet = operator_job.launch_packet(
        identity=identity,
        launch_gate=launch_gate,
        plan=plan,
        request=request,
        preflight_launch_result=preflight,
        source_preview=source_preview,
        manifest_sha256=expected_gpu_sha,
        dev_preview=gpu_previews[0],
        dev_preview_provenance=provenance,
        duplicate_proof=host_identity,
        capacity_census=capacity,
        predecessor_evidence=evidence,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    _validate_topology(expected, package)
    operator_previews = operator_launch.server_previews(package)
    operator_duplicate = operator_launch.duplicate_proof(package, previews=operator_previews)
    margin = validate_create_margin(
        gpu_previews=gpu_previews,
        host_identity=host_identity,
        operator_previews=operator_previews,
        operator_duplicate=operator_duplicate,
        capacity=capacity,
    )
    artifacts = {
        "PLAN.json": plan,
        "REQUEST.json": request,
        "SUCCESSOR_MANIFEST.json": data_manifest,
        "GENERATION_RETRY_POLICY.json": policy,
        "FAILURE_DIAGNOSTIC.json": diagnostic,
        "FAST2_RETIREMENT.json": retirement,
        "PREDECESSOR_SCIENCE.json": science,
        "PREDECESSOR_EVIDENCE.json": evidence,
        "LAUNCH_GATE.json": launch_gate,
        "SOURCE_PREVIEW.json": source_preview,
        "GPU_MANIFEST.json": expected,
        "GPU_PREVIEWS.json": gpu_previews,
        "HOST_IDENTITY_PROOF.json": host_identity,
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
        raise ValueError("sealed Fast3 package did not rebuild exactly")
    summary = {
        "schema": "cyber_skyrl_prod11_fast3_jit_launch_summary_v1",
        "status": "sealed_and_validated_no_create" if not args.create else "sealed_for_create",
        "successor_class": "bounded_retry_policy_successor_not_exact_runtime_parity",
        "source_head": args.source_head,
        "launcher_head": args.launcher_head,
        "operation_directory": str(output),
        "identity_sha256": identity.sealed_mapping()["sha256"],
        "run_name": identity.run_name,
        "output_root": identity.output_root,
        "plan_sha256": expected_plan_sha,
        "request_sha256": expected_request_sha,
        "data_manifest_sha256": expected_data_sha,
        "gpu_manifest_sha256": expected_gpu_sha,
        "preflight_launch_sha256": preflight["sha256"],
        "generation_retry_policy_file_sha256": policy_file_sha,
        "generation_retry_policy_sha256": policy_self_sha,
        "failure_diagnostic_file_sha256": diagnostic_file_sha,
        "failure_diagnostic_self_sha256": diagnostic_self_sha,
        "fast2_retirement_file_sha256": retirement_file_sha,
        "fast2_retirement_self_sha256": retirement_self_sha,
        "predecessor_science_file_sha256": science_file_sha,
        "predecessor_science_sha256": science_self_sha,
        "predecessor_evidence_sha256": evidence["sha256"],
        "launch_gate_sha256": launch_gate["sha256"],
        "packet_sha256": packet["sha256"],
        "package": proof,
        "gpu_preview_sha256": [value["sha256"] for value in gpu_previews],
        "host_identity_proof_sha256": host_identity["sha256"],
        "host_sfs_absence_claimed": False,
        "runtime_jit_sfs_absence_required": True,
        "capacity_sha256": "sha256:" + capacity["sha256"],
        "operator_preview_sha256": [value["sha256"] for value in operator_previews],
        "operator_duplicate_sha256": operator_duplicate["sha256"],
        "ttl_margin": margin,
        "outer_gpus": 0,
        "inner_nodes": 1,
        "inner_gpus": 8,
        "create_authorized": args.create,
    }
    prod10_finalizer._write_once(output, "SUMMARY.json", summary)
    if not args.create:
        return summary
    validate_create_margin(
        gpu_previews=gpu_previews,
        host_identity=host_identity,
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
    value.add_argument("--launcher-head", required=True)
    value.add_argument("--launch-gate", type=Path, required=True)
    value.add_argument("--identity", type=Path, required=True)
    value.add_argument("--plan-sha256", required=True)
    value.add_argument("--request-sha256", required=True)
    value.add_argument("--data-manifest-sha256", required=True)
    value.add_argument("--gpu-manifest-sha256", required=True)
    value.add_argument("--preflight-directory", type=Path, required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    value.add_argument("--failure-diagnostic", type=Path, required=True)
    value.add_argument("--fast2-retirement", type=Path, required=True)
    value.add_argument("--generation-retry-policy", type=Path, required=True)
    value.add_argument("--predecessor-science", type=Path, required=True)
    value.add_argument("--create", action="store_true")
    return value


def main() -> None:
    print(json.dumps(finalize(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
