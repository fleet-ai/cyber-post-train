"""Append-only release authority for the non-scoring hosted c2/c4 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_concurrency4_qualification_package_v1 as package

SCHEMA = "fleet-hosted-concurrency4-qualification-release-v1"
PACKAGE_COMMIT = "cbe8b2c7654cbda1f5392c8466b953e335be2641"
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-launch-release-v1.json"
)
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-held-v1.json"
)
MODULE_PATH = Path("evals/fleet/hosted_concurrency4_qualification_release_v1.py")
SUBMIT_PATH = Path("evals/fleet/scripts/submit_hosted_concurrency4_qualification_release_v1.sh")
RELEASE_INTENT = "chris-cyber-hosted-c4-qualification-release-v1"
G5 = {
    "qwen3.8-27b": {
        "job": "chris-q38-ac-r004-a1-g5-v1",
        "spec": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation5-v1.json",
        "release": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-autocontinue-generation5-scoring-release-v1.json"
        ),
    },
    "glm-5.3": {
        "job": "chris-glm53-ac-r013-a1-g5-v1",
        "spec": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation5-v1.json",
        "release": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-glm53-autocontinue-generation5-scoring-release-v1.json"
        ),
    },
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReleaseError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReleaseError("release_target_unsafe")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ReleaseError("receipt_shape_invalid")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError("duplicate_json_key")
        result[key] = value
    return result


def _git_file(root: Path, commit: str, path: Path) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise ReleaseError("commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError("git_file_absent") from exc


def _held_binding(root: Path) -> dict[str, str]:
    raw = _git_file(root, PACKAGE_COMMIT, HELD_PATH)
    held = json.loads(raw, object_pairs_hook=_strict_object)
    if (
        not isinstance(held, dict)
        or held.get("status") != "HELD"
        or held.get("launch_authorized") is not False
        or held.get("objects_created") is not False
        or held.get("receipt_sha256") != digest_without(held, "receipt_sha256")
    ):
        raise ReleaseError("held_authority_invalid")
    return {
        "path": str(HELD_PATH),
        "file_sha256": sha256(raw),
        "receipt_sha256": held["receipt_sha256"],
    }


def build_release(root: Path, implementation_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(implementation_commit) is None:
        raise ReleaseError("implementation_commit_invalid")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                PACKAGE_COMMIT,
                implementation_commit,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError("implementation_not_descended_from_package") from exc
    for path in (MODULE_PATH, SUBMIT_PATH):
        _git_file(root, implementation_commit, path)
    configmap = package.build_configmap(root, PACKAGE_COMMIT)
    package_value = json.loads(configmap["data"]["package.json"])
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "non_scoring": True,
            "chat_completion_requests": 24,
            "task_instance_session_scoring_or_verifier_calls": 0,
            "scored_bulk_launch_authorized": False,
            "maximum_concurrent_requests_per_model": 4,
            "models_run_sequentially": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "package": {
            "commit": PACKAGE_COMMIT,
            "schema_version": package.SCHEMA,
            "sha256": package_value["package_sha256"],
            "held_launch_authorized": False,
        },
        "implementation": {
            "commit": implementation_commit,
            "module": {
                "path": str(MODULE_PATH),
                "sha256": sha256(_git_file(root, implementation_commit, MODULE_PATH)),
            },
            "submitter": {
                "path": str(SUBMIT_PATH),
                "sha256": sha256(_git_file(root, implementation_commit, SUBMIT_PATH)),
            },
        },
        "held_authority": _held_binding(root),
        "runtime_prerequisites": {
            "generation5_qwen_and_glm_exclusively_complete_and_accepted": True,
            "no_active_generation5_or_exact_pass4_bulk_controller": True,
            "exact_secret_uid_required": True,
            "hosted_route_identity_checked_in_job": True,
            "output_root_and_all_object_names_absent": True,
            "separate_non_scoring_lease_required": True,
        },
        "interpretation": {
            "classification": "non_scoring_operational_capacity_gate",
            "passing_only_proposes_hosted_stream_cap_four": True,
            "task_capability_claim": False,
            "hosted_weight_bytes_claim": False,
            "standalone_context_length_claim": False,
        },
        "privacy": {
            "credentials_included": False,
            "request_or_response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def validate_release(receipt: dict[str, Any], root: Path) -> None:
    implementation_commit = receipt.get("implementation", {}).get("commit")
    if not isinstance(implementation_commit, str) or receipt != build_release(
        root, implementation_commit
    ):
        raise ReleaseError("release_not_authoritative")


def released_objects(receipt: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    validate_release(receipt, root)
    bootstrap = package.build_configmap(root, PACKAGE_COMMIT)
    job = yaml.safe_load(_git_file(root, PACKAGE_COMMIT, Path(package.MANIFEST_PATH)))
    job["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/package-commit": PACKAGE_COMMIT,
        "cyber-post-train.fleet.ai/release-receipt-sha256": receipt["receipt_sha256"],
    }
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env.extend(
        [
            {"name": "RELEASE_RECEIPT_SHA256", "value": receipt["receipt_sha256"]},
            {"name": "RELEASE_IMPLEMENTATION_COMMIT", "value": receipt["implementation"]["commit"]},
        ]
    )
    intent = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": RELEASE_INTENT,
            "namespace": package.NAMESPACE,
            "annotations": {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
            },
        },
        "immutable": True,
        "data": {
            "job": package.JOB_NAME,
            "bootstrap_configmap": package.CONFIGMAP_NAME,
            "package_commit": PACKAGE_COMMIT,
            "package_sha256": receipt["package"]["sha256"],
            "release_implementation_commit": receipt["implementation"]["commit"],
            "release_receipt_sha256": receipt["receipt_sha256"],
            "launch_authorized": "true",
            "scored_bulk_launch_authorized": "false",
        },
    }
    _validate_rendered(bootstrap, intent, job)
    return [bootstrap, intent, job]


def _validate_rendered(
    bootstrap: dict[str, Any], intent: dict[str, Any], job: dict[str, Any]
) -> None:
    pod = job["spec"]["template"]["spec"]
    resources = pod["containers"][0]["resources"]
    if (
        bootstrap.get("immutable") is not True
        or json.loads(bootstrap["data"]["package.json"]).get("launch_authorized") is not False
        or intent.get("immutable") is not True
        or intent["data"].get("launch_authorized") != "true"
        or job["metadata"]["name"] != package.JOB_NAME
        or job["metadata"]["namespace"] != package.NAMESPACE
        or job["metadata"]["annotations"].get("cyber-post-train.fleet.ai/launch-authorized")
        != "true"
        or job["spec"].get("backoffLimit") != 0
        or pod.get("priorityClassName") != "fleet-serve-low"
        or pod.get("preemptionPolicy") != "Never"
        or any("gpu" in key.lower() for bucket in resources.values() for key in bucket)
    ):
        raise ReleaseError("released_object_contract_invalid")


def validate_generation5_evidence(
    root: Path,
    model: str,
    terminal_path: Path,
    claim_path: Path,
    *,
    live_job_uid: str,
    live_pod_uid: str,
) -> None:
    """Validate the exact accepted G5 chain without exposing scores or content."""
    from evals.fleet import autocontinue_generation5_authority_v1 as authority
    from evals.fleet import autocontinue_generation5_canary as generation5

    binding = G5.get(model)
    if binding is None:
        raise ReleaseError("generation5_model_invalid")
    spec = authority.load(root / binding["spec"])
    plan = generation5.validate_spec(spec, root)
    release_path = root / binding["release"]
    release = authority.load(release_path)
    package_commit = release.get("package_commit")
    if not isinstance(package_commit, str):
        raise ReleaseError("generation5_package_commit_invalid")
    terminal = load(terminal_path)
    claim = load(claim_path)
    root_authority = authority.load(root / authority.AUTH_PATH)
    authority.validate_terminal(
        terminal,
        spec,
        plan,
        claim,
        release,
        authority.file_sha256(release_path),
        root_authority,
        root,
        package_commit,
    )
    result = terminal.get("result") or {}
    if (
        terminal.get("job_uid") != live_job_uid
        or terminal.get("pod_uid") != live_pod_uid
        or result.get("accepted") is not True
        or result.get("quarantined") is not False
        or result.get("session_ingest_completed") is not True
        or result.get("cleanup_completed") is not True
    ):
        raise ReleaseError("generation5_terminal_not_accepted")


def validate_terminal_job(
    job: dict[str, Any], pods: dict[str, Any], *, job_name: str
) -> tuple[str, str]:
    conditions = (job.get("status") or {}).get("conditions") or []
    items = pods.get("items")
    if (
        (job.get("metadata") or {}).get("name") != job_name
        or not any(
            isinstance(row, dict) and row.get("type") == "Complete" and row.get("status") == "True"
            for row in conditions
        )
        or any(
            isinstance(row, dict) and row.get("type") == "Failed" and row.get("status") == "True"
            for row in conditions
        )
        or (job.get("status") or {}).get("active", 0) not in (0, None)
        or not isinstance(items, list)
        or len(items) != 1
    ):
        raise ReleaseError("generation5_job_not_exclusively_complete")
    pod = items[0]
    statuses = (pod.get("status") or {}).get("containerStatuses")
    if (
        (pod.get("status") or {}).get("phase") != "Succeeded"
        or not isinstance(statuses, list)
        or len(statuses) != 1
        or statuses[0].get("restartCount") != 0
        or ((statuses[0].get("state") or {}).get("terminated") or {}).get("exitCode") != 0
    ):
        raise ReleaseError("generation5_pod_not_cleanly_succeeded")
    try:
        return str(uuid.UUID(job["metadata"]["uid"])), str(uuid.UUID(pod["metadata"]["uid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseError("generation5_uid_invalid") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "render-release",
            "validate-release",
            "render-bundle",
            "validate-terminal-job",
            "validate-generation5-evidence",
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-json", type=Path)
    parser.add_argument("--pods-json", type=Path)
    parser.add_argument("--job-name")
    parser.add_argument("--model")
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--live-job-uid")
    parser.add_argument("--live-pod-uid")
    args = parser.parse_args()
    if args.command == "render-release":
        if not args.implementation_commit or not args.output:
            parser.error("render-release requires implementation commit and output")
        expected = (args.repo / RELEASE_PATH).resolve()
        if args.output.resolve() != expected:
            parser.error("render-release output must be the exact append-only release path")
        write_json_once(args.output, build_release(args.repo, args.implementation_commit))
        return 0
    if args.command == "validate-terminal-job":
        if not args.job_json or not args.pods_json or not args.job_name:
            parser.error("validate-terminal-job requires job, pods, and name")
        validate_terminal_job(load(args.job_json), load(args.pods_json), job_name=args.job_name)
        return 0
    if args.command == "validate-generation5-evidence":
        if not all(
            (
                args.model,
                args.terminal,
                args.claim,
                args.live_job_uid,
                args.live_pod_uid,
            )
        ):
            parser.error("validate-generation5-evidence requires model, evidence, and UIDs")
        validate_generation5_evidence(
            args.repo,
            args.model,
            args.terminal,
            args.claim,
            live_job_uid=args.live_job_uid,
            live_pod_uid=args.live_pod_uid,
        )
        return 0
    if not args.release:
        parser.error(f"{args.command} requires release")
    receipt = load(args.release)
    validate_release(receipt, args.repo)
    if args.command == "render-bundle":
        if not args.output:
            parser.error("render-bundle requires output")
        args.output.write_text(
            yaml.safe_dump(
                {"apiVersion": "v1", "kind": "List", "items": released_objects(receipt, args.repo)},
                sort_keys=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
