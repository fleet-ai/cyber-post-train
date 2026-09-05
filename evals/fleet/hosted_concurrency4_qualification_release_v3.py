"""G7-gated append-only release authority for the non-scoring hosted c2/c4 probe."""

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

from evals.fleet import autocontinue_generation7_authority_v1 as g7_authority
from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import hosted_concurrency4_qualification_package_v1 as package

SCHEMA = "fleet-hosted-concurrency4-qualification-release-v3"
HELD_SCHEMA = "fleet-hosted-concurrency4-qualification-g7-gated-held-v3"
ACCEPTANCE_SCHEMA = "fleet-hosted-concurrency4-qualification-g7-acceptance-v1"
PROBE_PACKAGE_COMMIT = "cbe8b2c7654cbda1f5392c8466b953e335be2641"
G7_PACKAGE_COMMIT = "e549ed9588159af9aaf6186e5b458a9eb070c114"
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-launch-release-v3.json"
)
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-g7-gated-held-v3.json"
)
PACKAGE_HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-held-v1.json"
)
MODULE_PATH = Path("evals/fleet/hosted_concurrency4_qualification_release_v3.py")
SUBMIT_PATH = Path("evals/fleet/scripts/submit_hosted_concurrency4_qualification_release_v3.sh")
RELEASE_INTENT = "chris-cyber-hosted-c4-qualification-release-v3"
PRIOR_RELEASE_INTENTS = (
    "chris-cyber-hosted-c4-qualification-release-v1",
    "chris-cyber-hosted-c4-qualification-release-v2",
)
G7 = {
    "qwen3.8-27b": {
        "job": "chris-q38-ac-r004-a1-g7-v1",
        "spec": ("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation7-v1.json"),
        "release": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-autocontinue-generation7-scoring-release-v1.json"
        ),
        "execution": "77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535",
    },
    "glm-5.3": {
        "job": "chris-glm53-ac-r013-a1-g7-v1",
        "spec": ("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation7-v1.json"),
        "release": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-glm53-autocontinue-generation7-scoring-release-v1.json"
        ),
        "execution": "b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524",
    },
}
G7_IMMUTABLE_PATHS = (
    Path(g7_authority.AUTH_PATH),
    Path(g7_authority.MODULE_PATH),
    Path(g7_authority.PACKAGE_MODULE_PATH),
    Path(g7_authority.MANIFEST_PATH),
    Path(g7_authority.RUN_PATH),
    Path(g7_authority.SUBMIT_PATH),
    Path(generation7.FAILURE_PATH),
    *(Path(row["spec"]) for row in G7.values()),
)
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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError("duplicate_json_key")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ReleaseError("receipt_shape_invalid")
    return value


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


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    if COMMIT_RE.fullmatch(descendant) is None:
        raise ReleaseError("implementation_commit_invalid")
    try:
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError("implementation_not_descended_from_required_package") from exc


def _g7_source_binding(root: Path, implementation_commit: str) -> dict[str, Any]:
    _require_ancestor(root, G7_PACKAGE_COMMIT, implementation_commit)
    entries: list[dict[str, str]] = []
    for path in G7_IMMUTABLE_PATHS:
        frozen = _git_file(root, G7_PACKAGE_COMMIT, path)
        if _git_file(root, implementation_commit, path) != frozen:
            raise ReleaseError("generation7_source_drifted")
        entries.append({"path": str(path), "sha256": sha256(frozen)})
    binding: dict[str, Any] = {
        "commit": G7_PACKAGE_COMMIT,
        "files": entries,
    }
    binding["aggregate_sha256"] = digest_without(binding, "aggregate_sha256")
    return binding


def _package_held_binding(root: Path) -> dict[str, str]:
    raw = _git_file(root, PROBE_PACKAGE_COMMIT, PACKAGE_HELD_PATH)
    held = json.loads(raw, object_pairs_hook=_strict_object)
    if (
        not isinstance(held, dict)
        or held.get("status") != "HELD"
        or held.get("launch_authorized") is not False
        or held.get("objects_created") is not False
        or held.get("receipt_sha256") != digest_without(held, "receipt_sha256")
    ):
        raise ReleaseError("package_held_authority_invalid")
    return {
        "path": str(PACKAGE_HELD_PATH),
        "file_sha256": sha256(raw),
        "receipt_sha256": held["receipt_sha256"],
    }


def expected_held() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "launch_authorized": False,
        "objects_created": False,
        "reason": "generation7_canaries_not_yet_terminally_accepted",
        "successor_of": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-hosted-concurrency4-qualification-g6-gated-held-v2.json"
        ),
        "probe_package_commit": PROBE_PACKAGE_COMMIT,
        "generation7_package_commit": G7_PACKAGE_COMMIT,
        "preserved_non_scoring_contract": {
            "chat_completion_requests": 24,
            "task_instance_session_scoring_or_verifier_calls": 0,
            "maximum_concurrent_requests_per_model": 4,
            "models_run_sequentially": True,
            "request_retries": 0,
        },
        "preserved_acceptance_thresholds": {
            "maximum_request_latency_seconds": 300.0,
            "maximum_wave_elapsed_seconds": 600.0,
            "minimum_concurrency4_over_concurrency2_throughput_ratio": 1.25,
            "maximum_concurrency4_over_concurrency2_p95_stream_latency_ratio": 2.5,
            "zero_errors_required": True,
        },
        "release_conditions": {
            "qwen_and_glm_generation7_exclusively_complete_and_accepted": True,
            "exact_generation7_spec_release_terminal_and_claim_chain_required": True,
            "exact_job_pod_uid_and_owner_binding_required": True,
            "exact_session_ingest_cleanup_and_verifier_evidence_required": True,
            "all_create_once_names_and_output_root_absent": True,
            "exact_pass4_bulk_jobs_configmaps_pods_and_output_roots_absent": True,
            "exact_secret_uid_and_hosted_routes_revalidated": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def validate_held(receipt: dict[str, Any]) -> None:
    if receipt != expected_held():
        raise ReleaseError("g7_gated_held_authority_invalid")


def build_release(
    root: Path, implementation_commit: str, generation7_acceptance: dict[str, Any]
) -> dict[str, Any]:
    _require_ancestor(root, PROBE_PACKAGE_COMMIT, implementation_commit)
    g7_source = _g7_source_binding(root, implementation_commit)
    validate_acceptance_binding(generation7_acceptance, root)
    for path in (MODULE_PATH, SUBMIT_PATH, HELD_PATH):
        _git_file(root, implementation_commit, path)
    validate_held(json.loads(_git_file(root, implementation_commit, HELD_PATH)))
    configmap = package.build_configmap(root, PROBE_PACKAGE_COMMIT)
    package_value = json.loads(configmap["data"]["package.json"])
    root_authority_raw = _git_file(root, G7_PACKAGE_COMMIT, Path(g7_authority.AUTH_PATH))
    root_authority = json.loads(root_authority_raw, object_pairs_hook=_strict_object)
    g7_authority.validate_root_authorization(root_authority, root)
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
            "request_retries": 0,
            "scored_bulk_launch_authorized": False,
            "maximum_concurrent_requests_per_model": 4,
            "models_run_sequentially": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "preserved_acceptance_thresholds": {
            "maximum_request_latency_seconds": 300.0,
            "maximum_wave_elapsed_seconds": 600.0,
            "minimum_concurrency4_over_concurrency2_throughput_ratio": 1.25,
            "maximum_concurrency4_over_concurrency2_p95_stream_latency_ratio": 2.5,
            "zero_errors_required": True,
        },
        "probe_package": {
            "commit": PROBE_PACKAGE_COMMIT,
            "schema_version": package.SCHEMA,
            "sha256": package_value["package_sha256"],
            "held_launch_authorized": False,
        },
        "generation7_package": g7_source,
        "generation7_root_authorization": {
            "path": g7_authority.AUTH_PATH,
            "file_sha256": sha256(root_authority_raw),
            "receipt_sha256": root_authority["receipt_sha256"],
        },
        "generation7_acceptance": generation7_acceptance,
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
        "package_held_authority": _package_held_binding(root),
        "g7_gated_held_authority": {
            "path": str(HELD_PATH),
            "file_sha256": sha256(_git_file(root, implementation_commit, HELD_PATH)),
            "receipt_sha256": expected_held()["receipt_sha256"],
        },
        "runtime_prerequisites": {
            "generation7_qwen_and_glm_exclusively_complete_and_accepted": True,
            "exact_generation7_job_pod_uid_and_owner_binding": True,
            "exact_generation7_session_ingest_cleanup_and_verifier_evidence": True,
            "no_active_generation7_controller": True,
            "exact_pass4_bulk_jobs_configmaps_pods_and_output_roots_absent": True,
            "exact_secret_uid_required": True,
            "hosted_route_identity_checked_in_job": True,
            "output_root_and_all_object_names_absent": True,
            "prior_release_intents_must_be_absent": True,
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
    acceptance = receipt.get("generation7_acceptance")
    if (
        not isinstance(implementation_commit, str)
        or not isinstance(acceptance, dict)
        or receipt != build_release(root, implementation_commit, acceptance)
    ):
        raise ReleaseError("release_not_authoritative")


def released_objects(receipt: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    validate_release(receipt, root)
    bootstrap = package.build_configmap(root, PROBE_PACKAGE_COMMIT)
    job = yaml.safe_load(_git_file(root, PROBE_PACKAGE_COMMIT, Path(package.MANIFEST_PATH)))
    job["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/preview-only": "false",
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/package-commit": PROBE_PACKAGE_COMMIT,
        "cyber-post-train.fleet.ai/generation7-package-commit": G7_PACKAGE_COMMIT,
        "cyber-post-train.fleet.ai/generation7-acceptance-sha256": receipt[
            "generation7_acceptance"
        ]["binding_sha256"],
        "cyber-post-train.fleet.ai/release-receipt-sha256": receipt["receipt_sha256"],
        "cyber-post-train.fleet.ai/release-generation": "g7-gated-v3",
    }
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env.extend(
        [
            {"name": "RELEASE_RECEIPT_SHA256", "value": receipt["receipt_sha256"]},
            {
                "name": "RELEASE_IMPLEMENTATION_COMMIT",
                "value": receipt["implementation"]["commit"],
            },
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
            "probe_package_commit": PROBE_PACKAGE_COMMIT,
            "generation7_package_commit": G7_PACKAGE_COMMIT,
            "generation7_acceptance_binding_sha256": receipt["generation7_acceptance"][
                "binding_sha256"
            ],
            "package_sha256": receipt["probe_package"]["sha256"],
            "release_implementation_commit": receipt["implementation"]["commit"],
            "release_receipt_sha256": receipt["receipt_sha256"],
            "launch_authorized": "true",
            "scored_bulk_launch_authorized": "false",
            "generation7_evidence_required": "true",
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
        or intent["data"].get("generation7_evidence_required") != "true"
        or intent["data"].get("generation7_acceptance_binding_sha256")
        != job["metadata"]["annotations"].get(
            "cyber-post-train.fleet.ai/generation7-acceptance-sha256"
        )
        or job["metadata"]["name"] != package.JOB_NAME
        or job["metadata"]["namespace"] != package.NAMESPACE
        or job["metadata"]["annotations"].get("cyber-post-train.fleet.ai/launch-authorized")
        != "true"
        or job["spec"].get("backoffLimit") != 0
        or pod.get("restartPolicy") != "Never"
        or pod.get("priorityClassName") != "fleet-serve-low"
        or pod.get("preemptionPolicy") != "Never"
        or any("gpu" in key.lower() for bucket in resources.values() for key in bucket)
    ):
        raise ReleaseError("released_object_contract_invalid")


def validate_generation7_evidence(
    root: Path,
    model: str,
    terminal_path: Path,
    claim_path: Path,
    *,
    live_job_uid: str,
    live_pod_uid: str,
) -> None:
    """Validate one accepted G7 chain without exposing scores or task content."""
    binding = G7.get(model)
    if binding is None:
        raise ReleaseError("generation7_model_invalid")
    spec = g7_authority.load(root / binding["spec"])
    plan = generation7.validate_spec(spec, root)
    release_path = root / binding["release"]
    scoring_release = g7_authority.load(release_path)
    package_commit = scoring_release.get("package_commit")
    if package_commit != G7_PACKAGE_COMMIT:
        raise ReleaseError("generation7_package_commit_invalid")
    root_authority = g7_authority.load(root / g7_authority.AUTH_PATH)
    g7_authority.validate_release(scoring_release, spec, root_authority, root, package_commit)
    terminal = load(terminal_path)
    claim = load(claim_path)
    g7_authority.validate_claim(
        claim,
        spec,
        plan,
        scoring_release,
        g7_authority.file_sha256(release_path),
        root_authority,
        root,
        package_commit,
    )
    result = terminal.get("result")
    if not isinstance(result, dict):
        raise ReleaseError("generation7_result_invalid")
    expected = {
        "schema_version": g7_authority.TERMINAL_SCHEMA,
        "generation7_spec_sha256": spec["generation7_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 7,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": terminal.get("terminal_at_utc"),
        "scoring_release": g7_authority.release_binding(
            scoring_release, g7_authority.file_sha256(release_path), package_commit
        ),
        "root_authorization": g7_authority.authority_binding(root_authority, root),
        "result": g7_authority.generation2._validated_result(result, plan),
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not generation7.generation6.generation5.generation3._canonical_utc(
            terminal.get("terminal_at_utc")
        )
        or terminal.get("receipt_sha256") != digest_without(terminal, "receipt_sha256")
        or {key: value for key, value in terminal.items() if key != "receipt_sha256"} != expected
        or terminal.get("job_uid") != live_job_uid
        or terminal.get("pod_uid") != live_pod_uid
        or claim.get("job_uid") != live_job_uid
        or claim.get("pod_uid") != live_pod_uid
        or result.get("accepted") is not True
        or result.get("quarantined") is not False
        or result.get("session_ingest_completed") is not True
        or result.get("cleanup_completed") is not True
        or not isinstance(result.get("session_id"), str)
        or not isinstance(result.get("verifier_execution_id"), str)
    ):
        raise ReleaseError("generation7_terminal_not_accepted")


def generation7_acceptance_binding(root: Path, evidence_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model, source in G7.items():
        job_path = evidence_dir / f"{model}-job.json"
        pods_path = evidence_dir / f"{model}-pods.json"
        terminal_path = evidence_dir / f"{model}-terminal.json"
        claim_path = evidence_dir / f"{model}-claim.json"
        job_uid, pod_uid = validate_terminal_job(
            load(job_path), load(pods_path), job_name=source["job"]
        )
        validate_generation7_evidence(
            root,
            model,
            terminal_path,
            claim_path,
            live_job_uid=job_uid,
            live_pod_uid=pod_uid,
        )
        spec = g7_authority.load(root / source["spec"])
        release_path = root / source["release"]
        scoring_release = g7_authority.load(release_path)
        terminal = load(terminal_path)
        claim = load(claim_path)
        result = terminal["result"]
        rows.append(
            {
                "model": model,
                "job_name": source["job"],
                "job_uid": job_uid,
                "pod_uid": pod_uid,
                "pod_owner_binding": {
                    "api_version": "batch/v1",
                    "kind": "Job",
                    "name": source["job"],
                    "uid": job_uid,
                    "controller": True,
                },
                "generation7_spec_path": source["spec"],
                "generation7_spec_sha256": spec["generation7_spec_sha256"],
                "execution_id": "sha256:" + source["execution"],
                "scoring_release": {
                    "path": source["release"],
                    "file_sha256": g7_authority.file_sha256(release_path),
                    "receipt_sha256": scoring_release["receipt_sha256"],
                    "package_commit": scoring_release["package_commit"],
                },
                "terminal": {
                    "file_sha256": sha256(terminal_path.read_bytes()),
                    "receipt_sha256": terminal["receipt_sha256"],
                },
                "claim": {
                    "file_sha256": sha256(claim_path.read_bytes()),
                    "receipt_sha256": claim["receipt_sha256"],
                },
                "accepted_result": {
                    "accepted": True,
                    "quarantined": False,
                    "acceptance_receipt_sha256": result["acceptance_receipt_sha256"],
                    "session_id": result["session_id"],
                    "verifier_execution_id": result["verifier_execution_id"],
                    "session_ingest_completed": True,
                    "cleanup_completed": True,
                },
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "generation7_package_commit": G7_PACKAGE_COMMIT,
        "models": rows,
        "all_exclusively_complete_and_accepted": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["binding_sha256"] = digest_without(receipt, "binding_sha256")
    validate_acceptance_binding(receipt, root)
    return receipt


def validate_acceptance_binding(receipt: dict[str, Any], root: Path) -> None:
    rows = receipt.get("models")
    if (
        receipt.get("schema_version") != ACCEPTANCE_SCHEMA
        or receipt.get("generation7_package_commit") != G7_PACKAGE_COMMIT
        or receipt.get("all_exclusively_complete_and_accepted") is not True
        or receipt.get("scores_included") is not False
        or receipt.get("prompts_or_traces_included") is not False
        or receipt.get("binding_sha256") != digest_without(receipt, "binding_sha256")
        or not isinstance(rows, list)
        or [row.get("model") for row in rows if isinstance(row, dict)] != list(G7)
    ):
        raise ReleaseError("generation7_acceptance_binding_invalid")
    for row, (model, source) in zip(rows, G7.items(), strict=True):
        spec = g7_authority.load(root / source["spec"])
        generation7.validate_spec(spec, root)
        release_path = root / source["release"]
        scoring_release = g7_authority.load(release_path)
        root_authority = g7_authority.load(root / g7_authority.AUTH_PATH)
        g7_authority.validate_release(
            scoring_release, spec, root_authority, root, G7_PACKAGE_COMMIT
        )
        accepted = row.get("accepted_result")
        terminal = row.get("terminal")
        claim = row.get("claim")
        if (
            not isinstance(accepted, dict)
            or not isinstance(terminal, dict)
            or not isinstance(claim, dict)
        ):
            raise ReleaseError("generation7_acceptance_binding_invalid")
        for field in ("job_uid", "pod_uid"):
            try:
                uuid.UUID(str(row.get(field)))
            except (ValueError, AttributeError) as exc:
                raise ReleaseError("generation7_acceptance_binding_invalid") from exc
        for field in ("session_id", "verifier_execution_id"):
            try:
                uuid.UUID(str(accepted.get(field)))
            except (ValueError, AttributeError) as exc:
                raise ReleaseError("generation7_acceptance_binding_invalid") from exc
        dynamic_hashes = (
            terminal.get("file_sha256"),
            terminal.get("receipt_sha256"),
            claim.get("file_sha256"),
            claim.get("receipt_sha256"),
            accepted.get("acceptance_receipt_sha256"),
        )
        if any(SHA_RE.fullmatch(str(value)) is None for value in dynamic_hashes):
            raise ReleaseError("generation7_acceptance_binding_invalid")
        expected = {
            "model": model,
            "job_name": source["job"],
            "job_uid": row["job_uid"],
            "pod_uid": row["pod_uid"],
            "pod_owner_binding": {
                "api_version": "batch/v1",
                "kind": "Job",
                "name": source["job"],
                "uid": row["job_uid"],
                "controller": True,
            },
            "generation7_spec_path": source["spec"],
            "generation7_spec_sha256": spec["generation7_spec_sha256"],
            "execution_id": "sha256:" + source["execution"],
            "scoring_release": {
                "path": source["release"],
                "file_sha256": g7_authority.file_sha256(release_path),
                "receipt_sha256": scoring_release["receipt_sha256"],
                "package_commit": G7_PACKAGE_COMMIT,
            },
            "terminal": row["terminal"],
            "claim": row["claim"],
            "accepted_result": {
                "accepted": True,
                "quarantined": False,
                "acceptance_receipt_sha256": accepted["acceptance_receipt_sha256"],
                "session_id": accepted["session_id"],
                "verifier_execution_id": accepted["verifier_execution_id"],
                "session_ingest_completed": True,
                "cleanup_completed": True,
            },
        }
        if row != expected:
            raise ReleaseError("generation7_acceptance_binding_invalid")


def validate_live_acceptance_binding(
    release: dict[str, Any], root: Path, evidence_dir: Path
) -> None:
    validate_release(release, root)
    if release["generation7_acceptance"] != generation7_acceptance_binding(root, evidence_dir):
        raise ReleaseError("generation7_live_acceptance_binding_mismatch")


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
        or (job.get("status") or {}).get("succeeded") != 1
        or (job.get("status") or {}).get("failed", 0) not in (0, None)
        or not isinstance(items, list)
        or len(items) != 1
    ):
        raise ReleaseError("generation7_job_not_exclusively_complete")
    pod = items[0]
    statuses = (pod.get("status") or {}).get("containerStatuses")
    init_statuses = (pod.get("status") or {}).get("initContainerStatuses")
    try:
        job_uid = str(uuid.UUID(job["metadata"]["uid"]))
        pod_uid = str(uuid.UUID(pod["metadata"]["uid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseError("generation7_uid_invalid") from exc
    owners = (pod.get("metadata") or {}).get("ownerReferences")
    matching_owners = [
        owner
        for owner in owners or []
        if isinstance(owner, dict)
        and owner.get("apiVersion") == "batch/v1"
        and owner.get("kind") == "Job"
        and owner.get("name") == job_name
        and owner.get("uid") == job_uid
        and owner.get("controller") is True
    ]
    controller_owners = [
        owner
        for owner in owners or []
        if isinstance(owner, dict) and owner.get("controller") is True
    ]
    if (
        not isinstance(owners, list)
        or len(matching_owners) != 1
        or len(controller_owners) != 1
        or (pod.get("status") or {}).get("phase") != "Succeeded"
        or not isinstance(statuses, list)
        or len(statuses) != 1
        or statuses[0].get("name") != "evaluator"
        or any(status.get("restartCount") != 0 for status in statuses)
        or any(
            ((status.get("state") or {}).get("terminated") or {}).get("exitCode") != 0
            for status in statuses
        )
        or not isinstance(init_statuses, list)
        or len(init_statuses) != 2
        or {status.get("name") for status in init_statuses} != {"docker-cli", "dind"}
        or any(status.get("restartCount") != 0 for status in init_statuses)
        or any(
            ((status.get("state") or {}).get("terminated") or {}).get("exitCode") != 0
            for status in init_statuses
        )
    ):
        raise ReleaseError("generation7_pod_not_cleanly_succeeded")
    return job_uid, pod_uid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-held",
            "render-release",
            "validate-release",
            "render-bundle",
            "validate-terminal-job",
            "validate-generation7-evidence",
            "render-acceptance",
            "validate-live-binding",
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--implementation-commit")
    parser.add_argument("--held", type=Path)
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
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.command == "validate-held":
        if not args.held:
            parser.error("validate-held requires held receipt")
        validate_held(load(args.held))
        return 0
    if args.command == "render-release":
        if not args.implementation_commit or not args.output or not args.evidence_dir:
            parser.error("render-release requires implementation commit, evidence, and output")
        expected = (args.repo / RELEASE_PATH).resolve()
        if args.output.resolve() != expected:
            parser.error("render-release output must be the exact append-only release path")
        acceptance = generation7_acceptance_binding(args.repo, args.evidence_dir)
        write_json_once(
            args.output, build_release(args.repo, args.implementation_commit, acceptance)
        )
        return 0
    if args.command == "validate-terminal-job":
        if not args.job_json or not args.pods_json or not args.job_name:
            parser.error("validate-terminal-job requires job, pods, and name")
        validate_terminal_job(load(args.job_json), load(args.pods_json), job_name=args.job_name)
        return 0
    if args.command == "validate-generation7-evidence":
        if not all((args.model, args.terminal, args.claim, args.live_job_uid, args.live_pod_uid)):
            parser.error("validate-generation7-evidence requires model, evidence, and UIDs")
        validate_generation7_evidence(
            args.repo,
            args.model,
            args.terminal,
            args.claim,
            live_job_uid=args.live_job_uid,
            live_pod_uid=args.live_pod_uid,
        )
        return 0
    if args.command == "render-acceptance":
        if not args.evidence_dir or not args.output:
            parser.error("render-acceptance requires evidence and output")
        write_json_once(args.output, generation7_acceptance_binding(args.repo, args.evidence_dir))
        return 0
    if args.command == "validate-live-binding":
        if not args.release or not args.evidence_dir:
            parser.error("validate-live-binding requires release and evidence")
        validate_live_acceptance_binding(load(args.release), args.repo, args.evidence_dir)
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
