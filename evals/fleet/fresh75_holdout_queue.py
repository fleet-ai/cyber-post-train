"""Validate the inert Fresh75 Fleet heldout queue and run GET-only drift checks.

This module intentionally has no launch, staging, registration, cancellation or
write path.  Its live mode only reads Fleet task/session metadata and inference
serving identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import evaluate as evaluation
from evals.fleet import opencode_self_hosted as harness
from evals.fleet import rollout_worker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = REPO_ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-queue-v1.json"
ACCEPTED_ARMS = {
    "b8-lr1e5-e2",
    "b8-lr3e6-e2",
    "b32-lr1e5-e2",
    "b64-lr1e5-e2",
    "b8-lr1e5-e1",
    "b16-lr1e5-e2",
    "b8-lr1e6-e2",
}
PENDING_ARMS = {"b8-lr1e5-e4"}
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not an exact SHA-256 identity")


def _check_file(reference: dict[str, Any], root: Path, label: str) -> Path:
    path = root / reference["path"]
    expected = reference.get("file_sha256")
    _require_sha256(expected, f"{label}.file_sha256")
    if _file_sha256(path) != expected:
        raise ValueError(f"{label} file digest changed")
    return path


def _binding_digest(binding: tuple[dict, dict, dict]) -> str:
    task, environment, verifier = binding
    return "sha256:" + digest({"task": task, "environment": environment, "verifier": verifier})


def _base_eval_config(queue: dict[str, Any], tasks: dict[str, Any]) -> dict[str, Any]:
    treatment = queue["treatment"]
    base = queue["base_control"]
    harness_fields = {key: treatment[key] for key in evaluation.TREATMENT_FIELDS}
    return {
        "name": base["campaign_id"],
        "task_set": queue["task_set"]["path"],
        "models": {
            "upstream-base": {
                "repository": base["model"]["repository"],
                "revision": base["model"]["revision"],
                "session_model": base["model"]["session_model"],
            }
        },
        "routes": {
            "base": {
                "model": "upstream-base",
                "served_id": base["route"]["served_id"],
                "task_versions": [task["task_version_id"] for task in tasks["tasks"]],
                "endpoint_origin": base["route"]["endpoint_origin"],
                "catalog": base["route"]["catalog"],
                "model_info": base["route"]["model_info"],
                "server_info": base["route"]["server_info"],
            }
        },
        "harness": harness_fields,
        "images": treatment["images"],
        "sampling": treatment["sampling"],
        "pass_k": treatment["pass_k"],
        "concurrency": treatment["concurrency"],
        "training_data_eligible": treatment["training_data_eligible"],
    }


def validate_queue(queue_path: Path = DEFAULT_QUEUE, *, repo_root: Path = REPO_ROOT) -> dict:
    """Validate local immutable inputs without network access or filesystem writes."""
    queue = _read(queue_path)
    if queue.get("schema") != "qwen38_fresh75_fleet_dev17_queue_v1":
        raise ValueError("unexpected Fresh75 queue schema")
    if (
        queue.get("launch_authorized") is not False
        or queue.get("external_mutation_authorized") is not False
    ):
        raise ValueError("the heldout queue must remain mutation-free")
    budget = queue.get("failure_budget") or {}
    if budget.get("limit") != 10 or budget.get("consumed") != 10 or budget.get("remaining") != 0:
        raise ValueError("the exhausted 10/10 failure budget must be explicit")

    training_queue_path = _check_file(
        queue["provenance"]["training_queue"], repo_root, "training queue"
    )
    training_queue = _read(training_queue_path)
    if training_queue.get("queue_sha256") != queue["provenance"]["training_queue"]["queue_sha256"]:
        raise ValueError("training queue object identity changed")

    task_path = _check_file(queue["task_set"], repo_root, "task set")
    tasks = _read(task_path)
    selected = tasks.get("tasks") or []
    versions = [task.get("task_version_id") for task in selected]
    if (
        len(selected) != 17
        or len(set(versions)) != 17
        or tasks.get("selection_role") != "dev"
        or tasks.get("selection_sha256") != queue["task_set"]["selection_sha256"]
        or tasks.get("split_manifest", {}).get("object_sha256")
        != queue["task_set"]["split_manifest_sha256"]
    ):
        raise ValueError("heldout task selection is incomplete or drifted")

    census_path = _check_file(queue["provenance"]["get_only_census"], repo_root, "GET-only census")
    census = _read(census_path)
    if (
        census.get("external_methods_used") != ["GET"]
        or census.get("external_mutation_performed") is not False
    ):
        raise ValueError("census must be read-only")
    binding_receipts = {
        row["task_version_id"]: row["binding_sha256"] for row in census.get("task_bindings", [])
    }
    if set(binding_receipts) != set(versions):
        raise ValueError("GET census does not bind every exact task version")
    for value in binding_receipts.values():
        _require_sha256(value, "task binding")

    treatment = queue.get("treatment") or {}
    if (
        treatment.get("harness") != "opencode"
        or treatment.get("harness_version") != "1.18.27"
        or treatment.get("provider_adapter") != "@ai-sdk/openai-compatible"
        or treatment.get("context_management")
        != "opencode_1.18.27_native_compaction_autocontinue_v2"
        or treatment.get("tools") != ["bash", "submit_report"]
        or treatment.get("pass_k") != 1
        or treatment.get("concurrency") != 1
        or treatment.get("training_data_eligible") is not False
    ):
        raise ValueError("Fleet/OpenCode treatment drifted")
    for key in ("release_asset_sha256", "tool_catalog_sha256"):
        _require_sha256(treatment.get(key), f"treatment.{key}")
    for relative, expected in treatment.get("runtime_files", {}).items():
        _require_sha256(expected, f"runtime file {relative}")
        if _file_sha256(repo_root / relative) != expected:
            raise ValueError(f"runtime file changed: {relative}")

    base = queue.get("base_control") or {}
    if base.get("arm_id") != "upstream-base" or base.get("readiness") != (
        "serving_ready_launch_blocked"
    ):
        raise ValueError("base matched control is not frozen")
    lock_path = _check_file(
        {
            "path": base["model"]["lock_path"],
            "file_sha256": base["model"]["lock_file_sha256"],
        },
        repo_root,
        "base model lock",
    )
    lock = _read(lock_path)
    if any(
        (
            lock.get("repo") != base["model"]["repository"],
            lock.get("revision") != base["model"]["revision"],
            lock.get("weights", {}).get("manifest_sha256")
            != base["model"]["weight_manifest_sha256"],
            lock.get("tokenizer", {}).get("manifest_sha256")
            != base["model"]["tokenizer_manifest_sha256"],
        )
    ):
        raise ValueError("base model lock and control identity differ")
    evaluation.compile_eval(_base_eval_config(queue, tasks), relative_to=repo_root)

    arms = queue.get("accepted_sft_arms") or []
    if {arm.get("arm_id") for arm in arms} != ACCEPTED_ARMS or len(arms) != 7:
        raise ValueError("accepted SFT checkpoint set changed")
    output_roots = [base.get("output_root")]
    campaigns = [base.get("campaign_id")]
    served_ids: list[str] = []
    for arm in arms:
        if (
            arm.get("readiness") != "blocked_checkpoint_payload_export_and_serving"
            or arm.get("model_repository") is not None
            or arm.get("model_revision") is not None
            or arm.get("checkpoint", {}).get("payload_seal_status") != "required_not_yet_accepted"
        ):
            raise ValueError(f"{arm.get('arm_id')} invents an unavailable export identity")
        handoff_path = _check_file(arm["handoff"], repo_root, f"{arm['arm_id']} handoff")
        handoff = _read(handoff_path)
        checkpoint = arm["checkpoint"]
        if any(
            (
                handoff.get("arm_id") != arm["arm_id"],
                handoff.get("status") != "training_accepted_waiting_for_checkpoint_payload_seal",
                handoff.get("handoff_sha256") != arm["handoff"]["handoff_sha256"],
                handoff.get("training_observation_sha256")
                != arm["handoff"]["training_observation_sha256"],
                handoff.get("checkpoint_selection", {}).get("optimizer_step")
                != checkpoint.get("optimizer_step"),
                handoff.get("checkpoint_selection", {}).get("checkpoint_path")
                != checkpoint.get("path"),
                handoff.get("checkpoint_selection", {}).get("checkpoint_saved_receipt_sha256")
                != checkpoint.get("saved_receipt_sha256"),
                handoff.get("next_stage", {}).get("serving_model_id")
                != arm.get("expected_served_id"),
            )
        ):
            raise ValueError(f"{arm['arm_id']} differs from its accepted handoff")
        output_roots.append(arm.get("output_root"))
        campaigns.append(arm.get("campaign_id"))
        served_ids.append(arm.get("expected_served_id"))
    if len(set(output_roots)) != 8 or any(
        not isinstance(root, str)
        or not root.startswith("/mnt/sfs/jobs/q38-f75v2-fleet-dev17-p1-v1/")
        for root in output_roots
    ):
        raise ValueError("evaluation output roots are not unique and frozen")
    if len(set(campaigns)) != 8 or any(
        not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", name) is None
        for name in campaigns
    ):
        raise ValueError("campaign identities are not unique evaluator names")
    if len(set(served_ids)) != 7:
        raise ValueError("candidate serving identities are not unique")

    pending = queue.get("pending_training_arms") or []
    if {arm.get("arm_id") for arm in pending} != PENDING_ARMS or any(
        arm.get("arm_id") in ACCEPTED_ARMS
        or arm.get("status") != "pending_not_terminal_not_accepted_not_in_evaluation_matrix"
        for arm in pending
    ):
        raise ValueError("e4 must remain pending and outside the evaluation matrix")
    if queue.get("retry_policy", {}).get("automatic_retry") is not False:
        raise ValueError("automatic retries are prohibited")
    if queue.get("duplicate_and_absence_policy", {}).get("external_check_methods") != ["GET"]:
        raise ValueError("duplicate checks must remain GET-only")

    return {
        "status": "valid_mutation_free_queue",
        "queue_file_sha256": _file_sha256(queue_path),
        "tasks": 17,
        "base_controls": 1,
        "accepted_sft_arms": 7,
        "pending_training_arms": 1,
        "launch_authorized": False,
        "binding_receipts": binding_receipts,
    }


def live_check(
    queue_path: Path = DEFAULT_QUEUE,
    *,
    repo_root: Path = REPO_ROOT,
    client: httpx.Client | None = None,
) -> dict:
    """Reconcile live state using GET requests only; never create or update state."""
    validated = validate_queue(queue_path, repo_root=repo_root)
    queue = _read(queue_path)
    tasks = _read(repo_root / queue["task_set"]["path"])
    census = _read(repo_root / queue["provenance"]["get_only_census"]["path"])
    expected_bindings = {
        row["task_version_id"]: row["binding_sha256"] for row in census["task_bindings"]
    }
    owns_client = client is None
    if client is None:
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise ValueError("FLEET_API_KEY is required for GET-only live checking")
        client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60)
    try:
        account = harness._request(client, "GET", "/v1/account")  # noqa: SLF001
        if account.get("team_name") != "fleet" or account.get("team_id") != harness.FLEET_TEAM_ID:
            raise RuntimeError("Fleet team identity required")
        bindings: dict[str, str] = {}
        sessions_by_task: dict[str, list[dict]] = {}
        for task in tasks["tasks"]:
            version = task["task_version_id"]
            binding = rollout_worker._task_binding(client, task, task)  # noqa: SLF001
            if binding[0].get("cyber_contract", {}).get("verifier_contract") != "3.0.0":
                raise RuntimeError("task verifier contract changed")
            bindings[version] = _binding_digest(binding)
            if bindings[version] != expected_bindings[version]:
                raise RuntimeError(f"task binding changed: {version}")
            sessions_by_task[version] = harness._task_sessions(  # noqa: SLF001
                client, task["task_key"]
            )

        base = queue["base_control"]
        base_route = {
            "served_id": base["route"]["served_id"],
            "catalog": base["route"]["catalog"],
            "model_info": base["route"]["model_info"],
            "server_info": base["route"]["server_info"],
        }
        base_model = {
            "repository": base["model"]["repository"],
            "revision": base["model"]["revision"],
            "session_model": base["model"]["session_model"],
        }
        base_proof = evaluation.check_route(base_route, base_model, client)
        response = client.get("https://inference.flt.build/fleet/v1/model-catalog")
        response.raise_for_status()
        catalog = response.json().get("data") or []
        candidates = [arm["expected_served_id"] for arm in queue["accepted_sft_arms"]]
        candidate_catalog_matches = {
            served_id: sum(row.get("id") == served_id for row in catalog)
            for served_id in candidates
        }
        if any(candidate_catalog_matches.values()):
            raise RuntimeError(
                "candidate route state advanced; freeze its exact export and profile"
            )

        candidate_session_matches = 0
        base_session_matches = 0
        for sessions in sessions_by_task.values():
            candidate_session_matches += sum(row.get("model") in candidates for row in sessions)
            base_session_matches += sum(
                row.get("model") == base["route"]["served_id"] for row in sessions
            )
        if candidate_session_matches:
            raise RuntimeError("planned candidate identity already has Fleet sessions")
        return {
            "status": "get_only_live_check_passed",
            "external_methods_used": ["GET"],
            "external_mutation_performed": False,
            "queue_file_sha256": validated["queue_file_sha256"],
            "task_bindings": len(bindings),
            "base_route": base_proof,
            "candidate_catalog_matches": candidate_catalog_matches,
            "candidate_session_matches": candidate_session_matches,
            "base_historical_session_matches": base_session_matches,
            "launch_authorized": False,
        }
    finally:
        if owns_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--live-get-only",
        action="store_true",
        help="also reconcile Fleet and serving state using GET requests only",
    )
    args = parser.parse_args()
    result = live_check(args.queue) if args.live_get_only else validate_queue(args.queue)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
