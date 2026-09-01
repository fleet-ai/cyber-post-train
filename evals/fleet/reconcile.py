"""Fail-closed reconciliation for interrupted self-hosted Fleet attempts.

This module never runs a model.  Its default CLI is read-only and emits a
content-free recovery plan from local receipts plus an independently collected
authoritative observation.  Mutating operators must consume that plan in a
separate, reviewed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

OBSERVATION_SCHEMA = "fleet-selfhosted-authority-observation-v1"
PLAN_SCHEMA = "fleet-selfhosted-reconcile-plan-v1"
RESOURCE_SNAPSHOT_SCHEMA = "fleet-selfhosted-resource-snapshot-v1"
RESOURCE_PLAN_SCHEMA = "fleet-selfhosted-resource-plan-v1"
SCORING_INTENT_SCHEMA = "fleet-selfhosted-scoring-intent-v1"


class ReconcileError(RuntimeError):
    """Local evidence is incomplete, contradictory, or unsafe to resume."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconcileError(f"cannot read valid JSON receipt: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReconcileError(f"receipt is not a JSON object: {path.name}")
    return value


def _digest(value: dict[str, Any]) -> str:
    return self_hosted.sha256(self_hosted.canonical_json(value))


def _terminal_stream_receipt(attempt_dir: Path) -> dict[str, Any]:
    path = attempt_dir / "agent-output" / "qwen-stream.jsonl"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        raise ReconcileError("Qwen stream is missing") from exc
    if not lines:
        raise ReconcileError("Qwen stream is empty")
    try:
        event = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ReconcileError("Qwen stream has no parseable terminal event") from exc
    if (
        not isinstance(event, dict)
        or event.get("type") != "result"
        or event.get("subtype") != "success"
        or event.get("is_error") is not False
    ):
        raise ReconcileError("Qwen stream is not a successful terminal model outcome")
    return {
        "terminal": True,
        "stream_sha256": self_hosted.sha256(path.read_bytes()),
        "num_turns": event.get("num_turns"),
    }


def _conversation_evidence(
    attempt_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    events, canonical_trace, malformed = self_hosted.load_qwen_chat_trace(attempt_dir / "qwen-home")
    if malformed:
        raise ReconcileError("Qwen chat contains malformed records")
    messages = self_hosted.normalize_qwen_conversation(events)
    calls: set[str] = set()
    results: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if str(function.get("name") or "").endswith("submit_report"):
                call_id = str(call.get("id") or "")
                if call_id:
                    calls.add(call_id)
        if message.get("role") == "tool" and str(
            (message.get("metadata") or {}).get("qwen_tool_name") or ""
        ).endswith("submit_report"):
            call_id = str(message.get("tool_call_id") or "")
            if call_id:
                results.add(call_id)
    final_answer = self_hosted.final_answer_from_conversation(messages)
    stream = attempt_dir / "agent-output" / "qwen-stream.jsonl"
    if not final_answer and stream.exists():
        final_answer = self_hosted.extract_final_answer(stream)
    receipt = {
        "canonical_trace_sha256": self_hosted.sha256(canonical_trace.read_bytes()),
        "final_answer_sha256": self_hosted.sha256(final_answer.encode()),
        "normalized_message_count": len(messages),
        "submit_report_call_count": len(calls),
        "submit_report_completed_count": len(calls & results),
        "submit_report_completed": bool(calls & results),
    }
    return receipt, messages, final_answer


def _validate_identity(
    record: dict[str, Any], binding: dict[str, Any], runtime: dict[str, Any], *, label: str
) -> None:
    expected = {
        "run_id": binding["run_id"],
        "instance_id": runtime["instance_id"],
        "evidence_run_id": runtime["evidence_run_id"],
    }
    for field, wanted in expected.items():
        if str(record.get(field) or "") != str(wanted):
            raise ReconcileError(f"{label} {field} does not match the interrupted attempt")


def _local_reward(
    attempt_dir: Path, binding: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any] | None:
    path = attempt_dir / "reward-result.json"
    if not path.exists():
        return None
    result = _load_object(path)
    expected = {
        "task_key": binding["task"]["key"],
        "task_version_id": binding["task"]["version_id"],
        "instance_id": runtime["instance_id"],
    }
    for field, wanted in expected.items():
        if str(result.get(field) or "") != str(wanted):
            raise ReconcileError(f"local reward {field} drifted")
    reward = result.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise ReconcileError("local reward is not finite numeric evidence")
    execution_id = result.get("verifier_execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise ReconcileError("local reward lacks a verifier execution identity")
    return {
        "reward": float(reward),
        "verifier_execution_id": execution_id,
        "receipt_sha256": self_hosted.sha256(path.read_bytes()),
    }


def _authority_results(
    observation: dict[str, Any] | None,
    binding: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[bool, str | None, list[dict[str, Any]]]:
    if observation is None:
        return False, None, []
    if observation.get("schema_version") != OBSERVATION_SCHEMA:
        raise ReconcileError("unsupported authority observation schema")
    expected_fields = {
        "schema_version",
        "source",
        "lookup_complete",
        "observed_at",
        "run_id",
        "instance_id",
        "evidence_run_id",
        "task_key",
        "task_version_id",
        "instance_status",
        "verifier_results",
        "observation_sha256",
    }
    if set(observation) != expected_fields:
        raise ReconcileError("authority observation fields are incomplete or unknown")
    unsigned = {key: value for key, value in observation.items() if key != "observation_sha256"}
    if observation["observation_sha256"] != _digest(unsigned):
        raise ReconcileError("authority observation self-digest mismatch")
    _validate_identity(observation, binding, runtime, label="authority observation")
    expected_task = {
        "task_key": binding["task"]["key"],
        "task_version_id": binding["task"]["version_id"],
    }
    for field, wanted in expected_task.items():
        if observation.get(field) != wanted:
            raise ReconcileError(f"authority observation {field} drifted")
    if observation.get("source") != "authoritative_verifier_store":
        raise ReconcileError("authority observation is not from the verifier store")
    if not isinstance(observation.get("observed_at"), str) or not observation["observed_at"]:
        raise ReconcileError("authority observation lacks an observation time")
    if observation.get("instance_status") not in {"running", "terminated"}:
        raise ReconcileError("authority observation has an unknown instance status")
    complete = observation.get("lookup_complete") is True
    results = observation.get("verifier_results")
    if not complete or not isinstance(results, list):
        raise ReconcileError("authority observation is not a complete lookup")
    validated: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            raise ReconcileError("authority verifier result is malformed")
        execution_id = result.get("verifier_execution_id")
        reward = result.get("reward")
        if not isinstance(execution_id, str) or not execution_id:
            raise ReconcileError("authority result lacks a verifier execution identity")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ReconcileError("authority result has a non-finite reward")
        validated.append({"verifier_execution_id": execution_id, "reward": float(reward)})
    return True, str(observation.get("instance_status") or ""), validated


def _resource_plan(
    attempt_dir: Path, binding: dict[str, Any], runtime: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    plan = _load_object(attempt_dir / "resource-plan.json")
    expected_fields = {
        "schema_version",
        "run_id",
        "instance_id",
        "evidence_run_id",
        "containers",
        "network",
        "resource_plan_sha256",
    }
    if set(plan) != expected_fields or plan.get("schema_version") != RESOURCE_PLAN_SCHEMA:
        raise ReconcileError("resource plan schema or fields are invalid")
    supplied_digest = plan["resource_plan_sha256"]
    unsigned = {key: value for key, value in plan.items() if key != "resource_plan_sha256"}
    if supplied_digest != _digest(unsigned):
        raise ReconcileError("resource plan self-digest mismatch")
    _validate_identity(plan, binding, runtime, label="resource plan")
    suffix = hashlib.sha256(binding["run_id"].encode()).hexdigest()[:8]
    expected_containers = {
        f"qwen-agent-{suffix}",
        f"qwen-model-proxy-{suffix}",
        f"qwen-mcp-proxy-{suffix}",
    }
    containers = plan.get("containers")
    if not isinstance(containers, list) or set(containers) != expected_containers:
        raise ReconcileError("resource plan container identities drifted")
    if len(containers) != len(expected_containers):
        raise ReconcileError("resource plan contains duplicate containers")
    network = plan.get("network")
    if not isinstance(network, str) or not network or network == "bridge":
        raise ReconcileError("resource plan network identity is invalid")
    return plan, supplied_digest


def _scoring_payload(
    binding: dict[str, Any], runtime: dict[str, Any], messages: list[dict[str, Any]], final: str
) -> dict[str, Any]:
    authority = binding.get("authority")
    if not isinstance(authority, dict):
        raise ReconcileError("binding lacks authoritative scoring controls")
    return {
        "instance_id": runtime["instance_id"],
        "final_answer": final,
        "conversation": messages,
        "scoring_mode": authority.get("scoring_mode"),
        "multi_app_aggregation_mode": authority.get("multi_app_aggregation_mode"),
    }


def _scoring_intent(
    attempt_dir: Path,
    binding: dict[str, Any],
    runtime: dict[str, Any],
    scoring_payload: dict[str, Any],
) -> bool:
    path = attempt_dir / "scoring-intent.json"
    if not path.exists():
        return False
    intent = _load_object(path)
    expected_fields = {
        "schema_version",
        "run_id",
        "instance_id",
        "evidence_run_id",
        "request_sha256",
        "scoring_intent_sha256",
    }
    if set(intent) != expected_fields or intent.get("schema_version") != SCORING_INTENT_SCHEMA:
        raise ReconcileError("scoring intent schema or fields are invalid")
    supplied_digest = intent["scoring_intent_sha256"]
    unsigned = {key: value for key, value in intent.items() if key != "scoring_intent_sha256"}
    if supplied_digest != _digest(unsigned):
        raise ReconcileError("scoring intent self-digest mismatch")
    _validate_identity(intent, binding, runtime, label="scoring intent")
    if intent.get("request_sha256") != self_hosted.sha256(
        self_hosted.canonical_json(scoring_payload)
    ):
        raise ReconcileError("scoring intent request digest does not match the preserved trace")
    return True


def _cleanup_plan(
    binding: dict[str, Any],
    resource_plan: dict[str, Any],
    resource_plan_sha256: str,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = set(resource_plan["containers"])
    planned_network = resource_plan["network"]
    if snapshot is None:
        return {"action": "needs_resource_snapshot", "targets": []}
    if snapshot.get("schema_version") != RESOURCE_SNAPSHOT_SCHEMA:
        raise ReconcileError("unsupported resource snapshot schema")
    expected_fields = {
        "schema_version",
        "lookup_complete",
        "run_id",
        "resource_plan_sha256",
        "containers",
        "networks",
    }
    if set(snapshot) != expected_fields or snapshot.get("lookup_complete") is not True:
        raise ReconcileError("resource snapshot fields are incomplete or unknown")
    if snapshot.get("run_id") != binding["run_id"]:
        raise ReconcileError("resource snapshot run identity drifted")
    if snapshot.get("resource_plan_sha256") != resource_plan_sha256:
        raise ReconcileError("resource snapshot is not bound to the durable resource plan")
    rows = snapshot.get("containers")
    if not isinstance(rows, list):
        raise ReconcileError("resource snapshot containers are malformed")
    targets: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("name") not in expected:
            raise ReconcileError("resource snapshot contains an unrelated container")
        names = row.get("networks")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ReconcileError("resource snapshot network bindings are malformed")
        targets.append(row["name"])
        if planned_network not in names or any(
            name not in {"bridge", planned_network} for name in names
        ):
            raise ReconcileError("resource snapshot proposes an unplanned network")
    if len(targets) != len(set(targets)):
        raise ReconcileError("resource snapshot contains duplicate containers")
    networks = snapshot.get("networks")
    if not isinstance(networks, list) or len(networks) != len(set(networks)):
        raise ReconcileError("resource snapshot networks are malformed")
    if any(name != planned_network for name in networks):
        raise ReconcileError("resource snapshot proposes a peer network")
    return {
        "action": "cleanup_once" if targets or networks else "none",
        "targets": sorted(targets),
        "networks": sorted(networks),
    }


def build_plan(
    attempt_dir: Path,
    *,
    authority_observation: dict[str, Any] | None = None,
    resource_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-free plan; never contacts Fleet, Docker, or the model."""
    binding = _load_object(attempt_dir / "binding.json")
    runtime = _load_object(attempt_dir / "runtime-binding.json")
    resources, resource_plan_sha256 = _resource_plan(attempt_dir, binding, runtime)
    stream = _terminal_stream_receipt(attempt_dir)
    conversation, messages, final_answer = _conversation_evidence(attempt_dir)
    scoring_payload = _scoring_payload(binding, runtime, messages, final_answer)
    scoring_intent_exists = _scoring_intent(attempt_dir, binding, runtime, scoring_payload)
    local = _local_reward(attempt_dir, binding, runtime)
    lookup_complete, instance_status, authoritative = _authority_results(
        authority_observation, binding, runtime
    )

    reasons: list[str] = []
    score_action = "refuse"
    recovered_reward: dict[str, Any] | None = None
    if len(authoritative) > 1:
        reasons.append("multiple_authoritative_verifier_results")
    elif (
        local
        and authoritative
        and (
            local["verifier_execution_id"] != authoritative[0]["verifier_execution_id"]
            or local["reward"] != authoritative[0]["reward"]
        )
    ):
        reasons.append("local_and_authoritative_reward_conflict")
    elif local:
        score_action = "none_already_scored"
        recovered_reward = local
    elif len(authoritative) == 1:
        score_action = "recover_authoritative_result"
        recovered_reward = authoritative[0]
    elif not lookup_complete:
        reasons.append("authoritative_lookup_required_before_scoring")
    elif instance_status.lower() != "running":
        reasons.append("instance_not_running_and_no_authoritative_result")
    else:
        # A pre-existing intent proves a score POST may have started.  A complete
        # empty authority lookup is the only evidence that makes another POST safe.
        score_action = "score_once_from_existing_trace"

    cleanup = _cleanup_plan(binding, resources, resource_plan_sha256, resource_snapshot)
    if score_action in {"refuse", "score_once_from_existing_trace"}:
        cleanup["eligible_action_after_score"] = cleanup["action"]
        cleanup["action"] = "blocked_until_score_reconciled"

    unsigned: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "run_id": binding["run_id"],
        "instance_id": runtime["instance_id"],
        "evidence_run_id": runtime["evidence_run_id"],
        "model_outcome": stream,
        "conversation": conversation,
        "scoring_intent_exists": scoring_intent_exists,
        "authority_lookup_complete": lookup_complete,
        "score_action": score_action,
        "recovered_reward": recovered_reward,
        "cleanup": cleanup,
        "rerun_model": False,
        "refusal_reasons": reasons,
    }
    unsigned["plan_sha256"] = _digest(unsigned)
    return unsigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    parser.add_argument("--authority-observation", type=Path)
    parser.add_argument("--resource-snapshot", type=Path)
    args = parser.parse_args()
    observation = _load_object(args.authority_observation) if args.authority_observation else None
    snapshot = _load_object(args.resource_snapshot) if args.resource_snapshot else None
    print(
        json.dumps(
            build_plan(
                args.attempt_dir,
                authority_observation=observation,
                resource_snapshot=snapshot,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
