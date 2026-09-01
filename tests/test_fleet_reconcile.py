from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import reconcile, self_hosted


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _with_digest(value: dict, field: str) -> dict:
    value[field] = self_hosted.sha256(self_hosted.canonical_json(value))
    return value


def _set_and_redigest(value: dict, field: str, replacement) -> None:
    value[field] = replacement
    value.pop("observation_sha256", None)
    _with_digest(value, "observation_sha256")


def _attempt(tmp_path: Path, *, report: bool = True) -> Path:
    attempt = tmp_path / "attempt"
    run_id = "chris-cyber-q36-reconcile-test"
    _write(
        attempt / "binding.json",
        {
            "run_id": run_id,
            "task": {"key": "train-task__blackbox_ctf_v1", "version_id": "version-1"},
            "authority": {
                "scoring_mode": "binary",
                "multi_app_aggregation_mode": "binary",
            },
        },
    )
    _write(
        attempt / "runtime-binding.json",
        {"instance_id": "instance-1", "evidence_run_id": "evidence-1"},
    )
    suffix = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    _write(
        attempt / "resource-plan.json",
        _with_digest(
            {
                "schema_version": reconcile.RESOURCE_PLAN_SCHEMA,
                "run_id": run_id,
                "instance_id": "instance-1",
                "evidence_run_id": "evidence-1",
                "containers": [
                    f"qwen-agent-{suffix}",
                    f"qwen-model-proxy-{suffix}",
                    f"qwen-mcp-proxy-{suffix}",
                ],
                "network": "private",
            },
            "resource_plan_sha256",
        ),
    )
    _write(
        attempt / "agent-output" / "qwen-stream.jsonl",
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 10,
        },
    )
    events = [
        {
            "type": "assistant",
            "message": {"role": "model", "parts": [{"text": "finished"}]},
        }
    ]
    if report:
        events.extend(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": "mcp__fleet__submit_report",
                                    "args": {"redacted": True},
                                }
                            }
                        ],
                    },
                },
                {
                    "type": "tool_result",
                    "message": {
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "call-1",
                                    "name": "mcp__fleet__submit_report",
                                    "response": "accepted",
                                }
                            }
                        ]
                    },
                },
            ]
        )
    chat = attempt / "qwen-home" / "projects" / "workspace" / "chats" / "trace.jsonl"
    chat.parent.mkdir(parents=True)
    chat.write_text("".join(json.dumps(event) + "\n" for event in events))
    return attempt


def _observation(*, results: list[dict] | None = None, status: str = "running") -> dict:
    return _with_digest(
        {
            "schema_version": reconcile.OBSERVATION_SCHEMA,
            "source": "authoritative_verifier_store",
            "lookup_complete": True,
            "observed_at": "2026-09-01T00:00:00Z",
            "run_id": "chris-cyber-q36-reconcile-test",
            "instance_id": "instance-1",
            "evidence_run_id": "evidence-1",
            "task_key": "train-task__blackbox_ctf_v1",
            "task_version_id": "version-1",
            "instance_status": status,
            "verifier_results": results or [],
        },
        "observation_sha256",
    )


def _snapshot(*, containers: list[dict], networks: list[str] | None = None) -> dict:
    plan = _with_digest(
        {
            "schema_version": reconcile.RESOURCE_PLAN_SCHEMA,
            "run_id": "chris-cyber-q36-reconcile-test",
            "instance_id": "instance-1",
            "evidence_run_id": "evidence-1",
            "containers": [],
            "network": "private",
        },
        "resource_plan_sha256",
    )
    # Only the digest is needed here; derive it from the exact fixture written by
    # _attempt rather than using this placeholder's container list.
    suffix = hashlib.sha256(b"chris-cyber-q36-reconcile-test").hexdigest()[:8]
    exact_plan = {
        **plan,
        "containers": [
            f"qwen-agent-{suffix}",
            f"qwen-model-proxy-{suffix}",
            f"qwen-mcp-proxy-{suffix}",
        ],
    }
    exact_plan.pop("resource_plan_sha256")
    digest = self_hosted.sha256(self_hosted.canonical_json(exact_plan))
    return {
        "schema_version": reconcile.RESOURCE_SNAPSHOT_SCHEMA,
        "lookup_complete": True,
        "run_id": "chris-cyber-q36-reconcile-test",
        "resource_plan_sha256": digest,
        "containers": containers,
        "networks": networks or [],
    }


def test_crash_after_agent_requires_authority_lookup_and_never_reruns_model(
    tmp_path: Path,
) -> None:
    attempt = _attempt(tmp_path, report=False)
    unknown = reconcile.build_plan(attempt)
    assert unknown["score_action"] == "refuse"
    assert unknown["rerun_model"] is False
    assert unknown["refusal_reasons"] == ["authoritative_lookup_required_before_scoring"]

    safe = reconcile.build_plan(attempt, authority_observation=_observation())
    assert safe["score_action"] == "score_once_from_existing_trace"
    assert safe["conversation"]["submit_report_completed"] is False
    assert safe["rerun_model"] is False


def test_crash_after_submit_can_score_only_after_complete_empty_lookup(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    _write(
        attempt / "scoring-intent.json",
        _valid_scoring_intent(attempt),
    )
    blocked = reconcile.build_plan(attempt)
    assert blocked["score_action"] == "refuse"
    assert blocked["scoring_intent_exists"] is True

    safe = reconcile.build_plan(attempt, authority_observation=_observation())
    assert safe["score_action"] == "score_once_from_existing_trace"
    assert safe["conversation"]["submit_report_completed"] is True


def test_crash_after_score_recovers_authority_and_refuses_duplicate_score(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    plan = reconcile.build_plan(
        attempt,
        authority_observation=_observation(
            results=[{"verifier_execution_id": "verify-1", "reward": 1}]
        ),
    )
    assert plan["score_action"] == "recover_authoritative_result"
    assert plan["recovered_reward"] == {
        "verifier_execution_id": "verify-1",
        "reward": 1.0,
    }
    assert plan["rerun_model"] is False


def test_local_score_is_terminal_and_conflict_fails_closed(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    _write(
        attempt / "reward-result.json",
        {
            "task_key": "train-task__blackbox_ctf_v1",
            "task_version_id": "version-1",
            "instance_id": "instance-1",
            "reward": 0,
            "verifier_execution_id": "verify-local",
        },
    )
    terminal = reconcile.build_plan(attempt)
    assert terminal["score_action"] == "none_already_scored"

    conflict = reconcile.build_plan(
        attempt,
        authority_observation=_observation(
            results=[{"verifier_execution_id": "verify-other", "reward": 1}]
        ),
    )
    assert conflict["score_action"] == "refuse"
    assert conflict["refusal_reasons"] == ["local_and_authoritative_reward_conflict"]


def test_orphan_proxy_cleanup_plan_is_exact_and_content_free(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    suffix = hashlib.sha256(b"chris-cyber-q36-reconcile-test").hexdigest()[:8]
    snapshot = _snapshot(
        containers=[
            {"name": f"qwen-model-proxy-{suffix}", "networks": ["bridge", "private"]},
            {"name": f"qwen-mcp-proxy-{suffix}", "networks": ["bridge", "private"]},
        ],
        networks=["private"],
    )
    plan = reconcile.build_plan(
        attempt,
        authority_observation=_observation(
            results=[{"verifier_execution_id": "verify-1", "reward": 0}]
        ),
        resource_snapshot=snapshot,
    )
    assert plan["cleanup"] == {
        "action": "cleanup_once",
        "targets": [f"qwen-mcp-proxy-{suffix}", f"qwen-model-proxy-{suffix}"],
        "networks": ["private"],
    }
    assert "prompt" not in json.dumps(plan)


def test_cleanup_snapshot_refuses_unrelated_container(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    snapshot = _snapshot(containers=[{"name": "peer-container", "networks": ["bridge"]}])
    with pytest.raises(reconcile.ReconcileError, match="unrelated container"):
        reconcile.build_plan(attempt, resource_snapshot=snapshot)


def _valid_scoring_intent(attempt: Path) -> dict:
    binding = json.loads((attempt / "binding.json").read_text())
    runtime = json.loads((attempt / "runtime-binding.json").read_text())
    events, _, _ = self_hosted.load_qwen_chat_trace(attempt / "qwen-home")
    messages = self_hosted.normalize_qwen_conversation(events)
    final = self_hosted.final_answer_from_conversation(messages)
    payload = {
        "instance_id": runtime["instance_id"],
        "final_answer": final,
        "conversation": messages,
        "scoring_mode": binding["authority"]["scoring_mode"],
        "multi_app_aggregation_mode": binding["authority"]["multi_app_aggregation_mode"],
    }
    return {
        "schema_version": reconcile.SCORING_INTENT_SCHEMA,
        "run_id": binding["run_id"],
        "instance_id": runtime["instance_id"],
        "evidence_run_id": runtime["evidence_run_id"],
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
    }


def test_pending_score_blocks_cleanup(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    suffix = hashlib.sha256(b"chris-cyber-q36-reconcile-test").hexdigest()[:8]
    snapshot = _snapshot(
        containers=[{"name": f"qwen-model-proxy-{suffix}", "networks": ["bridge", "private"]}],
        networks=["private"],
    )
    plan = reconcile.build_plan(
        attempt, authority_observation=_observation(), resource_snapshot=snapshot
    )
    assert plan["score_action"] == "score_once_from_existing_trace"
    assert plan["cleanup"]["action"] == "blocked_until_score_reconciled"
    assert plan["cleanup"]["eligible_action_after_score"] == "cleanup_once"


def test_scoring_intent_must_match_exact_preserved_request(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    intent = _valid_scoring_intent(attempt)
    intent["request_sha256"] = "sha256:" + "0" * 64
    _write(attempt / "scoring-intent.json", intent)
    with pytest.raises(reconcile.ReconcileError, match="preserved trace"):
        reconcile.build_plan(attempt)


def test_resource_plan_is_required_and_exact(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    (attempt / "resource-plan.json").unlink()
    with pytest.raises(reconcile.ReconcileError, match="resource-plan.json"):
        reconcile.build_plan(attempt)


def test_snapshot_cannot_propose_peer_network(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    suffix = hashlib.sha256(b"chris-cyber-q36-reconcile-test").hexdigest()[:8]
    snapshot = _snapshot(
        containers=[
            {
                "name": f"qwen-model-proxy-{suffix}",
                "networks": ["bridge", "private", "peer-network"],
            }
        ],
        networks=["peer-network"],
    )
    with pytest.raises(reconcile.ReconcileError, match="unplanned network|peer network"):
        reconcile.build_plan(attempt, resource_snapshot=snapshot)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("lookup_complete"), "fields"),
        (lambda value: _set_and_redigest(value, "lookup_complete", False), "complete lookup"),
        (lambda value: value.__setitem__("instance_id", "forged"), "self-digest"),
        (
            lambda value: value.__setitem__("observation_sha256", "sha256:" + "0" * 64),
            "self-digest",
        ),
    ],
)
def test_authority_observation_rejects_forged_or_incomplete_receipts(
    tmp_path: Path, mutate, message: str
) -> None:
    attempt = _attempt(tmp_path)
    observation = _observation()
    mutate(observation)
    with pytest.raises(reconcile.ReconcileError, match=message):
        reconcile.build_plan(attempt, authority_observation=observation)


def test_write_json_once_is_an_exclusive_durable_claim(tmp_path: Path) -> None:
    path = tmp_path / "intent.json"
    self_hosted.write_json_once(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        self_hosted.write_json_once(path, {"attempt": 2})
    assert json.loads(path.read_text()) == {"attempt": 1}
