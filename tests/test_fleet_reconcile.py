from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import reconcile, self_hosted


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _attempt(tmp_path: Path, *, report: bool = True) -> Path:
    attempt = tmp_path / "attempt"
    run_id = "chris-cyber-q36-reconcile-test"
    _write(
        attempt / "binding.json",
        {
            "run_id": run_id,
            "task": {"key": "train-task__blackbox_ctf_v1", "version_id": "version-1"},
        },
    )
    _write(
        attempt / "runtime-binding.json",
        {"instance_id": "instance-1", "evidence_run_id": "evidence-1"},
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
    return {
        "schema_version": reconcile.OBSERVATION_SCHEMA,
        "source": "authoritative_verifier_store",
        "lookup_complete": True,
        "run_id": "chris-cyber-q36-reconcile-test",
        "instance_id": "instance-1",
        "evidence_run_id": "evidence-1",
        "instance_status": status,
        "verifier_results": results or [],
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
        {
            "run_id": "chris-cyber-q36-reconcile-test",
            "instance_id": "instance-1",
            "evidence_run_id": "evidence-1",
        },
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
    snapshot = {
        "schema_version": reconcile.RESOURCE_SNAPSHOT_SCHEMA,
        "run_id": "chris-cyber-q36-reconcile-test",
        "containers": [
            {"name": f"qwen-model-proxy-{suffix}", "networks": ["bridge", "private"]},
            {"name": f"qwen-mcp-proxy-{suffix}", "networks": ["bridge", "private"]},
        ],
    }
    plan = reconcile.build_plan(
        attempt, authority_observation=_observation(), resource_snapshot=snapshot
    )
    assert plan["cleanup"] == {
        "action": "cleanup_once",
        "targets": [f"qwen-mcp-proxy-{suffix}", f"qwen-model-proxy-{suffix}"],
        "networks": ["private"],
    }
    assert "prompt" not in json.dumps(plan)


def test_cleanup_snapshot_refuses_unrelated_container(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path)
    snapshot = {
        "schema_version": reconcile.RESOURCE_SNAPSHOT_SCHEMA,
        "run_id": "chris-cyber-q36-reconcile-test",
        "containers": [{"name": "peer-container", "networks": ["bridge"]}],
    }
    with pytest.raises(reconcile.ReconcileError, match="unrelated container"):
        reconcile.build_plan(attempt, resource_snapshot=snapshot)


def test_write_json_once_is_an_exclusive_durable_claim(tmp_path: Path) -> None:
    path = tmp_path / "intent.json"
    self_hosted.write_json_once(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        self_hosted.write_json_once(path, {"attempt": 2})
    assert json.loads(path.read_text()) == {"attempt": 1}
