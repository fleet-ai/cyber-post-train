from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from training import miles96_reward_signal as signal
from training import miles96_reward_signal_job as job


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _campaign(root: Path, *, varying: bool = True) -> Path:
    output = root / "campaign"
    attempts = output / "attempts"
    attempts.mkdir(parents=True)
    terminal_body = {
        "schema": "fleet_eval_campaign_terminal_v1",
        "summary": {"total": signal.EPISODES, "by_state": {"accepted": signal.EPISODES}},
    }
    _write(
        output / "EVAL_TERMINAL.json",
        {**terminal_body, "sha256": signal.digest(terminal_body)},
    )
    for index in range(signal.EPISODES):
        attempt = attempts / f"attempt-{index}"
        attempt.mkdir()
        execution = f"verifier-{index}"
        reward = float(index % 2) if varying else 1.0
        _write(
            attempt / "binding.json",
            {
                "task": {"version_id": signal.TASK_VERSION_ID},
                "verifier": {"version_id": signal.VERIFIER_VERSION_ID},
                "model": {
                    "revision": signal.MODEL_REVISION,
                    "session_model": f"qwen/{signal.SERVED_ID}",
                },
            },
        )
        _write(
            attempt / "result.json",
            {
                "task_key": signal.TASK_KEY,
                "task_version_id": signal.TASK_VERSION_ID,
                "verifier_execution_id": execution,
                "score": reward,
                "agent_termination": "completed",
                "harness_config": {
                    "max_model_requests": signal.MAX_TURNS,
                    "max_output_tokens": signal.MAX_TOKENS_PER_TURN,
                    "timeout_seconds": signal.EPISODE_TIMEOUT_S,
                },
            },
        )
        _write(
            attempt / "reward-result.json",
            {
                "reward": reward,
                "verifier_execution_id": execution,
                "direct_authority_attestation": {
                    "context": {
                        "task_version_id": signal.TASK_VERSION_ID,
                        "verifier_version_id": signal.VERIFIER_VERSION_ID,
                    }
                },
            },
        )
        _write(
            attempt / "cleanup.json",
            {"instance_created": True, "instance_closed": True, "containers_removed": True},
        )
        _write(attempt / "ACCEPTED.json", {"accepted": True})
    return output


def test_seal_preserves_private_rewards_and_public_mixed_signal_receipt(tmp_path: Path) -> None:
    output = _campaign(tmp_path)
    receipt = signal.seal(output)
    assert receipt["reward_variation"] is True
    assert receipt["completed_episode_count"] == signal.EPISODES
    assert "rewards" not in receipt
    assert "verifier_execution_id" not in json.dumps(receipt)
    private = json.loads((output / ".private-signal.json").read_text())
    assert {row["reward"] for row in private["rows"]} == {0.0, 1.0}
    assert stat.S_IMODE((output / ".private-signal.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "SIGNAL_VALIDATED.json").stat().st_mode) == 0o444


def test_seal_rejects_uniform_rewards(tmp_path: Path) -> None:
    output = _campaign(tmp_path, varying=False)
    with pytest.raises(ValueError, match="no reward variation"):
        signal.seal(output)
    assert not (output / "SIGNAL_VALIDATED.json").exists()


def test_job_packet_is_exactly_eight_cpu_episodes_and_alerts_off() -> None:
    packet = job.build_packet()
    proof = job.validate_packet(packet)
    assert proof["planned_episodes"] == 8
    config_map, rendered_job = packet["bundle"]["items"]
    assert config_map["immutable"] is True
    assert rendered_job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert rendered_job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert rendered_job["spec"]["backoffLimit"] == 0
    assert rendered_job["spec"]["suspend"] is True
    assert rendered_job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(rendered_job["spec"]["template"]["spec"])
    assert packet["task_identity"] == {
        "task_key": signal.TASK_KEY,
        "task_version_id": signal.TASK_VERSION_ID,
        "verifier_version_id": signal.VERIFIER_VERSION_ID,
        "task_set_sha256": signal.TASK_SET_SHA256,
        "tool_catalog_sha256": signal.TOOL_CATALOG_SHA256,
    }
    assert packet["episode"] == {
        "count": 8,
        "concurrency": 8,
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "timeout_seconds": 2400,
    }
    sequence = packet["execution_sequence"]
    assert sequence["config_map_create_request_count"] == 1
    assert sequence["job_create_request_count"] == 1
    assert sequence["controller_managed_unsuspend"] is True
    assert sequence["operator_patch_request_count"] == 0
    assert sequence["post_create_or_patch_requests_allowed"] is False


def test_job_packet_embeds_current_sources() -> None:
    config_map = job.build_packet()["bundle"]["items"][0]
    assert config_map["data"]["signal.py"] == Path(signal.__file__).read_text()
    assert (
        config_map["data"]["run.sh"]
        == (job.ROOT / "evals/fleet/scripts/run_miles96_reward_signal_v1.sh").read_text()
    )
