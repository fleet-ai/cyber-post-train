"""Seal reward-only evidence for the bounded Miles96 task qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "cyber_qwen38_miles96_task_signal_evidence_v1"
PRIVATE_SCHEMA = "cyber_qwen38_miles96_private_signal_detail_v1"
TASK_KEY = "cysec1-2-fira-gen_blackbox-f8d6badcc03f916a8d15b9d0__blackbox_ctf_v1"
TASK_VERSION_ID = "0920e798-c7e7-4da6-9d5e-ebeba45ec05a"
VERIFIER_VERSION_ID = "9356b7ca-43b4-4926-a871-d9a95b41f6e5"
TASK_SET_SHA256 = "sha256:5de88eada94119fd21ac715bb1956a745376d775c13ce7d5584936e70b3a003a"
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
MAX_TURNS = 32
MAX_TOKENS_PER_TURN = 8192
EPISODE_TIMEOUT_S = 2400
EPISODES = 8
SERVED_ID = "chris-q38-base-pass4-v1"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required signal input is absent: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"required signal input is invalid: {path.name}")
    return value


def _write_once(path: Path, value: dict[str, Any], mode: int) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def seal(output: Path) -> dict[str, Any]:
    terminal = _read(output / "EVAL_TERMINAL.json")
    if terminal.get("schema") != "fleet_eval_campaign_terminal_v1":
        raise ValueError("evaluation terminal schema differs")
    if terminal.get("sha256") != digest({k: v for k, v in terminal.items() if k != "sha256"}):
        raise ValueError("evaluation terminal digest differs")
    if (
        terminal.get("summary", {}).get("total") != EPISODES
        or terminal.get("summary", {}).get("by_state", {}).get("accepted") != EPISODES
    ):
        raise ValueError("evaluation terminal did not accept exactly eight cells")
    attempts = sorted((output / "attempts").iterdir())
    if len(attempts) != EPISODES or any(
        path.is_symlink() or not path.is_dir() for path in attempts
    ):
        raise ValueError("bounded signal campaign did not preserve exactly eight attempts")
    private_rows = []
    for attempt in attempts:
        result = _read(attempt / "result.json")
        reward = _read(attempt / "reward-result.json")
        binding = _read(attempt / "binding.json")
        cleanup = _read(attempt / "cleanup.json")
        accepted = _read(attempt / "ACCEPTED.json")
        score = reward.get("reward")
        attestation = reward.get("direct_authority_attestation") or {}
        context = attestation.get("context") or {}
        if (
            result.get("task_key") != TASK_KEY
            or result.get("task_version_id") != TASK_VERSION_ID
            or binding.get("task", {}).get("version_id") != TASK_VERSION_ID
            or binding.get("verifier", {}).get("version_id") != VERIFIER_VERSION_ID
            or binding.get("model", {}).get("revision") != MODEL_REVISION
            or binding.get("model", {}).get("session_model") != f"qwen/{SERVED_ID}"
            or context.get("task_version_id") != TASK_VERSION_ID
            or context.get("verifier_version_id") != VERIFIER_VERSION_ID
            or result.get("verifier_execution_id") != reward.get("verifier_execution_id")
            or result.get("score") != score
            or result.get("agent_termination") != "completed"
            or result.get("harness_config", {}).get("max_model_requests") != MAX_TURNS
            or result.get("harness_config", {}).get("max_output_tokens") != MAX_TOKENS_PER_TURN
            or result.get("harness_config", {}).get("timeout_seconds") != EPISODE_TIMEOUT_S
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or cleanup.get("instance_closed") is not True
            or cleanup.get("containers_removed") is not True
            or accepted.get("accepted") is not True
        ):
            raise ValueError("one bounded signal attempt is incomplete")
        private_rows.append(
            {
                "attempt_sha256": "sha256:"
                + digest(
                    {
                        "result": result,
                        "reward": reward,
                        "binding": binding,
                        "cleanup": cleanup,
                        "accepted": accepted,
                    }
                ),
                "reward": float(score),
                "verifier_execution_id": result["verifier_execution_id"],
            }
        )
    rewards = [row["reward"] for row in private_rows]
    if len({row["verifier_execution_id"] for row in private_rows}) != EPISODES:
        raise ValueError("verifier execution identities are not unique")
    if len(set(rewards)) < 2:
        raise ValueError("exact task produced no reward variation")
    private_body = {
        "schema": PRIVATE_SCHEMA,
        "task_version_id": TASK_VERSION_ID,
        "verifier_version_id": VERIFIER_VERSION_ID,
        "episode_count": EPISODES,
        "rows": private_rows,
        "terminal_sha256": terminal["sha256"],
    }
    private = {**private_body, "sha256": "sha256:" + digest(private_body)}
    private_path = output / ".private-signal.json"
    _write_once(private_path, private, 0o600)
    body = {
        "schema": SCHEMA,
        "task_key": TASK_KEY,
        "task_version_id": TASK_VERSION_ID,
        "verifier_version_id": VERIFIER_VERSION_ID,
        "task_set_sha256": TASK_SET_SHA256,
        "tool_catalog_sha256": TOOL_CATALOG_SHA256,
        "max_turns": MAX_TURNS,
        "max_tokens_per_turn": MAX_TOKENS_PER_TURN,
        "episode_timeout_s": EPISODE_TIMEOUT_S,
        "completed_episode_count": EPISODES,
        "finite_rewards": True,
        "reward_variation": True,
        "all_instances_released": True,
        "source_receipt_sha256": private["sha256"],
    }
    receipt = {**body, "sha256": "sha256:" + digest(body)}
    _write_once(output / "SIGNAL_VALIDATED.json", receipt, 0o444)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
