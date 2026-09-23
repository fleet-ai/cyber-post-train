"""Provider-free science and identity checks for the parallel fast-update canary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import skyrl_reward_rayjob as historical

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs/qualification"
PROD11_RUN = QUALIFICATION / "qwen38-rl-reward-canary-prod-v11.json"
FAST_RUN = QUALIFICATION / "qwen38-rl-reward-canary-prod-v11-fast1.json"
PROD11_DATA = QUALIFICATION / "qwen38-rl-reward-canary-data-prod-v11.json"
FAST_DATA = QUALIFICATION / "qwen38-rl-reward-canary-data-prod11-fast1.json"
FAST_IDENTITY = QUALIFICATION / "qwen38-rl-reward-canary-prod11-fast1-identity-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_fast_update_changes_only_operational_identity_and_baseline_eval() -> None:
    prod11 = _load(PROD11_RUN)
    fast = _load(FAST_RUN)
    normalized = copy.deepcopy(fast)
    normalized["name"] = prod11["name"]
    normalized["output_root"] = prod11["output_root"]
    normalized["data"] = copy.deepcopy(prod11["data"])
    normalized["wandb"]["run_id"] = prod11["wandb"]["run_id"]
    assert normalized["recipe"].pop("eval_before_train") is False

    assert normalized == prod11
    assert fast["recipe"]["groups"] == 1
    assert fast["recipe"]["samples_per_prompt"] == 8
    assert fast["recipe"]["steps"] == 1
    assert fast["recipe"]["eval_interval"] == 1
    assert fast["recipe"]["checkpoint_interval"] == 1
    assert fast["recipe"]["keep_checkpoints"] == 2
    assert fast["cluster"]["priority"] == "c1"


def test_fast_update_reuses_exact_reviewed_task_and_compaction_contract() -> None:
    prod11 = _load(PROD11_DATA)
    fast = _load(FAST_DATA)
    normalized = copy.deepcopy(fast)
    normalized["name"] = prod11["name"]
    normalized["output"] = prod11["output"]

    assert normalized == prod11
    assert fast["task_set"] == "../data/qwen38-rl-reward-canary-task-set-v3.json"
    assert fast["limits"] == {
        "compaction_summary_tokens": 8192,
        "compaction_trigger_tokens": 163840,
        "context_tokens": 262144,
        "episode_seconds": 14400,
        "generation_chunk_tokens": 4096,
        "max_tokens_per_turn": 32768,
        "max_turns": 1200,
        "response_tokens": 4194304,
        "tool_result_chars": 50000,
        "tool_seconds": 330,
    }


def test_fast_update_identity_is_sealed_fresh_and_never_reuses_prod11_output() -> None:
    raw = _load(FAST_IDENTITY)
    identity = historical.load_identity(FAST_IDENTITY)
    body = {key: value for key, value in raw.items() if key != "sha256"}

    assert raw["sha256"] == "sha256:" + digest(body)
    assert identity.run_name == "chris-q38-rlreward-prod11-fast1"
    assert identity.predecessor_run_name == "chris-q38-rlreward-prod11"
    assert identity.output_root == "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1"
    assert identity.output_root != "/mnt/sfs/jobs/chris-q38-rlreward-prod11"
    assert identity.data_root != identity.predecessor_data_root
