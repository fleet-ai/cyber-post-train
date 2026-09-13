import hashlib
import json
from pathlib import Path

from training.io import canonical_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "data"
QUALIFICATION = ROOT / "configs" / "qualification"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def sealed(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return "sha256:" + hashlib.sha256(canonical_json(body).encode()).hexdigest()


def test_dev6_is_a_fresh_c1_one_update_canary_on_the_filtered_set() -> None:
    task_set = load(DATA / "qwen38-rl-filtered-canary-task-set-v2.json")
    split = load(DATA / "qwen38-rl-filtered-canary-split-v2.json")
    parent_task_set = load(DATA / "qwen38-rl-filtered-canary-task-set-v1.json")
    parent_split = load(DATA / "qwen38-rl-filtered-canary-split-v1.json")
    reward_prior = load(DATA / "qwen38-rl-reward-canary-exact-version-evidence-v1.json")
    data = load(QUALIFICATION / "qwen38-miles-rl-reward-canary-data-dev-v6.json")
    run = load(QUALIFICATION / "qwen38-miles-rl-reward-canary-dev-v6.json")
    launch = load(QUALIFICATION / "qwen38-miles-rl-reward-canary-launch-dev-v6.json")

    assert task_set["sha256"] == sealed(task_set)
    assert split["sha256"] == sealed(split)
    selected = {(row["task_key"], row["task_version_id"]) for row in task_set["tasks"]}
    parent = {
        (row["task_key"], row["task_version_id"]) for row in parent_task_set["tasks"]
    }
    assignments = {
        (row["task_key"], row["task_version_id"]): row["split"]
        for row in split["tasks"]
    }
    parent_assignments = {
        (row["task_key"], row["task_version_id"]): row["split"]
        for row in parent_split["tasks"]
    }
    assert selected == set(assignments) < parent
    assert assignments == {key: parent_assignments[key] for key in selected}
    assert sorted(assignments.values()) == ["dev", "train"]
    assert "reward_signal_provenance" not in task_set
    evidence = task_set["selection_evidence"]
    assert evidence["parent_task_set_self_sha256"] == parent_task_set["sha256"]
    assert evidence["parent_task_set_file_sha256"] == "sha256:" + hashlib.sha256(
        (DATA / evidence["parent_task_set_path"]).read_bytes()
    ).hexdigest()
    assert evidence["reward_prior_self_sha256"] == reward_prior["sha256"]
    assert evidence["reward_prior_file_sha256"] == "sha256:" + hashlib.sha256(
        (DATA / evidence["reward_prior_path"]).read_bytes()
    ).hexdigest()
    train = next(key for key, value in assignments.items() if value == "train")
    selected_prior = reward_prior["selected_version"]
    assert train == (selected_prior["task_key"], selected_prior["task_version_id"])

    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert data["name"] == "chris-q38-miles-rlreward-dev6"
    assert data["output"] == run["data"]["root"]
    assert data["limits"]["tool_result_chars"] == 4000
    assert run["recipe"]["steps"] == 1
    assert run["recipe"]["groups"] == 1
    assert run["recipe"]["samples_per_prompt"] == 8
    assert run["cluster"]["target"] == "dev"
    assert run["cluster"]["priority"] == "c1"
    assert launch["identities"] == {
        "run_name": run["name"],
        "cyber_run_id": run["name"],
        "data_root": data["output"],
        "prepared_root": "/mnt/sfs/jobs/chris-q38-miles-rlreward-preflight-dev6-v1",
        "output_root": run["output_root"],
        "wandb_run_id": run["wandb"]["run_id"],
    }
    assert launch["jobs_api_constraints"] == {
        "priority_request": "c1",
        "rendered_queue_priority": "q1",
        "effective_priority": 10000,
        "maximum_allowed_priority": "c1",
        "requeue_if_preempted": False,
        "current_rendered_shared_memory": "64Gi",
    }
    text = "\n".join(
        path.read_text()
        for path in (
            DATA / "qwen38-rl-filtered-canary-task-set-v2.json",
            DATA / "qwen38-rl-filtered-canary-split-v2.json",
            QUALIFICATION / "qwen38-miles-rl-reward-canary-data-dev-v6.json",
            QUALIFICATION / "qwen38-miles-rl-reward-canary-dev-v6.json",
            QUALIFICATION / "qwen38-miles-rl-reward-canary-launch-dev-v6.json",
        )
    ).lower()
    assert "webexploitbench" not in text
    assert '"c0"' not in text and '"q0"' not in text
