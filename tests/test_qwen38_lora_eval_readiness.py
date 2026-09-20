import hashlib
import json
from pathlib import Path

from evals.fleet import rollout_worker

ROOT = Path(__file__).parents[1]
DATA = ROOT / "configs" / "data"
EVALS = ROOT / "configs" / "evaluation"

TASK_FIELDS = {
    "task_key",
    "task_version_id",
    "env_key",
    "env_version",
    "environment_version_id",
    "data_key",
    "data_version",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def unsigned_sha256(value: dict, field: str = "sha256") -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    body = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def test_fleet_final_task_set_is_exact_launcher_ready_projection():
    final_lock = read(DATA / "qwen-blackbox-study-final-test-v1.json")
    binding_source = read(DATA / "qwen-blackbox-eligible-v1.json")
    selection = read(EVALS / "qwen38-lora-fleet-final-task-set-v1.json")

    assert final_lock["sha256"] == unsigned_sha256(final_lock)
    assert "sha256:" + binding_source["sha256"] == unsigned_sha256(binding_source)
    assert selection["sha256"] == unsigned_sha256(selection)
    assert selection["selection_basis"] == {
        "final_test_lock_path": "configs/data/qwen-blackbox-study-final-test-v1.json",
        "final_test_lock_sha256": final_lock["sha256"],
        "runtime_binding_source_path": "configs/data/qwen-blackbox-eligible-v1.json",
        "runtime_binding_source_sha256": "sha256:" + binding_source["sha256"],
        "outcomes_or_losses_used": False,
    }

    expected_identity = [
        (task["task_key"], task["task_version_id"]) for task in final_lock["tasks"]
    ]
    actual_identity = [(task["task_key"], task["task_version_id"]) for task in selection["tasks"]]
    assert actual_identity == expected_identity
    assert len(set(actual_identity)) == 10

    source_by_identity = {}
    for task in binding_source["task_versions"]:
        identity = (task["task_key"], task["task_version_id"])
        assert identity not in source_by_identity
        source_by_identity[identity] = task
    for selected in selection["tasks"]:
        assert set(selected) == TASK_FIELDS
        source = source_by_identity[(selected["task_key"], selected["task_version_id"])]
        environment = source["environment"]
        assert selected == {
            "task_key": source["task_key"],
            "task_version_id": source["task_version_id"],
            "env_key": environment["id"],
            "env_version": environment["version"],
            "environment_version_id": environment["version_id"],
            "data_key": environment["data_id"],
            "data_version": environment["data_version"],
        }

    # This is the exact parser used by `cyber-post-train eval prepare`.
    assert set(rollout_worker._selection_index(selection)) == {
        task["task_version_id"] for task in final_lock["tasks"]
    }

    protocol = read(EVALS / "qwen38-lora-fleet-final-opencode-pass1-v1.template.json")
    assert protocol["benchmark"]["task_manifest_sha256"] == final_lock["sha256"]
    assert protocol["benchmark"]["environment_manifest_sha256"] == selection["sha256"]
