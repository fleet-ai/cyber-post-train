import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training.sft import compile_sft, job_request
from training.sft_runtime import optimizer_schedule

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/qualification"
BASE = CONFIG_DIR / "qwen38-teacher-production-layout-dev-v1.json"
STUDY = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v5.json"
CASES = {
    "qwen38-teacher-lr1-cosine-layout-dev-v1.json": (1e-6, 6, 6),
    "qwen38-teacher-lr30-cosine-layout-dev-v1.json": (3e-5, 6, 6),
    "qwen38-teacher-lr100-cosine-layout-dev-v1.json": (1e-4, 21, 20),
}


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def scientific_view(value: dict) -> dict:
    value = json.loads(json.dumps(value))
    for key in ("name", "output_root"):
        value.pop(key)
    value["wandb"].pop("run_id")
    value["wandb"].pop("name")
    value["wandb"].pop("group")
    value["wandb"].pop("tags")
    value["recipe"].pop("lr")
    value["recipe"].pop("checkpoint_interval")
    value.pop("pause_after_step")
    return value


def test_lr_layout_configs_are_exact_production_layout_variants():
    base = load(BASE)
    seen = set()
    for filename, (lr, pause_step, checkpoint_interval) in CASES.items():
        config = load(CONFIG_DIR / filename)
        assert scientific_view(config) == scientific_view(base)
        assert config["recipe"]["lr"] == lr
        assert config["recipe"]["checkpoint_interval"] == checkpoint_interval
        assert config["pause_after_step"] == pause_step
        identities = (
            config["name"],
            config["output_root"],
            config["wandb"]["run_id"],
            config["wandb"]["name"],
        )
        assert len(set(identities[0:1] + identities[2:])) == 1
        assert identities not in seen
        seen.add(identities)


def test_lr_layout_configs_match_frozen_v5_dev_waves_and_compile_safely(tmp_path):
    study = load(STUDY)
    expected = {
        cell["lr"]: cell["pause_after_step"]
        for wave in study["remaining_lr_qualification"]["dev_waves"]
        for cell in wave["cells"]
    }
    assert expected == {1e-6: 6, 3e-5: 6, 1e-4: 21}

    # Compilation itself is offline. Replace only the unavailable SFS manifest
    # locator with a digest-bound synthetic manifest of the same 602-row shape;
    # every model, recipe, topology, outcome protocol and request assertion below
    # still comes from the checked-in qualification configs.
    manifest = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": (
            "sha256:3c2c65eed748b16b50ef992327324b1fcaad84b190456c4aefe69c6efe748f75"
        ),
        "files": {
            "train": {
                "path": "train.parquet",
                "rows": 602,
                "sha256": "b" * 64,
                "task_keys": ["synthetic-offline-task"],
            }
        },
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    for filename, (lr, pause_step, _) in CASES.items():
        path = CONFIG_DIR / filename
        config = load(path)
        config["data"]["manifest"] = str(manifest_path)
        assert expected[lr] == pause_step
        plan = compile_sft(config, relative_to=path.parent)
        request = job_request(plan)
        assert plan["recipe"]["max_steps"] == 76
        assert optimizer_schedule(plan["recipe"])["num_warmup_steps"] == 4
        assert plan["pause_after_step"] == pause_step
        assert request["workers"] == 1
        assert request["gpus_per_worker"] == 8
        assert request["priority_class"] == "c1"
        assert request["requeueIfPreempted"] is False
        assert request["secrets"] == ["wandb-api"]
