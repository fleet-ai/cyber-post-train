"""The published Fleet RL examples must stay loadable and topology-consistent."""

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from training import miles

ROOT = Path(__file__).resolve().parents[1] / "configs/runs"
DATA = ROOT / "qwen38-27b-fleet-rl-data.example.yaml"
CONVERT = ROOT / "qwen38-27b-fleet-miles-convert.example.yaml"
TRAIN = ROOT / "qwen38-27b-fleet-rl-multinode.example.yaml"
MODEL_ROOT = "/mnt/sfs/models/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def load(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", [DATA, CONVERT, TRAIN])
def test_examples_are_placeholders_without_credentials(path):
    text = path.read_text()
    assert not re.search(r"(?i)(api[_-]?key|token|password|secret)\s*:", text)
    # Unreplaced values must be obvious; a real identity must never be implied.
    for value in re.findall(r"<[^>]*>", text):
        assert re.fullmatch(r"<[a-z0-9.<>/-]+>", value), value
    assert load(path)


def test_model_identity_is_the_reviewed_pinned_lock():
    data, convert, train = load(DATA), load(CONVERT), load(TRAIN)
    assert data["model_lock"] == convert["model"]["lock"] == train["model"]["lock"]
    assert data["model_root"] == convert["model"]["root"] == train["model"]["root"] == MODEL_ROOT
    assert data["backend"] == train["backend"] == "miles"
    assert Path(data["model_lock"]).exists() and Path(train["model"]["weights"]).exists()


def test_example_recipe_matches_the_supported_multi_node_topology():
    recipe = load(TRAIN)["recipe"]
    fields = {field.name for field in dataclasses.fields(miles.MilesConfig)}
    assert set(recipe) <= fields and recipe["nodes"] > 1
    config = miles.MilesConfig(
        name="synthetic-rl",
        output_root="/mnt/sfs/jobs/synthetic-rl",
        model_root="/mnt/sfs/models/synthetic-hf",
        torch_dist_root="/mnt/sfs/models/synthetic-dist",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
        **recipe,
    )
    layout = miles.topology(config)
    assert layout["data_parallel_size"] == recipe["nodes"] <= miles.MAX_NODES
    assert not layout["global_batch_size"] % layout["data_parallel_size"]


def test_example_limits_match_the_native_context_envelope():
    limits = load(DATA)["limits"]
    config = miles.MilesConfig(
        name="synthetic-rl",
        output_root="/mnt/sfs/jobs/synthetic-rl",
        model_root="/mnt/sfs/models/synthetic-hf",
        torch_dist_root="/mnt/sfs/models/synthetic-dist",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
        context_tokens=limits["context_tokens"],
        response_tokens=limits["response_tokens"],
        tokens_per_turn=limits["max_tokens_per_turn"],
    )
    config.validate()
