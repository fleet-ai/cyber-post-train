"""Historical RL preview semantics remain testable without a second launcher."""

import pytest

from training.io import digest_json
from training.rl_preview import RL_TOOL_EVIDENCE_SCHEMA, rl_paid_launch_blockers


@pytest.fixture
def pair():
    config = {
        "grpo": {"max_steps": 130},
        "trainer": {"args": ["trainer.max_training_steps=130"]},
        "tasks": {"task_versions": [{"task_version_id": "train"}]},
        "eval": {"task_versions": [{"task_version_id": "dev"}]},
    }
    bindings = [
        {"task_version_id": version, "tools": ["bash", "submit_report"]}
        for version in ("train", "dev")
    ]
    preview = {
        "manifest_yaml": (
            "spec:\n  entrypoint: python train.py trainer.epochs=130 "
            "trainer.max_training_steps=130\n"
        ),
        "task_tool_allowlist_evidence": {
            "schema": RL_TOOL_EVIDENCE_SCHEMA,
            "source": "authoritative_task_version_metadata",
            "source_field": "metadata.tools",
            "bindings": bindings,
            "bindings_sha256": digest_json(bindings),
        },
    }
    return config, preview


def test_exact_bound_preview(pair):
    assert rl_paid_launch_blockers(*pair) == []
    config, preview = pair
    config["eval"]["task_versions"] = []
    evidence = preview["task_tool_allowlist_evidence"]
    evidence["bindings"].pop()
    evidence["bindings_sha256"] = digest_json(evidence["bindings"])
    assert rl_paid_launch_blockers(config, preview) == []


@pytest.mark.parametrize("config", [{}, {"kind": "sft", "grpo": {}, "tasks": {}}])
def test_not_an_rl_request(config):
    assert "expected a typed RL request" in rl_paid_launch_blockers(config, {})[0]


@pytest.mark.parametrize("value", [None, {}, 0, True, "1"])
def test_invalid_step_limit(pair, value):
    config, preview = pair
    config["grpo"] = value if value is None else {"max_steps": value}
    assert "positive integer" in ";".join(rl_paid_launch_blockers(config, preview))


@pytest.mark.parametrize(
    "trainer",
    [
        None,
        {},
        {"args": [1]},
        {"args": []},
        {"args": ["trainer.max_training_steps=12"]},
        {"args": ["trainer.max_training_steps=130"] * 2},
    ],
)
def test_request_must_bound_optimizer_steps(pair, trainer):
    config, preview = pair
    config["trainer"] = trainer
    assert "trainer.max_training_steps=130" in ";".join(rl_paid_launch_blockers(config, preview))


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        (None, "omitted manifest_yaml"),
        ("", "omitted manifest_yaml"),
        ("[", "invalid manifest_yaml"),
        ("[]", "omitted spec.entrypoint"),
        ("spec: null", "omitted spec.entrypoint"),
        ("spec: {}", "omitted spec.entrypoint"),
        ("spec:\n  entrypoint: 1", "omitted spec.entrypoint"),
        ('spec:\n  entrypoint: "\'"', "not valid shell-word syntax"),
        ("spec:\n  entrypoint: python trainer.epochs=130", "exactly one"),
        ("spec:\n  entrypoint: python train.py", "exactly one"),
    ],
)
def test_rendered_preview_not_request_intent_is_authoritative(pair, manifest, error):
    config, preview = pair
    preview["manifest_yaml"] = manifest
    assert error in ";".join(rl_paid_launch_blockers(config, preview))


@pytest.mark.parametrize(
    ("section", "value", "error"),
    [
        ("tasks", None, "object at tasks"),
        ("tasks", {}, "exact tasks.task_versions"),
        ("tasks", {"task_versions": []}, "at least one"),
        ("eval", {"task_versions": [], "task_keys": ["mutable"]}, "cannot substitute"),
        ("tasks", {"task_versions": [None]}, "no task_version_id"),
        ("tasks", {"task_versions": [{"task_version_id": "dev"}]}, "unique and disjoint"),
    ],
)
def test_task_versions_must_be_exact_unique_disjoint(pair, section, value, error):
    config, preview = pair
    config[section] = value
    assert error in ";".join(rl_paid_launch_blockers(config, preview))


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        (None, None, "lacks authoritative"),
        ("schema", "unknown", "unsupported"),
        ("source", "client", "authoritative metadata.tools"),
        ("source_field", "prompt", "authoritative metadata.tools"),
        ("bindings", None, "must be a list"),
        ("bindings_sha256", "wrong", "sha256 mismatch"),
    ],
)
def test_tool_evidence_integrity(pair, key, value, error):
    config, preview = pair
    if key is None:
        preview["task_tool_allowlist_evidence"] = value
    else:
        preview["task_tool_allowlist_evidence"][key] = value
    assert error in ";".join(rl_paid_launch_blockers(config, preview))


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (None, "not an object"),
        ({}, "no task_version_id"),
        ({"task_version_id": "wrong", "tools": ["bash", "submit_report"]}, "ordered train+eval"),
        *[
            ({"task_version_id": "train", "tools": value}, "non-empty, duplicate-free")
            for value in (None, [], [1], [""], ["bash", "bash"])
        ],
        *[
            ({"task_version_id": "train", "tools": value}, "exactly ordered")
            for value in (["submit_report", "bash"], ["bash", "submit_report", "text_editor"])
        ],
    ],
)
def test_tool_surface_cannot_drift(pair, row, error):
    config, preview = pair
    evidence = preview["task_tool_allowlist_evidence"]
    evidence["bindings"][0] = row
    evidence["bindings_sha256"] = digest_json(evidence["bindings"])
    assert error in ";".join(rl_paid_launch_blockers(config, preview))
