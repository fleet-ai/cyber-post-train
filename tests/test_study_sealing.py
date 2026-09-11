"""Offline tests for score-blind study launch and result sealing."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals import study_sealing as sealing
from evals.fleet import dev_outcome_protocol as fleet
from evals.webexploitbench import study_parent_protocol as web
from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seal(value: dict) -> dict:
    return {**value, "sha256": digest_json(value)}


def reference(path: Path) -> dict:
    value = load(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": value["sha256"],
        "file_sha256": file_sha256(path),
    }


def exact(fields) -> dict:
    values = {}
    for index, key in enumerate(fields, 1):
        if key.endswith("sha256"):
            values[key] = "sha256:" + f"{index:064x}"[-64:]
        elif key == "checkpoint_step":
            values[key] = 1
        elif key == "sandbox_resources":
            values[key] = {"cpu": 8, "memory_mb": 32768, "disk_size_mb": 102400}
        elif key in {"evaluator_image", "agent_image", "netproxy_image"}:
            values[key] = f"registry/{key}@sha256:" + f"{index:064x}"[-64:]
        else:
            values[key] = f"exact-{key}"
    return values


def runtime() -> dict:
    value = exact(sealing.MATCHED_RUNTIME_FIELDS)
    value.update(
        {
            "endpoint_origin": "https://inference.flt.build",
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "context_length": 262144,
            "quantization": "none",
        }
    )
    return value


def child(surface: str, private_root: Path, *, launchable: bool = True) -> dict:
    private_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if surface == "fleet_dev":
        parent_path = CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json"
        base_path = CONFIG / "qwen38-blackbox-fleet-dev-a-base-control-v1.json"
        post, base = exact(fleet.UNBOUND_CHECKPOINT_FIELDS), exact(
            fleet.UNBOUND_BASE_SERVING_FIELDS
        )
        base["weights_manifest_sha256"] = "sha256:" + "a" * 64
        post["weights_manifest_sha256"] = "sha256:" + "b" * 64
        for key in (
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "agent_image_digest",
            "proxy_image_digest",
        ):
            post[key] = base[key]
        bindings = {"post_sft": post, "base": base, "matched_runtime": runtime()}
        base_ref = reference(base_path)
    elif surface == "fleet_final":
        parent_path = CONFIG / "qwen38-blackbox-fleet-final-outcome-protocol-v1.json"
        base, post = exact(fleet.UNBOUND_CHECKPOINT_FIELDS), exact(
            fleet.UNBOUND_CHECKPOINT_FIELDS
        )
        base["weights_manifest_sha256"] = "sha256:" + "a" * 64
        post["weights_manifest_sha256"] = "sha256:" + "b" * 64
        for key in (
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "agent_image_digest",
            "proxy_image_digest",
            "live_parity_receipt_sha256",
        ):
            post[key] = base[key]
        bindings = {
            "arms": {"base": base, "post_sft": post},
            "matched_runtime": runtime(),
        }
        base_ref = None
    else:
        parent_path = CONFIG / "qwen38-sft-webexploitbench-tensorlake-parent-v1.json"
        bindings = {
            "checkpoint_and_serving": exact(web.CHECKPOINT_FIELDS),
            "tensorlake_qualification": exact(web.TENSORLAKE_FIELDS),
            "matched_runtime": runtime(),
        }
        base_ref = None
    value = {
        "schema": sealing.CHILD_SCHEMA,
        "study_id": "qwen-study-v1",
        "arm_id": "teacher-a-lr3e6-v1",
        "blinded_evaluation_id": f"blind-{surface.replace('_', '-')}-001",
        "surface": surface,
        "parent_protocol": reference(parent_path),
        "base_control": base_ref,
        "bindings": bindings,
        "execution": {
            "campaign_id": f"qwen-{surface.replace('_', '-')}-001",
            "private_results_root": str(private_root),
            "claim_journal": str(private_root / "claims.jsonl"),
        },
        "result_policy": copy.deepcopy(sealing.RESULT_POLICY),
        "launchable": launchable,
    }
    return seal(value)


@pytest.mark.parametrize(
    "surface", ["fleet_dev", "fleet_final", "webexploitbench_tensorlake"]
)
def test_exact_children_are_launchable_only_after_all_bindings(surface, tmp_path):
    value = child(surface, tmp_path / surface)
    assert sealing.validate_child(value, root=ROOT) is True

    held = copy.deepcopy(value)
    held["bindings"]["matched_runtime"]["serving_image_digest"] = None
    held["launchable"] = False
    held["sha256"] = digest_json({key: item for key, item in held.items() if key != "sha256"})
    assert sealing.validate_child(held, root=ROOT) is False

    held["launchable"] = True
    held["sha256"] = digest_json({key: item for key, item in held.items() if key != "sha256"})
    with pytest.raises(ValueError, match="serving_image_digest is absent"):
        sealing.validate_child(held, root=ROOT)


def test_child_rejects_wandb_or_unmatched_base_post_treatment(tmp_path):
    value = child("fleet_dev", tmp_path / "private")
    value["result_policy"]["wandb_export_allowed"] = True
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})
    with pytest.raises(ValueError, match="outside W&B"):
        sealing.validate_child(value, root=ROOT)

    value = child("fleet_dev", tmp_path / "private")
    value["bindings"]["post_sft"]["tokenizer_manifest_sha256"] = "sha256:" + "f" * 64
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})
    with pytest.raises(ValueError, match="tokenizer_manifest_sha256 differs"):
        sealing.validate_child(value, root=ROOT)


def test_launchable_child_requires_an_existing_private_result_root(tmp_path):
    private_root = tmp_path / "private"
    value = child("fleet_dev", private_root)
    private_root.rmdir()
    with pytest.raises(FileNotFoundError):
        sealing.validate_child(value, root=ROOT)

    private_root.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private directory"):
        sealing.validate_child(value, root=ROOT)


def private_completion(value: dict) -> dict:
    unsigned = {
        "schema": sealing.COMPLETION_SCHEMA,
        "child_sha256": value["sha256"],
        "blinded_evaluation_id": value["blinded_evaluation_id"],
        "terminal_classification": "valid_outcome_set",
        "terminal_evidence_sha256": "sha256:" + "c" * 64,
        "raw_result_manifest_sha256": "sha256:" + "d" * 64,
        "private_results_root": value["execution"]["private_results_root"],
        "resources_released": True,
        "wandb_exported": False,
    }
    return seal(unsigned)


def test_completion_emits_only_blinded_digests_and_never_opens_raw_files(tmp_path):
    private_root = tmp_path / "private-results"
    private_root.mkdir(mode=0o700)
    unreadable = private_root / "raw-results.bin"
    unreadable.write_bytes(b"synthetic-private-payload")
    unreadable.chmod(0)
    value = child("fleet_dev", private_root)
    completion = private_completion(value)

    blinded = sealing.blinded_completion(value, completion, root=ROOT)
    assert blinded["sealed_terminal"] is True
    assert blinded["outcome_or_validity_exposed"] is False
    assert blinded["wandb_exported"] is False
    assert "terminal_classification" not in blinded
    assert "child_sha256" not in blinded
    assert "score" not in json.dumps(blinded).lower()


def test_completion_rejects_added_outcome_fields_and_nonprivate_root(tmp_path):
    private_root = tmp_path / "private-results"
    private_root.mkdir(mode=0o700)
    value = child("fleet_dev", private_root)
    completion = private_completion(value)
    completion["score"] = 1.0
    completion["sha256"] = digest_json(
        {key: item for key, item in completion.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="outcome-bearing"):
        sealing.blinded_completion(value, completion, root=ROOT)

    completion.pop("score")
    completion["sha256"] = digest_json(
        {key: item for key, item in completion.items() if key != "sha256"}
    )
    private_root.chmod(0o755)
    with pytest.raises(ValueError, match="private directory"):
        sealing.blinded_completion(value, completion, root=ROOT)


def outcome_receipt(value: dict, completion: dict) -> dict:
    unsigned = sealing.blinded_completion(value, completion, root=ROOT)
    return seal(unsigned)


def freeze_for(receipt: dict, *, checkpoint: bool) -> dict:
    unsigned = {
        "schema": sealing.FREEZE_SCHEMA,
        "study_id": "qwen-study-v1",
        "surface": receipt["surface"],
        "parent_protocol_sha256": receipt["parent_protocol_sha256"],
        "expected_blinded_evaluation_ids": [receipt["blinded_evaluation_id"]],
        "gate": receipt["outcome_access_gate"],
        "selection_rule_sha256": "sha256:" + "e" * 64,
        "selected_checkpoint_seal_sha256": "sha256:" + "f" * 64 if checkpoint else None,
        "frozen_before_outcome_access": True,
    }
    return seal(unsigned)


@pytest.mark.parametrize(
    ("surface", "checkpoint"),
    [
        ("fleet_dev", False),
        ("fleet_final", True),
        ("webexploitbench_tensorlake", True),
    ],
)
def test_outcome_access_requires_the_surface_specific_frozen_gate(
    surface, checkpoint, tmp_path
):
    private_root = tmp_path / surface
    private_root.mkdir(mode=0o700)
    value = child(surface, private_root)
    receipt = outcome_receipt(value, private_completion(value))
    freeze = freeze_for(receipt, checkpoint=checkpoint)
    sealing.validate_outcome_access(receipt, freeze)

    freeze["frozen_before_outcome_access"] = False
    freeze["sha256"] = digest_json({key: item for key, item in freeze.items() if key != "sha256"})
    with pytest.raises(ValueError, match="not frozen"):
        sealing.validate_outcome_access(receipt, freeze)
