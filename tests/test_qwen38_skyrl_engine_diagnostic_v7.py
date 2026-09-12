"""Frozen preparation and terminal evidence for the retired SkyRL dev7 smoke."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_V6 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v6.json"
)
DATA_V7 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v7.json"
)
RUN_V6 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v6.json"
RUN_V7 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v7.json"
QUALIFICATION = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-replacement-image-cpu-qualification-v1.json"
)
PREPARATION = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-engine-diagnostic-dev7-offcluster-preparation-v1.json"
)
TERMINAL = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-skyrl-engine-diagnostic-dev7-terminal-v1.json"
)
LEGACY_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
DEV7_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "9b6f43938f9b28aaff7ba91edd59be3d18b01f9a22078ba5475cdac5e6bfcca6"
)
CORRECTED_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "e48827529b1cf5fafa153b2aed1b774c2eec86905baf5ccb62b36300533e252b"
)


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dev7_image_remains_bound_to_its_historical_cpu_qualification():
    receipt = load(QUALIFICATION)
    canonical = json.dumps(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["requested_image"] == DEV7_IMAGE
    assert receipt["status"] == "qualified"
    assert receipt["classification"] == "operational_gate"
    assert receipt["expected_source_commit"] == ("de9e6b7cf087d12c5ca371c9ec916a54757d369f")

    assert skyrl_training.IMAGE == CORRECTED_IMAGE
    assert receipt["requested_image"] != skyrl_training.IMAGE


def test_dev7_and_its_predecessor_are_both_engine_start_disqualified():
    for image in (LEGACY_IMAGE, DEV7_IMAGE):
        identity = ("Qwen/Qwen3.8-27B", image)
        assert identity in skyrl_training._ENGINE_START_DISQUALIFIED
        plan = {"model": {"repo": identity[0]}, "execution": {"image": image}}
        try:
            skyrl_training._require_engine_start_qualified_image(plan)
        except ValueError as error:
            assert "terminal dev evidence" in str(error)
        else:
            raise AssertionError("historically disqualified image was accepted")


def test_dev7_has_fresh_create_once_identities_and_unchanged_science():
    data_v6, data_v7 = load(DATA_V6), load(DATA_V7)
    run_v6, run_v7 = load(RUN_V6), load(RUN_V7)

    assert data_v7["name"] == run_v7["name"] == run_v7["wandb"]["run_id"]
    assert data_v7["name"] == "chris-q38-rldiag-dev7"
    assert run_v7["output_root"] == "/mnt/sfs/jobs/chris-q38-rldiag-dev7"
    assert data_v7["output"] == run_v7["data"]["root"]
    assert run_v7["data"]["manifest"] == data_v7["output"] + "/manifest.json"
    assert data_v7["output"] != data_v6["output"]
    assert run_v7["output_root"] != run_v6["output_root"]
    assert run_v7["wandb"]["run_id"] != run_v6["wandb"]["run_id"]

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v7[key] == data_v6[key]
    assert data_v7["limits"] == data_v6["limits"]
    for key in ("backend", "model", "recipe", "cluster"):
        assert run_v7[key] == run_v6[key]


def test_dev7_terminal_evidence_is_nonqualifying_zero_work_and_released():
    value = load(TERMINAL)
    assert value["classification"] == "infrastructure_invalid_relay_transport"
    assert value["bindings"]["image"] == DEV7_IMAGE
    assert value["receipt"]["status"] == "engine_start_rejected"
    assert value["receipt"]["sanitized_error_class"] == "UnserializableException"
    assert value["receipt"]["engine_start_qualified"] is False
    assert value["receipt"]["training_qualified"] is False
    assert value["receipt"]["cleanup_proven"] is True
    assert value["kubernetes"]["gpu_release_proven"] is True
    assert all(
        value["receipt"][key] == 0
        for key in (
            "task_rows_read",
            "rollouts",
            "verifier_calls",
            "optimizer_steps",
            "checkpoints_created",
        )
    )
    assert value["decision"]["production_rl_open"] is False


def test_dev7_offcluster_evidence_is_self_digesting_and_truthful():
    value = load(PREPARATION)
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    successor = value["successor"]
    assert successor["data_config_sha256"] == file_sha256(DATA_V7)
    assert successor["run_config_sha256"] == file_sha256(RUN_V7)
    # Historical preparation remains bound to the exact runtime it submitted;
    # later fail-closed guards must not rewrite that immutable provenance.
    assert successor["skyrl_training_source_sha256"] == (
        "7ac081a3f595ba4a9cada3dea0b151aa155b3b728286229406ece95004763dec"
    )
    assert successor["plan_sha256"] is successor["request_sha256"] is None
    assert successor["submitted"] is successor["gpu_engine_start_qualified"] is False
    assert (
        value["replacement_image"]["cpu_qualification"]["receipt_sha256"]
        == (load(QUALIFICATION)["receipt_sha256"])
    )
    assert value["offcluster_package"]["create_once_prepared_directory_absent"] is True
    assert not any(value["audit_boundaries"].values())
