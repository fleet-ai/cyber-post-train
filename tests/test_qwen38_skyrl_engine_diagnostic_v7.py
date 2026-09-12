"""Replacement-image and fresh-identity gates for the SkyRL dev7 smoke."""
# ruff: noqa: F811

import hashlib
import json
from pathlib import Path

from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

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
OLD_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
NEW_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "9b6f43938f9b28aaff7ba91edd59be3d18b01f9a22078ba5475cdac5e6bfcca6"
)


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replacement_image_is_bound_to_its_cpu_qualification(skyrl_prepared):
    receipt = load(QUALIFICATION)
    canonical = json.dumps(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["requested_image"] == NEW_IMAGE
    assert receipt["status"] == "qualified"
    assert receipt["classification"] == "operational_gate"
    assert receipt["expected_source_commit"] == ("de9e6b7cf087d12c5ca371c9ec916a54757d369f")

    plan = skyrl_prepared.plan
    assert skyrl_training.IMAGE == plan["execution"]["image"] == NEW_IMAGE
    assert plan["execution"]["image_cpu_qualification"] == (
        skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION
    )
    binding = skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION
    assert binding["receipt_sha256"] == receipt["receipt_sha256"]
    assert binding["source_commit"] == receipt["expected_source_commit"]
    assert binding["evidence_path"] == str(QUALIFICATION.relative_to(ROOT))


def test_old_pair_remains_disqualified_while_replacement_is_only_cpu_qualified(
    skyrl_prepared,
):
    old = {
        **skyrl_prepared.plan,
        "execution": {**skyrl_prepared.plan["execution"], "image": OLD_IMAGE},
    }
    assert ("Qwen/Qwen3.8-27B", OLD_IMAGE) in skyrl_training._ENGINE_START_DISQUALIFIED
    try:
        skyrl_training._require_engine_start_qualified_image(old)
    except ValueError as error:
        assert "dev5/dev6" in str(error)
    else:
        raise AssertionError("historically disqualified image was accepted")

    skyrl_training._require_engine_start_qualified_image(skyrl_prepared.plan)
    assert skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION["status"] == "qualified"


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


def test_dev7_request_is_exact_zero_work_tp4x2_and_clean_terminal(skyrl_prepared):
    plan = skyrl_prepared.plan
    request = skyrl_training.engine_diagnostic_request(plan)

    assert request["image"] == NEW_IMAGE
    assert (request["workers"], request["gpus_per_worker"]) == (2, 4)
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert "WANDB" not in json.dumps(request["env"])
    assert digest(request) != digest(skyrl_training.job_request(plan))
    assert plan["arguments"]["steps"] == 2  # inert in diagnostic mode
    assert plan["arguments"]["engine_start_timeout_seconds"] == 1800
    assert plan["arguments"]["engine_cleanup_timeout_seconds"] == 300

    # Runtime tests exercise both success and sanitized-rejection paths.  This
    # immutable request can qualify engine startup only; it cannot read tasks,
    # initialize W&B, call a verifier, optimize, or create a checkpoint.
    assert request["title"].endswith("SkyRL engine-start diagnostic")
    assert request["run_dir"] == plan["output_root"]


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
