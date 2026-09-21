"""Build the non-authorizing prod8 step-1 reload/continuation packet.

The packet deliberately leaves live step-1 receipt and checkpoint-manifest
digests unbound.  A later successor may bind those fields only after prod8 has
finished, its native checkpoint has been sealed, and independent acceptance
has verified the reward and optimizer-update evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod8-launch-packet-v1.json"
OUTPUT = ROOT / "configs/qualification/qwen38-rl-reward-prod8-resume-qualification-v1.json"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)


def digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def stage(name: str, kind: str, output_root: str, source_step: int, target_step: int) -> dict:
    updates = target_step - source_step
    return {
        "name": name,
        "kind": kind,
        "output_root": output_root,
        "wandb_run_id": name,
        "source_optimizer_step": source_step,
        "target_optimizer_step": target_step,
        "optimizer_updates": updates,
        "resume_mode": "from_path",
        "source_checkpoint": "/mnt/sfs/jobs/chris-q38-rlreward-prod8/checkpoints/global_step_1",
        "resources": {
            "priority": "c1",
            "queue_priority": "q1",
            "nodes": 1,
            "gpus_per_node": 8,
            "total_gpus": 8,
            "image": IMAGE,
            "requeue_if_preempted": False,
            "failure_alerts": "off",
        },
        "launch_authorized": False,
    }


def build() -> dict:
    source = json.loads(SOURCE.read_bytes())
    if (
        source.get("schema") != "cyber_qwen38_skyrl_prod8_launch_packet_v1"
        or source.get("identity", {}).get("name") != "chris-q38-rlreward-prod8"
        or source.get("digests", {}).get("plan_sha256")
        != "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
        or source.get("digests", {}).get("runtime_sha256")
        != "acb1d1a1ff5aec9d8c153dfc57e52a13bb0b081fc666a4a0d0db139e8cb6831b"
        or source.get("checkpoint_policy", {}).get("current_resume_mode") != "none"
        or source.get("checkpoint_policy", {}).get("expected_step1_path")
        != "/mnt/sfs/jobs/chris-q38-rlreward-prod8/checkpoints/global_step_1"
    ):
        raise ValueError("prod8 launch packet identity changed")

    reload1 = stage(
        "chris-q38-rlreward-step1-reload-v1",
        "zero_update_step1_reload",
        "/mnt/sfs/jobs/chris-q38-rlreward-step1-reload-v1",
        1,
        1,
    )
    reload1["required_result"] = {
        "optimizer_calls": 0,
        "finite_held_out_inference": True,
        "restored": [
            "all eight model ranks",
            "all eight optimizer ranks",
            "scheduler step 1",
            "all eight rank-local RNG states",
            "trainer global step 1",
            "sampler cursor after step 1",
        ],
        "source_checkpoint_byte_identical": True,
    }

    continuation = stage(
        "chris-q38-rlreward-resume-step2-v1",
        "one_update_step1_to_step2",
        "/mnt/sfs/jobs/chris-q38-rlreward-resume-step2-v1",
        1,
        2,
    )
    continuation["science"] = {
        **source["science"],
        "steps": 2,
        "new_prompt_groups": 1,
        "new_samples": 8,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
    }
    continuation["required_result"] = {
        "new_train_batches": [2],
        "forbidden_train_batches": [1],
        "step1_episode_ids_replayed": False,
        "step1_verifier_execution_ids_replayed": False,
        "authoritative_rewards_vary": True,
        "finite_nonzero_optimizer_update": True,
        "checkpoint": "checkpoints/global_step_2",
        "source_checkpoint_byte_identical": True,
    }

    reload2 = stage(
        "chris-q38-rlreward-step2-reload-v1",
        "zero_update_step2_reload",
        "/mnt/sfs/jobs/chris-q38-rlreward-step2-reload-v1",
        2,
        2,
    )
    reload2["source_checkpoint"] = (
        "/mnt/sfs/jobs/chris-q38-rlreward-resume-step2-v1/checkpoints/global_step_2"
    )
    reload2["required_result"] = {
        "optimizer_calls": 0,
        "finite_held_out_inference": True,
        "restored": [
            "all eight model ranks",
            "all eight optimizer ranks",
            "scheduler step 2",
            "all eight rank-local RNG states",
            "trainer global step 2",
            "sampler cursor after step 2",
        ],
        "step1_and_step2_checkpoints_byte_identical": True,
    }

    packet = {
        "schema": "cyber_qwen38_skyrl_prod8_resume_qualification_v1",
        "status": "awaiting_accepted_prod8_step1",
        "launch_authorized": False,
        "source": {
            "run_name": source["identity"]["name"],
            "output_root": source["identity"]["output_root"],
            "plan_sha256": source["digests"]["plan_sha256"],
            "runtime_sha256": source["digests"]["runtime_sha256"],
            "request_sha256": source["digests"]["request_sha256"],
            "checkpoint_path": source["checkpoint_policy"]["expected_step1_path"],
            "terminal_receipt_sha256": None,
            "checkpoint_manifest_sha256": None,
            "checkpoint_inventory_sha256": None,
            "reward_variation_receipt_sha256": None,
            "finite_update_receipt_sha256": None,
        },
        "binding_rule": (
            "A new immutable successor may replace the five null source evidence fields only "
            "after prod8 terminal acceptance. This template is never edited in place."
        ),
        "shared_invariants": {
            "model_data_recipe_unchanged": True,
            "source_checkpoint_read_only": True,
            "distinct_run_output_and_wandb_id_per_stage": True,
            "root_job_or_rayjob_annotation": {"fleet.ai/failure-alerts": "off"},
            "fresh_jobs_kubernetes_sfs_wandb_absence_required": True,
            "server_preview_required_before_each_create": True,
            "uid_bound_observer_and_terminal_release_required": True,
        },
        "implementation_gate": {
            "ready": False,
            "required_runtime": "separately versioned native SkyRL resume validator",
            "historical_prod8_runtime_must_not_change": True,
            "required_native_mode": "trainer.resume_mode=from_path",
            "required_proofs": [
                "strict all-rank model optimizer scheduler and RNG reload",
                "strict trainer-step and sampler-cursor reload without warning fallback",
                "optimizer call guard for both zero-update reload stages",
                "one-and-only-one optimizer call for the step1-to-step2 continuation",
                "source inventory and digests unchanged before and after every stage",
                "step-2 checkpoint sealed before its independent reload",
            ],
        },
        "stages": [reload1, continuation, reload2],
        "final_acceptance": {
            "resume_qualified": False,
            "requires_all_three_stages": True,
            "requires_all_owned_gpus_released": True,
            "broad_training_stays_blocked": True,
        },
    }
    packet["sha256"] = digest(packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build()
    if args.check:
        if json.loads(OUTPUT.read_bytes()) != expected:
            raise SystemExit("prod8 resume qualification packet drift")
        return
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
