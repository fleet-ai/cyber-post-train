#!/usr/bin/env python3
"""Seal the non-authorizing prod8 SkyRL launch packet.

The packet recompiles the scientific plan from committed, sanitized inputs and
binds the current read-only server previews.  It deliberately cannot authorize
or create a workload: private-data staging, the exact-image CPU preflight,
fresh absence checks, and the cleanup observer remain live gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, digest
from scripts import audit_qwen38_skyrl_launch_readiness as readiness
from training import skyrl_reward_rayjob as direct

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod8-launch-packet-v1.json"
PREVIEW_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-21-skyrl-prod8-read-only-previews-v1.json"
)
PRIVATE_SOURCE_CANDIDATE = "/private/tmp/q38-prod8-data.PxTZax/prod8"
EXPECTED = {
    "plan_sha256": "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de",
    "request_sha256": "7c2df31feceb5c741cb16463b554b91d00b11c6bf50ec85b3203706823c2fb44",
    "runtime_sha256": "acb1d1a1ff5aec9d8c153dfc57e52a13bb0b081fc666a4a0d0db139e8cb6831b",
    "manifest_sha256": "sha256:5561f1a349abbe1a580dd5763368c1e6c1524861c39d744f7dad4f25d9950dd3",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def _validate_preview_evidence(value: dict, plan: dict, request: dict) -> None:
    if (
        value.get("schema") != "cyber_skyrl_prod8_read_only_preview_evidence_v1"
        or value != _seal({key: item for key, item in value.items() if key != "sha256"})
        or value.get("external_mutations") != 0
        or value.get("private_payloads_read") is not False
        or value.get("authorization_reusable") is not False
    ):
        raise ValueError("prod8 read-only preview evidence changed")
    packet = direct._validate_seal(value.get("direct_packet"), direct.PACKET_SCHEMA)
    if (
        packet.get("plan_sha256") != digest(plan)
        or packet.get("request_sha256") != digest(request)
        or packet.get("name") != direct.RUN_NAME
        or packet.get("submitted") is not False
    ):
        raise ValueError("prod8 read-only direct packet changed")
    contexts = {direct.DEV_CONTEXT, direct.PROD_CONTEXT}
    if set(value.get("server_previews", {})) != contexts:
        raise ValueError("prod8 direct server preview contexts changed")
    for context, proof in value["server_previews"].items():
        direct._validate_seal(proof, direct.PREVIEW_SCHEMA)
        if (
            proof.get("context") != context
            or proof.get("plan_sha256") != digest(plan)
            or proof.get("request_sha256") != digest(request)
            or proof.get("failure_alerts") != "off"
            or proof.get("submitted") is not False
        ):
            raise ValueError("prod8 direct server preview changed")
    if set(value.get("cpu_preflight_previews", {})) != contexts:
        raise ValueError("prod8 CPU preflight preview contexts changed")
    for context, proof in value["cpu_preflight_previews"].items():
        direct._validate_seal(proof, direct.CPU_PREVIEW_SCHEMA)
        if (
            proof.get("context") != context
            or proof.get("purpose") != "preflight"
            or proof.get("gpus") != 0
            or proof.get("failure_alerts") != "off"
            or proof.get("submitted") is not False
        ):
            raise ValueError("prod8 CPU preflight server preview changed")


def build() -> dict:
    run = readiness.load(readiness.CANARY_RUN)
    manifest = readiness.prod8_metadata(run, {})
    plan, request = readiness.compile_prod8(run, manifest)
    observed = {
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "runtime_sha256": plan["runtime_sha256"],
        "manifest_sha256": manifest["sha256"],
    }
    if observed != EXPECTED:
        raise ValueError("prod8 scientific packet changed")
    preview_evidence = _load(PREVIEW_EVIDENCE)
    _validate_preview_evidence(preview_evidence, plan, request)
    preflight = direct.preflight_job_manifest(plan)
    pod = preflight["spec"]["template"]["spec"]
    if (
        preflight["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or pod.get("priorityClassName") != "c1"
        or "nvidia.com/gpu" in json.dumps(preflight, sort_keys=True)
    ):
        raise ValueError("prod8 CPU preflight resource/alert contract changed")
    args = plan["arguments"]
    if (
        args["checkpoint_interval"] != 1
        or args["keep_checkpoints"] != 2
        or plan["native_overrides"].get("trainer.resume_mode") != "none"
    ):
        raise ValueError("prod8 checkpoint/resume contract changed")
    return _seal(
        {
            "schema": "cyber_qwen38_skyrl_prod8_launch_packet_v1",
            "status": "prepared_external_stage_and_preflight_pending",
            "launch_authorized": False,
            "external_mutations_by_builder": 0,
            "identity": {
                "name": plan["run_name"],
                "output_root": plan["output_root"],
                "wandb_run_id": args["wandb_run_id"],
                "namespace": direct.NAMESPACE,
                "cluster_target": plan["execution"]["cluster_target"],
            },
            "digests": observed,
            "inputs": {
                "run_config": readiness.source(readiness.CANARY_RUN),
                "data_config": readiness.source(readiness.CANARY_DATA),
                "sanitized_manifest": readiness.source(readiness.CANARY_MANIFEST),
                "qualification": readiness.source(readiness.CANARY_QUALIFICATION),
                "model_lock": readiness.source(readiness.MODEL_LOCK),
                "model_weights": readiness.source(readiness.MODEL_WEIGHTS),
                "read_only_preview_evidence": readiness.source(PREVIEW_EVIDENCE),
            },
            "resources": {
                "priority": request["priority_class"],
                "queue_priority": "q1",
                "nodes": request["workers"],
                "gpus_per_node": request["gpus_per_worker"],
                "total_gpus": request["workers"] * request["gpus_per_worker"],
                "requeue_if_preempted": request["requeueIfPreempted"],
                "failure_alerts": request["failureAlerts"],
                "image": request["image"],
            },
            "science": {
                key: args[key]
                for key in (
                    "steps",
                    "groups",
                    "samples_per_prompt",
                    "lr",
                    "eval_interval",
                    "seed",
                    "context_tokens",
                    "response_tokens",
                    "tokens_per_turn",
                    "generation_chunk_tokens",
                    "compaction_trigger_tokens",
                    "compaction_summary_tokens",
                    "max_turns",
                )
            },
            "checkpoint_policy": {
                "checkpoint_interval_optimizer_steps": args["checkpoint_interval"],
                "keep_latest": args["keep_checkpoints"],
                "expected_step1_path": plan["output_root"] + "/checkpoints/global_step_1",
                "expected_native_state": [
                    "data.pt sampler cursor",
                    "trainer_state.pt global step",
                    "policy/fsdp_config.json",
                    "eight model shards",
                    "eight optimizer shards",
                    "eight extra-state shards containing rank-local continuation state",
                    "policy/huggingface/config.json and tokenizer sidecars",
                ],
                "current_resume_mode": "none",
                "resume_qualified": False,
                "observed_checkpoint_bytes": None,
                "observed_checkpoint_write_seconds": None,
            },
            "local_cpu_preflight_shape": {
                "name": direct.PREFLIGHT_NAME,
                "manifest_sha256": digest(preflight),
                "gpus": 0,
                "priority": "c1",
                "failure_alerts": "off",
                "server_previewed_in_dev_and_prod": True,
                "executed": False,
            },
            "read_only_preview_observation": {
                "observed_at": preview_evidence["observed_at"],
                "source_commit": preview_evidence["source_commit"],
                "direct_manifest_sha256": preview_evidence["direct_packet"]["manifest_sha256"],
                "dev_direct_preview_sha256": preview_evidence["server_previews"][
                    direct.DEV_CONTEXT
                ]["sha256"],
                "prod_direct_preview_sha256": preview_evidence["server_previews"][
                    direct.PROD_CONTEXT
                ]["sha256"],
                "authorization_reusable": False,
            },
            "private_data_stage": {
                "operator_candidate_source": PRIVATE_SOURCE_CANDIDATE,
                "candidate_source_is_ephemeral_and_must_be_rehashed": True,
                "destination": args["data_manifest"].removesuffix("/manifest.json"),
                "create_once": True,
                "staged": False,
            },
            "open_gates_in_order": [
                (
                    "rehash the exact private source and prove the destination, stage Job, and "
                    "archive are absent"
                ),
                "server-preview the zero-GPU create-once stage Job in dev and prod",
                (
                    "create the stage Job once with root failure-alert opt-out and monitor "
                    "terminal release"
                ),
                "independently validate the staged receipt, ownership, inventory, and every digest",
                "freshly server-preview and create the exact-image zero-GPU CPU preflight once",
                (
                    "accept only a fresh preflight receipt with both output-limit proofs "
                    "literally true"
                ),
                "repeat Jobs, Kubernetes, SFS, W&B, and output absence checks",
                "repeat direct RayJob server previews and arm the exact UID-bound cleanup observer",
                "authorize exactly one GPU RayJob create; never retry an ambiguous create",
            ],
            "resume_qualification_after_step1": {
                "successor_identity": "chris-q38-rlreward-resume-step2-v1",
                "source_checkpoint": plan["output_root"] + "/checkpoints/global_step_1",
                "required_implementation": (
                    "add a separately sealed native resume mode; current code hard-codes none"
                ),
                "must_restore": [
                    "all-rank model shards",
                    "all-rank optimizer and scheduler state",
                    "all-rank RNG/extra state",
                    "trainer global step",
                    "sampler cursor",
                ],
                "must_not_replay": [
                    "step-1 rollout episodes",
                    "step-1 verifier execution IDs",
                    "step-1 optimizer update",
                ],
                "acceptance": [
                    "source checkpoint inventory and digests remain unchanged",
                    "trainer resumes at global step 1 and executes exactly one new group of eight",
                    (
                        "authoritative rewards vary and exactly one finite nonzero update reaches "
                        "step 2"
                    ),
                    "a complete step-2 checkpoint is sealed",
                    (
                        "an independent zero-update GPU reload proves model, optimizer, scheduler, "
                        "RNG, sampler, and finite inference"
                    ),
                    "all eight GPUs and owned descendants are released",
                ],
                "qualified": False,
            },
            "broad_followup": {
                "first_arm": "configs/runs/qwen38-skyrl-production-a1-v1.json",
                "launch_safe": False,
                "blockers": [
                    "prod8 has not completed and been accepted",
                    "native step1-to-step2 resume is not qualified",
                    "the arm still uses the retired 98,304-token non-compacting horizon",
                    "the arm must inherit the prod8 output-limit reward contract",
                ],
            },
        }
    )


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = build()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(raw(value))
    elif not OUTPUT.exists() or OUTPUT.read_bytes() != raw(value):
        raise SystemExit("prod8 launch packet is stale")
    print(
        json.dumps(
            {
                "packet": str(OUTPUT.relative_to(ROOT)),
                "sha256": value["sha256"],
                "launch_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
