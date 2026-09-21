"""Regression contract for the non-authorizing prod8 launch packet."""

from __future__ import annotations

import json

from cyber_post_train.jobs import digest
from scripts import prepare_qwen38_skyrl_prod8_launch_packet as packet


def test_packet_is_current_self_sealed_and_never_authorizes_launch() -> None:
    value = packet.build()
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert packet.OUTPUT.read_bytes() == packet.raw(value)
    assert value["launch_authorized"] is False
    assert value["external_mutations_by_builder"] == 0
    assert value["read_only_preview_observation"]["authorization_reusable"] is False


def test_packet_binds_prod8_science_resources_and_alert_opt_out() -> None:
    value = packet.build()
    assert value["digests"] == packet.EXPECTED
    assert value["resources"] == {
        "priority": "c1",
        "queue_priority": "q1",
        "nodes": 1,
        "gpus_per_node": 8,
        "total_gpus": 8,
        "requeue_if_preempted": False,
        "failure_alerts": False,
        "image": (
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
            "sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
        ),
    }
    assert value["science"] == {
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 1,
        "seed": 42,
        "context_tokens": 262144,
        "response_tokens": 4194304,
        "tokens_per_turn": 32768,
        "generation_chunk_tokens": 4096,
        "compaction_trigger_tokens": 163840,
        "compaction_summary_tokens": 8192,
        "max_turns": 1200,
    }
    assert value["local_cpu_preflight_shape"] == {
        "name": "chris-q38-prod8-preflight-v1",
        "manifest_sha256": "995a8c0eb5795920dc554fdcf9f332484a137e7f3d9ca2fa8b3efcdcdb820bd5",
        "gpus": 0,
        "priority": "c1",
        "failure_alerts": "off",
        "server_previewed_in_dev_and_prod": True,
        "executed": False,
    }


def test_checkpoint_and_followup_resume_remain_truthfully_unqualified() -> None:
    value = packet.build()
    policy = value["checkpoint_policy"]
    resume = value["resume_qualification_after_step1"]
    assert policy["checkpoint_interval_optimizer_steps"] == 1
    assert policy["keep_latest"] == 2
    assert policy["current_resume_mode"] == "none"
    assert policy["resume_qualified"] is False
    assert policy["observed_checkpoint_bytes"] is None
    assert policy["observed_checkpoint_write_seconds"] is None
    assert resume["qualified"] is False
    assert "step-1 rollout episodes" in resume["must_not_replay"]
    assert value["broad_followup"]["launch_safe"] is False


def test_read_only_preview_evidence_is_self_sealed_and_contains_no_mutation() -> None:
    evidence = json.loads(packet.PREVIEW_EVIDENCE.read_bytes())
    assert evidence["sha256"] == "sha256:" + digest(
        {key: item for key, item in evidence.items() if key != "sha256"}
    )
    assert evidence["external_mutations"] == 0
    assert evidence["authorization_reusable"] is False
    assert {proof["failure_alerts"] for proof in evidence["server_previews"].values()} == {"off"}
    assert {proof["failure_alerts"] for proof in evidence["cpu_preflight_previews"].values()} == {
        "off"
    }
