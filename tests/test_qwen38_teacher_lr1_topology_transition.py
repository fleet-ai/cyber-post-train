"""Frozen cancellation and successor-admission contract for the LR1 topology switch."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-dev-fallback-v1.json"
PREPARATION = ROOT / (
    "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-fallback-preparation-v1.json"
)
EVIDENCE = ROOT / ("docs/evidence/qwen38-study/2026-09-12-lr1-topology-transition-dev-v1.json")


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_transition_evidence_is_self_digesting_and_source_bound() -> None:
    evidence = read(EVIDENCE)
    assert evidence["sha256"] == digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )

    source = evidence["source"]
    assert source["config_commit"].startswith(source["config_commit_abbrev"])
    assert source["config_commit_abbrev"] == "97d761d8"
    assert source["config_commit"] == "97d761d8f5fcb3fa59ba2bbd036e11e911c4a4f5"
    assert source["config_file_sha256"] == file_sha256(CONFIG)

    preparation = read(PREPARATION)
    binding = source["preparation_evidence"]
    assert binding["file_sha256"] == file_sha256(PREPARATION)
    assert binding["embedded_sha256"] == preparation["sha256"]
    assert preparation["fallback"]["config_path"] == source["config_path"]
    assert preparation["fallback"]["config_file_sha256"] == file_sha256(CONFIG)


def test_one_by_eight_predecessor_was_unadmitted_and_reconciled_absent() -> None:
    predecessor = read(EVIDENCE)["predecessor"]
    assert {
        key: predecessor[key] for key in ("api_name", "rayjob_name", "rayjob_uid", "workload_uid")
    } == {
        "api_name": "chris-q38-ta8-lr1-dev-v2-e9d44c23",
        "rayjob_name": "chris-q38-ta8-lr1-dev-v2-e9d44c23",
        "rayjob_uid": "d7aeaa33-7c0f-416f-b231-cdb1f939e654",
        "workload_uid": "a9fdc7bd-1456-48cc-aa74-cc2d882387ae",
    }
    topology = predecessor["requested_topology"]
    queue = predecessor["queue_observation"]
    assert topology == {"workers": 1, "gpus_per_worker": 8, "total_gpus": 8}
    assert queue["rayjob_suspended"] is True
    assert queue["workload_admitted"] is False
    assert queue["free_gpus_by_node"] == [6, 6]
    assert max(queue["free_gpus_by_node"]) < topology["gpus_per_worker"]
    assert queue["single_node_eight_gpu_fit"] is False
    assert predecessor["zero_execution"] == {
        "rayclusters_created": 0,
        "pods_created": 0,
        "optimizer_steps": 0,
        "gpus_allocated": 0,
    }

    reconciliation = predecessor["cancellation_and_reconciliation"]
    assert reconciliation["sequence"] == [
        "one_jobs_api_delete_returned_204",
        "jobs_api_read_returned_404",
        "uid_bound_kubernetes_objects_and_descendants_confirmed_absent",
    ]
    assert reconciliation["jobs_api_delete_calls"] == 1
    assert reconciliation["delete_http_status"] == 204
    assert reconciliation["delete_retries"] == 0
    assert reconciliation["post_delete_api_http_status"] == 404
    absence = reconciliation["kubernetes_absence"]
    assert absence["rayjob"] == {
        "name": predecessor["rayjob_name"],
        "uid": predecessor["rayjob_uid"],
        "absent": True,
    }
    assert absence["workload"] == {
        "uid": predecessor["workload_uid"],
        "absent": True,
    }
    assert absence["raycluster_descendants"] == absence["pod_descendants"] == 0
    assert reconciliation["gpus_held_after_reconciliation"] == 0
    assert reconciliation["complete_before_successor_post"] is True


def test_exactly_one_two_by_four_successor_was_admitted_and_started() -> None:
    evidence = read(EVIDENCE)
    successor = evidence["successor"]
    config = read(CONFIG)
    assert successor["config_name"] == config["name"]
    assert {
        key: successor[key]
        for key in (
            "api_name",
            "rayjob_name",
            "rayjob_uid",
            "workload_name",
            "workload_uid",
            "raycluster_name",
            "raycluster_uid",
            "head_pod_uid",
            "worker_pod_uid",
        )
    } == {
        "api_name": "chris-q38-ta8-lr1-2x4-dev-v1-0a67736f",
        "rayjob_name": "chris-q38-ta8-lr1-2x4-dev-v1-0a67736f",
        "rayjob_uid": "70161760-16ae-4f8e-8809-e6a024a16188",
        "workload_name": "rayjob-chris-q38-ta8-lr1-2x4-dev-v1-0a67736f-c5802",
        "workload_uid": "f2ee4249-2554-4433-a6c3-53140433f6b8",
        "raycluster_name": "chris-q38-ta8-lr1-2x4-dev-v1-0a67736f-6jdc6",
        "raycluster_uid": "08369b1f-956b-46fb-bf5a-aed9a450c180",
        "head_pod_uid": "524ab333-11d9-4dc1-9643-b70595de0d20",
        "worker_pod_uid": "56e3a6b5-d2ad-45c8-84be-30a4f77a9880",
    }
    assert successor["pre_submit_identity_check"] == {
        "wandb_run_id": config["wandb"]["run_id"],
        "wandb_run_id_absent": True,
        "observed_before_post": True,
    }
    assert successor["submission"] == {
        "jobs_api_post_calls": 1,
        "posted_at": "2026-09-12T13:52:18Z",
        "post_retries": 0,
    }

    topology = successor["admission"]["requested_topology"]
    assert topology == {"workers": 2, "gpus_per_worker": 4, "total_gpus": 8}
    assert topology["workers"] == config["recipe"]["nodes"]
    assert topology["gpus_per_worker"] == config["recipe"]["gpus_per_node"]
    assert topology["total_gpus"] == evidence["predecessor"]["requested_topology"]["total_gpus"]

    admission = successor["admission"]
    assert admission["workload_admitted"] is True
    assert admission["distinct_nodes"] == topology["workers"] == 2
    assert admission["gpus_per_node"] == topology["gpus_per_worker"] == 4
    assert admission["rayjob_status"] == "RUNNING"
    assert admission["ready_pods"] == 2
    assert admission["running_and_ready_by"] == "2026-09-12T13:52:49Z"
    assert admission["image"] == (
        "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train"
        "@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
    )
    assert admission["head_pod_restarts"] == 0
    assert admission["worker_pod_restarts"] == 0
    assert admission["total_restarts"] == 0

    prepared = successor["prepared_artifacts"]
    assert prepared == {
        "directory": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/lr-dev-v2-qualified-inputs-v1/"
            "lr1-2x4-fallback-v1"
        ),
        "plan_sha256": "3b554d48bb61eeac041415c951f1bc327f70288aa8e2590246bd5eecb2f04e58",
        "request_sha256": "698dd2804479ad58c19430688b061cd571a64e3324ee2f3eef2cf1c195f79ebd",
        "preflight_sha256": "da57e6b1ac279d5d7aaf07d07a5be1bbaf3328732556278ee46a86189c908924",
        "preview_manifest_sha256": (
            "cfb644a942d0664a4302f9a91c0a9116cd36e6e4326337d5aef9ad4ad72642aa"
        ),
    }
    started = successor["started_receipt"]
    assert prepared["directory"].endswith("/lr1-2x4-fallback-v1")
    assert started["present"] is started["bound_to_prepared_plan"] is True
    assert started["plan_sha256"] == prepared["plan_sha256"]


def test_transition_has_one_live_producer_and_no_terminal_claim() -> None:
    evidence = read(EVIDENCE)
    transition = evidence["transition_invariants"]
    assert transition == {
        "predecessor_absent_before_successor_post": True,
        "predecessor_and_successor_world_size_equal": True,
        "overlapping_gpu_allocation": False,
        "successor_posts": 1,
        "live_lr1_producers_after_transition": 1,
        "operational_gate_only": True,
    }
    assert (
        transition["successor_posts"] == evidence["successor"]["submission"]["jobs_api_post_calls"]
    )
    assert evidence["successor"]["terminal"] == {
        "outcome_known": False,
        "outcome": None,
        "model_or_scientific_outcome_claimed": False,
    }
    assert not any(
        evidence["evidence_authoring_boundaries"][key]
        for key in (
            "jobs_api_calls",
            "kubernetes_calls",
            "cluster_mutations",
            "submissions",
            "cancellations",
            "operator_files_modified",
            "credentials_read_or_persisted",
        )
    )
