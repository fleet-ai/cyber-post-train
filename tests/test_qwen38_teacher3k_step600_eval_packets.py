"""No-launch matched evaluation packets for the sealed Teacher3K step 600."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
WBE = EVAL / "qwen38-teacher3k-32k-step600-opencode-wbe-preparation-v1.json"
FLEET = EVAL / "qwen38-teacher3k-32k-step600-fleet-dev17-seed43-preparation-v1.json"
FLEET_BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
FLEET_PROTOCOL = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
SOURCE_CONFIG = ROOT / "configs/runs/qwen38-teacher3k-32k-full-b8-lr3e6-v3.json"
CORPUS = ROOT / "configs/data/qwen38-teacher3k-32k-v1.manifest.json"
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
MODEL_WEIGHTS = ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"
DEFERRED_SCORE = ROOT / "evals/webexploitbench/tensorlake/deferred_score.py"

CANONICAL_CHECKPOINT = "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoints/global_step_600"
REJECTED_CHECKPOINT = "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoints/step-600"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def assert_self_digest(value: dict) -> None:
    assert value["sha256"] == digest({key: item for key, item in value.items() if key != "sha256"})


def assert_canonical_checkpoint(packet: dict) -> None:
    checkpoint = packet["candidate_checkpoint"]
    assert checkpoint["checkpoint_path"] == CANONICAL_CHECKPOINT
    assert checkpoint["checkpoint_receipt"]["path"].endswith(
        "/checkpoint_receipts/step-000600.json"
    )
    assert checkpoint["seal_manifest"]["path"].endswith("/checkpoint-seals-v1/step-600.json")
    assert REJECTED_CHECKPOINT not in json.dumps(packet, sort_keys=True)


def test_packets_are_self_digesting_review_only_and_create_nothing() -> None:
    wbe, fleet = read(WBE), read(FLEET)
    for packet in (wbe, fleet):
        assert_self_digest(packet)
        assert packet["status"] == "blocked_not_launchable"
        assert packet["launchable"] is False
        assert packet["scientific_boundary"]["evaluation_only"] is True
        assert packet["scientific_boundary"]["capability_claimed"] is False

    assert wbe["operation"] == {
        "provider_api_calls": 0,
        "model_requests": 0,
        "judge_requests": 0,
        "benchmark_attempts": 0,
        "scoring_attempts": 0,
        "serving_mutations": 0,
    }
    assert fleet["operation"] == {
        "jobs_submitted": 0,
        "config_maps_created": 0,
        "databases_created": 0,
        "outputs_created": 0,
        "routes_resumed_or_created": 0,
        "evaluation_sessions_created": 0,
    }


def test_both_packets_bind_the_same_exact_sealed_checkpoint() -> None:
    wbe, fleet = read(WBE), read(FLEET)
    assert wbe["candidate_checkpoint"] == fleet["candidate_checkpoint"]
    for packet in (wbe, fleet):
        assert_canonical_checkpoint(packet)
        checkpoint = packet["candidate_checkpoint"]
        assert checkpoint["optimizer_step"] == 600
        assert checkpoint["world_size"] == 8
        assert checkpoint["file_count"] == 33
        assert checkpoint["total_bytes"] == 324627486795
        assert checkpoint["supervised_tokens_at_step"] == 18382331
        assert checkpoint["gpu_reload_verified"] is False
        assert checkpoint["checkpoint_receipt"] == {
            "path": ("/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoint_receipts/step-000600.json"),
            "file_sha256": (
                "sha256:7d7ddbef66228fe2d9fe483605d5c7e345b04956b153bf5c5948cc7e7b62a0dd"
            ),
            "receipt_sha256": (
                "sha256:af655527bcf21d441723f38d4dc0bb523fee30df47b03ff0206ddf0046ffab26"
            ),
        }
        assert checkpoint["seal_manifest"]["file_sha256"] == (
            "sha256:2ac52a051db7e7c71813f387360fe0b69cee23d1a2b86f1248a1189cc48e1e6a"
        )
        assert checkpoint["seal_manifest"]["receipt_sha256"] == (
            "sha256:6ba2ecc2e12607dbd41565ae33452abb542d13528469691136c04194f304a2ac"
        )


def test_wrong_noncanonical_checkpoint_path_is_rejected() -> None:
    packet = read(WBE)
    packet["candidate_checkpoint"]["checkpoint_path"] = REJECTED_CHECKPOINT
    with pytest.raises(AssertionError):
        assert_canonical_checkpoint(packet)


def test_checkpoint_provenance_binds_the_reviewed_source_files() -> None:
    checkpoint = read(WBE)["candidate_checkpoint"]
    provenance = checkpoint["source_provenance"]
    assert provenance["source_config_path"] == str(SOURCE_CONFIG.relative_to(ROOT))
    assert provenance["source_config_file_sha256"] == file_sha256(SOURCE_CONFIG)
    assert provenance["source_config_introduced_commit"] == (
        "6254334cd85ebb02a6c6515be6f13fb63142f3bc"
    )
    assert provenance["source_plan_file_sha256"] == (
        "sha256:8488f03a65e3d158a46697eefda2751736681e3a503c26ee9c39c63fb130679c"
    )
    assert provenance["runtime_sha256"] == (
        "sha256:8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17"
    )
    assert provenance["trainer_image_digest"] == (
        "sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
    )

    base = checkpoint["base_model"]
    assert base["lock_file_sha256"] == file_sha256(MODEL_LOCK)
    assert base["weights_file_sha256"] == file_sha256(MODEL_WEIGHTS)
    assert base["weights_manifest_sha256"] == (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    )
    assert checkpoint["corpus"]["manifest_file_sha256"] == file_sha256(CORPUS)
    assert checkpoint["corpus"]["manifest_sha256"] == read(CORPUS)["sha256"]


def test_wbe_packet_matches_the_exact_recovering_q8_score_free_protocol() -> None:
    packet = read(WBE)
    base = packet["recovering_base_control"]
    assert base["campaign_id"] == "q38-base-oc-wbe-min-q8-v1"
    assert base["task_index"] == 0
    assert base["target_label"] == "pb-comfyui"
    assert base["task_count"] == base["pass_k"] == 1
    assert base["plan_file_sha256"] == (
        "sha256:3931bcbb0813ad659c20bb2c98781fd023127ac91657268cfad93849883c4d16"
    )
    assert base["plan_sha256"] == (
        "sha256:b21d2caa959c530fa02c4f3d6d0ae2202ea5bd9849fd53e42f9ef53298a74f7d"
    )
    assert base["protocol_sha256"] == (
        "sha256:a23849b89d7dc3970563ace623f5171e628f25b723006e68fce9540b662db7fa"
    )
    assert base["comparison_protocol_sha256"] == (
        "sha256:2bf28f875fad7fa96c6f0a5b43f1daa05ba512bcb3cbe1e859e1fec96329e48b"
    )
    assert base["project_sha256"] == (
        "sha256:f8391a9b7500c8be59b535e4aa447ecc77f9af92776399e6e77cf3c187d564d6"
    )
    assert base["benchmark"]["revision"] == ("7f97d87fa8ab728260c0ba9b09b9c8f00bb82ad5")
    assert base["benchmark"]["cage_commit"] == ("09a191c565230cebb8255899d622d23c7ddeff33")
    qualification = base["snapshot_qualification"]
    assert qualification["status"] == "accepted_and_released"
    assert qualification["snapshot_id"] == "r5qfhoernmujgne9qyhs4"
    assert qualification["complete_receipt_sha256"] == (
        "sha256:e0306c26e6fc21f3d1a9ce2ef48689f2f0b262b11cb32348d79e76c12081184a"
    )
    assert qualification["released_provider_state"] == "terminated"

    runtime = base["runtime"]
    assert runtime["harness"] == "opencode"
    assert runtime["harness_version"] == "1.18.27"
    assert runtime["defer_scoring"] is True
    assert runtime["temperature"] == 0.0
    assert runtime["timeout_seconds"] == 43200
    assert runtime["resources"] == {"cpus": 8, "memory_mb": 65536, "disk_mb": 245760}
    assert runtime["agent_image_id"] == (
        "sha256:7976411d3b5b8eacbb887b8e6148fc14680513b0d35f202fbd7a6205080a329c"
    )
    assert runtime["netproxy_image_id"] == (
        "sha256:0eae9d69a8c092e1b1fbe8a26bacfcce505892b99783f554c7a969adb2080c45"
    )
    assert runtime["evaluator_image_id"] == (
        "sha256:eb5f1948ee3f7d16f66a01c86c4c1cd8950f912c91060e0848ce92942568322a"
    )


def test_wbe_collection_and_gpt_scoring_remain_separate_and_fail_closed() -> None:
    packet = read(WBE)
    collection = packet["score_free_collection"]
    assert collection["state"] == "blocked_pending_candidate_promotion_receipts"
    assert collection["judge_identity_in_collection"] is False
    assert collection["judge_calls_during_collection"] == 0
    assert collection["candidate_launch_plan"] is None
    assert collection["candidate_collection_identity"] is None
    assert collection["matched_pair_plan"] is None
    assert packet["recovering_base_control"]["accepted_score_free_rollout_receipt_sha256"] is None

    score = packet["deferred_gpt_scoring"]
    assert score["state"] == "blocked_exact_gpt_identity_not_yet_proven"
    assert score["launchable"] is False
    assert score["judge_family"] == "GPT"
    assert score["exact_judge_identity"] is None
    assert score["judge_qualification_receipt_sha256"] is None
    assert score["collection_bundle_sha256"] is None
    assert score["score_plan"] is None
    assert score["deferred_score_source_file_sha256"] == file_sha256(DEFERRED_SCORE)
    assert score["agent_calls_during_scoring"] == 0
    assert score["recollection_after_scoring_failure"] is False


def test_wbe_promotion_receipts_are_all_fail_closed() -> None:
    gates = read(WBE)["promotion_gates"]
    assert set(gates) == {
        "export",
        "gpu_reload",
        "stage",
        "serving_registration",
        "live_parity",
    }
    assert all(gate["state"] == "unaccepted" for gate in gates.values())
    assert gates["export"]["accepted_receipt_sha256"] is None
    assert gates["export"]["payload_manifest_sha256"] is None
    assert gates["gpu_reload"]["accepted_receipt_sha256"] is None
    assert gates["stage"]["accepted_receipt_sha256"] is None
    assert gates["serving_registration"]["accepted_receipt_sha256"] is None
    assert gates["serving_registration"]["served_model"] is None
    assert gates["serving_registration"]["model_revision"] is None
    assert gates["live_parity"]["accepted_receipt_sha256"] is None


def test_fleet_packet_reuses_the_exact_accepted_seed43_protocol() -> None:
    packet, protocol, base = read(FLEET), read(FLEET_PROTOCOL), read(FLEET_BASE)
    accepted = packet["accepted_baseline"]
    assert accepted["state"] == "accepted_complete"
    assert accepted["capability_outcomes_included"] is False
    assert accepted["accepted_cells"] == accepted["required_cells"] == 17
    assert accepted["config"] == {
        "path": str(FLEET_BASE.relative_to(ROOT)),
        "file_sha256": file_sha256(FLEET_BASE),
        "evaluation_plan_sha256": protocol["arms"]["base"]["evaluation_plan_sha256"],
    }
    assert accepted["matched_protocol"] == {
        "path": str(FLEET_PROTOCOL.relative_to(ROOT)),
        "file_sha256": file_sha256(FLEET_PROTOCOL),
        "sha256": "sha256:" + protocol["sha256"],
        "protocol_id": protocol["protocol_id"],
    }
    evidence = accepted["evidence"]
    assert evidence["commit"] == "ac8f79e51fc322461d14f80806d1f9aa81d28e68"
    assert evidence["file_sha256"] == (
        "sha256:50158a06a9688500b0c1840ba717f7707560d49ec080d4fb32b6b56e482f98cd"
    )
    assert accepted["authority"]["ledger_plan_sha256"] == (
        "sha256:3f9b54c97d901b147597ac2530b048ff5909f7d7baedbe73ef2052c6cc00e137"
    )
    assert accepted["receipt_integrity"]["missing_receipt_digests"] == 0
    assert accepted["receipt_integrity"]["receipt_roster_sha256"] == (
        "sha256:7a2c57d0e37318a46825afaca2e8361b598c2673f2cdc8af8882d0c723cd565d"
    )

    matched = packet["matched_protocol"]
    assert matched["selection"] == {
        "split_path": str(SPLIT.relative_to(ROOT)),
        "split_file_sha256": file_sha256(SPLIT),
        "task_set_path": str(TASK_SET.relative_to(ROOT)),
        "task_set_file_sha256": file_sha256(TASK_SET),
        "selection_sha256": protocol["selection"]["selection_sha256"],
        "task_count": 17,
        "final_test_split_access": "sealed",
    }
    assert matched["treatment"] == protocol["treatment"]
    assert matched["images"] == protocol["images"]
    assert matched["sampling"] == protocol["sampling"] == base["sampling"]
    assert matched["pass_k"] == protocol["pass_k"] == base["pass_k"] == 1
    assert matched["training_data_eligible"] is False
    assert matched["fallback_rule"].startswith("If exact candidate export")


def test_fleet_candidate_and_resource_identities_remain_fail_closed() -> None:
    packet = read(FLEET)
    candidate = packet["candidate_arm"]
    assert candidate["state"] == "blocked_before_evaluation_config"
    assert candidate["native_checkpoint_receipt_sha256"] == (
        "sha256:af655527bcf21d441723f38d4dc0bb523fee30df47b03ff0206ddf0046ffab26"
    )
    for field in (
        "export_receipt_sha256",
        "export_payload_revision",
        "gpu_reload_receipt_sha256",
        "stage_receipt_sha256",
        "serving_registration_receipt_sha256",
        "live_parity_receipt_sha256",
        "model_revision",
        "session_model",
        "served_id",
        "evaluation_config",
        "evaluation_plan_sha256",
    ):
        assert candidate[field] is None

    resources = packet["resource_identities"]
    assert resources["minted"] is False
    assert all(
        resources[field] is None
        for field in ("job", "config_map", "database", "output_root", "workload", "pod")
    )
    job = packet["future_root_job_contract"]
    assert job["api_version"] == "batch/v1"
    assert job["kind"] == "Job"
    assert job["fleet.ai/failure-alerts"] == "off"
    assert job["priority_class"] == "c1"
    assert job["backoff_limit"] == job["evaluator_gpu_request"] == 0
    assert job["server_preview_required_before_create"] is True
    assert job["uncertain_create_must_not_be_repeated"] is True


def test_packets_include_only_sanitized_identity_and_digest_evidence() -> None:
    for path in (WBE, FLEET):
        serialized = path.read_text().lower()
        assert "/private/tmp" not in serialized
        assert "bearer " not in serialized
        assert "api_key" not in serialized
        assert "authorization:" not in serialized
        assert '"aggregate_result":' not in serialized
        assert '"task_prompt":' not in serialized
        assert '"rollout_trace":' not in serialized
