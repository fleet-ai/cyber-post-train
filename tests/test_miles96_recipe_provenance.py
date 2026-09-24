from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-24-miles96-recipe-provenance-v1.json"
DOC = ROOT / "docs/QWEN38_MILES96_RECIPE_PROVENANCE_V1.md"


def _load() -> tuple[dict, str]:
    serialized = EVIDENCE.read_text(encoding="utf-8")
    return json.loads(serialized), serialized


def _digest_without_self(value: dict) -> str:
    body = dict(value)
    body.pop("sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tier(value: dict, tier_id: str) -> dict:
    return next(item for item in value["evidence_tiers"] if item["id"] == tier_id)


def test_provenance_separates_broad_mechanics_from_exact_recipe_proof() -> None:
    value, _ = _load()
    assert value["schema"] == "cyber_qwen38_miles96_recipe_provenance_v1"
    assert value["scope"] == "sanitized_static_reported_and_terminal_evidence_review"
    assert value["sha256"] == _digest_without_self(value)

    assert value["frozen_adapter"]["commit"] == ("978df19a1f6b344e2f88d9502060700a59294681")
    assert set(value["frozen_adapter"]["trainer_source_sha256"]) == {
        "fti_trainers_miles_run_fleet",
        "fti_miles_v1_common",
        "fti_miles_v1_client_recording",
        "fti_fleet_v1",
        "miles_inference_rollout_common",
        "miles_inference_rollout_eval",
        "miles_http_utils",
        "miles_megatron_actor",
        "miles_hf_export",
        "miles_wandb_utils",
    }
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        for digest in value["frozen_adapter"]["trainer_source_sha256"].values()
    )
    assert (
        value["conclusion"]["exact_frozen_adapter_terminal_receipt_present_in_reviewed_scope"]
        is False
    )
    assert value["frozen_adapter"]["phase2_prepared_model"]["status"] == (
        "fresh_hf_and_megatron_inventory_not_yet_bound"
    )

    v004 = _tier(value, "neeraj_dataminer_v004")
    maintained = _tier(value, "maintained_fti_01027_miles96")
    tier_ids = [tier["id"] for tier in value["evidence_tiers"]]
    assert len(tier_ids) == len(set(tier_ids))
    assert set(tier_ids) == {
        "neeraj_dataminer_v004",
        "maintained_fti_01027_miles96",
        "deniz_maintained_miles_reports",
        "qwen36_miles_canary03_operational_precedent",
        "maintained_miles256_compatibility_route",
    }
    assert v004["evidence_state"] == "executed_historical_reference"
    assert v004["classification"] == "cluster_proven_broad_mechanics_only"
    assert v004["executed_code_commit"] == "8452eda94667567e9357dac0d92a05fc48e2d727"
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        for digest in v004["source_files_sha256"].values()
    )
    assert maintained["evidence_state"] == "static_bound_unqualified"
    assert maintained["classification"] == (
        "exact_source_bound_recipe_delta_without_terminal_receipt_in_reviewed_scope"
    )
    assert maintained["static_contract"] == [
        "pins the source commits, image digest and individual trainer source hashes",
        "implements runtime drift checks for maintained FTI and Miles files",
        "invokes the direct supported runner instead of a copied experiment harness",
    ]
    assert v004["proves"] == [
        "one-node TP4/CP2 Qwen3.8 rollout-to-update-to-checkpoint mechanics",
        "the retained state identifies executed code commit "
        "8452eda94667567e9357dac0d92a05fc48e2d727",
        "resume from completed iteration 75 through steps 76-79",
        "terminal iteration-79 checkpoint and completed launcher",
    ]
    assert v004["does_not_prove"] == [
        "Fleet cyber task or verifier suitability",
        "the maintained FTI 0.10.27 source closure",
        "49152 train tokens per GPU",
        "whole-trainer CPU phase offload without optimizer CPU offload",
        "the frozen adapter's task-session and evidence contracts",
        "an independent sealed checkpoint reload",
        "trained-tensor finiteness after every update",
        "capability or transfer improvement",
    ]
    assert maintained["does_not_prove"] == [
        "a completed exact-recipe optimizer update",
        "a reloadable exact-recipe checkpoint",
        "useful Fleet cyber reward variation",
    ]
    assert v004["recipe"]["max_train_tokens_per_gpu"] == 8192
    assert v004["recipe"]["optimizer_cpu_offload"] is True
    assert maintained["recipe"]["max_train_tokens_per_gpu"] == 49152
    assert maintained["recipe"]["whole_trainer_phase_offload"] is True
    assert maintained["recipe"]["optimizer_cpu_offload"] is False


def test_signal_wave_remains_zero_update_and_successor_is_minimal() -> None:
    value, _ = _load()
    wave = value["planned_signal_wave"]
    shape = wave["shape"]
    assert wave["evidence_state"] == "planned_parent_review_required"
    assert wave["authority_source_commit"] == ("9859a27d09b4fd881dfe04bbf9a1dcf7413083b8")
    assert wave["authority_config"] == {
        "repository": "fleet-ai/cyber-post-train",
        "commit": "978df19a1f6b344e2f88d9502060700a59294681",
        "path": "configs/qualification/qwen38-miles-signal-wave-v1.json",
        "self_sha256": ("sha256:8c3c029c45b0a40832b66a6c9fff02f68b9ccd3805e9e12c9afd551375415a93"),
    }
    assert wave["review_config"]["launchable"] is False
    assert (shape["lanes"], shape["nodes_per_lane"], shape["gpus_per_node"]) == (4, 1, 8)
    assert shape["samples_per_lane"] == 8
    assert shape["max_concurrent_episodes_per_lane"] == 2
    assert (shape["priority_class"], shape["queue_priority"]) == ("c1", "q1")
    assert shape["failure_alerts"] is False
    assert shape["root_annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert shape["requeue_if_preempted"] is False
    assert shape["backoff_limit"] == 0
    assert shape["outer_replacements"] == 0
    assert shape["optimizer_steps"] == 0
    assert shape["checkpoint"] is False
    assert shape["mode"] == "eval"
    assert {lane["id"] for lane in wave["lanes"]} == {"2c50", "7317", "8095", "f294"}

    successor = value["minimum_one_update_successor"]
    assert successor["evidence_state"] == "future_blocked_on_accepted_zero_update"
    assert successor["status"] == (
        "no_launch_or_terminal_receipt_in_reviewed_scope_and_not_authorized_by_this_evidence"
    )
    assert successor["additional_fresh_bindings"] == [
        "accepted zero-update predecessor receipt",
        "fresh prepared-model inventory over HF and Megatron trees",
        "fresh task and runtime preflight receipts",
    ]
    assert successor["intended_training_recipe_delta"] == [
        "mode eval to normal",
        "optimizer_steps 0 to 1",
        "learning_rate fixed at 1e-6",
        "dynamic filter check_no_aborted_and_nonzero_std",
        "checkpoint interval fixed at 1",
        "independent one-GPU reload after complete-model assembly",
    ]
    assert successor["acceptance"] == [
        "one mixed-reward group selected without fabricated zeros",
        "one optimizer update completes",
        "at least one trained tensor changes",
        "every trained tensor is finite",
        "step-0 Megatron checkpoint and complete HF export sealed",
        "independent reload executes one ordinary zero-update episode",
        "all training and reload resources released",
    ]


def test_public_provenance_contains_no_private_task_or_secret_material() -> None:
    value, serialized = _load()
    public_text = serialized + DOC.read_text(encoding="utf-8")
    assert value["privacy"] == {
        "prompt_text_included": False,
        "task_keys_or_version_ids_included": False,
        "environment_or_verifier_ids_included": False,
        "instance_or_execution_ids_included": False,
        "environment_variables_included": False,
        "reward_values_from_private_cyber_tasks_included": False,
        "credentials_included": False,
        "trajectory_content_included": False,
    }
    for forbidden in (
        '"task_key":',
        '"task_version_id":',
        '"environment_version_id":',
        '"verifier_version_id":',
        "FLEET_API_KEY",
        "WANDB_API_KEY",
        "ghp_",
        "sk_pw",
        "Bearer ",
    ):
        assert forbidden not in public_text
    assert (
        re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            public_text,
        )
        is None
    )
    assert re.search(r"\bcysec[0-9][A-Za-z0-9_.:/-]*", public_text) is None
    assert "/mnt/sfs/" not in public_text

    forbidden_semantic_keys = {
        "prompt",
        "prompt_text",
        "messages",
        "task_key",
        "task_version_id",
        "environment_id",
        "environment_version_id",
        "verifier_id",
        "verifier_version_id",
        "instance_id",
        "verifier_execution_id",
        "attempt_id",
        "session_id",
        "run_id",
        "reward",
        "reward_value",
        "raw_reward",
        "trajectory",
        "trajectories",
        "environment_variables",
        "env",
        "credentials",
        "secrets",
        "api_key",
        "access_token",
    }

    def assert_sanitized(node: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                child_path = (*path, key)
                if path == () and key == "privacy":
                    continue
                assert key not in forbidden_semantic_keys, child_path
                assert_sanitized(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                assert_sanitized(child, (*path, str(index)))

    assert_sanitized(value)


def test_every_source_location_is_commit_and_line_bound() -> None:
    value, _ = _load()
    sources = list(value["frozen_adapter"]["source_locations"])
    for tier in value["evidence_tiers"]:
        sources.extend(tier.get("source_locations", []))
    sources.extend(value["planned_signal_wave"]["source_locations"])
    assert len(sources) == 21
    claims = {source["claim"] for source in sources}
    assert len(claims) == len(sources)
    assert claims == {
        "exact-field update and reload schemas have never admitted an external run",
        "image, source commits, FTI version, 96K budgets and reward filter",
        "one-node canary plan and explicit v004 scope limitation",
        "direct maintained fti.trainers.miles.run_fleet invocation",
        "runtime source hashes and exact TP4/CP2 recipe drift gate",
        "maintained recipe bindings and first-qualification status",
        "non-launchable one-update template and exact source-hash body",
        "recipe, task-distribution boundary and 80-step ceiling",
        "custom runner, sampling, optimizer, offload and parallelism",
        "one-node manifest, exact image, LR, concurrency and retry behavior",
        "checkpoint-75 resume, steps 76-79, final save and completion",
        "96K recipe, TP4/CP2, 49152-token packing and offload semantics",
        "rollout, reward filtering, optimizer defaults and eval-only path",
        "zero-spread failure mode, radix observation and reported full-weight updates",
        "unverified disposition and missing direct terminal receipts",
        "terminal identity, one-node recipe, zero learning signal and checkpoint observation",
        "exact four-node TP8/CP4 256K recipe",
        "four-node, eight-sample-group maintained payload",
        "iteration-19 save, preemption, restore and weight-equality failure",
        "fixed slots, zero-update eval invocation and exact runtime gates",
        "static-review state, shape, no-update contract and privacy",
    }
    locations = set()
    for source in sources:
        assert re.fullmatch(r"fleet-ai/[A-Za-z0-9_.-]+", source["repository"])
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["path"]
        assert re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", source["lines"])
        location = (source["repository"], source["commit"], source["path"], source["lines"])
        assert location not in locations
        locations.add(location)

    cpt = "fleet-ai/cyber-post-train"
    frozen = "978df19a1f6b344e2f88d9502060700a59294681"
    theseus = "fleet-ai/theseus"
    maintained = "d23116f018cb9213f0a3ee6c228d213abcd85d80"
    integration = "68da13aa9c226d1bfed0ad59b990b99321d44239"
    dataminer = "fleet-ai/dataminer_v2"
    v004_run = "8452eda94667567e9357dac0d92a05fc48e2d727"
    v004_evidence = "10afa8d064bb3dd1c11c50768590e432dfa69097"
    review = "7858ead1f2c498850816168fded1be9d3927fa32"
    expected_locations = {
        (cpt, frozen, "training/miles96_mechanics_canary.py", "37-45"),
        (cpt, frozen, "training/miles96_mechanics_canary.py", "49-73"),
        (cpt, frozen, "training/miles96_mechanics_canary.py", "235-328"),
        (cpt, frozen, "training/miles96_mechanics_canary.py", "748-821"),
        (cpt, frozen, "training/miles96_mechanics_canary.py", "889-942"),
        (cpt, frozen, "docs/QWEN38_MILES96_MECHANICS_CANARY.md", "18-45"),
        (
            cpt,
            frozen,
            "configs/qualification/qwen38-miles96-mechanics-canary-v1.template.json",
            "2-35",
        ),
        (dataminer, v004_run, "experiments/rl-transfer-v004/protocol.md", "41-53"),
        (
            dataminer,
            v004_run,
            "experiments/rl-transfer-v004/run_v004.py",
            "21-27,47-67,91-138,151-208",
        ),
        (
            dataminer,
            v004_run,
            "experiments/rl-transfer-v004/jobs/neeraj-dataminer-transfer-v004-base-easy.yaml",
            "1-70",
        ),
        (dataminer, v004_evidence, "experiments/rl-transfer-v004/state.md", "2889-2963"),
        (
            theseus,
            maintained,
            "services/fti/src/fti/trainers/miles/run_fleet.py",
            "143-265",
        ),
        (
            theseus,
            maintained,
            "services/fti/src/fti/trainers/miles/run_fleet.py",
            "1080-1196,1213-1315",
        ),
        (theseus, integration, "services/fti/CHANGELOG.md", "29-30,104"),
        (
            cpt,
            frozen,
            "configs/discovery/qwen38-lora-evidence-index-v1.json",
            "348-358",
        ),
        (
            cpt,
            frozen,
            "docs/evidence/qwen36-study/2026-08-31-miles-canary03-terminal-v1.json",
            "2-44,66-108,126-130",
        ),
        (
            theseus,
            maintained,
            "services/fti/src/fti/trainers/miles/run_fleet.py",
            "520-541",
        ),
        (
            theseus,
            integration,
            "services/fti/payloads/tool-use-qwen38-256k-v1.json",
            "1-14",
        ),
        (theseus, integration, "services/fti/CHANGELOG.md", "104"),
        (
            cpt,
            frozen,
            "training/miles96_signal_qualification.py",
            "113-185,356-412,421-515",
        ),
        (
            cpt,
            review,
            "configs/qualification/qwen38-miles-signal-score-blind-launch-review-v1.json",
            "1-68,252-287",
        ),
    }
    assert locations == expected_locations

    assert all(
        re.fullmatch(r"[0-9a-f]{40}", commit)
        for commit in (
            value["frozen_adapter"]["commit"],
            value["frozen_adapter"]["reviewed_theseus_integration_commit"],
            value["frozen_adapter"]["image_source_commit"],
            value["frozen_adapter"]["miles_source_commit"],
            value["planned_signal_wave"]["review_commit"],
            value["planned_signal_wave"]["authority_source_commit"],
        )
    )
