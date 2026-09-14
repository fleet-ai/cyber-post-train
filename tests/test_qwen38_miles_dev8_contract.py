import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from training import miles

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs" / "qualification"
EVIDENCE = ROOT / "docs" / "evidence" / "qwen38-study"


def load(name: str) -> dict:
    return json.loads((QUALIFICATION / name).read_bytes())


def test_dev8_changes_only_fresh_identities_and_the_rejected_generation_cap() -> None:
    data7 = load("qwen38-miles-rl-reward-canary-data-dev-v7.json")
    data8 = load("qwen38-miles-rl-reward-canary-data-dev-v8.json")
    run7 = load("qwen38-miles-rl-reward-canary-dev-v7.json")
    run8 = load("qwen38-miles-rl-reward-canary-dev-v8.json")
    launch = load("qwen38-miles-rl-reward-canary-launch-dev-v8.json")

    assert data7["limits"]["max_tokens_per_turn"] == 4096
    assert data8["limits"]["max_tokens_per_turn"] == 8192
    normalized7 = json.loads(json.dumps(data7))
    normalized8 = json.loads(json.dumps(data8))
    for value in (normalized7, normalized8):
        value["name"] = "RUN"
        value["output"] = "DATA_ROOT"
        value["limits"].pop("max_tokens_per_turn")
    assert normalized7 == normalized8

    normalized_run7 = json.loads(json.dumps(run7))
    normalized_run8 = json.loads(json.dumps(run8))
    for value in (normalized_run7, normalized_run8):
        value["name"] = "RUN"
        value["output_root"] = "OUTPUT_ROOT"
        value["data"] = {"manifest": "MANIFEST", "root": "DATA_ROOT"}
        value["wandb"]["run_id"] = "WANDB_RUN"
    assert normalized_run7 == normalized_run8

    assert launch["scientific_delta"] == {
        "field": "limits.max_tokens_per_turn",
        "from": 4096,
        "to": 8192,
        "reason": (
            "The predecessor ended before reward or optimization because one model turn "
            "reached the 4096-token generation cap."
        ),
        "all_other_data_model_task_split_harness_reward_and_training_fields_unchanged": True,
    }


def test_dev8_is_fresh_dev_c1_and_free_of_external_benchmark_scope() -> None:
    data = load("qwen38-miles-rl-reward-canary-data-dev-v8.json")
    run = load("qwen38-miles-rl-reward-canary-dev-v8.json")
    launch = load("qwen38-miles-rl-reward-canary-launch-dev-v8.json")
    run_id = "chris-q38-miles-rlreward-dev8"
    data_root = "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev8-v1/data"
    output_root = "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev8"

    assert data["name"] == run["name"] == run["wandb"]["run_id"] == run_id
    assert data["output"] == run["data"]["root"] == data_root
    assert run["data"]["manifest"] == data_root + "/manifest.json"
    assert run["output_root"] == output_root
    assert run["cluster"]["target"] == "dev"
    assert run["cluster"]["priority"] == "c1"
    assert launch["status"] == "configuration_only_unsubmitted"
    assert launch["cluster_target"] == "dev"
    assert launch["identities"] == {
        "run_name": run_id,
        "cyber_run_id": run_id,
        "data_root": data_root,
        "prepared_root": "/mnt/sfs/jobs/chris-q38-miles-rlreward-preflight-dev8-v1",
        "output_root": output_root,
        "wandb_run_id": run_id,
    }
    api = launch["jobs_api_constraints"]
    assert api["priority_request"] == "c1"
    assert api["rendered_queue_priority"] == "q1"
    assert api["effective_priority"] == 10000
    assert api["maximum_allowed_priority"] == "c1"
    assert api["requeue_if_preempted"] is False

    text = json.dumps((data, run, launch)).lower()
    assert "webexploitbench" not in text
    assert '"c0"' not in text and '"q0"' not in text
    assert "priority_reason" not in text


def test_dev8_binds_the_exact_clean_dev7_rejection_and_release() -> None:
    launch = load("qwen38-miles-rl-reward-canary-launch-dev-v8.json")
    previous = launch["supersedes"]

    assert previous["run_name"] == "chris-q38-miles-rlreward-dev7"
    assert previous["terminal_receipt"] == {
        "path": "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev7/NATIVE_REJECTED.json",
        "file_sha256": (
            "sha256:8f4d524da088e0db2ee086d8cbf2085799530e7b8899e601b417477bb7e0fa4a"
        ),
        "self_digest_valid": True,
        "status": "rejected",
        "reason": "generation_incomplete_length",
    }
    assert previous["scientific_accounting"] == {
        "accepted_reward_batches": 0,
        "optimizer_updates": 0,
        "checkpoints": 0,
        "training_accepted": False,
    }
    assert previous["release"] == {
        "jobs_api_delete_calls": 1,
        "jobs_api_delete_status": 204,
        "api_absent": True,
        "rayjob_absent": True,
        "workload_absent": True,
        "raycluster_absent": True,
        "gpu_pods_absent": True,
        "evidence_preserved": True,
    }


def test_dev8_native_argument_builder_emits_the_corrected_generation_cap(
    monkeypatch, tmp_path
) -> None:
    data = load("qwen38-miles-rl-reward-canary-data-dev-v8.json")
    run = load("qwen38-miles-rl-reward-canary-dev-v8.json")
    template = tmp_path / "qwen.jinja"
    template.write_text("exact test template")
    monkeypatch.setattr(
        miles, "TEMPLATE_SHA256", hashlib.sha256(template.read_bytes()).hexdigest()
    )
    profile = SimpleNamespace(
        backend="megatron",
        vision=False,
        tito_model="qwen35",
        chat_template=template.name,
        megatron_model_type="qwen3.8-27B",
        parallel_args_by_shape={
            (1, 8): (
                "--tensor-model-parallel-size 4 --sequence-parallel "
                "--pipeline-model-parallel-size 1 --context-parallel-size 2 "
                "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1"
            )
        },
        extra_train_args="--offload-train-target cpu",
        extra_sglang_args="--sglang-disable-radix-cache",
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.8,
        max_tokens_per_gpu=49152,
        log_prob_pass_multiplier=2,
    )
    runtime = ModuleType("fti.trainers.miles.run_fleet")
    runtime.TEMPLATES = tmp_path
    runtime._RECIPES = {"qwen3.8-27b": profile}
    model_args = ModuleType("miles.utils.external_utils.model_args_utils")
    model_args.load_model_args = lambda _name: "--num-layers 64"
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, model_args.__name__, model_args)

    recipe = run["recipe"]
    config = miles.MilesConfig(
        name=run["name"],
        output_root=run["output_root"],
        model_root=run["model"]["root"],
        torch_dist_root="/mnt/sfs/jobs/exact-base-checkpoint/root",
        train_data=run["data"]["root"] + "/train.jsonl",
        dev_data=run["data"]["root"] + "/dev.jsonl",
        data_manifest=run["data"]["manifest"],
        wandb_entity=run["wandb"]["entity"],
        wandb_project=run["wandb"]["project"],
        wandb_run_id=run["wandb"]["run_id"],
        context_tokens=data["limits"]["context_tokens"],
        response_tokens=data["limits"]["response_tokens"],
        tokens_per_turn=data["limits"]["max_tokens_per_turn"],
        **recipe,
    )
    argv = miles.arguments(config)
    assert argv[argv.index("--fleet-max-tokens-per-turn") + 1] == "8192"
    assert argv[argv.index("--max-tokens-per-gpu") + 1] == "8192"
    assert argv[argv.index("--rollout-max-response-len") + 1] == "81920"


def test_dev7_terminal_and_dev8_preflight_evidence_keep_training_closed() -> None:
    terminal = json.loads(
        (EVIDENCE / "2026-09-14-miles-rlreward-dev7-terminal-release-v1.json").read_bytes()
    )
    preflight = json.loads(
        (EVIDENCE / "2026-09-14-miles-rlreward-dev8-zero-gpu-preflight-v1.json").read_bytes()
    )

    assert terminal["terminal_receipt"]["reason"] == "generation_incomplete_length"
    assert terminal["terminal_receipt"]["self_digest_valid"] is True
    assert terminal["scientific_accounting"]["accepted_reward_batches"] == 0
    assert terminal["scientific_accounting"]["optimizer_updates"] == 0
    assert terminal["scientific_accounting"]["valid_training_outcome"] is False
    assert terminal["release"]["jobs_api_delete_calls"] == 1
    assert terminal["release"]["resource_release_uncertain"] is False

    assert preflight["qualification"]["status"] == "cpu_qualified_parser_blocked"
    assert preflight["qualification"]["native_argument_builder_observed_tokens_per_turn"] == 8192
    assert preflight["qualification"]["jobs_api_run_create_posts"] == 0
    assert preflight["native_parser"]["cpu_node_attempt"]["passed"] is False
    assert preflight["native_parser"]["gpu_capable_host_zero_gpu_attempt"]["passed"] is False
    assert preflight["native_parser"]["gpu_capable_host_zero_gpu_attempt"]["gpus_requested"] == 0
    assert preflight["decision"]["gpu_training_submitted"] is False
    assert preflight["decision"]["production_training_submitted"] is False
    assert preflight["decision"]["gpu_submission_authorized_by_this_record"] is False
    assert not any(terminal["privacy"].values())
    assert not any(preflight["privacy"].values())
