import ast
import base64
import copy
import gzip
import hashlib
import inspect
import json
import shlex
from pathlib import Path

import pytest

from cyber_post_train import cli
from cyber_post_train.jobs import digest, validate_preview
from cyber_post_train.sft_cpu_preflight_job import build_sft_cpu_preflight_job
from tests.test_direct_submit import manifest, preview
from training import sft_262k_4node_v1 as compiler
from training import sft_262k_runtime as hooks
from training import sft_runtime as base_runtime
from training.sft import read_mapping

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json"
PARENT = ROOT / "configs/qualification/qwen38-teacher3k-262k-v12-parent-plan.json"
HOOK_EVIDENCE = ROOT / "configs/qualification/qwen38-teacher3k-262k-v12-hook-asts.json"


def candidate():
    source = read_mapping(CONFIG)
    return compiler.compile_sft(source, relative_to=CONFIG.parent)


def request_bundle(request):
    encoded = request["env"].get("CYBER_SFT_BUNDLE")
    if encoded is None:
        parts = sorted(
            (key, value)
            for key, value in request["env"].items()
            if key.startswith("CYBER_SFT_BUNDLE_")
        )
        encoded = "".join(value for _, value in parts)
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def test_recovered_parent_is_the_exact_retained_v12_plan():
    parent = read_mapping(PARENT)
    assert digest(parent) == compiler.PARENT_PLAN_SHA256
    assert parent["runtime_sha256"] == compiler.PARENT_RUNTIME_SHA256
    assert parent["recipe"] == {
        "batch_size": 64,
        "checkpoint_interval": 1,
        "epochs": 1,
        "eval_interval": 0,
        "gdn_chunk_tokens": 512,
        "gpus_per_node": 8,
        "keep_checkpoints": 2,
        "layer_checkpoint_group_size": 1,
        "lm_head_chunk_tokens": 1024,
        "lr": 3e-6,
        "max_length": 262_144,
        "max_steps": 2,
        "microbatch_per_gpu": 1,
        "mlp_chunk_tokens": 1024,
        "nodes": 8,
        "rmsnorm_chunk_tokens": 1024,
        "seed": 20260916,
        "sequence_parallel_size": 1,
    }


def test_candidate_changes_only_reviewed_topology_and_create_once_identity():
    parent = read_mapping(PARENT)
    plan = candidate()
    assert plan["runtime_variant"] == compiler.VARIANT
    assert plan["recipe"] == {
        **parent["recipe"],
        "nodes": 4,
        "batch_size": 32,
        "max_steps": 4,
    }
    assert plan["model"] == parent["model"]
    assert plan["datasets"] == parent["datasets"]
    assert plan["split_manifest_sha256"] == parent["split_manifest_sha256"]
    assert plan["corpus_manifest_sha256"] == parent["corpus_manifest_sha256"]
    assert plan["execution"] == {
        **parent["execution"],
        "priority": "c1",
        "cluster_target": "prod",
        "jobs_api_base_url": "https://api.ft.flt.build",
    }
    assert plan["long_context_qualification"] == hooks.QUALIFICATION
    assert plan["qualification"]["submission_gate"] == {
        "preview_authorized": True,
        "preflight_authorized": True,
        "submission_authorized": False,
        "blockers": [
            "zero-GPU preflight receipt absent",
            "four-node GPU launch has not received root review",
        ],
        "required_standard_jobs_rail": hooks.STANDARD_JOBS_RAIL,
    }
    request = compiler.job_request(plan)
    assert (request["workers"], request["gpus_per_worker"]) == (4, 8)
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["image"] == compiler.IMAGE
    assert request["run_dir"] == "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1"


def test_candidate_scientific_or_gate_drift_fails_closed():
    plan = candidate()
    for path, replacement in (
        (("recipe", "batch_size"), 64),
        (("recipe", "layer_checkpoint_group_size"), 2),
        (("execution", "priority"), "c2"),
        (("datasets", "train", "rows"), 113),
        (("pause_after_step",), 2),
        (("wandb", "group"), "drifted"),
        (("qualification", "submission_gate", "submission_authorized"), True),
    ):
        broken = copy.deepcopy(plan)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(ValueError):
            hooks.validate_plan(broken, check_files=False)
    broken = copy.deepcopy(plan)
    broken["unexpected"] = True
    with pytest.raises(ValueError):
        hooks.validate_plan(broken, check_files=False)


def test_candidate_allows_preview_and_zero_gpu_preflight_but_blocks_gpu_submit():
    plan = candidate()
    cli._external_action_gate(plan, "preview")
    cli._external_action_gate(plan, "preflight")
    with pytest.raises(ValueError, match="submit blocked by qualification gate"):
        cli._external_action_gate(plan, "submit")


def test_candidate_zero_gpu_preflight_job_has_immediate_terminal_ttl(tmp_path):
    plan = candidate()
    request = compiler.job_request(plan)
    prepared = tmp_path / "prepared"
    cli._prepare(prepared, plan, request)
    package = build_sft_cpu_preflight_job(
        prepared,
        source_commit="a" * 40,
        attempt=1,
    )
    assert package.job["spec"]["ttlSecondsAfterFinished"] == 0
    assert "nvidia.com/gpu" not in json.dumps(package.job)
    assert plan["qualification"]["submission_gate"]["submission_authorized"] is False


def test_candidate_standard_jobs_rail_matches_runtime_watchdog():
    plan = candidate()
    rail = plan["qualification"]["submission_gate"]["required_standard_jobs_rail"]
    assert rail["submission_route"] == "Jobs.submit_once"
    assert rail["operator_preview_required_before_submit"] is True
    assert rail["submit_repeats_server_preview"] is True
    assert rail["complete_duplicate_census_required"] is True
    assert rail["durable_intent_journal"] == "SUBMISSION.jsonl"
    assert rail["runtime_watchdog_bounds"] == {
        "anchor": "ProgressWatchdog construction in _wait_for_training",
        "startup_seconds": base_runtime.WATCHDOG_STARTUP_SECONDS,
        "idle_seconds": base_runtime.WATCHDOG_IDLE_SECONDS,
        "hard_seconds": base_runtime.sft_watchdog_hard_seconds(plan),
        "checkpoint_drain_seconds": base_runtime.WATCHDOG_DRAIN_SECONDS,
    }
    assert rail["rendered_root_contract"] == {
        "failure_alerts": "off",
        "priority_class": "c1",
        "queue_priority": "q1",
        "shutdown_after_job_finishes": True,
        "ttl_seconds_after_finished": 0,
    }
    assert rail["monitoring"] == {
        "identity": "exact Jobs API run name and Kubernetes UIDs",
        "release_route": "one exact Jobs API DELETE after confirmed failure or stall",
        "uncertain_delete_policy": ("reconcile exact API and Kubernetes state before any retry"),
    }


def test_current_hooks_are_ast_identical_to_recovered_v12_hooks():
    evidence = read_mapping(HOOK_EVIDENCE)
    assert evidence["source"] == hooks.QUALIFICATION["historical_runtime_source"]
    assert evidence["runtime_sha256"] == hooks.QUALIFICATION["historical_parent_runtime_sha256"]
    assert evidence["functions"] == hooks.QUALIFICATION["hook_ast_sha256"]
    for name, expected in evidence["functions"].items():
        node = ast.parse(inspect.getsource(getattr(hooks, name))).body[0]
        observed = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        assert observed == expected


def test_candidate_selects_exact_long_context_overrides():
    options = hooks.sft_overrides(candidate())
    assert options["sequence_parallel_size"] == 1
    assert options["model_config_kwargs.fleet_sft_lm_head_chunk_tokens"] == 1024
    assert options["model_config_kwargs.fleet_sft_mlp_chunk_tokens"] == 1024
    assert options["model_config_kwargs.fleet_sft_rmsnorm_chunk_tokens"] == 1024
    assert options["model_config_kwargs.fleet_sft_gdn_chunk_tokens"] == 512
    assert options["model_config_kwargs.fleet_sft_layer_checkpoint_group_size"] == 1
    assert options["fsdp_config.cpu_offload"] is False
    assert options["optimizer_config.offload_after_step"] is False


def test_candidate_stages_isolated_runtime_without_changing_shared_runtime():
    shared = Path(base_runtime.__file__).read_bytes()
    assert hashlib.sha256(shared).hexdigest() == hooks.BASE_RUNTIME_SHA256
    request = compiler.job_request(candidate())
    bundle = request_bundle(request)
    wrapper = Path(hooks.__file__).read_text()
    assert bundle["runtime"] == wrapper
    assert bundle["extra_files"] == {
        "training/__init__.py": "",
        "training/sft_runtime.py": shared.decode(),
        "training/sft_262k_runtime.py": wrapper,
    }
    command = shlex.split(request["command"])
    assert command[:2] == ["python", "-c"]
    assert "runpy.run_path(str(p/'sft_runtime.py'),run_name='__main__')" in command[2]


def test_candidate_exact_render_is_alert_off_c1_q1_and_four_by_eight():
    plan = candidate()
    request = compiler.job_request(plan)
    rendered = manifest(request)
    rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "off"
    rendered["spec"]["ttlSecondsAfterFinished"] = 0
    proof = validate_preview(request, preview(rendered))
    assert rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert rendered["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert rendered["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    groups = [
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"],
        *rendered["spec"]["rayClusterSpec"]["workerGroupSpecs"],
    ]
    assert len(groups) == 2
    assert groups[1]["replicas"] == 3
    for group in groups:
        pod = group["template"]["spec"]
        assert pod["priorityClassName"] == "c1"
        container = pod["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == 8
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert rendered["spec"]["shutdownAfterJobFinishes"] is True
    assert rendered["spec"]["ttlSecondsAfterFinished"] == 0
    assert proof["nodes"] == 4
    assert proof["gpus"] == 32


def test_development_qualification_is_zero_gpu_only_and_bounded_to_thirty_minutes():
    value = read_mapping(ROOT / "configs/qualification/qwen38-teacher3k-262k-4node-canary-v1.json")[
        "dev_qualification"
    ]
    assert value == {
        "exact_four_node_gpu_shape_available": False,
        "allowed_work": "zero_gpu_preflight_only",
        "cleanup_maximum_seconds": 1800,
        "priority": "c1",
    }


def test_config_and_parent_files_are_json_objects():
    assert isinstance(json.loads(CONFIG.read_text()), dict)
    assert isinstance(json.loads(PARENT.read_text()), dict)
