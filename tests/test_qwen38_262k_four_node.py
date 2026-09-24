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
from cyber_post_train.direct_submit import SFT_PRODUCTION_CONTEXT, render_sft_rayjob
from cyber_post_train.jobs import digest
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
            "independent exact-UID release supervision absent",
            "four-node GPU launch has not received root review",
        ],
        "required_release_supervision": hooks.RELEASE_SUPERVISION,
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


def test_candidate_release_supervision_bounds_match_runtime_watchdog():
    plan = candidate()
    supervision = plan["qualification"]["submission_gate"]["required_release_supervision"]
    assert supervision["status"] == "absent_blocks_submission"
    assert supervision["runtime_bounds"] == {
        "startup_seconds": base_runtime.WATCHDOG_STARTUP_SECONDS,
        "idle_seconds": base_runtime.WATCHDOG_IDLE_SECONDS,
        "hard_seconds": base_runtime.sft_watchdog_hard_seconds(plan),
        "checkpoint_drain_seconds": base_runtime.WATCHDOG_DRAIN_SECONDS,
    }
    assert supervision["terminal_release_grace_seconds"] == 300
    assert supervision["required_uid_bindings"] == [
        "RayJob",
        "RayCluster",
        "Kueue Workload",
        "all Pods",
    ]
    assert supervision["terminal_proof"] == [
        "all bound Kubernetes objects absent",
        "all 32 requested GPUs released",
    ]


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
    source = manifest(request)
    rendered, proof = render_sft_rayjob(
        plan,
        request,
        preview(source),
        kubernetes_context=SFT_PRODUCTION_CONTEXT,
        run_id="12345678-1234-4234-9234-123456789abc",
    )
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
        assert pod["priority"] == 10_000
        container = pod["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == 8
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert proof["bound_gpu_cluster_templates"] == 2


def test_config_and_parent_files_are_json_objects():
    assert isinstance(json.loads(CONFIG.read_text()), dict)
    assert isinstance(json.loads(PARENT.read_text()), dict)
