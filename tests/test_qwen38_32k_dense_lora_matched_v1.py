import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli, qwen38_lora_sft_preflight
from cyber_post_train.jobs import digest
from training import sft_32k_matched_v1 as compiler
from training import sft_dispatch
from training import sft_runtime as base_runtime
from training import sft_runtime_32k_matched_v1 as runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
DENSE_CONFIG = RUNS / "qwen38-27b-sft-32k-dense-matched-v1.json"
LORA_CONFIG = RUNS / "qwen38-27b-sft-32k-lora-r64-a32-matched-v1.json"
DENSE_SOURCE = RUNS / "qwen38-teacher3k-32k-full-b8-lr3e6-v3.json"
LORA_SOURCE = RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json"
A2_TERMINAL = (
    ROOT / "docs" / "evidence" / "qwen38-lora-anchor-a2-runtime-root-terminal-20260921.json"
)
BASE_RUNTIME_SHA256 = "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17"
RUNNER = CliRunner()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _compile(path: Path) -> tuple[dict, dict, dict]:
    config = _read(path)
    selected = sft_dispatch.compiler_for_config(config)
    plan = selected.compile_sft(config, relative_to=RUNS)
    return config, plan, selected.job_request(plan)


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_SFT_BUNDLE")
    if encoded is None:
        encoded = "".join(
            value
            for _, value in sorted(
                (int(name.rsplit("_", 1)[1]), value)
                for name, value in request["env"].items()
                if name.startswith("CYBER_SFT_BUNDLE_")
            )
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def test_matched_successors_leave_both_create_once_source_identities_immutable():
    assert hashlib.sha256(DENSE_SOURCE.read_bytes()).hexdigest() == (
        "f2de71872423fcdc08ba3139728daf4c2948f2e1fd88a723a128e4568209576f"
    )
    assert hashlib.sha256(LORA_SOURCE.read_bytes()).hexdigest() == (
        "c7f847f8b9d45051f8f5e730dcb6f2009fafecccdba465001a6c57c7ec37f483"
    )
    dense_source, dense = _read(DENSE_SOURCE), _read(DENSE_CONFIG)
    lora_source, lora = _read(LORA_SOURCE), _read(LORA_CONFIG)

    for key in ("model", "data", "cluster"):
        assert dense[key] == dense_source[key]
        assert lora[key] == lora_source[key]
    dense_recipe_drift = {
        key for key in dense["recipe"] if dense["recipe"][key] != dense_source["recipe"][key]
    }
    assert dense_recipe_drift == {"seed"}
    assert {key for key in lora["recipe"] if lora["recipe"][key] != lora_source["recipe"][key]} == {
        "seed"
    }
    assert lora["lora"] == lora_source["lora"]
    assert lora["runtime"] == lora_source["runtime"]
    assert dense["name"] == dense["wandb"]["run_id"] == dense["wandb"]["name"]
    assert lora["name"] == lora["wandb"]["run_id"] == lora["wandb"]["name"]
    assert dense["name"] != dense_source["name"]
    assert lora["name"] != lora_source["name"]
    assert dense["output_root"] != dense_source["output_root"]
    assert lora["output_root"] != lora_source["output_root"]


def test_a2_terminal_evidence_preserves_actual_runtime_cause_and_release_boundary():
    evidence = _read(A2_TERMINAL)
    receipt_sha256 = evidence.pop("receipt_sha256")

    assert (
        hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == receipt_sha256
    )
    assert evidence["run"]["fleet_api_run_id"] == "a237beb8-e93f-4423-8737-21d1ecb0af55"
    assert evidence["run"]["rayjob_uid"] == "9055944a-542c-420d-8543-8a81b17f7754"
    assert evidence["run"]["workload_uid"] == "c7895748-c0a5-44ae-bbe1-7e5c1d644fec"
    assert evidence["terminal"]["error"] == (
        "Qwen3.8 LoRA recovery root differs from its reviewed broad plan"
    )
    assert evidence["terminal"]["optimizer_steps_executed"] == 0
    assert evidence["run"]["failure_alerts"] == "off"
    assert evidence["admission"]["admitted_selector"] == {
        "topology.nebius.com/gpu-cluster-id": "computegpucluster-e04x263hvn91b321fq",
        "workload": "fleetai-training-ng-gpu",
    }
    assert evidence["resource_release"] == {
        "workload_finished": True,
        "raycluster_name_suffix": "-qqhft",
        "raycluster_absent": True,
        "pod_count": 0,
        "allocated_gpus": 0,
    }
    assert evidence["cleanup"] == {
        "authority": "Jobs API",
        "attempts": 1,
        "request": {
            "method": "DELETE",
            "path": "/v1/runs/chris-q38-lora-sft-a2-v1-a237beb8",
        },
        "response": {"http_status": 204, "deleted": True},
        "recorded_at": "2026-09-21T21:13:04Z",
        "reconciliation": {
            "rayjob": {
                "uid": "9055944a-542c-420d-8543-8a81b17f7754",
                "absent": True,
            },
            "workload": {
                "uid": "c7895748-c0a5-44ae-bbe1-7e5c1d644fec",
                "absent": True,
            },
            "raycluster": {"name_suffix": "-qqhft", "absent": True},
        },
        "owned_pod_count": 0,
        "private_output_included": False,
    }
    assert evidence["retirement"]["exact_failed_objects_deleted_via_jobs_api"] is True
    assert evidence["root_cause"]["delegated_identity_reconciliation"] == {
        "equal": True,
        "diff": {},
    }


def test_matched_pair_freezes_data_exposure_seed_batch_schedule_and_optimizer():
    _, dense, dense_request = _compile(DENSE_CONFIG)
    _, lora, lora_request = _compile(LORA_CONFIG)

    assert dense["model"] == lora["model"]
    assert dense["datasets"] == lora["datasets"]
    assert dense["split_manifest_sha256"] == lora["split_manifest_sha256"]
    assert dense["corpus_manifest_sha256"] == lora["corpus_manifest_sha256"]
    train = dense["datasets"]["train"]
    assert train["rows"] == 14_693
    assert train["source_sessions"] == 2_886
    assert train["assistant_responses"] == 176_654
    assert train["supervised_tokens"] == 57_384_881
    assert len(train["task_keys"]) == 496

    for field in (
        "epochs",
        "batch_size",
        "microbatch_per_gpu",
        "nodes",
        "gpus_per_node",
        "max_length",
        "eval_interval",
        "seed",
        "max_steps",
    ):
        assert dense["recipe"][field] == lora["recipe"][field]
    assert dense["recipe"]["epochs"] == 1
    assert dense["recipe"]["batch_size"] == 8
    assert dense["recipe"]["seed"] == runtime.MATCHED_SEED
    assert dense["recipe"]["max_steps"] == 1_837
    assert dense["recipe"]["lr"] == 3e-6
    assert lora["recipe"]["lr"] == 3e-5
    assert dense["optimizer"] == lora["optimizer"] == runtime.MATCHED_OPTIMIZER

    dense_options = runtime.sft_overrides(dense)
    lora_options = runtime.sft_overrides(lora)
    for field, value in {
        "optimizer_config.adam_betas": [0.9, 0.999],
        "optimizer_config.weight_decay": 0.01,
        "optimizer_config.max_grad_norm": 1.0,
        "optimizer_config.scheduler": "constant_with_warmup",
        "optimizer_config.num_warmup_steps": 0,
        "optimizer_config.offload_after_step": False,
    }.items():
        assert dense_options[field] == lora_options[field] == value
    assert dense_options["strategy"] == "fsdp"
    assert lora_options["strategy"] == "megatron"
    assert lora_options["megatron_config.tensor_model_parallel_size"] == 8
    assert dense_request["image"] == runtime.DENSE_IMAGE
    assert lora_request["image"] != dense_request["image"]
    for request in (dense_request, lora_request):
        assert request["workers"] == 1
        assert request["gpus_per_worker"] == 8
        assert request["priority_class"] == "c1"
        assert request["failureAlerts"] is False


def test_matched_lora_uses_the_exact_typed_zero_gpu_preflight_receipt():
    _, dense, dense_request = _compile(DENSE_CONFIG)
    _, lora, lora_request = _compile(LORA_CONFIG)

    assert (
        frozenset({"qwen38_lora_anchor_a2_v1", runtime.RUNTIME_VARIANT})
        == qwen38_lora_sft_preflight.RUNTIME_VARIANTS
    )
    assert not qwen38_lora_sft_preflight.is_qwen38_lora_plan(dense)
    assert qwen38_lora_sft_preflight.is_qwen38_lora_plan(lora)
    assert lora["runtime_variant"]["name"] == (qwen38_lora_sft_preflight.MATCHED_RUNTIME_VARIANT)
    qwen38_lora_sft_preflight.validate_plan_request(lora, lora_request)

    native = {
        "schema": qwen38_lora_sft_preflight.NATIVE_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(lora),
        "request_sha256": digest(lora_request),
    }
    native["sha256"] = digest(native)
    receipt = qwen38_lora_sft_preflight.build_receipt(lora, lora_request, native)

    qwen38_lora_sft_preflight.validate_receipt(receipt, lora, lora_request)
    assert receipt["runtime_variant"]["name"] == runtime.RUNTIME_VARIANT
    assert receipt["gpus"] == 0
    assert dense_request["failureAlerts"] is False

    unknown = copy.deepcopy(lora)
    unknown["runtime_variant"]["name"] = "unreviewed_lora_runtime"
    assert not qwen38_lora_sft_preflight.is_qwen38_lora_plan(unknown)
    with pytest.raises(ValueError, match="reviewed rank-64 runtime"):
        qwen38_lora_sft_preflight.validate_plan_request(unknown, lora_request)


@pytest.mark.parametrize("path", [DENSE_CONFIG, LORA_CONFIG])
def test_matched_runtime_and_base_runtime_are_independently_digest_bound(path):
    _, plan, request = _compile(path)
    bundle = _bundle(request)

    assert plan["runtime_variant"] == {
        "schema": runtime.RUNTIME_BINDING_SCHEMA,
        "name": runtime.RUNTIME_VARIANT,
        "base_runtime_sha256": BASE_RUNTIME_SHA256,
    }
    assert hashlib.sha256(bundle["runtime"].encode()).hexdigest() == plan["runtime_sha256"]
    assert (
        hashlib.sha256(bundle["extra_files"]["training/sft_runtime.py"].encode()).hexdigest()
        == BASE_RUNTIME_SHA256
    )
    assert request["env"]["PYTHONPATH"].startswith(plan["output_root"] + "/.runtime")
    expected_extras = {"training/__init__.py", "training/sft_runtime.py"}
    if "lora" in plan:
        expected_extras |= {"training/io.py", "training/qwen38_lora_artifacts.py"}
    assert set(bundle["extra_files"]) == expected_extras
    assert sft_dispatch.compiler_for_plan(plan) is compiler


def test_lora_entrypoint_reopens_qualification_against_delegated_base_plan(monkeypatch):
    _, plan, _ = _compile(LORA_CONFIG)
    qualification = plan["qualification_gate"]
    reference = qualification["export_receipt"]
    receipt = {"exact_remote_receipt": True}
    reopened = []
    validated = []

    def reopen(path, expected_sha256):
        reopened.append((str(path), expected_sha256))
        return receipt

    monkeypatch.setattr(base_runtime, "_verified_json_file", reopen)
    monkeypatch.setattr(
        base_runtime,
        "_validate_qwen38_production_export_receipt",
        lambda value: validated.append(value),
    )

    with (
        runtime.base_plan_context(),
        pytest.raises(
            ValueError,
            match="LoRA recovery root differs from its reviewed broad plan",
        ),
    ):
        runtime._BASE_VERIFY_PRODUCTION_QUALIFICATION(plan)

    delegated = runtime.delegated_plan(plan)
    with runtime.base_plan_context():
        assert (
            base_runtime._qwen38_lora_one_step_identity(delegated)
            == runtime.lora_base_plan_binding()
        )
    runtime.verify_production_qualification(plan)
    assert "runtime_variant" not in delegated
    assert "optimizer" not in delegated
    assert delegated["runtime_sha256"] == BASE_RUNTIME_SHA256
    assert reopened == [(reference["path"], reference["file_sha256"])]
    assert validated == [receipt]

    original = base_runtime._verify_qwen38_production_qualification
    monkeypatch.setattr(
        base_runtime,
        "main",
        lambda: base_runtime._verify_qwen38_production_qualification(plan),
    )
    runtime.main()
    assert base_runtime._verify_qwen38_production_qualification is original
    assert reopened == [(reference["path"], reference["file_sha256"])] * 2
    assert validated == [receipt, receipt]


@pytest.mark.parametrize("path", [DENSE_CONFIG, LORA_CONFIG])
def test_versioned_request_render_does_not_recurse_inside_cpu_preflight_context(path):
    _, plan, expected = _compile(path)

    with compiler._preflight_context():
        assert compiler.job_request(plan) == expected


@pytest.mark.parametrize("path", [DENSE_CONFIG, LORA_CONFIG])
def test_matched_configs_prepare_locally_without_external_work(path, tmp_path):
    prepared = tmp_path / path.stem
    result = RUNNER.invoke(cli.app, ["train", str(path), "--output", str(prepared)])

    assert result.exit_code == 0, result.output
    plan, request = cli._prepared(prepared)
    assert plan["runtime_variant"]["name"] == runtime.RUNTIME_VARIANT
    assert plan["optimizer"] == runtime.MATCHED_OPTIMIZER
    assert request == compiler.job_request(plan)
    assert not (prepared / "PREFLIGHT.json").exists()


@pytest.mark.parametrize(
    ("path", "keys", "replacement"),
    [
        (DENSE_CONFIG, ("recipe", "seed"), 20260920),
        (LORA_CONFIG, ("recipe", "seed"), 20260919),
        (DENSE_CONFIG, ("recipe", "batch_size"), 16),
        (LORA_CONFIG, ("recipe", "epochs"), 2),
        (DENSE_CONFIG, ("optimizer", "weight_decay"), 0.0),
        (LORA_CONFIG, ("optimizer", "max_grad_norm"), 0.5),
        (DENSE_CONFIG, ("optimizer", "adam_betas"), [0.9, 0.95]),
        (LORA_CONFIG, ("optimizer", "scheduler"), "cosine"),
        (DENSE_CONFIG, ("optimizer", "num_warmup_steps"), 10),
        (LORA_CONFIG, ("wandb", "group"), "unmatched"),
    ],
)
def test_matched_pair_fails_closed_on_scientific_or_optimizer_drift(path, keys, replacement):
    config = copy.deepcopy(_read(path))
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = replacement

    with pytest.raises(ValueError):
        compiler.compile_sft(config, relative_to=RUNS)
