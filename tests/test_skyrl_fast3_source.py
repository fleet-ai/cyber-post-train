"""Fast3 source-profile tests; local compilation only and no provider calls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import rl_reward_fast3 as canary
from training import (
    sft,
    skyrl,
    skyrl_fast3_posttrain,
    skyrl_fast3_reload,
    skyrl_fast3_retry,
    skyrl_posttrain,
    skyrl_prod9_reload,
)
from training import skyrl_fast3_training as fast3_training

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / canary.FAST3_RUN_PATH
DATA = ROOT / canary.FAST3_DATA_PATH
IDENTITY = ROOT / canary.FAST3_IDENTITY_PATH
POLICY = ROOT / canary.FAST3_POLICY_PATH
SCIENCE = ROOT / canary.FAST3_SCIENCE_PATH
RUNTIME = ROOT / canary.FAST3_RUNTIME_EVIDENCE_PATH
QUALIFICATION = ROOT / canary.FAST3_QUALIFICATION_PATH
PREDECESSOR_MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_file_sha256(value: dict) -> str:
    return fleet.sha256(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _assert_sealed(value: dict, schema: str) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    assert value["schema"] == schema
    assert value["sha256"] == "sha256:" + digest(body)


def _manifest() -> dict:
    value = _load(PREDECESSOR_MANIFEST)
    value["name"] = canary.FAST3_IDENTITY["run_name"]
    value["sha256"] = "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def _compile(run: dict | None = None) -> tuple[dict, dict]:
    selected = run or _load(RUN)
    manifest = _manifest()
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(selected["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = fast3_training.compile_rl(selected, relative_to=RUN.parent)
    return plan, fast3_training.job_request(plan)


def test_fast3_compiles_with_v9_closure_and_historical_eval_accounting() -> None:
    run, data = _load(RUN), _load(DATA)
    identity, policy = _load(IDENTITY), _load(POLICY)
    science, runtime, qualification = _load(SCIENCE), _load(RUNTIME), _load(QUALIFICATION)

    assert run["recipe"] == canary.RECIPE
    assert "eval_before_train" not in run["recipe"]
    assert run["name"] == data["name"] == identity["run_name"]
    assert run["qualification"] == QUALIFICATION.name
    assert runtime["base_commit"] == canary.FAST3_BASE_COMMIT
    assert runtime["eval_schedule"] == {
        "eval_before_train": True,
        "post_update_eval_required": True,
    }
    assert runtime["port_successor_files"] == canary._fast3_port_successor_files()
    assert runtime["port_successor_sha256"] == canary.fast3_port_successor_sha256()
    assert set(runtime["port_successor_files"]) == set(canary.FAST3_PORT_FILES)
    assert fast3_training.RUNTIME_FILES == canary.FAST3_PORT_FILES
    assert set(canary.skyrl_prod9_training.RUNTIME_FILES) < set(canary.FAST3_PORT_FILES)
    assert "training/skyrl_fast3_posttrain.py" in canary.FAST3_PORT_FILES
    assert policy == skyrl_fast3_retry.expected_policy()

    _assert_sealed(identity, "cyber_skyrl_reward_direct_identity_v1")
    _assert_sealed(policy, "cyber_skyrl_generation_http_retry_policy_v1")
    _assert_sealed(science, "cyber_skyrl_fast3_predecessor_science_v1")
    _assert_sealed(runtime, "cyber_rl_fast3_current_main_runtime_evidence_v1")
    _assert_sealed(qualification, "cyber_qwen38_skyrl_reward_canary_port_v9")
    assert qualification["parent_qualification"] == {
        "path": Path(canary.QUALIFICATION_PATH).name,
        "file_sha256": canary.QUALIFICATION_FILE_SHA256,
        "self_sha256": canary.QUALIFICATION_SELF_SHA256,
    }
    assert qualification["predecessor_science"]["file_sha256"] == _file_sha256(SCIENCE)
    assert qualification["predecessor_science"]["self_sha256"] == science["sha256"]
    assert qualification["runtime_evidence"]["file_sha256"] == _file_sha256(RUNTIME)
    assert qualification["runtime_evidence"]["self_sha256"] == runtime["sha256"]
    assert qualification["source"] == {
        "base_commit": canary.FAST3_BASE_COMMIT,
        "port_successor_sha256": runtime["port_successor_sha256"],
    }
    assert qualification["submission_gate"]["submission_authorized"] is False

    scientific = science["scientific_predecessor"]
    assert scientific["run_name"] == "chris-q38-rlreward-prod11"
    assert scientific["eval_before_train"] is True
    assert science["normalized_science"]["recipe"] == canary.FAST3_SCIENCE_RECIPE
    assert scientific["run_config"]["file_sha256"] == (
        "sha256:a7f34702b4c19df80128ed1fd8e0114502605eafd3dd63daf3105f171092038c"
    )
    assert scientific["failure_diagnostic"]["plan_sha256"] == (
        "f86ca0c93754def88d2f9053fb7fa012af3b6d29046846fa5ad229972ce9910f"
    )
    assert scientific["failure_diagnostic"]["http_status_known"] is False
    assert scientific["failure_diagnostic"]["retryability_proven"] is False
    operational = science["operational_predecessor"]
    assert operational["source_commit"] == "58f81904a7478fd90dfe988d97b8fa831fbeadc1"
    assert operational["run_name"] == "chris-q38-rlreward-prod11-fast2"
    assert operational["scientific_parity_claim"] is False
    assert operational["retirement"]["gpu_launch_attempted"] is False
    assert operational["retirement"]["fresh_sfs_lstat"] is False

    for binding, path in (
        (scientific["failure_diagnostic"], ROOT / canary.PROD11_FAILURE_DIAGNOSTIC_PATH),
        (operational["retirement"], ROOT / canary.FAST2_RETIREMENT_PATH),
    ):
        assert binding["file_sha256"] == _file_sha256(path)
        assert binding["self_sha256"] == _load(path)["sha256"]
    for path in (RUN, DATA, IDENTITY, POLICY, SCIENCE, RUNTIME, QUALIFICATION):
        assert b"PLACEHOLDER" not in path.read_bytes()

    plan, request = _compile(run)
    assert plan["schema"] == fast3_training.SCHEMA
    assert plan["fast3_runtime"] == fast3_training._binding()
    assert plan["runtime_sha256"] == digest(fast3_training._runtime())
    binding = plan["qualification"]
    assert binding["schema"] == "cyber_qwen38_skyrl_reward_canary_plan_binding_v9"
    assert binding["profile"] == "qwen38_skyrl_reward_canary_fast3_v1"
    assert binding["fast3"]["generation_retry_policy"] == policy
    assert binding["fast3"]["qualification"] == qualification
    assert canary.validate_plan_binding(binding, plan["data"], plan["arguments"]) == binding
    assert plan["native_overrides"]["trainer.eval_before_train"] is True
    assert "eval_before_train" not in plan["arguments"]
    args = skyrl.SkyRLConfig(**plan["arguments"])
    assert skyrl_posttrain._expected_batches(args) == {
        ("eval", 0),
        ("train", 1),
        ("eval", 1),
    }
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert fast3_training._legacy_plan(plan)["native_overrides"] == plan["native_overrides"]


@pytest.mark.parametrize("value", [False, True])
def test_fast3_rejects_recipe_eval_override(value: bool) -> None:
    run = _load(RUN)
    run["recipe"]["eval_before_train"] = value
    with pytest.raises(ValueError, match="unknown fields in SkyRL recipe"):
        _compile(run)


def test_fast3_rejects_partial_identity() -> None:
    run = _load(RUN)
    run["output_root"] += "-drift"
    with pytest.raises(ValueError):
        _compile(run)


def test_fast3_rejects_fully_resealed_fake_source_map() -> None:
    plan, _ = _compile()
    binding = copy.deepcopy(plan["qualification"])
    fast3 = binding["fast3"]
    policy = fast3["generation_retry_policy"]
    fake_files = {path: "sha256:" + "0" * 64 for path in canary.FAST3_PORT_FILES}
    runtime = canary._fast3_expected_runtime_evidence(files=fake_files)
    science = canary._fast3_expected_science(
        port_successor_sha256=runtime["port_successor_sha256"],
        identity_sha256=fast3["identity_sha256"],
        policy_sha256=policy["sha256"],
    )
    science_file_sha256 = _json_file_sha256(science)
    runtime_file_sha256 = _json_file_sha256(runtime)
    qualification = canary._fast3_expected_qualification(
        science=science,
        science_file_sha256=science_file_sha256,
        runtime=runtime,
        runtime_file_sha256=runtime_file_sha256,
    )
    fast3.update(
        {
            "qualification_file_sha256": _json_file_sha256(qualification),
            "qualification": qualification,
            "runtime_evidence": runtime,
            "predecessor_science_file_sha256": science_file_sha256,
            "predecessor_science": science,
            "port_successor_sha256": runtime["port_successor_sha256"],
        }
    )
    fast3["sha256"] = "sha256:" + digest(
        {key: item for key, item in fast3.items() if key != "sha256"}
    )
    binding.update(
        {
            "qualification_file_sha256": fast3["qualification_file_sha256"],
            "qualification_self_sha256": qualification["sha256"],
            "submission_gate": qualification["submission_gate"],
        }
    )
    binding["sha256"] = "sha256:" + digest(
        {key: item for key, item in binding.items() if key != "sha256"}
    )

    with pytest.raises(ValueError, match="embedded port source closure changed"):
        canary.validate_plan_binding(binding, plan["data"], plan["arguments"])


@pytest.mark.parametrize("target", ["inner", "historical", "outer"])
def test_fast3_rejects_resealed_unknown_plan_binding_fields(target: str) -> None:
    plan, _ = _compile()
    binding = copy.deepcopy(plan["qualification"])
    if target == "inner":
        binding["fast3"]["unexpected"] = "resealed"
        binding["fast3"]["sha256"] = "sha256:" + digest(
            {key: item for key, item in binding["fast3"].items() if key != "sha256"}
        )
    elif target == "historical":
        binding["historical_binding"]["unexpected"] = "resealed"
        binding["historical_binding"]["sha256"] = "sha256:" + digest(
            {key: item for key, item in binding["historical_binding"].items() if key != "sha256"}
        )
    else:
        binding["unexpected"] = "resealed"
    binding["sha256"] = "sha256:" + digest(
        {key: item for key, item in binding.items() if key != "sha256"}
    )

    with pytest.raises(ValueError, match="plan binding fields changed"):
        canary.validate_plan_binding(binding, plan["data"], plan["arguments"])


def test_fast3_port_closure_rejects_a_file_changed_during_read(monkeypatch) -> None:
    stable = {
        "st_dev": 1,
        "st_ino": 2,
        "st_size": 3,
        "st_mtime_ns": 4,
        "st_ctime_ns": 5,
    }

    class ChangedSource:
        reads = 0

        def stat(self):
            self.reads += 1
            return SimpleNamespace(**{**stable, "st_mtime_ns": 4 + self.reads})

        def read_bytes(self):
            return b"abc"

    monkeypatch.setattr(canary, "_path", lambda _path: ChangedSource())
    with pytest.raises(ValueError, match="port source changed while being read"):
        canary._fast3_port_successor_files()


def test_fast3_port_closure_rejects_a_symlink_member(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "source.py"
    target.write_text("pass\n")
    (tmp_path / "link.py").symlink_to(target)
    monkeypatch.setattr(canary, "ROOT", tmp_path)
    monkeypatch.setattr(canary, "FAST3_PORT_FILES", ("link.py",))
    with pytest.raises(ValueError, match="path escapes or is unavailable"):
        canary._fast3_port_successor_files()


def _checkpoint_manifest(plan: dict) -> dict:
    args = skyrl.SkyRLConfig(**plan["arguments"])
    world_size = args.nodes * 8
    names = {"data.pt", "trainer_state.pt", "policy/fsdp_config.json"}
    names |= {
        f"policy/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }
    names.add("policy/huggingface/config.json")
    files = {name: {"bytes": 1, "sha256": "0" * 64} for name in sorted(names)}
    retained = sorted(
        set(range(args.checkpoint_interval, args.steps + 1, args.checkpoint_interval))
        | {args.steps}
    )[-args.keep_checkpoints :]
    body = {
        "schema": skyrl_fast3_posttrain.MANIFEST_SCHEMA,
        "source_plan_sha256": digest(plan),
        "source_plan": copy.deepcopy(plan),
        "terminal_receipt_sha256": "sha256:" + "1" * 64,
        "checkpoint_path": str(
            Path(plan["output_root"]) / "checkpoints" / f"global_step_{args.steps}"
        ),
        "optimizer_step": args.steps,
        "world_size": world_size,
        "checkpoint_interval": args.checkpoint_interval,
        "expected_retained_steps": retained,
        "sampler_batches_in_epoch": (args.steps - 1) % (args.train_rows // args.groups) + 1,
        "rollout_evidence": {
            "train_rollouts": args.steps * args.groups * args.samples_per_prompt,
            "reward_min": 0.0,
            "reward_max": 1.0,
        },
        "update_evidence": {
            "optimizer_steps": args.steps,
            "changed_parameter_tensors": 1,
        },
        "files": files,
        "source_files": {"model/config.json": {"bytes": 1, "sha256": "2" * 64}},
        "evidence_files": {"NATIVE_TRAINING_COMPLETE.json": {"bytes": 1, "sha256": "3" * 64}},
        "total_bytes": len(files),
        "source_inputs_unchanged": True,
        "optimizer_update_verified": True,
        "gpu_reload_verified": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def test_fast3_posttrain_reopens_exact_plan_and_restores_historical_validator() -> None:
    plan, _ = _compile()
    manifest = _checkpoint_manifest(plan)
    original = skyrl_posttrain._validate_plan

    with pytest.raises(ValueError, match="not a native SkyRL training plan"):
        original(plan)
    skyrl_fast3_posttrain.verify_manifest(manifest, check_files=False)
    assert skyrl_posttrain._validate_plan is original

    drift = copy.deepcopy(plan)
    drift["runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Fast3 plan/runtime binding changed"):
        skyrl_fast3_posttrain.verify_manifest(_checkpoint_manifest(drift), check_files=False)
    assert skyrl_posttrain._validate_plan is original

    extra = copy.deepcopy(plan)
    extra["unexpected"] = "resealed-extra-field"
    with pytest.raises(ValueError):
        skyrl_fast3_posttrain.verify_manifest(_checkpoint_manifest(extra), check_files=False)
    assert skyrl_posttrain._validate_plan is original


def _export_receipt(plan: dict, manifest: dict, manifest_file_sha256: str) -> dict:
    args = skyrl.SkyRLConfig(**plan["arguments"])
    body = {
        "schema": skyrl_fast3_posttrain.EXPORT_SCHEMA,
        "source_checkpoint_receipt_sha256": manifest["receipt_sha256"],
        "source_manifest_file_sha256": manifest_file_sha256,
        "source_plan_sha256": digest(plan),
        "model_repo": plan["model"]["repo"],
        "model_revision": plan["model"]["revision"],
        "output_root": plan["output_root"] + f"/hf-export-step{args.steps}-v1",
        "optimizer_step": args.steps,
        "optimizer_steps_executed": 0,
        "gpu_reload_verified": False,
        "dtype": "BF16",
        "all_output_tensors_reopened_equal": True,
        "source_inventory_sizes_mtimes_unchanged": True,
    }
    return {**body, "receipt_sha256": digest(body)}


def test_fast3_reload_accepts_exact_plan_and_rejects_tamper(monkeypatch) -> None:
    plan, _ = _compile()
    manifest = _checkpoint_manifest(plan)
    manifest_file_sha256 = "4" * 64
    export = _export_receipt(plan, manifest, manifest_file_sha256)
    spec = skyrl_fast3_reload.build_spec(
        plan,
        manifest,
        export,
        checkpoint_manifest_file_sha256=manifest_file_sha256,
        export_file_sha256="5" * 64,
    )

    assert spec["schema"] == skyrl_fast3_reload.SPEC_SCHEMA
    assert spec["plan_sha256"] == digest(plan)
    assert spec["runtime"] == {
        "module": skyrl_fast3_reload.MODULE,
        "files_sha256": skyrl_fast3_reload._runtime_hashes(),
        "hard_child_seconds": skyrl_fast3_reload.HARD_CHILD_SECONDS,
    }
    request = skyrl_fast3_reload.job_request(spec)
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert ".fast3-reload-create-claim-v1" in request["command"]
    with pytest.raises(ValueError, match="fresh training schema"):
        skyrl_prod9_reload.build_spec(
            plan,
            manifest,
            export,
            checkpoint_manifest_file_sha256=manifest_file_sha256,
            export_file_sha256="5" * 64,
        )

    monkeypatch.setattr(skyrl_fast3_reload.sft_runtime, "_checked_file", lambda *_args: None)
    monkeypatch.setattr(skyrl_fast3_reload.checkpoints, "receipt", lambda _path: manifest)
    marker = ({"status": "synthetic-export"}, {"tensor": {"dtype": "BF16"}})
    monkeypatch.setattr(skyrl_fast3_reload, "_inspect_export", lambda _spec: marker)
    assert skyrl_fast3_reload.validate_source(spec) == marker

    tampered = copy.deepcopy(manifest)
    tampered["source_plan"]["runtime_sha256"] = "0" * 64
    tampered["source_plan_sha256"] = digest(tampered["source_plan"])
    tampered["receipt_sha256"] = digest(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    monkeypatch.setattr(skyrl_fast3_reload.checkpoints, "receipt", lambda _path: tampered)
    with pytest.raises(ValueError, match="Fast3 plan/runtime binding changed"):
        skyrl_fast3_reload.validate_source(spec)


def test_compiled_policy_is_passed_to_the_fast3_generator(monkeypatch) -> None:
    plan, _ = _compile()
    captured = {}

    class MarkerGenerator:
        def __init__(self, *args, **kwargs):
            captured.update(args=args, kwargs=kwargs)

    monkeypatch.setattr(fast3_training.skyrl_fast3_rollout, "Generator", MarkerGenerator)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    result = fast3_training._generator(plan, args, object(), object())

    assert isinstance(result, MarkerGenerator)
    assert captured["kwargs"]["generation_retry_policy"] == skyrl_fast3_retry.expected_policy()
    assert captured["kwargs"]["repetitions"] == {"train": args.samples_per_prompt, "eval": 1}
    assert captured["kwargs"]["concurrency"] == args.groups * args.samples_per_prompt
