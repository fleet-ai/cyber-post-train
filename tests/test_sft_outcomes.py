"""Outcome-only SFT boundaries; synthetic CPU metadata, no W&B/cluster traffic."""

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_checkpoints import fixture as checkpoint_fixture
from test_sft_runtime import plan

from training import checkpoints, recovery
from training.corpus import select_sources
from training.sft_runtime import (
    OUTCOME_WANDB_HISTORY_KEYS,
    PlannedPause,
    ResilientTracking,
    _configure_wandb,
    _make_trainer_class,
    _preflight_wandb_environment,
    _unsigned_digest,
    selection_evidence,
    sft_overrides,
    training_result,
    uses_reference_ce,
    validate_plan,
    write_receipt,
)

POLICY = {"mode": "task_outcomes_only", "fleet_dev_protocol_sha256": "sha256:" + "d" * 64}


def outcome_plan(root):
    value = plan(root)
    value["validation_mode"] = POLICY["mode"]
    value["fleet_dev_protocol_sha256"] = POLICY["fleet_dev_protocol_sha256"]
    value["datasets"].pop("dev")
    value["recipe"]["eval_interval"] = 0
    return value


def test_outcome_plan_disables_all_native_ce_paths(tmp_path):
    value = outcome_plan(tmp_path)
    validate_plan(value, check_files=False)
    cfg = sft_overrides(value)
    assert "eval_dataset_name" not in cfg
    assert cfg["eval_before_train"] is False and cfg["eval_interval"] == 0
    assert cfg["ckpt_interval"] == 2 and cfg["hf_save_interval"] == 0
    assert cfg["max_ckpts_to_keep"] == -1
    assert uses_reference_ce(plan(tmp_path)) is True


def test_outcome_file_preflight_only_checks_train_and_model(tmp_path):
    value = outcome_plan(tmp_path)
    value["model"]["root"] = str(tmp_path)
    train = value["datasets"]["train"]
    train["path"] = str(tmp_path / "train.parquet")
    for spec in [train, *value["model"]["files"]]:
        path = tmp_path / spec["path"]
        path.write_text("synthetic file identity only")
        spec["sha256"] = recovery.digest(path)
    validate_plan(value, check_files=True)
    assert not (tmp_path / "dev.parquet").exists()


@pytest.mark.parametrize(
    "split_name,expected",
    [
        ("a", "sha256:2d20dac7d5b9bb29f6aa44369c5dd2efed9c78bd3ca4d32ed021f04843ae6863"),
        ("b", "sha256:66f98eac3d1533d2c0e3dda99704020e5177ed56c25196f7fa1175307732ab06"),
    ],
)
def test_exact_frozen_train_only_study_manifests_are_supported(split_name, expected):
    # Only public identity metadata, plus one synthetic record; no source corpus read.
    root = Path(__file__).resolve().parents[1]
    split = json.loads(
        (root / f"configs/data/qwen-blackbox-study-train-{split_name}-v1.json").read_text()
    )
    assert split["sha256"] == expected
    assert len(split["tasks"]) == 59 and {t["split"] for t in split["tasks"]} == {"train"}
    first = split["tasks"][0]
    row = {
        "record_id": "synthetic",
        "lineage": {
            "task_key": first["task_key"],
            "eval_task_version_id": first["task_version_id"],
        },
        "source": {"model": "synthetic"},
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
    }
    train, dev, _ = select_sources([row], split, ["synthetic"], reference_validation=False)
    assert train == [row] and dev == []


@pytest.mark.parametrize(
    "defect",
    [
        "dev",
        "positive_interval",
        "boolean_interval",
        "policy",
        "digest",
        "unknown",
        "extra_dataset",
    ],
)
def test_outcome_plan_rejects_ce_and_ambiguous_policy(tmp_path, defect):
    value = outcome_plan(tmp_path)
    if defect in {"dev", "extra_dataset"}:
        value["datasets"]["dev" if defect == "dev" else "test"] = plan(tmp_path)["datasets"]["dev"]
    elif defect == "positive_interval":
        value["recipe"]["eval_interval"] = 2
    elif defect == "boolean_interval":
        value["recipe"]["eval_interval"] = False
    elif defect == "policy":
        value["validation_mode"] = "teacher_cross_entropy"
    elif defect == "digest":
        value["fleet_dev_protocol_sha256"] = "not-bound"
    else:
        value["validation_mode"] = "unknown"
    with pytest.raises(ValueError):
        validate_plan(value, check_files=False)


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    """Exercise actual wrapper methods, replacing only absent native transports."""

    class Native:
        def __init__(self, cfg, *, skyrl_cfg, callbacks):
            self.sft_cfg, self.cfg, self.callbacks = cfg, skyrl_cfg, callbacks

        def save_checkpoint(self):
            root = self.output / "checkpoints" / f"global_step_{self.global_step}"
            root.mkdir(parents=True)
            torch.save({"_num_yielded": (self.global_step - 1) % 3 + 1}, root / "data.pt")
            torch.save({"global_step": self.global_step}, root / "trainer_state.pt")
            (root.parent / "latest_ckpt_global_step.txt").write_text(str(self.global_step))
            return str(root)

    class Tracking:
        def __init__(self, **kwargs):
            self.logs = []

        def log(self, values, **kwargs):
            self.logs.append(dict(values))

    modules = {
        "skyrl.backends.skyrl_train.training_batch": {"pad_training_input_batch": None},
        "skyrl.train.sft_trainer": {"SFTTrainer": Native, "tokenize_chat_example": None},
        "skyrl.train.utils.callbacks": {"TrainingCallback": object},
        "skyrl.train.utils.tracking": {"Tracking": Tracking},
        "skyrl.train.utils.utils": {"Timer": lambda *a, **k: nullcontext()},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    value = outcome_plan(tmp_path)
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            project_name="cyber-post-train", run_name="test", logger="wandb", tags=[]
        )
    )
    trainer = _make_trainer_class()(None, cfg, value)
    trainer.tracker = Tracking()
    return trainer


def event(step=1, total=6, logs=None):
    return SimpleNamespace(
        global_step=step, total_steps=total, steps_per_epoch=3, logs={} if logs is None else logs
    )


@pytest.mark.parametrize("pause", [False, True])
def test_callback_exports_only_loss_step_and_keeps_private_diagnostics(trainer, pause):
    if pause:
        trainer.plan["pause_after_step"] = 1
    trainer.extra_train_metrics = {
        "train/supervised_tokens": 16,
        "train/total_supervised_tokens": 16,
        "train/lr": 1e-6,
        "train/grad_norm": 2,
        "train/grad_norm_finite": 1,
        "train/supervised_tokens_per_second": 32.0,
    }
    record = event(logs={"train/loss": 1.5, "train/grad_norm": 2, "timing/step": 0.1})
    callback = trainer.callbacks[0]
    if pause:
        with pytest.raises(PlannedPause):
            callback.on_log(trainer, record, None)
        assert trainer.tracker.logs == [record.logs]
    else:
        callback.on_log(trainer, record, None)
    assert tuple(record.logs) == OUTCOME_WANDB_HISTORY_KEYS
    local = json.loads((trainer.output / "metrics.jsonl").read_text())
    assert local["train/supervised_tokens"] == 16 and local["train/lr"] == 1e-6
    assert local["train/grad_norm"] == 2
    control = SimpleNamespace(should_save=False, should_evaluate=True)
    callback.on_step_end(trainer, record, control)
    assert control.should_evaluate is False
    assert control.should_save is pause
    callback.on_step_end(trainer, event(step=6), control)
    assert control.should_save is True


def test_no_dev_loading_forward_evaluation_or_ce_callbacks(trainer):
    trainer._load_split = lambda split: pytest.fail("dev source accessed")
    assert trainer.load_eval_dataset() is None
    with pytest.raises(ValueError, match="evaluation is disabled"):
        trainer.run_eval()
    with pytest.raises(ValueError, match="selection is disabled"):
        trainer.callbacks[0].on_eval_end(trainer, event(), None)
    with pytest.raises(ValueError, match="telemetry is disabled"):
        trainer.callbacks[0].on_log(trainer, event(logs={"eval/eval_loss": 1}), None)


def test_tracker_defines_only_training_series_and_explicit_selection(trainer, monkeypatch):
    definitions, config = [], {}
    run = SimpleNamespace(
        id=trainer.plan["wandb"]["run_id"],
        entity="thefleet",
        project="cyber-post-train",
        url="https://example.invalid/synthetic",
        define_metric=lambda key, **kwargs: definitions.append((key, kwargs)),
        config=SimpleNamespace(update=lambda values, **kwargs: config.update(values)),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=run))
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-not-a-real-key")
    trainer._init_tracker()
    assert [key for key, _ in definitions] == [
        "train/global_step",
        "train/total_supervised_tokens",
        *[
            key
            for key in OUTCOME_WANDB_HISTORY_KEYS
            if key not in {"train/global_step", "train/total_supervised_tokens"}
        ],
    ]
    assert all(
        options.get("step_metric") == "train/total_supervised_tokens"
        for key, options in definitions
        if key not in {"train/global_step", "train/total_supervised_tokens"}
    )
    assert config["selection"] == POLICY and config["reference_ce_enabled"] is False
    assert config["checkpoint_selection"] == "external_fleet_dev_task_outcomes"
    assert config["checkpoint_retention"] == "latest_by_step"
    assert config["dev_rows"] == 0 and config["dev_tasks"] == 0
    assert config["dev_target_policy"] == "none_fresh_task_outcomes_after_training"
    receipt = json.loads((trainer.output / "WANDB.json").read_text())
    assert receipt["receipt_sha256"] == _unsigned_digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["identity"] == {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": trainer.plan["wandb"]["run_id"],
    }


def test_resilient_tracking_never_uploads_unreviewed_fields_and_records_sync():
    calls, changes = [], []
    state = {
        "outcome_only": True,
        "status": "active",
        "reason_codes": [],
        "attempted_training_events": 0,
        "accepted_training_events": 0,
        "last_accepted_global_step": None,
        "last_accepted_total_supervised_tokens": None,
        "finish_acknowledged": False,
    }
    delegate = SimpleNamespace(
        log=lambda values, **kwargs: calls.append(dict(values)),
        finish=lambda: calls.append("finish"),
    )
    tracking = ResilientTracking(delegate, state, lambda: changes.append(dict(state)))
    values = {
        "train/global_step": 1,
        "train/loss": 1.5,
        "train/total_supervised_tokens": 16,
        "train/supervised_tokens": 16,
        "train/lr": 1e-6,
        "train/grad_norm": 2.0,
        "train/grad_norm_finite": 1,
        "train/supervised_tokens_per_second": 32.0,
        "private/example": "must-not-upload",
    }
    tracking.log(values, step=1)
    tracking.finish()
    assert tuple(calls[0]) == OUTCOME_WANDB_HISTORY_KEYS
    assert calls[0].keys().isdisjoint({"private/example"})
    assert calls[1] == "finish"
    assert state["status"] == "synced" and state["finish_acknowledged"] is True
    assert state["accepted_training_events"] == state["attempted_training_events"] == 1
    assert changes[-1]["status"] == "synced"


def test_resilient_tracking_preserves_progress_after_sdk_log_failure():
    state = {
        "outcome_only": True,
        "status": "active",
        "reason_codes": [],
        "attempted_training_events": 0,
        "accepted_training_events": 0,
        "last_accepted_global_step": None,
        "last_accepted_total_supervised_tokens": None,
        "finish_acknowledged": False,
    }
    values = dict.fromkeys(OUTCOME_WANDB_HISTORY_KEYS, 1)
    tracking = ResilientTracking(
        SimpleNamespace(
            log=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("synthetic network")),
            finish=lambda: pytest.fail("disabled tracker must not finish"),
        ),
        state,
        lambda: None,
    )
    tracking.log(values, step=1)
    tracking.log(values, step=2)
    tracking.finish()
    assert state["status"] == "unavailable"
    assert state["reason_codes"] == ["history_log_failed"]
    assert state["attempted_training_events"] == 2
    assert state["accepted_training_events"] == 0


def test_train_step_emits_finite_token_normalized_scalar_metrics(trainer):
    trainer._torch_profiler_enabled = False
    trainer.dispatch = SimpleNamespace(
        forward_backward=lambda *args, **kwargs: SimpleNamespace(
            metrics={"final_loss": 1.25, "lr": 3e-6}
        ),
        optim_step=lambda *args, **kwargs: 0.75,
    )
    result = trainer.train_step({"loss_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])}, 1)
    assert result["loss"] == 1.25 and result["grad_norm"] == 0.75
    assert trainer.extra_train_metrics["train/supervised_tokens"] == 3
    assert trainer.extra_train_metrics["train/total_supervised_tokens"] == 3
    assert trainer.extra_train_metrics["train/lr"] == 3e-6
    assert trainer.extra_train_metrics["train/grad_norm_finite"] == 1
    assert trainer.extra_train_metrics["train/supervised_tokens_per_second"] > 0


def test_outcome_wandb_disables_automatic_system_history(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("os"), "environ", dict(__import__("os").environ))
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-not-a-real-key")
    _configure_wandb(outcome_plan(tmp_path))
    assert __import__("os").environ["WANDB__DISABLE_STATS"] == "true"


def test_cluster_preflight_requires_wandb_secret_and_sdk(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="secret injection missing"):
        _preflight_wandb_environment()
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-not-a-real-key")
    monkeypatch.setitem(sys.modules, "wandb", ModuleType("wandb"))
    _preflight_wandb_environment()


def test_terminal_wandb_evidence_requires_exact_synced_scalar_coverage(trainer):
    trainer.wandb_state.update(
        status="synced",
        finish_acknowledged=True,
        receipt_written=True,
        url="https://example.invalid/synthetic",
        attempted_training_events=2,
        accepted_training_events=2,
        last_accepted_global_step=2,
        last_accepted_total_supervised_tokens=32,
    )
    result = {"optimizer_steps_executed": 2, "optimizer_step": 2, "supervised_tokens": 32}
    assert trainer.wandb_evidence(result)["acceptance_ready"] is True
    trainer.wandb_state["accepted_training_events"] = 1
    assert trainer.wandb_evidence(result)["acceptance_ready"] is False


def test_installed_wandb_resolves_outcome_system_history_setting():
    wandb = pytest.importorskip("wandb", reason="requires pinned training image SDK; no network")
    settings = wandb.Settings()
    settings.update_from_env_vars({"WANDB__DISABLE_STATS": "true"})
    assert getattr(settings, "x_disable_stats", getattr(settings, "_disable_stats", False)) is True


def test_actual_save_receipts_bind_outcome_policy_and_recency(trainer):
    trainer.target_tokens_seen = 16
    for step in (2, 4, 6):
        trainer.global_step = step
        trainer.save_checkpoint()
    assert trainer.saved_steps == {6}
    assert {p.name for p in (trainer.output / "checkpoints").glob("global_step_*")} == {
        "global_step_6"
    }
    proof = checkpoints.receipt(trainer.output / "checkpoint_receipts/step-000006.json")
    assert all(proof[key] == value for key, value in selection_evidence(trainer.plan).items())
    result = training_result(trainer, paused=False)
    assert result["best"] is None and result["reference_ce_enabled"] is False
    assert result["selection"] == POLICY and result["optimizer_step"] == 6
    assert not (trainer.output / "validation").exists()


@pytest.mark.parametrize("defect", ["missing_policy", "protocol", "validation", "best"])
def test_outcome_terminal_rejects_missing_bindings_or_ce_artifacts(trainer, defect):
    trainer.target_tokens_seen, trainer.global_step = 16, 6
    trainer.save_checkpoint()
    path = trainer.output / "checkpoint_receipts/step-000006.json"
    if defect in {"missing_policy", "protocol"}:
        proof = checkpoints.receipt(path)
        proof.pop("receipt_sha256")
        if defect == "missing_policy":
            proof.pop("selection")
        else:
            proof["selection"]["fleet_dev_protocol_sha256"] = "sha256:" + "e" * 64
        proof["receipt_sha256"] = _unsigned_digest(proof)
        path.write_text(json.dumps(proof))
    elif defect == "validation":
        (trainer.output / "validation").mkdir()
    else:
        trainer.best = {"optimizer_step": 2}
    with pytest.raises(ValueError):
        training_result(trainer, paused=False)


def sealed_outcome(tmp_path):
    value, root, output = checkpoint_fixture(tmp_path)
    value.update(
        validation_mode=POLICY["mode"],
        fleet_dev_protocol_sha256=POLICY["fleet_dev_protocol_sha256"],
        run_name="source",
        execution={"image": "synthetic"},
    )
    value["datasets"].pop("dev")
    value["recipe"]["eval_interval"] = 0
    path = tmp_path / "checkpoint_receipts/step-000002.json"
    path.unlink()
    write_receipt(
        path,
        {
            "optimizer_step": 2,
            "checkpoint_path": str(root),
            "plan_sha256": _unsigned_digest(value),
            "training_progress": {"supervised_tokens": 32, "best": None},
            **selection_evidence(value),
        },
    )
    manifest = checkpoints.seal(value, 2, output)
    return value, manifest, output


def test_sealed_outcome_checkpoint_and_recovery_preserve_protocol(tmp_path):
    value, manifest, output = sealed_outcome(tmp_path)
    checkpoints.verify(manifest)
    assert manifest["selection"] == POLICY and manifest["reference_ce_enabled"] is False
    new = copy.deepcopy(value)
    new.update(run_name="resumed", output_root=str(tmp_path.parent / "resumed"))
    new["wandb"]["run_id"] = "resumed"
    recovery.bind(
        new,
        {"manifest": str(output), "sha256": recovery.digest(output), "mode": "resume"},
        relative_to=tmp_path,
    )
    recovery.validate(new, check_files=True)
    new["fleet_dev_protocol_sha256"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="selection mode or the Fleet dev protocol"):
        recovery.validate(new, check_files=False)


def test_outcome_sealed_checkpoint_cannot_gain_ce_selection_progress(tmp_path):
    _, manifest, _ = sealed_outcome(tmp_path)
    manifest["training_progress"]["best"] = {
        "optimizer_step": 1,
        "checkpoint_path": "/synthetic",
        "task_macro_loss": 1,
        "token_weighted_loss": 1,
    }
    manifest["receipt_sha256"] = _unsigned_digest(
        {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="must not contain reference CE"):
        checkpoints.verify(manifest, check_files=False)


def test_outcome_zero_step_recovery_never_loads_or_evaluates_ce(tmp_path, monkeypatch):
    value = outcome_plan(tmp_path)
    value["recovery"] = {"checkpoint": {"receipt_sha256": "a" * 64}}
    logs = []
    trainer = SimpleNamespace(
        plan=value,
        load_dataset=lambda: [1],
        build_train_dataloader=lambda rows: rows,
        load_eval_dataset=lambda: pytest.fail("CE dev loaded"),
        build_eval_dataloader=lambda rows: pytest.fail("CE loader built"),
        run_eval=lambda: pytest.fail("CE evaluated"),
        tracker=SimpleNamespace(log=lambda values, **kwargs: logs.append(values)),
        shutdown=lambda: None,
    )
    monkeypatch.setattr(recovery, "load", lambda trainer: 2)
    result = recovery.validate_only(trainer)
    assert result["optimizer_steps_executed"] == 0 and result["selection"] == POLICY
    assert result["validation_scope"] == "checkpoint_and_sampler_reload_only_no_ce"
    assert "held_out" not in result
    assert logs == [{"train/global_step": 2}]
