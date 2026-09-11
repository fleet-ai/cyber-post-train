"""Scientific bookkeeping tests; native-loop test also runs CPU-only in image."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.sft_runtime import (
    DENSE_FORMAT,
    DENSE_SCHEMA,
    EvalAccumulator,
    ProgressWatchdog,
    _make_trainer_class,
    _unsigned_digest,
    build_runtime_configs,
    dense_rows,
    explicit_tracking_class,
    finalize_failed_run,
    retention_steps,
    sft_overrides,
    tokenize_rows,
    validate_plan,
    validate_runtime_sources,
    write_receipt,
)


def plan(tmp_path):
    return {
        "schema": "cyber_sft_runtime_v2",
        "output_root": str(tmp_path),
        "plan_sha256": "a" * 64,
        "model": {
            "root": "/exact/base",
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "b" * 40,
            "files": [
                {"path": x, "sha256": "c" * 64} for x in ("config.json", "tokenizer_config.json")
            ],
        },
        "datasets": {
            "train": {
                "path": "/data/train.parquet",
                "sha256": "d" * 64,
                "rows": 17,
                "task_keys": ["train"],
            },
            "dev": {
                "path": "/data/dev.parquet",
                "sha256": "e" * 64,
                "rows": 20,
                "task_keys": [f"dev{i}" for i in range(20)],
            },
        },
        "recipe": {
            "epochs": 2,
            "batch_size": 8,
            "microbatch_per_gpu": 1,
            "nodes": 1,
            "gpus_per_node": 8,
            "lr": 1e-6,
            "max_length": 16384,
            "eval_interval": 2,
            "checkpoint_interval": 2,
            "keep_checkpoints": 1,
            "max_steps": 6,
            "seed": 42,
        },
        "wandb": {
            "project": "cyber-post-train",
            "entity": "thefleet",
            "group": "test",
            "run_id": "synthetic-test",
            "name": "synthetic-test",
            "tags": ["test"],
        },
    }


def test_recipe_keeps_tail_batch_and_disables_inline_export(tmp_path):
    value = plan(tmp_path)
    validate_plan(value, check_files=False)
    options = sft_overrides(value)
    assert options["max_training_steps"] == 6  # ceil(17/8) * 2, not floor(34/8)
    assert options["hf_save_interval"] == 0
    assert options["eval_before_train"] is True
    assert options["eval_interval"] == 2
    assert options["max_ckpts_to_keep"] == -1  # custom latest-plus-best retention
    assert options["logger"] == "wandb"


@pytest.mark.parametrize(
    "defect", ["overlap", "wrong_steps", "duplicate_dev", "same_file", "fractional_steps", "resume"]
)
def test_plan_rejects_leakage_and_wrong_step_semantics(tmp_path, defect):
    value = plan(tmp_path)
    if defect == "overlap":
        value["datasets"]["train"]["task_keys"] = ["dev0"]
    elif defect == "wrong_steps":
        value["recipe"]["max_steps"] = 5
    elif defect == "duplicate_dev":
        value["datasets"]["dev"]["task_keys"].append("dev0")
    elif defect == "same_file":
        value["datasets"]["dev"]["path"] = value["datasets"]["train"]["path"]
    elif defect == "fractional_steps":
        value["recipe"]["max_steps"] = 6.0
    else:
        value["resume_from"] = "/old/run"
    with pytest.raises(ValueError):
        validate_plan(value, check_files=False)


def test_token_weighted_and_task_macro_are_distinct_and_reconcile():
    accumulator = EvalAccumulator()
    # Different target lengths: dev0 gets 100 tokens at loss1; others 1 token at loss3.
    metadata = [{"task_key": f"dev{i}"} for i in range(20)]
    counts = [100] + [1] * 19
    total = sum(counts)
    outputs = [
        {"elementwise_loss": [(1 if i == 0 else 3) / total] * count}
        for i, count in enumerate(counts)
    ] + [{"elementwise_loss": []}]
    accumulator.add_batch(metadata, counts, outputs, (100 + 57) / 119)
    metrics = accumulator.metrics({f"dev{i}" for i in range(20)})
    assert metrics["eval_loss"] == pytest.approx(157 / 119)
    assert metrics["task_macro_loss"] == pytest.approx((1 + 19 * 3) / 20)
    assert metrics["supervised_tokens"] == 119
    assert metrics["windows"] == metrics["tasks"] == 20


@pytest.mark.parametrize("task_count", [1, 2, 20, 21])
def test_eval_uses_configured_task_set_not_historical_twenty(task_count):
    accumulator = EvalAccumulator()
    expected = {f"task-{i}" for i in range(task_count)}
    accumulator.add_batch(
        [{"task_key": task} for task in sorted(expected)],
        [1] * task_count,
        [{"elementwise_loss": [2.0 / task_count]} for _ in expected],
        2.0,
    )
    assert accumulator.metrics(expected)["tasks"] == task_count
    assert accumulator.metrics(expected)["eval_loss"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="coverage"):
        accumulator.metrics(expected | {"missing"})
    with pytest.raises(ValueError, match="coverage"):
        EvalAccumulator().metrics(set())


@pytest.mark.parametrize("defect", ["missing_row", "bad_aggregate", "nan", "padding_loss"])
def test_eval_rejects_corrupt_or_unreconciled_outputs(defect):
    outputs = [{"elementwise_loss": [1.0]}, {"elementwise_loss": []}]
    batch_loss = 1.0
    if defect == "missing_row":
        outputs = []
    elif defect == "bad_aggregate":
        batch_loss = 2.0
    elif defect == "nan":
        outputs[0]["elementwise_loss"] = [float("nan")]
    else:
        outputs[1]["elementwise_loss"] = [0.1]
    with pytest.raises(ValueError):
        EvalAccumulator().add_batch([{"task_key": "x"}], [1], outputs, batch_loss)


def test_tokenization_keeps_metadata_and_rejects_truncation():
    rows = [
        {
            "messages": [{"role": "assistant", "content": "synthetic"}],
            "task_key": "x",
            "window_id": "1",
            "token_count": 4,
        }
    ]
    spec = {"rows": 1, "task_keys": ["x"]}

    def tokenize(row, tokenizer, max_length):
        assert max_length is None
        return {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1] * 4,
            "num_actions": 2,
            "loss_mask": [1, 1],
        }

    actual = tokenize_rows(rows, spec, None, tokenize, max_length=4)
    assert actual[0]["task_key"] == "x" and actual[0]["window_id"] == "1"
    with pytest.raises(ValueError):
        tokenize_rows(rows, spec, None, tokenize, max_length=3)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["token_count"] = 5
    with pytest.raises(ValueError):
        tokenize_rows(corrupted, spec, None, tokenize, max_length=8)


def dense_fixture(eligibility_hole=False):
    def row(ids, ranges, copied, number):
        mask = [0] * len(ids)
        spans = []
        for index, start, end in ranges:
            mask[start:end] = [1] * (end - start)
            spans.append(
                {
                    "assistant_index": index,
                    "source_message_index": 2 + 2 * index,
                    "token_start": start,
                    "token_end": end,
                    "source_target_sha256": _unsigned_digest(ids[start:end]),
                }
            )
        return {
            "input_ids": ids,
            "loss_mask": mask,
            "task_key": "train",
            "window_id": str(number),
            "source_session_id": "synthetic-source",
            "source_assistant_count": 3,
            "eligible_assistant_indices": [0, 1, 2],
            "excluded_assistant_targets": [],
            "copied_context_assistant_indices": copied,
            "target_spans": spans,
            "token_count": len(ids),
            "target_token_count": sum(mask),
        }

    rows = [
        row(list(range(9)), [(0, 2, 4), (1, 6, 8)], [], 0),
        row([1, 2, 3, 4, 5, 8, 9], [(2, 5, 7)], [0, 1], 1),
    ]
    spec = {
        "format": DENSE_FORMAT,
        "rows": 2,
        "task_keys": ["train"],
        "supervised_tokens": 6,
        "assistant_responses": 3,
        "source_sessions": 1,
        "source_total_assistant_responses": 3,
        "excluded_assistant_responses": 0,
    }
    if eligibility_hole:
        spec.update(source_total_assistant_responses=4, excluded_assistant_responses=1)
        for item in rows:
            item.update(
                source_assistant_count=4,
                eligible_assistant_indices=[0, 2, 3],
                excluded_assistant_targets=[
                    {
                        "assistant_index": 1,
                        "source_message_index": 4,
                        "reason": "overlength_required_previous_round",
                    }
                ],
            )
            for span in item["target_spans"]:
                if span["assistant_index"]:
                    span["assistant_index"] += 1
                    span["source_message_index"] += 2
        # An excluded assistant may be retained as context, never as a target.
        rows[1]["copied_context_assistant_indices"] = [0, 1, 2]
    return rows, spec


def test_dense_targets_once_and_context_holes_masked():
    rows, spec = dense_fixture()
    actual = dense_rows(rows, spec, max_length=16, vocab_size=32)
    assert actual[0]["num_actions"] == 7  # suffix width, NOT four target tokens
    assert actual[0]["loss_mask"] == [1, 1, 0, 0, 1, 1, 0]
    assert actual[1]["loss_mask"] == [1, 1]  # copied earlier responses are outside target suffix
    assert sum(sum(x["loss_mask"]) for x in actual) == 6


def test_dense_original_ordinal_holes_are_explicit_and_exhaustive():
    rows, spec = dense_fixture(eligibility_hole=True)
    actual = dense_rows(rows, spec, max_length=16, vocab_size=32)
    assert [s["assistant_index"] for r in rows for s in r["target_spans"]] == [0, 2, 3]
    assert sum(sum(x["loss_mask"]) for x in actual) == 6


@pytest.mark.parametrize(
    "defect",
    [
        "overlap",
        "gap",
        "duplicate_exclusion",
        "unknown_reason",
        "changing_inventory",
        "provenance_order",
        "excluded_target",
        "excluded_total",
        "original_total",
    ],
)
def test_dense_rejects_invalid_eligibility_or_exclusion_inventory(defect):
    rows, spec = dense_fixture(eligibility_hole=True)
    if defect == "overlap":
        rows[0]["eligible_assistant_indices"] = [0, 1, 2, 3]
    elif defect == "gap":
        rows[0]["eligible_assistant_indices"] = [0, 2]
    elif defect == "duplicate_exclusion":
        rows[0]["excluded_assistant_targets"] *= 2
    elif defect == "unknown_reason":
        rows[0]["excluded_assistant_targets"][0]["reason"] = "arbitrary"
    elif defect == "changing_inventory":
        rows[1]["excluded_assistant_targets"][0]["reason"] = "overlength_assistant_target"
    elif defect == "provenance_order":
        for r in rows:
            r["excluded_assistant_targets"][0]["source_message_index"] = 6
    elif defect == "excluded_target":
        rows[0]["target_spans"][1]["assistant_index"] = 1
    elif defect == "excluded_total":
        spec["excluded_assistant_responses"] = 0
    else:
        spec["source_total_assistant_responses"] += 1
    with pytest.raises(ValueError):
        dense_rows(rows, spec, max_length=16, vocab_size=32)


@pytest.mark.parametrize(
    "defect",
    [
        "tool_target",
        "duplicate",
        "copied_target",
        "missing_response",
        "span_digest",
        "bad_total",
        "overlength",
        "invalid_token",
    ],
)
def test_dense_rejects_mask_or_coverage_defects(defect):
    rows, spec = dense_fixture()
    maximum = 16
    if defect == "tool_target":
        rows[0]["loss_mask"][4] = 1
        rows[0]["target_token_count"] += 1
    elif defect == "duplicate":
        rows[1]["target_spans"][0]["assistant_index"] = 1
    elif defect == "copied_target":
        rows[1]["copied_context_assistant_indices"] = [2]
    elif defect == "missing_response":
        rows.pop()
        spec.update(rows=1, supervised_tokens=4, assistant_responses=2)
    elif defect == "span_digest":
        rows[0]["target_spans"][0]["source_target_sha256"] = "a" * 64
    elif defect == "bad_total":
        spec["supervised_tokens"] += 1
    elif defect == "overlength":
        maximum = 8
    else:
        rows[0]["input_ids"][0] = -1
    with pytest.raises(ValueError):
        dense_rows(rows, spec, max_length=maximum, vocab_size=32)


def test_dense_plan_supports_configured_epochs_and_heldout_size(tmp_path):
    value = plan(tmp_path)
    value["schema"] = DENSE_SCHEMA
    value["recipe"].update(epochs=1, max_steps=3)
    value["datasets"]["train"].update(
        format=DENSE_FORMAT,
        supervised_tokens=60,
        assistant_responses=30,
        source_sessions=10,
        source_total_assistant_responses=31,
        excluded_assistant_responses=1,
    )
    validate_plan(value, check_files=False)
    assert sft_overrides(value)["train_on_what"] == "all_assistant_messages"
    # The held-out set is bound by the input manifest, not a global task count.
    value["datasets"]["dev"]["rows"] = 80
    value["datasets"]["dev"]["sha256"] = "f" * 64
    validate_plan(value, check_files=False)
    value["datasets"]["dev"]["format"] = DENSE_FORMAT
    with pytest.raises(ValueError, match="evaluation must remain"):
        validate_plan(value, check_files=False)
    value["datasets"]["dev"].pop("format")
    value["datasets"]["dev"]["task_keys"] = ["dev0", "dev1"]
    value["recipe"].update(epochs=2, max_steps=6)
    validate_plan(value, check_files=False)


def test_retention_preserves_best_even_when_old():
    assert retention_steps({50, 100, 150, 200}, 50, 2) == {50, 150, 200}
    assert retention_steps({50, 100}, 0, 1) == {100}


def test_receipts_are_create_once_and_digest_bound(tmp_path):
    path = tmp_path / "progress.json"
    write_receipt(path, {"step": 1})
    with pytest.raises(FileExistsError):
        write_receipt(path, {"step": 2})
    first = json.loads(path.read_text())["receipt_sha256"]
    write_receipt(path, {"step": 2}, replace=True)
    assert json.loads(path.read_text())["receipt_sha256"] != first


def test_watchdog_startup_idle_and_unavailable_evidence():
    watch = ProgressWatchdog(0)
    assert watch.observe(60, ("same",), 0.0, 0) is None
    for now in range(120, 1800, 60):
        assert watch.observe(now, ("same",), 0.0, 0) is None
    assert watch.observe(1800, ("same",), 0.0, 0) == "confirmed_no_progress_idle"
    unknown = ProgressWatchdog(0)
    for now in range(60, 7200, 60):
        assert unknown.observe(now, ("same",), None, None) is None


@pytest.mark.parametrize("activity", ["progress", "gpu", "io"])
def test_watchdog_does_not_kill_useful_work(activity):
    watch = ProgressWatchdog(0)
    for now in range(60, 7200, 60):
        marker = now if activity == "progress" else "same"
        util = 10 if activity == "gpu" else 0
        io_bytes = now * 1024 * 1024 if activity == "io" else 0
        assert watch.observe(now, marker, util, io_bytes) is None


def test_watchdog_hard_bound_and_fixed_checkpoint_drain():
    watch = ProgressWatchdog(0)
    assert watch.observe(28800, "saving", 80, 100, checkpoint_advancing=True) is None
    assert watch.observe(29040, "saving-more", 80, 200, checkpoint_advancing=True) is None
    assert (
        watch.observe(29100, "saving-still", 80, 300, checkpoint_advancing=True)
        == "hard_runtime_bound"
    )
    assert ProgressWatchdog(0).observe(28800, "training", 80, 100) == "hard_runtime_bound"


@pytest.mark.parametrize(
    "disk_failure,summary_failure", [(True, False), (False, True), (True, True)]
)
def test_failure_finalize_is_independent_of_disk_and_summary(
    tmp_path, monkeypatch, disk_failure, summary_failure
):
    calls = []

    class Summary:
        def update(self, values):
            assert values["status"] == "failed"
            calls.append("summary")
            if summary_failure:
                raise RuntimeError("synthetic SDK failure")

    sdk = SimpleNamespace(
        run=SimpleNamespace(summary=Summary()), finish=lambda exit_code: calls.append(exit_code)
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    original_open = Path.open

    def controlled_open(path, *args, **kwargs):
        if disk_failure and path.name == "private-runtime-failure.log":
            raise OSError("synthetic disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", controlled_open)
    # Tracker may be None if W&B created a run but native construction then failed.
    trainer = SimpleNamespace(tracker=None, _ray_gpu_monitor=None, global_step=17)
    failures = finalize_failed_run(trainer, tmp_path, ValueError("synthetic original failure"))
    assert calls == ["summary", 1]
    assert ("private_log" in failures) == disk_failure
    assert ("tracking_summary" in failures) == summary_failure


def test_native_destructor_cannot_mark_failure_successful():
    calls = []

    class NativeTracking:
        def finish(self):
            calls.append(0)

        def __del__(self):
            self.finish()

    tracker = explicit_tracking_class(NativeTracking)()
    tracker.__del__()
    assert calls == []
    # Successful shutdown still explicitly invokes normal native finish.
    tracker.finish()
    assert calls == [0]


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
def test_dense_resolved_config_and_dispatch_cannot_offload(tmp_path):
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    value = plan(tmp_path)
    value["schema"] = DENSE_SCHEMA
    cfg, backend = build_runtime_configs(value)
    assert cfg.fsdp_config.cpu_offload is False
    assert cfg.optimizer_config.offload_after_step is False
    assert backend.trainer.placement.colocate_all is False
    assert backend.trainer.placement.colocate_policy_ref is False

    def forbidden(**kwargs):
        raise AssertionError("single-policy dense SFT must not offload/onload")

    actor = SimpleNamespace(offload_to_cpu=forbidden, backload_to_gpu=forbidden)
    dispatch = WorkerDispatch(backend, policy_actor_group=actor)
    assert dispatch._should_manage_offload("policy") is False
    dispatch._ensure_on_gpu("policy", need_optimizer=True)
    dispatch._offload("policy")
    # Existing staged v2 arms keep their historical config behavior.
    legacy, _ = build_runtime_configs(plan(tmp_path))
    assert legacy.optimizer_config.offload_after_step is True


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
@pytest.mark.parametrize("dense", [False, True])
def test_initial_backload_completes_dense_worker_setup_once(tmp_path, monkeypatch, dense):
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl.train.sft_trainer import SFTTrainer

    calls = []
    value = plan(tmp_path)
    if dense:
        value["schema"] = DENSE_SCHEMA
    cfg, backend = build_runtime_configs(value)

    def backload(**kwargs):
        calls.append(("backload", kwargs))
        return [None] * 8

    actor = SimpleNamespace(
        backload_to_gpu=backload, offload_to_cpu=lambda **kwargs: calls.append(("offload", kwargs))
    )

    def init(self):
        calls.append(("init", {}))
        self.dispatch = WorkerDispatch(backend, policy_actor_group=actor)

    monkeypatch.setattr(SFTTrainer, "_init_workers", init)
    trainer = _make_trainer_class()(cfg, backend, value)
    trainer._init_workers()
    expected = [("init", {})]
    if dense:
        expected.append(("backload", {"backload_optimizer": False, "backload_model": True}))
        # Dedicated single-policy operation must not repeat transfers each step.
        for _ in range(3):
            trainer.dispatch._ensure_on_gpu("policy")
            trainer.dispatch._offload("policy")
        receipt = json.loads((tmp_path / "DEVICE_INITIALIZED.json").read_text())
        assert receipt["ranks_acknowledged"] == 8 and receipt["optimizer_steps"] == 0
    else:
        assert not (tmp_path / "DEVICE_INITIALIZED.json").exists()
    assert calls == expected


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
@pytest.mark.parametrize("replies", [None, [None] * 7, ["unexpected"] * 8])
def test_initial_backload_rejects_incomplete_acknowledgement(tmp_path, monkeypatch, replies):
    from skyrl.train.sft_trainer import SFTTrainer

    value = plan(tmp_path)
    value["schema"] = DENSE_SCHEMA
    cfg, backend = build_runtime_configs(value)
    actor = SimpleNamespace(backload_to_gpu=lambda **kwargs: replies)
    monkeypatch.setattr(
        SFTTrainer,
        "_init_workers",
        lambda self: setattr(self, "dispatch", SimpleNamespace(_actor_groups={"policy": actor})),
    )
    trainer = _make_trainer_class()(cfg, backend, value)
    with pytest.raises(ValueError, match="not acknowledged"):
        trainer._init_workers()
    assert not (tmp_path / "DEVICE_INITIALIZED.json").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
def test_native_backload_moves_nonpersistent_buffers_without_changing_values(monkeypatch):
    import torch
    from skyrl.backends.skyrl_train.distributed import fsdp_utils
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.register_buffer("inv_freq", torch.tensor([1.0, 0.01]), persistent=False)

    model = TinyModel()
    before = {n: t.clone() for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    assert "inv_freq" not in model.state_dict()  # exact missing-from-state-dict boundary
    # Execute the pinned native broadcast + backload with CPU transport, not a
    # mock of the strategy. Actual CUDA placement is separately tested by the run.
    monkeypatch.setattr(fsdp_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_utils.dist, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self.clone())
    fsdp_utils._sync_non_persistent_buffers(model, {})
    assert model.inv_freq.device.type == "cpu"
    moves = []
    native_to = model.to

    def checked_to(device, **kwargs):
        moves.append(device)
        return native_to(device, **kwargs)

    monkeypatch.setattr(model, "to", checked_to)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    strategy = SimpleNamespace(manual_offload=True, manual_offload_optimizer=False)
    FSDPStrategy.backload_to_gpu(
        strategy, model, None, backload_optimizer=False, backload_model=True
    )
    assert moves == ["cpu"]
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name])
    assert "inv_freq" not in model.state_dict()


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
def test_dense_wandb_config_has_only_bound_metadata(tmp_path, monkeypatch):
    import skyrl.train.utils.tracking as tracking
    import wandb

    value = plan(tmp_path)
    value["schema"] = DENSE_SCHEMA
    value["datasets"]["train"].update(
        supervised_tokens=123,
        source_sessions=4,
        assistant_responses=9,
        source_total_assistant_responses=11,
        excluded_assistant_responses=2,
    )
    value["corpus_manifest_sha256"] = "f" * 64
    value["execution"] = {
        "resources": {
            "cpu": {"request": "64", "limit": "64"},
            "memory": {"request": "512Gi", "limit": "768Gi"},
        }
    }
    updates, definitions = {}, []

    class NativeTracking:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(tracking, "Tracking", NativeTracking)
    monkeypatch.setattr(
        wandb,
        "run",
        SimpleNamespace(
            id=value["wandb"]["run_id"],
            entity="thefleet",
            project="cyber-post-train",
            url="https://example.invalid/synthetic",
            define_metric=lambda *args, **kwargs: definitions.append((args, kwargs)),
            config=SimpleNamespace(update=lambda values, **kwargs: updates.update(values)),
        ),
    )
    cfg, backend = build_runtime_configs(value)
    trainer = _make_trainer_class()(cfg, backend, value)
    trainer._init_tracker()
    assert updates["target_policy"] == "visible_all_assistant_once"
    assert updates["dev_target_policy"] == "visible_last_assistant_message"
    assert updates["train_expected_supervised_tokens"] == 123
    assert updates["train_source_sessions"] == 4
    assert updates["train_assistant_responses"] == 9
    assert updates["train_original_assistant_responses"] == 11
    assert updates["train_excluded_assistant_responses"] == 2
    assert updates["corpus_manifest_sha256"] == "f" * 64
    assert updates["execution_resources"] == value["execution"]["resources"]
    assert not any(key in updates for key in ("messages", "input_ids", "loss_mask", "target_spans"))
    assert (tmp_path / "WANDB.json").is_file()


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
@pytest.mark.parametrize("interval", [2, 4])
@pytest.mark.parametrize("dense", [False, True])
def test_exact_native_loop_eval_never_optimizes_and_saves_final(tmp_path, interval, dense):
    """Exercise the real SkyRL loop/collator with synthetic CPU worker outputs."""
    import torch
    from skyrl.train.config.sft_config import SFTConfig, build_skyrl_config_for_sft
    from skyrl.train.dataset.collators import DefaultCollator

    validate_runtime_sources()
    value = plan(tmp_path)
    if dense:
        value["schema"] = DENSE_SCHEMA
        value["recipe"].update(epochs=1, max_steps=3)
        value["datasets"]["train"].update(
            format=DENSE_FORMAT, supervised_tokens=51, assistant_responses=34, source_sessions=17
        )
    value["recipe"]["eval_interval"] = value["recipe"]["checkpoint_interval"] = interval
    cfg = SFTConfig.from_cli_overrides(sft_overrides(value))
    cls = _make_trainer_class()
    trainer = cls(cfg, build_skyrl_config_for_sft(cfg), value)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.collator = DefaultCollator(trainer.tokenizer, 1)
    trainer._ray_gpu_monitor = None

    def row(task, index):
        return {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1] * 4,
            "num_actions": 2,
            "loss_mask": [1, 1],
            "task_key": task,
            "window_id": str(index),
        }

    train_rows = [row("train", i) for i in range(17)]
    if dense:
        train_rows = [
            {
                **r,
                "input_ids": [1, 2, 3, 4, 5, 6, 7],
                "attention_mask": [1] * 7,
                "num_actions": 5,
                "loss_mask": [1, 1, 0, 0, 1],
            }
            for r in train_rows
        ]
    dev_rows = [row(f"dev{i}", i) for i in range(20)]
    trainer.load_dataset = lambda: train_rows

    def load_dev():
        trainer.dev_rows = dev_rows
        return dev_rows

    trainer.load_eval_dataset = load_dev

    class Dispatcher:
        steps = 0
        eval_at = []
        checkpoints = []

        def dp_size(self, model):
            return 8

        def forward_backward(self, model, batch, loss_fn):
            assert loss_fn == "cross_entropy"
            return SimpleNamespace(metrics={"loss": 1.0, "lr": 1e-6})

        def optim_step(self, model):
            self.steps += 1
            return 1.0

        def forward(self, model, batch, loss_fn, loss_fn_config):
            self.eval_at.append(self.steps)
            # Preserve a selected earlier checkpoint when the final step is worse.
            raw_loss = 0.5 if self.steps == interval else 1.0
            mask = batch["loss_mask"]
            outputs = [{"elementwise_loss": (r[r > 0] * raw_loss).tolist()} for r in mask]
            return SimpleNamespace(
                metrics={"loss": float(mask.sum()) * raw_loss}, loss_fn_outputs=outputs
            )

        def save_checkpoint(self, model, path, tokenizer):
            self.checkpoints.append(self.steps)
            Path(path).mkdir(parents=True)
            torch.save({"synthetic_optimizer_step": self.steps}, Path(path) / "state.pt")

    trainer.dispatch = Dispatcher()
    logs = []
    trainer.tracker = SimpleNamespace(
        log=lambda data, step, commit: logs.append((step, dict(data)))
    )
    trainer.train()
    final_step = value["recipe"]["max_steps"]
    assert trainer.dispatch.steps == final_step
    expected_steps = sorted({0, final_step} | set(range(interval, final_step + 1, interval)))
    assert sorted(set(trainer.dispatch.eval_at)) == expected_steps
    assert trainer.dispatch.checkpoints == expected_steps[1:]
    assert trainer.global_step == final_step
    assert int((tmp_path / "checkpoints/latest_ckpt_global_step.txt").read_text()) == final_step
    best_step = interval if interval <= final_step else 0
    retained = {f"global_step_{final_step}"} | (
        {f"global_step_{best_step}"} if best_step else set()
    )
    assert {p.name for p in (tmp_path / "checkpoints").glob("global_step_*")} == retained
    assert trainer.best["optimizer_step"] == best_step
    assert len(list((tmp_path / "validation").glob("*.json"))) == len(expected_steps)
    assert len([item for _, item in logs if "train/loss" in item]) == final_step
    assert all("train/lr" in item for _, item in logs if "train/loss" in item)
    assert logs[-1][1]["train/global_step"] == final_step
    assert logs[-1][0] == final_step + int(final_step % interval != 0)  # native W&B commit ordering
    assert trainer.target_tokens_seen == (51 if dense else 17 * 2 * 2)
    assert not (tmp_path / "hf_exports").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
@pytest.mark.parametrize("dp_size", [1, 8])
def test_native_sparse_ce_forward_backward_matches_causal_reference(monkeypatch, dp_size):
    """Real native wrapper/worker/CE, tiny CPU logits; only CUDA transport is stubbed."""
    from contextlib import nullcontext

    import torch
    from skyrl.backends.skyrl_train.training_batch import pad_training_input_batch
    from skyrl.backends.skyrl_train.utils import torch_utils
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
    from skyrl.train.dataset.collators import DefaultCollator

    validate_runtime_sources()
    raw, spec = dense_fixture(eligibility_hole=True)
    rows = dense_rows(raw, spec, max_length=16, vocab_size=32)
    batch = DefaultCollator(SimpleNamespace(pad_token_id=0), 1)(rows, batch_size=2)
    batch = pad_training_input_batch(batch, 1)
    assert (batch["loss_mask"] > 0).sum().item() == 6
    assert batch["loss_mask"].sum().item() == pytest.approx(1.0)
    assert batch["loss_mask"][-1].count_nonzero() == 0
    size, length = batch["sequences"].shape
    values = torch.arange(size * length * 32, dtype=torch.float32).reshape(size, length, 32) / 1000

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(values.clone())

        def forward(self, *args, **kwargs):
            return {"logits": self.logits.clone()}

    wrapper = HFModelWrapper.__new__(HFModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = TinyModel()
    wrapper.is_vlm = wrapper.remove_microbatch_padding = False
    wrapper.sequence_parallel_size = 1
    wrapper.attn_implementation = "eager"
    wrapper.logprobs_chunk_size = 4
    wrapper.chunked_entropy_from_logits_fn = torch_utils.chunked_entropy_from_logits
    # CPU qualification uses the native non-Flash fallback; never allocates CUDA.
    monkeypatch.setattr(torch_utils, "FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    width = batch["loss_mask"].shape[1]
    experience = SimpleNamespace(
        sequences=batch["sequences"],
        attention_mask=batch["attention_mask"],
        loss_mask=batch["loss_mask"],
        num_actions=width,
        action_mask=None,
        action_log_probs=None,
        base_action_log_probs=None,
        advantages=None,
        rollout_logprobs=None,
        pixel_values=None,
        image_grid_thw=None,
        to_device=lambda device: None,
    )
    worker = SimpleNamespace(
        model=wrapper,
        cfg=SimpleNamespace(algorithm=SimpleNamespace(temperature=1.0, use_entropy_loss=False)),
        mesh_rank=SimpleNamespace(dp_size=dp_size),
        strategy=SimpleNamespace(backward=lambda loss, model, optimizer: loss.backward()),
        optimizer=None,
        scheduler=SimpleNamespace(get_last_lr=lambda: [3e-6]),
    )
    status = PolicyWorkerBase._forward_backward_micro(
        worker, experience, 1.0, loss_fn="cross_entropy"
    )
    reference_logits = values.clone().requires_grad_(True)
    full_mask = torch.cat([torch.zeros(size, length - width), batch["loss_mask"]], dim=1)
    per_token = torch.nn.functional.cross_entropy(
        reference_logits[:, :-1, :].transpose(1, 2), batch["sequences"][:, 1:], reduction="none"
    )
    expected = (per_token * full_mask[:, 1:]).sum()
    (expected * dp_size).backward()
    assert status["loss"] == pytest.approx(expected.item(), rel=1e-6)
    assert status["lr"] == 3e-6
    assert torch.allclose(wrapper.model.logits.grad, reference_logits.grad, atol=2e-7, rtol=1e-5)
    # Tool/context/padding targets induce exactly zero direct token-loss gradient.
    shifted = torch.cat([full_mask[:, 1:], torch.zeros(size, 1)], dim=1)
    assert torch.count_nonzero(wrapper.model.logits.grad[shifted == 0]).item() == 0
    before = wrapper.model.logits.grad.clone()
    forward_only = PolicyWorkerBase._forward_micro_with_loss(worker, experience, "cross_entropy")
    assert forward_only["loss"] == pytest.approx(expected.item(), rel=1e-6)
    assert torch.equal(wrapper.model.logits.grad, before)  # validation never does backward
