"""No paid calls: exact recovery binding and native sampler/optimizer continuation."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_checkpoints import fixture

from training import checkpoints, recovery
from training.sft_runtime import _unsigned_digest, write_receipt


def source(tmp_path, *, progress=True, complete=False):
    p, root, out = fixture(tmp_path)
    p.update(run_name="old-run", execution={"image": "image@sha256:" + "a" * 64})
    if complete:
        p["recipe"].update(epochs=1, max_steps=2)
        p["datasets"]["train"]["rows"] = 9
    record = {"optimizer_step": 2, "checkpoint_path": str(root), "plan_sha256": _unsigned_digest(p)}
    if progress:
        record["training_progress"] = {"supervised_tokens": 64, "best": None}
    path = tmp_path / "checkpoint_receipts/step-000002.json"
    path.unlink()
    write_receipt(path, record)
    manifest = checkpoints.seal(p, 2, out)
    new = copy.deepcopy(p)
    new.update(run_name="new-run", output_root=str(tmp_path.parent / "new-run"))
    new["wandb"]["run_id"] = "new-run"
    return new, manifest, out


def test_recovery_binds_exact_sealed_source_and_never_changes_recipe(tmp_path):
    p, manifest, path = source(tmp_path)
    recovery.bind(
        p,
        {"manifest": path.name, "sha256": recovery.digest(path), "mode": "resume"},
        relative_to=tmp_path,
    )
    recovery.validate(p, check_files=True)
    assert p["recovery"]["checkpoint"] == manifest
    assert manifest["training_progress"] == {"supervised_tokens": 64, "best": None}
    assert not Path(p["output_root"]).exists()


@pytest.mark.parametrize(
    "defect",
    [
        "mode",
        "image",
        "name",
        "output",
        "nested",
        "parent",
        "wandb",
        "recipe",
        "model",
        "data",
        "runtime",
        "hash",
        "unknown",
    ],
)
def test_changed_or_ambiguous_recovery_rejected(tmp_path, defect):
    p, _, path = source(tmp_path)
    cfg = {"manifest": path.name, "sha256": recovery.digest(path), "mode": "validate"}
    if defect == "hash":
        cfg["sha256"] = "0" * 64
    elif defect == "unknown":
        cfg["extra_epochs"] = 1
    elif defect == "mode":
        cfg["mode"] = "latest"
    elif defect == "image":
        p["execution"]["image"] = "another-image"
    elif defect == "name":
        p["run_name"] = "old-run"
    elif defect in {"output", "nested", "parent"}:
        p["output_root"] = str(
            tmp_path
            if defect == "output"
            else tmp_path / "child"
            if defect == "nested"
            else tmp_path.parent
        )
    elif defect == "wandb":
        p["wandb"]["run_id"] = "synthetic-test"
    elif defect == "recipe":
        p["recipe"]["lr"] *= 2
    elif defect == "model":
        p["model"]["revision"] = "c" * 40
    elif defect == "data":
        p["datasets"]["train"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        recovery.bind(p, cfg, relative_to=tmp_path)
        if defect == "runtime":
            p["recovery_runtime_sha256"] = "0" * 64
        recovery.validate(p, check_files=True)


def test_old_checkpoint_can_be_validated_but_not_silently_resumed(tmp_path):
    p, _, path = source(tmp_path, progress=False)
    config = {"manifest": path.name, "sha256": recovery.digest(path), "mode": "validate"}
    recovery.bind(p, config, relative_to=tmp_path)
    recovery.validate(p, check_files=True)
    p["recovery"]["mode"] = "resume"
    with pytest.raises(ValueError, match="saved training progress"):
        recovery.validate(p, check_files=False)
    # A complete training arm cannot be replayed as continuation.
    p["recovery"]["checkpoint"]["source_plan"]["recipe"]["epochs"] = 1
    # Binding validation catches this tampered source, even before continuation.
    with pytest.raises(ValueError, match="digest/schema"):
        recovery.validate(p, check_files=False)


def test_completed_arm_cannot_be_replayed_as_a_resume(tmp_path):
    p, _, path = source(tmp_path, complete=True)
    with pytest.raises(ValueError, match="already completed"):
        recovery.bind(
            p,
            {"manifest": path.name, "sha256": recovery.digest(path), "mode": "resume"},
            relative_to=tmp_path,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"supervised_tokens": 0, "best": None},
        {"supervised_tokens": True, "best": None},
        {"supervised_tokens": 10, "best": {}},
    ],
)
def test_malformed_progress_cannot_authorize_continuation(tmp_path, value):
    _, manifest, _ = source(tmp_path)
    manifest["training_progress"] = value
    manifest["receipt_sha256"] = _unsigned_digest(
        {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError):
        checkpoints.verify(manifest, check_files=False)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            {"x": [torch.tensor([1, 2]), np.array([3]), (4,)]},
            {"x": [torch.tensor([1, 2]), np.array([3]), (4,)]},
            True,
        ),
        (torch.ones(2), torch.zeros(2), False),
        (torch.tensor([1], dtype=torch.int64), torch.tensor([1.0]), False),
        (torch.ones(1), [1], False),
        (np.array([1]), [1], False),
        (np.array([1]), np.array([2]), False),
        (np.array([1], dtype=np.int64), np.array([1.0]), False),
        ([1, 2], [1], False),
        ((1,), [1], False),
        ({"x": 1}, {"y": 1}, False),
    ],
)
def test_native_state_comparison_is_exact(a, b, expected):
    assert recovery.equal_state(a, b) == expected


def test_validation_never_enters_train_or_saves_or_optimizes(tmp_path, monkeypatch):
    p, _, path = source(tmp_path)
    recovery.bind(
        p,
        {"manifest": path.name, "sha256": recovery.digest(path), "mode": "validate"},
        relative_to=tmp_path,
    )
    p["plan_sha256"] = _unsigned_digest(p)
    calls = []
    trainer = SimpleNamespace(
        plan=p,
        load_dataset=lambda: [1],
        load_eval_dataset=lambda: [2],
        build_train_dataloader=lambda rows: rows,
        build_eval_dataloader=lambda rows: rows,
        run_eval=lambda: ({"eval_loss": 1.0, "task_macro_loss": 1.0}, 1),
        tracker=SimpleNamespace(log=lambda *args, **kwargs: calls.append("logged")),
        shutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setattr(recovery, "load", lambda value: 2)
    result = recovery.validate_only(trainer)
    assert result["optimizer_steps_executed"] == 0 and result["optimizer_step"] == 2
    assert calls == ["logged", "shutdown"]
    assert not (tmp_path / "RELOAD_VALIDATED.json").exists()  # producer publishes after return


@pytest.mark.parametrize(
    "defect", [None, "missing_rank", "duplicate_rank", "wrong_step", "sampler"]
)
def test_all_ranks_and_sampler_must_acknowledge_recovery(tmp_path, monkeypatch, defect):
    p, manifest, _ = source(tmp_path)
    p.update(plan_sha256="a" * 64, recovery={"mode": "validate", "checkpoint": manifest})
    # Exercise the dispatch boundary with two simulated ranks. Real payload
    # inventory/topology checks are tested separately and run before dispatch.
    manifest["world_size"] = 2
    replies = [{"rank": i, "optimizer_step": 2} for i in range(2)]
    if defect == "missing_rank":
        replies.pop()
    elif defect == "duplicate_rank":
        replies[1]["rank"] = 0
    elif defect == "wrong_step":
        replies[1]["optimizer_step"] = 1
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))
    state = {}
    dataloader = SimpleNamespace(
        load_state_dict=lambda value: state.update(value),
        state_dict=lambda: {} if defect == "sampler" else state,
    )
    trainer = SimpleNamespace(
        plan=p,
        output=tmp_path / "new",
        train_dataloader=dataloader,
        dispatch=SimpleNamespace(
            _ensure_on_gpu=lambda *a, **k: None,
            _actor_groups={"policy": SimpleNamespace(async_run_ray_method=lambda *a, **k: replies)},
        ),
    )
    if defect:
        with pytest.raises(ValueError):
            recovery.load(trainer)
        assert not (trainer.output / "RECOVERED.json").exists()
    else:
        assert recovery.load(trainer) == 2
        assert trainer.target_tokens_seen == 64
        assert checkpoints.receipt(trainer.output / "RECOVERED.json")["ranks"] == replies


@pytest.mark.skipif(importlib.util.find_spec("skyrl") is None, reason="pinned native image")
@pytest.mark.parametrize(
    "defect", [None, "path", "no_optimizer", "step", "empty", "scheduler", "rng"]
)
def test_native_worker_requires_loaded_optimizer_scheduler_and_rng(tmp_path, monkeypatch, defect):
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    p, manifest, _ = source(tmp_path)
    p["recovery"] = {"mode": "validate", "checkpoint": manifest}
    cls = recovery.worker_class(p)
    worker = cls.__new__(cls)
    expected_scheduler = {"last_epoch": 2, "other": 3}
    rng = {"cpu": torch.tensor([1, 2], dtype=torch.uint8), "numpy": np.random.get_state()}
    extra = Path(manifest["checkpoint_path"]) / "policy/extra_state_world_size_1_rank_0.pt"
    torch.save({"lr_scheduler": expected_scheduler, "rng": rng}, extra)

    def native_load(self, path, optim, sched):
        assert optim and sched
        self.optimizer = SimpleNamespace(
            state={}
            if defect == "empty"
            else {"parameter": {"step": torch.tensor(1 if defect == "step" else 2)}}
        )
        self.scheduler = SimpleNamespace(
            state_dict=lambda: {**expected_scheduler, "other": 4 if defect == "scheduler" else 3}
        )
        self.strategy = SimpleNamespace(get_rng_state=lambda: {} if defect == "rng" else rng)

    monkeypatch.setattr(fsdp_worker.FSDPPolicyWorkerBase, "load_checkpoint", native_load)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    path = str(Path(manifest["checkpoint_path"]) / ("wrong" if defect == "path" else "policy"))
    if defect:
        with pytest.raises(ValueError):
            worker.load_checkpoint(path, load_optimizer_states=defect != "no_optimizer")
    else:
        assert worker.load_checkpoint(path)["optimizer_step"] == 2
    with pytest.raises(ValueError, match="never call the optimizer"):
        worker.optim_step()
    original = fsdp_worker.PolicyWorker
    with recovery.use_worker(p):
        assert fsdp_worker.PolicyWorker is not original
    assert fsdp_worker.PolicyWorker is original


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="pinned native image; CPU only"
)
@pytest.mark.parametrize("epochs", [1, 2])
def test_native_resume_restores_exact_update_and_complete_epoch_coverage(
    tmp_path, monkeypatch, epochs
):
    import ray
    from skyrl.train.dataset.collators import DefaultCollator
    from test_sft_runtime import plan

    from training.sft_runtime import _make_trainer_class, build_runtime_configs

    monkeypatch.setattr(ray, "get", lambda value: value)
    completed_batches = {}
    boundary_states = {}

    def trainer_at(output, manifest=None, stop=None):
        output.mkdir()
        p = plan(output)
        p.pop("plan_sha256")
        p.update(run_name=output.name, execution={"image": "synthetic-pinned-image"})
        p["recipe"]["gpus_per_node"] = 1
        p["recipe"].update(epochs=epochs, max_steps=3 * epochs)
        p["wandb"]["run_id"] = output.name
        if manifest:
            p["recovery"] = {"mode": "resume", "checkpoint": manifest}
            p["recovery_runtime_sha256"] = recovery.digest(Path(recovery.__file__))
        cfg, skyrl = build_runtime_configs(p)
        t = _make_trainer_class()(cfg, skyrl, {**p, "plan_sha256": _unsigned_digest(p)})
        t.tokenizer = SimpleNamespace(pad_token_id=0)
        t.collator = DefaultCollator(t.tokenizer, 1)
        t._ray_gpu_monitor = None
        t.tracker = SimpleNamespace(log=lambda *a, **k: None)
        rows = [
            {
                "input_ids": [1, i + 2, 3, 4],
                "attention_mask": [1] * 4,
                "num_actions": 2,
                "loss_mask": [1, 1],
                "task_key": "train",
                "window_id": str(i),
            }
            for i in range(17)
        ]
        t.load_dataset = lambda: rows

        def dev():
            t.dev_rows = [{**rows[0], "task_key": f"dev{i}"} for i in range(20)]
            return t.dev_rows

        t.load_eval_dataset = dev
        torch.manual_seed(77)
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["recipe"]["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        steps = [0]
        worker_rng = [torch.get_rng_state()]
        completed_batches[output.name] = []

        class Dispatcher:
            def dp_size(self, role):
                return 1

            def forward_backward(self, role, batch, loss_fn):
                if t.global_step == stop:
                    raise RuntimeError("synthetic interruption after saved step")
                valid = batch["loss_mask"].sum(1) > 0
                values = batch["sequences"][:, 1:2].float()[valid] / 32
                completed_batches[output.name].append(values.detach().clone())
                model.train()
                # Real workers are separate Ray processes: creating a driver
                # dataloader must not consume the model worker's dropout RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(worker_rng[0])
                    loss = model(values).square().mean()
                    worker_rng[0] = torch.get_rng_state()
                loss.backward()
                return SimpleNamespace(
                    metrics={"loss": float(loss.detach()), "lr": p["recipe"]["lr"]}
                )

            def optim_step(self, role):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                steps[0] += 1
                if steps[0] == 3:
                    boundary_states[output.name] = copy.deepcopy(
                        (model.state_dict(), optimizer.state_dict())
                    )
                return 1.0

            def forward(self, role, batch, **kwargs):
                mask = batch["loss_mask"]
                return SimpleNamespace(
                    metrics={"loss": float(mask.sum())},
                    loss_fn_outputs=[{"elementwise_loss": r[r > 0].tolist()} for r in mask],
                )

            def save_checkpoint(self, role, path, tokenizer):
                path = Path(path)
                (path / "huggingface").mkdir(parents=True)
                (path / "huggingface/config.json").write_text("{}")
                (path / "fsdp_config.json").write_text(
                    json.dumps({"fsdp_strategy": "fsdp", "world_size": 1})
                )
                for name, value in {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "extra_state": {
                        "scheduler": scheduler.state_dict(),
                        "rng": worker_rng[0],
                    },
                }.items():
                    torch.save(value, path / f"{name}_world_size_1_rank_0.pt")

            def _ensure_on_gpu(self, *a, **k):
                pass  # CPU fixture substitutes workers only, not the native trainer/sampler.

            def async_run_ray_method(self, dispatch, method, *, ckpt_dir, **kwargs):
                assert dispatch == "pass_through" and method == "load_checkpoint"
                path = Path(ckpt_dir)
                model.load_state_dict(
                    torch.load(path / "model_world_size_1_rank_0.pt", weights_only=True)
                )
                optimizer.load_state_dict(
                    torch.load(path / "optim_world_size_1_rank_0.pt", weights_only=True)
                )
                extra = torch.load(path / "extra_state_world_size_1_rank_0.pt", weights_only=True)
                scheduler.load_state_dict(extra["scheduler"])
                worker_rng[0] = extra["rng"]
                steps[0] = int(next(iter(optimizer.state.values()))["step"])
                return [
                    {
                        "rank": 0,
                        "optimizer_step": steps[0],
                        "optimizer_states": 2,
                        "scheduler_restored": True,
                    }
                ]

        t.dispatch = Dispatcher()
        t.dispatch._actor_groups = {"policy": t.dispatch}
        return t, p, model, optimizer

    whole, _, whole_model, whole_optim = trainer_at(tmp_path / "whole")
    whole.train()
    interrupted, p, _, _ = trainer_at(tmp_path / "interrupted", stop=3)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted.train()
    manifest = checkpoints.seal(p, 2, tmp_path / "source.json")
    source_files = {
        name: recovery.digest(Path(manifest["checkpoint_path"]) / name)
        for name in manifest["files"]
    }
    resumed, _, resumed_model, resumed_optim = trainer_at(tmp_path / "resumed", manifest=manifest)
    resumed.train()
    # Native SkyRL explicitly promises exact order through the resumed epoch,
    # not identical reshuffles of later epochs. Check exact optimizer values at
    # that boundary, and lossless/no-duplicate coverage in every later epoch.
    assert recovery.equal_state(boundary_states["whole"], boundary_states["resumed"])
    assert recovery.equal_state(completed_batches["whole"][2], completed_batches["resumed"][0])
    if epochs == 1:
        assert recovery.equal_state(whole_model.state_dict(), resumed_model.state_dict())
        assert recovery.equal_state(whole_optim.state_dict(), resumed_optim.state_dict())
    else:
        for batches in (completed_batches["whole"][3:], completed_batches["resumed"][1:]):
            actual = torch.cat(batches).flatten().sort().values
            assert torch.equal(actual, torch.arange(2, 19).float() / 32)
    assert resumed.target_tokens_seen == whole.target_tokens_seen == 34 * epochs
    assert resumed.global_step == whole.global_step == 3 * epochs
    assert checkpoints.receipt(resumed.output / "RECOVERED.json")["optimizer_step"] == 2
    assert source_files == {
        name: recovery.digest(Path(manifest["checkpoint_path"]) / name)
        for name in manifest["files"]
    }
