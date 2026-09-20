from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.sft_runtime import (
    _qwen38_checkpoint_finalization,
    _qwen38_checkpoint_inventory,
    _qwen38_reconcile_evidence,
    _tensor_sha256,
    _unsigned_digest,
)

ADAPTER = "model.layers.0.linear_qkv.adapter.linear_in"


def finalization() -> list[dict]:
    return [{"world_rank": rank, "tp_rank": rank, "finalized": True} for rank in range(8)]


def snapshot(rank: int, tensor: torch.Tensor, *, updates: int, gradient: float | None) -> dict:
    trainable_rows = [{"name": ADAPTER, "dtype": "BF16", "shape": [2, 4], "numel": 8}]
    return {
        "rank": {
            "world_rank": rank,
            "tp_rank": rank,
            "tp_size": 8,
            "pp_rank": 0,
            "pp_size": 1,
            "cp_rank": 0,
            "cp_size": 1,
            "dp_rank": 0,
            "dp_size": 1,
        },
        "target_census": {"selector": "all-linear", "target_counts": {"linear_qkv": 1}},
        "adapters": {
            ADAPTER: {
                "logical_target": "linear_qkv",
                "dtype": "BF16",
                "global_shape": [2, 32],
                "local_shape": [2, 4],
                "sharding": {
                    "tensor_parallel": True,
                    "partition_dim": 1,
                    "partition_stride": 1,
                },
                "sha256": _tensor_sha256(tensor),
            }
        },
        "trainable": {
            "parameter_count": 1,
            "elements": 8,
            "manifest_sha256": _unsigned_digest(trainable_rows),
            "unexpected_parameters": [],
            "nontrainable_adapter_parameters": [],
        },
        "frozen_base": {
            "parameter_count": 2,
            "elements": 16,
            "bytes": 32,
            "manifest_sha256": f"{100 + rank:064x}",
        },
        "successful_optimizer_updates": updates,
        "last_gradient_norm": gradient,
    }


def finalized_checkpoint(root: Path) -> tuple[list[dict], list[dict]]:
    policy = root / "policy"
    (policy / "huggingface").mkdir(parents=True)
    before = []
    after = []
    for rank in range(8):
        first = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + rank
        last = first + 1
        before.append(snapshot(rank, first, updates=0, gradient=None))
        after.append(snapshot(rank, last, updates=1, gradient=0.25))
        torch.save(
            {"model_state_dict": {ADAPTER: last}},
            policy / f"adapter_tp{rank}_pp0_cp0_dp0_ep0_etp{rank}.pt",
        )
    for path in (
        root / "data.pt",
        root / "trainer_state.pt",
        policy / ".metadata",
        policy / "__0_0.distcp",
        policy / "huggingface" / "config.json",
    ):
        path.write_bytes(b"evidence")
    return before, after


def test_reconcile_reopens_every_adapter_shard_and_accepts_one_update(
    tmp_path: Path,
) -> None:
    before, after = finalized_checkpoint(tmp_path)
    files, roles = _qwen38_checkpoint_inventory(tmp_path)
    evidence = _qwen38_reconcile_evidence(
        before,
        after,
        tmp_path,
        forward_loss=1.5,
        trainer_gradient_norm=0.25,
    )

    assert len(roles["adapter"]) == 8
    assert set(files) == {name for values in roles.values() for name in values}
    shards = evidence["adapter_parameters"][ADAPTER]["rank_shards"]
    assert all(row["after_sha256"] == row["checkpoint_sha256"] for row in shards)
    assert evidence["adapter_updated_tensor_count"] == 8
    assert "tensor(" not in repr(evidence)


def test_checkpoint_inventory_rejects_mismatched_native_rank_coordinates(
    tmp_path: Path,
) -> None:
    finalized_checkpoint(tmp_path)
    source = tmp_path / "policy/adapter_tp4_pp0_cp0_dp0_ep0_etp4.pt"
    source.rename(tmp_path / "policy/adapter_tp4_pp0_cp0_dp0_ep0_etp0.pt")

    with pytest.raises(ValueError, match="file roles are incomplete"):
        _qwen38_checkpoint_inventory(tmp_path)


def test_checkpoint_finalization_rejects_duplicate_or_missing_rank() -> None:
    assert _qwen38_checkpoint_finalization(finalization())["async_writes_finalized"] is True
    broken = finalization()
    broken[7] = copy.deepcopy(broken[6])
    with pytest.raises(ValueError, match="duplicate"):
        _qwen38_checkpoint_finalization(broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda before, after: after.__setitem__(7, copy.deepcopy(after[6])),
            "duplicate",
        ),
        (
            lambda before, after: after[3].__setitem__("successful_optimizer_updates", 0),
            "exactly one successful",
        ),
        (
            lambda before, after: after[2]["frozen_base"].__setitem__("manifest_sha256", "f" * 64),
            "frozen base bytes changed",
        ),
        (
            lambda before, after: after[5]["trainable"].__setitem__("elements", 7),
            "adapter-only trainability",
        ),
    ],
)
def test_reconcile_rejects_incomplete_or_false_rank_evidence(
    tmp_path: Path, mutate, message: str
) -> None:
    before, after = finalized_checkpoint(tmp_path)
    mutate(before, after)
    with pytest.raises(ValueError, match=message):
        _qwen38_reconcile_evidence(
            before,
            after,
            tmp_path,
            forward_loss=1.5,
            trainer_gradient_norm=0.25,
        )


def test_reconcile_rejects_saved_adapter_that_differs_from_live_post_update(
    tmp_path: Path,
) -> None:
    before, after = finalized_checkpoint(tmp_path)
    path = tmp_path / "policy/adapter_tp4_pp0_cp0_dp0_ep0_etp4.pt"
    torch.save({"model_state_dict": {ADAPTER: torch.ones(2, 4, dtype=torch.bfloat16)}}, path)
    with pytest.raises(ValueError, match="differs from the live post-update"):
        _qwen38_reconcile_evidence(
            before,
            after,
            tmp_path,
            forward_loss=1.5,
            trainer_gradient_norm=0.25,
        )


def test_checkpoint_receipt_uses_one_post_step_source_read(monkeypatch, tmp_path):
    from training import sft_runtime as runtime

    plan = {
        "output_root": str(tmp_path),
        "model": {"repo": "Qwen/Qwen3.8-27B"},
        "lora": {},
        "wandb": {"run_id": "run"},
    }
    plan["plan_sha256"] = _unsigned_digest(plan)
    trainer = SimpleNamespace(
        plan=plan,
        output=tmp_path,
        last_step_evidence={"forward_loss": 1.5, "lora_gradient_norm": 0.25},
    )
    source = {
        "file_count": 29,
        "total_bytes": 55_563_006_776,
        "manifest_sha256": "a" * 64,
    }
    runtime_source = {
        "file_count": 14,
        "total_bytes": 1234,
        "manifest_sha256": "b" * 64,
    }
    events = []
    monkeypatch.setattr(
        runtime,
        "_qwen38_checkpoint_inventory",
        lambda _checkpoint: (
            {"policy/adapter.pt": {"bytes": 7, "sha256": "c" * 64}},
            {"adapter": ["policy/adapter.pt"]},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_qwen38_reconcile_evidence",
        lambda *_args, **_kwargs: {
            "target_census": {},
            "adapter_parameters": {},
            "trainable_parameter_census": {},
            "frozen_base": {},
            "forward_loss": 1.5,
            "lora_gradient_norm": 0.25,
            "adapter_updated_tensor_count": 1,
        },
    )
    monkeypatch.setattr(
        runtime,
        "validate_plan",
        lambda _plan, *, check_files: events.append(("plan", check_files)),
    )
    monkeypatch.setattr(
        runtime,
        "validate_runtime_sources",
        lambda *_args, **_kwargs: pytest.fail("duplicate runtime source read"),
    )
    monkeypatch.setattr(
        runtime,
        "_qwen38_source_inventory",
        lambda _plan: events.append(("model_and_data", True)) or dict(source),
    )
    monkeypatch.setattr(
        runtime,
        "_qwen38_runtime_source_inventory",
        lambda _plan: events.append(("runtime", True)) or dict(runtime_source),
    )
    monkeypatch.setattr(
        runtime,
        "_qwen38_checkpoint_finalization",
        lambda _rows: {"async_writes_finalized": True},
    )

    receipt = runtime._prepare_qwen38_checkpoint_receipt(
        trainer,
        [],
        [],
        [],
        source,
        runtime_source,
    )

    assert events == [
        ("plan", False),
        ("model_and_data", True),
        ("runtime", True),
    ]
    assert receipt["evidence"]["source_inventory_unchanged"] is True
    assert receipt["source_inventory"]["model_and_data"] == {
        "file_count": 29,
        "total_bytes": 55_563_006_776,
        "before_sha256": "a" * 64,
        "after_sha256": "a" * 64,
    }
