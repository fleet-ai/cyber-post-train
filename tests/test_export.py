import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from test_checkpoints import fixture as checkpoint_fixture
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from training import export as e
from training.checkpoints import seal
from training.sft_runtime import _unsigned_digest, digest, write_receipt


def distributed(local, shape, placement, mesh=(0, 1)):
    """Real DTensor storage/spec; no collectives are needed for offline reads."""
    spec = DTensorSpec(
        SimpleNamespace(mesh=torch.tensor(mesh)),
        (placement,),
        TensorMeta(torch.Size(shape), tuple(torch.empty(shape).stride()), local.dtype),
    )
    return DTensor(local, spec, requires_grad=False)


@pytest.mark.parametrize(("shape", "dim"), [([5, 2], 0), ([2, 5], 1), ([1], 0)])
def test_native_dtensor_shards_reassemble_in_rank_order(shape, dim):
    full = torch.arange(torch.empty(shape).numel(), dtype=torch.float32).reshape(shape)
    local = list(torch.chunk(full, 2, dim=dim))
    if len(local) == 1:
        empty_shape = list(shape)
        empty_shape[dim] = 0
        local.append(torch.empty(empty_shape))
    parts = [distributed(value, shape, Shard(dim)) for value in local]
    assert torch.equal(e.reassemble(parts, shape), full.bfloat16())


def test_replicas_and_single_rank_tensor():
    value = torch.tensor([1.125, 2.25])
    assert torch.equal(e.reassemble([value], [2]), value.bfloat16())
    parts = [distributed(value.clone(), [2], Replicate()) for _ in range(2)]
    assert torch.equal(e.reassemble(parts, [2]), value.bfloat16())


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "plain_multirank",
        "mesh",
        "placement",
        "global_shape",
        "local_shape",
        "dtype",
        "device",
        "shard_dimension",
        "replica_disagrees",
        "partial",
        "nan",
        "oversize",
    ],
)
def test_bad_rank_tensors_fail_closed(defect, monkeypatch):
    shape = [4]
    parts = [distributed(torch.ones(2), shape, Shard(0)) for _ in range(2)]
    if defect == "missing":
        parts = []
    elif defect == "plain_multirank":
        parts = [torch.ones(2), torch.ones(2)]
    elif defect == "mesh":
        parts[1] = distributed(torch.ones(2), shape, Shard(0), (1, 0))
    elif defect == "placement":
        parts[1] = distributed(torch.ones(2), shape, Replicate())
    elif defect == "global_shape":
        shape = [5]
    elif defect == "local_shape":
        parts[1] = distributed(torch.ones(3), shape, Shard(0))
    elif defect == "dtype":
        parts = [torch.ones(4, dtype=torch.bfloat16)]
    elif defect == "device":
        parts = [torch.ones(4, device="meta")]
    elif defect == "shard_dimension":
        parts = [distributed(torch.ones(2), shape, Shard(1)) for _ in range(2)]
    elif defect == "replica_disagrees":
        parts = [distributed(torch.ones(4) * i, shape, Replicate()) for i in range(2)]
    elif defect == "partial":
        parts = [distributed(torch.ones(4), shape, Partial()) for _ in range(2)]
    elif defect == "nan":
        parts = [torch.full((4,), float("nan"))]
    else:
        monkeypatch.setattr(e, "MAX_SOURCE_BYTES", 4)
    with pytest.raises(ValueError):
        e.reassemble(parts, shape)


def fixture(tmp_path, *, state=None, base_dtype=torch.bfloat16, model="Qwen/Qwen3.8-27B"):
    plan, source, manifest_path = checkpoint_fixture(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    tensors = {
        key: torch.tensor([float(i)], dtype=base_dtype) for i, key in enumerate(e.FROZEN_MTP_KEYS)
    }
    tensors["weight"] = torch.arange(12, dtype=base_dtype).reshape(3, 4)
    save_file(tensors, base / "model-00001-of-00001.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors},
            }
        )
    )
    for name in e.SIDECARS:
        (base / name).write_text(json.dumps({"synthetic": name}))
    plan["model"].update(
        {
            "root": str(base),
            "repo": model,
            "files": [{"path": p.name, "sha256": digest(p)} for p in sorted(base.iterdir())],
        }
    )
    trained = {"weight": tensors["weight"].float() + 0.125} if state is None else state
    torch.save(trained, source / "policy/model_world_size_1_rank_0.pt")
    saved = tmp_path / "checkpoint_receipts/step-000002.json"
    saved.unlink()
    write_receipt(
        saved,
        {
            "plan_sha256": _unsigned_digest(plan),
            "optimizer_step": 2,
            "checkpoint_path": str(source),
        },
    )
    seal(plan, 2, manifest_path)
    return manifest_path, digest(manifest_path), tmp_path / "export", trained, tensors


def test_export_is_complete_bf16_create_once_and_preserves_inputs(tmp_path, monkeypatch):
    manifest, sha, out, trained, base = fixture(tmp_path)
    monkeypatch.setattr(e, "MAX_SHARD_BYTES", 22)
    # One tensor must fit the chosen shard bound; the real bound is 3 GiB.
    with pytest.raises(ValueError, match="exceeds the fixed shard"):
        e.export(manifest, sha, out)
    monkeypatch.setattr(e, "MAX_SHARD_BYTES", 32)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    progress = []
    result = e.export(manifest, sha, out, progress=lambda *args: progress.append(args))
    assert result["optimizer_steps_executed"] == 0 and result["gpu_reload_verified"] is False
    assert result["trained_tensors"] == 1 and len(result["restored_base_tensors"]) == 15
    assert result["dtype"] == "BF16" and result["all_output_tensors_reopened_equal"]
    assert len(list(out.glob("*.safetensors"))) > 1
    assert before == {p: p.read_bytes() for p in before}
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for key in base:
        expected = trained[key].bfloat16() if key in trained else base[key]
        assert torch.equal(e._load_tensor(out, index[key], key), expected)
    for name, spec in result["files"].items():
        assert digest(out / name) == spec["sha256"]
    assert progress[-1] == ("export_shards", 16, result["tensor_bytes"])
    with pytest.raises(FileExistsError):
        e.export(manifest, sha, out)


@pytest.mark.parametrize("defect", ["unexpected", "missing", "dtype", "nan", "base_dtype", "model"])
def test_export_rejects_unknown_or_broken_weight_contract(tmp_path, defect):
    state = {"weight": torch.ones(3, 4)}
    if defect == "unexpected":
        state["surprise"] = torch.ones(1)
    elif defect == "missing":
        state = {}
    elif defect == "dtype":
        state["weight"] = state["weight"].bfloat16()
    elif defect == "nan":
        state["weight"][0, 0] = float("nan")
    manifest, sha, out, *_ = fixture(
        tmp_path,
        state=state,
        base_dtype=torch.float32 if defect == "base_dtype" else torch.bfloat16,
        model="zai-org/GLM-5.3" if defect == "model" else "Qwen/Qwen3.8-27B",
    )
    with pytest.raises(ValueError):
        e.export(manifest, sha, out)
    assert not out.exists()


def test_bound_digest_no_gpu_and_no_source_write(tmp_path, monkeypatch):
    manifest, sha, out, *_ = fixture(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        e.export(manifest, "0" * 64, out)
    with pytest.raises(ValueError, match="inside source"):
        e.export(manifest, sha, tmp_path / "base/new")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="does not need a GPU"):
        e.export(manifest, sha, out)


def test_partial_and_concurrent_destination_are_preserved(tmp_path, monkeypatch):
    manifest, sha, out, *_ = fixture(tmp_path)
    partial = out.with_name(out.name + ".partial")
    partial.mkdir()
    with pytest.raises(FileExistsError):
        e.export(manifest, sha, out)
    partial.rmdir()
    rename = e._rename_noreplace

    def racing(source, destination):
        destination.mkdir()
        (destination / "existing").write_text("keep")
        rename(source, destination)

    monkeypatch.setattr(e, "_rename_noreplace", racing)
    with pytest.raises(FileExistsError):
        e.export(manifest, sha, out)
    assert (out / "existing").read_text() == "keep"
    assert (partial / "EXPORT.json").exists()


def test_bad_reopen_never_publishes(tmp_path, monkeypatch):
    manifest, sha, out, *_ = fixture(tmp_path)
    load = e._load_tensor

    def bad(root, shard, key):
        value = load(root, shard, key)
        return value + 1 if root.name.endswith(".partial") else value

    monkeypatch.setattr(e, "_load_tensor", bad)
    with pytest.raises(ValueError, match="write/reopen"):
        e.export(manifest, sha, out)
    assert not out.exists()


@pytest.mark.parametrize("defect", ["source_mutation", "layout", "nonfinite_base", "inventory"])
def test_post_write_integrity_gates(tmp_path, monkeypatch, defect):
    manifest, sha, out, *_ = fixture(tmp_path)
    layout = e._safetensor_layout
    load = e._load_tensor

    def changed_layout(root):
        value, shards = layout(root)
        if defect == "inventory" and root.name == "base":
            shards.append("unbound.safetensors")
        if defect == "layout" and root.name.endswith(".partial"):
            value["weight"]["shape"] = [12]
        return value, shards

    def changed_tensor(root, shard, key):
        value = load(root, shard, key)
        return value * float("nan") if defect == "nonfinite_base" and root.name == "base" else value

    def mutate(phase, *_):
        if defect == "source_mutation" and phase == "export_shards":
            (tmp_path / "checkpoints/global_step_2/data.pt").write_bytes(b"changed")

    monkeypatch.setattr(e, "_safetensor_layout", changed_layout)
    monkeypatch.setattr(e, "_load_tensor", changed_tensor)
    with pytest.raises(ValueError):
        e.export(manifest, sha, out, progress=mutate)
    assert not out.exists()


def test_mismatched_rank_keys_are_rejected(tmp_path, monkeypatch):
    manifest, sha, out, *_ = fixture(tmp_path)
    value = json.loads(manifest.read_text())
    value["world_size"] = 2
    monkeypatch.setattr(e, "receipt", lambda _: value)
    monkeypatch.setattr(e, "verify", lambda _: None)
    # Exercise key agreement after the separately tested manifest/inventory gates.
    monkeypatch.setattr(e, "checkpoint_files", lambda *_: {})
    states = iter([{"weight": torch.ones(3, 4)}, {"wrong": torch.ones(3, 4)}])
    monkeypatch.setattr(torch, "load", lambda *_, **__: next(states))
    with pytest.raises(ValueError, match="ranks disagree"):
        e.export(manifest, sha, out)
    assert not out.exists()


def test_native_skyrl_writer_exports_without_gpu(tmp_path, monkeypatch):
    native = pytest.importorskip("skyrl.backends.skyrl_train.distributed.fsdp_strategy")
    from skyrl.train.config import FSDPConfig
    from transformers import PretrainedConfig

    manifest, _, out, trained, _ = fixture(tmp_path)
    plan = json.loads(manifest.read_text())["source_plan"]
    source = tmp_path / "checkpoints/global_step_2"
    model = torch.nn.Linear(4, 3, bias=False)
    model.config = PretrainedConfig()
    model.load_state_dict(trained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 4)).sum().backward()
        optimizer.step()
    strategy = native.FSDPStrategy(FSDPConfig())
    strategy.world_size = 1
    monkeypatch.setattr(native.dist, "barrier", lambda: None)
    monkeypatch.setattr(native.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(native.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(strategy, "get_rng_state", lambda: {"torch": torch.get_rng_state()})
    strategy.save_checkpoint(model, str(source / "policy"), 0, optimizer=optimizer)
    manifest.unlink()  # This test's synthetic seal preceded the native writer.
    seal(plan, 2, manifest)
    result = e.export(manifest, digest(manifest), out)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert torch.equal(e._load_tensor(out, index["weight"], "weight"), model.weight.bfloat16())
    assert result["optimizer_steps_executed"] == 0
