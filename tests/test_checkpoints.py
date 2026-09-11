import copy
import json

import pytest
import torch
from test_sft_runtime import plan

from training import checkpoints as c
from training.sft_runtime import _unsigned_digest, write_receipt


def fixture(tmp_path):
    p = plan(tmp_path)
    p.pop("plan_sha256")
    p["recipe"]["nodes"] = p["recipe"]["gpus_per_node"] = 1
    root = tmp_path / "checkpoints/global_step_2"
    policy = root / "policy"
    (policy / "huggingface").mkdir(parents=True)
    (policy / "huggingface/config.json").write_text("{}")
    (policy / "fsdp_config.json").write_text(json.dumps({"fsdp_strategy": "fsdp", "world_size": 1}))
    torch.save({"global_step": 2}, root / "trainer_state.pt")
    torch.save({"_num_yielded": 2}, root / "data.pt")
    for kind in ("model", "optim", "extra_state"):
        torch.save({"synthetic": torch.ones(2)}, policy / f"{kind}_world_size_1_rank_0.pt")
    write_receipt(
        tmp_path / "checkpoint_receipts/step-000002.json",
        {"plan_sha256": _unsigned_digest(p), "optimizer_step": 2, "checkpoint_path": str(root)},
    )
    return p, root, tmp_path / "sealed.json"


def test_seal_is_complete_create_once_and_preserves_source(tmp_path):
    p, root, out = fixture(tmp_path)
    before = {str(x): x.read_bytes() for x in root.rglob("*") if x.is_file()}
    progress = []
    result = c.seal(p, 2, out, progress=lambda count, size: progress.append((count, size)))
    c.verify(result)
    assert result["source_plan_sha256"] == _unsigned_digest(p)
    assert result["gpu_reload_verified"] is False
    assert result["sampler_batches_in_epoch"] == 2
    assert progress[-1] == (7, result["total_bytes"])
    assert before == {str(x): x.read_bytes() for x in root.rglob("*") if x.is_file()}
    with pytest.raises(FileExistsError):
        c.seal(p, 2, out)


@pytest.mark.parametrize(
    "defect", ["missing", "empty", "extra", "symlink", "topology", "step", "sampler", "receipt"]
)
def test_bad_checkpoint_rejected_before_handoff(tmp_path, defect):
    p, root, out = fixture(tmp_path)
    file = root / "policy/model_world_size_1_rank_0.pt"
    if defect == "missing":
        file.unlink()
    elif defect == "empty":
        file.write_bytes(b"")
    elif defect == "extra":
        (root / "policy/model_world_size_2_rank_0.pt").write_bytes(b"partial")
    elif defect == "symlink":
        file.unlink()
        file.symlink_to(root / "data.pt")
    elif defect == "topology":
        (root / "policy/fsdp_config.json").write_text("{}")
    elif defect == "step":
        torch.save({"global_step": 1}, root / "trainer_state.pt")
    elif defect == "sampler":
        torch.save({"_num_yielded": 1}, root / "data.pt")
    else:
        p["recipe"]["lr"] = 2e-6
    with pytest.raises(ValueError):
        c.seal(p, 2, out)
    assert not out.exists()


def test_change_during_hashing_and_tamper_after_sealing_are_detected(tmp_path):
    p, root, out = fixture(tmp_path)

    def mutate(count, size):
        if count == 7:
            (root / "policy/huggingface/config.json").write_text('{"changed":true}')

    with pytest.raises(ValueError, match="changed while"):
        c.seal(p, 2, out, progress=mutate)
    result = c.seal(p, 2, out)
    (root / "policy/huggingface/config.json").write_text("{}")
    with pytest.raises(ValueError, match="digest/size"):
        c.verify(result)


@pytest.mark.parametrize(
    "defect",
    [
        "plan",
        "step",
        "topology",
        "path",
        "traversal",
        "size",
        "digest",
        "file_digest",
        "cursor",
        "reload",
        "missing",
        "aliased_path",
    ],
)
def test_manifest_cross_bindings_checked_even_without_files(tmp_path, defect):
    p, root, out = fixture(tmp_path)
    result = copy.deepcopy(c.seal(p, 2, out))
    if defect == "plan":
        result["source_plan_sha256"] = "0" * 64
    elif defect == "step":
        result["optimizer_step"] = 0
    elif defect == "topology":
        result["world_size"] = 2
    elif defect == "path":
        result["checkpoint_path"] = "/other/global_step_2"
    elif defect == "traversal":
        result["files"]["../escape"] = result["files"].pop("data.pt")
    elif defect == "size":
        result["total_bytes"] += 1
    elif defect == "file_digest":
        result["files"]["data.pt"]["sha256"] = "not-a-digest"
    elif defect == "cursor":
        result["sampler_batches_in_epoch"] = 0
    elif defect == "reload":
        result["gpu_reload_verified"] = True
    elif defect == "missing":
        result["total_bytes"] -= result["files"].pop("data.pt")["bytes"]
    elif defect == "aliased_path":
        result["files"]["policy/huggingface/./config.json"] = result["files"].pop(
            "policy/huggingface/config.json"
        )
    else:
        result["receipt_sha256"] = "0" * 64
    if defect != "digest":
        result["receipt_sha256"] = _unsigned_digest(
            {k: v for k, v in result.items() if k != "receipt_sha256"}
        )
    with pytest.raises(ValueError):
        c.verify(result, check_files=False)


def test_never_writes_manifest_inside_checkpoint(tmp_path):
    p, root, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="source checkpoint"):
        c.seal(p, 2, root / "manifest.json")


def test_cpu_only_and_create_once_receipt(tmp_path, monkeypatch):
    p, _, out = fixture(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="does not need a GPU"):
        c.seal(p, 2, out)
    out.write_text("[]")
    with pytest.raises(ValueError, match="receipt digest"):
        c.receipt(out)


def test_missing_roots_config_symlink_receipts_and_invalid_steps(tmp_path):
    p, root, out = fixture(tmp_path)
    with pytest.raises(ValueError, match="real directory"):
        c.checkpoint_files(root / "absent", 1)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "checkpoint_receipts/step-000002.json")
    with pytest.raises(ValueError, match="symlink receipt"):
        c.receipt(link)
    with pytest.raises(ValueError, match="outside the frozen plan"):
        c.seal(p, 0, out)
    (root / "policy/huggingface/config.json").unlink()
    with pytest.raises(ValueError, match="missing native model configuration"):
        c.seal(p, 2, out)


def test_manifest_checks_positive_sizes_and_added_files(tmp_path):
    p, root, out = fixture(tmp_path)
    value = c.seal(p, 2, out)
    c.verify(value, check_files=False)
    broken = copy.deepcopy(value)
    broken["total_bytes"] -= broken["files"]["data.pt"]["bytes"]
    broken["files"]["data.pt"]["bytes"] = 0
    broken["receipt_sha256"] = _unsigned_digest(
        {k: v for k, v in broken.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="invalid checkpoint file size"):
        c.verify(broken, check_files=False)
    (root / "policy/huggingface/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        c.verify(value)


def test_checkpoint_seal_public_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from cyber_post_train import cli

    p, _, out = fixture(tmp_path)
    monkeypatch.setattr(cli, "_prepared", lambda directory: (p, {}))
    result = CliRunner().invoke(
        cli.app, ["checkpoint-seal", str(tmp_path), "2", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.stdout)) == {"optimizer_step", "total_bytes", "receipt_sha256"}
    assert (
        CliRunner()
        .invoke(cli.app, ["checkpoint-seal", str(tmp_path), "2", "--output", str(out)])
        .exit_code
        == 2
    )


def test_native_producer_metadata_is_accepted(tmp_path, monkeypatch):
    """Run the real pinned writer, stubbing only distributed/CUDA synchronization."""
    native = pytest.importorskip("skyrl.backends.skyrl_train.distributed.fsdp_strategy")
    from skyrl.train.config import FSDPConfig
    from transformers import PretrainedConfig

    p, root, out = fixture(tmp_path)
    model = torch.nn.Linear(2, 2)
    model.config = PretrainedConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    strategy = native.FSDPStrategy(FSDPConfig())
    strategy.world_size = 1
    monkeypatch.setattr(native.dist, "barrier", lambda: None)
    monkeypatch.setattr(native.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(native.torch.cuda, "synchronize", lambda: None)
    # No CUDA state exists in this CPU fixture; production uses the native RNG getter.
    monkeypatch.setattr(strategy, "get_rng_state", lambda: {"torch": torch.get_rng_state()})
    strategy.save_checkpoint(model, str(root / "policy"), 0, optimizer=optimizer)
    c.verify(c.seal(p, 2, out))
