"""CPU-only logical-device regression for the teacher's first-forward failure."""

import importlib.util
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("skyrl") is None, reason="requires pinned training image; CPU-only"
)
def test_rope_device_failure_and_native_initial_backload(monkeypatch):
    import torch
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from torch._subclasses.fake_tensor import FakeTensorMode

    # FakeTensorMode represents distinct devices without allocating any GPU.
    # The separate real-CPU native test checks bitwise value/dtype preservation.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cuda:0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with FakeTensorMode():
        module = torch.nn.Module()
        module.register_buffer("inv_freq", torch.empty(1, 2, 1, device="cpu"), persistent=False)
        positions = torch.empty(1, 1, 3, device="cuda:0")
        with pytest.raises(RuntimeError, match="device"):
            torch.bmm(module.inv_freq, positions)
        strategy = SimpleNamespace(manual_offload=True, manual_offload_optimizer=False)
        FSDPStrategy.backload_to_gpu(
            strategy, module, None, backload_optimizer=False, backload_model=True
        )
        output = torch.bmm(module.inv_freq, positions)
        assert output.device == positions.device
        assert output.shape == (1, 2, 3)
        assert "inv_freq" not in module.state_dict()
