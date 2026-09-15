import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PATCHER = (
    ROOT
    / "training/images/miles-opencode-long-context/patch_qwen3_asr_docstrings.py"
)
SPEC = importlib.util.spec_from_file_location("qwen3_asr_patch", PATCHER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture() -> bytes:
    filler = "x"
    source = MODULE._MODEL_FORWARD + filler + MODULE._CONDITIONAL_FORWARD
    return source.encode()


def test_patch_is_exact_and_refuses_source_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _fixture()
    expected = raw.decode().replace(
        MODULE._MODEL_FORWARD, MODULE._MODEL_FORWARD_PATCHED
    ).replace(MODULE._CONDITIONAL_FORWARD, MODULE._CONDITIONAL_FORWARD_PATCHED).encode()
    monkeypatch.setattr(MODULE, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(MODULE, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    assert MODULE.patch(raw) == expected
    with pytest.raises(ValueError, match="unexpected"):
        MODULE.patch(raw + b"drift")


def test_pinned_bridge_source_and_patched_digests_are_fixed() -> None:
    assert MODULE.SOURCE_SHA256 == (
        "bfdb9bff47ea68a1e72924d7f7ea83a16bde921c731298f1cd346d99c4b46b40"
    )
    assert MODULE.PATCHED_SHA256 == (
        "90ebb373c06195ad4a8d117e17ed154ffc67c78ccc370fe7f5dcab78ae864355"
    )
    dockerfile = (PATCHER.parent / "Dockerfile").read_text()
    assert "python /tmp/patch_qwen3_asr_docstrings.py" in dockerfile
