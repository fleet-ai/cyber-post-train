import json
from pathlib import Path

import pytest

from evals.fleet import fixed_proxy_qualification as qualification

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/evaluation/qwen38-blackbox-fleet-dev-a-base-certification-v1.json"
SOURCE = ROOT / "evals/fleet/fixed_proxy.py"
IMAGE = "ghcr.io/astral-sh/uv@sha256:" + "a" * 64


def test_exact_policy_and_zero_gpu_pod() -> None:
    config = qualification.build_config(json.loads(PLAN.read_text()), image=IMAGE, source=SOURCE)
    pod = qualification.build_pod(
        config,
        source=SOURCE,
        name="chris-fixed-proxy-test",
        namespace="default",
        node="cpu-node",
    )
    assert config["sampling_policy"] == {
        "model": "qwen3.8-27b",
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 42,
        "max_tokens": 32768,
    }
    container = pod["spec"]["containers"][0]
    assert pod["spec"]["nodeName"] == "cpu-node"
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert container["imagePullPolicy"] == "Always"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]


def test_source_change_is_rejected(tmp_path: Path) -> None:
    config = qualification.build_config(json.loads(PLAN.read_text()), image=IMAGE, source=SOURCE)
    changed = tmp_path / "fixed_proxy.py"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        qualification.build_pod(
            config,
            source=changed,
            name="chris-fixed-proxy-test",
            namespace="default",
            node="cpu-node",
        )


def test_mutable_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="digest pinned"):
        qualification.build_config(
            json.loads(PLAN.read_text()), image="ghcr.io/astral-sh/uv:latest", source=SOURCE
        )
