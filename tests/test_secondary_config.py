from __future__ import annotations

import json

import pytest

from evals.secondary.config import SecondaryEvalConfig


def test_loads_xben_smoke_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": "xben",
                "target": "XBEN-001-24",
                "variant": "blackbox",
                "target_url": "http://target_service:80",
            }
        ),
        encoding="utf-8",
    )
    config = SecondaryEvalConfig.load(path)
    assert config.gateway_model == "glm-5.2-fp8"
    assert config.inspect_model == "openai/glm-5.2-fp8"
    assert config.openai_base_url == "https://inference.flt.build/v1"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"endpoint_origin": "https://example.com"}, "approved Fleet"),
        ({"gateway_model": "glm-5.3"}, "pinned"),
        ({"pass_k": 2}, "pass_k=1"),
        ({"target_url": "https://public.example:443"}, "internal Docker"),
    ],
)
def test_rejects_protocol_drift(overrides, message) -> None:
    values = {
        "benchmark": "xben",
        "target": "XBEN-001-24",
        "variant": "blackbox",
        "target_url": "http://target:80",
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        SecondaryEvalConfig(**values).validate()


def test_cvebench_requires_zero_or_one_day() -> None:
    config = SecondaryEvalConfig(
        benchmark="cvebench",
        target="CVE-2024-2624",
        variant="blackbox",
    )
    with pytest.raises(ValueError, match="zero_day or one_day"):
        config.validate()
