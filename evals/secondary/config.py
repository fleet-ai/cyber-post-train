"""Secret-free configuration for the secondary external evaluations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .pins import BENCHMARK_PINS, BenchmarkName

_XBEN_TARGET = re.compile(r"^XBEN-\d{3}-24$")
_CVE_TARGET = re.compile(r"^CVE-\d{4}-\d{4,7}$")
_INTERNAL_TARGET = re.compile(r"^http://[a-zA-Z0-9_.-]+:\d{1,5}/?$")
Variant = Literal["blackbox", "zero_day", "one_day"]


@dataclass(frozen=True)
class SecondaryEvalConfig:
    benchmark: BenchmarkName
    target: str
    variant: Variant
    target_url: str | None = None
    endpoint_origin: str = "https://inference.flt.build"
    gateway_model: str = "glm-5.2-fp8"
    inspect_model: str = "openai/glm-5.2-fp8"
    pass_k: int = 1
    max_messages: int = 60
    command_timeout_seconds: int = 180
    run_id: str = "glm52-secondary-smoke-v1"

    def validate(self) -> None:
        if self.benchmark not in BENCHMARK_PINS:
            raise ValueError(f"unsupported benchmark: {self.benchmark}")
        if self.endpoint_origin != "https://inference.flt.build":
            raise ValueError("endpoint_origin must be the approved Fleet inference gateway")
        if self.gateway_model != "glm-5.2-fp8":
            raise ValueError("the baseline is pinned to glm-5.2-fp8")
        if self.inspect_model != "openai/glm-5.2-fp8":
            raise ValueError("inspect_model must use Inspect's pinned OpenAI-compatible adapter")
        if self.pass_k != 1:
            raise ValueError("external baseline smoke runs must use pass_k=1")
        if not 1 <= self.max_messages <= 300:
            raise ValueError("max_messages must be between 1 and 300")
        if not 1 <= self.command_timeout_seconds <= 900:
            raise ValueError("command timeout must be between 1 and 900 seconds")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.run_id):
            raise ValueError("run_id must be a safe lowercase identifier")

        if self.benchmark == "xben":
            if self.variant != "blackbox" or not _XBEN_TARGET.fullmatch(self.target):
                raise ValueError("XBEN requires variant=blackbox and an XBEN-NNN-24 target")
            if not self.target_url or not _INTERNAL_TARGET.fullmatch(self.target_url):
                raise ValueError("XBEN target_url must be an internal Docker HTTP service URL")
        elif self.benchmark == "cvebench":
            if self.variant not in {"zero_day", "one_day"}:
                raise ValueError("CVE-Bench requires zero_day or one_day")
            if not _CVE_TARGET.fullmatch(self.target):
                raise ValueError("CVE-Bench target must be a CVE identifier")
            if self.target_url is not None:
                raise ValueError("CVE-Bench owns its target URL inside its sandbox")

    @property
    def openai_base_url(self) -> str:
        return f"{self.endpoint_origin}/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> SecondaryEvalConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("secondary eval config must be a JSON object")
        config = cls(**raw)
        config.validate()
        return config
