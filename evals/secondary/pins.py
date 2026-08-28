"""Reviewed, immutable references to official upstream benchmark repositories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BenchmarkName = Literal["xben", "cvebench"]


@dataclass(frozen=True)
class BenchmarkPin:
    name: BenchmarkName
    repository: str
    commit: str
    release: str | None
    license_id: str
    task_count: int
    evaluation_only: bool = True

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "name": self.name,
            "repository": self.repository,
            "commit": self.commit,
            "release": self.release,
            "license_id": self.license_id,
            "task_count": self.task_count,
            "evaluation_only": self.evaluation_only,
        }


BENCHMARK_PINS: dict[BenchmarkName, BenchmarkPin] = {
    "xben": BenchmarkPin(
        name="xben",
        repository="https://github.com/xbow-engineering/validation-benchmarks.git",
        commit="a16cb1ba0701b20d63913846536fc7b0672082af",
        release=None,
        license_id="Apache-2.0",
        task_count=104,
    ),
    "cvebench": BenchmarkPin(
        name="cvebench",
        repository="https://github.com/uiuc-kang-lab/cve-bench.git",
        commit="332f0ec0fc95c4a4d35fe7458a73969aad1aa1d4",
        release="v2.1.0",
        license_id="Apache-2.0",
        task_count=40,
    ),
}
