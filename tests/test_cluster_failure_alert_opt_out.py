from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
FROZEN_HISTORICAL_MANIFESTS = {
    Path("cluster/jobs/chris-cyber-qwen38-canonical-link.yaml"),
    Path("cluster/jobs/chris-cyber-qwen38-stage-1d4bf0f2.yaml"),
}


def _jobs(value: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(value, dict):
        return
    if value.get("kind") in {"Job", "RayJob"}:
        yield value
    if value.get("kind") == "List":
        for item in value.get("items", []):
            yield from _jobs(item)


def test_every_committed_cluster_job_disables_failed_job_alerts() -> None:
    missing: list[str] = []
    manifests = sorted(
        {
            *list((ROOT / "cluster").rglob("*.yaml")),
            *list((ROOT / "cluster").rglob("*.yml")),
            *list((ROOT / "evals").rglob("*.yaml")),
            *list((ROOT / "evals").rglob("*.yml")),
        }
    )
    for path in manifests:
        relative_path = path.relative_to(ROOT)
        if relative_path in FROZEN_HISTORICAL_MANIFESTS:
            # These bytes are bound into a self-digested 2026-09-01 qualification receipt.
            # They are historical evidence, not reusable submission templates.
            continue
        for document_number, document in enumerate(yaml.safe_load_all(path.read_text()), start=1):
            for job in _jobs(document):
                annotations = job.get("metadata", {}).get("annotations", {})
                if annotations.get(FAILURE_ALERT_ANNOTATION) != "off":
                    name = job.get("metadata", {}).get("name", "<unnamed>")
                    missing.append(f"{relative_path} document {document_number}: {name}")
    assert not missing, "Jobs missing fleet.ai/failure-alerts: off:\n" + "\n".join(missing)
