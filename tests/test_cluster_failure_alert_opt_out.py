from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from training.io import digest_json

ROOT = Path(__file__).resolve().parents[1]
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
ALERT_RECEIPT = ROOT / "docs/evidence/failed-job-alert-opt-out-20260920.json"
READINESS_RECEIPT = ROOT / "configs/qualification/qwen38-27b-readiness-2026-09-01-v2.json"
QUALIFICATION = ROOT / "configs/qualification/qwen38-27b-v1.json"


def _jobs(value: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(value, dict):
        return
    if value.get("kind") in {"Job", "RayJob"}:
        yield value
    if value.get("kind") == "List":
        for item in value.get("items", []):
            yield from _jobs(item)


def _historical_manifest(path: Path) -> str:
    text = path.read_text()
    before, marker, remainder = text.partition("```yaml\n")
    manifest, closing, after = remainder.partition("```\n")
    assert marker and closing
    assert "non-launchable evidence" in before
    assert "Do not pass this file or the fenced bytes to `kubectl`" in before
    assert not after.strip()
    with pytest.raises(yaml.YAMLError):
        list(yaml.safe_load_all(text))
    return manifest


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
        for document_number, document in enumerate(yaml.safe_load_all(path.read_text()), start=1):
            for job in _jobs(document):
                annotations = job.get("metadata", {}).get("annotations", {})
                if annotations.get(FAILURE_ALERT_ANNOTATION) != "off":
                    name = job.get("metadata", {}).get("name", "<unnamed>")
                    missing.append(f"{relative_path} document {document_number}: {name}")
    assert not missing, "Jobs missing fleet.ai/failure-alerts: off:\n" + "\n".join(missing)


def test_retired_jobs_are_non_launchable_digest_bound_evidence() -> None:
    receipt = json.loads(ALERT_RECEIPT.read_text())
    embedded_digest = receipt.pop("receipt_sha256")
    assert embedded_digest == digest_json(receipt)
    assert receipt["status"] == "guard_enforced_no_launch"
    assert receipt["fail_closed_preview"] == {
        "source_repository_commit": "7a455f4470086a6edd3a6fbd919f18a082e8258b",
        "source_path": "configs/qualification/qwen38-lr30-step76-gpu-reload-v1.json",
        "source_file_sha256": (
            "sha256:1c571d6a05aa543acf0737b7da1100a115ff824896965f50898b464c0cf2465e"
        ),
        "source_receipt_sha256": (
            "sha256:841425ab2f818991ffd021666e5d5cff994f79803a43321f1e087d47d6f4a8c8"
        ),
        "performed_at_utc_date": "2026-09-21",
        "request_failure_alerts": False,
        "rendered_kind": "RayJob",
        "rendered_namespace": "fleet-train-jobs",
        "rendered_manifest_sha256": (
            "sha256:9ffbe0a4bbbf82ac5c4ab9bd100a85c7a4493f5afa7826ce49fa7e06b37b0954"
        ),
        "root_failure_alert_annotation": None,
        "qualified_for_create": False,
        "jobs_submitted": 0,
        "resources_created": 0,
        "gpus_allocated": 0,
        "raw_manifest_included": False,
    }

    readiness = json.loads(READINESS_RECEIPT.read_text())
    qualification = json.loads(QUALIFICATION.read_text())
    staging = qualification["model_staging"]
    expected_hashes = {
        readiness["staging_plan"]["stage_job_name"]: readiness["staging_plan"][
            "stage_manifest_sha256"
        ],
        readiness["staging_plan"]["canonical_link_job_name"]: readiness["staging_plan"][
            "canonical_link_manifest_sha256"
        ],
    }
    evidence_fields = {
        "chris-cyber-qwen38-stage-1d4bf0f2": "historical_stage_job_evidence",
        "chris-cyber-qwen38-canonical-link": "historical_canonical_link_job_evidence",
    }

    assert staging["status"] == "historical_not_launchable"
    assert staging["launchable"] is False
    assert receipt["successor_contract"] == {
        "reuse_historical_manifest_bytes": False,
        "new_unique_name_required": True,
        "new_render_required": True,
        "root_failure_alert_annotation": "off",
        "preview_must_prove_annotation_before_create": True,
        "patch_after_create_is_acceptable": False,
    }
    for record in receipt["retired_historical_manifests"]:
        original = ROOT / record["original_supported_path"]
        evidence = ROOT / record["evidence_path"]
        assert not original.exists()
        assert evidence.is_file() and evidence.suffix == ".md"
        assert evidence.is_relative_to(ROOT / "docs/evidence")
        assert record["launchable"] is False

        manifest_text = _historical_manifest(evidence)
        assert (
            "sha256:" + hashlib.sha256(manifest_text.encode()).hexdigest()
            == record["manifest_sha256"]
        )
        job = yaml.safe_load(manifest_text)
        assert job["kind"] == record["root_kind"] == "Job"
        assert job["metadata"]["name"] == record["job_name"]
        assert job["metadata"].get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) is None
        assert record["root_failure_alert_annotation"] is None
        assert record["manifest_sha256"] == expected_hashes[record["job_name"]]
        assert staging[evidence_fields[record["job_name"]]] == record["evidence_path"]
