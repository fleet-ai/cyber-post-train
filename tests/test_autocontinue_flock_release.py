from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import autocontinue_flock_release as release_validator
from evals.fleet import self_hosted

RELEASE = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-flock-preflight-release-v1.json"
)


def test_flock_preflight_release_is_exact_and_non_scored() -> None:
    release = release_validator.load_object(RELEASE)
    release_validator.validate_release(release)
    assert release["preflight_launch_authorized"] is True
    assert release["scored_launch_authorized"] is False
    assert release["model_or_fleet_api_calls"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "campaign_sha256",
        "manifest_sha256",
        "source_sha256",
        "job_creation_order",
        "sfs_root",
        "required_priority_class",
        "create_once",
        "require_server_dry_run",
    ],
)
def test_flock_preflight_release_rejects_resealed_tampering(field: str) -> None:
    release = json.loads(RELEASE.read_text())
    release[field] = "changed"
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    with pytest.raises(ValueError, match="release drifted"):
        release_validator.validate_release(release)
