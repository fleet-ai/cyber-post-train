from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals.fleet import autocontinue_flock_release as release_validator
from evals.fleet import self_hosted

RELEASE = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-flock-preflight-release-v1.json"
)
TERMINAL = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-flock-preflight-terminal-v1.json"
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
        "campaign_file_sha256",
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


def test_flock_preflight_terminal_evidence_is_exact_and_non_scored() -> None:
    evidence = release_validator.load_object(TERMINAL)
    release_validator.validate_terminal_evidence(evidence)
    assert evidence["classification"]["shared_pvc_cross_pod_flock_preflight_passed"]
    assert evidence["classification"]["scored_sessions_created"] == 0


@pytest.mark.parametrize(
    ("section", "field", "changed"),
    [
        (None, "launch_commit", "0" * 40),
        ("authorization", "release_file_sha256", "sha256:" + "0" * 64),
        ("implementation", "submit_script_sha256", "sha256:" + "0" * 64),
        ("implementation", "server_dry_run_passed", False),
        ("configmap", "name", "changed"),
        ("configmap", "uid", "00000000-0000-0000-0000-000000000000"),
        ("configmap", "immutable", False),
        ("holder", "job_name", "changed"),
        ("holder", "job_uid", "00000000-0000-0000-0000-000000000000"),
        ("prober", "pod_name", "changed"),
        ("prober", "pod_uid", "00000000-0000-0000-0000-000000000000"),
        ("terminal", "terminal_file_sha256", "sha256:" + "0" * 64),
        ("terminal", "holder_ready_sha256", "sha256:" + "0" * 64),
        ("terminal", "contention_receipt_sha256", "sha256:" + "0" * 64),
        ("terminal", "holder_released_sha256", "sha256:" + "0" * 64),
        ("terminal", "contention_excluded_second_pod", False),
        ("terminal", "slot_reacquired_after_release", False),
        ("classification", "scored_sessions_created", 1),
    ],
)
def test_flock_preflight_terminal_rejects_resealed_tampering(
    section: str | None, field: str, changed: object
) -> None:
    evidence = release_validator.load_object(TERMINAL)
    tampered = deepcopy(evidence)
    target = tampered if section is None else tampered[section]
    target[field] = changed
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="terminal evidence drifted"):
        release_validator.validate_terminal_evidence(tampered)
