from __future__ import annotations

import copy
import subprocess

import pytest

from evals.credential_rotation import (
    CredentialRotationError,
    build_credential_rotation_receipt,
    read_secret_metadata,
    read_secret_snapshot_in_memory,
    validate_credential_rotation,
)

BEFORE = {
    "namespace": "fleet-train-jobs",
    "name": "fleet-api",
    "uid": "11111111-1111-4111-8111-111111111111",
    "resource_version": "10",
    "data_key_present": True,
}
AFTER = {
    **BEFORE,
    "uid": "22222222-2222-4222-8222-222222222222",
    "resource_version": "11",
}
SECRET = "must-not-appear-in-an-error-or-receipt"


def _receipt() -> dict:
    return build_credential_rotation_receipt(
        BEFORE,
        AFTER,
        confirmed_by="credential-owner",
        rotation_completed_at="2026-09-02T00:00:00Z",
    )


def test_receipt_is_self_digested_secret_free_and_matches_live_metadata() -> None:
    receipt = _receipt()
    assert (
        validate_credential_rotation(receipt, AFTER, rotation_not_before="2026-09-01T23:39:05Z")
        == AFTER
    )
    assert SECRET not in repr(receipt)
    assert receipt["secret_value_observed"] is False
    assert receipt["secret_value_persisted"] is False


@pytest.mark.parametrize("field", ["uid", "resource_version", "data_key_present"])
def test_live_metadata_drift_fails_closed(field: str) -> None:
    live = copy.deepcopy(AFTER)
    live[field] = False if field == "data_key_present" else "999"
    with pytest.raises(CredentialRotationError):
        validate_credential_rotation(_receipt(), live, rotation_not_before="2026-09-01T23:39:05Z")


def test_rotation_must_postdate_incident_boundary() -> None:
    with pytest.raises(CredentialRotationError, match="does not postdate"):
        validate_credential_rotation(_receipt(), AFTER, rotation_not_before="2026-09-02T00:00:00Z")


def test_metadata_reader_never_requests_secret_value() -> None:
    seen: list[str] = []

    def runner(command, **kwargs):
        del kwargs
        seen.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "fleet-train-jobs\nfleet-api\n22222222-2222-4222-8222-222222222222\n11\ntrue\n"
            ),
            stderr="",
        )

    assert read_secret_metadata(runner=runner) == AFTER
    rendered = " ".join(seen)
    assert "index .data" in rendered
    assert "true" in rendered
    assert "base64" not in rendered


def test_snapshot_reader_keeps_value_in_memory_and_returns_bound_metadata() -> None:
    encoded = "bXVzdC1ub3QtYXBwZWFyLWluLWFuLWVycm9yLW9yLXJlY2VpcHQ="

    def runner(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "fleet-train-jobs\nfleet-api\n"
                "22222222-2222-4222-8222-222222222222\n11\ntrue\n"
                f"{encoded}"
            ),
            stderr="",
        )

    metadata, secret = read_secret_snapshot_in_memory(runner=runner)
    assert metadata == AFTER
    assert secret == SECRET


def test_kubectl_error_is_sanitized() -> None:
    def runner(command, **kwargs):
        del kwargs
        raise subprocess.CalledProcessError(1, command, output=SECRET, stderr=SECRET)

    with pytest.raises(CredentialRotationError) as exc:
        read_secret_snapshot_in_memory(runner=runner)
    assert SECRET not in str(exc.value)
