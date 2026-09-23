from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from training import skyrl_prod10_operator as operator
from training import skyrl_prod10_preview_diff as preview_diff


def _manifest(
    *, email: str = "host@example.com", profile: str = "00000000-0000-4000-8000-000000000001"
) -> dict:
    return {
        "metadata": {
            "annotations": {
                "fleet.ai/submitted-by": email,
                "fleet.ai/submitted-by-profile": profile,
            }
        }
    }


def test_pointer_inventory_is_fixed_unique_and_exactly_130() -> None:
    assert len(preview_diff.ALLOWED_POINTERS) == 130
    assert len(set(preview_diff.ALLOWED_POINTERS)) == 130
    assert all(value.startswith("/") for value in preview_diff.ALLOWED_POINTERS)
    assert "/metadata/annotations/fleet.ai~1submitted-by" in preview_diff.ALLOWED_POINTERS
    assert "/metadata/annotations/fleet.ai~1submitted-by-profile" in (preview_diff.ALLOWED_POINTERS)


def test_diagnosis_exports_only_allowlisted_differing_pointer_hashes() -> None:
    left = _manifest()
    right = _manifest(email="runtime@example.com")
    result = preview_diff.diagnose(preview_diff.pointer_proof(left), right)

    assert result["manifest_equal"] is False
    assert result["total_diff_count"] == 1
    assert result["unrecognized_count"] == 0
    assert result["safe_for_repair"] is True
    assert result["differences"] == [
        {
            "pointer_id": "/metadata/annotations/fleet.ai~1submitted-by",
            "left_type": "string",
            "right_type": "string",
            "left_sha256": result["differences"][0]["left_sha256"],
            "right_sha256": result["differences"][0]["right_sha256"],
            "equal": False,
        }
    ]
    encoded = json.dumps(result, sort_keys=True)
    assert "host@example.com" not in encoded
    assert "runtime@example.com" not in encoded


def test_unallowlisted_pointer_is_counted_but_never_exported() -> None:
    left = _manifest()
    right = copy.deepcopy(left)
    right["private_dynamic_field"] = "never-export-this"
    result = preview_diff.diagnose(preview_diff.pointer_proof(left), right)

    assert result["unrecognized_count"] == 1
    assert result["safe_for_repair"] is False
    encoded = json.dumps(result, sort_keys=True)
    assert "private_dynamic_field" not in encoded
    assert "never-export-this" not in encoded


def test_run_calls_preview_once_and_has_no_create_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    left = _manifest()
    right = _manifest(email="runtime@example.com")
    calls = {"preview": 0}

    class FakeJobs:
        def __init__(self, token: str, *, base_url: str) -> None:
            assert token == "token"
            assert base_url

        def __enter__(self) -> FakeJobs:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def preview(self, request: dict) -> dict:
            calls["preview"] += 1
            return {"request": request}

    identity = SimpleNamespace()
    plan, request = {}, {"request": "exact"}
    pointer = preview_diff.pointer_proof(left)
    submitter = preview_diff.submitter_identity_proof(left)
    monkeypatch.setattr(preview_diff.direct, "_identity", lambda *_args: identity)
    monkeypatch.setattr(preview_diff.training, "job_request", lambda _plan: request)
    monkeypatch.setattr(preview_diff.direct, "manifest", lambda *_args, **_kwargs: right)
    monkeypatch.setattr(preview_diff, "EXPECTED_POINTER_PROOF_SHA256", pointer["sha256"])
    monkeypatch.setattr(preview_diff, "EXPECTED_SUBMITTER_PROOF_SHA256", submitter["sha256"])
    result = preview_diff.run(
        plan,
        request,
        pointer,
        submitter,
        {},
        identity=identity,
        token="token",
        launch_v9_failure_sha256="sha256:" + "1" * 64,
        source_preview_sha256="sha256:" + "2" * 64,
        jobs_factory=FakeJobs,
    )

    assert calls == {"preview": 1}
    assert result["jobs_api_calls"] == {"preview": 1, "create": 0}
    assert result["gpus"] == 0
    assert result["values_exported"] is False
    assert result["raw_manifests_exported"] is False
    assert result["secrets_exported"] is False
    assert result["submitter_identity_format"] == {
        "left_email_valid": True,
        "left_profile_uuid_valid": True,
        "right_email_valid": True,
        "right_profile_uuid_valid": True,
    }


def test_pointer_and_submitter_proofs_reject_resealed_semantic_drift() -> None:
    manifest = _manifest()
    pointer = preview_diff.pointer_proof(manifest)
    changed = copy.deepcopy(pointer)
    changed["pointers"][0]["value_sha256"] = "sha256:" + "0" * 64
    changed = preview_diff._seal({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="pointer proof changed"):
        preview_diff.validate_pointer_proof(changed, expected_sha256=pointer["sha256"])

    submitter = preview_diff.submitter_identity_proof(manifest)
    changed_submitter = copy.deepcopy(submitter)
    changed_submitter["email_valid"] = False
    changed_submitter = preview_diff._seal(
        {key: value for key, value in changed_submitter.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="submitter identity proof changed"):
        preview_diff.validate_submitter_identity_proof(
            changed_submitter, manifest_sha256=submitter["manifest_sha256"]
        )


def test_preview_difference_job_is_alert_off_c1_q1_zero_gpu_fleet_secret_only() -> None:
    packet = {
        "phase": "preview-diff",
        "sha256": "sha256:" + "3" * 64,
        "writable_controls_probe": False,
    }
    job = operator_job._job(
        phase="preview-diff",
        source_sha256="sha256:" + "4" * 64,
        packet=packet,
        packet_bytes=b"{}\n",
    )
    pod = job["spec"]["template"]["spec"]
    [container] = pod["containers"]
    encoded = json.dumps(job, sort_keys=True)

    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert pod["priorityClassName"] == "c1"
    assert job["spec"]["backoffLimit"] == 0
    assert pod["automountServiceAccountToken"] is False
    assert container["envFrom"] == [{"secretRef": {"name": "fleet-api"}}]
    assert "wandb-api" not in encoded
    assert "controls-rw" not in encoded
    assert "nvidia.com/gpu" not in encoded
    assert pod["volumes"][-1]["persistentVolumeClaim"]["readOnly"] is True


def test_sanitized_difference_receipt_stays_below_termination_limit() -> None:
    difference = {
        "left_manifest_sha256": "sha256:" + "1" * 64,
        "right_manifest_sha256": "sha256:" + "2" * 64,
        "manifest_equal": False,
        "differences": [
            {
                "pointer_id": pointer,
                "left_type": "string",
                "right_type": "string",
                "left_sha256": "sha256:" + "3" * 64,
                "right_sha256": "sha256:" + "4" * 64,
                "equal": False,
            }
            for pointer in sorted(preview_diff.ALLOWED_POINTERS, key=len, reverse=True)[
                : preview_diff.MAX_REPORTED_DIFFERENCES
            ]
        ],
        "total_diff_count": preview_diff.MAX_REPORTED_DIFFERENCES,
        "reported_diff_count": preview_diff.MAX_REPORTED_DIFFERENCES,
        "unreported_allowlisted_count": 0,
        "unrecognized_count": 0,
        "safe_for_repair": True,
    }
    result = preview_diff._seal(
        {
            "schema": preview_diff.RESULT_SCHEMA,
            "status": "diagnostic_completed",
            "phase": "preview-diff",
            **difference,
            "jobs_api_calls": {"preview": 1, "create": 0},
            "launch_v9_failure_sha256": "sha256:" + "5" * 64,
            "source_preview_sha256": "sha256:" + "6" * 64,
            "expected_pointer_proof_sha256": "sha256:" + "7" * 64,
            "expected_submitter_identity_proof_sha256": "sha256:" + "8" * 64,
            "submitter_identity_format": {
                "left_email_valid": True,
                "left_profile_uuid_valid": True,
                "right_email_valid": True,
                "right_profile_uuid_valid": True,
            },
            "values_exported": False,
            "raw_manifests_exported": False,
            "logs_exported": False,
            "secrets_exported": False,
            "gpus": 0,
        }
    )
    assert len((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()) < 3900
    assert operator.OPERATOR_NAMES["preview-diff"] == "chris-q38-prod10-preview-diff-v1"
