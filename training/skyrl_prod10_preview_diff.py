"""Hashed-only JSON-pointer diagnosis for a prod10 Jobs preview difference."""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from cyber_post_train.jobs import API_URLS, Jobs, JobsError, digest

from . import skyrl_prod9_direct as direct
from . import skyrl_prod9_training as training
from . import skyrl_reward_rayjob as historical

PROOF_SCHEMA = "cyber_skyrl_prod10_preview_pointer_proof_v1"
SUBMITTER_PROOF_SCHEMA = "cyber_skyrl_prod10_submitter_identity_format_v1"
RESULT_SCHEMA = "cyber_skyrl_prod10_preview_difference_result_v1"
MAX_REPORTED_DIFFERENCES = 7
EXPECTED_POINTER_PROOF_SHA256 = (
    "sha256:bea833b73442d3eb6380382d1ae3f2d1cb7ff858db415686685e4d6a25744e14"
)
EXPECTED_SUBMITTER_PROOF_SHA256 = (
    "sha256:3961950bd1a645134476e471e8d34260799d8a497e7aefddf4c906ac68eb70f4"
)
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_MISSING = {"state": "missing"}

# Exact reviewed leaf inventory of the frozen prod10 RayJob. A server-added
# leaf is counted but its arbitrary path is never exported.
ALLOWED_POINTERS = (
    "/apiVersion",
    "/kind",
    "/metadata/annotations/fleet.ai~1failure-alerts",
    "/metadata/annotations/fleet.ai~1job-image",
    "/metadata/annotations/fleet.ai~1run-dir",
    "/metadata/annotations/fleet.ai~1run-id",
    "/metadata/annotations/fleet.ai~1submitted-by",
    "/metadata/annotations/fleet.ai~1submitted-by-profile",
    "/metadata/labels/app",
    "/metadata/labels/fleet.ai~1requeue-if-preempted",
    "/metadata/labels/fleet.ai~1run-id",
    "/metadata/labels/fleet.ai~1run-name",
    "/metadata/labels/kueue.x-k8s.io~1priority-class",
    "/metadata/labels/kueue.x-k8s.io~1queue-name",
    "/metadata/name",
    "/metadata/namespace",
    "/spec/entrypoint",
    "/spec/rayClusterSpec/enableInTreeAutoscaling",
    "/spec/rayClusterSpec/headGroupSpec/rayStartParams/dashboard-host",
    "/spec/rayClusterSpec/headGroupSpec/template/metadata/annotations/kueue.x-k8s.io~1podset-preferred-topology",
    "/spec/rayClusterSpec/headGroupSpec/template/metadata/labels/fleet.ai~1run-id",
    "/spec/rayClusterSpec/headGroupSpec/template/metadata/labels/fleet.ai~1run-name",
    *tuple(
        f"/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/env/{index}/{field}"
        for index in range(25)
        for field in ("name", "value")
    ),
    *tuple(
        f"/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/envFrom/{index}/secretRef/name"
        for index in range(3)
    ),
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/image",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/limits/cpu",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/limits/memory",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/limits/nvidia.com~1gpu",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/requests/cpu",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/requests/memory",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/resources/requests/nvidia.com~1gpu",
    *tuple(
        f"/spec/rayClusterSpec/headGroupSpec/template/spec/containers/0/volumeMounts/{index}/{field}"
        for index in range(3)
        for field in ("mountPath", "name")
    ),
    *tuple(
        f"/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/command/{index}"
        for index in range(3)
    ),
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/image",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/securityContext/runAsUser",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/volumeMounts/0/mountPath",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/0/volumeMounts/0/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/args/0",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/command/0",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/command/1",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/image",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/allowPrivilegeEscalation",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/capabilities/add/0",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/capabilities/add/1",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/capabilities/drop/0",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/readOnlyRootFilesystem",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/runAsNonRoot",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/securityContext/runAsUser",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/volumeMounts/0/mountPath",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/initContainers/1/volumeMounts/0/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/nodeSelector/kubernetes.io~1os",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/nodeSelector/workload",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/priorityClassName",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/securityContext/supplementalGroups/0",
    *tuple(
        f"/spec/rayClusterSpec/headGroupSpec/template/spec/tolerations/0/{field}"
        for field in ("effect", "key", "operator", "value")
    ),
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/0/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/0/persistentVolumeClaim/claimName",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/1/emptyDir/medium",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/1/emptyDir/sizeLimit",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/1/name",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/2/hostPath/path",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/2/hostPath/type",
    "/spec/rayClusterSpec/headGroupSpec/template/spec/volumes/2/name",
    "/spec/shutdownAfterJobFinishes",
    "/spec/submissionMode",
    "/spec/suspend",
)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _leaves(value: object, pointer: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {pointer: value}
        result: dict[str, object] = {}
        for key in sorted(value):
            token = str(key).replace("~", "~0").replace("/", "~1")
            result.update(_leaves(value[key], pointer + "/" + token))
        return result
    if isinstance(value, list):
        if not value:
            return {pointer: value}
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, pointer + "/" + str(index)))
        return result
    return {pointer: value}


def _kind(value: object) -> str:
    if value == _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if type(value) is int:
        return "integer"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise ValueError("preview pointer contains a non-JSON value")


def submitter_identity_validation(manifest: dict[str, Any]) -> dict[str, bool]:
    try:
        annotations = manifest["metadata"]["annotations"]
        submitted_by = annotations["fleet.ai/submitted-by"]
        profile = annotations["fleet.ai/submitted-by-profile"]
    except (KeyError, TypeError):
        return {"email_valid": False, "profile_uuid_valid": False}
    email_valid = (
        isinstance(submitted_by, str)
        and len(submitted_by) <= 254
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@[A-Za-z0-9]"
            r"(?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?",
            submitted_by,
        )
        is not None
    )
    try:
        profile_valid = isinstance(profile, str) and str(UUID(profile)) == profile
    except (TypeError, ValueError):
        profile_valid = False
    return {"email_valid": email_valid, "profile_uuid_valid": profile_valid}


def submitter_identity_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    return _seal(
        {
            "schema": SUBMITTER_PROOF_SCHEMA,
            "manifest_sha256": "sha256:" + digest(manifest),
            **submitter_identity_validation(manifest),
            "values_exported": False,
        }
    )


def validate_submitter_identity_proof(
    value: object, *, manifest_sha256: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("submitter identity proof is not an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        set(value)
        != {
            "schema",
            "manifest_sha256",
            "email_valid",
            "profile_uuid_valid",
            "values_exported",
            "sha256",
        }
        or value.get("schema") != SUBMITTER_PROOF_SCHEMA
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("email_valid") is not True
        or value.get("profile_uuid_valid") is not True
        or value.get("values_exported") is not False
        or value.get("sha256") != "sha256:" + digest(body)
        or (expected_sha256 is not None and value.get("sha256") != expected_sha256)
    ):
        raise ValueError("submitter identity proof changed")
    return value


def pointer_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("preview manifest is not an object")
    leaves = _leaves(manifest)
    allowed = set(ALLOWED_POINTERS)
    rows = [
        {
            "pointer_id": pointer,
            "value_type": _kind(leaves.get(pointer, _MISSING)),
            "value_sha256": "sha256:" + digest(leaves.get(pointer, _MISSING)),
        }
        for pointer in ALLOWED_POINTERS
    ]
    return _seal(
        {
            "schema": PROOF_SCHEMA,
            "status": "hashed_pointers_only",
            "manifest_sha256": "sha256:" + digest(manifest),
            "pointers": rows,
            "unrecognized_count": len(set(leaves) - allowed),
            "leaf_count": len(leaves),
            "values_exported": False,
            "raw_manifest_exported": False,
        }
    )


def validate_pointer_proof(value: object, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("preview pointer proof is not an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    rows = value.get("pointers")
    if (
        set(value)
        != {
            "schema",
            "status",
            "manifest_sha256",
            "pointers",
            "unrecognized_count",
            "leaf_count",
            "values_exported",
            "raw_manifest_exported",
            "sha256",
        }
        or value.get("schema") != PROOF_SCHEMA
        or value.get("status") != "hashed_pointers_only"
        or value.get("sha256") != "sha256:" + digest(body)
        or (expected_sha256 is not None and value.get("sha256") != expected_sha256)
        or _HASH.fullmatch(str(value.get("manifest_sha256", ""))) is None
        or not isinstance(rows, list)
        or [row.get("pointer_id") for row in rows if isinstance(row, dict)]
        != list(ALLOWED_POINTERS)
        or any(not isinstance(row, dict) for row in rows)
        or any(
            set(row) != {"pointer_id", "value_type", "value_sha256"}
            or row.get("value_type")
            not in {"missing", "null", "boolean", "integer", "string", "array", "object"}
            or _HASH.fullmatch(str(row.get("value_sha256", ""))) is None
            for row in rows
            if isinstance(row, dict)
        )
        or type(value.get("unrecognized_count")) is not int
        or value["unrecognized_count"] < 0
        or type(value.get("leaf_count")) is not int
        or value["leaf_count"] < 0
        or value.get("values_exported") is not False
        or value.get("raw_manifest_exported") is not False
    ):
        raise ValueError("preview pointer proof changed")
    return value


def diagnose(expected: object, live_manifest: dict[str, Any]) -> dict[str, Any]:
    left = validate_pointer_proof(expected)
    right = pointer_proof(live_manifest)
    differing = []
    for left_row, right_row in zip(left["pointers"], right["pointers"], strict=True):
        if (
            left_row["value_type"] == right_row["value_type"]
            and left_row["value_sha256"] == right_row["value_sha256"]
        ):
            continue
        differing.append(
            {
                "pointer_id": left_row["pointer_id"],
                "left_type": left_row["value_type"],
                "right_type": right_row["value_type"],
                "left_sha256": left_row["value_sha256"],
                "right_sha256": right_row["value_sha256"],
                "equal": False,
            }
        )
    unrecognized = right["unrecognized_count"]
    unreported = max(0, len(differing) - MAX_REPORTED_DIFFERENCES)
    return {
        "left_manifest_sha256": left["manifest_sha256"],
        "right_manifest_sha256": right["manifest_sha256"],
        "manifest_equal": left["manifest_sha256"] == right["manifest_sha256"],
        "differences": differing[:MAX_REPORTED_DIFFERENCES],
        "total_diff_count": len(differing) + unrecognized,
        "reported_diff_count": min(len(differing), MAX_REPORTED_DIFFERENCES),
        "unreported_allowlisted_count": unreported,
        "unrecognized_count": unrecognized,
        "safe_for_repair": unrecognized == 0 and unreported == 0,
    }


def run(
    plan: dict[str, Any],
    request: dict[str, Any],
    expected: object,
    expected_submitter_identity: object,
    image_identity_receipt: dict[str, Any],
    *,
    identity: historical.RailIdentity,
    token: str,
    launch_v9_failure_sha256: str,
    source_preview_sha256: str,
    jobs_factory: Any = Jobs,
) -> dict[str, Any]:
    direct._identity(plan, identity)
    if training.job_request(plan) != request:
        raise JobsError("preview-difference request changed")
    checked = validate_pointer_proof(expected, expected_sha256=EXPECTED_POINTER_PROOF_SHA256)
    submitter = validate_submitter_identity_proof(
        expected_submitter_identity,
        manifest_sha256=checked["manifest_sha256"],
        expected_sha256=EXPECTED_SUBMITTER_PROOF_SHA256,
    )
    if (
        not token
        or _HASH.fullmatch(launch_v9_failure_sha256) is None
        or _HASH.fullmatch(source_preview_sha256) is None
    ):
        raise JobsError("preview-difference credential is unavailable")
    with jobs_factory(token, base_url=API_URLS["prod"]) as client:
        live_source = client.preview(request)
    live_manifest = direct.manifest(
        plan,
        request,
        live_source,
        identity=identity,
        image_identity_receipt=image_identity_receipt,
    )
    live_submitter = submitter_identity_validation(live_manifest)
    if live_submitter != {"email_valid": True, "profile_uuid_valid": True}:
        raise JobsError("preview-difference submitter identity is invalid")
    difference = diagnose(expected, live_manifest)
    return _seal(
        {
            "schema": RESULT_SCHEMA,
            "status": (
                "diagnostic_completed"
                if difference["safe_for_repair"]
                else "diagnostic_fail_closed"
            ),
            "phase": "preview-diff",
            **difference,
            "launch_v9_failure_sha256": launch_v9_failure_sha256,
            "source_preview_sha256": source_preview_sha256,
            "expected_pointer_proof_sha256": checked["sha256"],
            "expected_submitter_identity_proof_sha256": submitter["sha256"],
            "submitter_identity_format": {
                "left_email_valid": True,
                "left_profile_uuid_valid": True,
                "right_email_valid": live_submitter["email_valid"],
                "right_profile_uuid_valid": live_submitter["profile_uuid_valid"],
            },
            "jobs_api_calls": {"preview": 1, "create": 0},
            "values_exported": False,
            "raw_manifests_exported": False,
            "logs_exported": False,
            "secrets_exported": False,
            "gpus": 0,
        }
    )
