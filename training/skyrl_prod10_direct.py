"""Isolated direct-v3 authorization and one-POST rail for prod10.

The completed prod10 stage is sealed to the frozen prod9 source bytes.  This
module therefore imports that reviewed rail but never changes or synthesizes
its legacy authorization schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from cyber_post_train.gpu_capacity import ROLE_LABELS
from cyber_post_train.jobs import API_URLS, Jobs, JobsError, digest

from . import skyrl_prod9_direct as direct
from . import skyrl_prod9_hardening as hardening
from . import skyrl_prod9_training as training
from . import skyrl_reward_rayjob as historical

AUTHORIZATION_SCHEMA = "cyber_skyrl_prod10_direct_authorization_v4"
PREFLIGHT_RESULT_SCHEMA = "cyber_skyrl_prod10_operator_result_v1"
DUPLICATE_SCHEMA = "cyber_skyrl_prod10_direct_duplicate_absence_v1"
JIT_DUPLICATE_SCHEMA = "cyber_skyrl_prod10_jit_duplicate_absence_v2"
CAPACITY_SCHEMA = "cyber_skyrl_prod10_direct_capacity_gate_v1"
CREATED_SCHEMA = "cyber_skyrl_prod10_direct_created_v1"
ACCEPTANCE_SCHEMA = "cyber_skyrl_prod10_terminal_acceptance_v1"
REVALIDATION_SCHEMA = "cyber_skyrl_prod10_preflight_revalidation_v1"
SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA = (
    "cyber_skyrl_prod10_sealed_external_dev_server_preview_provenance_v1"
)
SUBMITTER_NORMALIZATION_SCHEMA = "cyber_skyrl_prod10_submitter_normalization_v1"
_SUBMITTER_ANNOTATIONS = (
    "fleet.ai/submitted-by",
    "fleet.ai/submitted-by-profile",
)
MAX_NODES = 10
MAX_GPUS = 80


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return direct._seal(value)


def _submitter_identity_valid(manifest: object) -> bool:
    if not isinstance(manifest, dict):
        return False
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return False
    annotations = metadata.get("annotations")
    if not isinstance(annotations, dict):
        return False
    email = annotations.get(_SUBMITTER_ANNOTATIONS[0])
    profile = annotations.get(_SUBMITTER_ANNOTATIONS[1])
    if (
        not isinstance(email, str)
        or len(email) > 254
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}@[A-Za-z0-9]"
            r"(?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?",
            email,
        )
        is None
    ):
        return False
    try:
        return isinstance(profile, str) and str(UUID(profile)) == profile
    except (TypeError, ValueError):
        return False


def live_submitter_normalization(expected: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Allow only the two authenticated, server-owned submitter values to differ."""
    if not _submitter_identity_valid(expected) or not _submitter_identity_valid(live):
        raise JobsError("prod10 live Jobs API submitter identity is invalid")
    normalized = deepcopy(live)
    for key in _SUBMITTER_ANNOTATIONS:
        normalized["metadata"]["annotations"][key] = expected["metadata"]["annotations"][key]
    if normalized != expected:
        raise JobsError("prod10 live Jobs API manifest changed")
    return _seal(
        {
            "schema": SUBMITTER_NORMALIZATION_SCHEMA,
            "status": "two_server_owned_annotations_normalized",
            "annotations": list(_SUBMITTER_ANNOTATIONS),
            "expected_manifest_sha256": "sha256:" + digest(expected),
            "live_manifest_sha256": "sha256:" + digest(live),
            "normalized_manifest_sha256": "sha256:" + digest(normalized),
            "expected_email_valid": True,
            "expected_profile_uuid_valid": True,
            "live_email_valid": True,
            "live_profile_uuid_valid": True,
            "values_exported": False,
        }
    )


def _preflight_launch(
    value: object,
    plan: dict[str, Any],
    *,
    identity: historical.RailIdentity,
    operator_name: str,
) -> dict[str, Any]:
    """Bind the released zero-GPU producer, without treating age as truth."""
    direct._identity(plan, identity)
    launch = direct._validate_seal(value, direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA)
    package, created = launch.get("package"), launch.get("created")
    if not isinstance(package, dict) or not isinstance(created, dict):
        raise JobsError("prod10 preflight launch evidence is incomplete")
    job, source, packet = (
        created.get("job"),
        created.get("source_config_map"),
        created.get("packet_config_map"),
    )
    if not all(isinstance(item, dict) for item in (job, source, packet)):
        raise JobsError("prod10 preflight created identities are incomplete")
    release = direct._validate_seal(
        launch.get("observer"), "cyber_direct_cleanup_observer_result_v1"
    )
    receipt = direct._validate_seal(
        release.get("receipt"), direct.STAGE_OPERATOR_TERMINATION_SCHEMA
    )
    try:
        job_uid = str(UUID(job.get("uid")))
        UUID(source.get("uid"))
        UUID(packet.get("uid"))
        UUID(release.get("workload_uid"))
        for uid in release.get("pod_uids", []):
            UUID(uid)
    except (TypeError, ValueError) as exc:
        raise JobsError("prod10 preflight launch identity is invalid") from exc
    expected_path = str(hardening.training_operation_root(plan) / "PREFLIGHT_OPERATOR_RESULT.json")
    absent = (
        "target_present",
        "pods_present",
        "workload_present",
        "rayjob_present",
        "raycluster_present",
    )
    if (
        launch.get("status") != "operator_succeeded_and_released"
        or launch.get("gpus") != 0
        or package.get("name") != operator_name
        or package.get("phase") != "preflight"
        or package.get("failure_alerts") != "off"
        or package.get("priority") != "c1"
        or package.get("queue_priority") != "q1"
        or package.get("gpus") != 0
        or source.get("name") != operator_name + "-source"
        or packet.get("name") != operator_name + "-packet"
        or job.get("name") != operator_name
        or job.get("uid") != job_uid
        or job.get("manifest_sha256") != package.get("job_manifest_sha256")
        or release.get("status") != "released"
        or release.get("context") != direct.PROD_CONTEXT
        or release.get("namespace") != direct.NAMESPACE
        or release.get("kind") != "job"
        or release.get("name") != operator_name
        or release.get("uid") != job_uid
        or release.get("plan_sha256") != package.get("packet_sha256")
        or release.get("manifest_sha256") != package.get("job_manifest_sha256")
        or release.get("terminal_status") != "Succeeded"
        or release.get("exit_codes") != [0]
        or release.get("restarts") != 0
        or release.get("expected_gpus") != 0
        or release.get("peak_gpus") != 0
        or release.get("active_gpus") != 0
        or release.get("image_ids") != [training.historical.IMAGE]
        or any(release.get(key) is not False for key in absent)
        or receipt.get("status") != "passed"
        or receipt.get("phase") != "preflight"
        or receipt.get("result_path") != expected_path
        or not isinstance(receipt.get("result_sha256"), str)
        or receipt.get("gpus") != 0
    ):
        raise JobsError("prod10 preflight launch evidence changed")
    return launch


def preflight_result(
    plan: dict[str, Any],
    request: dict[str, Any],
    value: object,
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Strictly validate the immutable v3 result; launch rechecks live state."""
    bound = direct._identity(plan, identity)
    result = direct._validate_seal(value, PREFLIGHT_RESULT_SCHEMA)
    authorization = direct._validate_seal(
        result.get("authorization"), direct.PREFLIGHT_AUTHORIZATION_DIRECT_MANIFEST_SCHEMA
    )
    stage_result = authorization.get("stage_result")
    if not isinstance(stage_result, dict) or not isinstance(stage_result.get("stage"), dict):
        raise JobsError("prod10 direct-v3 preflight stage evidence is missing")
    expected_manifest = direct.preflight_job_manifest(plan, identity=bound)
    expected_authorization = direct._preflight_authorization_direct_manifest(
        plan,
        request,
        stage_result["stage"],
        stage_result,
        authorization["stage_launch_result"],
        authorization["manifest_launch_result"],
        expected_manifest,
        stage_operator_name=authorization["stage_operator_name"],
        manifest_operator_name=authorization["manifest_operator_name"],
        dev_preview=authorization["dev_preview"],
        prod_preview=authorization["prod_preview"],
        observer=authorization["observer"],
        identity=bound,
        require_live_observer=False,
        fresh_manifest_release=False,
        fresh_previews=False,
    )
    if authorization != expected_authorization:
        raise JobsError("prod10 direct-v3 preflight authorization changed")
    created = direct._cpu_created(
        result.get("created"),
        purpose="preflight",
        name=bound.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected_manifest),
        authorization_sha256=authorization["sha256"],
    )
    receipt = direct._preflight_receipt(plan, request, result.get("receipt"), identity=bound)
    release = direct._cpu_release(
        result.get("release"),
        receipt,
        name=bound.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected_manifest),
        fresh=False,
    )
    if (
        result.get("status") != "preflight_ready"
        or result.get("phase") != "preflight"
        or result.get("plan_sha256") != "sha256:" + digest(plan)
        or result.get("request_sha256") != "sha256:" + digest(request)
        or result.get("stage_result_sha256") != stage_result["sha256"]
        or result.get("stage_launch_result_sha256")
        != authorization["stage_launch_result"]["sha256"]
        or result.get("manifest_launch_result_sha256")
        != authorization["manifest_launch_result"]["sha256"]
        or result.get("authorization") != authorization
        or result.get("created") != created
        or result.get("receipt") != receipt
        or result.get("release") != release
        or release.get("uid") != created["job_uid"]
        or result.get("gpus") != 0
    ):
        raise JobsError("prod10 direct-v3 preflight result changed")
    return result


def image_identity(request: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    receipt, release = preflight.get("receipt"), preflight.get("release")
    if (
        not isinstance(receipt, dict)
        or not isinstance(release, dict)
        or receipt.get("runtime_user") != {"uid": direct.RUNTIME_UID, "gid": direct.RUNTIME_GID}
        or receipt.get("gpus") != 0
        or release.get("image_ids") != [request.get("image")]
    ):
        raise JobsError("prod10 direct-v3 exact-image evidence changed")
    body = {
        "schema": direct.IMAGE_DEFAULT_IDENTITY_SCHEMA,
        "status": "passed",
        "image": request["image"],
        "uid": direct.RUNTIME_UID,
        "gid": direct.RUNTIME_GID,
        "gpus": 0,
    }
    return {**body, "receipt_sha256": digest(body)}


def _seal_fresh_preflight_receipt(value: object) -> dict[str, Any]:
    """Apply the exact digest that the CPU Job's termination writer adds."""
    if not isinstance(value, dict) or "receipt_sha256" in value:
        raise JobsError("prod10 fresh preflight receipt body changed")
    return {**value, "receipt_sha256": digest(value)}


def revalidate_preflight(
    plan: dict[str, Any],
    request: dict[str, Any],
    preflight: dict[str, Any],
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Rerun the exact-image CPU gate; never extend the old result's TTL."""
    checked = preflight_result(plan, request, preflight, identity=identity)
    receipt = _seal_fresh_preflight_receipt(training.preflight(plan))
    receipt = direct._preflight_receipt(plan, request, receipt, identity=identity)
    if receipt != checked["receipt"]:
        raise JobsError("prod10 fresh preflight differs from the sealed result")
    return _seal(
        {
            "schema": REVALIDATION_SCHEMA,
            "status": "fresh_exact_image_preflight_passed",
            "preflight_result_sha256": checked["sha256"],
            "receipt": receipt,
            "revalidated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gpus": 0,
        }
    )


def _revalidation(
    value: object,
    plan: dict[str, Any],
    request: dict[str, Any],
    preflight: dict[str, Any],
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    checked = direct._validate_seal(value, REVALIDATION_SCHEMA)
    expected = preflight_result(plan, request, preflight, identity=identity)
    if (
        checked.get("status") != "fresh_exact_image_preflight_passed"
        or checked.get("preflight_result_sha256") != expected["sha256"]
        or checked.get("receipt") != expected["receipt"]
        or checked.get("gpus") != 0
    ):
        raise JobsError("prod10 preflight revalidation changed")
    direct._fresh_at(checked.get("revalidated_at"))
    return checked


def duplicate_proof(
    identity: historical.RailIdentity,
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    checked = direct._direct_duplicate_checks(
        identity, token=token, runner=runner, jobs_factory=jobs_factory
    )
    return _seal(
        {
            "schema": DUPLICATE_SCHEMA,
            "status": "identity_and_output_absent",
            "identity_sha256": identity.sealed_mapping()["sha256"],
            "run_name": identity.run_name,
            "output_root": identity.output_root,
            **checked,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _duplicate(
    value: object,
    identity: historical.RailIdentity,
    *,
    fresh: bool = True,
) -> dict[str, Any]:
    checked = direct._validate_seal(value, DUPLICATE_SCHEMA)
    if (
        checked.get("status") != "identity_and_output_absent"
        or checked.get("identity_sha256") != identity.sealed_mapping()["sha256"]
        or checked.get("run_name") != identity.run_name
        or checked.get("output_root") != identity.output_root
        or checked.get("kubernetes_inventories_checked") != 10
        or type(checked.get("jobs_api_rows_checked")) is not int
        or checked["jobs_api_rows_checked"] < 0
    ):
        raise JobsError("prod10 direct-v3 duplicate proof changed")
    if fresh:
        direct._fresh_at(checked.get("checked_at"))
    return checked


def jit_duplicate_proof(
    identity: historical.RailIdentity,
    host_duplicate: object,
    *,
    token: str,
    prior_duplicate: object | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    """Refresh prod state while preserving fresh sealed host dev provenance."""
    host = _duplicate(host_duplicate, identity, fresh=True)
    prior_sha256 = None
    if prior_duplicate is not None:
        prior_sha256 = _jit_duplicate(
            prior_duplicate,
            identity,
            host_duplicate,
            prior_sha256=None,
        )["sha256"]
    names = {identity.run_name, identity.stage_name, identity.preflight_name}
    output_root = identity.output_root
    inventories = 0
    for resource in ("rayjob", "raycluster", "job", "workload", "pod"):
        result = direct._kubectl(
            runner,
            direct.PROD_CONTEXT,
            "get",
            resource,
            "--output=json",
        )
        if result.returncode:
            raise JobsError("prod10 JIT Kubernetes duplicate inventory failed")
        inventories += 1
        try:
            items = json.loads(result.stdout).get("items", [])
        except (AttributeError, ValueError) as exc:
            raise JobsError("prod10 JIT Kubernetes duplicate inventory is invalid") from exc
        if not isinstance(items, list):
            raise JobsError("prod10 JIT Kubernetes duplicate inventory is invalid")
        for item in items:
            if not isinstance(item, dict):
                raise JobsError("prod10 JIT Kubernetes duplicate inventory is invalid")
            metadata = item.get("metadata", {})
            labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
            annotations = metadata.get("annotations", {}) if isinstance(metadata, dict) else {}
            values = [metadata.get("name", "")]
            if isinstance(labels, dict):
                values.extend(labels.values())
            if isinstance(annotations, dict):
                values.extend(annotations.values())
            serialized = json.dumps(item.get("spec", {}), sort_keys=True)
            if (
                any(
                    value in names or any(str(value).startswith(name + "-") for name in names)
                    for value in values
                )
                or output_root in values
                or output_root in serialized
            ):
                raise JobsError("prod10 JIT Kubernetes identity/output already exists")
    api_rows = 0
    with jobs_factory(token, base_url=API_URLS["prod"]) as client:
        rows = client.all_runs()
    api_rows += len(rows)
    for row in rows:
        name = str(row.get("name", ""))
        if (
            row.get("run_dir") == output_root
            or name in names
            or any(name.startswith(value + "-") for value in names)
        ):
            raise JobsError("prod10 JIT Jobs API history already owns this identity/output")
    output = Path(identity.output_root)
    if output.exists() or output.is_symlink():
        raise JobsError("prod10 output root already exists")
    return _seal(
        {
            "schema": JIT_DUPLICATE_SCHEMA,
            "status": "jit_identity_and_output_absent",
            "identity_sha256": identity.sealed_mapping()["sha256"],
            "run_name": identity.run_name,
            "output_root": identity.output_root,
            "fresh_host_all_context_duplicate_sha256": host["sha256"],
            "fresh_host_all_context_checked_at": host["checked_at"],
            "prior_jit_duplicate_sha256": prior_sha256,
            "fresh_host_all_context_kubernetes_inventories_checked": (
                host["kubernetes_inventories_checked"]
            ),
            "fresh_host_all_context_jobs_api_rows_checked": host["jobs_api_rows_checked"],
            "runtime_prod_kubernetes_inventories_checked": inventories,
            "runtime_jobs_api_targets_checked": ["prod"],
            "runtime_jobs_api_rows_checked": api_rows,
            "output_absent": True,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _jit_duplicate(
    value: object,
    identity: historical.RailIdentity,
    host_duplicate: object,
    *,
    prior_sha256: str | None = None,
) -> dict[str, Any]:
    host = _duplicate(host_duplicate, identity, fresh=True)
    checked = direct._validate_seal(value, JIT_DUPLICATE_SCHEMA)
    if (
        checked.get("status") != "jit_identity_and_output_absent"
        or checked.get("identity_sha256") != identity.sealed_mapping()["sha256"]
        or checked.get("run_name") != identity.run_name
        or checked.get("output_root") != identity.output_root
        or checked.get("fresh_host_all_context_duplicate_sha256") != host["sha256"]
        or checked.get("fresh_host_all_context_checked_at") != host["checked_at"]
        or checked.get("prior_jit_duplicate_sha256") != prior_sha256
        or checked.get("fresh_host_all_context_kubernetes_inventories_checked") != 10
        or checked.get("fresh_host_all_context_jobs_api_rows_checked")
        != host["jobs_api_rows_checked"]
        or checked.get("runtime_prod_kubernetes_inventories_checked") != 5
        or checked.get("runtime_jobs_api_targets_checked") != ["prod"]
        or type(checked.get("runtime_jobs_api_rows_checked")) is not int
        or checked["runtime_jobs_api_rows_checked"] < 0
        or checked.get("output_absent") is not True
    ):
        raise JobsError("prod10 JIT duplicate proof changed")
    direct._fresh_at(checked.get("checked_at"))
    return checked


def capacity_gate(
    plan: dict[str, Any],
    request: dict[str, Any],
    expected: dict[str, Any],
    census: object,
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    direct._identity(plan, identity)
    if not isinstance(census, dict):
        raise JobsError("prod10 capacity census is invalid")
    unsigned = {key: item for key, item in census.items() if key != "sha256"}
    current, projected = census.get("current"), census.get("projected")
    planned = {
        "nodes": request.get("workers"),
        "gpus": request.get("workers", 0) * request.get("gpus_per_worker", 0),
    }
    expected_scope = {
        "kubernetes_namespaces": "all",
        "owner_prefixes": list(hardening.PROJECT_OWNER_PREFIXES),
        "ownership_labels": ROLE_LABELS,
    }
    if (
        census.get("sha256") != digest(unsigned)
        or census.get("schema") != "cyber_project_gpu_capacity_census_v1"
        or census.get("scope") != expected_scope
        or census.get("limits") != {"nodes": MAX_NODES, "gpus": MAX_GPUS}
        or census.get("planned") != planned
        or census.get("qualified") is not True
        or census.get("problems") != []
        or not isinstance(current, dict)
        or current.get("role_pod_counts", {}).get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected
        != {"nodes": current["nodes"] + planned["nodes"], "gpus": current["gpus"] + planned["gpus"]}
        or projected["nodes"] > MAX_NODES
        or projected["gpus"] > MAX_GPUS
    ):
        raise JobsError("prod10 capacity census changed")
    direct._fresh_at(census.get("observed_at"), maximum_age=hardening.CAPACITY_MAX_AGE_SECONDS)
    return _seal(
        {
            "schema": CAPACITY_SCHEMA,
            "status": "passed",
            "identity_sha256": identity.sealed_mapping()["sha256"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest(expected),
            "planned": planned,
            "observed_at": census["observed_at"],
            "capacity_census": census,
        }
    )


def _preview_binding(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    value: object,
    *,
    context: str,
    identity: historical.RailIdentity,
    fresh: bool,
) -> dict[str, Any]:
    checked = direct._validate_seal(value, direct.PREVIEW_SCHEMA)
    if (
        checked.get("status") != "passed"
        or checked.get("context") != context
        or checked.get("plan_sha256") != digest(plan)
        or checked.get("request_sha256") != digest(request)
        or checked.get("manifest_sha256") != digest(expected)
        or checked.get("name") != source_preview.get("name")
        or checked.get("run_name_prefix") != identity.run_name
        or checked.get("nodes") != 1
        or checked.get("gpus") != 8
        or checked.get("priority") != "c1"
        or checked.get("queue_priority") != "q1"
        or checked.get("failure_alerts") != "off"
        or checked.get("submitted") is not False
    ):
        raise JobsError("prod10 direct-v3 server preview changed")
    if fresh:
        direct._fresh_at(checked.get("checked_at"))
    return checked


def sealed_dev_preview_provenance(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    dev_preview: dict[str, Any],
    *,
    image_identity_receipt: dict[str, Any],
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Validate the sealed host dev render at time of use without redating it."""
    bound = direct._identity(plan, identity)
    sealed_dev = _preview_binding(
        plan,
        request,
        source_preview,
        expected,
        dev_preview,
        context=direct.DEV_CONTEXT,
        identity=bound,
        fresh=True,
    )
    if expected != direct.manifest(
        plan,
        request,
        source_preview,
        identity=bound,
        image_identity_receipt=image_identity_receipt,
    ):
        raise JobsError("prod10 sealed dev Jobs manifest changed")
    return _seal(
        {
            "schema": SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA,
            "status": "fresh_sealed_external_dev_server_preview_validated",
            "context": direct.DEV_CONTEXT,
            "identity_sha256": bound.sealed_mapping()["sha256"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "source_preview_sha256": "sha256:" + digest(source_preview),
            "manifest_sha256": "sha256:" + digest(expected),
            "sealed_dev_server_preview_sha256": sealed_dev["sha256"],
            "failure_alerts": "off",
            "priority": "c1",
            "queue_priority": "q1",
            "gpus": 8,
            "submitted": False,
            # Preserve the server proof's timestamp.  Validation must never
            # manufacture a new freshness window inside the production Pod.
            "checked_at": sealed_dev["checked_at"],
        }
    )


def _sealed_dev_preview_provenance(
    value: object,
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    dev_preview: dict[str, Any],
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    bound = direct._identity(plan, identity)
    sealed_dev = _preview_binding(
        plan,
        request,
        source_preview,
        expected,
        dev_preview,
        context=direct.DEV_CONTEXT,
        identity=bound,
        fresh=False,
    )
    checked = direct._validate_seal(value, SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA)
    if (
        checked.get("status") != "fresh_sealed_external_dev_server_preview_validated"
        or checked.get("context") != direct.DEV_CONTEXT
        or checked.get("identity_sha256") != bound.sealed_mapping()["sha256"]
        or checked.get("plan_sha256") != "sha256:" + digest(plan)
        or checked.get("request_sha256") != "sha256:" + digest(request)
        or checked.get("source_preview_sha256") != "sha256:" + digest(source_preview)
        or checked.get("manifest_sha256") != "sha256:" + digest(expected)
        or checked.get("sealed_dev_server_preview_sha256") != sealed_dev["sha256"]
        or checked.get("failure_alerts") != "off"
        or checked.get("priority") != "c1"
        or checked.get("queue_priority") != "q1"
        or checked.get("gpus") != 8
        or checked.get("submitted") is not False
    ):
        raise JobsError("prod10 sealed dev server-preview provenance changed")
    return checked


def authorize(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    preflight: dict[str, Any],
    revalidation: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    dev_provenance: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    bound = direct._identity(plan, identity)
    checked = preflight_result(plan, request, preflight, identity=bound)
    fresh = _revalidation(revalidation, plan, request, checked, identity=bound)
    image = image_identity(request, checked)
    if expected != direct.manifest(
        plan, request, source_preview, identity=bound, image_identity_receipt=image
    ):
        raise JobsError("prod10 direct-v3 GPU manifest changed")
    checked_dev = _preview_binding(
        plan,
        request,
        source_preview,
        expected,
        dev_preview,
        context=direct.DEV_CONTEXT,
        identity=bound,
        fresh=False,
    )
    provenance = _sealed_dev_preview_provenance(
        dev_provenance,
        plan,
        request,
        source_preview,
        expected,
        checked_dev,
        identity=bound,
    )
    checked_prod = _preview_binding(
        plan,
        request,
        source_preview,
        expected,
        prod_preview,
        context=direct.PROD_CONTEXT,
        identity=bound,
        fresh=True,
    )
    plan_sha, manifest_sha = "sha256:" + digest(plan), "sha256:" + digest(expected)
    root = hardening.training_operation_root(plan)
    armed = direct._jobs_api_prefix_guard(
        observer,
        operation_root=root,
        purpose="training",
        request=request,
        plan_sha256=plan_sha,
        manifest_sha256=manifest_sha,
        gpus=request["workers"] * request["gpus_per_worker"],
        maximum_seconds=direct.MAXIMUM_SECONDS,
    )
    direct._canonical_operation_root(root)
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_jobs_api_post",
            "plan_sha256": plan_sha,
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": manifest_sha,
            "preflight_result": checked,
            "preflight_revalidation": fresh,
            "image_identity_receipt": image,
            "dev_preview": checked_dev,
            "sealed_dev_preview_provenance": provenance,
            "prod_preview": checked_prod,
            "observer": armed,
            "operation_root": str(root),
            "output_absence_enforcement": "sealed_preflight_plus_fresh_launch_revalidation",
        }
    )


def create_once(
    directory: Path,
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    token: str,
    identity: historical.RailIdentity,
    duplicate: dict[str, Any],
    final_duplicate: dict[str, Any],
    host_duplicate: dict[str, Any],
    census: dict[str, Any],
    live_source_preview: dict[str, Any],
    live_preview: dict[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    """Revalidate live state, journal intent, and perform exactly one API POST."""
    bound = direct._identity(plan, identity)
    root = hardening.training_operation_root(plan)
    journal = root / "PROD10_DIRECT_V3_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("prod10 create intent exists; reconcile, never retry")
    auth = direct._validate_seal(authorization, AUTHORIZATION_SCHEMA)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve() != root
        or Path(auth.get("operation_root", "")) != root
    ):
        raise JobsError("prod10 create directory differs from its sealed operation root")
    expected_auth = authorize(
        plan,
        request,
        source_preview,
        expected,
        auth["preflight_result"],
        auth["preflight_revalidation"],
        dev_preview=auth["dev_preview"],
        dev_provenance=auth["sealed_dev_preview_provenance"],
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
        identity=bound,
    )
    if auth != expected_auth:
        raise JobsError("prod10 direct-v3 authorization changed")
    live_manifest = direct.manifest(
        plan,
        request,
        live_source_preview,
        identity=bound,
        image_identity_receipt=auth["image_identity_receipt"],
    )
    submitter_normalization = live_submitter_normalization(expected, live_manifest)
    absence_before_guard = _jit_duplicate(duplicate, bound, host_duplicate)
    absence_before_intent = _jit_duplicate(
        final_duplicate,
        bound,
        host_duplicate,
        prior_sha256=absence_before_guard["sha256"],
    )
    expected_wandb = {
        "entity": plan["arguments"].get("wandb_entity"),
        "project": plan["arguments"].get("wandb_project"),
        "run_id": bound.wandb_run_id,
        "resume": "never",
    }
    if (
        not os.environ.get("WANDB_API_KEY")
        or auth["preflight_result"]["receipt"].get("wandb_create_once") != expected_wandb
    ):
        raise JobsError("prod10 W&B runtime create-once binding changed")
    wandb_runtime = {
        "credential_present": True,
        "wandb_create_once": expected_wandb,
        "remote_lookup_performed": False,
        "enforced_by": "training.skyrl_training.ScalarTracking.wandb.init",
    }
    plan_sha, request_sha, manifest_sha = (
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(expected),
    )
    checked_live_preview = _preview_binding(
        plan,
        request,
        source_preview,
        live_manifest,
        live_preview,
        context=direct.PROD_CONTEXT,
        identity=bound,
        fresh=True,
    )
    capacity = capacity_gate(plan, request, expected, census, identity=bound)
    direct._jobs_api_prefix_guard(
        auth["observer"],
        operation_root=root,
        purpose="training",
        request=request,
        plan_sha256=plan_sha,
        manifest_sha256=manifest_sha,
        gpus=request["workers"] * request["gpus_per_worker"],
        maximum_seconds=direct.MAXIMUM_SECONDS,
    )
    for preview in (auth["prod_preview"], checked_live_preview):
        direct._fresh_at(preview.get("checked_at"))
    direct._write_once_fsynced(
        journal,
        {
            "state": "POST_INTENT_DO_NOT_RETRY",
            "plan_sha256": plan_sha,
            "request_sha256": request_sha,
            "manifest_sha256": manifest_sha,
            "authorization_sha256": auth["sha256"],
            "sealed_jobs_preview_sha256": "sha256:" + digest(source_preview),
            "live_jobs_preview_sha256": "sha256:" + digest(live_source_preview),
            "submitter_normalization": submitter_normalization,
            "live_preview_proof": checked_live_preview,
            "capacity_gate": capacity,
            "duplicate_checks_before_guard": absence_before_guard,
            "duplicate_checks_before_intent": absence_before_intent,
            "fresh_host_all_context_duplicate_sha256": absence_before_intent[
                "fresh_host_all_context_duplicate_sha256"
            ],
            "wandb_runtime_create_once": wandb_runtime,
        },
    )
    with jobs_factory(token, base_url=API_URLS["prod"]) as client:
        response = client.request("POST", "/v1/runs", json=request)
    try:
        name, run_id = response["name"], str(UUID(response["job_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("prod10 Jobs API response is ambiguous; reconcile, never retry") from exc
    if (
        not isinstance(name, str)
        or re.fullmatch(re.escape(request["name"]) + r"-[a-f0-9]{8}", name) is None
        or response.get("run_dir") not in (None, request["run_dir"])
    ):
        raise JobsError("prod10 Jobs API returned another identity; reconcile, never retry")
    normalized = {
        "name": name,
        "job_id": run_id,
        "run_dir": response.get("run_dir") or request["run_dir"],
        "status": response.get("status"),
        "created_at": response.get("created_at"),
    }
    with journal.open("a") as stream:
        stream.write(
            json.dumps(
                {"state": "POST_RESPONSE", **normalized}, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    binding = direct._bind_jobs_api_created(
        operation_root=root,
        purpose="training",
        request=request,
        plan_sha256=plan_sha,
        manifest_sha256=manifest_sha,
        maximum_seconds=direct.MAXIMUM_SECONDS,
        expected_gpus=request["workers"] * request["gpus_per_worker"],
        response=normalized,
        runner=runner,
    )
    proof = _seal(
        {
            "schema": CREATED_SCHEMA,
            "status": "submitted_once_and_bound_exact_uid",
            "plan_sha256": plan_sha,
            "request_sha256": request_sha,
            "manifest_sha256": manifest_sha,
            "authorization_sha256": auth["sha256"],
            "sealed_jobs_preview_sha256": "sha256:" + digest(source_preview),
            "live_jobs_preview_sha256": "sha256:" + digest(live_source_preview),
            "submitter_normalization_sha256": submitter_normalization["sha256"],
            "live_preview_proof_sha256": live_preview["sha256"],
            "capacity_gate_sha256": capacity["sha256"],
            "jit_duplicate_before_guard_sha256": absence_before_guard["sha256"],
            "jit_duplicate_before_intent_sha256": absence_before_intent["sha256"],
            "jobs_api_run_name": name,
            "jobs_api_run_id": run_id,
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "creator_binding_sha256": binding["sha256"],
            "created_at": binding["rayjob_created_at"],
            "failure_alerts": binding["failure_alerts"],
            "priority": "c1",
            "queue_priority": "q1",
            "nodes": request["workers"],
            "gpus": request["workers"] * request["gpus_per_worker"],
        }
    )
    with journal.open("a") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def terminal_paths(plan: dict[str, Any]) -> dict[str, Path]:
    """Return the exact prod10 training evidence paths plus the proven reload paths."""
    paths = hardening.terminal_paths(plan)
    operation_root = hardening.training_operation_root(plan)
    return {
        **paths,
        "training_observer": operation_root / "PROD10_EXACT_OBSERVER_RESULT.json",
        "training_create_journal": operation_root / "PROD10_DIRECT_V3_CREATE.jsonl",
    }


def _training_create_journal(
    path: Path,
    *,
    plan_sha256: str,
    request_sha256: str,
    creator: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate the immutable prod10 three-row create journal without rewriting it."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect prod10 Jobs API create journal")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("prod10 Jobs API create journal changed while being read")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prod10 Jobs API create journal is not valid JSONL") from exc
    if len(rows) != 3 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("prod10 Jobs API create journal must contain exactly three rows")
    intent, response, created = rows
    capacity = intent.get("capacity_gate")
    preview = intent.get("live_preview_proof")
    if not isinstance(capacity, dict) or not isinstance(preview, dict):
        raise ValueError("prod10 create journal lacks capacity or preview evidence")
    try:
        capacity = direct._validate_seal(capacity, CAPACITY_SCHEMA)
        preview = direct._validate_seal(preview, direct.PREVIEW_SCHEMA)
        created = direct._validate_seal(created, CREATED_SCHEMA)
    except JobsError as exc:
        raise ValueError("prod10 create journal evidence schema/digest mismatch") from exc
    census = capacity.get("capacity_census")
    current = census.get("current") if isinstance(census, dict) else None
    projected = census.get("projected") if isinstance(census, dict) else None
    try:
        UUID(response["job_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("prod10 create journal response lacks an exact ID") from exc
    if (
        intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("plan_sha256") != plan_sha256
        or intent.get("request_sha256") != request_sha256
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("manifest_sha256"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("authorization_sha256"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("live_jobs_preview_sha256"))) is None
        or response.get("state") != "POST_RESPONSE"
        or response.get("name") != creator["jobs_api_run_name"]
        or response.get("job_id") != creator["jobs_api_run_id"]
        or response.get("run_dir") != creator["run_dir"]
        or created.get("status") != "submitted_once_and_bound_exact_uid"
        or created.get("plan_sha256") != plan_sha256
        or created.get("request_sha256") != request_sha256
        or created.get("manifest_sha256") != intent["manifest_sha256"]
        or created.get("authorization_sha256") != intent["authorization_sha256"]
        or created.get("live_jobs_preview_sha256") != intent["live_jobs_preview_sha256"]
        or created.get("jobs_api_run_name") != creator["jobs_api_run_name"]
        or created.get("jobs_api_run_id") != creator["jobs_api_run_id"]
        or created.get("rayjob_name") != creator["rayjob_name"]
        or created.get("rayjob_uid") != creator["rayjob_uid"]
        or created.get("creator_binding_sha256") != creator["sha256"]
        or created.get("created_at") != creator["rayjob_created_at"]
        or created.get("failure_alerts") != "off"
        or created.get("priority") != "c1"
        or created.get("queue_priority") != "q1"
        or created.get("nodes") != 1
        or created.get("gpus") != 8
        or capacity.get("status") != "passed"
        or capacity.get("plan_sha256") != plan_sha256
        or capacity.get("request_sha256") != request_sha256
        or capacity.get("manifest_sha256") != intent["manifest_sha256"]
        or capacity.get("planned") != {"nodes": 1, "gpus": 8}
        or created.get("capacity_gate_sha256") != capacity["sha256"]
        or not isinstance(census, dict)
        or census.get("sha256")
        != digest({key: item for key, item in census.items() if key != "sha256"})
        or census.get("qualified") is not True
        or census.get("problems") != []
        or census.get("planned") != {"nodes": 1, "gpus": 8}
        or census.get("limits") != {"nodes": MAX_NODES, "gpus": MAX_GPUS}
        or not isinstance(current, dict)
        or not isinstance(projected, dict)
        or current.get("role_pod_counts", {}).get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected != {"nodes": current["nodes"] + 1, "gpus": current["gpus"] + 8}
        or projected["nodes"] > MAX_NODES
        or projected["gpus"] > MAX_GPUS
        or preview.get("status") != "passed"
        or preview.get("context") != direct.PROD_CONTEXT
        or preview.get("plan_sha256") != plan_sha256.removeprefix("sha256:")
        or preview.get("request_sha256") != request_sha256.removeprefix("sha256:")
        or preview.get("manifest_sha256") != intent["manifest_sha256"].removeprefix("sha256:")
        or preview.get("failure_alerts") != "off"
        or preview.get("priority") != "c1"
        or preview.get("queue_priority") != "q1"
        or preview.get("nodes") != 1
        or preview.get("gpus") != 8
        or created.get("live_preview_proof_sha256") != preview["sha256"]
    ):
        raise ValueError("prod10 create journal/CREATED/capacity binding is incomplete")
    return created, hashlib.sha256(payload).hexdigest(), capacity


def accept_terminal(
    plan: dict[str, Any],
    *,
    checkpoint_manifest: Path,
    export: Path,
    training_observer: Path,
    training_creator_binding: Path,
    training_create_journal: Path,
    gpu_check: Path,
    reload_observer: Path,
    reload_creator_binding: Path,
    reload_create_journal: Path,
    output: Path,
) -> dict[str, Any]:
    """Accept only exact prod10 create evidence and the shared reload/release proof."""
    from . import skyrl_posttrain, skyrl_prod9_reload
    from .sft_runtime import write_receipt

    args = skyrl_posttrain._validate_plan(plan)
    paths = terminal_paths(plan)
    supplied = {
        "checkpoint_manifest": checkpoint_manifest,
        "export": export,
        "training_observer": training_observer,
        "training_creator_binding": training_creator_binding,
        "training_create_journal": training_create_journal,
        "gpu_check": gpu_check,
        "reload_observer": reload_observer,
        "accepted": output,
    }
    if any(Path(value) != paths[name] for name, value in supplied.items()):
        raise ValueError("terminal acceptance paths differ from the exact prod10 handoff")
    if output.exists() or output.is_symlink():
        raise FileExistsError("prod10 terminal acceptance already exists")
    manifest, manifest_file_sha256 = hardening._receipt_file(checkpoint_manifest)
    skyrl_posttrain.verify_manifest(manifest)
    if (
        manifest.get("source_plan_sha256") != digest(plan)
        or manifest.get("optimizer_step") != args.steps
        or manifest.get("optimizer_update_verified") is not True
        or manifest.get("source_inputs_unchanged") is not True
    ):
        raise ValueError("terminal checkpoint seal differs from the exact run")
    export_file_sha256 = hardening._hash(export)
    export_receipt, _ = hardening._inspect_export(export, export_file_sha256)
    if (
        export_receipt.get("source_checkpoint_receipt_sha256") != manifest["receipt_sha256"]
        or export_receipt.get("source_manifest_file_sha256") != manifest_file_sha256
        or export_receipt.get("source_plan_sha256") != digest(plan)
        or export_receipt.get("optimizer_step") != args.steps
        or export_receipt.get("optimizer_steps_executed") != 0
        or export_receipt.get("gpu_reload_verified") is not False
        or export_receipt.get("output_root") != str(export.parent)
    ):
        raise ValueError("BF16 export differs from the exact terminal checkpoint")
    reload_spec = skyrl_prod9_reload.build_spec(
        plan,
        manifest,
        export_receipt,
        checkpoint_manifest_file_sha256=manifest_file_sha256,
        export_file_sha256=export_file_sha256,
    )
    reload_operation = hardening.reload_operation_root(reload_spec)
    if (
        reload_creator_binding != hardening.creator_binding_path(reload_operation, "reload")
        or reload_create_journal != reload_operation / "PROD9_RELOAD_RAYJOB_CREATE.jsonl"
    ):
        raise ValueError("reload create evidence paths differ from the exact handoff")
    creator, creator_file_sha256 = hardening._exact_creator_binding(
        training_creator_binding,
        run_name_prefix=plan["run_name"],
        run_dir=plan["output_root"],
        expected_gpus=8,
        maximum_seconds=direct.MAXIMUM_SECONDS,
    )
    created, journal_file_sha256, capacity = _training_create_journal(
        training_create_journal,
        plan_sha256="sha256:" + digest(plan),
        request_sha256="sha256:" + digest(training.job_request(plan)),
        creator=creator,
    )
    candidate = skyrl_posttrain._json(training_observer)
    training_receipt = candidate.get("receipt")
    receipt_body = (
        {key: item for key, item in training_receipt.items() if key != "sha256"}
        if isinstance(training_receipt, dict)
        else {}
    )
    if (
        not isinstance(training_receipt, dict)
        or training_receipt.get("sha256") != digest(receipt_body)
        or training_receipt.get("status") != "native_loop_returned"
        or training_receipt.get("plan_sha256") != digest(plan)
        or training_receipt.get("checkpoint_global_step") != manifest["optimizer_step"]
        or training_receipt.get("completed_batches") != len(skyrl_posttrain._expected_batches(args))
        or training_receipt.get("optimizer_update_independently_verified") is not False
        or training_receipt.get("checkpoint_reload_verified") is not False
        or manifest.get("terminal_receipt_sha256") != training_receipt["sha256"]
    ):
        raise ValueError("training observer termination receipt differs from the exact run")
    training_release, training_observer_file_sha256 = hardening._exact_observer(
        training_observer,
        creator=creator,
        expected_gpus=8,
        expected_receipt=training_receipt,
    )
    gpu, gpu_file_sha256 = hardening._receipt_file(gpu_check)
    if (
        gpu.get("schema") != "cyber_hf_export_check_v1"
        or gpu.get("status") != "passed"
        or gpu.get("export_sha256") != export_file_sha256
        or gpu.get("export_receipt_sha256") != export_receipt["receipt_sha256"]
        or gpu.get("checkpoint_manifest_file_sha256") != manifest_file_sha256
        or gpu.get("checkpoint_receipt_sha256") != manifest["receipt_sha256"]
        or gpu.get("source_plan_sha256") != digest(plan)
        or gpu.get("model_repo") != plan["model"]["repo"]
        or gpu.get("model_revision") != plan["model"]["revision"]
        or gpu.get("checker_sha256") != hardening._hash(Path(skyrl_prod9_reload.__file__))
        or gpu.get("optimizer_steps_executed") != 0
        or gpu.get("gpus") != 1
        or gpu.get("gpu_reload_verified") is not True
        or gpu.get("source_unchanged") is not True
        or gpu.get("finite_logits") is not True
        or gpu.get("generated_tokens") != 2
        or gpu.get("serving_qualified") is not False
    ):
        raise ValueError("one-GPU BF16 reload proof is incomplete")
    reload_request = skyrl_prod9_reload.job_request(reload_spec)
    reload_creator, reload_creator_file_sha256 = hardening._exact_creator_binding(
        reload_creator_binding,
        run_name_prefix=reload_request["name"],
        run_dir=reload_spec["run_dir"],
        expected_gpus=1,
        maximum_seconds=skyrl_prod9_reload.MAXIMUM_SECONDS,
    )
    reload_created, reload_journal_file_sha256, reload_capacity = hardening._create_journal(
        reload_create_journal,
        created_schema=skyrl_prod9_reload.CREATED_SCHEMA,
        plan_sha256="sha256:" + digest(plan),
        request_sha256="sha256:" + digest(reload_request),
        creator=reload_creator,
        expected_gpus=1,
    )
    if reload_created.get("spec_sha256") != reload_spec["sha256"]:
        raise ValueError("reload CREATED receipt differs from the exact reload specification")
    observer, observer_file_sha256 = hardening._exact_observer(
        reload_observer,
        creator=reload_creator,
        expected_gpus=1,
        expected_receipt=gpu,
    )
    result = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "source_plan_sha256": digest(plan),
        "checkpoint_manifest_file_sha256": manifest_file_sha256,
        "checkpoint_manifest_receipt_sha256": manifest["receipt_sha256"],
        "export_file_sha256": export_file_sha256,
        "export_receipt_sha256": export_receipt["receipt_sha256"],
        "training_creator_binding_file_sha256": creator_file_sha256,
        "training_creator_binding_receipt_sha256": creator["sha256"],
        "training_create_journal_file_sha256": journal_file_sha256,
        "training_created_receipt_sha256": created["sha256"],
        "training_capacity_gate_sha256": capacity["sha256"],
        "training_observer_file_sha256": training_observer_file_sha256,
        "training_observer_receipt_sha256": training_release["sha256"],
        "training_manifest_sha256": created["manifest_sha256"],
        "training_rayjob_name": training_release["rayjob_name"],
        "training_rayjob_uid": training_release["rayjob_uid"],
        "training_workload_name": training_release["workloads"][0]["name"],
        "training_workload_uid": training_release["workloads"][0]["uid"],
        "training_raycluster_name": training_release["rayclusters"][0]["name"],
        "training_raycluster_uid": training_release["rayclusters"][0]["uid"],
        "training_pod_names": [row["name"] for row in training_release["pods"]],
        "training_pod_uids": [row["uid"] for row in training_release["pods"]],
        "gpu_check_file_sha256": gpu_file_sha256,
        "gpu_check_receipt_sha256": gpu["receipt_sha256"],
        "reload_checker_file_sha256": gpu["checker_sha256"],
        "reload_observer_file_sha256": observer_file_sha256,
        "reload_observer_receipt_sha256": observer["sha256"],
        "reload_creator_binding_file_sha256": reload_creator_file_sha256,
        "reload_creator_binding_receipt_sha256": reload_creator["sha256"],
        "reload_create_journal_file_sha256": reload_journal_file_sha256,
        "reload_created_receipt_sha256": reload_created["sha256"],
        "reload_capacity_gate_sha256": reload_capacity["sha256"],
        "reload_rayjob_uid": observer["rayjob_uid"],
        "reload_pod_uid": observer["pods"][0]["uid"],
        "optimizer_step": args.steps,
        "optimizer_updates_verified": True,
        "terminal_checkpoint_sealed": True,
        "complete_bf16_reload_verified": True,
        "gpu_resources_released": True,
        "private_payloads_included": False,
        "serving_qualified": False,
    }
    write_receipt(output, result)
    accepted, _ = hardening._receipt_file(output)
    return accepted
