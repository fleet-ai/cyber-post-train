"""Isolated direct-v3 authorization and one-POST rail for prod10.

The completed prod10 stage is sealed to the frozen prod9 source bytes.  This
module therefore imports that reviewed rail but never changes or synthesizes
its legacy authorization schema.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
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
REVALIDATION_SCHEMA = "cyber_skyrl_prod10_preflight_revalidation_v1"
SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA = (
    "cyber_skyrl_prod10_sealed_external_dev_server_preview_provenance_v1"
)
MAX_NODES = 10
MAX_GPUS = 80


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return direct._seal(value)


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
    if expected != direct.manifest(
        plan,
        request,
        live_source_preview,
        identity=bound,
        image_identity_receipt=auth["image_identity_receipt"],
    ):
        raise JobsError("prod10 live Jobs API manifest changed")
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
        expected,
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
            "live_jobs_preview_sha256": "sha256:" + digest(live_source_preview),
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
