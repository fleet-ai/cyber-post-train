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

AUTHORIZATION_SCHEMA = "cyber_skyrl_prod10_direct_authorization_v3"
PREFLIGHT_RESULT_SCHEMA = "cyber_skyrl_prod10_operator_result_v1"
DUPLICATE_SCHEMA = "cyber_skyrl_prod10_direct_duplicate_absence_v1"
CAPACITY_SCHEMA = "cyber_skyrl_prod10_direct_capacity_gate_v1"
CREATED_SCHEMA = "cyber_skyrl_prod10_direct_created_v1"
REVALIDATION_SCHEMA = "cyber_skyrl_prod10_preflight_revalidation_v1"
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


def _duplicate(value: object, identity: historical.RailIdentity) -> dict[str, Any]:
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


def authorize(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    preflight: dict[str, Any],
    revalidation: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
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
    previews = direct._direct_previews(
        plan, request, source_preview, expected, [dev_preview, prod_preview], identity=bound
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
            "dev_preview": previews[0],
            "prod_preview": previews[1],
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
    census: dict[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
    wandb_exists: Callable[[str, str, str], bool] = direct._wandb_exists_default,
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
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
        identity=bound,
    )
    if auth != expected_auth:
        raise JobsError("prod10 direct-v3 authorization changed")
    absence = _duplicate(duplicate, bound)
    output = Path(bound.output_root)
    if output.exists() or output.is_symlink():
        raise JobsError("prod10 output root already exists")
    arguments = plan.get("arguments", {})
    if not isinstance(arguments, dict) or wandb_exists(
        arguments.get("wandb_entity", ""),
        arguments.get("wandb_project", ""),
        arguments.get("wandb_run_id", ""),
    ):
        raise JobsError("W&B run ID already exists")
    plan_sha, request_sha, manifest_sha = (
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(expected),
    )
    with jobs_factory(token, base_url=API_URLS["prod"]) as client:
        live_source = client.preview(request)
        live_expected = direct.manifest(
            plan,
            request,
            live_source,
            identity=bound,
            image_identity_receipt=auth["image_identity_receipt"],
        )
        if live_source != source_preview or live_expected != expected:
            raise JobsError("prod10 live Jobs API preview changed")
        live_preview = direct.validate_preview(
            plan,
            request,
            live_source,
            live_expected,
            direct.server_dry_run(live_expected, context=direct.PROD_CONTEXT, runner=runner),
            context=direct.PROD_CONTEXT,
            identity=bound,
            image_identity_receipt=auth["image_identity_receipt"],
        )
        capacity = capacity_gate(plan, request, live_expected, census, identity=bound)
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
        for preview in (auth["dev_preview"], auth["prod_preview"], live_preview):
            direct._fresh_at(preview.get("checked_at"))
        direct._write_once_fsynced(
            journal,
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "plan_sha256": plan_sha,
                "request_sha256": request_sha,
                "manifest_sha256": manifest_sha,
                "authorization_sha256": auth["sha256"],
                "live_jobs_preview_sha256": "sha256:" + digest(live_source),
                "live_preview_proof": live_preview,
                "capacity_gate": capacity,
                "duplicate_checks": absence,
                "wandb_run_id_absent": True,
            },
        )
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
