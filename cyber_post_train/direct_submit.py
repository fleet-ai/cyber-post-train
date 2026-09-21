"""Fail-closed, create-once SFT fallback when the Jobs API omits one annotation.

The Jobs API remains the rendering authority.  This module accepts its live
preview, changes only the run identity, removes the API-only Fleet credential
Secret that SFT does not consume, and adds the project-required root alert
annotation.  It never calls the Jobs API create endpoint and never applies or
patches a Kubernetes object.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .jobs import (
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    JobsError,
    digest,
    validate_preview,
    validate_request,
)

NAMESPACE = "fleet-train-jobs"
ZERO_RUN_ID = "00000000-0000-0000-0000-000000000000"
SFT_SCHEMAS = {"cyber_sft_runtime_v2", "cyber_sft_runtime_dense_v1"}
SFT_SECRET = "wandb-api"
DIRECT_JOURNAL = "DIRECT_SUBMISSION.jsonl"
CPU_CHECKPOINT_OPERATION_ANNOTATION = "cyber-post-train.fleet.ai/cpu-checkpoint-operation"
CPU_CHECKPOINT_OPERATIONS = {"seal", "verify"}
CPU_NODE_SELECTOR = {
    "kubernetes.io/arch": "amd64",
    "workload": "fleetai-training-ng-cpu",
}


def validate_cpu_checkpoint_pod(manifest: dict) -> dict:
    """Reject GPU use and host-specific placement for a CPU checkpoint operation."""
    try:
        metadata = manifest["metadata"]
        spec = manifest["spec"]
        annotations = metadata["annotations"]
        containers = spec["containers"]
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed CPU checkpoint Pod") from exc
    if manifest.get("apiVersion") != "v1" or manifest.get("kind") != "Pod":
        raise JobsError("CPU checkpoint operation must be one v1 Pod")
    if metadata.get("namespace") != NAMESPACE or not isinstance(metadata.get("name"), str):
        raise JobsError("CPU checkpoint Pod namespace/name drift")
    if annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF:
        raise JobsError("CPU checkpoint Pod must opt out of failed-job alerts before create")
    if annotations.get(CPU_CHECKPOINT_OPERATION_ANNOTATION) not in CPU_CHECKPOINT_OPERATIONS:
        raise JobsError("CPU checkpoint Pod must name a supported seal/verify operation")
    if spec.get("priorityClassName") != "c1":
        raise JobsError("CPU checkpoint Pod must use c1 priority")
    if spec.get("nodeSelector") != CPU_NODE_SELECTOR:
        raise JobsError("CPU checkpoint Pod must select only the shared CPU pool and architecture")
    if spec.get("nodeName") is not None or spec.get("affinity") is not None:
        raise JobsError("CPU checkpoint Pod must not pin one host or add placement affinity")
    if spec.get("restartPolicy") != "Never":
        raise JobsError("CPU checkpoint Pod must use restartPolicy Never")
    if not isinstance(containers, list) or not containers:
        raise JobsError("CPU checkpoint Pod has no containers")
    for container in [*spec.get("initContainers", []), *containers]:
        if not isinstance(container, dict):
            raise JobsError("CPU checkpoint Pod contains a malformed container")
        resources = container.get("resources", {})
        if not isinstance(resources, dict):
            raise JobsError("CPU checkpoint Pod resources are malformed")
        for field in ("requests", "limits"):
            values = resources.get(field, {})
            if not isinstance(values, dict):
                raise JobsError("CPU checkpoint Pod resource quantities are malformed")
            if "nvidia.com/gpu" in values:
                raise JobsError("CPU checkpoint Pod must not request or limit GPUs")
    return manifest


def _templates(obj: dict) -> list[tuple[str, dict]]:
    try:
        cluster = obj["spec"]["rayClusterSpec"]
        values = [("head", cluster["headGroupSpec"]["template"])]
        values.extend(
            (f"worker[{index}]", group["template"])
            for index, group in enumerate(cluster.get("workerGroupSpecs", []))
        )
        return values
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed RayJob pod templates") from exc


def _env(container: dict) -> dict[str, str]:
    entries = container.get("env", [])
    if not isinstance(entries, list):
        raise JobsError("preview container environment is not a list")
    result: dict[str, str] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or set(entry) != {"name", "value"}
            or not isinstance(entry.get("value"), str)
            or entry["name"] in result
        ):
            raise JobsError("preview has ambiguous runtime environment")
        result[entry["name"]] = entry["value"]
    return result


def _secret_names(container: dict) -> list[str]:
    entries = container.get("envFrom", [])
    if not isinstance(entries, list):
        raise JobsError("preview container envFrom is not a list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"secretRef"}:
            raise JobsError("preview contains an unsupported envFrom source")
        ref = entry["secretRef"]
        if (
            not isinstance(ref, dict)
            or set(ref) != {"name"}
            or not isinstance(ref.get("name"), str)
        ):
            raise JobsError("preview Secret reference is not exact and required")
        result.append(ref["name"])
    if len(result) != len(set(result)):
        raise JobsError("preview has duplicate Secret references")
    return result


def _assert_sft_contract(plan: dict, request: dict) -> None:
    validate_request(request)
    if plan.get("schema") not in SFT_SCHEMAS:
        raise JobsError("direct fallback is restricted to the qualified SFT runtime")
    if request.get("secrets") != [SFT_SECRET]:
        raise JobsError("SFT direct fallback requires exactly the W&B Secret")
    if any("fleet" in name.lower() for name in request["secrets"]):
        raise JobsError("SFT request unexpectedly requires a Fleet credential Secret")
    if request.get("priority_class") != "c1":
        raise JobsError("direct fallback is restricted to current c1/q1 policy")


def _expected_generated_env(request: dict, placeholder_name: str) -> dict[str, str]:
    return {
        "FLEET_EXTERNAL_RAY": "1",
        "FLEET_GPUS_PER_WORKER": str(request["gpus_per_worker"]),
        "FLEET_RUN_ID": ZERO_RUN_ID,
        "FLEET_RUN_NAME": placeholder_name,
        "FLEET_TRACE_ROOT": "/mnt/fleet/trajectory-spool",
        "RAY_memory_usage_threshold": "0.98",
        "RUN_DIR": request["run_dir"],
        "SKYPILOT_NUM_GPUS_PER_NODE": str(request["gpus_per_worker"]),
        "WORKERS": str(request["workers"]),
    }


def _walk_strings(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.extend(_walk_strings(item, (*path, key)))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_walk_strings(item, (*path, index)))
        return result
    return [(path, value)] if isinstance(value, str) else []


def _parse_preview(preview: dict) -> dict:
    if preview.get("errors") or preview.get("warnings"):
        raise JobsError("preview reported errors/warnings; direct rendering stopped")
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed Jobs API preview") from exc
    if not isinstance(obj, dict):
        raise JobsError("Jobs API preview is not one Kubernetes object")
    return obj


def render_sft_rayjob(
    plan: dict, request: dict, preview: dict, *, run_id: str | None = None
) -> tuple[dict, dict]:
    """Return one reviewed RayJob and a sanitized proof; create nothing."""
    _assert_sft_contract(plan, request)
    source = _parse_preview(preview)
    obj = deepcopy(source)
    expected_placeholder_name = request["name"] + "-00000000"
    try:
        meta = obj["metadata"]
        if obj.get("apiVersion") != "ray.io/v1" or obj.get("kind") != "RayJob":
            raise JobsError("preview is not a ray.io/v1 RayJob")
        if meta.get("name") != expected_placeholder_name or meta.get("namespace") != NAMESPACE:
            raise JobsError("preview placeholder name/namespace drift")
        if meta.get("generateName") is not None or any(
            key in meta for key in ("uid", "resourceVersion", "creationTimestamp")
        ):
            raise JobsError("preview unexpectedly contains persisted object identity")
        labels = meta["labels"]
        annotations = meta["annotations"]
        if labels.get("fleet.ai/run-id") != ZERO_RUN_ID:
            raise JobsError("preview root run ID is not the zero placeholder")
        if labels.get("fleet.ai/run-name") != request["name"]:
            raise JobsError("preview root run-name label drift")
        if annotations.get("fleet.ai/run-id") != ZERO_RUN_ID:
            raise JobsError("preview root run ID annotation drift")
        if FAILURE_ALERT_ANNOTATION in annotations:
            raise JobsError("preview already carries an alert setting; use the normal API rail")
        if annotations.get("fleet.ai/job-image") != request["image"]:
            raise JobsError("preview root image annotation drift")
        if annotations.get("fleet.ai/run-dir") != request["run_dir"]:
            raise JobsError("preview root output annotation drift")
        if not annotations.get("fleet.ai/submitted-by") or not re.fullmatch(
            r"[a-f0-9-]{36}", annotations.get("fleet.ai/submitted-by-profile", "")
        ):
            raise JobsError("preview submitter identity is missing")
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed Jobs API preview identity") from exc

    template_records = _templates(obj)
    expected_env = {
        **request.get("env", {}),
        **_expected_generated_env(request, expected_placeholder_name),
    }
    generated_secret = expected_placeholder_name + "-fleet-key"
    removed = 0
    for group_name, template in template_records:
        try:
            pod_labels = template["metadata"]["labels"]
            if pod_labels.get("fleet.ai/run-id") != ZERO_RUN_ID:
                raise JobsError(f"{group_name} template run ID drift")
            if pod_labels.get("fleet.ai/run-name") != request["name"]:
                raise JobsError(f"{group_name} template run-name drift")
            containers = template["spec"]["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError(f"{group_name} must contain exactly one SFT container")
            container = containers[0]
            if _env(container) != expected_env:
                raise JobsError(f"{group_name} generated environment drift")
            names = _secret_names(container)
            if names != [SFT_SECRET, generated_secret]:
                raise JobsError(f"{group_name} Secret injection drift")
            container["envFrom"] = [
                entry
                for entry in container["envFrom"]
                if entry["secretRef"]["name"] != generated_secret
            ]
            removed += 1
        except (KeyError, TypeError) as exc:
            raise JobsError(f"malformed {group_name} SFT template") from exc

    raw_strings = _walk_strings(source)
    allowed_old_name_paths = {("metadata", "name")}
    allowed_zero_paths = {
        ("metadata", "labels", "fleet.ai/run-id"),
        ("metadata", "annotations", "fleet.ai/run-id"),
    }
    for group_index, (_, template) in enumerate(template_records):
        # Resolve the original path from object identity instead of assuming a
        # worker exists for one-node previews.
        prefix: tuple[Any, ...]
        if group_index == 0:
            prefix = ("spec", "rayClusterSpec", "headGroupSpec", "template")
        else:
            prefix = (
                "spec",
                "rayClusterSpec",
                "workerGroupSpecs",
                group_index - 1,
                "template",
            )
        allowed_zero_paths.add((*prefix, "metadata", "labels", "fleet.ai/run-id"))
        # Environment ordering is platform-owned, so find the identity entries.
        entries = template["spec"]["containers"][0]["env"]
        for env_index, entry in enumerate(entries):
            if entry["name"] == "FLEET_RUN_ID":
                allowed_zero_paths.add(
                    (*prefix, "spec", "containers", 0, "env", env_index, "value")
                )
            if entry["name"] == "FLEET_RUN_NAME":
                allowed_old_name_paths.add(
                    (*prefix, "spec", "containers", 0, "env", env_index, "value")
                )

    actual_zero_paths = {path for path, value in raw_strings if value == ZERO_RUN_ID}
    actual_old_name_paths = {
        path for path, value in raw_strings if value == expected_placeholder_name
    }
    if actual_zero_paths != allowed_zero_paths or actual_old_name_paths != allowed_old_name_paths:
        raise JobsError("preview placeholder identity appeared at an unreviewed location")
    if sum(value == generated_secret for _, value in raw_strings) != len(template_records):
        raise JobsError("preview generated Fleet Secret appeared at an unreviewed location")

    selected_run_id = run_id or str(uuid.uuid4())
    uuid4_pattern = r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}"
    if not re.fullmatch(uuid4_pattern, selected_run_id):
        raise JobsError("direct run identity must be a lowercase UUIDv4")
    run_name = request["name"] + "-" + selected_run_id[:8]
    meta["name"] = run_name
    meta["labels"]["fleet.ai/run-id"] = selected_run_id
    meta["annotations"]["fleet.ai/run-id"] = selected_run_id
    meta["annotations"][FAILURE_ALERT_ANNOTATION] = FAILURE_ALERT_OFF
    for _, template in template_records:
        template["metadata"]["labels"]["fleet.ai/run-id"] = selected_run_id
        entries = template["spec"]["containers"][0]["env"]
        for entry in entries:
            if entry["name"] == "FLEET_RUN_ID":
                entry["value"] = selected_run_id
            elif entry["name"] == "FLEET_RUN_NAME":
                entry["value"] = run_name

    final_strings = _walk_strings(obj)
    if any(
        value in {ZERO_RUN_ID, expected_placeholder_name, generated_secret}
        for _, value in final_strings
    ):
        raise JobsError("placeholder identity or generated Fleet Secret survived rendering")
    if removed != len(template_records):
        raise JobsError("generated Fleet Secret removal count drift")

    validated = validate_preview(
        request,
        {"manifest_yaml": yaml.safe_dump(obj, sort_keys=False), "warnings": [], "errors": []},
    )
    return obj, {
        **validated,
        "preview_manifest_sha256": digest(source),
        "manifest_sha256": digest(obj),
        "run_id": selected_run_id,
        "name": run_name,
        "removed_api_fleet_secrets": removed,
    }


def _json_object(payload: str, operation: str) -> dict:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JobsError(f"kubectl {operation} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise JobsError(f"kubectl {operation} returned a non-object")
    return value


class Kubectl:
    """Small no-shell Kubernetes boundary; mutation is exactly one create."""

    def __init__(self, context: str, *, binary: str = "kubectl"):
        valid_context = (
            isinstance(context, str)
            and context
            and not context.startswith("-")
            and re.fullmatch(r"[-A-Za-z0-9_.:@/]+", context)
        )
        if not valid_context:
            raise JobsError("an explicit valid Kubernetes context is required")
        if not binary or binary.startswith("-"):
            raise JobsError("invalid kubectl binary")
        self.context = context
        self.binary = binary

    def _run(self, args: list[str], *, manifest: dict | None = None) -> dict:
        command = [self.binary, "--context", self.context, "--request-timeout=60s", *args]
        try:
            result = subprocess.run(
                command,
                input=(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    if manifest is not None
                    else None
                ),
                text=True,
                capture_output=True,
                timeout=75,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise JobsError("kubectl transport failed; do not infer cluster state") from None
        if result.returncode:
            raise JobsError("kubectl operation failed; private server output suppressed")
        return _json_object(result.stdout, args[0])

    def list(self, resource: str) -> dict:
        if resource not in {"rayjobs.ray.io", "jobs.batch"}:
            raise JobsError("unsupported duplicate-check resource")
        return self._run(["get", resource, "--namespace", NAMESPACE, "--output=json"])

    def dry_run(self, manifest: dict) -> dict:
        return self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=manifest,
        )

    def create_once(self, manifest: dict) -> dict:
        return self._run(["create", "--filename=-", "--output=json"], manifest=manifest)

    def dry_run_cpu_checkpoint_pod(self, manifest: dict) -> dict:
        """Server-preview one CPU seal/verifier after the local placement gate."""
        validate_cpu_checkpoint_pod(manifest)
        return self._run(
            ["create", "--dry-run=server", "--filename=-", "--output=json"],
            manifest=manifest,
        )

    def create_cpu_checkpoint_pod_once(self, manifest: dict) -> dict:
        """Create exactly one locally validated CPU seal/verifier Pod."""
        validate_cpu_checkpoint_pod(manifest)
        return self._run(["create", "--filename=-", "--output=json"], manifest=manifest)


def _assert_api_unique(rows: list[dict], request: dict) -> None:
    if not isinstance(rows, list):
        raise JobsError("Jobs API history is incomplete")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise JobsError("Jobs API history contains an invalid record")
        if (
            row.get("run_dir") == request["run_dir"]
            or row["name"] == request["name"]
            or row["name"].startswith(request["name"] + "-")
        ):
            raise JobsError("a Jobs API run already owns this name/output")


def _assert_kubernetes_unique(inventories: list[dict], request: dict, proof: dict) -> None:
    for inventory in inventories:
        items = inventory.get("items")
        kind = inventory.get("kind")
        if not isinstance(kind, str) or not kind.endswith("List") or not isinstance(items, list):
            raise JobsError("Kubernetes duplicate inventory is incomplete")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
                raise JobsError("Kubernetes duplicate inventory contains an invalid object")
            meta = item["metadata"]
            name = meta.get("name", "")
            labels = meta.get("labels") or {}
            annotations = meta.get("annotations") or {}
            if (
                not isinstance(name, str)
                or not isinstance(labels, dict)
                or not isinstance(annotations, dict)
            ):
                raise JobsError("Kubernetes duplicate inventory contains invalid metadata")
            if (
                name == proof["name"]
                or name == request["name"]
                or name.startswith(request["name"] + "-")
                or labels.get("fleet.ai/run-name") == request["name"]
                or labels.get("fleet.ai/run-id") == proof["run_id"]
                or annotations.get("fleet.ai/run-id") == proof["run_id"]
                or annotations.get("fleet.ai/run-dir") == request["run_dir"]
            ):
                raise JobsError("a Kubernetes object already owns this name/output/run identity")


def _assert_created_identity(obj: dict, proof: dict, *, require_uid: bool) -> None:
    try:
        meta = obj["metadata"]
        if (
            obj["apiVersion"] != "ray.io/v1"
            or obj["kind"] != "RayJob"
            or meta["name"] != proof["name"]
            or meta["namespace"] != NAMESPACE
            or meta["labels"]["fleet.ai/run-id"] != proof["run_id"]
            or meta["annotations"]["fleet.ai/run-id"] != proof["run_id"]
            or meta["annotations"][FAILURE_ALERT_ANNOTATION] != FAILURE_ALERT_OFF
        ):
            raise JobsError("Kubernetes response identity differs from create intent")
        if require_uid and not re.fullmatch(r"[a-f0-9-]{36}", meta.get("uid", "")):
            raise JobsError("created RayJob response omitted its immutable UID")
    except (KeyError, TypeError) as exc:
        raise JobsError("malformed Kubernetes create response") from exc


def _write_intent(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise JobsError("direct-create journal already exists; reconcile, never retry")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append_journal(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def direct_submit_sft_once(
    *,
    plan: dict,
    request: dict,
    jobs: Any,
    kubectl: Kubectl,
    journal: Path,
    run_id: str | None = None,
) -> dict:
    """Preview, prove, journal and issue exactly one direct Kubernetes create."""
    _assert_sft_contract(plan, request)
    from training.sft import job_request

    if job_request(plan) != request:
        raise JobsError("saved SFT request differs from the current source-bound renderer")
    if journal.exists() or journal.is_symlink():
        raise JobsError("direct-create journal already exists; reconcile, never retry")

    _assert_api_unique(jobs.all_runs(), request)
    preview = jobs.raw_preview(request)
    manifest, proof = render_sft_rayjob(plan, request, preview, run_id=run_id)
    _assert_kubernetes_unique(
        [kubectl.list("rayjobs.ray.io"), kubectl.list("jobs.batch")], request, proof
    )
    server_object = kubectl.dry_run(manifest)
    _assert_created_identity(server_object, proof, require_uid=False)

    # Close the read/dry-run race as far as the two authorities permit.  The
    # exact-name Kubernetes create remains the final atomic create-once gate.
    _assert_api_unique(jobs.all_runs(), request)
    _assert_kubernetes_unique(
        [kubectl.list("rayjobs.ray.io"), kubectl.list("jobs.batch")], request, proof
    )
    _write_intent(
        journal,
        {
            "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
            "request_sha256": digest(request),
            "plan_sha256": digest(plan),
            "preview_manifest_sha256": proof["preview_manifest_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "run_id": proof["run_id"],
            "name": proof["name"],
            "namespace": NAMESPACE,
            "kubernetes_context": kubectl.context,
        },
    )

    # Never wrap this call in retry logic.  Any error after the durable intent
    # is ambiguous until the exact name/run ID is reconciled read-only.
    created = kubectl.create_once(manifest)
    _assert_created_identity(created, proof, require_uid=True)
    result = {
        "name": proof["name"],
        "run_id": proof["run_id"],
        "namespace": NAMESPACE,
        "uid": created["metadata"]["uid"],
        "manifest_sha256": proof["manifest_sha256"],
        "submitted": True,
        "transport": "direct-kubectl-create",
    }
    _append_journal(journal, {"state": "KUBECTL_CREATE_RESPONSE", **result})
    return result
