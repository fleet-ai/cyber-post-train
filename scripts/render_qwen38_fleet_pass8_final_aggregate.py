#!/usr/bin/env python3
"""Render, but never create, the private Fleet dev17 pass@8 final gate."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
import shutil
import stat
import uuid
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import final_pass8_aggregate as aggregate
from evals.fleet.evaluate import stable_job_preview

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json"
ROSTER = ROOT / "configs/evaluation/qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
BASE_CONFIG = ROOT / "configs/evaluation/qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
NAMESPACE = "fleet-train-jobs"
NAME = "chris-q38-dev17-pass8-final-v2"
CONFIG_MAP = f"{NAME}-code"
SOURCE = ROOT / "evals/fleet/final_pass8_aggregate.py"
RUNNER = r"""from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

source = Path("aggregate.py")
actual = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
if actual != os.environ["FINAL_AGGREGATE_MODULE_SHA256"]:
    raise RuntimeError("private final aggregate module bytes differ")
spec = importlib.util.spec_from_file_location("fleet_final_pass8_aggregate", source)
if spec is None or spec.loader is None:
    raise RuntimeError("private final aggregate module cannot be loaded")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module.run_from_environment(Path("study.json"))
print(json.dumps({
    "status": result["status"],
    "task_count": result["task_count"],
    "valid_outcomes_per_task_arm": result["valid_outcomes_per_task_arm"],
    "receipt_sha256": result["receipt_sha256"],
}, sort_keys=True))
"""


class RenderError(ValueError):
    """The rendered private final gate is incomplete or preview-mismatched."""


def _canonical_digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular_once(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RenderError(f"{label} is not an exact regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RenderError(f"{label} is not an exact regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (
            identity != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RenderError(f"{label} changed while it was read")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _bundle(plan: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    files = {
        "aggregate.py": SOURCE.read_text(encoding="utf-8"),
        "run.py": RUNNER,
        "study.json": json.dumps(plan, indent=2, sort_keys=True) + "\n",
    }
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    if len(base64.b64encode(compressed)) > 900_000:
        raise RenderError("private final aggregate bundle exceeds the ConfigMap ceiling")
    return compressed, digests


def _objects(plan: dict[str, Any], compressed: bytes, digests: dict[str, str]) -> tuple[dict, dict]:
    bundle_digest = "sha256:" + hashlib.sha256(compressed).hexdigest()
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP, "namespace": NAMESPACE},
        "immutable": True,
        "binaryData": {"bundle.json.gz": base64.b64encode(compressed).decode()},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/experiment": NAME,
                "cyber-post-train.fleet.ai/owner": "chris",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "fleet.ai/failure-alerts": "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": NAME,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "aggregate",
                            "image": IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "python - <<'PY'\n"
                                "import gzip,hashlib,json,os,pathlib\n"
                                "bundle=pathlib.Path('/bootstrap/bundle.json.gz').read_bytes()\n"
                                "actual='sha256:'+hashlib.sha256(bundle).hexdigest()\n"
                                "assert actual == os.environ['FINAL_BUNDLE_SHA256']\n"
                                "root=pathlib.Path('/workspace/source')\n"
                                "root.mkdir(parents=True,exist_ok=False)\n"
                                "for name,text in json.loads(gzip.decompress(bundle)).items():\n"
                                " p=root/name; p.parent.mkdir(parents=True,exist_ok=True)\n"
                                " p.write_text(text,encoding='utf-8')\n"
                                "PY\n"
                                "cd /workspace/source\n"
                                "exec uv run --no-project --with 'psycopg[binary]==3.3.5' "
                                "python run.py\n"
                            ],
                            "env": [
                                {"name": "FINAL_BUNDLE_SHA256", "value": bundle_digest},
                                {
                                    "name": "FINAL_AGGREGATE_MODULE_SHA256",
                                    "value": digests["aggregate.py"],
                                },
                                {
                                    "name": "FINAL_STUDY_PLAN_FILE_SHA256",
                                    "value": digests["study.json"],
                                },
                                {
                                    "name": "FINAL_OUTPUT_ROOT",
                                    "value": plan["private_output_root"],
                                },
                                {
                                    "name": "ROLLOUT_DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-rollout-postgres-v1",
                                            "key": "ROLLOUT_DATABASE_URL",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIG_MAP}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    return config_map, job


def _render_receipt(
    *,
    plan: dict[str, Any],
    compressed: bytes,
    digests: dict[str, str],
    rendered_bundle_file_sha256: str,
) -> dict[str, Any]:
    receipt = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_job_render_v1",
        "namespace": NAMESPACE,
        "config_map_name": CONFIG_MAP,
        "job_name": NAME,
        "study_plan_sha256": plan["sha256"],
        "migration_receipt_sha256": plan["migration_receipt_sha256"],
        "migration_receipt_file_sha256": plan["migration_receipt_file_sha256"],
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "comparison_definition_file_sha256": plan["comparison_definition_file_sha256"],
        "source_files": digests,
        "compressed_bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
        "rendered_bundle_file_sha256": rendered_bundle_file_sha256,
        "failure_alerts": "off",
        "priority_class": "c1",
        "gpu_requests": 0,
        "create_once": True,
        "two_server_previews_required_before_create": True,
        "external_mutations": 0,
        "launch_performed": False,
    }
    return {**receipt, "sha256": _canonical_digest(receipt)}


def render(*, output: Path, migration_receipt: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("private final aggregate render already exists")
    if not output.parent.is_dir():
        raise RenderError("private final aggregate render parent is missing")
    plan = aggregate.build_current_study_plan(
        task_set_path=TASK_SET,
        roster_path=ROSTER,
        base_config_path=BASE_CONFIG,
        migration_receipt_path=migration_receipt,
    )
    compressed, digests = _bundle(plan)
    config_map, job = _objects(plan, compressed, digests)
    output.mkdir(mode=0o700)
    try:
        bundle = {"apiVersion": "v1", "kind": "List", "items": [config_map, job]}
        bundle_path = output / "final-aggregate.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = _render_receipt(
            plan=plan,
            compressed=compressed,
            digests=digests,
            rendered_bundle_file_sha256=_file_digest(bundle_path),
        )
        (output / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return receipt


def _items(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("kind") == "List" and isinstance(value.get("items"), list):
        return value["items"]
    return [value]


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return actual == expected


def _contains_accelerator_resource(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            isinstance(key, str)
            and ("gpu" in key.lower() or "mig" in key.lower() or key.startswith("nvidia.com/"))
            for key in value
        ) or any(_contains_accelerator_resource(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_accelerator_resource(item) for item in value)
    return False


def _extra_keys(
    actual: Mapping[str, Any], expected: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unexpected = set(actual) - set(expected) - allowed
    if unexpected:
        raise RenderError(f"server-rendered {label} added unreviewed fields")


def _server_defaults_only(job: dict[str, Any], expected: dict[str, Any]) -> None:
    normalized = stable_job_preview(job)
    reviewed = stable_job_preview(expected)
    _extra_keys(normalized, reviewed, set(), "Job")
    _extra_keys(normalized["metadata"], reviewed["metadata"], set(), "Job metadata")
    actual_annotations = normalized["metadata"].get("annotations", {})
    expected_annotations = reviewed["metadata"].get("annotations", {})
    if actual_annotations != expected_annotations:
        raise RenderError("server-rendered root Job annotations differ")
    root_labels = normalized["metadata"].get("labels", {})
    expected_root_labels = reviewed["metadata"].get("labels", {})
    _extra_keys(
        root_labels,
        expected_root_labels,
        {
            "batch.kubernetes.io/controller-uid",
            "batch.kubernetes.io/job-name",
            "controller-uid",
            "job-name",
        },
        "root Job labels",
    )
    raw_uid = job.get("metadata", {}).get("uid")
    if any(
        root_labels.get(key) not in (None, raw_uid)
        for key in ("batch.kubernetes.io/controller-uid", "controller-uid")
    ) or any(
        root_labels.get(key) not in (None, NAME)
        for key in ("batch.kubernetes.io/job-name", "job-name")
    ):
        raise RenderError("server-rendered root Job generated labels differ")
    _extra_keys(
        normalized["spec"],
        reviewed["spec"],
        {
            "completionMode",
            "completions",
            "manualSelector",
            "parallelism",
            "podReplacementPolicy",
            "suspend",
        },
        "Job spec",
    )
    job_spec = normalized["spec"]
    safe_job_defaults = {
        "completionMode": {None, "NonIndexed"},
        "completions": {None, 1},
        "manualSelector": {None, False},
        "parallelism": {None, 1},
        "podReplacementPolicy": {None, "Failed", "TerminatingOrFailed"},
        "suspend": {None, False},
    }
    if any(job_spec.get(field) not in values for field, values in safe_job_defaults.items()):
        raise RenderError("server-rendered Job has a non-default execution policy")
    selector = job.get("spec", {}).get("selector")
    if selector is not None:
        if not isinstance(selector, dict) or set(selector) - {"matchLabels", "matchExpressions"}:
            raise RenderError("server-rendered Job selector differs")
        selector_labels = selector.get("matchLabels")
        expressions = selector.get("matchExpressions")
        raw_uid = job.get("metadata", {}).get("uid")
        if (
            not isinstance(selector_labels, dict)
            or not selector_labels
            or set(selector_labels) - {"batch.kubernetes.io/controller-uid", "controller-uid"}
            or any(value != raw_uid for value in selector_labels.values())
            or expressions not in (None, [])
        ):
            raise RenderError("server-rendered Job selector differs")
    actual_template = normalized["spec"]["template"]
    expected_template = reviewed["spec"]["template"]
    _extra_keys(actual_template, expected_template, set(), "Pod template")
    _extra_keys(
        actual_template["metadata"],
        expected_template["metadata"],
        {"creationTimestamp"},
        "Pod metadata",
    )
    if actual_template["metadata"].get("creationTimestamp") is not None:
        raise RenderError("server-rendered Pod creation timestamp is not the API default")
    labels = actual_template["metadata"].get("labels", {})
    expected_labels = expected_template["metadata"].get("labels", {})
    _extra_keys(
        labels,
        expected_labels,
        {"batch.kubernetes.io/job-name", "job-name"},
        "Pod labels",
    )
    if any(
        labels.get(key) not in (None, NAME) for key in ("batch.kubernetes.io/job-name", "job-name")
    ):
        raise RenderError("server-rendered Pod job-name label differs")
    pod = actual_template["spec"]
    expected_pod = expected_template["spec"]
    _extra_keys(
        pod,
        expected_pod,
        {
            "dnsPolicy",
            "enableServiceLinks",
            "preemptionPolicy",
            "priority",
            "schedulerName",
            "schedulingGates",
            "securityContext",
            "serviceAccount",
            "serviceAccountName",
            "terminationGracePeriodSeconds",
        },
        "Pod spec",
    )
    safe_pod_defaults = {
        "dnsPolicy": {None, "ClusterFirst"},
        "enableServiceLinks": {None, True},
        "preemptionPolicy": {None, "PreemptLowerPriority"},
        "priority": {None, 10_000},
        "schedulerName": {None, "default-scheduler"},
        "terminationGracePeriodSeconds": {None, 30},
    }
    if any(pod.get(field) not in values for field, values in safe_pod_defaults.items()):
        raise RenderError("server-rendered Pod has a non-default scheduling policy")
    if any(pod.get(field) is True for field in ("hostIPC", "hostNetwork", "hostPID")):
        raise RenderError("server-rendered Pod enables a host namespace")
    if pod.get("securityContext") not in (None, {}):
        raise RenderError("server-rendered Pod adds a security context")
    if pod.get("serviceAccountName") not in (None, "default") or pod.get("serviceAccount") not in (
        None,
        "default",
    ):
        raise RenderError("server-rendered Pod changes the service account")
    gates = pod.get("schedulingGates", [])
    if gates not in ([], [{"name": "kueue.x-k8s.io/admission"}]):
        raise RenderError("server-rendered Pod adds an unknown scheduling gate")
    if pod.get("initContainers") or pod.get("ephemeralContainers") or pod.get("resourceClaims"):
        raise RenderError("server-rendered Pod adds an unreviewed container or resource claim")
    if pod.get("nodeSelector") != expected_pod.get("nodeSelector"):
        raise RenderError("server-rendered Pod node selector differs")
    if pod.get("tolerations") != expected_pod.get("tolerations"):
        raise RenderError("server-rendered Pod tolerations differ")
    containers = pod.get("containers")
    expected_containers = expected_pod["containers"]
    if not isinstance(containers, list) or len(containers) != len(expected_containers):
        raise RenderError("server-rendered Pod changes the container roster")
    for container, expected_container in zip(containers, expected_containers, strict=True):
        _extra_keys(
            container,
            expected_container,
            {"imagePullPolicy", "terminationMessagePath", "terminationMessagePolicy"},
            "container",
        )
        safe_container_defaults = {
            "imagePullPolicy": {None, "IfNotPresent"},
            "terminationMessagePath": {None, "/dev/termination-log"},
            "terminationMessagePolicy": {None, "File"},
        }
        if any(
            container.get(field) not in values for field, values in safe_container_defaults.items()
        ):
            raise RenderError("server-rendered container has an unsafe runtime default")
        if container.get("securityContext") not in (None, {}):
            raise RenderError("server-rendered container adds a security context")
        if container.get("env") != expected_container.get("env"):
            raise RenderError("server-rendered container environment differs")
        if container.get("volumeMounts") != expected_container.get("volumeMounts"):
            raise RenderError("server-rendered container volume mounts differ")
        resources = container.get("resources", {})
        expected_resources = expected_container.get("resources", {})
        _extra_keys(resources, expected_resources, set(), "container resources")
        for kind in ("requests", "limits"):
            if set(resources.get(kind, {})) != set(expected_resources.get(kind, {})):
                raise RenderError("server-rendered container resource keys differ")
    volumes = pod.get("volumes")
    expected_volumes = expected_pod["volumes"]
    if not isinstance(volumes, list) or len(volumes) != len(expected_volumes):
        raise RenderError("server-rendered Pod changes the volume roster")
    for volume, expected_volume in zip(volumes, expected_volumes, strict=True):
        _extra_keys(volume, expected_volume, set(), "volume")
        if "configMap" in volume:
            _extra_keys(
                volume["configMap"],
                expected_volume["configMap"],
                {"defaultMode"},
                "ConfigMap volume",
            )
            if volume["configMap"].get("defaultMode") not in (None, 0o644):
                raise RenderError("server-rendered ConfigMap volume mode differs")
        if "persistentVolumeClaim" in volume:
            _extra_keys(
                volume["persistentVolumeClaim"],
                expected_volume["persistentVolumeClaim"],
                set(),
                "persistent volume",
            )


def _job_uid(value: dict[str, Any]) -> str:
    jobs = [item for item in _items(value) if item.get("kind") == "Job"]
    if len(jobs) != 1:
        raise RenderError("server preview lacks the exact Job")
    raw = jobs[0].get("metadata", {}).get("uid")
    try:
        return str(uuid.UUID(raw))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RenderError("server preview lacks a valid server-assigned Job UID") from exc


def _normalized_preview(value: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    items = _items(value)
    jobs = [item for item in items if item.get("kind") == "Job"]
    maps = [item for item in items if item.get("kind") == "ConfigMap"]
    if len(jobs) != 1 or len(maps) != 1:
        raise RenderError("server preview lacks the exact Job and ConfigMap")
    job = jobs[0]
    config_map = maps[0]
    expected_items = _items(expected)
    expected_job = next(item for item in expected_items if item["kind"] == "Job")
    expected_map = next(item for item in expected_items if item["kind"] == "ConfigMap")
    if job.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") != "off":
        raise RenderError("server-rendered root Job did not retain failure alerts off")
    if _contains_accelerator_resource(job):
        raise RenderError("server-rendered Job unexpectedly requests an accelerator")
    normalized_job = stable_job_preview(job)
    if not _contains(normalized_job, stable_job_preview(expected_job)):
        raise RenderError("server-rendered Job differs from the immutable render")
    _server_defaults_only(job, expected_job)
    normalized_map = {
        "apiVersion": config_map.get("apiVersion"),
        "kind": config_map.get("kind"),
        "metadata": {
            "name": config_map.get("metadata", {}).get("name"),
            "namespace": config_map.get("metadata", {}).get("namespace"),
        },
        "immutable": config_map.get("immutable"),
        "binaryData": config_map.get("binaryData"),
    }
    expected_normalized_map = {
        "apiVersion": expected_map["apiVersion"],
        "kind": expected_map["kind"],
        "metadata": {
            "name": expected_map["metadata"]["name"],
            "namespace": expected_map["metadata"]["namespace"],
        },
        "immutable": expected_map["immutable"],
        "binaryData": expected_map["binaryData"],
    }
    if normalized_map != expected_normalized_map:
        raise RenderError("server-rendered ConfigMap differs from the immutable render")
    return {"job": normalized_job, "config_map": normalized_map}


def _validated_render(render_root: Path) -> tuple[dict[str, Any], str]:
    bundle_path = render_root / "final-aggregate.yaml"
    receipt_path = render_root / "RENDER.json"
    bundle_bytes, _ = _read_regular_once(bundle_path, "rendered bundle")
    receipt_bytes, _ = _read_regular_once(receipt_path, "render receipt")
    try:
        expected = yaml.safe_load(bundle_bytes)
        render_receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RenderError("rendered bundle or receipt is unreadable") from exc
    if not isinstance(expected, dict) or not isinstance(render_receipt, dict):
        raise RenderError("rendered bundle or receipt is malformed")
    claimed = render_receipt.get("sha256")
    if claimed != _canonical_digest(
        {key: value for key, value in render_receipt.items() if key != "sha256"}
    ):
        raise RenderError("render receipt self digest differs")
    items = _items(expected)
    maps = [item for item in items if item.get("kind") == "ConfigMap"]
    jobs = [item for item in items if item.get("kind") == "Job"]
    if len(items) != 2 or len(maps) != 1 or len(jobs) != 1:
        raise RenderError("rendered bundle lacks the exact Job and ConfigMap")
    encoded = maps[0].get("binaryData", {}).get("bundle.json.gz")
    if not isinstance(encoded, str):
        raise RenderError("rendered ConfigMap lacks the immutable source bundle")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        files = json.loads(gzip.decompress(compressed))
    except (
        ValueError,
        binascii.Error,
        gzip.BadGzipFile,
        zlib.error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise RenderError("rendered source bundle is unreadable") from exc
    if (
        not isinstance(files, dict)
        or set(files) != {"aggregate.py", "run.py", "study.json"}
        or any(not isinstance(value, str) for value in files.values())
    ):
        raise RenderError("rendered source bundle file roster differs")
    if files["aggregate.py"] != SOURCE.read_text(encoding="utf-8") or files["run.py"] != RUNNER:
        raise RenderError("rendered executable bytes differ from the reviewed source")
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    try:
        plan = json.loads(files["study.json"])
    except json.JSONDecodeError as exc:
        raise RenderError("rendered study plan is unreadable") from exc
    if not isinstance(plan, dict):
        raise RenderError("rendered study plan is malformed")
    try:
        aggregate.validate_plan(plan)
    except aggregate.FinalAggregateError as exc:
        raise RenderError("rendered study plan differs from the frozen comparison") from exc
    expected_compressed, expected_digests = _bundle(plan)
    if compressed != expected_compressed or digests != expected_digests:
        raise RenderError("rendered source bundle differs from the canonical package")
    expected_map, expected_job = _objects(plan, compressed, digests)
    if maps[0] != expected_map or jobs[0] != expected_job:
        raise RenderError("rendered Kubernetes objects differ from the immutable source bundle")
    bundle_file_sha256 = "sha256:" + hashlib.sha256(bundle_bytes).hexdigest()
    exact_receipt = _render_receipt(
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=bundle_file_sha256,
    )
    if render_receipt != exact_receipt:
        raise RenderError("render receipt policy or source identity differs")
    return expected, "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()


def validate_previews(
    *, render_root: Path, first: Path, second: Path, output: Path
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("server preview receipt already exists")
    expected, render_receipt_file_sha256 = _validated_render(render_root)
    first_bytes, first_identity = _read_regular_once(first, "first server preview")
    second_bytes, second_identity = _read_regular_once(second, "second server preview")
    if first_identity == second_identity:
        raise RenderError("two server previews must be distinct regular files")
    try:
        one = json.loads(first_bytes)
        two = json.loads(second_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError("server preview is unreadable") from exc
    if not isinstance(one, dict) or not isinstance(two, dict):
        raise RenderError("server preview is malformed")
    first_uid = _job_uid(one)
    second_uid = _job_uid(two)
    if first_uid == second_uid:
        raise RenderError("two independent server previews must have distinct Job UIDs")
    stable_one = _normalized_preview(one, expected)
    stable_two = _normalized_preview(two, expected)
    first_digest = _canonical_digest(stable_one)
    second_digest = _canonical_digest(stable_two)
    if first_digest != second_digest:
        raise RenderError("identical server dry-runs produced different stable previews")
    receipt = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_server_preview_v1",
        "render_receipt_file_sha256": render_receipt_file_sha256,
        "first_preview_file_sha256": "sha256:" + hashlib.sha256(first_bytes).hexdigest(),
        "second_preview_file_sha256": "sha256:" + hashlib.sha256(second_bytes).hexdigest(),
        "stable_server_preview_sha256": first_digest,
        "first_job_uid_sha256": "sha256:" + hashlib.sha256(first_uid.encode()).hexdigest(),
        "second_job_uid_sha256": "sha256:" + hashlib.sha256(second_uid.encode()).hexdigest(),
        "root_failure_alerts": "off",
        "priority_class": "c1",
        "gpu_requests": 0,
        "create_performed": False,
    }
    receipt["sha256"] = _canonical_digest(receipt)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--migration-receipt", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--preview-one", type=Path)
    parser.add_argument("--preview-two", type=Path)
    parser.add_argument("--preview-receipt", type=Path)
    args = parser.parse_args()
    if (
        args.output is not None
        and args.migration_receipt is not None
        and all(
            value is None
            for value in (
                args.render_root,
                args.preview_one,
                args.preview_two,
                args.preview_receipt,
            )
        )
    ):
        result = render(output=args.output, migration_receipt=args.migration_receipt)
    elif (
        args.output is None
        and args.migration_receipt is None
        and all(
            value is not None
            for value in (
                args.render_root,
                args.preview_one,
                args.preview_two,
                args.preview_receipt,
            )
        )
    ):
        result = validate_previews(
            render_root=args.render_root,
            first=args.preview_one,
            second=args.preview_two,
            output=args.preview_receipt,
        )
    else:
        parser.error("choose exactly one render or two-preview validation operation")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
