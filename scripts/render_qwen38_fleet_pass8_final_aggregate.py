#!/usr/bin/env python3
"""Render, but never create, the private Fleet dev17 pass@8 final gate."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import tempfile
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
NAME = "chris-q38-dev17-pass8-final-v1"
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


def render(*, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("private final aggregate render already exists")
    if not output.parent.is_dir():
        raise RenderError("private final aggregate render parent is missing")
    plan = aggregate.build_current_study_plan(
        task_set_path=TASK_SET,
        roster_path=ROSTER,
        base_config_path=BASE_CONFIG,
    )
    compressed, digests = _bundle(plan)
    config_map, job = _objects(plan, compressed, digests)
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-pass8-final-", dir=output.parent))
    try:
        bundle = {"apiVersion": "v1", "kind": "List", "items": [config_map, job]}
        bundle_path = temporary / "final-aggregate.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = {
            "schema": "cyber_fleet_matched_pass8_final_job_render_v1",
            "namespace": NAMESPACE,
            "config_map_name": CONFIG_MAP,
            "job_name": NAME,
            "study_plan_sha256": plan["sha256"],
            "source_files": {
                "aggregate.py": digests["aggregate.py"],
                "run.py": digests["run.py"],
                "study.json": digests["study.json"],
            },
            "compressed_bundle_sha256": ("sha256:" + hashlib.sha256(compressed).hexdigest()),
            "rendered_bundle_file_sha256": _file_digest(bundle_path),
            "failure_alerts": job["metadata"]["annotations"]["fleet.ai/failure-alerts"],
            "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
            "gpu_requests": 0,
            "create_once": True,
            "two_server_previews_required_before_create": True,
            "external_mutations": 0,
            "launch_performed": False,
        }
        receipt["sha256"] = _canonical_digest(receipt)
        (temporary / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
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
    normalized_job = stable_job_preview(job)
    if not _contains(normalized_job, stable_job_preview(expected_job)):
        raise RenderError("server-rendered Job differs from the immutable render")
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


def validate_previews(
    *, render_root: Path, first: Path, second: Path, output: Path
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("server preview receipt already exists")
    bundle_path = render_root / "final-aggregate.yaml"
    receipt_path = render_root / "RENDER.json"
    render_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claimed = render_receipt.get("sha256")
    if claimed != _canonical_digest(
        {key: value for key, value in render_receipt.items() if key != "sha256"}
    ):
        raise RenderError("render receipt self digest differs")
    if render_receipt.get("rendered_bundle_file_sha256") != _file_digest(bundle_path):
        raise RenderError("rendered bundle differs from its immutable receipt")
    expected = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    one = json.loads(first.read_text(encoding="utf-8"))
    two = json.loads(second.read_text(encoding="utf-8"))
    stable_one = _normalized_preview(one, expected)
    stable_two = _normalized_preview(two, expected)
    first_digest = _canonical_digest(stable_one)
    second_digest = _canonical_digest(stable_two)
    if first_digest != second_digest:
        raise RenderError("identical server dry-runs produced different stable previews")
    receipt = {
        "schema": "cyber_fleet_matched_pass8_final_server_preview_v1",
        "render_receipt_file_sha256": _file_digest(receipt_path),
        "first_preview_file_sha256": _file_digest(first),
        "second_preview_file_sha256": _file_digest(second),
        "stable_server_preview_sha256": first_digest,
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
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--preview-one", type=Path)
    parser.add_argument("--preview-two", type=Path)
    parser.add_argument("--preview-receipt", type=Path)
    args = parser.parse_args()
    if args.output is not None and all(
        value is None
        for value in (args.render_root, args.preview_one, args.preview_two, args.preview_receipt)
    ):
        result = render(output=args.output)
    elif args.output is None and all(
        value is not None
        for value in (args.render_root, args.preview_one, args.preview_two, args.preview_receipt)
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
