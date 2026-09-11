"""Create and execute a zero-GPU, read-only base-model payload probe.

The same stdlib-only file is embedded in the helper Pod.  It hashes paths,
sizes and bytes, emits only sanitized digests/counts, and never contacts a
model endpoint.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cyber_fleet_base_payload_readback_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def digest_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("expected a canonical relative model path")
    return value


def probe(config: dict[str, Any], *, environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    if set(config) != {
        "plan_sha256",
        "root",
        "repository",
        "revision",
        "weights",
        "tokenizer",
        "configuration",
        "index_sha256",
        "source_serving_pod",
        "source_serving_pod_uid",
        "source_node",
    }:
        raise ValueError("probe configuration has unknown or missing fields")
    root = Path(config["root"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("served model root is not an absolute ordinary directory")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("served model payload contains symlinks")
    files = sorted(path for path in paths if path.is_file())
    rows = []
    for path in files:
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    observed = {row["path"]: row for row in rows}

    expected_weights = sorted(config["weights"], key=lambda row: row["path"])
    for row in expected_weights:
        _safe_relative(row["path"])
        if observed.get(row["path"]) != row:
            raise ValueError("served weight bytes differ from the exact model lock")
    expected_tokenizer = sorted(config["tokenizer"], key=lambda row: row["path"])
    for row in expected_tokenizer:
        _safe_relative(row["path"])
        actual = observed.get(row["path"])
        if actual is None or actual["sha256"] != row["sha256"]:
            raise ValueError("served tokenizer bytes differ from the exact model lock")
    for path, expected_sha in sorted(config["configuration"].items()):
        _safe_relative(path)
        actual = observed.get(path)
        if actual is None or actual["sha256"] != expected_sha:
            raise ValueError("served configuration bytes differ from the exact model lock")
    index = observed.get("model.safetensors.index.json")
    if index is None or index["sha256"] != config["index_sha256"]:
        raise ValueError("served safetensors index differs from the exact model lock")
    chat = next(row for row in expected_tokenizer if row["path"] == "chat_template.jinja")

    unsigned = {
        "schema": SCHEMA,
        "status": "passed",
        "plan_sha256": config["plan_sha256"],
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": {
            "read_only_host_path": True,
            "gpu_requests": 0,
            "prompt_requests": 0,
            "completion_requests": 0,
            "scoring_requests": 0,
            "task_or_grading_requests": 0,
        },
        "helper": {
            "pod": env.get("POD_NAME", "local-test"),
            "pod_uid": env.get("POD_UID", "local-test"),
            "node": env.get("NODE_NAME", "local-test"),
        },
        "source": {
            "serving_pod": config["source_serving_pod"],
            "serving_pod_uid": config["source_serving_pod_uid"],
            "node": config["source_node"],
            "root": str(root),
        },
        "model": {
            "repository": config["repository"],
            "revision": config["revision"],
            "weights_manifest_sha256": digest_json(expected_weights),
            "tokenizer_manifest_sha256": digest_json(expected_tokenizer),
            "chat_template_sha256": "sha256:" + chat["sha256"],
            "configuration_manifest_sha256": digest_json(
                [
                    {"path": path, "sha256": sha}
                    for path, sha in sorted(config["configuration"].items())
                ]
            ),
            "index_sha256": "sha256:" + index["sha256"],
        },
        "payload": {
            "file_count": len(rows),
            "total_bytes": sum(row["size"] for row in rows),
            "manifest_sha256": digest_json(rows),
            "payload_rehashed": True,
            "symlinks_absent": True,
        },
    }
    return {**unsigned, "sha256": digest_json(unsigned)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def build_config(
    plan: dict[str, Any],
    lock: dict[str, Any],
    weights: dict[str, Any],
    **source: str,
) -> dict[str, Any]:
    rows = weights.get("files")
    if (
        weights.get("repo") != lock.get("repo")
        or weights.get("revision") != lock.get("revision")
        or not isinstance(rows, list)
        or digest_json(sorted(rows, key=lambda row: row["path"]))
        != lock.get("weights", {}).get("manifest_sha256")
    ):
        raise ValueError("weight inventory differs from the exact model lock")
    configuration = {
        key.removesuffix("_sha256") + ".json": value.removeprefix("sha256:")
        for key, value in lock["configuration"].items()
    }
    return {
        "plan_sha256": plan["sha256"],
        "root": source["root"],
        "repository": lock["repo"],
        "revision": lock["revision"],
        "weights": sorted(rows, key=lambda row: row["path"]),
        "tokenizer": sorted(lock["tokenizer"]["files"], key=lambda row: row["path"]),
        "configuration": configuration,
        "index_sha256": lock["weights"]["index_sha256"].removeprefix("sha256:"),
        "source_serving_pod": source["serving_pod"],
        "source_serving_pod_uid": source["serving_pod_uid"],
        "source_node": source["node"],
    }


def build_pod(config: dict[str, Any], *, name: str, namespace: str, image: str) -> dict[str, Any]:
    if "@sha256:" not in image:
        raise ValueError("helper image must be digest pinned")
    source = Path(__file__).read_text(encoding="utf-8")
    encoded = base64.b64encode(canonical_json(config).encode()).decode()
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/purpose": "qwen38-base-payload-readback",
            },
            "annotations": {"cyber-post-train.fleet.ai/plan-sha256": config["plan_sha256"]},
        },
        "spec": {
            "activeDeadlineSeconds": 1800,
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "nodeName": config["source_node"],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 10,
            "tolerations": [
                {
                    "key": "workload",
                    "operator": "Equal",
                    "value": "fleetai-training-ng-gpu",
                    "effect": "NoSchedule",
                },
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            ],
            "containers": [
                {
                    "name": "readback",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["python3", "-c", source, "probe"],
                    "env": [
                        {"name": "PROBE_CONFIG_B64", "value": encoded},
                        {
                            "name": "POD_NAME",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                        },
                        {
                            "name": "POD_UID",
                            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                        },
                        {
                            "name": "NODE_NAME",
                            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                        },
                    ],
                    "resources": {
                        "requests": {"cpu": "500m", "memory": "512Mi"},
                        "limits": {"cpu": "2", "memory": "2Gi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                        "readOnlyRootFilesystem": True,
                    },
                    "volumeMounts": [
                        {"name": "served-scratch", "mountPath": "/scratch", "readOnly": True}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "served-scratch",
                    "hostPath": {"path": "/scratch", "type": "Directory"},
                }
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    render = sub.add_parser("render-pod")
    render.add_argument("plan", type=Path)
    render.add_argument("--lock", type=Path, required=True)
    render.add_argument("--weights", type=Path, required=True)
    render.add_argument("--name", required=True)
    render.add_argument("--namespace", default="inference")
    render.add_argument("--image", required=True)
    render.add_argument("--root", required=True)
    render.add_argument("--serving-pod", required=True)
    render.add_argument("--serving-pod-uid", required=True)
    render.add_argument("--node", required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        config = json.loads(base64.b64decode(os.environ["PROBE_CONFIG_B64"], validate=True))
        print(canonical_json(probe(config)), flush=True)
        return 0
    config = build_config(
        _load(args.plan),
        _load(args.lock),
        _load(args.weights),
        root=args.root,
        serving_pod=args.serving_pod,
        serving_pod_uid=args.serving_pod_uid,
        node=args.node,
    )
    print(
        canonical_json(
            build_pod(config, name=args.name, namespace=args.namespace, image=args.image)
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(canonical_json({"schema": SCHEMA, "status": "failed", "error": str(error)}))
        raise
