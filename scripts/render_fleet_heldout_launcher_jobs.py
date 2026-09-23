#!/usr/bin/env python3
"""Render CPU-only in-cluster launchers for sealed Fleet held-out packets.

The desktop cannot inspect ``/mnt/sfs/jobs`` or the rollout PostgreSQL catalog,
so the packet launcher must execute inside the production cluster.  This
renderer makes that boundary explicit: each arm gets one create-once CPU Job
which runs ``heldout_launch.launch_once`` against the mounted SFS, database and
in-cluster Kubernetes API.  This module renders files only; it never submits.
"""

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
from typing import Any, Literal

import yaml

from evals.fleet import heldout_launch

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
SOURCE_ROOTS = (
    ROOT / "cyber_post_train" / "__init__.py",
    ROOT / "cyber_post_train" / "jobs.py",
    ROOT / "evals" / "__init__.py",
    ROOT / "evals" / "fleet" / "__init__.py",
)
LAUNCHER = r"""from __future__ import annotations

import json
import os
from pathlib import Path

from evals.fleet.heldout_launch import KubectlCluster, PostgresDatabase, launch_once

workspace = Path(__file__).resolve().parent
token_root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
kubeconfig = workspace / "kubeconfig.yaml"
kubeconfig.write_text(
    "apiVersion: v1\n"
    "kind: Config\n"
    "clusters:\n"
    "- name: incluster\n"
    "  cluster:\n"
    "    server: https://kubernetes.default.svc\n"
    f"    certificate-authority: {token_root / 'ca.crt'}\n"
    "users:\n"
    "- name: launcher\n"
    "  user:\n"
    f"    tokenFile: {token_root / 'token'}\n"
    "contexts:\n"
    "- name: incluster\n"
    "  context:\n"
    "    cluster: incluster\n"
    "    user: launcher\n"
    "    namespace: fleet-train-jobs\n"
    "current-context: incluster\n",
    encoding="utf-8",
)
os.chmod(kubeconfig, 0o600)
os.environ["KUBECONFIG"] = str(kubeconfig)
journal = Path(os.environ["CREATE_JOURNAL"])
journal.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
result = launch_once(
    workspace / os.environ["PACKET_PATH"],
    cluster=KubectlCluster("incluster"),
    database=PostgresDatabase(),
    journal=journal,
)
print(json.dumps(result, sort_keys=True))
"""

TERMINAL_COLLECTOR = r"""from __future__ import annotations

import json
import os
from pathlib import Path

from evals.fleet.heldout_launch import KubectlCluster, PostgresDatabase, collect_terminal

workspace = Path(__file__).resolve().parent
token_root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
kubeconfig = workspace / "kubeconfig.yaml"
kubeconfig.write_text(
    "apiVersion: v1\n"
    "kind: Config\n"
    "clusters:\n"
    "- name: incluster\n"
    "  cluster:\n"
    "    server: https://kubernetes.default.svc\n"
    f"    certificate-authority: {token_root / 'ca.crt'}\n"
    "users:\n"
    "- name: collector\n"
    "  user:\n"
    f"    tokenFile: {token_root / 'token'}\n"
    "contexts:\n"
    "- name: incluster\n"
    "  context:\n"
    "    cluster: incluster\n"
    "    user: collector\n"
    "    namespace: fleet-train-jobs\n"
    "current-context: incluster\n",
    encoding="utf-8",
)
os.chmod(kubeconfig, 0o600)
os.environ["KUBECONFIG"] = str(kubeconfig)
packet_path = workspace / os.environ["PACKET_PATH"]
packet = json.loads(packet_path.read_text(encoding="utf-8"))
receipt_path = Path(packet["output_root"]) / "TERMINAL_OBSERVATION.json"
result = collect_terminal(
    packet_path,
    cluster=KubectlCluster("incluster"),
    database=PostgresDatabase(),
    receipt_path=receipt_path,
)
print(json.dumps(result, sort_keys=True))
"""


def _canonical(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(packet_dir: Path, *, program: str = LAUNCHER) -> tuple[bytes, str]:
    packet = heldout_launch.build_package(packet_dir / "LAUNCH_PACKET.json")
    files: dict[str, str] = {"launch.py": program}
    for source in SOURCE_ROOTS:
        files[str(source.relative_to(ROOT))] = source.read_text(encoding="utf-8")
    for source in sorted((ROOT / "evals" / "fleet").glob("*.py")):
        files[str(source.relative_to(ROOT))] = source.read_text(encoding="utf-8")
    for source in sorted(packet_dir.iterdir()):
        if source.is_file() and not source.is_symlink():
            files[str(Path("packet") / source.name)] = source.read_text(encoding="utf-8")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    if len(base64.b64encode(compressed)) > 900_000:
        raise ValueError("launcher ConfigMap bundle exceeds the conservative size ceiling")
    return compressed, packet.packet.job_name


def _objects(
    *,
    replica: str,
    compressed: bytes,
    evaluator_job: str,
    operation: Literal["launch", "terminal"] = "launch",
    terminal_generation: int = 1,
) -> tuple[dict, dict]:
    # ``-launch`` was the first operational attempt.  Those Pods could not
    # reach the rollout ledger because they lacked the NetworkPolicy client
    # label.  Keep the deterministic repair create-once under a new name; the
    # evaluator identity itself remains unchanged and is still protected by
    # its SFS, PostgreSQL, Kubernetes, and ledger duplicate gates.
    if operation == "launch":
        name = f"{evaluator_job}-launch-v2"
        operation_environment = [
            {"name": "PACKET_PATH", "value": "packet/LAUNCH_PACKET.json"},
            {
                "name": "CREATE_JOURNAL",
                "value": (
                    "/mnt/sfs/jobs/chris-q38-fleet-dev17-s46to53-launch-control-v1/"
                    f"{replica}-CREATE_INTENT.jsonl"
                ),
            },
        ]
    else:
        if type(terminal_generation) is not int or not 1 <= terminal_generation <= 99:
            raise ValueError("terminal collector generation must be an integer from 1 through 99")
        name = f"{evaluator_job}-terminal-v{terminal_generation}"
        operation_environment = [{"name": "PACKET_PATH", "value": "packet/LAUNCH_PACKET.json"}]
    if len(name) > 63:
        raise ValueError("launcher Kubernetes name is too long")
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": heldout_launch.NAMESPACE},
        "immutable": True,
        "binaryData": {"bundle.json.gz": base64.b64encode(compressed).decode()},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": heldout_launch.NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/experiment": name,
                "cyber-post-train.fleet.ai/owner": "chris",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                heldout_launch.FAILURE_ALERT_ANNOTATION: heldout_launch.FAILURE_ALERT_OFF,
                heldout_launch.CREATE_ONCE_ANNOTATION: "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": name,
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
                            "name": "launcher",
                            "image": IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "apt-get update\n"
                                "apt-get install --yes --no-install-recommends kubernetes-client\n"
                                "mkdir -p /workspace/source\n"
                                "python - <<'PY'\n"
                                "import gzip,json,pathlib\n"
                                "root=pathlib.Path('/workspace/source')\n"
                                "data=json.loads(gzip.decompress(\n"
                                " pathlib.Path('/bootstrap/bundle.json.gz').read_bytes()))\n"
                                "for name,text in data.items():\n"
                                " p=root/name; p.parent.mkdir(parents=True,exist_ok=True)\n"
                                " p.write_text(text)\n"
                                "PY\n"
                                "cd /workspace/source\n"
                                "exec uv run --no-project --with httpx==0.28.1 "
                                "--with pyyaml==6.0.3 "
                                "--with 'psycopg[binary]==3.3.5' python launch.py\n"
                            ],
                            "env": [
                                *operation_environment,
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
                        {"name": "bootstrap", "configMap": {"name": name}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    return config_map, job


def render_terminal_collectors(
    *, packets: Path, output: Path, terminal_generation: int = 1
) -> dict[str, Any]:
    """Render score-blind collectors; never create them before evaluator terminal state."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("terminal collector output already exists")
    if not output.parent.is_dir():
        raise ValueError("terminal collector output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-terminal-collectors-", dir=output.parent))
    try:
        items = []
        arms = []
        packet_paths = sorted(packets.glob("seed*/base/LAUNCH_PACKET.json")) + sorted(
            packets.glob("seed*/candidate/LAUNCH_PACKET.json")
        )
        if len(packet_paths) != 16:
            raise ValueError("exactly eight base and eight candidate packets are required")
        for packet_path in packet_paths:
            relative = packet_path.relative_to(packets)
            replica = f"{relative.parts[0]}-{relative.parts[1]}"
            compressed, evaluator_job = _bundle(packet_path.parent, program=TERMINAL_COLLECTOR)
            config_map, job = _objects(
                replica=replica,
                compressed=compressed,
                evaluator_job=evaluator_job,
                operation="terminal",
                terminal_generation=terminal_generation,
            )
            items.extend([config_map, job])
            arms.append(
                {
                    "replica": replica,
                    "collector_generation": terminal_generation,
                    "evaluator_job": evaluator_job,
                    "collector_job": job["metadata"]["name"],
                    "bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
                    "failure_alerts": job["metadata"]["annotations"][
                        heldout_launch.FAILURE_ALERT_ANNOTATION
                    ],
                    "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
                    "gpu_requests": 0,
                }
            )
        bundle = {"apiVersion": "v1", "kind": "List", "items": items}
        bundle_path = temporary / "terminal-collectors.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = {
            "schema": "cyber_fleet_heldout_terminal_collector_render_v1",
            "bundle_path": "terminal-collectors.yaml",
            "bundle_file_sha256": _file_sha(bundle_path),
            "collector_generation": terminal_generation,
            "arms": arms,
            "external_mutations": 0,
            "launch_performed": False,
        }
        receipt["sha256"] = _canonical(receipt)
        (temporary / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def render(*, packets: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("launcher output already exists")
    if not output.parent.is_dir():
        raise ValueError("launcher output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-launchers-", dir=output.parent))
    try:
        items = []
        arms = []
        packet_paths = sorted(packets.glob("seed*/base/LAUNCH_PACKET.json")) + sorted(
            packets.glob("seed*/candidate/LAUNCH_PACKET.json")
        )
        if len(packet_paths) != 16:
            raise ValueError("exactly eight base and eight candidate packets are required")
        for packet_path in packet_paths:
            relative = packet_path.relative_to(packets)
            replica = f"{relative.parts[0]}-{relative.parts[1]}"
            compressed, evaluator_job = _bundle(packet_path.parent)
            config_map, job = _objects(
                replica=replica, compressed=compressed, evaluator_job=evaluator_job
            )
            items.extend([config_map, job])
            arms.append(
                {
                    "replica": replica,
                    "evaluator_job": evaluator_job,
                    "launcher_job": job["metadata"]["name"],
                    "bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
                    "bundle_bytes": len(compressed),
                    "failure_alerts": job["metadata"]["annotations"][
                        heldout_launch.FAILURE_ALERT_ANNOTATION
                    ],
                    "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
                    "gpu_requests": 0,
                }
            )
        bundle = {"apiVersion": "v1", "kind": "List", "items": items}
        bundle_path = temporary / "launchers.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = {
            "schema": "cyber_fleet_heldout_incluster_launcher_render_v1",
            "bundle_path": "launchers.yaml",
            "bundle_file_sha256": _file_sha(bundle_path),
            "arms": arms,
            "external_mutations": 0,
            "launch_performed": False,
        }
        receipt["sha256"] = _canonical(receipt)
        (temporary / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-collectors", action="store_true")
    parser.add_argument("--terminal-generation", type=int, default=1)
    args = parser.parse_args()
    if args.terminal_collectors:
        result = render_terminal_collectors(
            packets=args.packets,
            output=args.output,
            terminal_generation=args.terminal_generation,
        )
    else:
        if args.terminal_generation != 1:
            raise ValueError("terminal generation applies only to terminal collectors")
        result = render(packets=args.packets, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
