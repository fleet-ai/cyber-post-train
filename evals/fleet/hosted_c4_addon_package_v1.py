"""Render the immutable held-only package for the hosted Qwen +2 qualifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "fleet-hosted-qwen-c4-addon-package-v1"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-cyber-hosted-q38-c4-addon-v2"
CONFIGMAP_NAME = "chris-cyber-hosted-q38-c4-addon-bootstrap-v2"
MANIFEST_PATH = "evals/fleet/cluster/hosted-q38-c4-addon-held-v1.yaml"
OUT_ROOT = "/mnt/sfs/jobs/chris-cyber-hosted-q38-c4-addon-v2"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FILES = {
    "package.py": "evals/fleet/hosted_c4_addon_package_v1.py",
    "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
    "probe.py": "evals/fleet/hosted_concurrency4_qualification_v1.py",
    "addon.py": "evals/fleet/hosted_c4_addon_qualifier_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_q38_c4_addon_v1.sh",
}


class PackageError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def _git_file(root: Path, commit: str, path: str) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise PackageError("package_commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root.resolve(strict=True)), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError("package_source_absent") from exc


def build_configmap(root: Path, commit: str) -> dict[str, Any]:
    data: dict[str, str] = {}
    bindings: dict[str, dict[str, str]] = {}
    for key, source in FILES.items():
        raw = _git_file(root, commit, source)
        data[key] = raw.decode()
        bindings[key] = {"source_path": source, "sha256": sha256(raw)}
    manifest = _git_file(root, commit, MANIFEST_PATH)
    package: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "HELD",
        "package_commit": commit,
        "files": bindings,
        "manifest": {"source_path": MANIFEST_PATH, "sha256": sha256(manifest)},
        "runtime_contract": {
            "model": "qwen3.8-27b",
            "exact_existing_controller_slots": [1, 2],
            "addon_slots": [3, 4],
            "maximum_total_owned_streams": 4,
            "synthetic_chat_completion_requests": 4,
            "task_instance_session_scoring_or_verifier_calls": 0,
            "output_root": OUT_ROOT,
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "launch_authorized": False,
        "scored_bulk_launch_authorized": False,
    }
    package["package_sha256"] = digest_without(package, "package_sha256")
    data["package.json"] = canonical_json(package).decode() + "\n"
    data["package_commit"] = commit
    data["package_sha256"] = package["package_sha256"]
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": CONFIGMAP_NAME,
            "namespace": NAMESPACE,
            "annotations": {
                "cyber-post-train.fleet.ai/preview-only": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "false",
            },
        },
        "immutable": True,
        "data": data,
    }


def materialize(projected: Path, destination: Path, expected_commit: str) -> dict[str, Any]:
    package = json.loads((projected / "package.json").read_bytes())
    if (
        package.get("schema_version") != SCHEMA
        or package.get("status") != "HELD"
        or package.get("package_commit") != expected_commit
        or package.get("launch_authorized") is not False
        or package.get("package_sha256") != digest_without(package, "package_sha256")
    ):
        raise PackageError("projected_package_invalid")
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    for key, binding in package["files"].items():
        raw = (projected / key).read_bytes()
        if sha256(raw) != binding["sha256"]:
            raise PackageError("projected_file_digest_mismatch")
        target = destination / binding["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render-configmap", "render-job", "materialize"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--projected", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "render-configmap":
        print(json.dumps(build_configmap(args.repo, args.package_commit), sort_keys=True))
    elif args.command == "render-job":
        print(_git_file(args.repo, args.package_commit, MANIFEST_PATH).decode(), end="")
    else:
        if args.projected is None or args.destination is None:
            parser.error("materialize requires --projected and --destination")
        print(materialize(args.projected, args.destination, args.package_commit)["package_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
