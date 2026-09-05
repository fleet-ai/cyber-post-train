"""Render the immutable held-only hosted concurrency qualification package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "fleet-hosted-concurrency4-qualification-package-v1"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-cyber-hosted-c4-qualification-v1"
CONFIGMAP_NAME = "chris-cyber-hosted-c4-qualification-bootstrap-v1"
INTENT_NAME = "chris-cyber-hosted-c4-qualification-intent-v1"
MANIFEST_PATH = "evals/fleet/cluster/hosted-concurrency4-qualification-held-v1.yaml"
OUT_ROOT = "/mnt/sfs/jobs/chris-cyber-hosted-c4-qualification-v1"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = {
    "package.py": "evals/fleet/hosted_concurrency4_qualification_package_v1.py",
    "runner.py": "evals/fleet/hosted_concurrency4_qualification_v1.py",
    "run.sh": "evals/fleet/scripts/run_hosted_concurrency4_qualification_v1.sh",
}


class PackageError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def _commit(value: str) -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise PackageError("package_commit_invalid")
    return value


def _git_file(root: Path, commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root.resolve(strict=True)), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError("package_source_absent") from exc


def build_configmap(root: Path, package_commit: str) -> dict[str, Any]:
    package_commit = _commit(package_commit)
    data: dict[str, str] = {}
    entries: dict[str, dict[str, Any]] = {}
    for key, source in FILES.items():
        raw = _git_file(root, package_commit, source)
        data[key] = raw.decode("utf-8")
        entries[key] = {"source_path": source, "sha256": sha256(raw)}
    manifest_raw = _git_file(root, package_commit, MANIFEST_PATH)
    package = {
        "schema_version": SCHEMA,
        "package_commit": package_commit,
        "files": entries,
        "job_manifest": {"source_path": MANIFEST_PATH, "sha256": sha256(manifest_raw)},
        "runtime_contract": {
            "hosted_origin": "https://inference.flt.build",
            "models": ["qwen3.8-27b", "glm-5.3"],
            "concurrency_waves_per_model": [2, 4],
            "chat_completion_requests": 24,
            "task_instance_session_scoring_or_verifier_calls": 0,
            "output_root": OUT_ROOT,
            "lease_namespace": "hosted-concurrency-qualification-v1",
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "launch_authorized": False,
        "scored_bulk_launch_authorized": False,
    }
    package["package_sha256"] = digest_without(package, "package_sha256")
    data["package.json"] = canonical_json(package).decode() + "\n"
    data["package_commit"] = package_commit
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


def _load_package(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PackageError("package_shape_invalid")
    if (
        value.get("schema_version") != SCHEMA
        or value.get("launch_authorized") is not False
        or value.get("scored_bulk_launch_authorized") is not False
        or value.get("package_sha256") != digest_without(value, "package_sha256")
    ):
        raise PackageError("package_validation_failed")
    return value


def build_intent(configmap: dict[str, Any]) -> dict[str, Any]:
    package = _load_package(configmap["data"]["package.json"])
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": INTENT_NAME,
            "namespace": NAMESPACE,
            "annotations": {
                "cyber-post-train.fleet.ai/preview-only": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "false",
            },
        },
        "immutable": True,
        "data": {
            "job": JOB_NAME,
            "bootstrap_configmap": CONFIGMAP_NAME,
            "package_commit": package["package_commit"],
            "package_sha256": package["package_sha256"],
            "job_manifest_sha256": package["job_manifest"]["sha256"],
            "output_root": OUT_ROOT,
            "launch_authorized": "false",
            "scored_bulk_launch_authorized": "false",
        },
    }


def materialize(projected: Path, destination: Path, expected_commit: str) -> dict[str, Any]:
    package = _load_package((projected / "package.json").read_text())
    if package["package_commit"] != _commit(expected_commit):
        raise PackageError("projected_commit_mismatch")
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
    parser.add_argument(
        "command", choices=("render-configmap", "render-intent", "render-job", "materialize")
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--configmap", type=Path)
    parser.add_argument("--projected", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "render-configmap":
        print(json.dumps(build_configmap(args.repo, args.package_commit), sort_keys=True))
    elif args.command == "render-intent":
        if args.configmap is None:
            parser.error("render-intent requires --configmap")
        print(json.dumps(build_intent(json.loads(args.configmap.read_text())), sort_keys=True))
    elif args.command == "render-job":
        os.write(1, _git_file(args.repo, _commit(args.package_commit), MANIFEST_PATH))
    else:
        if args.projected is None or args.destination is None:
            parser.error("materialize requires --projected and --destination")
        package = materialize(args.projected, args.destination, args.package_commit)
        print(package["package_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
