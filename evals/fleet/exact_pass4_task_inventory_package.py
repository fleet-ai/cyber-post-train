"""Build and verify the exact easiest-100 inventory observer ConfigMap.

The package is deliberately small and self-contained.  It carries only the
GET-only observer, its local Python dependencies, and the two immutable
campaign inputs.  Kubernetes projects an immutable ConfigMap; the entrypoint
validates every projected byte before reconstructing a private runtime tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-exact-pass4-task-inventory-package-v1"
CONFIGMAP_NAME = "chris-cyber-exact100-pass4-inventory-bootstrap-v1"
INTENT_NAME = "chris-cyber-exact100-pass4-inventory-intent-v1"
NAMESPACE = "fleet-train-jobs"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_PATH = "evals/fleet/cluster/exact-pass4-task-inventory-observer-v1.yaml"

PACKAGE_FILES = {
    "package.py": (
        "evals/fleet/exact_pass4_task_inventory_package.py",
        "evals/fleet/exact_pass4_task_inventory_package.py",
    ),
    "inventory.py": (
        "evals/fleet/exact_pass4_task_inventory.py",
        "evals/fleet/exact_pass4_task_inventory.py",
    ),
    "exact_universe.py": (
        "evals/fleet/exact_pass4_universe.py",
        "evals/fleet/exact_pass4_universe.py",
    ),
    "crypto.py": (
        "evals/fleet/exact_pass4_crypto.py",
        "evals/fleet/exact_pass4_crypto.py",
    ),
    "campaign.json": (
        "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
        "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
    ),
    "selection.json": (
        "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
        "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    ),
    "run.sh": (
        "evals/fleet/scripts/run_exact_pass4_task_inventory_v1.sh",
        None,
    ),
}


class PackageError(RuntimeError):
    """A stable, content-free package validation error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError("duplicate_json_key")
        result[key] = value
    return result


def _load_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError("package_json_invalid") from exc
    if not isinstance(parsed, dict):
        raise PackageError("package_json_invalid")
    return parsed


def _commit(value: str) -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise PackageError("package_commit_invalid")
    return value


def _git_file(repo_root: Path, package_commit: str, source_path: str) -> bytes:
    """Read one source exclusively from an exact Git commit snapshot."""
    root = repo_root.resolve(strict=True)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{package_commit}:{source_path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError("package_commit_source_absent") from exc
    return result.stdout


def job_manifest_bytes(repo_root: Path, package_commit: str) -> bytes:
    return _git_file(repo_root, _commit(package_commit), MANIFEST_PATH)


def build_configmap(repo_root: Path, package_commit: str) -> dict[str, Any]:
    """Render one immutable ConfigMap from the exact checked-out package bytes."""
    package_commit = _commit(package_commit)
    data: dict[str, str] = {}
    manifest_files: dict[str, dict[str, Any]] = {}
    for key, (source_path, install_path) in PACKAGE_FILES.items():
        raw = _git_file(repo_root, package_commit, source_path)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PackageError("package_source_not_utf8") from exc
        data[key] = text
        manifest_files[key] = {
            "source_path": source_path,
            "install_path": install_path,
            "sha256": sha256(raw),
        }
    package = {
        "schema_version": SCHEMA,
        "package_commit": package_commit,
        "files": manifest_files,
        "job_manifest": {
            "source_path": MANIFEST_PATH,
            "sha256": sha256(job_manifest_bytes(repo_root, package_commit)),
        },
        "runtime_contract": {
            "fleet_methods": ["GET"],
            "task_version_gets": 100,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
            "mutation_calls": 0,
            "sfs_terminal_path": (
                "/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v1/TERMINAL.json"
            ),
        },
    }
    package["package_sha256"] = digest_without(package, "package_sha256")
    data["package.json"] = canonical_json(package).decode() + "\n"
    data["package_commit"] = package_commit
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_intent(configmap: dict[str, Any]) -> dict[str, Any]:
    """Bind the one-shot submission intent to the exact immutable package."""
    if (
        configmap.get("kind") != "ConfigMap"
        or configmap.get("metadata", {}).get("name") != CONFIGMAP_NAME
        or configmap.get("immutable") is not True
    ):
        raise PackageError("package_configmap_invalid")
    package = _load_object(configmap.get("data", {}).get("package.json", ""))
    validate_package_manifest(package)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": INTENT_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": {
            "package_commit": package["package_commit"],
            "package_sha256": package["package_sha256"],
            "job_manifest_sha256": package["job_manifest"]["sha256"],
            "bootstrap_configmap": CONFIGMAP_NAME,
            "job": "chris-cyber-exact100-pass4-inventory-v1",
            "output_root": "/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v1",
        },
    }


def validate_package_manifest(package: dict[str, Any]) -> None:
    if set(package) != {
        "schema_version",
        "package_commit",
        "files",
        "job_manifest",
        "runtime_contract",
        "package_sha256",
    }:
        raise PackageError("package_shape_mismatch")
    if package.get("schema_version") != SCHEMA:
        raise PackageError("package_schema_mismatch")
    _commit(package.get("package_commit", ""))
    if package.get("package_sha256") != digest_without(package, "package_sha256"):
        raise PackageError("package_digest_mismatch")
    files = package.get("files")
    if not isinstance(files, dict) or set(files) != set(PACKAGE_FILES):
        raise PackageError("package_file_set_mismatch")
    for key, (source_path, install_path) in PACKAGE_FILES.items():
        binding = files.get(key)
        if not isinstance(binding, dict) or binding != {
            "source_path": source_path,
            "install_path": install_path,
            "sha256": binding.get("sha256") if isinstance(binding, dict) else None,
        }:
            raise PackageError("package_file_binding_mismatch")
        digest = binding.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise PackageError("package_file_digest_invalid")
    job_manifest = package.get("job_manifest")
    if (
        not isinstance(job_manifest, dict)
        or set(job_manifest) != {"source_path", "sha256"}
        or job_manifest.get("source_path") != MANIFEST_PATH
        or not isinstance(job_manifest.get("sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", job_manifest["sha256"]) is None
    ):
        raise PackageError("job_manifest_binding_invalid")
    if package.get("runtime_contract") != {
        "fleet_methods": ["GET"],
        "task_version_gets": 100,
        "model_or_scoring_calls": 0,
        "session_calls": 0,
        "mutation_calls": 0,
        "sfs_terminal_path": (
            "/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v1/TERMINAL.json"
        ),
    }:
        raise PackageError("package_runtime_contract_mismatch")


def _read_projected(projected_root: Path, key: str) -> bytes:
    root = projected_root.resolve(strict=True)
    path = projected_root / key
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PackageError("projected_package_file_absent") from exc
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise PackageError("projected_package_file_unsafe")
    return resolved.read_bytes()


def materialize_projected(
    projected_root: Path, destination: Path, *, expected_commit: str
) -> dict[str, Any]:
    """Verify projected ConfigMap bytes and reconstruct an empty runtime root."""
    expected_commit = _commit(expected_commit)
    package = _load_object(_read_projected(projected_root, "package.json").decode())
    validate_package_manifest(package)
    if package["package_commit"] != expected_commit:
        raise PackageError("projected_package_commit_mismatch")
    if _read_projected(projected_root, "package_commit").decode() != expected_commit:
        raise PackageError("projected_package_commit_mismatch")

    raw_files: dict[str, bytes] = {}
    for key, binding in package["files"].items():
        raw = _read_projected(projected_root, key)
        if sha256(raw) != binding["sha256"]:
            raise PackageError("projected_package_file_digest_mismatch")
        raw_files[key] = raw

    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    for init_path in ("evals/__init__.py", "evals/fleet/__init__.py"):
        target = destination / init_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    for key, binding in package["files"].items():
        install_path = binding["install_path"]
        if install_path is None:
            continue
        target = destination / install_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(raw_files[key])
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    return {
        "schema_version": SCHEMA,
        "package_commit": expected_commit,
        "package_sha256": package["package_sha256"],
        "installed_file_count": sum(
            binding["install_path"] is not None for binding in package["files"].values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-configmap")
    render.add_argument("--repo", type=Path, required=True)
    render.add_argument("--package-commit", required=True)
    intent = subparsers.add_parser("render-intent")
    intent.add_argument("--configmap", type=Path, required=True)
    manifest = subparsers.add_parser("render-job-manifest")
    manifest.add_argument("--repo", type=Path, required=True)
    manifest.add_argument("--package-commit", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--projected", type=Path, required=True)
    materialize.add_argument("--destination", type=Path, required=True)
    materialize.add_argument("--package-commit", required=True)
    args = parser.parse_args()
    if args.command == "render-configmap":
        value = build_configmap(args.repo, args.package_commit)
    elif args.command == "render-intent":
        value = build_intent(_load_object(args.configmap.read_text()))
    elif args.command == "render-job-manifest":
        sys.stdout.buffer.write(job_manifest_bytes(args.repo, args.package_commit))
        return 0
    else:
        value = materialize_projected(
            args.projected, args.destination, expected_commit=args.package_commit
        )
    print(canonical_json(value).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
