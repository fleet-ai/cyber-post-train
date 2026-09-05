"""Render exact create-only source or accept objects for one final-v5 group."""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import exact_pass4_final_duplicate_observer_v5 as observer
from evals.fleet import exact_pass4_final_duplicate_package_v5 as package
from evals.fleet import kubernetes_create_relay as relay

SCHEMA = "fleet-exact-pass4-final-duplicate-render-v5"
INTENT_SCHEMA = "fleet-exact-pass4-final-duplicate-intent-v5"


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.write(fd, raw) != len(raw):
            raise ValueError("render_write_failed")
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_job(root: Path, name: str) -> dict[str, Any]:
    held = yaml.safe_load((root / observer.MANIFEST_PATH).read_text())
    rows = [
        row
        for row in held.get("items", [])
        if isinstance(row, dict)
        and row.get("kind") == "Job"
        and (row.get("metadata") or {}).get("name") == name
    ]
    if len(rows) != 1:
        raise ValueError("exact_held_job_absent")
    return copy.deepcopy(rows[0])


def _released_job(value: dict[str, Any]) -> dict[str, Any]:
    annotations = value["metadata"].setdefault("annotations", {})
    annotations["cyber-post-train.fleet.ai/preview-only"] = "false"
    annotations["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    return value


def _configmap(name: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": observer.NAMESPACE},
        "immutable": True,
        "data": data,
    }


def _allowlist(objects: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "api_version": row["apiVersion"],
            "kind": row["kind"],
            "namespace": row["metadata"]["namespace"],
            "name": row["metadata"]["name"],
            "sha256": relay.sha256(relay.canonical_json(row)),
        }
        for row in objects
    ]
    body = {
        "schema_version": relay.ALLOWLIST_SCHEMA,
        "namespace": relay.NAMESPACE,
        "allie_dev": {"name": relay.ALLIE_NAME, "uid": relay.ALLIE_UID},
        "objects": rows,
    }
    body["allowlist_sha256"] = relay.digest_without(body, "allowlist_sha256")
    return body


def render(
    root: Path,
    release: dict[str, Any],
    group: str,
    mode: str,
) -> dict[str, Any]:
    observer.validate_release(release, root)
    if group not in observer._groups():  # noqa: SLF001 - exact release authority
        raise ValueError("group_invalid")
    if mode not in {"source", "accept"}:
        raise ValueError("mode_invalid")
    built = package.build_package(root)
    if built["aggregate_sha256"] != release["package_aggregate_sha256"]:
        raise ValueError("released_package_drifted")
    release_raw = observer.canonical(release) + b"\n"
    release_name = f"chris-final-v5-dup-release-{group}"
    intent_name = f"chris-final-v5-dup-intent-{group}"
    binding = observer.group_binding(root, group)
    intent_body = {
        "schema_version": INTENT_SCHEMA,
        "group": group,
        "package_commit": release["package_commit"],
        "package_aggregate_sha256": built["aggregate_sha256"],
        "binding": binding,
        "source_job": observer.source_job(group),
        "accept_job": observer.accept_job(group),
        "output_root": str(observer.output_root(group)),
        "create_once": True,
        "methods": ["GET"],
        "prompts_traces_flags_or_scores_included": False,
    }
    intent_body["intent_sha256"] = observer.digest(intent_body, "intent_sha256")
    job_name = observer.source_job(group) if mode == "source" else observer.accept_job(group)
    if mode == "source":
        configmaps = copy.deepcopy(built["configmaps"])
        runtime = configmaps[package.OBSERVER_NAME]
        runtime["data"]["package_aggregate_sha256"] = built["aggregate_sha256"]
        runtime["data"]["package_commit"] = release["package_commit"]
        packaged = []
        for generic, component in zip(
            (*package.CORE_NAMES, package.OBSERVER_NAME),
            ("core-a", "core-b", "core-c", "core-d", "runtime"),
            strict=True,
        ):
            value = configmaps[generic]
            value["metadata"]["name"] = observer.package_configmap(group, component)
            packaged.append(value)
        objects = [
            *packaged,
            _configmap(
                release_name,
                {
                    "release.json": release_raw.decode(),
                    "release_file_sha256": observer.sha256(release_raw),
                },
            ),
            _configmap(
                intent_name,
                {"intent.json": observer.canonical(intent_body).decode() + "\n"},
            ),
            _released_job(_load_job(root, job_name)),
        ]
    else:
        objects = [_released_job(_load_job(root, job_name))]
    allowlist = _allowlist(objects)
    envelope = relay.build_envelope(objects, allowlist)
    return {
        "schema_version": SCHEMA,
        "group": group,
        "mode": mode,
        "manifest": {"apiVersion": "v1", "kind": "List", "items": objects},
        "allowlist": allowlist,
        "envelope_sha256": envelope["envelope_sha256"],
        "object_count": len(objects),
        "launch_authorized": True,
        "prompts_traces_flags_or_scores_included": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--mode", choices=("source", "accept"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    release = observer.load(args.release)
    result = render(root, release, args.group, args.mode)
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise ValueError("output_dir_exists")
    output.mkdir(mode=0o700, parents=True)
    _write_once(output / "manifest.json", observer.canonical(result["manifest"]) + b"\n")
    _write_once(output / "allowlist.json", observer.canonical(result["allowlist"]) + b"\n")
    print(
        json.dumps(
            {
                "status": "RENDERED",
                "group": result["group"],
                "mode": result["mode"],
                "object_count": result["object_count"],
                "envelope_sha256": result["envelope_sha256"],
                "objects_created": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
