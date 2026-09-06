"""Preview or create-once submit the released peer-free rank-30 controller."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_peer_free_package_v2 as controller_package
from evals.fleet import hosted_glm_rank30_peer_free_release_observer_v1 as observer
from evals.fleet import hosted_glm_rank30_peer_free_release_package_v1 as release_package
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor

SCHEMA = "fleet-hosted-glm-rank30-peer-free-submit-v1"
PREVIEW_SCHEMA = "fleet-hosted-glm-rank30-peer-free-create-preview-v1"
SUBMIT_RECEIPT = Path("/mnt/sfs/jobs/chris-glm53-peer-free-submit-v1/SUBMITTED.json")
PREVIEW_MAX_AGE_SECONDS = 60
RECONCILIATION_ZERO_FIELDS = (
    "all_generation_claim_collisions",
    "authoritative_session_collisions",
    "accepted_evidence_collisions",
    "output_root_collisions",
    "new_job_collisions",
    "new_pod_collisions",
    "new_configmap_collisions",
    "active_hosted_controllers",
)


class SubmitError(RuntimeError):
    """The create-once release or submit gate failed closed."""


def preview_digest(value: dict[str, Any]) -> str:
    return observer.sha256(
        observer.canonical({key: item for key, item in value.items() if key != "preview_sha256"})
    )


def _load_release(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > observer.MAX_RECEIPT_BYTES:
        raise SubmitError("release_path_unsafe")
    return observer.strict_json(path.read_bytes())


def _runtime_plan(root: Path) -> tuple[dict[str, Any], str]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    source_sha = controller_package.source_package_sha256(root)
    return plan, source_sha


def validate_release(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    release = _load_release(path)
    plan, source_sha = _runtime_plan(root)
    successor.validate_release(release, plan, source_sha)
    binding = release_package.build_binding(root)
    if binding["held_source_package_sha256"] != source_sha:
        raise SubmitError("release_source_binding_drifted")
    return release, binding


def _validate_immediate_reconciliation(value: dict[str, Any]) -> None:
    if any(value.get(key) != 0 for key in RECONCILIATION_ZERO_FIELDS) or any(
        (
            value.get("endpoint_lease_slots_available") != 2,
            value.get("both_endpoint_lease_slots_simultaneously_free") is not True,
        )
    ):
        raise SubmitError("immediate_create_reconciliation_failed")


def render(root: Path, release_path: Path, *, api_key: str) -> dict[str, Any]:
    release, binding = validate_release(release_path, root)
    current = observer.recheck(binding, api_key=api_key)
    _validate_immediate_reconciliation(current)
    rendered = controller_package.render(root, release_value=release)
    objects = rendered.get("objects") or {}
    items = objects.get("items") if isinstance(objects, dict) else None
    if any(
        (
            rendered.get("launch_authorized") is not True,
            rendered.get("scoring_authorized") is not True,
            not isinstance(items, list),
            len(items or []) != 2,
            [row.get("kind") for row in items or []] != ["ConfigMap", "Job"],
            [row.get("metadata", {}).get("name") for row in items or []]
            != [successor.CONFIGMAP_NAME, successor.JOB_NAME],
            items[0].get("immutable") is not True if items else True,
            items[1]
            .get("metadata", {})
            .get("annotations", {})
            .get("cyber-post-train.fleet.ai/create-once")
            != "true"
            if items
            else True,
        )
    ):
        raise SubmitError("rendered_create_objects_invalid")
    body = {
        "schema_version": PREVIEW_SCHEMA,
        "apiVersion": "v1",
        "kind": "List",
        "items": items,
        "release_receipt_sha256": release["receipt_sha256"],
        "immediate_reconciliation": current,
        "previewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "launch_authorized": True,
        "scoring_authorized": True,
    }
    return {**body, "preview_sha256": preview_digest(body)}


def validate_preview(manifest: dict[str, Any], *, root: Path, api_key: str) -> None:
    if any(
        (
            set(manifest)
            != {
                "schema_version",
                "apiVersion",
                "kind",
                "items",
                "release_receipt_sha256",
                "immediate_reconciliation",
                "previewed_at_utc",
                "launch_authorized",
                "scoring_authorized",
                "preview_sha256",
            },
            manifest.get("schema_version") != PREVIEW_SCHEMA,
            manifest.get("apiVersion") != "v1",
            manifest.get("kind") != "List",
            manifest.get("launch_authorized") is not True,
            manifest.get("scoring_authorized") is not True,
            observer.SHA_RE.fullmatch(str(manifest.get("release_receipt_sha256"))) is None,
            manifest.get("preview_sha256") != preview_digest(manifest),
        )
    ):
        raise SubmitError("create_preview_invalid")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 2:
        raise SubmitError("create_preview_objects_invalid")
    if [item.get("kind") for item in items if isinstance(item, dict)] != ["ConfigMap", "Job"]:
        raise SubmitError("create_preview_objects_invalid")
    configmap, job = items
    config_data = configmap.get("data") or {}
    try:
        embedded_release = observer.strict_json(str(config_data["release.json"]).encode())
    except (KeyError, TypeError, observer.ObserverError) as exc:
        raise SubmitError("create_preview_embedded_release_invalid") from exc
    try:
        expected = controller_package.render(root, release_value=embedded_release)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SubmitError("create_preview_expected_objects_invalid") from exc
    expected_objects = expected.get("objects")
    expected_items = expected_objects.get("items") if isinstance(expected_objects, dict) else None
    if any(
        (
            expected.get("launch_authorized") is not True,
            expected.get("scoring_authorized") is not True,
            not isinstance(expected_items, list),
            items != expected_items,
            configmap.get("apiVersion") != "v1",
            configmap.get("metadata", {}).get("name") != successor.CONFIGMAP_NAME,
            configmap.get("metadata", {}).get("namespace") != observer.NAMESPACE,
            configmap.get("immutable") is not True,
            embedded_release.get("receipt_sha256") != manifest.get("release_receipt_sha256"),
            job.get("apiVersion") != "batch/v1",
            job.get("metadata", {}).get("name") != successor.JOB_NAME,
            job.get("metadata", {}).get("namespace") != observer.NAMESPACE,
            job.get("metadata", {})
            .get("annotations", {})
            .get("cyber-post-train.fleet.ai/create-once")
            != "true",
            job.get("metadata", {})
            .get("annotations", {})
            .get("cyber-post-train.fleet.ai/launch-authorized")
            != "true",
            job.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("volumes", [{}])[0]
            .get("configMap", {})
            .get("name")
            != successor.CONFIGMAP_NAME,
        )
    ):
        raise SubmitError("create_preview_objects_invalid")
    binding = release_package.build_binding(root)
    current = observer.recheck(binding, api_key=api_key)
    _validate_immediate_reconciliation(current)
    try:
        observed = datetime.fromisoformat(str(manifest["previewed_at_utc"]).replace("Z", "+00:00"))
        age = (datetime.now(UTC) - observed).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise SubmitError("create_preview_timestamp_invalid") from None
    if age < -60 or age > PREVIEW_MAX_AGE_SECONDS:
        raise SubmitError("create_preview_stale")


def _default_runner(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "create", "-f", "-"],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def submit_once(
    manifest: dict[str, Any],
    receipt_path: Path,
    *,
    root: Path,
    api_key: str,
    runner: Callable[[str], subprocess.CompletedProcess[str]] = _default_runner,
) -> dict[str, Any]:
    if receipt_path != SUBMIT_RECEIPT:
        raise SubmitError("submit_receipt_identity_invalid")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SubmitError("submit_receipt_exists_do_not_repeat")
    validate_preview(manifest, root=root, api_key=api_key)
    payload = yaml.safe_dump(
        {key: manifest[key] for key in ("apiVersion", "kind", "items")}, sort_keys=False
    )
    result = runner(payload)
    if result.returncode != 0:
        raise SubmitError("kubectl_create_failed_or_indeterminate_do_not_retry")
    body = {
        "schema_version": SCHEMA,
        "status": "CREATE_ACCEPTED_ONCE",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "job_name": successor.JOB_NAME,
        "configmap_name": successor.CONFIGMAP_NAME,
        "release_receipt_sha256": manifest["release_receipt_sha256"],
        "create_invocations": 1,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    receipt = {**body, "receipt_sha256": observer.digest(body)}
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(receipt_path, flags, 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SubmitError("submit_receipt_not_regular")
        raw = observer.canonical(receipt) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise SubmitError("submit_receipt_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo.resolve(strict=True)
    api_key = os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        raise SubmitError("fleet_api_key_absent")
    manifest = render(root, args.release, api_key=api_key)
    if args.submit:
        receipt = submit_once(manifest, SUBMIT_RECEIPT, root=root, api_key=api_key)
        print(
            json.dumps(
                {"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]},
                sort_keys=True,
            )
        )
        return 0
    payload = yaml.safe_dump(manifest, sort_keys=False)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
