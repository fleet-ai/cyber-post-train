"""Fail-closed immutable base-model materialization for training SFS."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "cyber_model_stage_bridge_plan_v1"
RECEIPT_SCHEMA = "cyber_model_stage_bridge_acceptance_v1"
CHECKPOINT_LOCK_SCHEMA = "cyber_post_train_checkpoint_lock_v1"
RENAME_NOREPLACE = 1
EXPECTED_STORAGE_TOPOLOGY = {
    "inference_claim": {
        "namespace": "inference",
        "name": "hf-cache-shared",
        "uid": "04703d36-2cec-4146-a35c-36f457462012",
        "volume_name": "pvc-04703d36-2cec-4146-a35c-36f457462012",
        "pv_uid": "d6f4aae9-de43-4909-acc9-3abd8c0c719a",
    },
    "training_claim": {
        "namespace": "fleet-train-jobs",
        "name": "sfs-shared",
        "uid": "34cb6b11-8766-4294-9f9e-332064ea17d5",
        "volume_name": "sfs-shared-fleet-train-jobs",
        "pv_uid": "fabf4936-1034-47ec-b07a-dd770e94fa23",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _as_digest(value: str) -> str:
    return value if value.startswith("sha256:") else "sha256:" + value


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}") from error
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return observed


def _exact_tree_inventory(root: Path, expected_files: set[str]) -> None:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as error:
        raise ValueError("staged root is missing") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("staged root must be a real directory")
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(root).as_posix()
                observed = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(observed.st_mode):
                    raise ValueError(f"staged tree contains symlink: {relative}")
                if stat.S_ISREG(observed.st_mode):
                    observed_files.add(relative)
                elif stat.S_ISDIR(observed.st_mode):
                    if relative not in expected_directories:
                        raise ValueError(f"staged tree contains unexpected directory: {relative}")
                    observed_directories.add(relative)
                    visit(Path(entry.path))
                else:
                    raise ValueError(f"staged tree contains non-regular member: {relative}")

    visit(root)
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("staged tree type inventory differs from the frozen plan")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _digest_field(value: Mapping[str, Any], field: str) -> None:
    expected = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if not isinstance(expected, str) or expected != digest_json(unsigned):
        raise ValueError(f"{field} does not validate")


def load_and_validate(plan_path: Path, lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _object(json.loads(plan_path.read_text()), "stage plan")
    lock = _object(json.loads(lock_path.read_text()), "model lock")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported stage-plan schema")
    if plan.get("status") not in {"dry_run_only_unapproved", "approved_for_execution"}:
        raise ValueError("stage plan has an unsupported approval status")
    _digest_field(plan, "plan_sha256")
    if lock.get("schema") != "huggingface_model_lock_v1":
        raise ValueError("unsupported model-lock schema")
    if plan.get("model_lock_file_sha256") != file_sha256(lock_path):
        raise ValueError("model-lock file bytes differ from the frozen stage plan")
    source = _object(plan.get("source"), "source")
    weights = _object(lock.get("weights"), "model-lock weights")
    tokenizer = _object(lock.get("tokenizer"), "model-lock tokenizer")
    config = _object(lock.get("configuration"), "model-lock configuration")
    expected = {
        "repository": lock.get("repo"),
        "revision": lock.get("revision"),
        "source_api": lock.get("source_api"),
        "weight_shards": weights.get("shards"),
        "weight_bytes": weights.get("bytes"),
        "weights_manifest_sha256": weights.get("manifest_sha256"),
        "tokenizer_manifest_sha256": tokenizer.get("manifest_sha256"),
    }
    if source != expected:
        raise ValueError("stage source differs from the immutable model lock")
    files = _object(plan.get("files"), "files")
    expected_small = {
        "LICENSE": _as_digest(plan["license_sha256"]),
        "chat_template.jinja": _as_digest(
            next(
                row["sha256"] for row in tokenizer["files"] if row["path"] == "chat_template.jinja"
            )
        ),
        "config.json": _as_digest(config["config_sha256"]),
        "generation_config.json": _as_digest(config["generation_config_sha256"]),
        "merges.txt": _as_digest(
            next(row["sha256"] for row in tokenizer["files"] if row["path"] == "merges.txt")
        ),
        "model.safetensors.index.json": _as_digest(weights["index_sha256"]),
        "preprocessor_config.json": _as_digest(config["preprocessor_config_sha256"]),
        "tokenizer.json": _as_digest(
            next(row["sha256"] for row in tokenizer["files"] if row["path"] == "tokenizer.json")
        ),
        "tokenizer_config.json": _as_digest(
            next(
                row["sha256"]
                for row in tokenizer["files"]
                if row["path"] == "tokenizer_config.json"
            )
        ),
        "video_preprocessor_config.json": _as_digest(config["video_preprocessor_config_sha256"]),
        "vocab.json": _as_digest(
            next(row["sha256"] for row in tokenizer["files"] if row["path"] == "vocab.json")
        ),
    }
    if files != expected_small or any(
        not str(value).startswith("sha256:") for value in files.values()
    ):
        raise ValueError("stage sidecar file map differs from the immutable model lock")
    destination = _object(plan.get("destination"), "destination")
    if destination.get("atomic_transaction") != "directory_rename_noreplace_v1":
        raise ValueError("destination must use atomic no-replace publication")
    root = Path(str(destination.get("immutable_root") or ""))
    if not root.is_absolute() or root.name != source["revision"]:
        raise ValueError("destination root is not the exact immutable revision path")
    execution = _object(plan.get("execution"), "execution")
    resources = _object(execution.get("resources"), "execution resources")
    if (
        execution.get("namespace") != "fleet-train-jobs"
        or execution.get("queue") != "training-lq"
        or execution.get("suspend") is not True
        or resources.get("gpus") != 0
        or "@sha256:" not in str(execution.get("image") or "")
    ):
        raise ValueError("execution is not a suspended, digest-pinned, zero-GPU training-queue job")
    if execution.get("code_sha256") != file_sha256(Path(__file__)):
        raise ValueError("executor code bytes differ from the frozen plan")
    topology = _object(plan.get("storage_topology"), "storage topology")
    if (
        topology.get("claims_are_distinct") is not True
        or topology.get("direct_cross_namespace_mount_supported") is not False
        or topology.get("selected_bridge") != "immutable_upstream_rematerialization"
    ):
        raise ValueError("storage topology did not select the fail-closed bridge")
    claims = {
        label: _object(topology.get(label), label)
        for label in ("inference_claim", "training_claim")
    }
    for label, expected_claim in EXPECTED_STORAGE_TOPOLOGY.items():
        claim = claims[label]
        if any(claim.get(field) != value for field, value in expected_claim.items()):
            raise ValueError(f"{label} identity differs from the reviewed storage topology")
        claim_ref = _object(claim.get("pv_claim_ref"), f"{label} PV claimRef")
        if claim_ref != {
            "namespace": claim["namespace"],
            "name": claim["name"],
            "uid": claim["uid"],
        }:
            raise ValueError(f"{label} PV claimRef does not bind the exact PVC")
        for digest_field in ("snapshot_sha256", "pv_snapshot_sha256"):
            digest = str(claim.get(digest_field) or "")
            if len(digest) != 71 or not digest.startswith("sha256:"):
                raise ValueError(f"{label} {digest_field} is invalid")
    inference, training = claims["inference_claim"], claims["training_claim"]
    for field in ("uid", "volume_name", "pv_uid"):
        if inference[field] == training[field]:
            raise ValueError(f"storage claims are not distinct by {field}")
    return plan, lock


def fetch_shard_manifest(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _object(plan.get("source"), "source")
    with urllib.request.urlopen(str(source["source_api"]), timeout=60) as response:
        tree = json.load(response)
    count = int(source["weight_shards"])
    suffix = f"-of-{count:05d}.safetensors"
    rows = sorted(
        (
            {"path": row["path"], "size": row["size"], "sha256": row["lfs"]["oid"]}
            for row in tree
            if row.get("path", "").startswith("model-") and row.get("path", "").endswith(suffix)
        ),
        key=lambda row: row["path"],
    )
    names = [f"model-{index:05d}-of-{count:05d}.safetensors" for index in range(1, count + 1)]
    if [row["path"] for row in rows] != names:
        raise ValueError("remote shard names differ from the frozen model shape")
    if sum(int(row["size"]) for row in rows) != source["weight_bytes"]:
        raise ValueError("remote shard bytes differ from the frozen model lock")
    if digest_json(rows) != source["weights_manifest_sha256"]:
        raise ValueError("remote shard manifest differs from the frozen model lock")
    return rows


def _payload_manifest(
    root: Path, plan: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {
        str(name): str(value) for name, value in _object(plan.get("files"), "files").items()
    }
    expected.update({str(row["path"]): "sha256:" + str(row["sha256"]) for row in shards})
    permitted = set(expected) | {"source-tree.json", ".cyber-post-train-lock.json"}
    receipt_path = root / ".fleet-stage-receipt.json"
    try:
        receipt_stat = receipt_path.lstat()
    except FileNotFoundError:
        receipt_stat = None
    if receipt_stat is not None:
        if not stat.S_ISREG(receipt_stat.st_mode):
            raise ValueError("stage receipt must be a regular non-symlink file")
        permitted.add(".fleet-stage-receipt.json")
    _exact_tree_inventory(root, permitted)
    rows: list[dict[str, Any]] = []
    for relative, digest in sorted(expected.items()):
        path = root / relative
        observed = _require_regular_file(path, f"staged file {relative}")
        actual = file_sha256(path)
        if actual != digest:
            raise ValueError(f"staged file digest mismatch: {relative}")
        rows.append({"path": relative, "size": observed.st_size, "sha256": actual})
    tree_path = root / "source-tree.json"
    lock_path = root / ".cyber-post-train-lock.json"
    _require_regular_file(tree_path, "source-tree provenance")
    _require_regular_file(lock_path, "checkpoint lock")
    if json.loads(tree_path.read_text()) != {
        "revision": plan["source"]["revision"],
        "shards": list(shards),
    }:
        raise ValueError("source-tree provenance differs from the frozen shard manifest")
    if json.loads(lock_path.read_text()) != _checkpoint_lock(plan):
        raise ValueError("checkpoint lock differs from the frozen stage plan")
    for path in (lock_path, tree_path):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    rows.sort(key=lambda row: row["path"])
    return {
        "files": rows,
        "file_count": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "manifest_sha256": digest_json(rows),
    }


def _checkpoint_lock(plan: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(plan.get("source"), "source")
    return {
        "schema": CHECKPOINT_LOCK_SCHEMA,
        "repo": source["repository"],
        "revision": source["revision"],
        "weights_manifest_sha256": source["weights_manifest_sha256"],
        "verified_shards": source["weight_shards"],
        "verified_bytes": source["weight_bytes"],
        "small_file_sha256": {
            name: digest.removeprefix("sha256:")
            for name, digest in sorted(_object(plan.get("files"), "files").items())
        },
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), RENAME_NOREPLACE)
    if result != 0:
        code = ctypes.get_errno()
        raise (
            FileExistsError(code, os.strerror(code), destination)
            if code == errno.EEXIST
            else OSError(code, os.strerror(code), destination)
        )


def _validate_existing(
    final: Path,
    plan: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    model_lock_path: Path,
) -> dict[str, Any]:
    receipt_path = final / ".fleet-stage-receipt.json"
    lock_path = final / ".cyber-post-train-lock.json"
    _exact_tree_inventory(
        final,
        {
            *plan["files"],
            *(row["path"] for row in shards),
            "source-tree.json",
            ".cyber-post-train-lock.json",
            ".fleet-stage-receipt.json",
        },
    )
    _require_regular_file(receipt_path, "stage receipt")
    _require_regular_file(lock_path, "checkpoint lock")
    receipt = _object(json.loads(receipt_path.read_text()), "stage receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("existing stage receipt schema differs")
    _digest_field(receipt, "receipt_sha256")
    if receipt.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("existing destination was produced by a different plan")
    expected_destination = {
        "path": str(final),
        "atomic_transaction": "directory_rename_noreplace_v1",
        "no_replace": True,
    }
    if (
        receipt.get("model_lock_sha256") != file_sha256(model_lock_path)
        or receipt.get("source") != plan["source"]
        or receipt.get("destination") != expected_destination
        or receipt.get("execution") != plan["execution"]
    ):
        raise ValueError("existing stage receipt bindings differ from the exact plan")
    if json.loads(lock_path.read_text()) != _checkpoint_lock(plan):
        raise ValueError("existing checkpoint lock differs from the exact plan")
    if receipt.get("payload") != _payload_manifest(final, plan, shards):
        raise ValueError("existing payload differs from its acceptance receipt")
    return receipt


def execute(plan_path: Path, lock_path: Path, work_root: Path) -> dict[str, Any]:
    plan, _lock = load_and_validate(plan_path, lock_path)
    if plan.get("status") != "approved_for_execution":
        raise PermissionError("stage plan is dry-run only and has not been approved for execution")
    shards = fetch_shard_manifest(plan)
    final = Path(plan["destination"]["immutable_root"])
    if os.path.lexists(final):
        return _validate_existing(final, plan, shards, lock_path)
    suffix = plan["plan_sha256"].removeprefix("sha256:")[:16]
    partial = work_root / f".partial-{suffix}"
    if os.path.lexists(partial):
        raise FileExistsError(f"single-use partial path already exists: {partial}")
    partial.mkdir(parents=True, mode=0o700)
    selected = sorted([*plan["files"], *(row["path"] for row in shards)])
    subprocess.run(
        [
            "hf",
            "download",
            plan["source"]["repository"],
            *selected,
            "--revision",
            plan["source"]["revision"],
            "--local-dir",
            str(partial),
            "--max-workers",
            "32",
        ],
        check=True,
    )
    cache = partial / ".cache"
    if os.path.lexists(cache):
        if stat.S_ISLNK(cache.lstat().st_mode):
            raise ValueError("downloader cache path must not be a symlink")
        shutil.rmtree(cache)
        if os.path.lexists(cache):
            raise ValueError("downloader cache removal was incomplete")
    _write_json_new(
        partial / "source-tree.json",
        {"revision": plan["source"]["revision"], "shards": list(shards)},
    )
    _write_json_new(partial / ".cyber-post-train-lock.json", _checkpoint_lock(plan))
    payload = _payload_manifest(partial, plan, shards)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "model_lock_sha256": file_sha256(lock_path),
        "source": plan["source"],
        "payload": payload,
        "destination": {
            "path": str(final),
            "atomic_transaction": "directory_rename_noreplace_v1",
            "no_replace": True,
        },
        "execution": plan["execution"],
    }
    receipt["receipt_sha256"] = digest_json(receipt)
    _write_json_new(partial / ".fleet-stage-receipt.json", receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    _rename_noreplace(partial, final)
    return _validate_existing(final, plan, shards, lock_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "remote-dry-run", "execute"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    plan, _lock = load_and_validate(args.plan, args.model_lock)
    if args.command == "validate":
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "status": "valid"}, sort_keys=True))
    elif args.command == "remote-dry-run":
        rows = fetch_shard_manifest(plan)
        print(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "shards": len(rows),
                    "weights_manifest_sha256": digest_json(rows),
                    "status": "remote_identity_valid",
                },
                sort_keys=True,
            )
        )
    else:
        if args.work_root is None:
            parser.error("execute requires --work-root")
        print(json.dumps(execute(args.plan, args.model_lock, args.work_root), sort_keys=True))


if __name__ == "__main__":
    main()
