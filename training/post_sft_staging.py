"""Fail-closed, CPU-only transport and composition of the final SFT inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .io import digest_json, file_sha256
from .post_sft_artifacts import inspect_hf_export

STAGE_SCHEMA = "cyber_sft_inference_stage_input_v1"
RECEIPT_SCHEMA = "cyber_sft_inference_stage_receipt_v1"
FILEBROWSER_ORIGIN = "http://filebrowser.fleet-train-data-plane.svc.cluster.local"
EXPORT_SOURCE = "/exports/cyber-sft/ft-run-574bd7b3/step-318-v1/global_step_318/policy"
DESTINATION = "/models/cyber-sft/ft-run-574bd7b3/step-318"
BASE_ROOT = "/models/qwen3.6-27b/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != "cyber_sft_full_file_manifest_v1":
        raise ValueError("raw export manifest has an unsupported schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("raw export manifest files must be an array of objects")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("raw export manifest paths must be unique and non-empty")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("raw export manifest contains an unsafe path")
        if not isinstance(size, int) or size < 0:
            raise ValueError("raw export manifest contains an invalid size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("raw export manifest contains an invalid SHA-256")
        int(digest, 16)
        seen.add(name)
        normalized.append({"path": name, "size": size, "sha256": digest})
    normalized.sort(key=lambda row: row["path"])
    if manifest.get("file_count") != len(normalized):
        raise ValueError("raw export manifest file_count is inconsistent")
    if manifest.get("total_bytes") != sum(row["size"] for row in normalized):
        raise ValueError("raw export manifest total_bytes is inconsistent")
    if manifest.get("manifest_sha256") != digest_json(normalized):
        raise ValueError("raw export manifest digest does not validate")
    return normalized


def verify_full_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read each file once and require exact path, size, and SHA-256 equality."""

    resolved = root.resolve(strict=True)
    expected = _manifest_rows(manifest)
    observed_paths = sorted(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if any(path.is_symlink() for path in resolved.rglob("*")):
        raise ValueError("staged export must not contain symlinks")
    if observed_paths != [row["path"] for row in expected]:
        raise ValueError("staged export paths differ from the source manifest")
    for row in expected:
        path = resolved / row["path"]
        if path.stat().st_size != row["size"] or file_sha256(path) != (
            "sha256:" + row["sha256"]
        ):
            raise ValueError(f"staged export content differs for {row['path']}")
    return {
        "file_count": len(expected),
        "total_bytes": sum(row["size"] for row in expected),
        "manifest_sha256": digest_json(expected),
    }


def _safe_extract_zip(archive: Path, output: Path) -> Path:
    output.mkdir(mode=0o700)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members:
            raise ValueError("filebrowser returned an empty ZIP archive")
        for member in members:
            pure = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if pure.is_absolute() or ".." in pure.parts or stat.S_ISLNK(mode):
                raise ValueError("filebrowser ZIP contains an unsafe member")
        bundle.extractall(output)
    if (output / "model.safetensors.index.json").is_file():
        return output
    roots = [path for path in output.iterdir() if path.is_dir()]
    files = [path for path in output.iterdir() if path.is_file()]
    if len(roots) == 1 and not files and (roots[0] / "model.safetensors.index.json").is_file():
        return roots[0]
    raise ValueError("filebrowser ZIP does not contain one recognizable HF export root")


def _download_export(url: str, destination: Path, forwarded_user: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"X-Forwarded-User": forwarded_user})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as stream:
        if response.status != 200:
            raise ValueError(f"filebrowser returned HTTP {response.status}")
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    if size == 0:
        raise ValueError("filebrowser returned an empty response")
    return {"zip_bytes": size, "zip_sha256": "sha256:" + digest.hexdigest()}


def _weight_files(raw_root: Path) -> list[str]:
    index = _read(raw_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("raw export has no safetensors weight map")
    names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and PurePosixPath(name).name == name for name in names):
        raise ValueError("raw export weight map contains an unsafe shard name")
    return ["model.safetensors.index.json", *names]


def compose_bundle(
    raw_root: Path,
    base_root: Path,
    output: Path,
    runtime_sidecars: Mapping[str, str],
) -> None:
    """Compose post weights/index with exact base runtime sidecars into an empty directory."""

    output.mkdir(mode=0o700)
    names = _weight_files(raw_root)
    if set(names) & set(runtime_sidecars):
        raise ValueError("runtime sidecars overlap the model weights/index")
    for name in names:
        source = raw_root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"raw export is missing regular weight file {name}")
        shutil.copyfile(source, output / name)
    for name, expected_sha256 in sorted(runtime_sidecars.items()):
        if PurePosixPath(name).name != name:
            raise ValueError("runtime sidecar names must be top-level files")
        source = base_root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"base checkpoint is missing regular sidecar {name}")
        if file_sha256(source) != expected_sha256:
            raise ValueError(f"base runtime sidecar hash differs for {name}")
        shutil.copyfile(source, output / name)


def build_stage_input(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    *,
    tokenizer_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind the post-export evidence needed by the inference-side transport job."""

    if observation.get("schema") != "fleet_sft_sfs_checkpoint_observation_v1":
        raise ValueError("unsupported post-export observation schema")
    if observation.get("run_name") != "ft-run-574bd7b3" or observation.get("step") != 318:
        raise ValueError("post-export observation names the wrong checkpoint")
    inspection = observation.get("output_inspection")
    if not isinstance(inspection, Mapping):
        raise ValueError("post-export observation has no HF output inspection")
    expected_source = plan["export"]["expected_output_path"]
    if inspection.get("root") != expected_source:
        raise ValueError("HF output inspection names an unexpected raw export path")
    rows = _manifest_rows(raw_manifest)
    if raw_manifest.get("root") != expected_source:
        raise ValueError("raw export manifest names an unexpected source path")
    if inspection.get("files_manifest_sha256") != raw_manifest.get("manifest_sha256"):
        raise ValueError("raw export inspection and full manifest disagree")
    if observation.get("raw_export_full_manifest_sha256") != raw_manifest.get(
        "manifest_sha256"
    ) or observation.get("raw_export_file_count") != raw_manifest.get("file_count"):
        raise ValueError("checkpoint observation does not bind the supplied raw export manifest")
    if observation.get("raw_export_total_bytes") != raw_manifest.get("total_bytes"):
        raise ValueError("checkpoint observation raw export byte count differs")
    evidence_binding = plan["base_model"]["tokenizer_equivalence_evidence"]
    if tokenizer_evidence_sha256 != evidence_binding.get("sha256"):
        raise ValueError("tokenizer equivalence evidence differs from the frozen plan")
    observation_sha256 = observation.get("observation_sha256")
    if not isinstance(observation_sha256, str) or not observation_sha256.startswith("sha256:"):
        raise ValueError("post-export observation must be digest-bound")
    undigested = {key: value for key, value in observation.items() if key != "observation_sha256"}
    if digest_json(undigested) != observation_sha256:
        raise ValueError("post-export observation digest does not validate")
    source_url = FILEBROWSER_ORIGIN + "/api/resources/download?" + urllib.parse.urlencode(
        {"file": EXPORT_SOURCE, "source": "sfs"}
    )
    stage_input = {
        "schema": STAGE_SCHEMA,
        "source": {
            "sfs_path": expected_source,
            "filebrowser_url": source_url,
            "raw_full_manifest": dict(raw_manifest),
            "raw_inspection": dict(inspection),
            "observation_sha256": observation_sha256,
        },
        "composition": {
            "base_root": BASE_ROOT,
            "runtime_sidecar_sha256": plan["base_model"]["runtime_sidecar_sha256"],
            "tokenizer_equivalence_evidence_sha256": tokenizer_evidence_sha256,
            "expected_parameter_count": plan["base_model"]["parameter_count"],
            "expected_tokenizer_manifest_sha256": plan["base_model"][
                "tokenizer_manifest_sha256"
            ],
            "expected_chat_template_sha256": plan["base_model"]["chat_template_sha256"],
            "expected_config_sha256": plan["base_model"]["config_sha256"],
        },
        "destination": {"path": DESTINATION, "must_be_absent": True},
        "raw_file_count": len(rows),
    }
    stage_input["stage_input_sha256"] = digest_json(stage_input)
    return stage_input


def execute_stage(stage_input: Mapping[str, Any], *, work_root: Path, forwarded_user: str) -> dict:
    """Download, verify, compose, re-hash, and atomically promote one exact bundle."""

    if stage_input.get("schema") != STAGE_SCHEMA:
        raise ValueError("unsupported inference stage input schema")
    expected_input_sha256 = stage_input.get("stage_input_sha256")
    undigested = {key: value for key, value in stage_input.items() if key != "stage_input_sha256"}
    if digest_json(undigested) != expected_input_sha256:
        raise ValueError("inference stage input digest does not validate")
    if not forwarded_user.strip():
        raise ValueError("FILEBROWSER_USER is required")
    source = _mapping(stage_input.get("source"), "source")
    composition = _mapping(stage_input.get("composition"), "composition")
    destination = _mapping(stage_input.get("destination"), "destination")
    final = Path(str(destination.get("path")))
    if final != Path(DESTINATION) or destination.get("must_be_absent") is not True:
        raise ValueError("inference destination differs from the frozen path")
    if final.exists():
        raise ValueError("inference destination already exists; refusing to overwrite")
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    partial = parent / f".partial-{final.name}-{expected_input_sha256.removeprefix('sha256:')[:12]}"
    if partial.exists():
        raise ValueError("inference partial destination already exists; refusing to reuse it")
    base_root = Path(str(composition.get("base_root"))).resolve(strict=True)
    if base_root != Path(BASE_ROOT):
        raise ValueError("composition base root differs from the frozen revision")

    work_root.mkdir(parents=True, exist_ok=False)
    archive = work_root / "raw-export.zip"
    extracted = work_root / "raw-export"
    transport = _download_export(str(source.get("filebrowser_url")), archive, forwarded_user)
    raw_root = _safe_extract_zip(archive, extracted)
    raw_manifest = _mapping(source.get("raw_full_manifest"), "raw full manifest")
    raw_verified = verify_full_manifest(raw_root, raw_manifest)
    if raw_verified["manifest_sha256"] != source.get("raw_inspection", {}).get(
        "files_manifest_sha256"
    ):
        raise ValueError("downloaded raw export differs from its inspection receipt")

    compose_bundle(
        raw_root,
        base_root,
        partial,
        _mapping(composition.get("runtime_sidecar_sha256"), "runtime sidecar hashes"),
    )
    composed = inspect_hf_export(
        partial,
        expected_tokenizer_manifest_sha256=str(
            composition.get("expected_tokenizer_manifest_sha256")
        ),
        expected_chat_template_sha256=str(composition.get("expected_chat_template_sha256")),
        expected_config_sha256=str(composition.get("expected_config_sha256")),
        expected_parameter_count=int(composition.get("expected_parameter_count")),
        expected_sidecar_sha256=dict(composition["runtime_sidecar_sha256"]),
        require_base_sidecars=True,
    )
    raw_inspection = _mapping(source.get("raw_inspection"), "raw inspection")
    if composed["weights_manifest_sha256"] != raw_inspection.get("weights_manifest_sha256"):
        raise ValueError("composed bundle weights differ from the raw export")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage_input_sha256": expected_input_sha256,
        "source_observation_sha256": source.get("observation_sha256"),
        "source_raw_manifest_sha256": raw_verified["manifest_sha256"],
        "transport": transport,
        "composition": {
            "policy": "raw_post_weights_and_index_plus_exact_base_runtime_sidecars_v1",
            "base_root": str(base_root),
            "tokenizer_equivalence_evidence_sha256": composition.get(
                "tokenizer_equivalence_evidence_sha256"
            ),
            "inspection": composed,
        },
        "destination": {"path": str(final), "atomic_promotion": True},
    }
    receipt["staging_receipt_sha256"] = digest_json(receipt)
    receipt_path = partial / ".fleet-acceptance.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, final)
    if not (final / ".fleet-acceptance.json").is_file():
        raise ValueError("atomic promotion did not preserve the acceptance receipt")
    return receipt


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--observation", type=Path, required=True)
    build.add_argument("--raw-manifest", type=Path, required=True)
    build.add_argument("--tokenizer-evidence", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    if args.command == "build-input":
        result = build_stage_input(
            _read(args.plan),
            _read(args.observation),
            _read(args.raw_manifest),
            tokenizer_evidence_sha256=file_sha256(args.tokenizer_evidence),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    work_root = args.work_root
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="post-sft-stage-"))
        work_root.rmdir()
    receipt = execute_stage(
        _read(args.input),
        work_root=work_root,
        forwarded_user=os.environ.get("FILEBROWSER_USER", ""),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
