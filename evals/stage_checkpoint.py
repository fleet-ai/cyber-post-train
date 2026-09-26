"""Copy one verified SFT export to inference storage without allocating GPUs."""

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("source is missing or linked")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("source changed during hashing")
    return digest.hexdigest()


def receipt(path, expected):
    if sha(path) != expected:
        raise ValueError("receipt file digest differs")
    value = json.loads(path.read_bytes())
    claimed = value.get("receipt_sha256") if isinstance(value, dict) else None
    if not isinstance(claimed, str) or claimed != hashlib.sha256(canonical({
        key: item for key, item in value.items() if key != "receipt_sha256"
    })).hexdigest():
        raise ValueError("receipt self-digest differs")
    return value


def stage(source, destination, expected_ready):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("model destination already exists")
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("source and destination must be absolute")
    ready = receipt(source / "CHECKPOINT_READY.json", expected_ready)
    export = receipt(source / "bf16/EXPORT.json", ready["export_sha256"])
    gpu = receipt(source / "GPU_CHECK.json", ready["gpu_check_sha256"])
    files = export.get("files")
    if (ready.get("schema") != "qwen38_checkpoint_ready_v1"
        or ready.get("status") != "ready_for_route_parity"
        or ready.get("serving_qualified") is not False
        or ready.get("task_evaluated") is not False
        or export.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or export.get("optimizer_step") != ready.get("optimizer_step")
        or export.get("source_plan_sha256") != ready.get("plan_sha256")
        or export.get("output_root") != str(source / "bf16")
        or export.get("dtype") != "BF16"
        or export.get("optimizer_steps_executed") != 0
        or gpu.get("schema") != "cyber_hf_export_check_v1"
        or gpu.get("status") != "passed"
        or gpu.get("export_sha256") != ready["export_sha256"]
        or gpu.get("gpu_reload_verified") is not True
        or gpu.get("optimizer_steps_executed") != 0
        or gpu.get("finite_logits") is not True
        or not isinstance(files, dict) or len(files) != 29
        or hashlib.sha256(canonical(files)).hexdigest() != ready["export_payload_sha256"]):
        raise ValueError("checkpoint readiness bindings differ")
    total = 0
    for name, spec in files.items():
        if (not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}
            or not isinstance(spec, dict) or type(spec.get("bytes")) is not int
            or not 0 <= spec["bytes"] <= 4 << 30
            or not isinstance(spec.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", spec["sha256"])):
            raise ValueError("unsafe export file manifest")
        total += spec["bytes"]
    if not 50_000_000_000 < total < 65_000_000_000:
        raise ValueError("unexpected export size")
    if shutil.disk_usage(destination.parent).free < total + (1 << 30):
        raise ValueError("insufficient destination free space")
    partial = destination.with_name("." + destination.name + ".partial")
    partial.mkdir(mode=0o755, exist_ok=False)
    for name, spec in sorted(files.items()):
        original = source / "bf16" / name
        if original.stat().st_size != spec["bytes"] or sha(original) != spec["sha256"]:
            raise ValueError("export payload changed before copy")
        target = partial / name
        digest = hashlib.sha256()
        with original.open("rb") as inp, target.open("xb") as out:
            for block in iter(lambda: inp.read(8 << 20), b""):
                out.write(block)
                digest.update(block)
            out.flush()
            os.fsync(out.fileno())
        if target.stat().st_size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise ValueError("staged payload digest differs")
    acceptance = {"schema": "qwen38_inference_stage_v1", "ready_file_sha256": expected_ready,
                  "export_file_sha256": ready["export_sha256"],
                  "payload_sha256": ready["export_payload_sha256"],
                  "file_count": len(files), "total_bytes": total}
    acceptance["receipt_sha256"] = hashlib.sha256(canonical(acceptance)).hexdigest()
    with (partial / ".fleet-acceptance.json").open("xb") as out:
        out.write(canonical(acceptance) + b"\n")
        out.flush()
        os.fsync(out.fileno())
    parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(parent, partial.name.encode(), parent, destination.name.encode(), 1):
            raise OSError(ctypes.get_errno(), "atomic no-replace rename failed")
        os.fsync(parent)
    finally:
        os.close(parent)
    print(json.dumps({"status": "staged", "file_count": len(files), "total_bytes": total,
                      "acceptance_sha256": acceptance["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("ready_file_sha256")
    arguments = parser.parse_args()
    stage(arguments.source, arguments.destination, arguments.ready_file_sha256)
