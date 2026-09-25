"""CPU-only worker embedded verbatim in the 96k preflight Job.

The only output is a sanitized Kubernetes termination message. Never print
trainer output, paths from exceptions, or private dataset contents.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run() -> dict:
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or
        os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none" or
        os.environ.get("WANDB_MODE") != "disabled"):
        raise ValueError("isolation drift")
    chunks = {int(k.removeprefix("Q38_BUNDLE_")): v for k, v in os.environ.items()
              if k.startswith("Q38_BUNDLE_") and k.removeprefix("Q38_BUNDLE_").isdigit()}
    if not chunks or sorted(chunks) != list(range(len(chunks))) or len(chunks) > 32:
        raise ValueError("bundle chunks drift")
    blob = base64.b64decode("".join(chunks[i] for i in sorted(chunks)), validate=True)
    if hashlib.sha256(blob).hexdigest() != os.environ["Q38_BUNDLE_SHA256"]:
        raise ValueError("bundle hash drift")
    payload = json.loads(gzip.decompress(blob))
    meta, files = payload["manifest"], payload["files"]
    if (meta.get("schema") != "qwen38_cpu_preflight_bundle_v1" or
        set(files) != set(meta.get("files", {})) or
        meta.get("sha256") != hashlib.sha256(canonical({k: v for k, v in meta.items()
                                                        if k != "sha256"})).hexdigest()):
        raise ValueError("bundle manifest drift")
    run_dir = PurePosixPath(meta["output_root"])
    if (run_dir.parts[:4] != ("/", "mnt", "sfs", "jobs") or
        len(run_dir.parts) != 5 or run_dir.name != meta["run_name"]):
        raise ValueError("output identity drift")
    parent = Path(str(run_dir.parent))
    if not stat.S_ISDIR(parent.lstat().st_mode) or parent.is_symlink():
        raise ValueError("SFS root drift")
    try:
        Path(str(run_dir)).lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("output already exists")
    with tempfile.TemporaryDirectory(prefix="q38-preflight-") as temporary:
        root = Path(temporary)
        for name, text in files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or
                not name.startswith("src/") or
                hashlib.sha256(text.encode()).hexdigest() != meta["files"][name]):
                raise ValueError("source inventory drift")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        sys.path.insert(0, str(root / "src"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from training.sft import job_request, preflight

            plan = payload["plan"]
            if (hashlib.sha256(canonical(plan)).hexdigest() != meta["plan_sha256"] or
                hashlib.sha256(canonical(job_request(plan))).hexdigest() != meta["request_sha256"]):
                raise ValueError("prepared plan drift")
            native = preflight(plan)
    if (native.get("status") != "passed" or native.get("gpus") != 0 or
        native.get("plan_sha256") != meta["plan_sha256"] or
        native.get("request_sha256") != meta["request_sha256"]):
        raise ValueError("native preflight drift")
    return {"schema": "qwen38_cpu_preflight_observation_v1", "status": "passed",
            "output_absent": True, "observed_at_unix": time.time(),
            "bundle_sha256": os.environ["Q38_BUNDLE_SHA256"], "native": native}


if __name__ == "__main__":
    try:
        result = run()
        code = 0
    except BaseException as exc:
        result = {"schema": "qwen38_cpu_preflight_observation_v1", "status": "failed",
                  "error_class": type(exc).__name__}
        code = 2
    Path("/dev/termination-log").write_bytes(canonical(result))
    raise SystemExit(code)
