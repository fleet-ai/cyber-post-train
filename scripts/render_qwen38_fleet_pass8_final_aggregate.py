#!/usr/bin/env python3
"""Render, but never create, the private Fleet dev17 pass@8 final gate."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
import shutil
import stat
import uuid
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import final_pass8_aggregate as aggregate
from evals.fleet.evaluate import stable_job_preview

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json"
ROSTER = ROOT / "configs/evaluation/qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
BASE_CONFIG = ROOT / "configs/evaluation/qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
NAMESPACE = "fleet-train-jobs"
NAME = "chris-q38-dev17-pass8-final-v2"
CONFIG_MAP = f"{NAME}-code"
SOURCE = ROOT / "evals/fleet/final_pass8_aggregate.py"
WHEEL_LOCK = [
    {
        "filename": "psycopg-3.3.5-py3-none-any.whl",
        "url": (
            "https://files.pythonhosted.org/packages/3d/2e/"
            "d0a645bcaadde68bd6d93c43f02f14b0191bdda367ce3f7722abe3da744a/"
            "psycopg-3.3.5-py3-none-any.whl"
        ),
        "size": 213598,
        "sha256": "ce5aa5cdb4f9379f00f487590e5890bfa7df9a164648c969ffa628505e21af4e",
    },
    {
        "filename": (
            "psycopg_binary-3.3.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
        ),
        "url": (
            "https://files.pythonhosted.org/packages/21/d1/"
            "0f244dfef389e52e9dc3056f2a9033d1f6901e97d24d9a9c8b836e32ab6b/"
            "psycopg_binary-3.3.5-cp312-cp312-manylinux2014_x86_64."
            "manylinux_2_17_x86_64.whl"
        ),
        "size": 5227752,
        "sha256": "682a17a57415c3ca1731eec018ed031f012ffcb81ba74806eb219cb396065672",
    },
    {
        "filename": "typing_extensions-4.16.0-py3-none-any.whl",
        "url": (
            "https://files.pythonhosted.org/packages/49/d3/"
            "b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/"
            "typing_extensions-4.16.0-py3-none-any.whl"
        ),
        "size": 45571,
        "sha256": "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
    },
]
RUNNER = r"""from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

source = Path("aggregate.py")
actual = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
if actual != os.environ["FINAL_AGGREGATE_MODULE_SHA256"]:
    raise RuntimeError("private final aggregate module bytes differ")
spec = importlib.util.spec_from_file_location("fleet_final_pass8_aggregate", source)
if spec is None or spec.loader is None:
    raise RuntimeError("private final aggregate module cannot be loaded")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module.run_from_environment(Path("study.json"))
staging = Path(os.environ["FINAL_STAGING_ROOT"])
if staging.as_posix() != module.PRIVATE_STAGING_ROOT:
    raise RuntimeError("private final staging root differs")
for child in staging.iterdir():
    if not child.is_file() or child.is_symlink():
        raise RuntimeError("private final staging file roster differs")
    child.chmod(0o640)
staging.chmod(0o750)
print(json.dumps({
    "status": result["status"],
    "task_count": result["task_count"],
    "valid_outcomes_per_task_arm": result["valid_outcomes_per_task_arm"],
    "receipt_sha256": result["receipt_sha256"],
}, sort_keys=True), flush=True)
ready = staging.parent / "READY"
descriptor = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    stream.write(result["receipt_sha256"] + "\n")
    stream.flush()
    os.fsync(stream.fileno())
"""
PUBLISHER = r"""from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path

FILES = {
    "PRIVATE_TERMINAL_INDEX.json",
    "PRIVATE_SCORED_OUTCOME_INDEX.json",
    "PRIVATE_ANONYMIZATION.json",
    "SANITIZED_AGGREGATE.json",
    "FINAL.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read_regular_once(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "staged output is not a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "staged output changed while read",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def checked_json(payload: bytes) -> dict:
    value = json.loads(payload)
    require(isinstance(value, dict), "staged output is not an object")
    claimed = value.get("receipt_sha256")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    actual = "sha256:" + hashlib.sha256(canonical(body)).hexdigest()
    require(claimed == actual, "staged output self digest differs")
    return value


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "atomic no-replace rename is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("final aggregate output already exists")
        raise OSError(error, os.strerror(error))


source = Path(os.environ["FINAL_STAGING_ROOT"])
target = Path(os.environ["FINAL_OUTPUT_ROOT"])
require(source == Path("/result-staging/chris-q38-dev17-pass8-final-v2"), "stage differs")
require(
    target
    == Path(
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/"
        "chris-q38-dev17-pass8-final-v2"
    ),
    "output differs",
)
require((os.geteuid(), os.getegid()) == (1000, 100), "publisher identity differs")
deadline = time.monotonic() + 1700
ready = source.parent / "READY"
failed = source.parent / "FAILED"
while not ready.is_file():
    require(not failed.exists(), "aggregate reader failed before publication")
    require(time.monotonic() < deadline, "aggregate reader did not finish in time")
    time.sleep(2)
identity = source.lstat()
require(
    stat.S_ISDIR(identity.st_mode)
    and not source.is_symlink()
    and (identity.st_uid, identity.st_gid) == (0, 100)
    and identity.st_mode & stat.S_IRGRP
    and identity.st_mode & stat.S_IXGRP,
    "staging directory identity differs",
)
require({item.name for item in source.iterdir()} == FILES, "staged file roster differs")
payloads = {name: read_regular_once(source / name) for name in FILES}
values = {name: checked_json(payload) for name, payload in payloads.items()}
terminal = values["PRIVATE_TERMINAL_INDEX.json"]
outcome = values["PRIVATE_SCORED_OUTCOME_INDEX.json"]
anonymization = values["PRIVATE_ANONYMIZATION.json"]
public = values["SANITIZED_AGGREGATE.json"]
final = values["FINAL.json"]
require(terminal.get("schema") == "cyber_fleet_matched_pass8_terminal_index_v2", "terminal schema")
require(
    outcome.get("schema")
    == "cyber_fleet_matched_pass8_protocol_v2_private_scored_outcome_index_v1",
    "outcome schema",
)
require(anonymization.get("schema") == "cyber_private_task_anonymization_v1", "mapping schema")
require(
    public.get("schema_version") == "cyber_sanitized_matched_pass8_aggregate_v1",
    "public schema",
)
require(
    final.get("schema") == "cyber_fleet_matched_pass8_protocol_v2_final_receipt_v1"
    and final.get("status") == "final"
    and final.get("private_terminal_index_sha256") == terminal["receipt_sha256"]
    and final.get("private_scored_outcome_index_sha256") == outcome["receipt_sha256"]
    and final.get("private_anonymization_receipt_sha256") == anonymization["receipt_sha256"]
    and final.get("sanitized_aggregate_receipt_sha256") == public["receipt_sha256"],
    "final cross-file identity differs",
)
require(read_regular_once(ready).decode().strip() == final["receipt_sha256"], "ready differs")
parent = target.parent
parent_identity = parent.lstat()
require(
    stat.S_ISDIR(parent_identity.st_mode)
    and not parent.is_symlink()
    and (parent_identity.st_uid, parent_identity.st_gid) == (1000, 100)
    and parent_identity.st_mode & stat.S_IWUSR
    and parent_identity.st_mode & stat.S_IXUSR,
    "output parent identity differs",
)
require(not target.exists() and not target.is_symlink(), "final output already exists")
probe = None
try:
    probe = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".pass8-publisher-probe-", dir=parent, delete=False
    )
    probe.write(b"pass8-publisher-probe-v1\n")
    probe.flush()
    os.fsync(probe.fileno())
    probe.close()
    Path(probe.name).unlink()
finally:
    if probe is not None and not probe.closed:
        probe.close()
    if probe is not None:
        Path(probe.name).unlink(missing_ok=True)
temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
os.chmod(temporary, 0o700)
for name in sorted(FILES):
    descriptor = os.open(temporary / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payloads[name])
        stream.flush()
        os.fsync(stream.fileno())
directory = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
rename_noreplace(temporary, target)
directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(json.dumps({"status": "published", "receipt_sha256": final["receipt_sha256"]}))
"""


class RenderError(ValueError):
    """The rendered private final gate is incomplete or preview-mismatched."""


def _canonical_digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular_once(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RenderError(f"{label} is not an exact regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RenderError(f"{label} is not an exact regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (
            identity != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RenderError(f"{label} changed while it was read")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _bundle(plan: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    files = {
        "aggregate.py": SOURCE.read_text(encoding="utf-8"),
        "publish.py": PUBLISHER,
        "run.py": RUNNER,
        "study.json": json.dumps(plan, indent=2, sort_keys=True) + "\n",
        "wheel-lock.json": json.dumps(WHEEL_LOCK, indent=2, sort_keys=True) + "\n",
    }
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    if len(base64.b64encode(compressed)) > 900_000:
        raise RenderError("private final aggregate bundle exceeds the ConfigMap ceiling")
    return compressed, digests


def _objects(plan: dict[str, Any], compressed: bytes, digests: dict[str, str]) -> tuple[dict, dict]:
    bundle_digest = "sha256:" + hashlib.sha256(compressed).hexdigest()
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP, "namespace": NAMESPACE},
        "immutable": True,
        "binaryData": {"bundle.json.gz": base64.b64encode(compressed).decode()},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/experiment": NAME,
                "cyber-post-train.fleet.ai/owner": "chris",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "fleet.ai/failure-alerts": "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": NAME,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": {
                    "priorityClassName": "c1",
                    "automountServiceAccountToken": False,
                    "restartPolicy": "Never",
                    "securityContext": {
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
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
                            "name": "aggregate",
                            "image": IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "trap 'status=$?; trap - EXIT; "
                                'if [ "$status" -ne 0 ] && '
                                "[ ! -f /result-staging/READY ]; then umask 027; "
                                ": > /result-staging/FAILED || true; fi; "
                                'exit "$status"\' EXIT\n'
                                "python - <<'PY'\n"
                                "import gzip,hashlib,io,json,os,pathlib,stat,tempfile\n"
                                "import urllib.request,zipfile\n"
                                "def require(condition,message):\n"
                                " if not condition: raise RuntimeError(message)\n"
                                "bundle=pathlib.Path('/bootstrap/bundle.json.gz').read_bytes()\n"
                                "actual='sha256:'+hashlib.sha256(bundle).hexdigest()\n"
                                "require(actual==os.environ['FINAL_BUNDLE_SHA256'],\n"
                                " 'final bundle digest differs')\n"
                                "root=pathlib.Path('/workspace/source')\n"
                                "root.mkdir(parents=True,exist_ok=False)\n"
                                "for name,text in json.loads(gzip.decompress(bundle)).items():\n"
                                " p=root/name; p.parent.mkdir(parents=True,exist_ok=True)\n"
                                " p.write_text(text,encoding='utf-8')\n"
                                "deps=pathlib.Path('/workspace/deps')\n"
                                "deps.mkdir(parents=True,exist_ok=False)\n"
                                "for item in json.loads((root/'wheel-lock.json').read_text()):\n"
                                " require(\n"
                                "  item['url'].startswith('https://files.pythonhosted.org/packages/'),\n"
                                "  'wheel source differs'\n"
                                " )\n"
                                " with urllib.request.urlopen(\n"
                                "  item['url'],timeout=120\n"
                                " ) as response:\n"
                                "  wheel=response.read(item['size']+1)\n"
                                " require(len(wheel)==item['size'],'wheel size differs')\n"
                                " require(\n"
                                "  hashlib.sha256(wheel).hexdigest()==item['sha256'],\n"
                                "  'wheel digest differs'\n"
                                " )\n"
                                " with zipfile.ZipFile(io.BytesIO(wheel)) as archive:\n"
                                "  seen=set()\n"
                                "  for member in archive.infolist():\n"
                                "   relative=pathlib.PurePosixPath(member.filename)\n"
                                "   require(\n"
                                "    bool(relative.parts) and not relative.is_absolute(),\n"
                                "    'wheel member path differs'\n"
                                "   )\n"
                                "   require('..' not in relative.parts,'wheel member escapes')\n"
                                "   require(member.filename not in seen,'duplicate wheel member')\n"
                                "   seen.add(member.filename)\n"
                                "   require(\n"
                                "    not stat.S_ISLNK(member.external_attr>>16),\n"
                                "    'wheel symlink is forbidden'\n"
                                "   )\n"
                                "   target=deps.joinpath(*relative.parts)\n"
                                "   if member.is_dir():\n"
                                "    target.mkdir(parents=True,exist_ok=True)\n"
                                "    continue\n"
                                "   target.parent.mkdir(parents=True,exist_ok=True)\n"
                                "   with target.open('xb') as stream:\n"
                                "    stream.write(archive.read(member))\n"
                                "output=pathlib.Path(os.environ['FINAL_OUTPUT_ROOT'])\n"
                                "expected=pathlib.Path(\n"
                                " '/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/'\n"
                                " 'chris-q38-dev17-pass8-final-v2'\n"
                                ")\n"
                                "require(output==expected,'final output path differs')\n"
                                "require(\n"
                                " (os.geteuid(),os.getegid())==(0,100),\n"
                                " 'reader identity differs'\n"
                                ")\n"
                                "require(\n"
                                " not output.exists() and not output.is_symlink(),\n"
                                " 'final output already exists'\n"
                                ")\n"
                                "staging=pathlib.Path(os.environ['FINAL_STAGING_ROOT'])\n"
                                "require(\n"
                                " staging==pathlib.Path(\n"
                                "  '/result-staging/chris-q38-dev17-pass8-final-v2'\n"
                                " ),'staging root differs'\n"
                                ")\n"
                                "staging_identity=staging.parent.lstat()\n"
                                "require(\n"
                                " stat.S_ISDIR(staging_identity.st_mode)\n"
                                " and not staging.parent.is_symlink(),\n"
                                " 'staging parent identity differs'\n"
                                ")\n"
                                "require(\n"
                                " not staging.exists() and not staging.is_symlink(),\n"
                                " 'staging output already exists'\n"
                                ")\n"
                                "plan=json.loads((root/'study.json').read_text())\n"
                                "for replica in plan['replicas']:\n"
                                " source=pathlib.Path(replica['output_root'])\n"
                                " source_identity=source.lstat()\n"
                                " require(\n"
                                "  stat.S_ISDIR(source_identity.st_mode)\n"
                                "  and not source.is_symlink(),\n"
                                "  'source output root is not readable as an exact directory'\n"
                                " )\n"
                                " terminal=pathlib.Path(replica['terminal_receipt_path'])\n"
                                " require(\n"
                                "  terminal.parent==source,\n"
                                "  'terminal path escapes source root'\n"
                                " )\n"
                                " for evidence in (source/'EVAL.json',terminal):\n"
                                "  descriptor=os.open(\n"
                                "   evidence,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)\n"
                                "  )\n"
                                "  try:\n"
                                "   require(\n"
                                "    stat.S_ISREG(os.fstat(descriptor).st_mode),\n"
                                "    'source evidence is not an exact regular file'\n"
                                "   )\n"
                                "  finally:\n"
                                "   os.close(descriptor)\n"
                                "probe=None\n"
                                "try:\n"
                                " probe=tempfile.NamedTemporaryFile(\n"
                                "  mode='wb',prefix='.pass8-staging-probe-',\n"
                                "  dir=staging.parent,delete=False\n"
                                " )\n"
                                " probe.write(b'pass8-write-probe-v1\\n')\n"
                                " probe.flush(); os.fsync(probe.fileno()); probe.close()\n"
                                " pathlib.Path(probe.name).unlink()\n"
                                "finally:\n"
                                " if probe is not None and not probe.closed: probe.close()\n"
                                " if probe is not None:\n"
                                "  pathlib.Path(probe.name).unlink(missing_ok=True)\n"
                                "PY\n"
                                "cd /workspace/source\n"
                                "export PYTHONPATH=/workspace/deps\n"
                                "python run.py\n"
                            ],
                            "env": [
                                {"name": "FINAL_BUNDLE_SHA256", "value": bundle_digest},
                                {
                                    "name": "FINAL_AGGREGATE_MODULE_SHA256",
                                    "value": digests["aggregate.py"],
                                },
                                {
                                    "name": "FINAL_STUDY_PLAN_FILE_SHA256",
                                    "value": digests["study.json"],
                                },
                                {
                                    "name": "FINAL_OUTPUT_ROOT",
                                    "value": plan["private_output_root"],
                                },
                                {
                                    "name": "FINAL_STAGING_ROOT",
                                    "value": aggregate.PRIVATE_STAGING_ROOT,
                                },
                                {
                                    "name": "ROLLOUT_DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-rollout-postgres-v1",
                                            "key": "ROLLOUT_DATABASE_URL",
                                        }
                                    },
                                },
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {"name": "TMPDIR", "value": "/tmp"},
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": False,
                                "runAsUser": 0,
                                "runAsGroup": 100,
                            },
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "tmp", "mountPath": "/tmp"},
                                {
                                    "name": "sfs-readonly",
                                    "mountPath": "/mnt/sfs",
                                    "readOnly": True,
                                },
                                {
                                    "name": "result-staging",
                                    "mountPath": "/result-staging",
                                },
                            ],
                        },
                        {
                            "name": "publish",
                            "image": IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "python - <<'PY'\n"
                                "import gzip,hashlib,json,os,pathlib\n"
                                "def require(condition,message):\n"
                                " if not condition: raise RuntimeError(message)\n"
                                "bundle=pathlib.Path('/bootstrap/bundle.json.gz').read_bytes()\n"
                                "require(\n"
                                " 'sha256:'+hashlib.sha256(bundle).hexdigest()\n"
                                " ==os.environ['FINAL_BUNDLE_SHA256'],\n"
                                " 'final bundle digest differs'\n"
                                ")\n"
                                "files=json.loads(gzip.decompress(bundle))\n"
                                "source=files.get('publish.py')\n"
                                "require(isinstance(source,str),'publisher source missing')\n"
                                "require(\n"
                                " 'sha256:'+hashlib.sha256(source.encode()).hexdigest()\n"
                                " ==os.environ['FINAL_PUBLISHER_MODULE_SHA256'],\n"
                                " 'publisher module digest differs'\n"
                                ")\n"
                                "exec(compile(source,'publish.py','exec'),{'__name__':'__main__'})\n"
                                "PY\n"
                            ],
                            "env": [
                                {"name": "FINAL_BUNDLE_SHA256", "value": bundle_digest},
                                {
                                    "name": "FINAL_PUBLISHER_MODULE_SHA256",
                                    "value": digests["publish.py"],
                                },
                                {
                                    "name": "FINAL_OUTPUT_ROOT",
                                    "value": plan["private_output_root"],
                                },
                                {
                                    "name": "FINAL_STAGING_ROOT",
                                    "value": aggregate.PRIVATE_STAGING_ROOT,
                                },
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {"name": "TMPDIR", "value": "/tmp"},
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "runAsGroup": 100,
                            },
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "publisher-tmp", "mountPath": "/tmp"},
                                {
                                    "name": "result-staging",
                                    "mountPath": "/result-staging",
                                    "readOnly": True,
                                },
                                {
                                    "name": "sfs-control",
                                    "mountPath": (
                                        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
                                    ),
                                    "subPath": ("jobs/chris-q38-study-corpora-v1/launch-controls"),
                                },
                            ],
                        },
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIG_MAP}},
                        {"name": "workspace", "emptyDir": {}},
                        {"name": "tmp", "emptyDir": {"sizeLimit": "128Mi"}},
                        {"name": "publisher-tmp", "emptyDir": {"sizeLimit": "32Mi"}},
                        {"name": "result-staging", "emptyDir": {"sizeLimit": "32Mi"}},
                        {
                            "name": "sfs-readonly",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                        {
                            "name": "sfs-control",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                    ],
                },
            },
        },
    }
    return config_map, job


def _render_receipt(
    *,
    plan: dict[str, Any],
    compressed: bytes,
    digests: dict[str, str],
    rendered_bundle_file_sha256: str,
) -> dict[str, Any]:
    receipt = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_job_render_v1",
        "namespace": NAMESPACE,
        "config_map_name": CONFIG_MAP,
        "job_name": NAME,
        "study_plan_sha256": plan["sha256"],
        "migration_receipt_sha256": plan["migration_receipt_sha256"],
        "migration_receipt_file_sha256": plan["migration_receipt_file_sha256"],
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "comparison_definition_file_sha256": plan["comparison_definition_file_sha256"],
        "source_files": digests,
        "compressed_bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
        "rendered_bundle_file_sha256": rendered_bundle_file_sha256,
        "failure_alerts": "off",
        "priority_class": "c1",
        "gpu_requests": 0,
        "source_reader_identity": {"uid": 0, "gid": 100, "sfs_access": "read_only"},
        "publisher_identity": {"uid": 1000, "gid": 100, "database_credential": False},
        "writable_sfs_subpath": "jobs/chris-q38-study-corpora-v1/launch-controls",
        "create_once": True,
        "two_server_previews_required_before_create": True,
        "external_mutations": 0,
        "launch_performed": False,
    }
    return {**receipt, "sha256": _canonical_digest(receipt)}


def render(*, output: Path, migration_receipt: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("private final aggregate render already exists")
    if not output.parent.is_dir():
        raise RenderError("private final aggregate render parent is missing")
    plan = aggregate.build_current_study_plan(
        task_set_path=TASK_SET,
        roster_path=ROSTER,
        base_config_path=BASE_CONFIG,
        migration_receipt_path=migration_receipt,
    )
    aggregate.validate_plan(plan)
    compressed, digests = _bundle(plan)
    config_map, job = _objects(plan, compressed, digests)
    output.mkdir(mode=0o700)
    try:
        bundle = {"apiVersion": "v1", "kind": "List", "items": [config_map, job]}
        bundle_path = output / "final-aggregate.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = _render_receipt(
            plan=plan,
            compressed=compressed,
            digests=digests,
            rendered_bundle_file_sha256=_file_digest(bundle_path),
        )
        (output / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return receipt


def _items(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("kind") == "List" and isinstance(value.get("items"), list):
        return value["items"]
    return [value]


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return actual == expected


def _contains_accelerator_resource(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            isinstance(key, str)
            and ("gpu" in key.lower() or "mig" in key.lower() or key.startswith("nvidia.com/"))
            for key in value
        ) or any(_contains_accelerator_resource(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_accelerator_resource(item) for item in value)
    return False


def _extra_keys(
    actual: Mapping[str, Any], expected: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unexpected = set(actual) - set(expected) - allowed
    if unexpected:
        raise RenderError(f"server-rendered {label} added unreviewed fields")


def _server_defaults_only(job: dict[str, Any], expected: dict[str, Any]) -> None:
    normalized = stable_job_preview(job)
    reviewed = stable_job_preview(expected)
    _extra_keys(normalized, reviewed, set(), "Job")
    _extra_keys(normalized["metadata"], reviewed["metadata"], set(), "Job metadata")
    actual_annotations = normalized["metadata"].get("annotations", {})
    expected_annotations = reviewed["metadata"].get("annotations", {})
    if actual_annotations != expected_annotations:
        raise RenderError("server-rendered root Job annotations differ")
    root_labels = normalized["metadata"].get("labels", {})
    expected_root_labels = reviewed["metadata"].get("labels", {})
    _extra_keys(
        root_labels,
        expected_root_labels,
        {
            "batch.kubernetes.io/controller-uid",
            "batch.kubernetes.io/job-name",
            "controller-uid",
            "job-name",
        },
        "root Job labels",
    )
    raw_uid = job.get("metadata", {}).get("uid")
    if any(
        root_labels.get(key) not in (None, raw_uid)
        for key in ("batch.kubernetes.io/controller-uid", "controller-uid")
    ) or any(
        root_labels.get(key) not in (None, NAME)
        for key in ("batch.kubernetes.io/job-name", "job-name")
    ):
        raise RenderError("server-rendered root Job generated labels differ")
    _extra_keys(
        normalized["spec"],
        reviewed["spec"],
        {
            "completionMode",
            "completions",
            "manualSelector",
            "parallelism",
            "podReplacementPolicy",
            "suspend",
        },
        "Job spec",
    )
    job_spec = normalized["spec"]
    safe_job_defaults = {
        "completionMode": {None, "NonIndexed"},
        "completions": {None, 1},
        "manualSelector": {None, False},
        "parallelism": {None, 1},
        "podReplacementPolicy": {None, "Failed", "TerminatingOrFailed"},
        "suspend": {None, False},
    }
    if any(job_spec.get(field) not in values for field, values in safe_job_defaults.items()):
        raise RenderError("server-rendered Job has a non-default execution policy")
    selector = job.get("spec", {}).get("selector")
    if selector is not None:
        if not isinstance(selector, dict) or set(selector) - {"matchLabels", "matchExpressions"}:
            raise RenderError("server-rendered Job selector differs")
        selector_labels = selector.get("matchLabels")
        expressions = selector.get("matchExpressions")
        raw_uid = job.get("metadata", {}).get("uid")
        if (
            not isinstance(selector_labels, dict)
            or not selector_labels
            or set(selector_labels) - {"batch.kubernetes.io/controller-uid", "controller-uid"}
            or any(value != raw_uid for value in selector_labels.values())
            or expressions not in (None, [])
        ):
            raise RenderError("server-rendered Job selector differs")
    actual_template = normalized["spec"]["template"]
    expected_template = reviewed["spec"]["template"]
    _extra_keys(actual_template, expected_template, set(), "Pod template")
    _extra_keys(
        actual_template["metadata"],
        expected_template["metadata"],
        {"creationTimestamp"},
        "Pod metadata",
    )
    if actual_template["metadata"].get("creationTimestamp") is not None:
        raise RenderError("server-rendered Pod creation timestamp is not the API default")
    labels = actual_template["metadata"].get("labels", {})
    expected_labels = expected_template["metadata"].get("labels", {})
    _extra_keys(
        labels,
        expected_labels,
        {"batch.kubernetes.io/job-name", "job-name"},
        "Pod labels",
    )
    if any(
        labels.get(key) not in (None, NAME) for key in ("batch.kubernetes.io/job-name", "job-name")
    ):
        raise RenderError("server-rendered Pod job-name label differs")
    pod = actual_template["spec"]
    expected_pod = expected_template["spec"]
    _extra_keys(
        pod,
        expected_pod,
        {
            "dnsPolicy",
            "enableServiceLinks",
            "preemptionPolicy",
            "priority",
            "schedulerName",
            "schedulingGates",
            "securityContext",
            "serviceAccount",
            "serviceAccountName",
            "terminationGracePeriodSeconds",
        },
        "Pod spec",
    )
    safe_pod_defaults = {
        "dnsPolicy": {None, "ClusterFirst"},
        "enableServiceLinks": {None, True},
        "preemptionPolicy": {None, "PreemptLowerPriority"},
        "priority": {None, 10_000},
        "schedulerName": {None, "default-scheduler"},
        "terminationGracePeriodSeconds": {None, 30},
    }
    if any(pod.get(field) not in values for field, values in safe_pod_defaults.items()):
        raise RenderError("server-rendered Pod has a non-default scheduling policy")
    if any(pod.get(field) is True for field in ("hostIPC", "hostNetwork", "hostPID")):
        raise RenderError("server-rendered Pod enables a host namespace")
    if pod.get("automountServiceAccountToken") is not False:
        raise RenderError("server-rendered Pod enables a service-account token")
    if pod.get("securityContext") != expected_pod.get("securityContext"):
        raise RenderError("server-rendered Pod weakens the reviewed security context")
    if pod.get("serviceAccountName") not in (None, "default") or pod.get("serviceAccount") not in (
        None,
        "default",
    ):
        raise RenderError("server-rendered Pod changes the service account")
    gates = pod.get("schedulingGates", [])
    if gates not in ([], [{"name": "kueue.x-k8s.io/admission"}]):
        raise RenderError("server-rendered Pod adds an unknown scheduling gate")
    if pod.get("initContainers") or pod.get("ephemeralContainers") or pod.get("resourceClaims"):
        raise RenderError("server-rendered Pod adds an unreviewed container or resource claim")
    if pod.get("nodeSelector") != expected_pod.get("nodeSelector"):
        raise RenderError("server-rendered Pod node selector differs")
    if pod.get("tolerations") != expected_pod.get("tolerations"):
        raise RenderError("server-rendered Pod tolerations differ")
    containers = pod.get("containers")
    expected_containers = expected_pod["containers"]
    if not isinstance(containers, list) or len(containers) != len(expected_containers):
        raise RenderError("server-rendered Pod changes the container roster")
    for container, expected_container in zip(containers, expected_containers, strict=True):
        _extra_keys(
            container,
            expected_container,
            {"imagePullPolicy", "terminationMessagePath", "terminationMessagePolicy"},
            "container",
        )
        safe_container_defaults = {
            "imagePullPolicy": {None, "IfNotPresent"},
            "terminationMessagePath": {None, "/dev/termination-log"},
            "terminationMessagePolicy": {None, "File"},
        }
        if any(
            container.get(field) not in values for field, values in safe_container_defaults.items()
        ):
            raise RenderError("server-rendered container has an unsafe runtime default")
        if container.get("securityContext") != expected_container.get("securityContext"):
            raise RenderError("server-rendered container weakens the reviewed security context")
        if container.get("env") != expected_container.get("env"):
            raise RenderError("server-rendered container environment differs")
        if container.get("volumeMounts") != expected_container.get("volumeMounts"):
            raise RenderError("server-rendered container volume mounts differ")
        resources = container.get("resources", {})
        expected_resources = expected_container.get("resources", {})
        _extra_keys(resources, expected_resources, set(), "container resources")
        for kind in ("requests", "limits"):
            if set(resources.get(kind, {})) != set(expected_resources.get(kind, {})):
                raise RenderError("server-rendered container resource keys differ")
    volumes = pod.get("volumes")
    expected_volumes = expected_pod["volumes"]
    if not isinstance(volumes, list) or len(volumes) != len(expected_volumes):
        raise RenderError("server-rendered Pod changes the volume roster")
    for volume, expected_volume in zip(volumes, expected_volumes, strict=True):
        _extra_keys(volume, expected_volume, set(), "volume")
        if "configMap" in volume:
            _extra_keys(
                volume["configMap"],
                expected_volume["configMap"],
                {"defaultMode"},
                "ConfigMap volume",
            )
            if volume["configMap"].get("defaultMode") not in (None, 0o644):
                raise RenderError("server-rendered ConfigMap volume mode differs")
        if "persistentVolumeClaim" in volume:
            _extra_keys(
                volume["persistentVolumeClaim"],
                expected_volume["persistentVolumeClaim"],
                set(),
                "persistent volume",
            )


def _job_uid(value: dict[str, Any]) -> str:
    jobs = [item for item in _items(value) if item.get("kind") == "Job"]
    if len(jobs) != 1:
        raise RenderError("server preview lacks the exact Job")
    raw = jobs[0].get("metadata", {}).get("uid")
    try:
        return str(uuid.UUID(raw))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RenderError("server preview lacks a valid server-assigned Job UID") from exc


def _normalized_preview(value: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    items = _items(value)
    jobs = [item for item in items if item.get("kind") == "Job"]
    maps = [item for item in items if item.get("kind") == "ConfigMap"]
    if len(items) != 2 or len(jobs) != 1 or len(maps) != 1:
        raise RenderError("server preview lacks the exact Job and ConfigMap")
    job = jobs[0]
    config_map = maps[0]
    expected_items = _items(expected)
    expected_job = next(item for item in expected_items if item["kind"] == "Job")
    expected_map = next(item for item in expected_items if item["kind"] == "ConfigMap")
    if job.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") != "off":
        raise RenderError("server-rendered root Job did not retain failure alerts off")
    if _contains_accelerator_resource(job):
        raise RenderError("server-rendered Job unexpectedly requests an accelerator")
    normalized_job = stable_job_preview(job)
    if not _contains(normalized_job, stable_job_preview(expected_job)):
        raise RenderError("server-rendered Job differs from the immutable render")
    _server_defaults_only(job, expected_job)
    _extra_keys(config_map, expected_map, set(), "ConfigMap")
    metadata = config_map.get("metadata")
    if not isinstance(metadata, dict):
        raise RenderError("server-rendered ConfigMap metadata is missing")
    _extra_keys(
        metadata,
        expected_map["metadata"],
        {"creationTimestamp", "generation", "managedFields", "resourceVersion", "uid"},
        "ConfigMap metadata",
    )
    if metadata.get("uid") is not None:
        try:
            uuid.UUID(str(metadata["uid"]))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RenderError("server-rendered ConfigMap UID is invalid") from exc
    normalized_map = {
        "apiVersion": config_map.get("apiVersion"),
        "kind": config_map.get("kind"),
        "metadata": {
            "name": config_map.get("metadata", {}).get("name"),
            "namespace": config_map.get("metadata", {}).get("namespace"),
        },
        "immutable": config_map.get("immutable"),
        "binaryData": config_map.get("binaryData"),
    }
    expected_normalized_map = {
        "apiVersion": expected_map["apiVersion"],
        "kind": expected_map["kind"],
        "metadata": {
            "name": expected_map["metadata"]["name"],
            "namespace": expected_map["metadata"]["namespace"],
        },
        "immutable": expected_map["immutable"],
        "binaryData": expected_map["binaryData"],
    }
    if normalized_map != expected_normalized_map:
        raise RenderError("server-rendered ConfigMap differs from the immutable render")
    return {"job": normalized_job, "config_map": normalized_map}


def _validated_render(render_root: Path, migration_receipt: Path) -> tuple[dict[str, Any], str]:
    bundle_path = render_root / "final-aggregate.yaml"
    receipt_path = render_root / "RENDER.json"
    bundle_bytes, _ = _read_regular_once(bundle_path, "rendered bundle")
    receipt_bytes, _ = _read_regular_once(receipt_path, "render receipt")
    try:
        expected = yaml.safe_load(bundle_bytes)
        render_receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RenderError("rendered bundle or receipt is unreadable") from exc
    if not isinstance(expected, dict) or not isinstance(render_receipt, dict):
        raise RenderError("rendered bundle or receipt is malformed")
    claimed = render_receipt.get("sha256")
    if claimed != _canonical_digest(
        {key: value for key, value in render_receipt.items() if key != "sha256"}
    ):
        raise RenderError("render receipt self digest differs")
    items = _items(expected)
    maps = [item for item in items if item.get("kind") == "ConfigMap"]
    jobs = [item for item in items if item.get("kind") == "Job"]
    if len(items) != 2 or len(maps) != 1 or len(jobs) != 1:
        raise RenderError("rendered bundle lacks the exact Job and ConfigMap")
    encoded = maps[0].get("binaryData", {}).get("bundle.json.gz")
    if not isinstance(encoded, str):
        raise RenderError("rendered ConfigMap lacks the immutable source bundle")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        files = json.loads(gzip.decompress(compressed))
    except (
        ValueError,
        binascii.Error,
        gzip.BadGzipFile,
        zlib.error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise RenderError("rendered source bundle is unreadable") from exc
    if (
        not isinstance(files, dict)
        or set(files) != {"aggregate.py", "publish.py", "run.py", "study.json", "wheel-lock.json"}
        or any(not isinstance(value, str) for value in files.values())
    ):
        raise RenderError("rendered source bundle file roster differs")
    if (
        files["aggregate.py"] != SOURCE.read_text(encoding="utf-8")
        or files["publish.py"] != PUBLISHER
        or files["run.py"] != RUNNER
        or files["wheel-lock.json"] != json.dumps(WHEEL_LOCK, indent=2, sort_keys=True) + "\n"
    ):
        raise RenderError("rendered executable bytes differ from the reviewed source")
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    try:
        plan = json.loads(files["study.json"])
    except json.JSONDecodeError as exc:
        raise RenderError("rendered study plan is unreadable") from exc
    if not isinstance(plan, dict):
        raise RenderError("rendered study plan is malformed")
    try:
        authoritative_plan = aggregate.build_current_study_plan(
            task_set_path=TASK_SET,
            roster_path=ROSTER,
            base_config_path=BASE_CONFIG,
            migration_receipt_path=migration_receipt,
        )
    except aggregate.FinalAggregateError as exc:
        raise RenderError("authoritative migration evidence is invalid") from exc
    if plan != authoritative_plan:
        raise RenderError("rendered study plan differs from authoritative migration evidence")
    expected_compressed, expected_digests = _bundle(authoritative_plan)
    if compressed != expected_compressed or digests != expected_digests:
        raise RenderError("rendered source bundle differs from the canonical package")
    expected_map, expected_job = _objects(plan, compressed, digests)
    if maps[0] != expected_map or jobs[0] != expected_job:
        raise RenderError("rendered Kubernetes objects differ from the immutable source bundle")
    bundle_file_sha256 = "sha256:" + hashlib.sha256(bundle_bytes).hexdigest()
    exact_receipt = _render_receipt(
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=bundle_file_sha256,
    )
    if render_receipt != exact_receipt:
        raise RenderError("render receipt policy or source identity differs")
    return expected, "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()


def validate_previews(
    *,
    render_root: Path,
    migration_receipt: Path,
    first: Path,
    second: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("server preview receipt already exists")
    expected, render_receipt_file_sha256 = _validated_render(render_root, migration_receipt)
    first_bytes, first_identity = _read_regular_once(first, "first server preview")
    second_bytes, second_identity = _read_regular_once(second, "second server preview")
    if first_identity == second_identity:
        raise RenderError("two server previews must be distinct regular files")
    try:
        one = json.loads(first_bytes)
        two = json.loads(second_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError("server preview is unreadable") from exc
    if not isinstance(one, dict) or not isinstance(two, dict):
        raise RenderError("server preview is malformed")
    first_uid = _job_uid(one)
    second_uid = _job_uid(two)
    if first_uid == second_uid:
        raise RenderError("two independent server previews must have distinct Job UIDs")
    stable_one = _normalized_preview(one, expected)
    stable_two = _normalized_preview(two, expected)
    first_digest = _canonical_digest(stable_one)
    second_digest = _canonical_digest(stable_two)
    if first_digest != second_digest:
        raise RenderError("identical server dry-runs produced different stable previews")
    receipt = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_server_preview_v1",
        "render_receipt_file_sha256": render_receipt_file_sha256,
        "first_preview_file_sha256": "sha256:" + hashlib.sha256(first_bytes).hexdigest(),
        "second_preview_file_sha256": "sha256:" + hashlib.sha256(second_bytes).hexdigest(),
        "stable_server_preview_sha256": first_digest,
        "first_job_uid_sha256": "sha256:" + hashlib.sha256(first_uid.encode()).hexdigest(),
        "second_job_uid_sha256": "sha256:" + hashlib.sha256(second_uid.encode()).hexdigest(),
        "root_failure_alerts": "off",
        "priority_class": "c1",
        "gpu_requests": 0,
        "create_performed": False,
    }
    receipt["sha256"] = _canonical_digest(receipt)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--migration-receipt", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--preview-one", type=Path)
    parser.add_argument("--preview-two", type=Path)
    parser.add_argument("--preview-receipt", type=Path)
    args = parser.parse_args()
    if (
        args.output is not None
        and args.migration_receipt is not None
        and all(
            value is None
            for value in (
                args.render_root,
                args.preview_one,
                args.preview_two,
                args.preview_receipt,
            )
        )
    ):
        result = render(output=args.output, migration_receipt=args.migration_receipt)
    elif (
        args.output is None
        and args.migration_receipt is not None
        and all(
            value is not None
            for value in (
                args.render_root,
                args.preview_one,
                args.preview_two,
                args.preview_receipt,
            )
        )
    ):
        result = validate_previews(
            render_root=args.render_root,
            migration_receipt=args.migration_receipt,
            first=args.preview_one,
            second=args.preview_two,
            output=args.preview_receipt,
        )
    else:
        parser.error(
            "choose exactly one render or two-preview validation operation; both require "
            "--migration-receipt"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
