"""Reconstruct the exact historical 4×8 Qwen3.8 262K canary source.

The compiler and runtime are exact Git blobs from the held 65fcf038 candidate.
This bridge stages them in a temporary directory, then discards that directory.
The one-time preflight and submission are recorded in Git history and receipts.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/runs/qwen38-262k-four-node-canary.json"
BASE_SPEC_SHA256 = "c1f2e4006756f77b11dbc71b079146fcb19f67e5f96405fc876517471e6012b1"


REVISION = "65fcf03812e48cf6f0845c607152b2069f3275b4"
SUCCESSOR = ROOT / "configs/runs/qwen38-262k-four-node-canary-v3.json"
V4 = ROOT / "configs/runs/qwen38-262k-four-node-canary-v4.json"
OLD_NAME = "chris-q38-t3k262-4n-can-v1"
NEW_NAME = "chris-q38-t3k262-4n-can-v3"
V4_NAME = "chris-q38-t3k262-4n-can-v4"
SOURCES = (
    "training/__init__.py",
    "training/io.py",
    "training/models.py",
    "training/sft.py",
    "training/sft_runtime.py",
    "training/sft_262k_runtime.py",
    "training/sft_262k_4node_v1.py",
    "cyber_post_train/__init__.py",
    "cyber_post_train/jobs.py",
    "configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json",
    "configs/qualification/qwen38-teacher3k-262k-v12-parent-plan.json",
)
SOURCE_SHA256 = {
    "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
    "training/sft_262k_runtime.py": "fed7acaa5e08451da11231197aa20316de9cea2729fb31de22cc23cdb97d1001",
    "training/sft_262k_4node_v1.py": "edd5576689fa387836d2896daa647db3496e78d5d05bf8b519cb29399cba07ae",
    "cyber_post_train/jobs.py": "bc0d08a7a27b48ff5a21a5d824356d6e92e4856b17cc7296a976aa18a2baa40d",
}


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def base_spec() -> dict:
    blob = SPEC.read_bytes()
    if digest(blob) != BASE_SPEC_SHA256:
        raise ValueError("historical capacity spec changed")
    spec = json.loads(blob)
    for key in ("length_audit", "real_row_witness"):
        receipt = spec[key]
        if digest((ROOT / receipt["path"]).read_bytes()) != receipt["file_sha256"]:
            raise ValueError(f"historical {key} evidence changed")
    return spec


def successor_spec() -> dict:
    spec = json.loads(SUCCESSOR.read_text())
    base = base_spec()
    if (
        set(spec) != {"schema", "base_spec", "base_spec_sha256", "name", "output_root", "delta", "accepted", "submission_authorized", "root_review", "cpu_preflight", "remaining_gates"}
        or spec["schema"] != "qwen38_262k_four_node_repair_v1"
        or spec["base_spec"] != str(SPEC.relative_to(ROOT))
        or spec["base_spec_sha256"] != digest(SPEC.read_bytes())
        or spec["name"] != NEW_NAME
        or spec["output_root"] != f"/mnt/sfs/jobs/{NEW_NAME}"
        or spec["delta"] != "create-once identity; accept the entrypoint-added plan_sha256 field; no model, data, or recipe change"
        or spec["accepted"] is not False
        or spec["submission_authorized"] is not True
        or spec["root_review"] != {
            "scope": "one create-once v3 four-node capacity canary; one finite step, checkpoint, and reload; not scientific training",
            "v1_cpu_receipt_sha256": "39c827afa6db9733e175c6ebb3b5c75478e661d5266a265fa6402f40132b5cf0",
            "v2_failed_run_id": "1502ba9f-0506-43ac-93fb-c77f88c2a019",
            "v2_failed_rayjob_uid": "090b0a43-560d-42dc-8b49-6a96aaca4bc9",
        }
        or spec["cpu_preflight"] != {
            "path": "docs/evidence/qwen38-262k-four-node-cpu-preflight-v6-passed.json",
            "file_sha256": "5f4a88b8f8cb9f414ab0e6c4f374780a72a23b4e19ad0621426aa5bf6cca0bca",
        }
        or len(spec["remaining_gates"]) != 3
    ):
        raise ValueError("v2 successor intent or root review changed")
    receipt_blob = (ROOT / spec["cpu_preflight"]["path"]).read_bytes()
    receipt = json.loads(receipt_blob)
    if (
        digest(receipt_blob) != spec["cpu_preflight"]["file_sha256"]
        or receipt.get("status") != "Succeeded"
        or receipt.get("gpu_request") != 0
        or receipt.get("job_and_pod_released") is not True
        or receipt.get("candidate_plan_sha256") != "85cbab43a21e195e231176b3e6246dc017204c9955a27e018c8ff0a3a86977f4"
        or receipt.get("candidate_request_sha256") != "c6fe8a4b142673a6d2645c9e35dc0e6bd0a3e4721cc372795e275fa853f82192"
    ):
        raise ValueError("v2 exact-image native CPU preflight has not passed")
    return spec


def stage_old_code(root: Path, *, successor: bool = False) -> None:
    for name in SOURCES:
        blob = subprocess.run(
            ["git", "show", f"{REVISION}:{name}"], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout
        if name in SOURCE_SHA256 and digest(blob) != SOURCE_SHA256[name]:
            raise ValueError(f"historical source digest mismatch: {name}")
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    if successor:
        # Repair the exact entrypoint mismatch, leaving data and recipe intact.
        for name, expected in (
            ("training/sft_262k_4node_v1.py", 2),
            ("training/sft_262k_runtime.py", 4),
            ("configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json", 4),
        ):
            path = root / name
            source = path.read_text()
            if source.count(OLD_NAME) != expected:
                raise ValueError(f"unexpected v2 identity edit site count: {name}")
            path.write_text(source.replace(OLD_NAME, NEW_NAME))
        path = root / "training/sft_262k_runtime.py"
        source = path.read_text()
        old = '''    "submission_authorized": False,
    "blockers": [
        "zero-GPU preflight receipt absent",
        "four-node GPU launch has not received root review",
    ],'''
        approved = successor_spec()["root_review"]
        new = '    "submission_authorized": True,\n    "blockers": [],\n    "approval_evidence": ' + repr(approved) + ','
        if source.count(old) != 1:
            raise ValueError("historical submission gate changed")
        source = source.replace(old, new)
        old = '        set(plan) != expected_keys'
        new = '        set(plan) - {"plan_sha256"} != expected_keys'
        if source.count(old) != 1:
            raise ValueError("historical runtime plan-key gate changed")
        source = source.replace(old, new)
        old = '    _BASE_VALIDATE_PLAN(plan, check_files=check_files)'
        new = old + '\n    if "plan_sha256" in plan and plan["plan_sha256"] != base._unsigned_digest({k: v for k, v in plan.items() if k != "plan_sha256"}):\n        raise ValueError("runtime plan digest changed")'
        if source.count(old) != 1:
            raise ValueError("historical runtime validation hook changed")
        path.write_text(source.replace(old, new))


def v4_spec(*, for_cpu_preflight: bool = False) -> dict:
    spec = json.loads(V4.read_text())
    if (set(spec) != {"schema", "source_v3_sha256", "name", "output_root", "delta", "root_review", "submission_authorized", "cpu_preflight"}
            or spec["schema"] != "qwen38_262k_four_node_repair_v2"
            or spec["source_v3_sha256"] != digest(SUCCESSOR.read_bytes())
            or spec["name"] != V4_NAME
            or spec["output_root"] != f"/mnt/sfs/jobs/{V4_NAME}"
            or spec["root_review"]["v3_deleted_rayjob_uid"] != "0b135d1f-3326-401b-a83c-68923d6e1b6b"
            or spec["root_review"]["v3_deleted_workload_uid"] != "14050301-32dd-4855-bcc9-9d3919b5619f"
            or spec["root_review"]["v3_delete_http_status"] != 204):
        raise ValueError("v4 repair provenance changed")
    if not for_cpu_preflight:
        if spec["submission_authorized"] is not True or not spec["cpu_preflight"]:
            raise ValueError("v4 GPU submission is closed until exact CPU preflight")
        receipt = spec["cpu_preflight"]
        blob = (ROOT / receipt["path"]).read_bytes()
        record = json.loads(blob)
        if (digest(blob) != receipt["file_sha256"] or record.get("status") != "Succeeded"
                or record.get("candidate_plan_sha256") != "3c630d5d924767d4ed78a823d1d1d6f5793e650bcc5c3221e1e86228adc7b8d2"
                or record.get("candidate_request_sha256") != "f8ff4d2f8cbdf81f95547b0e9e1ec53b3015de313efe639f39624c753c7d0389"
                or record.get("root_failure_alert_annotation") != "off"
                or record.get("gpu_request") != 0 or record.get("pod_restarts") != 0
                or record.get("exit_code") != 0 or record.get("job_and_pod_released") is not True):
            raise ValueError("v4 CPU preflight receipt differs")
    return spec


def stage_v4_code(root: Path) -> None:
    stage_old_code(root, successor=True)
    spec = v4_spec(for_cpu_preflight=True)
    for name in ("training/sft_262k_4node_v1.py", "training/sft_262k_runtime.py", "configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json"):
        path = root / name
        source = path.read_text()
        if NEW_NAME not in source:
            raise ValueError(f"v3 identity missing in {name}")
        path.write_text(source.replace(NEW_NAME, V4_NAME))
    runtime = root / "training/sft_runtime.py"
    before = runtime.read_text()
    old = '                trainer.dispatch.finalize_pending_saves("policy")'
    if before.count(old) != 1:
        raise ValueError("unreviewed post-pause finalizer")
    after = before.replace(old, '                pass  # native FSDP save already waited for completion')
    runtime.write_text(after)
    port = root / "training/sft_262k_runtime.py"
    source = port.read_text()
    if source.count(SOURCE_SHA256["training/sft_runtime.py"]) != 1:
        raise ValueError("unreviewed base runtime pin")
    source = source.replace(SOURCE_SHA256["training/sft_runtime.py"], digest(after.encode()))
    old_review = repr(successor_spec()["root_review"])
    if source.count(old_review) != 1:
        raise ValueError("v3 approval binding changed")
    port.write_text(source.replace(old_review, repr(spec["root_review"])))


def _old_python(root: Path, program: str, *, stdin: str = "") -> dict:
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", program, str(root)], input=stdin,
        text=True, capture_output=True, env=env, cwd=root,
    )
    if result.returncode:
        # The old Jobs client sanitizes HTTP errors. Never print the full
        # exception stream: a third-party library may embed a private value.
        raise RuntimeError("historical compiler or Jobs API preview failed; inspect locally")
    return json.loads(result.stdout)


def _compiled(root: Path) -> tuple[dict, dict]:
    result = _old_python(root, """
import json, sys
from pathlib import Path
from training.sft_262k_4node_v1 import compile_sft, job_request
root = Path(sys.argv[1])
config = json.loads((root/'configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json').read_text())
plan = compile_sft(config, relative_to=root/'configs/runs')
print(json.dumps({'plan': plan, 'request': job_request(plan)}, sort_keys=True))
""")
    return result["plan"], result["request"]


def historical_request(*, successor: bool = False) -> tuple[dict, dict]:
    spec = successor_spec() if successor else base_spec()
    with tempfile.TemporaryDirectory(prefix="q38-262k-4n-") as tmp:
        root = Path(tmp)
        stage_old_code(root, successor=successor)
        plan, request = _compiled(root)
    if (
        plan["qualification"]["submission_gate"]["submission_authorized"] is not successor
        or request["name"] != spec["name"]
        or request["run_dir"] != spec["output_root"]
        or request["image"] != json.loads(SPEC.read_text())["cluster"]["image"]
        or request["workers"] != 4
        or request["gpus_per_worker"] != 8
        or request["priority_class"] != "c1"
        or request["failureAlerts"] is not False
        or request["requeueIfPreempted"] is not False
    ):
        raise ValueError("historical request differs from the held 4×8 canary")
    verify_bundle(plan, request)
    return plan, request


def v4_request(*, for_cpu_preflight: bool = False) -> tuple[dict, dict]:
    spec = v4_spec(for_cpu_preflight=for_cpu_preflight)
    with tempfile.TemporaryDirectory(prefix="q38-262k-4n-v4-") as tmp:
        root = Path(tmp)
        stage_v4_code(root)
        plan, request = _compiled(root)
    if (request["name"] != spec["name"] or request["run_dir"] != spec["output_root"]
            or request["workers"] != 4 or request["gpus_per_worker"] != 8
            or request["priority_class"] != "c1" or request["failureAlerts"] is not False
            or plan["pause_after_step"] != 1 or plan["recipe"]["max_length"] != 262144
            or plan["qualification"]["submission_gate"]["approval_evidence"] != spec["root_review"]):
        raise ValueError("v4 canary request differs from reviewed repair")
    verify_bundle(plan, request)
    if not for_cpu_preflight:
        receipt = json.loads((ROOT / spec["cpu_preflight"]["path"]).read_text())
        exact = lambda obj: digest(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if (exact(plan) != receipt["candidate_plan_sha256"]
                or exact(request) != receipt["candidate_request_sha256"]):
            raise ValueError("v4 request differs from its exact-image CPU preflight")
    return plan, request


def v4_cpu_job() -> dict:
    """Run the exact gate-true entrypoint through its native CPU checks only."""
    _, request = v4_request(for_cpu_preflight=True)
    program = shlex.split(request["command"])
    if program[:2] != ["python", "-c"] or len(program) != 3:
        raise ValueError("unexpected GPU bootstrap command")
    old = "];runpy.run_path(str(p/'sft_runtime.py'),run_name='__main__')"
    if program[2].count(old) != 1:
        raise ValueError("unexpected runtime argv bootstrap")
    program[2] = program[2].replace(old, ",'--validate-only','--preflight-tokenize'];runpy.run_path(str(p/'sft_runtime.py'),run_name='__main__')")
    env = {k: v for k, v in request["env"].items() if k.startswith("CYBER_SFT_BUNDLE")}
    env.update(RUN_DIR="/tmp/q38-v4-cpu-preflight", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {
        "name": "chris-q38-262k4n-cpu-pre-v7", "namespace": "fleet-train-jobs",
        "annotations": {"fleet.ai/failure-alerts": "off"},
        "labels": {"cyber-post-train.fleet.ai/owner": "chris", "kueue.x-k8s.io/queue-name": "training-lq", "kueue.x-k8s.io/priority-class": "q1"}},
        "spec": {"suspend": True, "backoffLimit": 0, "activeDeadlineSeconds": 3600, "ttlSecondsAfterFinished": 3600,
        "template": {"spec": {"restartPolicy": "Never", "priorityClassName": "c1",
            "nodeSelector": {"kubernetes.io/arch": "amd64", "workload": "fleetai-training-ng-cpu"},
            "tolerations": [{"key": "workload", "operator": "Equal", "value": "fleetai-training-ng-cpu", "effect": "NoSchedule"}],
            "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True}}],
            "containers": [{"name": "preflight", "image": request["image"], "command": program,
                "env": [{"name": k, "value": v} for k, v in env.items()],
                "resources": {"requests": {"cpu": "4", "memory": "16Gi"}, "limits": {"cpu": "8", "memory": "24Gi"}},
                "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}],
                "securityContext": {"allowPrivilegeEscalation": False, "privileged": False}}]}}}}


def verify_bundle(plan: dict, request: dict) -> None:
    """CPU-only entrypoint/bundle integrity, not a GPU or SFS preflight."""
    env = request["env"]
    keys = sorted((k for k in env if k.startswith("CYBER_SFT_BUNDLE")), key=lambda k: (len(k), k))
    if keys == ["CYBER_SFT_BUNDLE"]:
        encoded = env[keys[0]]
    elif keys == [f"CYBER_SFT_BUNDLE_{i}" for i in range(len(keys))]:
        encoded = "".join(env[k] for k in keys)
    else:
        raise ValueError("runtime transport chunks are not contiguous")
    bundle = json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))
    if json.loads(bundle["plan"]) != plan:
        raise ValueError("runtime bundle plan changed")
    for name in ("runtime", "training/sft_runtime.py", "training/sft_262k_runtime.py"):
        source = bundle["runtime"] if name == "runtime" else bundle["extra_files"][name]
        compile(source, name, "exec")
    if digest(bundle["extra_files"]["training/sft_262k_runtime.py"].encode()) != plan["runtime_sha256"]:
        raise ValueError("runtime bundle source digest changed")
