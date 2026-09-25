"""Prepare and preview the historical 4×8 Qwen3.8 262K canary; never submit.

The compiler and runtime are exact Git blobs from the held 65fcf038 candidate.
This bridge stages them in a temporary directory, then discards that directory.
Only a reviewed, create-once request may later be handed to an operator.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from training.long_context import ROOT, SPEC, validate_length_audit, validate_real_row_witness, validate_rendered_job, validate_spec


REVISION = "65fcf03812e48cf6f0845c607152b2069f3275b4"
SUCCESSOR = ROOT / "configs/runs/qwen38-262k-four-node-canary-v2.json"
OLD_NAME = "chris-q38-t3k262-4n-can-v1"
NEW_NAME = "chris-q38-t3k262-4n-can-v2"
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


def successor_spec() -> dict:
    spec = json.loads(SUCCESSOR.read_text())
    base = json.loads(SPEC.read_text())
    validate_spec(base)
    if (
        set(spec) != {"schema", "base_spec", "base_spec_sha256", "name", "output_root", "delta", "accepted", "submission_authorized", "root_review", "cpu_preflight", "remaining_gates"}
        or spec["schema"] != "qwen38_262k_four_node_successor_v1"
        or spec["base_spec"] != str(SPEC.relative_to(ROOT))
        or spec["base_spec_sha256"] != digest(SPEC.read_bytes())
        or spec["name"] != NEW_NAME
        or spec["output_root"] != f"/mnt/sfs/jobs/{NEW_NAME}"
        or spec["delta"] != "create-once identity and root-approved gate only; no trainer, model, data, or recipe change"
        or spec["accepted"] is not False
        or spec["submission_authorized"] is not True
        or spec["root_review"] != {
            "scope": "one create-once v2 four-node capacity canary; one finite step, checkpoint, and reload; not scientific training",
            "v1_cpu_receipt_sha256": "39c827afa6db9733e175c6ebb3b5c75478e661d5266a265fa6402f40132b5cf0",
            "v2_false_gate_preview_sha256": "19b4bd9265751fa96992daff8004ac6348de9a948e5f7403f597d28b11752963",
        }
        or spec["cpu_preflight"] != {
            "path": "docs/evidence/qwen38-262k-four-node-cpu-preflight-v5-passed.json",
            "file_sha256": "699dd2362f181f615cd4bbda8e969a97cc24d65286908dcd5935651340903235",
        }
        or len(spec["remaining_gates"]) != 3
    ):
        raise ValueError("v2 successor intent or root review changed")
    validate_length_audit(base)
    validate_real_row_witness(base)
    receipt_blob = (ROOT / spec["cpu_preflight"]["path"]).read_bytes()
    receipt = json.loads(receipt_blob)
    if (
        digest(receipt_blob) != spec["cpu_preflight"]["file_sha256"]
        or receipt.get("status") != "Succeeded"
        or receipt.get("gpu_request") != 0
        or receipt.get("job_and_pod_released") is not True
        or receipt.get("candidate_plan_sha256") != "c1aa4371a94efab1fee41bff436615db527afa779fd978abaa07d2527df6efec"
        or receipt.get("candidate_request_sha256") != "3cb50179559f6b819da1b477f6fcf0eff9a8a53a6d9b9540cf65805f1d9667c3"
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
        # Only identity and the now-reviewed capacity-canary gate change.
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
        path.write_text(source.replace(old, new))


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


def historical_request(*, successor: bool = False) -> tuple[dict, dict]:
    spec = successor_spec() if successor else json.loads(SPEC.read_text())
    if not successor:
        validate_spec(spec)
        validate_length_audit(spec)
        validate_real_row_witness(spec)
    with tempfile.TemporaryDirectory(prefix="q38-262k-4n-") as tmp:
        root = Path(tmp)
        stage_old_code(root, successor=successor)
        result = _old_python(root, """
import json, sys
from pathlib import Path
from training.sft_262k_4node_v1 import compile_sft, job_request
root = Path(sys.argv[1])
config = json.loads((root/'configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json').read_text())
plan = compile_sft(config, relative_to=root/'configs/runs')
print(json.dumps({'plan': plan, 'request': job_request(plan)}, sort_keys=True))
""")
    plan, request = result["plan"], result["request"]
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


def _write_new(path: Path, content: str) -> None:
    with path.open("x") as f:
        f.write(content)
    path.chmod(0o600)


def prepare(dest: Path, *, successor: bool = False) -> dict:
    plan, request = historical_request(successor=successor)
    dest.mkdir(parents=True, exist_ok=False)
    _write_new(dest / "plan.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _write_new(dest / "request.json", json.dumps(request, indent=2, sort_keys=True) + "\n")
    result = {
        "state": "GATE_TRUE_REVIEW_ONLY_NO_GPU_POST" if successor else "PREVIEW_ONLY_SUBMISSION_BLOCKED",
        "historical_revision": REVISION,
        "name": request["name"],
        "output_root": request["run_dir"],
        "nodes": 4,
        "gpus": 32,
        "plan_sha256": digest(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()),
        "request_sha256": digest(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()),
        "real_row_witness": str(ROOT / json.loads(SPEC.read_text())["real_row_witness"]["path"]),
        "limits": "No POST /v1/runs, GPU fit, optimizer update, checkpoint reload, or scientific acceptance.",
    }
    _write_new(dest / "prepare-summary.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def preview(dest: Path, *, successor: bool = False) -> dict:
    request = json.loads((dest / "request.json").read_text())
    _, exact_request = historical_request(successor=successor)
    if request != exact_request:
        raise ValueError("saved request changed after preparation")
    spec = json.loads(SPEC.read_text())
    with tempfile.TemporaryDirectory(prefix="q38-262k-4n-") as tmp:
        root = Path(tmp)
        stage_old_code(root, successor=successor)
        result = _old_python(root, """
import json, os, sys
from cyber_post_train.jobs import Jobs
request = json.loads(sys.stdin.read())
with Jobs(os.environ.get('FLEET_API_KEY', '')) as jobs:
    rows = jobs.all_runs()
    duplicates = [r['name'] for r in rows if r.get('name') == request['name'] or r.get('name', '').startswith(request['name']+'-') or r.get('run_dir') == request['run_dir'] or r.get('title') == request['title']]
    if duplicates: raise ValueError('exact name/title/output already recorded; no preview as launch candidate')
    p = jobs.preview(request)
print(json.dumps({'manifest_yaml': p['manifest_yaml'], 'warnings': p.get('warnings'), 'errors': p.get('errors'), 'run_count': len(rows)}))
""", stdin=json.dumps(request))
    rendered = yaml.safe_load(result["manifest_yaml"])
    validate_rendered_job(spec, rendered)
    _write_new(dest / "rendered-rayjob.yaml", result["manifest_yaml"])
    summary = {
        "state": "RENDERED_PREVIEW_REVIEW_REQUIRED_NO_SUBMISSION",
        "name": request["name"],
        "run_count_seen": result["run_count"],
        "duplicates": 0,
        "rendered_rayjob_sha256": digest(result["manifest_yaml"].encode()),
        "root_failed_job_alerts": rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"],
        "priority": rendered["metadata"]["labels"]["kueue.x-k8s.io/priority-class"],
        "nodes": 4,
        "gpus": 32,
        "submission_authorized": successor,
    }
    _write_new(dest / "preview-summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def cpu_preflight_job(*, successor: bool = False) -> dict:
    """A one-shot 0-GPU Job on the exact image; never creates it here."""
    plan, request = historical_request(successor=successor)
    with tempfile.TemporaryDirectory(prefix="q38-262k-4n-") as tmp:
        root = Path(tmp)
        stage_old_code(root, successor=successor)
        sources = {name: (root / name).read_text() for name in SOURCES}
    compressed = gzip.compress(json.dumps(sources, sort_keys=True).encode(), mtime=0)
    encoded = base64.b64encode(compressed).decode()
    if len(encoded) > 120000:
        raise ValueError("CPU preflight source bundle exceeds one environment value")
    code = f"""
import base64,gzip,hashlib,json,os,sys,tempfile
from pathlib import Path
stage='bundle_decode'
try:
    blob=base64.b64decode(os.environ.pop('CYBER_PREFLIGHT_BUNDLE'),validate=True)
    assert hashlib.sha256(blob).hexdigest()=={digest(compressed)!r}
    sources=json.loads(gzip.decompress(blob))
    with tempfile.TemporaryDirectory(prefix='q38-262k-cpu-') as tmp:
        stage='source_stage'
        root=Path(tmp)
        for name,text in sources.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        sys.path.insert(0,tmp)
        stage='compiler_import'
        from training.sft_262k_4node_v1 import compile_sft,job_request,preflight
        from cyber_post_train.jobs import digest
        stage='plan_compile'
        config=json.loads((root/'configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json').read_text())
        plan=compile_sft(config,relative_to=root/'configs/runs')
        stage='plan_request_binding'
        assert digest(plan)=={digest(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode())!r}
        assert digest(job_request(plan))=={digest(json.dumps(request, sort_keys=True, separators=(',', ':')).encode())!r}
        # The historical generic preflight's final receipt calls the generic
        # request builder, which rejects this candidate's different runtime
        # digest. Point only that receipt call at the exact candidate builder.
        from training import sft
        sft.job_request=job_request
        stage='native_preflight'
        result=preflight(plan)
        assert result['request_sha256']=={digest(json.dumps(request, sort_keys=True, separators=(',', ':')).encode())!r}
        assert result['plan_sha256']=={digest(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode())!r}
        print(json.dumps({{'status':'passed','candidate_plan_sha256':digest(plan),'candidate_request_sha256':digest(job_request(plan)),'native_checked':result.get('checked'),'counts':result.get('counts'),'gpus':0}},sort_keys=True))
except BaseException as exc:
    frames=[];trace=exc.__traceback__
    while trace:
        frames.append([Path(trace.tb_frame.f_code.co_filename).name,trace.tb_frame.f_code.co_name,trace.tb_lineno])
        trace=trace.tb_next
    print(json.dumps({{'status':'failed','stage':stage,'error_class':type(exc).__name__,'site_chain':frames[-5:]}},sort_keys=True))
    raise SystemExit(1)
"""
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "name": "chris-q38-262k4n-cpu-pre-v5" if successor else "chris-q38-262k4n-cpu-pre-v4", "namespace": "fleet-train-jobs",
            "annotations": {"fleet.ai/failure-alerts": "off"},
            "labels": {"kueue.x-k8s.io/queue-name": "training-lq", "kueue.x-k8s.io/priority-class": "q1"},
        },
        "spec": {
            "backoffLimit": 0, "activeDeadlineSeconds": 1800,
            "ttlSecondsAfterFinished": 600,
            "template": {"spec": {
                "restartPolicy": "Never", "priorityClassName": "c1",
                "containers": [{
                    "name": "preflight", "image": request["image"],
                    "command": ["python", "-c", code],
                    "env": [
                        {"name": "CYBER_PREFLIGHT_BUNDLE", "value": encoded},
                        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                        {"name": "HF_HUB_OFFLINE", "value": "1"},
                        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                    ],
                    "resources": {"requests": {"cpu": "4", "memory": "16Gi", "ephemeral-storage": "2Gi"}, "limits": {"cpu": "8", "memory": "24Gi", "ephemeral-storage": "4Gi"}},
                    "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}],
                }],
                "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True}}],
            }},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preview", "prepare-v2", "preview-v2"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    successor = args.action.endswith("-v2")
    print(json.dumps(prepare(args.directory, successor=successor) if args.action.startswith("prepare") else preview(args.directory, successor=successor), sort_keys=True))


if __name__ == "__main__":
    main()
