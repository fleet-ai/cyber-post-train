"""Create-once, digest-bound SkyRL checkpoint → BF16 reload handoff.

Run each stage in the qualified trainer image with the shared SFS mounted.
Seal/export/CPU check/ready need zero visible GPUs; GPU check needs exactly one.
This module creates artifacts only. It never submits a cluster job or serves a model.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "c908d3a828d070c6b27611fc388b1e7e3b4049dd"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "training/__init__.py": "ecf358039bbb9b6cbab6546c9b1e61b9bc06c5b2d5b19907ff303a9277d77e27",
    "training/io.py": "7a0b734a4ab7fb8b702430094c58c72f19ac8fc7cc5e056eb8410267e6bfdfe3",
    "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
    "training/checkpoints.py": "b2bfa604a45a7ea1ed1b195401ce3c489f6460b36d2cd39d68a43033d23873b7",
    "training/export.py": "af0fc6cc2c388f8948f6be02a3cfedc189bbafbffa9511aacae35a08ce9125b1",
    "training/export_check.py": "a04811409178eedc6969e34766ca70c82d84c27b718944ecab93407609c4dfe7",
    "training/post_sft_artifacts.py": "c09555c61cb25ad5867cce80ed8bef0684734231b5708b1be892e75b9dd980eb",
    "training/post_sft_cast.py": "9c531664dbd85b03c2f62d5847f0691ebe193154f72386c5ff395c54a74a8fc0",
    "training/post_sft_base_surface.py": "f2e82927348fe844ff685f6506e1799a53cb9ecfbe58d50c654f2b1ba70c4f4f",
}
STAGES = ("seal", "export", "cpu", "gpu", "ready")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable receipt file missing or symlinked")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("immutable file changed during hashing")
    return digest.hexdigest()


def _receipt(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable receipt file missing or symlinked")
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or not re.fullmatch(r"[a-f0-9]{64}", value.get("receipt_sha256", ""))
        or value["receipt_sha256"] != _sha(_canonical({k: v for k, v in value.items() if k != "receipt_sha256"}))):
        raise ValueError("receipt digest mismatch")
    return value


def _prepared(directory: Path) -> tuple[dict, dict, dict]:
    receipt = json.loads((directory / "PREPARED.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if (receipt.get("schema") not in {"qwen38_96k_debug_prepared_v1",
                                           "qwen38_96k_full_prepared_v1"}
        or receipt.get("historical_commit") != COMMIT
        or receipt.get("plan_sha256") != _sha(_canonical(plan))
        or receipt.get("request_sha256") != _sha(_canonical(request))
        or plan.get("runtime_sha256") != SOURCES["training/sft_runtime.py"]
        or request.get("image") != plan.get("execution", {}).get("image")):
        raise ValueError("prepared run digest/source binding changed")
    return plan, request, receipt


def _sources() -> dict[str, bytes]:
    result = {}
    for name, expected in SOURCES.items():
        if os.environ.get("CYBER_CKPT_SOURCE_DIR"):
            path = Path(os.environ["CYBER_CKPT_SOURCE_DIR"]) / name
            raw = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
        else:
            source = subprocess.run(
                ["git", "show", f"{COMMIT}:{name}"], cwd=ROOT,
                capture_output=True, check=False,
            )
            raw = source.stdout if source.returncode == 0 else b""
        if _sha(raw) != expected:
            raise ValueError("reviewed historical source unavailable or changed")
        result[name] = raw
    return result


def _historical(stage: str, value: dict) -> None:
    """Execute only reviewed historical source; never expose private exception text."""
    script = """
import json,sys
from pathlib import Path
v=json.load(sys.stdin)
try:
    if v['stage']=='seal':
        from training.checkpoints import seal
        seal(v['plan'],v['step'],Path(v['output']))
    elif v['stage']=='export':
        from training.export import export
        export(Path(v['seal']),v['seal_sha256'],Path(v['output']))
    elif v['stage'] in ('cpu','gpu'):
        from training.export_check import check
        check(Path(v['export']),v['export_sha256'],Path(v['output']),gpu=v['stage']=='gpu')
    else: raise ValueError('unsupported stage')
    print('passed')
except BaseException as exc:
    print(type(exc).__name__,file=sys.stderr)
    sys.exit(2)
"""
    with tempfile.TemporaryDirectory(prefix="q38-checkpoint-flow-") as temporary:
        root = Path(temporary)
        for name, raw in _sources().items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        result = subprocess.run(
            [sys.executable, "-c", script], input=_canonical({"stage": stage, **value}),
            cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True, check=False,
        )
    if result.returncode or result.stdout.strip() != b"passed":
        raise ValueError(f"historical {stage} gate failed; inspect private worker evidence")


def _paths(plan: dict, step: int) -> dict[str, Path]:
    recipe = plan["recipe"]
    if (type(step) is not int or not 0 < step <= recipe["max_steps"]
        or (step != recipe["max_steps"] and step % recipe["checkpoint_interval"])):
        raise ValueError("step is not a planned checkpoint")
    root = Path(plan["output_root"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("output root must be an absolute non-symlink path")
    slot = root / "checkpoint-eval" / f"step-{step:06d}"
    return {"slot": slot, "seal": slot / "SEAL.json", "model": slot / "bf16",
            "export": slot / "bf16" / "EXPORT.json", "cpu": slot / "CPU_CHECK.json",
            "gpu": slot / "GPU_CHECK.json", "ready": slot / "CHECKPOINT_READY.json"}


def _seal(paths: dict, plan: dict, step: int) -> tuple[dict, str]:
    path = paths["seal"]
    value, file_sha = _receipt(path), _file_sha(path)
    if (value.get("schema") != "cyber_skyrl_checkpoint_manifest_v1"
        or value.get("source_plan_sha256") != _sha(_canonical(plan))
        or value.get("source_plan") != plan or value.get("optimizer_step") != step
        or value.get("checkpoint_path") != str(Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}")
        or value.get("gpu_reload_verified") is not False):
        raise ValueError("seal differs from exact run/step")
    return value, file_sha


def _export(paths: dict, seal: dict, seal_sha: str, step: int) -> tuple[dict, str]:
    path = paths["export"]
    value, file_sha = _receipt(path), _file_sha(path)
    if (value.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or value.get("source_checkpoint_receipt_sha256") != seal["receipt_sha256"]
        or value.get("source_manifest_file_sha256") != seal_sha
        or value.get("source_plan_sha256") != seal["source_plan_sha256"]
        or value.get("optimizer_step") != step
        or value.get("output_root") != str(paths["model"])
        or value.get("dtype") != "BF16" or value.get("optimizer_steps_executed") != 0
        or value.get("source_inventory_sizes_mtimes_unchanged") is not True
        or value.get("all_output_tensors_reopened_equal") is not True
        or not value.get("files")):
        raise ValueError("BF16 export differs from sealed checkpoint")
    return value, file_sha


def _check(path: Path, export: dict, export_sha: str, gpu: bool) -> dict:
    value = _receipt(path)
    if (value.get("schema") != "cyber_hf_export_check_v1"
        or value.get("status") != "passed"
        or value.get("export_sha256") != export_sha
        or value.get("export_receipt_sha256") != export["receipt_sha256"]
        or value.get("optimizer_steps_executed") != 0
        or value.get("gpus") != int(gpu)
        or value.get("gpu_reload_verified") is not gpu
        or value.get("source_unchanged") is not True
        or value.get("serving_qualified") is not False
        or (gpu and (value.get("finite_logits") is not True or value.get("generated_tokens", 0) <= 0))):
        raise ValueError("export check failed exact CPU/GPU gate")
    return value


def _payload(paths: dict, exported: dict) -> None:
    root = paths["model"]
    for name, spec in exported["files"].items():
        part = Path(name)
        if (part.is_absolute() or len(part.parts) != 1 or part.name != name
            or type(spec.get("bytes")) is not int
            or not re.fullmatch(r"[a-f0-9]{64}", spec.get("sha256", ""))):
            raise ValueError("invalid BF16 payload manifest")
        path = root / name
        if path.stat().st_size != spec["bytes"] or _file_sha(path) != spec["sha256"]:
            raise ValueError("BF16 payload changed after reload check")


def _teacher_ce(plan: dict, prepared: dict, step: int) -> dict:
    path = Path(plan["output_root"]) / "validation" / f"step-{step:06d}.json"
    dev = _receipt(path)
    if dev.get("optimizer_step") != step or dev.get("plan_sha256") != prepared["plan_sha256"]:
        raise ValueError("teacher-loss development receipt differs from checkpoint")
    if (any(type(dev.get(k)) not in (int, float) or not math.isfinite(dev[k]) or dev[k] < 0
            for k in ("eval_loss", "task_macro_loss"))
        or any(type(dev.get(k)) is not int or dev[k] <= 0
               for k in ("supervised_tokens", "windows", "tasks"))
        or dev["tasks"] != len(set(plan["datasets"]["dev"]["task_keys"]))):
        raise ValueError("teacher-loss development metrics are incomplete")
    return {"teacher_loss_receipt_sha256": dev["receipt_sha256"],
            "teacher_loss_file_sha256": _file_sha(path)}


def run(directory: Path, step: int, stage: str) -> dict:
    if stage not in STAGES:
        raise ValueError("unknown checkpoint stage")
    plan, request, prepared = _prepared(directory)
    if (request.get("name") != plan.get("run_name")
        or request.get("run_dir") != plan.get("output_root")
        or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False):
        raise ValueError("prepared run binding/cluster policy differs")
    paths = _paths(plan, step)
    output = paths[stage] if stage != "export" else paths["model"]
    if output.exists() or output.is_symlink():
        raise ValueError("create-once checkpoint output already exists")
    if stage == "seal":
        paths["slot"].mkdir(parents=True, exist_ok=True)
        _historical(stage, {"plan": plan, "step": step, "output": str(output)})
        _seal(paths, plan, step)
    else:
        seal, seal_sha = _seal(paths, plan, step)
        if stage == "export":
            _historical(stage, {"seal": str(paths["seal"]), "seal_sha256": seal_sha,
                                "output": str(output)})
            _export(paths, seal, seal_sha, step)
        else:
            exported, export_sha = _export(paths, seal, seal_sha, step)
            if stage in ("cpu", "gpu"):
                _historical(stage, {"export": str(paths["export"]),
                                    "export_sha256": export_sha, "output": str(output)})
                _check(output, exported, export_sha, stage == "gpu")
            else:
                cpu = _check(paths["cpu"], exported, export_sha, False)
                gpu = _check(paths["gpu"], exported, export_sha, True)
                _payload(paths, exported)
                proof = {"schema": "qwen38_checkpoint_ready_v1", "status": "ready_for_route_parity",
                         "source_run": request["name"], "optimizer_step": step,
                         "historical_commit": COMMIT, "bridge_sha256": _file_sha(Path(__file__)),
                         "prepared_sha256": _file_sha(directory / "PREPARED.json"),
                         "plan_sha256": prepared["plan_sha256"],
                         "request_sha256": prepared["request_sha256"],
                         "checkpoint_sha256": seal_sha,
                         "seal_receipt_sha256": seal["receipt_sha256"],
                         "export_sha256": export_sha,
                         "export_receipt_sha256": exported["receipt_sha256"],
                         "export_payload_sha256": _sha(_canonical(exported["files"])),
                         "export_path": str(paths["model"]),
                         "cpu_check_sha256": _file_sha(paths["cpu"]),
                         "gpu_check_sha256": _file_sha(paths["gpu"]),
                         "cpu_check_receipt_sha256": cpu["receipt_sha256"],
                         "gpu_check_receipt_sha256": gpu["receipt_sha256"],
                         "serving_qualified": False, "task_evaluated": False}
                if plan.get("validation_mode") == "teacher_cross_entropy":
                    proof.update(_teacher_ce(plan, prepared, step))
                proof["receipt_sha256"] = _sha(_canonical(proof))
                with output.open("x") as stream:
                    stream.write(json.dumps(proof, sort_keys=True, indent=2) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
    return {"stage": stage, "source_run": request["name"], "optimizer_step": step,
            "output": str(output), "sha256": _file_sha(output if stage != "export" else paths["export"])}


def stage_spec(directory: Path, step: int, stage: str) -> dict:
    """Build, but do not submit, a c1/q1 stage Job or GPU Jobs-API request."""
    if stage not in STAGES:
        raise ValueError("unknown checkpoint stage")
    plan, request, receipt = _prepared(directory)
    paths = _paths(plan, step)
    if (request.get("name") != plan.get("run_name")
        or request.get("run_dir") != plan.get("output_root")
        or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False):
        raise ValueError("prepared run binding/cluster policy differs")
    destination = paths["model" if stage == "export" else stage]
    if destination.exists() or destination.is_symlink():
        raise ValueError("create-once checkpoint output already exists")
    if stage == "seal":
        source = _receipt(Path(plan["output_root"]) / "checkpoint_receipts" /
                          f"step-{step:06d}.json")
        if (source.get("plan_sha256") != receipt["plan_sha256"]
            or source.get("optimizer_step") != step
            or source.get("checkpoint_path") != str(Path(plan["output_root"]) /
                                                    "checkpoints" / f"global_step_{step}")):
            raise ValueError("saved source checkpoint receipt differs from prepared run")
    if stage != "seal":
        sealed, seal_sha = _seal(paths, plan, step)
        if stage != "export":
            exported, export_sha = _export(paths, sealed, seal_sha, step)
            if stage == "ready":
                _check(paths["cpu"], exported, export_sha, False)
                _check(paths["gpu"], exported, export_sha, True)
                if plan.get("validation_mode") == "teacher_cross_entropy":
                    _teacher_ce(plan, receipt, step)
    files = {**{name: raw.decode() for name, raw in _sources().items()},
             "training/checkpoint_flow.py": Path(__file__).read_text(),
             "prepared/plan.json": (directory / "plan.json").read_text(),
             "prepared/request.json": (directory / "request.json").read_text(),
             "prepared/PREPARED.json": (directory / "PREPARED.json").read_text()}
    package = {"files": files, "digests": {name: _sha(text.encode()) for name, text in files.items()}}
    blob = gzip.compress(_canonical(package), mtime=0)
    encoded = base64.b64encode(blob).decode()
    chunks = [encoded[i:i + 48000] for i in range(0, len(encoded), 48000)]
    if not 1 <= len(chunks) <= 32:
        raise ValueError("checkpoint stage bundle exceeds environment bound")
    env = {"CKPT_BUNDLE_SHA256": _sha(blob), "CKPT_BUNDLE_COUNT": str(len(chunks)),
           "CUDA_VISIBLE_DEVICES": "0" if stage == "gpu" else "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1",
           **{f"CKPT_BUNDLE_{i}": part for i, part in enumerate(chunks)}}
    worker = """import base64,gzip,hashlib,json,os,pathlib,runpy,sys,tempfile
raw=base64.b64decode(''.join(os.environ.pop('CKPT_BUNDLE_'+str(i)) for i in range(int(os.environ.pop('CKPT_BUNDLE_COUNT')))),validate=True)
if hashlib.sha256(raw).hexdigest()!=os.environ.pop('CKPT_BUNDLE_SHA256'): raise SystemExit('bundle digest mismatch')
v=json.loads(gzip.decompress(raw)); t=tempfile.TemporaryDirectory(prefix='q38-checkpoint-stage-'); root=pathlib.Path(t.name)
for name,content in v['files'].items():
    if name.startswith('/') or '..' in pathlib.PurePosixPath(name).parts or hashlib.sha256(content.encode()).hexdigest()!=v['digests'][name]: raise SystemExit('bundle file mismatch')
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
os.environ['CYBER_CKPT_SOURCE_DIR']=str(root);sys.path.insert(0,str(root));os.chdir(root)
sys.argv=['checkpoint_flow',os.environ['CKPT_STAGE'],str(root/'prepared'),os.environ['CKPT_STEP']]
runpy.run_module('training.checkpoint_flow',run_name='__main__')
"""
    env.update({"CKPT_STAGE": stage, "CKPT_STEP": str(step)})
    name = f"q38-ck-{receipt['plan_sha256'][:8]}-{step:06d}-{stage}"
    if len(name) > 31 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name):
        raise ValueError("checkpoint stage name is not a Kubernetes name")
    command = ["python", "-u", "-c", worker]
    if stage == "gpu":
        return {"api": "Fleet Jobs preview then create-once POST only",
                "requires_rendered_root_annotation": {"fleet.ai/failure-alerts": "off"},
                "request": {"name": name, "title": name, "run_dir": str(paths["slot"] / "gpu-check-run"),
                            "image": request["image"], "command": "python -u -c " + shlex.quote(worker),
                            "workers": 1, "gpus_per_worker": 1,
                            "resources": {"cpu_request": "16", "cpu_limit": "32",
                                          "memory_request": "96Gi", "memory_limit": "192Gi"},
                            "priority_class": "c1", "requeueIfPreempted": False,
                            "failureAlerts": False, "env": env}}
    return {"api": "Kubernetes server dry-run then create-once; unsuspend only after review",
            "job": {"apiVersion": "batch/v1", "kind": "Job",
                    "metadata": {"name": name, "namespace": "fleet-train-jobs",
                                 "annotations": {"fleet.ai/failure-alerts": "off",
                                                 "cyber-post-train.fleet.ai/plan-sha256": receipt["plan_sha256"],
                                                 "cyber-post-train.fleet.ai/checkpoint-step": str(step)},
                                 "labels": {"kueue.x-k8s.io/queue-name": "training-lq",
                                            "kueue.x-k8s.io/priority-class": "q1"}},
                    "spec": {"suspend": True, "backoffLimit": 0,
                             "activeDeadlineSeconds": 10800, "ttlSecondsAfterFinished": 7200,
                             "template": {"spec": {"restartPolicy": "Never",
                                "automountServiceAccountToken": False,
                                "priorityClassName": "c1", "priority": 10000,
                                "nodeSelector": {"kubernetes.io/arch": "amd64",
                                                 "workload": "fleetai-training-ng-cpu"},
                                "tolerations": [{"key": "workload", "operator": "Equal",
                                                 "value": "fleetai-training-ng-cpu", "effect": "NoSchedule"}],
                                "containers": [{"name": stage, "image": request["image"],
                                                "command": command,
                                                "env": [{"name": k, "value": v} for k, v in sorted(env.items())],
                                                "resources": {"requests": {"cpu": "16", "memory": "96Gi"},
                                                              "limits": {"cpu": "32", "memory": "192Gi"}},
                                                "securityContext": {"allowPrivilegeEscalation": False,
                                                                    "privileged": False},
                                                "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}]}],
                                "volumes": [{"name": "sfs", "persistentVolumeClaim":
                                             {"claimName": "sfs-shared"}}]}}}}}


def validate_stage_preview(spec: dict, server_preview: dict) -> dict:
    """Fail closed on the server-rendered root alert, c1/q1 and resources."""
    if "job" in spec:
        expected = spec["job"]
        meta, actual = server_preview.get("metadata", {}), server_preview.get("spec", {})
        pod = actual.get("template", {}).get("spec", {})
        wanted = expected["spec"]["template"]["spec"]
        if (server_preview.get("kind") != "Job"
            or meta.get("name") != expected["metadata"]["name"]
            or meta.get("namespace") != "fleet-train-jobs"
            or meta.get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
            or any(meta.get("annotations", {}).get(k) != v
                   for k, v in expected["metadata"]["annotations"].items())
            or meta.get("labels", {}).get("kueue.x-k8s.io/queue-name") != "training-lq"
            or meta.get("labels", {}).get("kueue.x-k8s.io/priority-class") != "q1"
            or actual.get("suspend") is not True or actual.get("backoffLimit") != 0
            or pod.get("priorityClassName") != "c1" or pod.get("priority") != 10000
            or pod.get("nodeSelector") != wanted["nodeSelector"]
            or pod.get("automountServiceAccountToken") is not False
            or pod.get("volumes") != wanted["volumes"]
            or len(pod.get("containers", [])) != 1
            or pod["containers"][0].get("image") != wanted["containers"][0]["image"]
            or pod["containers"][0].get("command") != wanted["containers"][0]["command"]
            or pod["containers"][0].get("env") != wanted["containers"][0]["env"]
            or pod["containers"][0].get("resources") != wanted["containers"][0]["resources"]
            or pod["containers"][0].get("securityContext") != wanted["containers"][0]["securityContext"]
            or pod["containers"][0].get("envFrom")
            or pod["containers"][0].get("volumeMounts") != wanted["containers"][0]["volumeMounts"]
            or "nvidia.com/gpu" in str(pod["containers"][0].get("resources", {}))):
            raise ValueError("server-rendered CPU checkpoint Job drifted")
        return {"status": "previewed_not_created", "kind": "Job",
                "job_sha256": _sha(_canonical(spec["job"]))}
    if "request" in spec:
        import yaml

        request = spec["request"]
        if server_preview.get("errors") or server_preview.get("warnings"):
            raise ValueError("GPU checkpoint preview reported errors/warnings")
        obj = yaml.safe_load(server_preview["manifest_yaml"])
        meta, actual = obj.get("metadata", {}), obj.get("spec", {})
        if (obj.get("kind") != "RayJob"
            or meta.get("name") != request["name"]
            or meta.get("namespace") != "fleet-train-jobs"
            or meta.get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
            or meta.get("annotations", {}).get("fleet.ai/run-dir") != request["run_dir"]
            or meta.get("labels", {}).get("kueue.x-k8s.io/queue-name") != "training-lq"
            or meta.get("labels", {}).get("kueue.x-k8s.io/priority-class") != "q1"
            or actual.get("suspend") is not True
            or actual.get("shutdownAfterJobFinishes") is not True
            or actual.get("entrypoint") != request["command"]):
            raise ValueError("server-rendered GPU checkpoint RayJob drifted")
        cluster = actual.get("rayClusterSpec", {})
        groups = [(1, cluster.get("headGroupSpec", {}).get("template", {}))] + [
            (group.get("replicas", 0), group.get("template", {}))
            for group in cluster.get("workerGroupSpecs", [])]
        count = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise ValueError("invalid GPU checkpoint replica count")
            pod = template.get("spec", {})
            if replicas and (pod.get("priorityClassName") != "c1" or pod.get("nodeName")):
                raise ValueError("GPU checkpoint pod priority/node drifted")
            for container in pod.get("containers", []):
                gpu = container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)
                if gpu:
                    if container.get("image") != request["image"]:
                        raise ValueError("GPU checkpoint image drifted")
                    count += replicas * int(gpu)
        if count != 1:
            raise ValueError("GPU checkpoint must allocate exactly one GPU")
        return {"status": "previewed_not_created", "kind": "RayJob",
                "request_sha256": _sha(_canonical(request)),
                "manifest_sha256": _sha(server_preview["manifest_yaml"].encode())}
    raise ValueError("checkpoint stage spec is missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "spec"))
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--for-stage", choices=STAGES, help="stage for spec generation")
    args = parser.parse_args()
    try:
        result = (stage_spec(args.prepared_dir, args.step, args.for_stage)
                  if args.stage == "spec" and args.for_stage
                  else run(args.prepared_dir, args.step, args.stage))
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        # Private trainer paths, model contents, or source may appear in exceptions.
        print(f"checkpoint stage rejected ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
