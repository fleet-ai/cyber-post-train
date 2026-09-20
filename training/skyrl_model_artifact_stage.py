"""Create one immutable dev FleetJob model artifact without allocating a GPU.

The proven development-cluster model bytes currently live under a historical
single-component SFS alias.  FleetJob deliberately accepts only a project and
artifact source.  This bounded, create-once stage links the exact digest-bound
files into a project/artifact directory on the same SFS filesystem.  It never
loads a task, starts an inference engine, performs a rollout, or imports a
trainer.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
from contextlib import suppress
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import JobsError, bundled_request, digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/qualification/qwen38-skyrl-model-artifact-stage-dev-v1.json"
)
CONFIG_SCHEMA = "cyber_skyrl_model_artifact_stage_config_v1"
PLAN_SCHEMA = "cyber_skyrl_model_artifact_stage_plan_v1"
RECEIPT_SCHEMA = "cyber_skyrl_model_artifact_stage_receipt_v1"
REJECTION_SCHEMA = "cyber_skyrl_model_artifact_stage_rejection_v1"
PACKET_SCHEMA = "cyber_skyrl_model_artifact_stage_job_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_model_artifact_stage_job_preview_v1"
MODULE = "training.skyrl_model_artifact_stage"
RECEIPT_PATH = "/dev/termination-log"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RUNTIME_FILES = (
    "training/skyrl_model_artifact_stage.py",
    "cyber_post_train/jobs.py",
)


class StageGateError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value != _seal(value):
        raise ValueError("SkyRL model artifact stage digest changed")


def _runtime() -> dict[str, str]:
    return {name: (ROOT / name).read_text() for name in RUNTIME_FILES}


def _read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("model artifact stage input must be a JSON mapping")
    return value


def _expected_execution() -> dict:
    return {
        "cluster_target": "dev",
        "kubernetes_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
        "namespace": "fleet-train-jobs",
        "priority": "c1",
        "pvc": "sfs-shared",
        "models_subpath": "models",
        "models_mount": "/mnt/models",
        "source_alias": "qwen3.8-27b-1d4bf0f2",
        "artifact_path": "fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v2",
        "runtime_uid": 1000,
        "runtime_gid": 100,
        "deadline_seconds": 1200,
    }


def _config(path: Path) -> dict:
    if path.resolve() != CONFIG_PATH.resolve() or path.is_symlink():
        raise ValueError("unknown SkyRL model artifact stage configuration")
    value = _read_mapping(path)
    _validate_seal(value, CONFIG_SCHEMA)
    if value["execution"] != _expected_execution():
        raise ValueError("SkyRL model artifact stage execution contract changed")
    return value


def compile_stage(path: Path) -> dict:
    # Compiler-only dependency: the runtime consumes the already sealed model
    # inventory and therefore does not carry the full training compiler graph.
    from .models import bound_model

    value = _config(path)
    model = bound_model(
        _read_mapping(CONFIG_PATH.parent / value["model"]["lock"]),
        _read_mapping(CONFIG_PATH.parent / value["model"]["weights"]),
        "/mnt/models/" + value["execution"]["source_alias"],
    )
    execution = value["execution"]
    plan = {
        "schema": PLAN_SCHEMA,
        "name": value["name"],
        "image": IMAGE,
        "model": model,
        "execution": execution,
        "source_root": execution["models_mount"] + "/" + execution["source_alias"],
        "artifact_root": execution["models_mount"] + "/" + execution["artifact_path"],
        "scientific_work": {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        },
        "runtime_sha256": digest(_runtime()),
    }
    _validate(plan)
    return plan


def _validate(plan: dict) -> None:
    execution = _expected_execution()
    artifact = PurePosixPath(execution["artifact_path"])
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("name") != "chris-q38-modelstage-v2"
        or plan.get("image") != IMAGE
        or plan.get("execution") != execution
        or plan.get("source_root")
        != execution["models_mount"] + "/" + execution["source_alias"]
        or plan.get("artifact_root")
        != execution["models_mount"] + "/" + execution["artifact_path"]
        or artifact.parts[:1] != ("fleetjob-dev",)
        or len(artifact.parts) != 2
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("scientific_work")
        != {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        }
        or not isinstance(plan.get("model", {}).get("files"), list)
        or not plan["model"]["files"]
        or plan["model"].get("root") != plan.get("source_root")
    ):
        raise ValueError("SkyRL model artifact stage plan changed")


def request(plan: dict) -> dict:
    _validate(plan)
    files = _runtime()
    files.update(
        {
            "training/__init__.py": "",
            "cyber_post_train/__init__.py": "",
            "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
        }
    )
    run_dir = "/mnt/sfs/jobs/" + plan["name"]
    return bundled_request(
        {
            "name": plan["name"],
            "title": plan["name"] + " zero-GPU immutable model stage",
            "run_dir": run_dir,
            "image": IMAGE,
            # This request is only the sealed code transport.  The Kubernetes
            # Job below is the sole resource authority and requests no GPU.
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "4",
                "cpu_limit": "8",
                "memory_request": "8Gi",
                "memory_limit": "16Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), "--receipt", RECEIPT_PATH],
        transport_split_threshold=30000,
        transport_chunk_size=30000,
    )


def job_manifest(plan: dict) -> dict:
    _validate(plan)
    execution = plan["execution"]
    bundled = request(plan)
    run_dir = bundled["run_dir"]
    env = {**bundled["env"], "RUN_DIR": run_dir}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": plan["name"], "namespace": execution["namespace"]},
        "spec": {
            "activeDeadlineSeconds": execution["deadline_seconds"],
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "stage",
                            "image": IMAGE,
                            "command": ["/bin/sh", "-lc", "exec " + bundled["command"]],
                            "env": [
                                {"name": key, "value": item}
                                for key, item in sorted(env.items())
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "privileged": False,
                                "runAsGroup": execution["runtime_gid"],
                                "runAsNonRoot": True,
                                "runAsUser": execution["runtime_uid"],
                            },
                            "resources": {
                                "requests": {"cpu": "4", "memory": "8Gi"},
                                "limits": {"cpu": "8", "memory": "16Gi"},
                            },
                            "terminationMessagePath": RECEIPT_PATH,
                            "terminationMessagePolicy": "File",
                            "volumeMounts": [
                                {
                                    "name": "models",
                                    "mountPath": execution["models_mount"],
                                    "subPath": execution["models_subpath"],
                                },
                                {"name": "runtime", "mountPath": run_dir},
                            ],
                        }
                    ],
                    "priorityClassName": execution["priority"],
                    "restartPolicy": "Never",
                    "volumes": [
                        {
                            "name": "models",
                            "persistentVolumeClaim": {"claimName": execution["pvc"]},
                        },
                        {"name": "runtime", "emptyDir": {}},
                    ],
                },
            },
        },
    }


def packet(plan: dict) -> dict:
    manifest = job_manifest(plan)
    return _seal(
        {
            "schema": PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": plan["name"],
            "submitted": False,
        }
    )


def _validate_server_render(expected: dict, rendered: dict) -> None:
    expected = copy.deepcopy(expected)
    actual = copy.deepcopy(rendered)
    status = actual.pop("status", None)
    metadata = actual.get("metadata", {})
    timestamp = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    uid = metadata.pop("uid", None)
    labels = metadata.pop("labels", None)
    name = expected["metadata"]["name"]
    generated_labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    spec = actual.get("spec", {})
    defaults = {
        key: spec.pop(key, None)
        for key in (
            "completionMode",
            "completions",
            "manualSelector",
            "parallelism",
            "podReplacementPolicy",
            "selector",
            "suspend",
        )
    }
    template = spec.get("template", {})
    template_labels = template.get("metadata", {}).pop("labels", None)
    pod = template.get("spec", {})
    pod_defaults = {
        key: pod.pop(key, None)
        for key in (
            "dnsPolicy",
            "schedulerName",
            "securityContext",
            "terminationGracePeriodSeconds",
        )
    }
    containers = pod.get("containers", [])
    pull = containers[0].pop("imagePullPolicy", None) if len(containers) == 1 else None
    if (
        status != {}
        or not isinstance(timestamp, str)
        or not timestamp
        or generation != 1
        or not isinstance(uid, str)
        or not uid
        or labels != generated_labels
        or template_labels != generated_labels
        or defaults
        != {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
        or pod_defaults
        != {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "securityContext": {},
            "terminationGracePeriodSeconds": 30,
        }
        or pull != "IfNotPresent"
        or actual != expected
    ):
        raise JobsError("model artifact stage server dry-run changed the exact Job")


def validate_preview(plan: dict, manifest: dict, rendered: dict) -> dict:
    if manifest != job_manifest(plan):
        raise JobsError("model artifact stage Job differs from its immutable plan")
    _validate_server_render(manifest, rendered)
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest),
            "server_render_sha256": digest(rendered),
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": plan["name"],
            "gpus": 0,
            "runtime_user": {
                "uid": plan["execution"]["runtime_uid"],
                "gid": plan["execution"]["runtime_gid"],
            },
            "submitted": False,
        }
    )


def _relative_file(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != value:
        raise StageGateError("model_inventory_path_invalid")
    return Path(*path.parts)


def _verify_source(plan: dict) -> tuple[Path, list[tuple[dict, Path]]]:
    root = Path(plan["source_root"])
    if root.is_symlink():
        raise StageGateError("source_root_symlink")
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise StageGateError("source_root_missing") from exc
    files: list[tuple[dict, Path]] = []
    for index, item in enumerate(plan["model"]["files"]):
        relative = _relative_file(item["path"])
        candidate = resolved_root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
            metadata = resolved.stat()
        except FileNotFoundError as exc:
            raise StageGateError(f"source_file_missing_{index:02d}") from exc
        except (OSError, ValueError) as exc:
            raise StageGateError(f"source_file_invalid_{index:02d}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise StageGateError(f"source_file_not_regular_{index:02d}")
        if metadata.st_size != item["size"]:
            raise StageGateError(f"source_file_size_mismatch_{index:02d}")
        with resolved.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"].removeprefix("sha256:"):
            raise StageGateError(f"source_file_digest_mismatch_{index:02d}")
        files.append((item, resolved))
    return resolved_root, files


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise StageGateError("atomic_noreplace_unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    result = function(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise StageGateError("artifact_destination_exists")
        raise StageGateError("artifact_atomic_publish_failed")


def _write_receipt(path: Path, value: dict, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o444)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def stage(plan: dict) -> dict:
    _validate(plan)
    execution = plan["execution"]
    if (os.geteuid(), os.getegid()) != (
        execution["runtime_uid"],
        execution["runtime_gid"],
    ):
        raise StageGateError("runtime_identity_mismatch")
    source_root, files = _verify_source(plan)
    artifact = Path(plan["artifact_root"])
    project_root = artifact.parent
    if project_root != Path(execution["models_mount"]) / "fleetjob-dev":
        raise StageGateError("artifact_project_binding_mismatch")
    if artifact.exists() or artifact.is_symlink():
        raise StageGateError("artifact_destination_exists")
    if not project_root.is_dir() or not os.access(project_root, os.W_OK | os.X_OK):
        raise StageGateError("artifact_project_not_writable")
    temp = project_root / ("." + artifact.name + ".tmp-" + digest(plan)[:12])
    if temp.exists() or temp.is_symlink():
        raise StageGateError("artifact_temp_exists")
    temp.mkdir(mode=0o700)
    try:
        total = 0
        for index, (item, source) in enumerate(files):
            relative = _relative_file(item["path"])
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination, follow_symlinks=True)
            except OSError as exc:
                raise StageGateError(f"hardlink_failed_{index:02d}") from exc
            if not os.path.samefile(source, destination):
                raise StageGateError(f"hardlink_identity_mismatch_{index:02d}")
            total += item["size"]
        receipt = _seal(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "published",
                "plan_sha256": digest(plan),
                "model_binding_sha256": digest(plan["model"]),
                "source_alias": execution["source_alias"],
                "artifact_path": execution["artifact_path"],
                "files": len(files),
                "bytes": total,
                "hardlinks_verified": len(files),
                **plan["scientific_work"],
            }
        )
        _write_receipt(temp / ".CYBER_ARTIFACT.json", receipt, exclusive=True)
        _rename_noreplace(temp, artifact)
        if not artifact.is_dir() or artifact.is_symlink():
            raise StageGateError("artifact_publish_not_visible")
        return receipt
    except BaseException:
        if temp.exists() and temp.parent == project_root and temp.name.startswith("."):
            with suppress(Exception):
                shutil.rmtree(temp)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    phase = "plan_loading"
    plan = None
    try:
        plan = json.loads(args.plan.read_bytes())
        phase = "plan_digest"
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        if args.receipt != Path(RECEIPT_PATH):
            raise ValueError("receipt path binding mismatch")
        phase = "request_validation"
        request(plan)
        phase = "artifact_stage"
        receipt = stage(plan)
        _write_receipt(Path(RECEIPT_PATH), receipt, exclusive=False)
        print(json.dumps({"status": receipt["status"], "sha256": receipt["sha256"]}))
    except BaseException as exc:
        rejection = _seal(
            {
                "schema": REJECTION_SCHEMA,
                "status": "rejected",
                "phase": phase,
                "error_class": type(exc).__name__,
                "error_code": getattr(exc, "code", "unclassified"),
                "plan_sha256": args.sha256,
                "task_rows": 0,
                "rollout_episodes": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
            }
        )
        if args.receipt == Path(RECEIPT_PATH):
            with suppress(Exception):
                _write_receipt(Path(RECEIPT_PATH), rejection, exclusive=False)
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "phase": phase,
                    "error_class": type(exc).__name__,
                    "error_code": getattr(exc, "code", "unclassified"),
                    "sha256": rejection["sha256"],
                }
            )
        )
        # Deterministic, sanitized admission rejection is not a training
        # failure.  The cleanup observer and receipt still prove no artifact
        # was accepted; unexpected cluster/runtime failures remain nonzero.
        if isinstance(exc, StageGateError):
            return
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
