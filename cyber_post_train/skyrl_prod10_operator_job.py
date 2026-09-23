"""Immutable package and zero-GPU Job for the bounded prod10 coordinator."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import dev_cleanup_observer as cleanup
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_training as training
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

from .jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest

NAMESPACE = "fleet-train-jobs"
SOURCE_ANNOTATION = "cyber-post-train.fleet.ai/operator-source-sha256"
PACKET_ANNOTATION = "cyber-post-train.fleet.ai/operator-packet-sha256"
PHASE_ANNOTATION = "cyber-post-train.fleet.ai/operator-phase"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
QUEUE = "training-lq"
QUEUE_PRIORITY = "q1"
IMAGE = training.historical.IMAGE
PVC = "sfs-shared"
CONTROLS_SUBPATH = "jobs/chris-q38-study-corpora-v1/launch-controls"
CONTROLS_PATH = "/mnt/sfs/" + CONTROLS_SUBPATH
LAUNCH_ACTIVE_DEADLINE_SECONDS = direct.MAXIMUM_SECONDS + 3600
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIRS = ("training", "cyber_post_train")
_SOURCE_FILES = (
    "scripts/probe_qwen38_prod8_terminal.py",
    "evals/fleet/opencode_self_hosted.py",
)
_BOOTSTRAP = r"""import gzip,hashlib,json,os,sys,tarfile
from pathlib import Path,PurePosixPath
def failure(error):
    body={"schema":"cyber_skyrl_prod10_bootstrap_failure_v1","status":"failed","phase":os.environ.get("OPERATOR_PHASE",""),"error_class":type(error).__name__,"error_code":"bootstrap_failure","gpus":0}
    body["sha256"]="sha256:"+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    encoded=(json.dumps(body,sort_keys=True,separators=(",",":"))+"\n").encode()
    descriptor=os.open("/dev/termination-log",os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,"wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
def bootstrap():
    archive=Path("/bundle/source.tgz")
    packet_gz=Path("/packet/packet.json.gz")
    source=archive.read_bytes()
    packet=packet_gz.read_bytes()
    if "sha256:"+hashlib.sha256(source).hexdigest()!=os.environ["OPERATOR_SOURCE_SHA256"]:
        raise SystemExit("operator source digest mismatch")
    if "sha256:"+hashlib.sha256(gzip.decompress(packet)).hexdigest()!=os.environ["OPERATOR_PACKET_FILE_SHA256"]:
        raise SystemExit("operator packet digest mismatch")
    root=Path("/runtime")
    with tarfile.open(fileobj=__import__("io").BytesIO(source),mode="r:gz") as bundle:
        members=bundle.getmembers()
        if not members or len(members)>512:
            raise SystemExit("operator source inventory invalid")
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise SystemExit("operator source member invalid")
            payload=bundle.extractfile(member)
            if payload is None:
                raise SystemExit("operator source member unreadable")
            target=root.joinpath(*path.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
            with os.fdopen(descriptor,"wb") as stream:
                stream.write(payload.read())
    packet_path=Path("/work/packet.json")
    descriptor=os.open(packet_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
    with os.fdopen(descriptor,"wb") as stream:
        stream.write(gzip.decompress(packet))
    os.environ["PYTHONPATH"]="/runtime"
    os.chdir("/runtime")
    os.execv(sys.executable,[sys.executable,"-u","-m","training.skyrl_prod10_operator","--packet",str(packet_path),"--phase",os.environ["OPERATOR_PHASE"]])
try:
    bootstrap()
except BaseException as error:
    failure(error)
    raise
"""


@dataclass(frozen=True)
class OperatorPackage:
    source_config_map: dict[str, Any]
    packet_config_map: dict[str, Any]
    job: dict[str, Any]
    packet: dict[str, Any]
    source_archive: bytes


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _tracked_sources() -> dict[str, bytes]:
    paths: set[Path] = set()
    for directory in _SOURCE_DIRS:
        paths.update((_ROOT / directory).glob("*.py"))
    paths.update(_ROOT / value for value in _SOURCE_FILES)
    result: dict[str, bytes] = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError("prod10 operator source contains an indirect file")
        relative = path.relative_to(_ROOT).as_posix()
        result[relative] = path.read_bytes()
    required = {
        "training/skyrl_prod10_operator.py",
        "training/incluster_kubernetes.py",
        "training/skyrl_prod9_direct.py",
        "training/dev_cleanup_observer.py",
        "cyber_post_train/jobs.py",
    }
    if not required.issubset(result):
        raise ValueError("prod10 operator source closure is incomplete")
    return result


def source_archive() -> tuple[bytes, str]:
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, payload in _tracked_sources().items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o444
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.mtime = 0
                archive.addfile(member, io.BytesIO(payload))
    value = stream.getvalue()
    return value, "sha256:" + hashlib.sha256(value).hexdigest()


def stage_packet(
    *,
    identity: historical.RailIdentity,
    stage: dict[str, Any],
) -> dict[str, Any]:
    checked, bound = training._stage_identity(stage)
    if bound != identity:
        raise ValueError("prod10 stage operator identity changed")
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "stage",
            "operator_name": operator.OPERATOR_NAMES["stage"],
            "identity": identity.sealed_mapping(),
            "stage": checked,
            "precreate_recovery": operator.stage_recovery_binding(),
        }
    )


def manifest_packet(
    *,
    identity: historical.RailIdentity,
    stage: dict[str, Any],
    stage_launch_result: dict[str, Any],
) -> dict[str, Any]:
    checked_stage, stage_identity = training._stage_identity(stage)
    if stage_identity != identity:
        raise ValueError("prod10 manifest stage identity changed")
    checked_launch = direct._direct_stage_launch(
        stage_launch_result,
        checked_stage,
        identity=identity,
        operator_name=operator.OPERATOR_NAMES["stage"],
        fresh=False,
    )
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "manifest",
            "operator_name": operator.OPERATOR_NAMES["manifest"],
            "identity": identity.sealed_mapping(),
            "stage": checked_stage,
            "stage_launch_result": checked_launch,
            "preflight_v1_failure": operator.preflight_v1_failure_binding(),
        }
    )


def preflight_packet(
    *,
    identity: historical.RailIdentity,
    plan: dict[str, Any],
    request: dict[str, Any],
    stage: dict[str, Any],
    stage_launch_result: dict[str, Any],
    manifest_launch_result: dict[str, Any],
    dev_preview: dict[str, Any],
    dev_duplicate_proof: dict[str, Any],
) -> dict[str, Any]:
    direct._identity(plan, identity)
    if training.job_request(plan) != request:
        raise ValueError("prod10 preflight operator request changed")
    checked_stage, stage_identity = training._stage_identity(stage)
    if stage_identity != identity:
        raise ValueError("prod10 preflight stage identity changed")
    checked_stage_launch = direct._direct_stage_launch(
        stage_launch_result,
        checked_stage,
        identity=identity,
        operator_name=operator.OPERATOR_NAMES["stage"],
        fresh=False,
    )
    checked_manifest_launch = direct._validate_seal(
        manifest_launch_result, direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA
    )
    manifest_package = checked_manifest_launch.get("package")
    manifest_release = direct._validate_seal(
        checked_manifest_launch.get("observer"), cleanup.DIRECT_RESULT_SCHEMA
    )
    manifest_receipt = direct._validate_seal(
        manifest_release.get("receipt"), direct.DIRECT_MANIFEST_RESULT_SCHEMA
    )
    if (
        checked_manifest_launch.get("status") != "operator_succeeded_and_released"
        or checked_manifest_launch.get("gpus") != 0
        or not isinstance(manifest_package, dict)
        or manifest_package.get("name") != operator.OPERATOR_NAMES["manifest"]
        or manifest_package.get("phase") != "manifest"
        or manifest_package.get("failure_alerts") != "off"
        or manifest_package.get("priority") != "c1"
        or manifest_package.get("queue_priority") != "q1"
        or manifest_package.get("gpus") != 0
        or manifest_release.get("status") != "released"
        or manifest_release.get("terminal_status") != "Succeeded"
        or manifest_receipt.get("status") != "passed"
        or manifest_receipt.get("successor_manifest") != plan.get("data")
        or manifest_receipt.get("successor_manifest_sha256") != plan.get("data", {}).get("sha256")
        or manifest_receipt.get("private_rows_exported") is not False
        or manifest_receipt.get("nested_jobs_created") != 0
        or manifest_receipt.get("gpus") != 0
    ):
        raise ValueError("prod10 manifest handoff changed")
    direct._fresh_at(
        manifest_release.get("release_observed_at"),
        maximum_age=direct.DIRECT_STAGE_RELEASE_MAX_AGE_SECONDS,
    )
    expected = direct.preflight_job_manifest(plan, identity=identity)
    direct.validate_cpu_preview_proof(
        expected,
        dev_preview,
        purpose="preflight",
        context=direct.DEV_CONTEXT,
        fresh=True,
    )
    duplicate = direct._validate_seal(dev_duplicate_proof, direct.CPU_DUPLICATE_PROOF_SCHEMA)
    if duplicate.get("name") != identity.preflight_name:
        raise ValueError("prod10 preflight duplicate proof changed")
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "preflight",
            "operator_name": operator.OPERATOR_NAMES["preflight"],
            "identity": identity.sealed_mapping(),
            "plan": plan,
            "request": request,
            "stage": checked_stage,
            "stage_launch_result": checked_stage_launch,
            "manifest_launch_result": checked_manifest_launch,
            "manifest_sha256": "sha256:" + digest(expected),
            "dev_preview": dev_preview,
            "dev_duplicate_proof": duplicate,
        }
    )


def launch_packet(
    *,
    identity: historical.RailIdentity,
    plan: dict[str, Any],
    request: dict[str, Any],
    preflight_launch_result: dict[str, Any],
    source_preview: dict[str, Any],
    manifest_sha256: str,
    dev_preview: dict[str, Any],
    duplicate_proof: dict[str, Any],
    capacity_census: dict[str, Any],
) -> dict[str, Any]:
    direct._identity(plan, identity)
    if training.job_request(plan) != request:
        raise ValueError("prod10 launch request changed")
    checked_launch = launch_direct._preflight_launch(
        preflight_launch_result,
        plan,
        identity=identity,
        operator_name=operator.OPERATOR_NAMES["preflight"],
    )
    direct._source(source_preview)
    direct._validate_seal(dev_preview, direct.PREVIEW_SCHEMA)
    duplicate = launch_direct._duplicate(duplicate_proof, identity)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("prod10 launch manifest digest changed")
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "launch",
            "operator_name": operator.OPERATOR_NAMES["launch"],
            "identity": identity.sealed_mapping(),
            "plan": plan,
            "request": request,
            "preflight_launch_result": checked_launch,
            "source_preview": source_preview,
            "manifest_sha256": manifest_sha256,
            "dev_preview": dev_preview,
            "duplicate_proof": duplicate,
            "capacity_census": capacity_census,
            "launch_v1_failure": operator.launch_v1_failure_binding(),
            "launch_v2_failure": operator.launch_v2_failure_binding(),
            "probe_v6_success": operator.probe_v6_success_binding(),
            "probe_v7_failure": operator.probe_v7_failure_binding(),
            "probe_v8_failure": operator.probe_v8_failure_binding(),
            "probe_v9_success": operator.probe_v9_success_binding(),
        }
    )


def inspect_packet(
    *,
    identity: historical.RailIdentity,
    plan: dict[str, Any],
    preflight_launch_result: dict[str, Any],
) -> dict[str, Any]:
    direct._identity(plan, identity)
    checked_launch = launch_direct._preflight_launch(
        preflight_launch_result,
        plan,
        identity=identity,
        operator_name=operator.OPERATOR_NAMES["preflight"],
    )
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "inspect",
            "operator_name": operator.OPERATOR_NAMES["inspect"],
            "identity": identity.sealed_mapping(),
            "plan": plan,
            "preflight_launch_result": checked_launch,
            "launch_v3_failure": operator.launch_v3_failure_binding(),
        }
    )


def probe_packet(
    *,
    launch_packet: dict[str, Any],
) -> dict[str, Any]:
    # The probe binds the exact, already-rendered launch-v2 predecessor after
    # its time-limited duplicate proof has expired.  Validate its immutable
    # packet contract without pretending the historical proof is fresh.
    checked_launch = operator._launch_v2_packet(launch_packet)
    failure = operator.launch_v2_failure_binding()
    if checked_launch.get("sha256") != failure["launch_packet_sha256"]:
        raise ValueError("prod10 launch probe packet predecessor changed")
    if checked_launch.get("phase") != "launch":
        raise ValueError("prod10 launch probe source packet changed")
    return _seal(
        {
            "schema": operator.PACKET_SCHEMA,
            "phase": "probe",
            "operator_name": operator.OPERATOR_NAMES["probe"],
            "identity": checked_launch["identity"],
            "launch_packet": checked_launch,
            "launch_v2_failure": failure,
            "inspect_v3_success": operator.inspect_v3_success_binding(),
            "probe_v7_failure": operator.probe_v7_failure_binding(),
            "probe_v8_failure": operator.probe_v8_failure_binding(),
            "writable_controls_probe": True,
            "expected_diagnosis": "before_guard_passed",
        }
    )


def _validate_packet_semantics(packet: dict[str, Any]) -> dict[str, Any]:
    checked = operator._packet(packet, packet.get("phase", ""))
    identity = historical.identity_from_mapping(checked["identity"])
    if checked["phase"] == "stage":
        expected = stage_packet(
            identity=identity,
            stage=checked["stage"],
        )
    elif checked["phase"] == "manifest":
        expected = manifest_packet(
            identity=identity,
            stage=checked["stage"],
            stage_launch_result=checked["stage_launch_result"],
        )
    elif checked["phase"] == "preflight":
        expected = preflight_packet(
            identity=identity,
            plan=checked["plan"],
            request=checked["request"],
            stage=checked["stage"],
            stage_launch_result=checked["stage_launch_result"],
            manifest_launch_result=checked["manifest_launch_result"],
            dev_preview=checked["dev_preview"],
            dev_duplicate_proof=checked["dev_duplicate_proof"],
        )
    elif checked["phase"] == "inspect":
        expected = inspect_packet(
            identity=identity,
            plan=checked["plan"],
            preflight_launch_result=checked["preflight_launch_result"],
        )
    elif checked["phase"] == "probe":
        expected = probe_packet(
            launch_packet=checked["launch_packet"],
        )
    else:
        expected = launch_packet(
            identity=identity,
            plan=checked["plan"],
            request=checked["request"],
            preflight_launch_result=checked["preflight_launch_result"],
            source_preview=checked["source_preview"],
            manifest_sha256=checked["manifest_sha256"],
            dev_preview=checked["dev_preview"],
            duplicate_proof=checked["duplicate_proof"],
            capacity_census=checked["capacity_census"],
        )
    if checked != expected:
        raise ValueError("prod10 operator packet differs from current exact renderer")
    return checked


def _config_maps(
    *, phase: str, source: bytes, source_sha256: str, packet: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    name = operator.OPERATOR_NAMES[phase]
    packet_bytes = (json.dumps(packet, sort_keys=True, separators=(",", ":")) + "\n").encode()
    packet_gz = gzip.compress(packet_bytes, mtime=0)
    annotations = {
        SOURCE_ANNOTATION: source_sha256,
        PACKET_ANNOTATION: packet["sha256"],
        PHASE_ANNOTATION: phase,
    }
    source_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name + "-source",
            "namespace": NAMESPACE,
            "annotations": deepcopy(annotations),
        },
        "immutable": True,
        "binaryData": {"source.tgz": base64.b64encode(source).decode()},
    }
    packet_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name + "-packet",
            "namespace": NAMESPACE,
            "annotations": deepcopy(annotations),
        },
        "immutable": True,
        "binaryData": {"packet.json.gz": base64.b64encode(packet_gz).decode()},
    }
    return source_map, packet_map, packet_bytes


def _job(
    *, phase: str, source_sha256: str, packet: dict[str, Any], packet_bytes: bytes
) -> dict[str, Any]:
    name = operator.OPERATOR_NAMES[phase]
    annotations = {
        FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF,
        SOURCE_ANNOTATION: source_sha256,
        PACKET_ANNOTATION: packet["sha256"],
        PHASE_ANNOTATION: phase,
    }
    labels = {
        "cyber-post-train.fleet.ai/role": "prod10-bounded-operator",
        QUEUE_LABEL: QUEUE,
        QUEUE_PRIORITY_LABEL: QUEUE_PRIORITY,
    }
    environment = [
        {"name": "OPERATOR_PHASE", "value": phase},
        {"name": "OPERATOR_PACKET_SHA256", "value": packet["sha256"]},
        {
            "name": "OPERATOR_PACKET_FILE_SHA256",
            "value": "sha256:" + hashlib.sha256(packet_bytes).hexdigest(),
        },
        {"name": "OPERATOR_SOURCE_SHA256", "value": source_sha256},
        {"name": "OPERATOR_JOB_NAME", "value": name},
        {"name": "OPERATOR_POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        {"name": "OPERATOR_POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "WANDB_MODE", "value": "disabled"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
    ]
    if phase in {"launch", "probe"}:
        environment.append(
            {"name": "HF_DATASETS_CACHE", "value": "/work/hf-datasets"}
        )
    if phase == "probe":
        environment.extend(
            [
                {"name": "WANDB_API_KEY", "value": "diagnostic-not-a-credential"},
            ]
        )
    sfs_mount: dict[str, Any] = {"name": "sfs", "mountPath": "/mnt/sfs"}
    sfs_claim: dict[str, Any] = {"claimName": PVC}
    if phase != "stage":
        sfs_mount["readOnly"] = True
        sfs_claim["readOnly"] = True
    container = {
        "name": "operator",
        "image": IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python", "-u", "-c", _BOOTSTRAP],
        "env": environment,
        "resources": {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "4", "memory": "16Gi"},
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "privileged": False,
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 100,
        },
        "terminationMessagePath": "/dev/termination-log",
        "terminationMessagePolicy": "File",
        "volumeMounts": [
            {"name": "source", "mountPath": "/bundle", "readOnly": True},
            {"name": "packet", "mountPath": "/packet", "readOnly": True},
            {"name": "runtime", "mountPath": "/runtime"},
            {"name": "work", "mountPath": "/work"},
            sfs_mount,
            *(
                []
                if phase == "inspect"
                or (phase == "probe" and packet.get("writable_controls_probe") is not True)
                else [
                    {
                        "name": "controls-rw",
                        "mountPath": CONTROLS_PATH,
                        "subPath": CONTROLS_SUBPATH,
                    }
                ]
            ),
        ],
    }
    if phase == "launch":
        container["envFrom"] = [
            {"secretRef": {"name": "fleet-api"}},
            {"secretRef": {"name": "wandb-api"}},
        ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": deepcopy(annotations),
            "labels": deepcopy(labels),
        },
        "spec": {
            "activeDeadlineSeconds": (
                LAUNCH_ACTIVE_DEADLINE_SECONDS if phase == "launch" else 2400
            ),
            "backoffLimit": 0,
            "suspend": True,
            "template": {
                "metadata": {
                    "annotations": deepcopy(annotations),
                    "labels": deepcopy(labels),
                },
                "spec": {
                    "automountServiceAccountToken": True,
                    "serviceAccountName": "default",
                    "containers": [container],
                    "hostIPC": False,
                    "hostNetwork": False,
                    "hostPID": False,
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "supplementalGroups": [2000],
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "terminationGracePeriodSeconds": 60,
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {
                            "name": "source",
                            "configMap": {"name": name + "-source", "defaultMode": 292},
                        },
                        {
                            "name": "packet",
                            "configMap": {"name": name + "-packet", "defaultMode": 292},
                        },
                        {"name": "runtime", "emptyDir": {"sizeLimit": "256Mi"}},
                        {"name": "work", "emptyDir": {"sizeLimit": "64Mi"}},
                        {"name": "sfs", "persistentVolumeClaim": sfs_claim},
                        *(
                            []
                            if phase == "inspect"
                            or (
                                phase == "probe"
                                and packet.get("writable_controls_probe") is not True
                            )
                            else [
                                {
                                    "name": "controls-rw",
                                    "persistentVolumeClaim": {"claimName": PVC},
                                }
                            ]
                        ),
                    ],
                },
            },
        },
    }


def build_operator_package(packet: dict[str, Any]) -> OperatorPackage:
    checked = _validate_packet_semantics(packet)
    source, source_sha256 = source_archive()
    source_map, packet_map, packet_bytes = _config_maps(
        phase=checked["phase"], source=source, source_sha256=source_sha256, packet=checked
    )
    package = OperatorPackage(
        source_config_map=source_map,
        packet_config_map=packet_map,
        job=_job(
            phase=checked["phase"],
            source_sha256=source_sha256,
            packet=checked,
            packet_bytes=packet_bytes,
        ),
        packet=deepcopy(checked),
        source_archive=source,
    )
    validate_operator_package(package)
    return package


def validate_operator_package(package: OperatorPackage) -> dict[str, Any]:
    if not isinstance(package, OperatorPackage):
        raise ValueError("prod10 operator requires one immutable package")
    packet = _validate_packet_semantics(package.packet)
    source, source_sha256 = source_archive()
    source_map, packet_map, packet_bytes = _config_maps(
        phase=packet["phase"], source=source, source_sha256=source_sha256, packet=packet
    )
    expected_job = _job(
        phase=packet["phase"],
        source_sha256=source_sha256,
        packet=packet,
        packet_bytes=packet_bytes,
    )
    if (
        package.source_archive != source
        or package.source_config_map != source_map
        or package.packet_config_map != packet_map
        or package.job != expected_job
    ):
        raise ValueError("prod10 operator package differs from tracked exact bytes")
    metadata = expected_job["metadata"]
    pod = expected_job["spec"]["template"]["spec"]
    [container] = pod["containers"]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    volumes = {item["name"]: item for item in pod["volumes"]}
    expected_sfs_mount: dict[str, Any] = {"name": "sfs", "mountPath": "/mnt/sfs"}
    expected_sfs_claim: dict[str, Any] = {"claimName": PVC}
    if packet["phase"] != "stage":
        expected_sfs_mount["readOnly"] = True
        expected_sfs_claim["readOnly"] = True
    if (
        metadata["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or metadata["labels"].get(QUEUE_LABEL) != QUEUE
        or metadata["labels"].get(QUEUE_PRIORITY_LABEL) != QUEUE_PRIORITY
        or pod.get("priorityClassName") != "c1"
        or expected_job["spec"].get("backoffLimit") != 0
        or expected_job["spec"].get("activeDeadlineSeconds")
        != (LAUNCH_ACTIVE_DEADLINE_SECONDS if packet["phase"] == "launch" else 2400)
        or expected_job["spec"].get("suspend") is not True
        or pod.get("serviceAccountName") != "default"
        or pod.get("automountServiceAccountToken") is not True
        or "nvidia.com/gpu" in json.dumps(expected_job, sort_keys=True)
        or container["securityContext"].get("runAsUser") != 1000
        or container["securityContext"].get("runAsGroup") != 100
        or container.get("resources", {}).get("requests") != {"cpu": "2", "memory": "8Gi"}
        or mounts.get("sfs") != expected_sfs_mount
        or volumes.get("sfs") != {"name": "sfs", "persistentVolumeClaim": expected_sfs_claim}
        or (packet["phase"] == "inspect" and "controls-rw" in mounts)
        or (packet["phase"] == "inspect" and "controls-rw" in volumes)
        or (
            packet["phase"] == "probe"
            and mounts.get("controls-rw")
            != {
                "name": "controls-rw",
                "mountPath": CONTROLS_PATH,
                "subPath": CONTROLS_SUBPATH,
            }
        )
        or (
            packet["phase"] == "probe"
            and volumes.get("controls-rw")
            != {"name": "controls-rw", "persistentVolumeClaim": {"claimName": PVC}}
        )
        or container.get("envFrom")
        != (
            [
                {"secretRef": {"name": "fleet-api"}},
                {"secretRef": {"name": "wandb-api"}},
            ]
            if packet["phase"] == "launch"
            else None
        )
    ):
        raise ValueError("prod10 operator execution policy changed")
    return {
        "phase": packet["phase"],
        "name": metadata["name"],
        "packet_sha256": packet["sha256"],
        "source_sha256": source_sha256,
        "source_bytes": len(source),
        "source_config_map_sha256": "sha256:" + digest(source_map),
        "packet_config_map_sha256": "sha256:" + digest(packet_map),
        "job_manifest_sha256": "sha256:" + digest(expected_job),
        "failure_alerts": "off",
        "priority": "c1",
        "queue_priority": "q1",
        "gpus": 0,
    }


def validate_server_response(
    actual: dict[str, Any], expected: dict[str, Any], *, require_uid: bool
) -> dict[str, Any]:
    """Validate reviewed fields after API defaulting without accepting mutation."""
    if (
        not isinstance(actual, dict)
        or actual.get("apiVersion") != expected.get("apiVersion")
        or actual.get("kind") != expected.get("kind")
    ):
        raise JobsError("prod10 operator server response kind changed")
    actual_meta = actual.get("metadata", {})
    expected_meta = expected.get("metadata", {})
    if (
        actual_meta.get("name") != expected_meta.get("name")
        or actual_meta.get("namespace") != NAMESPACE
        or actual_meta.get("annotations") != expected_meta.get("annotations")
        or any(
            actual_meta.get("labels", {}).get(key) != value
            for key, value in expected_meta.get("labels", {}).items()
        )
    ):
        raise JobsError("prod10 operator server response metadata changed")
    if require_uid and re.fullmatch(r"[0-9a-f-]{36}", str(actual_meta.get("uid", ""))) is None:
        raise JobsError("prod10 operator create response omitted its UID")
    actual_spec = actual.get("spec", {})
    expected_spec = expected.get("spec", {})
    for key in ("activeDeadlineSeconds", "backoffLimit", "suspend"):
        if actual_spec.get(key) != expected_spec.get(key):
            raise JobsError("prod10 operator server response Job policy changed")
    actual_template = actual_spec.get("template", {})
    expected_template = expected_spec.get("template", {})
    actual_pod = deepcopy(actual_template.get("spec", {}))
    expected_pod = expected_template.get("spec", {})
    for key in ("hostIPC", "hostNetwork", "hostPID"):
        if expected_pod.get(key) is False and key not in actual_pod:
            actual_pod[key] = False
    actual_containers = actual_pod.get("containers")
    expected_containers = expected_pod.get("containers")
    if (
        isinstance(actual_containers, list)
        and isinstance(expected_containers, list)
        and len(actual_containers) == len(expected_containers) == 1
    ):
        actual_env = actual_containers[0].get("env")
        expected_env = expected_containers[0].get("env")
        if (
            isinstance(actual_env, list)
            and isinstance(expected_env, list)
            and len(actual_env) == len(expected_env)
        ):
            for actual_entry, expected_entry in zip(actual_env, expected_env, strict=True):
                if expected_entry.get("value") == "" and "value" not in actual_entry:
                    actual_entry["value"] = ""
                actual_field = actual_entry.get("valueFrom", {}).get("fieldRef")
                expected_field = expected_entry.get("valueFrom", {}).get("fieldRef")
                if (
                    isinstance(actual_field, dict)
                    and isinstance(expected_field, dict)
                    and "apiVersion" not in expected_field
                    and actual_field.get("apiVersion") == "v1"
                ):
                    actual_field.pop("apiVersion")
    if actual_template.get("metadata", {}).get("annotations") != expected_template.get(
        "metadata", {}
    ).get("annotations"):
        raise JobsError("prod10 operator Pod annotations changed")
    for key in (
        "automountServiceAccountToken",
        "serviceAccountName",
        "containers",
        "hostIPC",
        "hostNetwork",
        "hostPID",
        "nodeSelector",
        "priorityClassName",
        "restartPolicy",
        "securityContext",
        "terminationGracePeriodSeconds",
        "tolerations",
        "volumes",
    ):
        if actual_pod.get(key) != expected_pod.get(key):
            raise JobsError(f"prod10 operator server response Pod {key} changed")
    if len(actual_pod.get("containers", [])) != 1 or actual_pod.get("initContainers") not in (
        None,
        [],
    ):
        raise JobsError("prod10 operator server response gained a container")
    return {
        "name": actual_meta["name"],
        "uid": actual_meta.get("uid"),
        "manifest_sha256": "sha256:" + digest(expected),
        "server_sha256": "sha256:" + digest(actual),
        "failure_alerts": actual_meta["annotations"][FAILURE_ALERT_ANNOTATION],
        "priority": actual_pod["priorityClassName"],
        "queue_priority": actual_meta["labels"][QUEUE_PRIORITY_LABEL],
        "gpus": 0,
    }
