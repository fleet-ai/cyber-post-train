#!/usr/bin/env python3
"""Build and verify the one-shot read-only prod8 terminal-evidence probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest
from training import skyrl_reward_rayjob as direct

NAME = "chris-q38-prod8-terminal-probe-v1"
TARGET = "/mnt/sfs/jobs/chris-q38-rlreward-prod8"
TRAINING_PLAN_SHA256 = "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
    "sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RECEIPT_SCHEMA = "cyber_qwen38_prod8_terminal_probe_receipt_v1"
PREVIEW_SCHEMA = "cyber_qwen38_prod8_terminal_probe_preview_v1"

# This program runs in the exact immutable training image with the SFS mount
# read-only. It opens only the public, sanitized lifecycle markers written by
# the reviewed runtime. It never opens trainer logs, trajectories, prompts,
# tool calls, scores, rewards, metrics, tensors, optimizer state, or secrets.
RUNTIME = r"""import datetime,hashlib,json,os,pathlib,re,stat

ROOT=pathlib.Path(os.environ["PROBE_TARGET"])
PLAN=os.environ["PROBE_TRAINING_PLAN_SHA256"]
SCHEMA=os.environ["PROBE_RECEIPT_SCHEMA"]
RECEIPT=pathlib.Path(os.environ.get("PROBE_RECEIPT_PATH","/dev/termination-log"))
MAX_MARKER_BYTES=65536
MAX_RECEIPT_BYTES=3500
ROOT_MARKERS=(
    "STARTED.json","FAILED.json","NATIVE_FAILURE.json","NATIVE_REJECTED.json",
    "REJECTED.json","NATIVE_TRAINING_COMPLETE.json","ACCEPTED.json",
)
BATCH_MARKERS=("STARTED.json","COLLECTED.json","REJECTED.json","FAILED.json")
SAFE_WATCHDOGS=(None,"hard_runtime_bound","confirmed_no_progress_idle")
SAFE_NAMES=re.compile(r"^[A-Za-z_][A-Za-z0-9_.<>-]{0,127}$")
SAFE_FILES=re.compile(r"^[A-Za-z0-9_.-]+\.py$")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_info(path,require_digest=True):
    try:
        fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    except FileNotFoundError:
        return None
    except OSError:
        return {"accepted":False,"reason":"not_direct_regular_file"}
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_MARKER_BYTES:
            return {"accepted":False,"reason":"not_small_regular_file"}
        raw=b""
        while len(raw)<=MAX_MARKER_BYTES:
            chunk=os.read(fd,MAX_MARKER_BYTES+1-len(raw))
            if not chunk:
                break
            raw+=chunk
        after=os.fstat(fd)
    finally:
        os.close(fd)
    if len(raw)>MAX_MARKER_BYTES:
        return {"accepted":False,"reason":"not_small_regular_file"}
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
        after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns
    ):
        return {"accepted":False,"reason":"changed_during_read"}
    try:
        value=json.loads(raw)
    except (UnicodeDecodeError,ValueError):
        return {"accepted":False,"reason":"invalid_json"}
    if not isinstance(value,dict):
        return {"accepted":False,"reason":"not_object"}
    supplied=value.get("sha256")
    expected=digest({k:v for k,v in value.items() if k!="sha256"})
    if (require_digest and supplied not in (expected,"sha256:"+expected)) or (
        not require_digest
        and supplied is not None
        and supplied not in (expected,"sha256:"+expected)
    ):
        return {"accepted":False,"reason":"digest_mismatch"}
    return {"accepted":True,"value":value,"size_bytes":before.st_size,
            "mtime_ns":before.st_mtime_ns,"file_sha256":hashlib.sha256(raw).hexdigest()}

def stat_info(path):
    try:
        current=path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(current.st_mode) or path.is_symlink():
        return {"accepted":False,"reason":"not_direct_regular_file"}
    return {"accepted":True,"size_bytes":current.st_size,"mtime_ns":current.st_mtime_ns}

def safe_frame(value):
    if not isinstance(value,dict) or set(value)!={"file","line","function"}:
        return None
    if (not isinstance(value["file"],str) or not SAFE_FILES.fullmatch(value["file"])
        or type(value["line"]) is not int or value["line"]<1
        or not isinstance(value["function"],str) or not SAFE_NAMES.fullmatch(value["function"])):
        return None
    return {"file":value["file"],"line":value["line"],"function":value["function"]}

def safe_cause(value):
    expected={"error_class","actor_init_failed","remote_error_classes","remote_frames","local_frames"}
    if not isinstance(value,dict) or set(value)!=expected:
        return None
    if (not isinstance(value["error_class"],str) or not SAFE_NAMES.fullmatch(value["error_class"])
        or type(value["actor_init_failed"]) is not bool
        or not isinstance(value["remote_error_classes"],list)
        or len(value["remote_error_classes"])>32
        or any(not isinstance(item,str) or not SAFE_NAMES.fullmatch(item)
               for item in value["remote_error_classes"])):
        return None
    result={
        "error_class":value["error_class"],"actor_init_failed":value["actor_init_failed"],
        "remote_error_classes":value["remote_error_classes"][:4],
        "remote_error_classes_total":len(value["remote_error_classes"]),
    }
    for key in ("remote_frames","local_frames"):
        if not isinstance(value[key],list) or len(value[key])>20:
            return None
        frames=[safe_frame(item) for item in value[key]]
        if any(item is None for item in frames):
            return None
        result[key]=frames[:2]
        result[key+"_total"]=len(frames)
    return result

def sanitize_root(name, info):
    if info is None:
        return {"name":name,"present":False}
    base={"name":name,"present":True,"accepted":False}
    if not info.get("accepted"):
        return {**base,"reason":info["reason"]}
    if name=="ACCEPTED.json":
        return {
            **base,"accepted":True,"size_bytes":info["size_bytes"],
            "mtime_ns":info["mtime_ns"],"selected":{"present":True},
        }
    value=info["value"]
    common={**base,"size_bytes":info["size_bytes"],"mtime_ns":info["mtime_ns"],
            "file_sha256":info["file_sha256"]}
    selected=None
    if name=="STARTED.json" and set(value)=={"plan_sha256","started_at","sha256"}:
        if value.get("plan_sha256")==PLAN and type(value.get("started_at")) in (int,float):
            selected={"plan_sha256":value["plan_sha256"],"started_at":value["started_at"]}
    elif name=="FAILED.json" and set(value)=={
        "status","plan_sha256","error_class","watchdog_reason","sha256"
    }:
        if (value.get("status")=="failed" and value.get("plan_sha256")==PLAN
            and isinstance(value.get("error_class"),str)
            and SAFE_NAMES.fullmatch(value["error_class"])
            and value.get("watchdog_reason") in SAFE_WATCHDOGS):
            selected={key:value[key] for key in (
                "status","plan_sha256","error_class","watchdog_reason"
            )}
    elif name=="NATIVE_FAILURE.json" and set(value)=={"plan_sha256","causes","sha256"}:
        causes=value.get("causes")
        safe=(
            [safe_cause(item) for item in causes]
            if isinstance(causes,list) and len(causes)<=8 else []
        )
        if value.get("plan_sha256")==PLAN and isinstance(causes,list) and safe and all(safe):
            selected={
                "plan_sha256":value["plan_sha256"],"causes":safe[:2],
                "causes_total":len(safe),
            }
    elif name in ("NATIVE_REJECTED.json","REJECTED.json") and set(value)=={
        "schema","status","reason","plan_sha256","sha256"
    }:
        if (value.get("schema")=="cyber_rl_native_rejection_v1" and value.get("status")=="rejected"
            and value.get("plan_sha256")==PLAN and isinstance(value.get("reason"),str)
            and SAFE_NAMES.fullmatch(value["reason"])):
            selected={key:value[key] for key in ("schema","status","reason","plan_sha256")}
    elif name=="NATIVE_TRAINING_COMPLETE.json" and set(value)=={
        "status","plan_sha256","checkpoint_global_step","completed_batches","completed_at",
        "optimizer_update_independently_verified","checkpoint_reload_verified","sha256"
    }:
        if (value.get("status")=="native_loop_returned" and value.get("plan_sha256")==PLAN
            and type(value.get("checkpoint_global_step")) is int
            and type(value.get("completed_batches")) is int
            and type(value.get("optimizer_update_independently_verified")) is bool
            and type(value.get("checkpoint_reload_verified")) is bool):
            selected={key:value[key] for key in (
                "status","plan_sha256","checkpoint_global_step","completed_batches",
                "optimizer_update_independently_verified","checkpoint_reload_verified")}
    if selected is None:
        return {**common,"reason":"schema_or_binding_mismatch"}
    return {**common,"accepted":True,"selected":selected}

def sanitize_batch(path):
    info=file_info(path,require_digest=False)
    if info is None:
        return None
    base={"present":True,"accepted":False}
    if not info.get("accepted"):
        return {**base,"reason":info["reason"]}
    value=info["value"]
    name=path.name
    selected=None
    if name in ("STARTED.json","COLLECTED.json"):
        required={"schema","phase","global_step","optimizer_step_verified"}
        if required.issubset(value) and value.get("schema")=="cyber_skyrl_batch_v1":
            if (value.get("phase") in ("train","eval") and type(value.get("global_step")) is int
                and value["global_step"]>=0 and type(value.get("optimizer_step_verified")) is bool):
                selected={key:value[key] for key in (
                    "schema","phase","global_step","optimizer_step_verified"
                )}
    elif name=="REJECTED.json" and set(value) in ({"reason"},{"reason","sha256"}):
        if isinstance(value.get("reason"),str) and SAFE_NAMES.fullmatch(value["reason"]):
            selected={"reason":value["reason"]}
    elif name=="FAILED.json" and set(value) in (
        {"reason"},{"reason","sha256"},{"error_type"},{"error_type","sha256"}
    ):
        key="reason" if "reason" in value else "error_type"
        if isinstance(value.get(key),str) and SAFE_NAMES.fullmatch(value[key]):
            selected={key:value[key]}
    if selected is None:
        return {**base,"reason":"schema_mismatch"}
    return {**base,"accepted":True,"selected":selected,"size_bytes":info["size_bytes"],
            "mtime_ns":info["mtime_ns"]}

def batch_inventory():
    parent=ROOT/"episodes"/"batches"
    if not parent.exists():
        return {
            "directory_present":False,"batch_directories":0,
            "markers":{
                name:{"count":0,"accepted_count":0,"unaccepted_count":0,"latest":None}
                for name in BATCH_MARKERS
            },
        }
    if parent.is_symlink() or not parent.is_dir():
        return {"directory_present":True,"accepted":False,"reason":"not_direct_directory"}
    markers={
        name:{
            "count":0,"accepted_count":0,"unaccepted_count":0,"latest":None,
            "latest_mtime_ns":-1,
        }
        for name in BATCH_MARKERS
    }
    count=0
    for entry in sorted(parent.iterdir(),key=lambda item:item.name):
        if entry.is_symlink() or not entry.is_dir():
            continue
        count+=1
        for name in BATCH_MARKERS:
            row=sanitize_batch(entry/name)
            if row is not None:
                summary=markers[name]
                summary["count"]+=1
                accepted=row.get("accepted") is True
                summary["accepted_count" if accepted else "unaccepted_count"]+=1
                mtime=row.get("mtime_ns",-1)
                if accepted and mtime>=summary["latest_mtime_ns"]:
                    summary["latest_mtime_ns"]=mtime
                    summary["latest"]={"selected":row["selected"],"mtime_ns":mtime}
    for summary in markers.values():
        summary.pop("latest_mtime_ns")
    return {"directory_present":True,"accepted":True,"batch_directories":count,"markers":markers}

def checkpoint_inventory():
    parent=ROOT/"checkpoints"
    if not parent.exists():
        return {"directory_present":False,"global_steps":[],"file_count":0,"total_bytes":0,
                "latest_pointer_step":None}
    if parent.is_symlink() or not parent.is_dir():
        return {"directory_present":True,"accepted":False,"reason":"not_direct_directory"}
    steps=[]
    files=0
    total=0
    for child in sorted(parent.iterdir(),key=lambda item:item.name):
        if child.is_symlink():
            continue
        match=re.fullmatch(r"global_step_(\d+)",child.name)
        if match and child.is_dir():
            steps.append(int(match.group(1)))
            for base,dirs,names in os.walk(child,followlinks=False):
                dirs[:]=[name for name in dirs if not (pathlib.Path(base)/name).is_symlink()]
                for name in names:
                    path=pathlib.Path(base)/name
                    try:
                        current=path.lstat()
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(current.st_mode) and not path.is_symlink():
                        files+=1
                        total+=current.st_size
    latest=None
    pointer=parent/"latest_ckpt_global_step.txt"
    try:
        current=pointer.lstat()
        if stat.S_ISREG(current.st_mode) and not pointer.is_symlink() and current.st_size<=32:
            text=pointer.read_text().strip()
            if text.isdecimal():
                latest=int(text)
    except (FileNotFoundError,UnicodeDecodeError):
        pass
    return {"directory_present":True,"accepted":True,"global_steps":steps,"file_count":files,
            "total_bytes":total,"latest_pointer_step":latest}

root_exists=ROOT.exists()
root_direct=root_exists and ROOT.is_dir() and not ROOT.is_symlink()
root_markers=(
    [
        sanitize_root(
            name,
            stat_info(ROOT/name) if name=="ACCEPTED.json" else file_info(ROOT/name),
        )
        for name in ROOT_MARKERS
    ]
    if root_direct else []
)
present={row["name"]:row for row in root_markers if row.get("present")}
if "FAILED.json" in present:
    terminal="failed" if present["FAILED.json"].get("accepted") else "unaccepted_failed_marker"
elif "REJECTED.json" in present:
    terminal=(
        "rejected"
        if present["REJECTED.json"].get("accepted") else "unaccepted_rejected_marker"
    )
elif "NATIVE_TRAINING_COMPLETE.json" in present:
    terminal=(
        "native_training_complete"
        if present["NATIVE_TRAINING_COMPLETE.json"].get("accepted")
        else "unaccepted_complete_marker"
    )
else:
    terminal="no_terminal_marker"
body={
    "schema":SCHEMA,"status":"inspected","target":str(ROOT),"training_plan_sha256":PLAN,
    "checked_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpus":0,"sfs_mount_read_only":True,"private_payloads_read":False,
    "root_exists":root_exists,"root_direct":root_direct,"terminal_classification":terminal,
    "root_markers":root_markers,
    "batch_inventory":batch_inventory() if root_direct else {"directory_present":False},
    "checkpoint_inventory":checkpoint_inventory() if root_direct else {"directory_present":False},
    "receipt_size_limit_bytes":MAX_RECEIPT_BYTES,
}
body["receipt_size_bytes"]=0
while True:
    body.pop("sha256",None)
    body["sha256"]="sha256:"+digest(body)
    payload=json.dumps(body,sort_keys=True,separators=(",",":"))+"\n"
    size=len(payload.encode())
    if body["receipt_size_bytes"]==size:
        break
    body["receipt_size_bytes"]=size
if size>MAX_RECEIPT_BYTES:
    raise RuntimeError("sanitized receipt exceeds Kubernetes termination-message bound")
RECEIPT.write_text(payload)
print(json.dumps({"status":body["status"],"terminal_classification":terminal,
                  "sha256":body["sha256"]},sort_keys=True))
"""


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def manifest() -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": direct.NAMESPACE,
            "annotations": {FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF},
        },
        "spec": {
            "activeDeadlineSeconds": 300,
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "terminal-probe",
                            "image": IMAGE,
                            "command": ["python", "-c", RUNTIME],
                            "env": [
                                {"name": "PROBE_TARGET", "value": TARGET},
                                {"name": "PROBE_RECEIPT_SCHEMA", "value": RECEIPT_SCHEMA},
                                {
                                    "name": "PROBE_TRAINING_PLAN_SHA256",
                                    "value": TRAINING_PLAN_SHA256,
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "privileged": False,
                                "readOnlyRootFilesystem": True,
                                "runAsGroup": direct.RUNTIME_GID,
                                "runAsNonRoot": True,
                                "runAsUser": direct.RUNTIME_UID,
                            },
                            "terminationMessagePath": direct.STAGE_RECEIPT,
                            "terminationMessagePolicy": "File",
                            "volumeMounts": [
                                {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}
                            ],
                        }
                    ],
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {"supplementalGroups": [2000]},
                    "volumes": [
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {
                                "claimName": direct.PVC,
                                "readOnly": True,
                            },
                        }
                    ],
                },
            },
        },
    }


def preview_evidence(rendered: dict[str, Any], context: str) -> dict[str, Any]:
    expected = manifest()
    direct.validate_cpu_preview(expected, rendered, context=context, purpose="data_stage")
    pod = rendered["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if (
        rendered["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or pod.get("priorityClassName") != "c1"
        or rendered["spec"].get("activeDeadlineSeconds") != 300
        or rendered["spec"].get("backoffLimit") != 0
        or container.get("volumeMounts")
        != [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}]
        or pod.get("volumes")
        != [
            {
                "name": "sfs",
                "persistentVolumeClaim": {"claimName": direct.PVC, "readOnly": True},
            }
        ]
        or "nvidia.com/gpu" in json.dumps(rendered, sort_keys=True)
    ):
        raise JobsError("prod8 terminal probe preview changed")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "context": context,
            "name": NAME,
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "target": TARGET,
            "training_plan_sha256": TRAINING_PLAN_SHA256,
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "sfs_mount_read_only": True,
            "submitted": False,
        }
    )


def validate_receipt(value: dict[str, Any], *, target: str = TARGET) -> dict[str, Any]:
    if (
        value != _seal(value)
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") != "inspected"
        or value.get("target") != target
        or value.get("training_plan_sha256") != TRAINING_PLAN_SHA256
        or value.get("gpus") != 0
        or value.get("sfs_mount_read_only") is not True
        or value.get("private_payloads_read") is not False
        or value.get("receipt_size_limit_bytes") != 3500
        or type(value.get("receipt_size_bytes")) is not int
        or value["receipt_size_bytes"] > value["receipt_size_limit_bytes"]
        or value["receipt_size_bytes"]
        != len((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        or value.get("terminal_classification")
        not in {
            "failed",
            "rejected",
            "native_training_complete",
            "unaccepted_failed_marker",
            "unaccepted_rejected_marker",
            "unaccepted_complete_marker",
            "no_terminal_marker",
        }
        or not isinstance(value.get("root_markers"), list)
        or not isinstance(value.get("batch_inventory"), dict)
        or not isinstance(value.get("checkpoint_inventory"), dict)
    ):
        raise JobsError("prod8 terminal probe receipt was not accepted")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--validate-preview", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--validate-receipt", type=Path)
    args = parser.parse_args()
    selected = sum(
        (args.manifest, args.validate_preview is not None, args.validate_receipt is not None)
    )
    if selected != 1:
        parser.error("select exactly one operation")
    if args.manifest:
        value = manifest()
    elif args.validate_preview is not None:
        if args.context not in {direct.DEV_CONTEXT, direct.PROD_CONTEXT}:
            parser.error("preview context is required")
        value = preview_evidence(json.loads(args.validate_preview.read_bytes()), args.context)
    else:
        value = validate_receipt(json.loads(args.validate_receipt.read_bytes()))
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
