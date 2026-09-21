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
# read-only. It opens only the public, sanitized root lifecycle markers written
# by the reviewed runtime. Batch markers and checkpoint payloads are inventoried
# with no-follow metadata only. It never opens trainer logs, trajectories,
# prompts, tool calls, scores, rewards, metrics, tensors, optimizer state, or
# secrets.
RUNTIME = r"""import datetime,hashlib,json,os,pathlib,re,stat

ROOT=pathlib.Path(os.environ["PROBE_TARGET"])
PLAN=os.environ["PROBE_TRAINING_PLAN_SHA256"]
SCHEMA=os.environ["PROBE_RECEIPT_SCHEMA"]
RECEIPT=pathlib.Path(os.environ.get("PROBE_RECEIPT_PATH","/dev/termination-log"))
MAX_MARKER_BYTES=131072
MAX_RECEIPT_BYTES=3500
ROOT_MARKERS=(
    "STARTED.json","FAILED.json","NATIVE_FAILURE.json","NATIVE_REJECTED.json",
    "REJECTED.json","NATIVE_TRAINING_COMPLETE.json","ACCEPTED.json",
)
BATCH_MARKERS=("STARTED.json","COLLECTED.json","REJECTED.json","FAILED.json")
SAFE_WATCHDOGS=(None,"hard_runtime_bound","confirmed_no_progress_idle")
SAFE_NAMES=re.compile(r"^[A-Za-z_][A-Za-z0-9_.<>-]{0,63}$")
SAFE_FILES=re.compile(r"^[A-Za-z0-9_.-]{1,60}\.py$")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_info_at(parent_fd,name,require_digest=True):
    try:
        fd=os.open(
            name,
            os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0),
            dir_fd=parent_fd,
        )
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
    except (UnicodeDecodeError,ValueError,RecursionError):
        return {"accepted":False,"reason":"invalid_json"}
    if not isinstance(value,dict):
        return {"accepted":False,"reason":"not_object"}
    supplied=value.get("sha256")
    try:
        expected=digest({k:v for k,v in value.items() if k!="sha256"})
    except (TypeError,ValueError,RecursionError):
        return {"accepted":False,"reason":"invalid_digest_input"}
    if (require_digest and supplied not in (expected,"sha256:"+expected)) or (
        not require_digest
        and supplied is not None
        and supplied not in (expected,"sha256:"+expected)
    ):
        return {"accepted":False,"reason":"digest_mismatch"}
    return {"accepted":True,"value":value,"size_bytes":before.st_size,
            "mtime_ns":before.st_mtime_ns,"file_sha256":hashlib.sha256(raw).hexdigest()}

def stat_info_at(parent_fd,name):
    try:
        current=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        return {"accepted":False,"reason":"metadata_unavailable"}
    if not stat.S_ISREG(current.st_mode):
        return {"accepted":False,"reason":"not_direct_regular_file"}
    return {"accepted":True,"size_bytes":current.st_size,"mtime_ns":current.st_mtime_ns}

def read_small_direct_at(parent_fd,name,limit):
    try:
        fd=os.open(
            name,
            os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0),
            dir_fd=parent_fd,
        )
    except (FileNotFoundError,OSError):
        return None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>limit:
            return None
        raw=b""
        while len(raw)<=limit:
            chunk=os.read(fd,limit+1-len(raw))
            if not chunk:
                break
            raw+=chunk
        after=os.fstat(fd)
    finally:
        os.close(fd)
    if (len(raw)>limit or len(raw)!=before.st_size or
        (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns
        )):
        return None
    return raw

def open_direct_directory_at(parent_fd,name):
    try:
        fd=os.open(
            name,
            os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return None,"missing"
    except OSError:
        return None,"not_direct_directory"
    try:
        current=os.fstat(fd)
    except OSError:
        os.close(fd)
        return None,"directory_metadata_unavailable"
    if not stat.S_ISDIR(current.st_mode):
        os.close(fd)
        return None,"not_direct_directory"
    return fd,None

def directory_tree_stats(root_fd,visited=None,depth=0):
    if depth>32:
        return None
    files=0
    total=0
    visited=set() if visited is None else visited
    current=os.fstat(root_fd)
    identity=(current.st_dev,current.st_ino)
    if identity in visited:
        return files,total
    visited.add(identity)
    try:
        names=os.listdir(root_fd)
    except OSError:
        return None
    for name in names:
        try:
            info=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
        except (FileNotFoundError,OSError):
            return None
        if stat.S_ISREG(info.st_mode):
            files+=1
            total+=info.st_size
        elif stat.S_ISDIR(info.st_mode):
            child_fd,_=open_direct_directory_at(root_fd,name)
            if child_fd is None:
                return None
            try:
                child=directory_tree_stats(child_fd,visited,depth+1)
            finally:
                os.close(child_fd)
            if child is None:
                return None
            child_files,child_total=child
            files+=child_files
            total+=child_total
    return files,total

def safe_frame(value):
    if not isinstance(value,dict) or set(value)!={"file","line","function"}:
        return None
    if (not isinstance(value["file"],str) or not SAFE_FILES.fullmatch(value["file"])
        or type(value["line"]) is not int or not 1<=value["line"]<=2147483647
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
            and 0<=value["checkpoint_global_step"]<=2147483647
            and type(value.get("completed_batches")) is int
            and 0<=value["completed_batches"]<=2147483647
            and type(value.get("completed_at")) in (int,float)
            and 0<=value["completed_at"]<1e20
            and value.get("optimizer_update_independently_verified") is False
            and value.get("checkpoint_reload_verified") is False):
            selected={key:value[key] for key in (
                "status","plan_sha256","checkpoint_global_step","completed_batches",
                "optimizer_update_independently_verified","checkpoint_reload_verified")}
    if selected is None:
        return {**common,"reason":"schema_or_binding_mismatch"}
    return {**common,"accepted":True,"selected":selected}

def batch_inventory(root_fd):
    episodes_fd,reason=open_direct_directory_at(root_fd,"episodes")
    if episodes_fd is None:
        if reason!="missing":
            return {"directory_present":True,"accepted":False,"reason":reason}
        return {
            "directory_present":False,"batch_directories":0,
            "markers":{
                name:{"count":0,"direct_regular_count":0,"indirect_count":0,"latest":None}
                for name in BATCH_MARKERS
            },
        }
    try:
        parent_fd,reason=open_direct_directory_at(episodes_fd,"batches")
    finally:
        os.close(episodes_fd)
    if parent_fd is None:
        if reason!="missing":
            return {"directory_present":True,"accepted":False,"reason":reason}
        return {
            "directory_present":False,"batch_directories":0,
            "markers":{
                name:{"count":0,"direct_regular_count":0,"indirect_count":0,"latest":None}
                for name in BATCH_MARKERS
            },
        }
    markers={
        name:{
            "count":0,"direct_regular_count":0,"indirect_count":0,"latest":None,
            "latest_mtime_ns":-1,
        }
        for name in BATCH_MARKERS
    }
    count=0
    try:
        try:
            names=os.listdir(parent_fd)
        except OSError:
            return {
                "directory_present":True,"accepted":False,
                "reason":"batch_inventory_incomplete",
            }
        for entry_name in sorted(names):
            try:
                entry_info=os.stat(entry_name,dir_fd=parent_fd,follow_symlinks=False)
            except (FileNotFoundError,OSError):
                return {
                    "directory_present":True,"accepted":False,
                    "reason":"batch_inventory_incomplete",
                }
            if not stat.S_ISDIR(entry_info.st_mode):
                continue
            entry_fd,_=open_direct_directory_at(parent_fd,entry_name)
            if entry_fd is None:
                return {
                    "directory_present":True,"accepted":False,
                    "reason":"batch_inventory_incomplete",
                }
            try:
                count+=1
                for name in BATCH_MARKERS:
                    row=stat_info_at(entry_fd,name)
                    if row is not None:
                        summary=markers[name]
                        summary["count"]+=1
                        accepted=row.get("accepted") is True
                        summary["direct_regular_count" if accepted else "indirect_count"]+=1
                        mtime=row.get("mtime_ns",-1)
                        if accepted and mtime>=summary["latest_mtime_ns"]:
                            summary["latest_mtime_ns"]=mtime
                            summary["latest"]={"size_bytes":row["size_bytes"],"mtime_ns":mtime}
            finally:
                os.close(entry_fd)
    finally:
        os.close(parent_fd)
    for summary in markers.values():
        summary.pop("latest_mtime_ns")
    return {"directory_present":True,"accepted":True,"batch_directories":count,"markers":markers}

def checkpoint_inventory(root_fd):
    parent_fd,reason=open_direct_directory_at(root_fd,"checkpoints")
    if parent_fd is None and reason=="missing":
        return {
            "directory_present":False,"global_step_count":0,"minimum_step":None,
            "maximum_step":None,"boundary_steps":[],"file_count":0,"total_bytes":0,
            "latest_pointer_step":None,
        }
    if parent_fd is None:
        return {"directory_present":True,"accepted":False,"reason":reason}
    try:
        pinned=os.fstat(parent_fd)
        if not stat.S_ISDIR(pinned.st_mode):
            return {"directory_present":True,"accepted":False,"reason":"not_direct_directory"}
        steps=[]
        files=0
        total=0
        try:
            names=os.listdir(parent_fd)
        except OSError:
            return {
                "directory_present":True,"accepted":False,
                "reason":"checkpoint_inventory_incomplete",
            }
        for name in sorted(names):
            match=re.fullmatch(r"global_step_(\d+)",name)
            if match:
                child_fd,_=open_direct_directory_at(parent_fd,name)
                if child_fd is None:
                    return {
                        "directory_present":True,"accepted":False,
                        "reason":"checkpoint_inventory_incomplete",
                    }
                try:
                    child=directory_tree_stats(child_fd)
                finally:
                    os.close(child_fd)
                if child is None:
                    return {
                        "directory_present":True,"accepted":False,
                        "reason":"checkpoint_inventory_incomplete",
                    }
                child_files,child_total=child
                steps.append(int(match.group(1)))
                files+=child_files
                total+=child_total
        steps=sorted(set(steps))
        boundary=steps[:2]+[step for step in steps[-2:] if step not in steps[:2]]
        latest=None
        try:
            raw=read_small_direct_at(parent_fd,"latest_ckpt_global_step.txt",32)
            text=raw.decode().strip() if raw is not None else ""
            if text.isdecimal():
                latest=int(text)
        except UnicodeDecodeError:
            pass
    finally:
        os.close(parent_fd)
    return {
        "directory_present":True,"accepted":True,"global_step_count":len(steps),
        "minimum_step":steps[0] if steps else None,"maximum_step":steps[-1] if steps else None,
        "boundary_steps":boundary,"file_count":files,"total_bytes":total,
        "latest_pointer_step":latest,
    }

try:
    root_fd=os.open(
        ROOT,
        os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0),
    )
except FileNotFoundError:
    root_fd=None
    root_exists=False
    root_direct=False
except OSError:
    root_fd=None
    root_exists=True
    root_direct=False
else:
    root_exists=True
    root_direct=stat.S_ISDIR(os.fstat(root_fd).st_mode)
root_markers=(
    [
        sanitize_root(
            name,
            stat_info_at(root_fd,name) if name=="ACCEPTED.json" else file_info_at(root_fd,name),
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

def cause_summary(markers):
    by_name={row["name"]:row for row in markers}
    result={}
    outer=by_name.get("FAILED.json",{}).get("selected",{})
    if outer:
        result["outer_error_class"]=outer.get("error_class")
        result["watchdog_reason"]=outer.get("watchdog_reason")
    native=by_name.get("NATIVE_FAILURE.json",{}).get("selected",{})
    causes=native.get("causes",[])
    if causes:
        first=causes[0]
        result.update({
            "native_causes_total":native.get("causes_total"),
            "native_error_class":first.get("error_class"),
            "native_actor_init_failed":first.get("actor_init_failed"),
            "native_remote_error_classes":first.get("remote_error_classes",[])[:2],
            "native_first_remote_frame":(
                first.get("remote_frames",[])[0] if first.get("remote_frames") else None
            ),
        })
    return result

def finalize(value):
    value=dict(value)
    value["receipt_size_bytes"]=0
    while True:
        value.pop("sha256",None)
        value["sha256"]="sha256:"+digest(value)
        payload=json.dumps(value,sort_keys=True,separators=(",",":"))+"\n"
        size=len(payload.encode())
        if value["receipt_size_bytes"]==size:
            return value,payload
        value["receipt_size_bytes"]=size

if root_direct:
    try:
        batches=batch_inventory(root_fd)
        checkpoints=checkpoint_inventory(root_fd)
    finally:
        os.close(root_fd)
else:
    batches={"directory_present":False}
    checkpoints={"directory_present":False}
base={
    "schema":SCHEMA,"status":"inspected","target":str(ROOT),"training_plan_sha256":PLAN,
    "checked_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpus":0,"sfs_mount_read_only":True,"private_payloads_read":False,
    "root_exists":root_exists,"root_direct":root_direct,"terminal_classification":terminal,
    "receipt_size_limit_bytes":MAX_RECEIPT_BYTES,
}
marker_summary=[
    {key:row[key] for key in ("name","present","accepted","reason") if key in row}
    for row in root_markers
]
batch_counts={
    name:item.get("count",0) for name,item in batches.get("markers",{}).items()
}
candidates=(
    {**base,"receipt_compaction_level":0,"cause_summary":cause_summary(root_markers),
     "root_markers":root_markers,"batch_inventory":batches,"checkpoint_inventory":checkpoints},
    {**base,"receipt_compaction_level":1,"cause_summary":cause_summary(root_markers),
     "root_markers":marker_summary,
     "batch_inventory":batches,
     "checkpoint_inventory":checkpoints},
    {**base,"receipt_compaction_level":2,"cause_summary":cause_summary(root_markers),
     "root_markers":[row["name"] for row in root_markers if row.get("present")],
     "batch_inventory":{"batch_directories":batches.get("batch_directories",0),
                        "marker_counts":batch_counts},
     "checkpoint_inventory":{
         key:checkpoints.get(key) for key in (
             "directory_present","global_step_count","minimum_step","maximum_step",
             "boundary_steps","file_count","total_bytes","latest_pointer_step"
         )
     }},
    {**base,"receipt_compaction_level":3,"cause_summary":cause_summary(root_markers),
     "root_markers":[row["name"] for row in root_markers if row.get("present")],
     "batch_inventory":{"batch_directories":batches.get("batch_directories",0)},
     "checkpoint_inventory":{
         key:checkpoints.get(key) for key in (
             "global_step_count","minimum_step","maximum_step","file_count","total_bytes",
             "latest_pointer_step"
         )
     }},
)
for candidate in candidates:
    body,payload=finalize(candidate)
    if body["receipt_size_bytes"]<=MAX_RECEIPT_BYTES:
        break
else:
    raise RuntimeError("fixed compact receipt exceeded termination-message bound")
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
        or type(value.get("gpus")) is not int
        or value["gpus"] != 0
        or value.get("sfs_mount_read_only") is not True
        or value.get("private_payloads_read") is not False
        or type(value.get("receipt_size_limit_bytes")) is not int
        or value["receipt_size_limit_bytes"] != 3500
        or type(value.get("receipt_size_bytes")) is not int
        or type(value.get("receipt_compaction_level")) is not int
        or value.get("receipt_compaction_level") not in {0, 1, 2, 3}
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
