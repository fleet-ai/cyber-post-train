#!/usr/bin/env python3
"""Build and verify the one-shot read-only prod8 terminal-evidence probe."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import (
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    JobsError,
    digest,
    quantity,
)
from training import skyrl_reward_rayjob as direct

NAME = "chris-q38-prod8-terminal-probe-v1"
TARGET = "/mnt/sfs/jobs/chris-q38-rlreward-prod8"
TRAINING_PLAN_SHA256 = "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
    "sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RECEIPT_SCHEMA = "cyber_qwen38_prod8_terminal_probe_receipt_v2"
PREVIEW_SCHEMA = "cyber_qwen38_prod8_terminal_probe_preview_v1"
MAX_RECEIPT_BYTES = 3500
MAX_SCAN_ATTEMPTS = 2
MAX_COUNT = 2_147_483_647
MAX_TOTAL_BYTES = 9_223_372_036_854_775_807
ROOT_MARKERS = (
    "STARTED.json",
    "FAILED.json",
    "NATIVE_FAILURE.json",
    "NATIVE_REJECTED.json",
    "REJECTED.json",
    "NATIVE_TRAINING_COMPLETE.json",
    "ACCEPTED.json",
)
BATCH_MARKERS = ("STARTED.json", "COLLECTED.json", "REJECTED.json", "FAILED.json")
MARKER_REASONS = frozenset(
    {
        "absent",
        "accepted",
        "not_direct_regular_file",
        "not_small_regular_file",
        "changed_during_read",
        "invalid_json",
        "not_object",
        "invalid_digest_input",
        "digest_mismatch",
        "schema_or_binding_mismatch",
        "scan_mutated",
    }
)
STABLE_REJECTED_MARKER_REASONS = MARKER_REASONS - {
    "absent",
    "accepted",
    "changed_during_read",
    "scan_mutated",
}
TERMINAL_CLASSIFICATIONS = frozenset(
    {
        "failed",
        "rejected",
        "native_training_complete",
        "unaccepted_failed_marker",
        "unaccepted_rejected_marker",
        "unaccepted_complete_marker",
        "no_terminal_marker",
        "root_missing",
        "unaccepted_root",
        "unaccepted_scan_mutation",
    }
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RFC3339 = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

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
MAX_SCAN_ATTEMPTS=2
MAX_COUNT=2147483647
MAX_TOTAL_BYTES=9223372036854775807
ROOT_MARKERS=(
    "STARTED.json","FAILED.json","NATIVE_FAILURE.json","NATIVE_REJECTED.json",
    "REJECTED.json","NATIVE_TRAINING_COMPLETE.json","ACCEPTED.json",
)
BATCH_MARKERS=("STARTED.json","COLLECTED.json","REJECTED.json","FAILED.json")
SAFE_WATCHDOGS=(None,"hard_runtime_bound","confirmed_no_progress_idle")
SAFE_NAMES=re.compile(r"^[A-Za-z_][A-Za-z0-9_.<>-]{0,63}$")
SAFE_FILES=re.compile(r"^[A-Za-z0-9_.-]{1,60}\.py$")

class ScanMutation(Exception):
    pass

def stat_signature(value):
    return (
        value.st_mode,value.st_dev,value.st_ino,value.st_size,
        value.st_mtime_ns,value.st_ctime_ns,
    )

def directory_snapshot(fd):
    before=os.fstat(fd)
    if not stat.S_ISDIR(before.st_mode):
        raise ScanMutation()
    try:
        names=os.listdir(fd)
    except OSError as exc:
        raise ScanMutation() from exc
    if len(names)>MAX_COUNT:
        raise ScanMutation()
    rows=[]
    for name in sorted(names):
        try:
            current=os.stat(name,dir_fd=fd,follow_symlinks=False)
        except (FileNotFoundError,OSError) as exc:
            raise ScanMutation() from exc
        rows.append((name,stat_signature(current)))
    after=os.fstat(fd)
    if stat_signature(before)!=stat_signature(after):
        raise ScanMutation()
    return stat_signature(before),tuple(rows)

def assert_directory_unchanged(fd,before):
    if directory_snapshot(fd)!=before:
        raise ScanMutation()

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
        raise ScanMutation()
    files=0
    total=0
    visited=set() if visited is None else visited
    current=os.fstat(root_fd)
    identity=(current.st_dev,current.st_ino)
    if identity in visited:
        return files,total
    visited.add(identity)
    before=directory_snapshot(root_fd)
    for name,expected in before[1]:
        try:
            info=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
        except (FileNotFoundError,OSError) as exc:
            raise ScanMutation() from exc
        if stat_signature(info)!=expected:
            raise ScanMutation()
        if stat.S_ISREG(info.st_mode):
            files+=1
            total+=info.st_size
            if files>MAX_COUNT or total>MAX_TOTAL_BYTES:
                raise ScanMutation()
        elif stat.S_ISDIR(info.st_mode):
            child_fd,_=open_direct_directory_at(root_fd,name)
            if child_fd is None:
                raise ScanMutation()
            try:
                child=directory_tree_stats(child_fd,visited,depth+1)
            finally:
                os.close(child_fd)
            child_files,child_total=child
            files+=child_files
            total+=child_total
            if files>MAX_COUNT or total>MAX_TOTAL_BYTES:
                raise ScanMutation()
    assert_directory_unchanged(root_fd,before)
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
        return {"name":name,"present":False,"accepted":False,"reason":"absent"}
    base={"name":name,"present":True,"accepted":False}
    if not info.get("accepted"):
        return {**base,"reason":info.get("reason","schema_or_binding_mismatch")}
    if name=="ACCEPTED.json":
        return {**base,"accepted":True,"reason":"accepted"}
    value=info["value"]
    accepted=False
    if name=="STARTED.json" and set(value)=={"plan_sha256","started_at","sha256"}:
        if value.get("plan_sha256")==PLAN and type(value.get("started_at")) in (int,float):
            accepted=True
    elif name=="FAILED.json" and set(value)=={
        "status","plan_sha256","error_class","watchdog_reason","sha256"
    }:
        if (value.get("status")=="failed" and value.get("plan_sha256")==PLAN
            and isinstance(value.get("error_class"),str)
            and SAFE_NAMES.fullmatch(value["error_class"])
            and value.get("watchdog_reason") in SAFE_WATCHDOGS):
            accepted=True
    elif name=="NATIVE_FAILURE.json" and set(value)=={"plan_sha256","causes","sha256"}:
        causes=value.get("causes")
        safe=(
            [safe_cause(item) for item in causes]
            if isinstance(causes,list) and len(causes)<=8 else []
        )
        if value.get("plan_sha256")==PLAN and isinstance(causes,list) and safe and all(safe):
            accepted=True
    elif name in ("NATIVE_REJECTED.json","REJECTED.json") and set(value)=={
        "schema","status","reason","plan_sha256","sha256"
    }:
        if (value.get("schema")=="cyber_rl_native_rejection_v1" and value.get("status")=="rejected"
            and value.get("plan_sha256")==PLAN and isinstance(value.get("reason"),str)
            and SAFE_NAMES.fullmatch(value["reason"])):
            accepted=True
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
            accepted=True
    if not accepted:
        return {**base,"reason":"schema_or_binding_mismatch"}
    return {**base,"accepted":True,"reason":"accepted"}

def batch_inventory(root_fd):
    empty={
        "directory_present":False,"accepted":True,"batch_directories":0,
        "markers":{
            name:{"count":0,"direct_regular_count":0,"indirect_count":0}
            for name in BATCH_MARKERS
        },
    }
    episodes_fd,reason=open_direct_directory_at(root_fd,"episodes")
    if episodes_fd is None:
        if reason!="missing":
            return {**empty,"directory_present":True,"accepted":False}
        return empty
    try:
        episodes_before=directory_snapshot(episodes_fd)
        parent_fd,reason=open_direct_directory_at(episodes_fd,"batches")
        if parent_fd is None:
            assert_directory_unchanged(episodes_fd,episodes_before)
            if reason!="missing":
                return {**empty,"directory_present":True,"accepted":False}
            return empty
        markers={
            name:{"count":0,"direct_regular_count":0,"indirect_count":0}
            for name in BATCH_MARKERS
        }
        count=0
        try:
            parent_before=directory_snapshot(parent_fd)
            for entry_name,expected in parent_before[1]:
                try:
                    entry_info=os.stat(entry_name,dir_fd=parent_fd,follow_symlinks=False)
                except (FileNotFoundError,OSError) as exc:
                    raise ScanMutation() from exc
                if stat_signature(entry_info)!=expected:
                    raise ScanMutation()
                if not stat.S_ISDIR(entry_info.st_mode):
                    continue
                entry_fd,_=open_direct_directory_at(parent_fd,entry_name)
                if entry_fd is None:
                    raise ScanMutation()
                try:
                    count+=1
                    if count>MAX_COUNT:
                        raise ScanMutation()
                    entry_before=directory_snapshot(entry_fd)
                    for name in BATCH_MARKERS:
                        row=stat_info_at(entry_fd,name)
                        if row is not None:
                            summary=markers[name]
                            summary["count"]+=1
                            if summary["count"]>MAX_COUNT:
                                raise ScanMutation()
                            accepted=row.get("accepted") is True
                            summary["direct_regular_count" if accepted else "indirect_count"]+=1
                    assert_directory_unchanged(entry_fd,entry_before)
                finally:
                    os.close(entry_fd)
            assert_directory_unchanged(parent_fd,parent_before)
        finally:
            os.close(parent_fd)
        assert_directory_unchanged(episodes_fd,episodes_before)
        return {
            "directory_present":True,
            "accepted":True,
            "batch_directories":count,
            "markers":markers,
        }
    finally:
        os.close(episodes_fd)

def checkpoint_inventory(root_fd):
    empty={
        "directory_present":False,"accepted":True,"global_step_count":0,
        "minimum_step":None,"maximum_step":None,"boundary_steps":[],
        "file_count":0,"total_bytes":0,"latest_pointer_step":None,
    }
    parent_fd,reason=open_direct_directory_at(root_fd,"checkpoints")
    if parent_fd is None and reason=="missing":
        return empty
    if parent_fd is None:
        return {**empty,"directory_present":True,"accepted":False}
    try:
        pinned=os.fstat(parent_fd)
        if not stat.S_ISDIR(pinned.st_mode):
            return {**empty,"directory_present":True,"accepted":False}
        steps=[]
        files=0
        total=0
        parent_before=directory_snapshot(parent_fd)
        for name,expected in parent_before[1]:
            try:
                current=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
            except (FileNotFoundError,OSError) as exc:
                raise ScanMutation() from exc
            if stat_signature(current)!=expected:
                raise ScanMutation()
            match=re.fullmatch(r"global_step_(\d+)",name)
            if match:
                step=int(match.group(1))
                if step>MAX_COUNT:
                    raise ScanMutation()
                child_fd,_=open_direct_directory_at(parent_fd,name)
                if child_fd is None:
                    raise ScanMutation()
                try:
                    child=directory_tree_stats(child_fd)
                finally:
                    os.close(child_fd)
                child_files,child_total=child
                steps.append(step)
                files+=child_files
                total+=child_total
                if len(steps)>MAX_COUNT or files>MAX_COUNT or total>MAX_TOTAL_BYTES:
                    raise ScanMutation()
        steps=sorted(set(steps))
        boundary=steps[:2]+[step for step in steps[-2:] if step not in steps[:2]]
        latest=None
        try:
            raw=read_small_direct_at(parent_fd,"latest_ckpt_global_step.txt",32)
            text=raw.decode().strip() if raw is not None else ""
            if text.isdecimal():
                latest=int(text)
                if latest>MAX_COUNT:
                    latest=None
        except UnicodeDecodeError:
            pass
        assert_directory_unchanged(parent_fd,parent_before)
    finally:
        os.close(parent_fd)
    return {
        "directory_present":True,"accepted":True,"global_step_count":len(steps),
        "minimum_step":steps[0] if steps else None,"maximum_step":steps[-1] if steps else None,
        "boundary_steps":boundary,"file_count":files,"total_bytes":total,
        "latest_pointer_step":latest,
    }

def marker_rows(reason):
    return [
        {"name":name,"present":False,"accepted":False,"reason":reason}
        for name in ROOT_MARKERS
    ]

def empty_batch(accepted):
    return {
        "directory_present":False,"accepted":accepted,"batch_directories":0,
        "markers":{
            name:{"count":0,"direct_regular_count":0,"indirect_count":0}
            for name in BATCH_MARKERS
        },
    }

def empty_checkpoint(accepted):
    return {
        "directory_present":False,"accepted":accepted,"global_step_count":0,
        "minimum_step":None,"maximum_step":None,"boundary_steps":[],
        "file_count":0,"total_bytes":0,"latest_pointer_step":None,
    }

def path_signature():
    try:
        return stat_signature(os.stat(ROOT,follow_symlinks=False))
    except FileNotFoundError:
        return None
    except OSError:
        return ("unavailable",)

def classify(root_exists,root_direct,markers,stable):
    if not stable:
        return "unaccepted_scan_mutation"
    if not root_exists:
        return "root_missing"
    if not root_direct:
        return "unaccepted_root"
    by_name={row["name"]:row for row in markers}
    for name,accepted,rejected in (
        ("FAILED.json","failed","unaccepted_failed_marker"),
        ("REJECTED.json","rejected","unaccepted_rejected_marker"),
        ("NATIVE_TRAINING_COMPLETE.json","native_training_complete","unaccepted_complete_marker"),
    ):
        row=by_name[name]
        if row["present"]:
            return accepted if row["accepted"] else rejected
    return "no_terminal_marker"

def scan_once():
    before_path=path_signature()
    try:
        root_fd=os.open(
            ROOT,
            os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0),
        )
    except FileNotFoundError:
        if before_path!=path_signature():
            raise ScanMutation()
        return False,False,marker_rows("absent"),empty_batch(True),empty_checkpoint(True)
    except OSError:
        if before_path!=path_signature():
            raise ScanMutation()
        return True,False,marker_rows("absent"),empty_batch(True),empty_checkpoint(True)
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            return True,False,marker_rows("absent"),empty_batch(True),empty_checkpoint(True)
        root_before=directory_snapshot(root_fd)
        markers=[
            sanitize_root(
                name,
                stat_info_at(root_fd,name) if name=="ACCEPTED.json" else file_info_at(root_fd,name),
            )
            for name in ROOT_MARKERS
        ]
        if any(row["reason"]=="changed_during_read" for row in markers):
            raise ScanMutation()
        batches=batch_inventory(root_fd)
        checkpoints=checkpoint_inventory(root_fd)
        assert_directory_unchanged(root_fd,root_before)
    finally:
        os.close(root_fd)
    if before_path!=path_signature():
        raise ScanMutation()
    return True,True,markers,batches,checkpoints

scan_attempts=0
scan_stable=False
root_exists=False
root_direct=False
root_markers=marker_rows("scan_mutated")
batches=empty_batch(False)
checkpoints=empty_checkpoint(False)
for scan_attempts in range(1,MAX_SCAN_ATTEMPTS+1):
    try:
        root_exists,root_direct,root_markers,batches,checkpoints=scan_once()
    except ScanMutation:
        continue
    scan_stable=True
    break
if not scan_stable:
    root_markers=marker_rows("scan_mutated")
    batches=empty_batch(False)
    checkpoints=empty_checkpoint(False)

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

terminal=classify(root_exists,root_direct,root_markers,scan_stable)
body,payload=finalize({
    "schema":SCHEMA,
    "status":"inspected" if scan_stable else "unaccepted",
    "target":str(ROOT),
    "training_plan_sha256":PLAN,
    "checked_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpus":0,
    "sfs_mount_read_only":True,
    "private_payloads_read":False,
    "root_exists":root_exists,
    "root_direct":root_direct,
    "scan":{"attempts":scan_attempts,"stable":scan_stable},
    "terminal_classification":terminal,
    "root_markers":root_markers,
    "batch_inventory":batches,
    "checkpoint_inventory":checkpoints,
    "receipt_size_limit_bytes":MAX_RECEIPT_BYTES,
})
if body["receipt_size_bytes"]>MAX_RECEIPT_BYTES:
    raise RuntimeError("canonical receipt exceeded termination-message bound")
RECEIPT.write_text(payload)
print(json.dumps({"status":body["status"],"terminal_classification":terminal,
                  "sha256":body["sha256"]},sort_keys=True))
"""


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate receipt key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite receipt number")


def parse_receipt_json(value: str | bytes) -> object:
    """Parse a receipt without duplicate keys or non-finite JSON numbers."""
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )


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


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobsError(f"prod8 terminal probe {label} is not an object")
    return value


def _image_digest(image: object, label: str) -> str:
    if not isinstance(image, str) or "@" not in image:
        raise JobsError(f"prod8 terminal probe {label} is not digest-pinned")
    digest_value = image.rsplit("@", 1)[1]
    if _DIGEST.fullmatch(digest_value) is None:
        raise JobsError(f"prod8 terminal probe {label} digest changed")
    return digest_value


def _zero_gpu(resources: object, label: str) -> None:
    if resources is None:
        return
    value = _mapping(resources, label)
    for key in ("requests", "limits"):
        selected = value.get(key, {})
        if not isinstance(selected, dict):
            raise JobsError(f"prod8 terminal probe {label} {key} changed")
        if "nvidia.com/gpu" in selected:
            try:
                gpu = quantity(selected["nvidia.com/gpu"])
            except JobsError as exc:
                raise JobsError(f"prod8 terminal probe {label} GPU quantity changed") from exc
            if gpu != 0:
                raise JobsError(f"prod8 terminal probe {label} requests GPU")


def _resource_shape(value: object, label: str) -> dict[str, dict[str, Any]]:
    resources = _mapping(value, label)
    if set(resources) != {"requests", "limits"}:
        raise JobsError(f"prod8 terminal probe {label} fields changed")
    _zero_gpu(resources, label)
    result: dict[str, dict[str, Any]] = {}
    for key in ("requests", "limits"):
        selected = resources[key]
        if not isinstance(selected, dict) or not all(isinstance(name, str) for name in selected):
            raise JobsError(f"prod8 terminal probe {label} {key} shape changed")
        result[key] = {name: selected[name] for name in sorted(selected)}
    return result


def _env_shape(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise JobsError("prod8 terminal probe environment is not a list")
    result: list[dict[str, str]] = []
    names: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name", "value"}:
            raise JobsError("prod8 terminal probe environment fields changed")
        name, item = row.get("name"), row.get("value")
        if not isinstance(name, str) or not name or not isinstance(item, str) or name in names:
            raise JobsError("prod8 terminal probe environment value changed")
        names.add(name)
        result.append({"name": name, "value": item})
    return sorted(result, key=lambda row: row["name"])


def _list_of_mappings(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise JobsError(f"prod8 terminal probe {label} shape changed")
    return value


def _exactly_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exactly_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exactly_equal(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _controller_labels(uid: str) -> dict[str, str]:
    return {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": NAME,
        "controller-uid": uid,
        "job-name": NAME,
    }


def _assert_manifest_matches_source(value: object) -> None:
    """Strip only known API defaults, then require the source Job byte shape."""
    actual = copy.deepcopy(_mapping(value, "Job"))
    status = actual.pop("status", None)
    if status is not None and not isinstance(status, dict):
        raise JobsError("prod8 terminal probe Job status changed")
    metadata = _mapping(actual.get("metadata"), "Job metadata")
    uid = metadata.get("uid")
    labels = metadata.pop("labels", None)
    if uid is None:
        if labels is not None:
            raise JobsError("prod8 terminal probe Job labels changed")
    elif not isinstance(uid, str) or not _exactly_equal(labels, _controller_labels(uid)):
        raise JobsError("prod8 terminal probe Job controller labels changed")
    generation = metadata.pop("generation", None)
    if generation is not None and (type(generation) is not int or generation != 1):
        raise JobsError("prod8 terminal probe Job generation changed")
    for key in ("creationTimestamp", "managedFields", "resourceVersion", "uid"):
        metadata.pop(key, None)

    spec = _mapping(actual.get("spec"), "Job spec")
    server_defaults: dict[str, Any] = {
        "completionMode": "NonIndexed",
        "completions": 1,
        "manualSelector": False,
        "parallelism": 1,
        "podReplacementPolicy": "TerminatingOrFailed",
        "suspend": False,
    }
    for key, expected in server_defaults.items():
        observed = spec.pop(key, None)
        if observed is not None and not _exactly_equal(observed, expected):
            raise JobsError("prod8 terminal probe Job server default changed")
    selector = spec.pop("selector", None)
    expected_selector = {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}}
    if (uid is not None and not _exactly_equal(selector, expected_selector)) or (
        uid is None and selector is not None
    ):
        raise JobsError("prod8 terminal probe Job selector changed")

    template = _mapping(spec.get("template"), "Job template")
    template_metadata = _mapping(template.get("metadata"), "Pod template metadata")
    template_labels = template_metadata.pop("labels", None)
    if uid is None:
        if template_labels is not None:
            raise JobsError("prod8 terminal probe Pod-template labels changed")
    elif not _exactly_equal(template_labels, _controller_labels(uid)):
        raise JobsError("prod8 terminal probe Pod-template controller labels changed")
    template_created = template_metadata.pop("creationTimestamp", None)
    if template_created is not None:
        raise JobsError("prod8 terminal probe Pod-template timestamp changed")

    pod = _mapping(template.get("spec"), "Pod spec")
    pod_defaults: dict[str, Any] = {
        "dnsPolicy": "ClusterFirst",
        "schedulerName": "default-scheduler",
        "terminationGracePeriodSeconds": 30,
    }
    for key, expected in pod_defaults.items():
        observed = pod.pop(key, None)
        if observed is not None and not _exactly_equal(observed, expected):
            raise JobsError("prod8 terminal probe Pod server default changed")
    containers = _list_of_mappings(pod.get("containers"), "containers")
    for container in containers:
        pull_policy = container.pop("imagePullPolicy", None)
        if pull_policy is not None and not _exactly_equal(pull_policy, "IfNotPresent"):
            raise JobsError("prod8 terminal probe image-pull policy changed")
    if not _exactly_equal(actual, manifest()):
        raise JobsError("prod8 terminal probe Job manifest changed")


def normalized_manifest(value: object) -> dict[str, Any]:
    """Return the safety-critical, server-stable Job projection or fail closed."""
    _assert_manifest_matches_source(value)
    job = _mapping(value, "Job")
    metadata = _mapping(job.get("metadata"), "Job metadata")
    spec = _mapping(job.get("spec"), "Job spec")
    template = _mapping(spec.get("template"), "Job template")
    pod = _mapping(template.get("spec"), "Pod spec")
    annotations = _mapping(metadata.get("annotations"), "Job annotations")
    containers = _list_of_mappings(pod.get("containers"), "containers")
    if len(containers) != 1:
        raise JobsError("prod8 terminal probe has not exactly one container")
    container = containers[0]
    command = container.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise JobsError("prod8 terminal probe command changed")
    security = _mapping(container.get("securityContext"), "container security context")
    pod_security = _mapping(pod.get("securityContext"), "Pod security context")
    mounts = _list_of_mappings(container.get("volumeMounts"), "volume mounts")
    volumes = _list_of_mappings(pod.get("volumes"), "volumes")
    init = pod.get("initContainers", [])
    ephemeral = pod.get("ephemeralContainers", [])
    if init not in ([], None) or ephemeral not in ([], None):
        raise JobsError("prod8 terminal probe has an unexpected extra container")
    overhead = pod.get("overhead", {})
    if overhead not in ({}, None):
        _zero_gpu(overhead, "Pod overhead")
        raise JobsError("prod8 terminal probe has Pod overhead")
    _zero_gpu({"requests": overhead or {}, "limits": overhead or {}}, "Pod overhead")
    return {
        "api_version": job.get("apiVersion"),
        "kind": job.get("kind"),
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "failure_alerts": annotations.get(FAILURE_ALERT_ANNOTATION),
        "active_deadline_seconds": spec.get("activeDeadlineSeconds"),
        "backoff_limit": spec.get("backoffLimit"),
        "automount_service_account_token": pod.get("automountServiceAccountToken"),
        "priority_class_name": pod.get("priorityClassName"),
        "restart_policy": pod.get("restartPolicy"),
        "pod_security_context": {
            "supplemental_groups": pod_security.get("supplementalGroups"),
        },
        "volumes": sorted(volumes, key=lambda row: json.dumps(row, sort_keys=True)),
        "container": {
            "name": container.get("name"),
            "image": container.get("image"),
            "image_digest": _image_digest(container.get("image"), "requested image"),
            "command_sha256": digest(command),
            "environment": _env_shape(container.get("env")),
            "resources": _resource_shape(container.get("resources"), "container resources"),
            "security_context": {
                "allow_privilege_escalation": security.get("allowPrivilegeEscalation"),
                "privileged": security.get("privileged"),
                "read_only_root_filesystem": security.get("readOnlyRootFilesystem"),
                "run_as_group": security.get("runAsGroup"),
                "run_as_non_root": security.get("runAsNonRoot"),
                "run_as_user": security.get("runAsUser"),
            },
            "termination_message_path": container.get("terminationMessagePath"),
            "termination_message_policy": container.get("terminationMessagePolicy"),
            "volume_mounts": sorted(mounts, key=lambda row: json.dumps(row, sort_keys=True)),
        },
    }


def manifest_digest() -> str:
    return digest(manifest())


def expected_normalized_manifest() -> dict[str, Any]:
    return normalized_manifest(manifest())


def _runtime_image_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"@(?P<digest>sha256:[0-9a-f]{64})\Z", value)
    return match.group("digest") if match else None


def normalized_terminal_pod(value: object, *, job_uid: str) -> dict[str, Any]:
    """Bind an observed Pod and its sole receipt container to the immutable Job."""
    pod = _mapping(value, "Pod")
    metadata = _mapping(pod.get("metadata"), "Pod metadata")
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": NAME,
        "uid": job_uid,
        "controller": True,
        "blockOwnerDeletion": True,
    }
    if not _exactly_equal(metadata.get("ownerReferences"), [owner]):
        raise JobsError("prod8 terminal probe Pod owner UID changed")
    labels = _mapping(metadata.get("labels"), "Pod labels")
    if (
        labels.get("batch.kubernetes.io/controller-uid") != job_uid
        or labels.get("job-name") != NAME
    ):
        raise JobsError("prod8 terminal probe Pod controller labels changed")
    spec = _mapping(pod.get("spec"), "Pod spec")
    if spec.get("priorityClassName") != "c1":
        raise JobsError("prod8 terminal probe Pod priority changed")
    init = spec.get("initContainers", [])
    ephemeral = spec.get("ephemeralContainers", [])
    if init not in ([], None) or ephemeral not in ([], None):
        raise JobsError("prod8 terminal probe Pod has an extra container")
    overhead = spec.get("overhead", {})
    _zero_gpu({"requests": overhead or {}, "limits": overhead or {}}, "Pod overhead")
    containers = _list_of_mappings(spec.get("containers"), "Pod containers")
    if len(containers) != 1:
        raise JobsError("prod8 terminal probe Pod does not have one container")
    container = containers[0]
    _zero_gpu(container.get("resources", {}), "Pod container resources")
    if container.get("name") != "terminal-probe" or container.get("image") != IMAGE:
        raise JobsError("prod8 terminal probe Pod requested image changed")
    statuses = _list_of_mappings(
        _mapping(pod.get("status", {}), "Pod status").get("containerStatuses", []),
        "container statuses",
    )
    if not statuses:
        return {
            "pod_name": metadata.get("name"),
            "pod_uid": metadata.get("uid"),
            "container_name": "terminal-probe",
            "requested_image": IMAGE,
            "requested_image_digest": _image_digest(IMAGE, "expected image"),
            "runtime_image_id": None,
            "runtime_image_digest": None,
            "exit_code": None,
            "termination_reason": None,
            "restart_count": 0,
            "termination_message": None,
        }
    if len(statuses) != 1:
        raise JobsError("prod8 terminal probe Pod status shape changed")
    status = statuses[0]
    if status.get("name") != "terminal-probe" or not _bounded_int(status.get("restartCount")):
        raise JobsError("prod8 terminal probe container state changed")
    runtime_image_id = status.get("imageID")
    runtime_digest = _runtime_image_digest(runtime_image_id)
    if runtime_image_id is not None and (
        not isinstance(runtime_image_id, str) or runtime_digest is None
    ):
        raise JobsError("prod8 terminal probe runtime image is unbound")
    if runtime_digest is not None and runtime_digest != _image_digest(IMAGE, "expected image"):
        raise JobsError("prod8 terminal probe runtime image changed")
    state = _mapping(status.get("state"), "container state")
    terminated = state.get("terminated")
    exit_code: int | None = None
    reason: str | None = None
    message: str | None = None
    if terminated is not None:
        terminal = _mapping(terminated, "container termination")
        if not _bounded_int(terminal.get("exitCode"), upper=255):
            raise JobsError("prod8 terminal probe exit code changed")
        exit_code = terminal["exitCode"]
        if terminal.get("reason") is not None and not isinstance(terminal.get("reason"), str):
            raise JobsError("prod8 terminal probe termination reason changed")
        if terminal.get("message") is not None and not isinstance(terminal.get("message"), str):
            raise JobsError("prod8 terminal probe termination message changed")
        reason = terminal.get("reason")
        message = terminal.get("message")
    return {
        "pod_name": metadata.get("name"),
        "pod_uid": metadata.get("uid"),
        "container_name": "terminal-probe",
        "requested_image": IMAGE,
        "requested_image_digest": _image_digest(IMAGE, "expected image"),
        "runtime_image_id": runtime_image_id,
        "runtime_image_digest": runtime_digest,
        "exit_code": exit_code,
        "termination_reason": reason,
        "restart_count": status["restartCount"],
        "termination_message": message,
    }


def preview_evidence(rendered: dict[str, Any], context: str) -> dict[str, Any]:
    expected = manifest()
    direct.validate_cpu_preview(expected, rendered, context=context, purpose="data_stage")
    if not _exactly_equal(normalized_manifest(rendered), expected_normalized_manifest()):
        raise JobsError("prod8 terminal probe preview changed")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "context": context,
            "name": NAME,
            "manifest_sha256": manifest_digest(),
            "normalized_manifest_sha256": digest(expected_normalized_manifest()),
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


def _bounded_int(value: object, *, upper: int = MAX_COUNT) -> bool:
    return type(value) is int and 0 <= value <= upper


def _rfc3339(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).tzinfo is UTC
    except ValueError:
        return False


def _empty_batch_inventory(*, accepted: bool) -> dict[str, Any]:
    return {
        "directory_present": False,
        "accepted": accepted,
        "batch_directories": 0,
        "markers": {
            name: {"count": 0, "direct_regular_count": 0, "indirect_count": 0}
            for name in BATCH_MARKERS
        },
    }


def _empty_checkpoint_inventory(*, accepted: bool) -> dict[str, Any]:
    return {
        "directory_present": False,
        "accepted": accepted,
        "global_step_count": 0,
        "minimum_step": None,
        "maximum_step": None,
        "boundary_steps": [],
        "file_count": 0,
        "total_bytes": 0,
        "latest_pointer_step": None,
    }


def _scan_mutated_markers() -> list[dict[str, Any]]:
    return [
        {"name": name, "present": False, "accepted": False, "reason": "scan_mutated"}
        for name in ROOT_MARKERS
    ]


def _validate_root_markers(value: object, *, stable: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(ROOT_MARKERS):
        raise JobsError("prod8 terminal probe root-marker schema changed")
    rows: list[dict[str, Any]] = []
    for name, row in zip(ROOT_MARKERS, value, strict=True):
        if not isinstance(row, dict) or set(row) != {"name", "present", "accepted", "reason"}:
            raise JobsError("prod8 terminal probe root-marker fields changed")
        if row.get("name") != name or type(row.get("present")) is not bool:
            raise JobsError("prod8 terminal probe root-marker identity changed")
        if type(row.get("accepted")) is not bool or row.get("reason") not in MARKER_REASONS:
            raise JobsError("prod8 terminal probe root-marker state changed")
        present = row["present"]
        accepted = row["accepted"]
        reason = row["reason"]
        if not stable:
            if row != _scan_mutated_markers()[len(rows)]:
                raise JobsError("prod8 terminal probe preserved unstable marker evidence")
        elif not present:
            if accepted or reason != "absent":
                raise JobsError("prod8 terminal probe absent marker state changed")
        elif accepted:
            if reason != "accepted":
                raise JobsError("prod8 terminal probe accepted marker state changed")
        elif reason not in STABLE_REJECTED_MARKER_REASONS:
            raise JobsError("prod8 terminal probe rejected marker reason changed")
        rows.append(row)
    return rows


def _validate_batch_inventory(value: object, *, stable: bool) -> dict[str, Any]:
    required = {"directory_present", "accepted", "batch_directories", "markers"}
    if not isinstance(value, dict) or set(value) != required:
        raise JobsError("prod8 terminal probe batch-inventory schema changed")
    if not stable:
        if value != _empty_batch_inventory(accepted=False):
            raise JobsError("prod8 terminal probe preserved unstable batch evidence")
        return value
    directory_present = value.get("directory_present")
    accepted = value.get("accepted")
    count = value.get("batch_directories")
    markers = value.get("markers")
    if type(directory_present) is not bool or type(accepted) is not bool or not _bounded_int(count):
        raise JobsError("prod8 terminal probe batch-inventory values changed")
    if not isinstance(markers, dict) or set(markers) != set(BATCH_MARKERS):
        raise JobsError("prod8 terminal probe batch-marker schema changed")
    if not directory_present:
        if value != _empty_batch_inventory(accepted=True):
            raise JobsError("prod8 terminal probe absent batch-inventory changed")
        return value
    for name in BATCH_MARKERS:
        row = markers[name]
        if not isinstance(row, dict) or set(row) != {
            "count",
            "direct_regular_count",
            "indirect_count",
        }:
            raise JobsError("prod8 terminal probe batch-marker fields changed")
        if not all(_bounded_int(row[key]) for key in row):
            raise JobsError("prod8 terminal probe batch-marker count changed")
        if row["count"] != row["direct_regular_count"] + row["indirect_count"]:
            raise JobsError("prod8 terminal probe batch-marker totals conflict")
        if row["count"] > count:
            raise JobsError("prod8 terminal probe batch-marker count exceeds directories")
    if not accepted and (count or any(any(row.values()) for row in markers.values())):
        raise JobsError("prod8 terminal probe rejected batch inventory retained evidence")
    return value


def _validate_checkpoint_inventory(value: object, *, stable: bool) -> dict[str, Any]:
    required = {
        "directory_present",
        "accepted",
        "global_step_count",
        "minimum_step",
        "maximum_step",
        "boundary_steps",
        "file_count",
        "total_bytes",
        "latest_pointer_step",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise JobsError("prod8 terminal probe checkpoint-inventory schema changed")
    if not stable:
        if value != _empty_checkpoint_inventory(accepted=False):
            raise JobsError("prod8 terminal probe preserved unstable checkpoint evidence")
        return value
    directory_present = value.get("directory_present")
    accepted = value.get("accepted")
    if type(directory_present) is not bool or type(accepted) is not bool:
        raise JobsError("prod8 terminal probe checkpoint-inventory state changed")
    if not directory_present:
        if value != _empty_checkpoint_inventory(accepted=True):
            raise JobsError("prod8 terminal probe absent checkpoint-inventory changed")
        return value
    fields = ("global_step_count", "file_count")
    if not all(_bounded_int(value.get(key)) for key in fields) or not _bounded_int(
        value.get("total_bytes"), upper=MAX_TOTAL_BYTES
    ):
        raise JobsError("prod8 terminal probe checkpoint-inventory count changed")
    minimum, maximum = value.get("minimum_step"), value.get("maximum_step")
    steps = value.get("global_step_count")
    if steps == 0:
        if (
            minimum is not None
            or maximum is not None
            or value.get("boundary_steps") != []
            or value.get("latest_pointer_step") is not None
        ):
            raise JobsError("prod8 terminal probe empty checkpoint inventory conflicts")
    else:
        if not _bounded_int(minimum) or not _bounded_int(maximum) or minimum > maximum:
            raise JobsError("prod8 terminal probe checkpoint bounds changed")
        boundary = value.get("boundary_steps")
        if (
            not isinstance(boundary, list)
            or not 1 <= len(boundary) <= 4
            or not all(_bounded_int(item) for item in boundary)
            or boundary != sorted(set(boundary))
            or boundary[0] != minimum
            or boundary[-1] != maximum
        ):
            raise JobsError("prod8 terminal probe checkpoint boundary evidence changed")
    pointer = value.get("latest_pointer_step")
    if pointer is not None and not _bounded_int(pointer):
        raise JobsError("prod8 terminal probe checkpoint pointer changed")
    if pointer is not None and steps and not minimum <= pointer <= maximum:
        raise JobsError("prod8 terminal probe checkpoint pointer conflicts")
    if not accepted and any(
        value[key]
        for key in (
            "global_step_count",
            "file_count",
            "total_bytes",
            "boundary_steps",
            "latest_pointer_step",
        )
    ):
        raise JobsError("prod8 terminal probe rejected checkpoint inventory retained evidence")
    return value


def _terminal_classification(
    *, root_exists: bool, root_direct: bool, stable: bool, markers: list[dict[str, Any]]
) -> str:
    if not stable:
        return "unaccepted_scan_mutation"
    if not root_exists:
        return "root_missing"
    if not root_direct:
        return "unaccepted_root"
    by_name = {row["name"]: row for row in markers}
    for name, accepted, rejected in (
        ("FAILED.json", "failed", "unaccepted_failed_marker"),
        ("REJECTED.json", "rejected", "unaccepted_rejected_marker"),
        ("NATIVE_TRAINING_COMPLETE.json", "native_training_complete", "unaccepted_complete_marker"),
    ):
        row = by_name[name]
        if row["present"]:
            return accepted if row["accepted"] else rejected
    return "no_terminal_marker"


def validate_receipt(value: object, *, target: str = TARGET) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "target",
        "training_plan_sha256",
        "checked_at",
        "gpus",
        "sfs_mount_read_only",
        "private_payloads_read",
        "root_exists",
        "root_direct",
        "scan",
        "terminal_classification",
        "root_markers",
        "batch_inventory",
        "checkpoint_inventory",
        "receipt_size_limit_bytes",
        "receipt_size_bytes",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise JobsError("prod8 terminal probe receipt schema changed")
    try:
        sealed = _seal(value)
    except (TypeError, ValueError, RecursionError):
        raise JobsError("prod8 terminal probe receipt cannot be sealed") from None
    if value != sealed:
        raise JobsError("prod8 terminal probe receipt integrity changed")
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("target") != target
        or not isinstance(value.get("target"), str)
        or value.get("training_plan_sha256") != TRAINING_PLAN_SHA256
        or not _rfc3339(value.get("checked_at"))
        or value.get("gpus") != 0
        or type(value.get("gpus")) is not int
        or value.get("sfs_mount_read_only") is not True
        or value.get("private_payloads_read") is not False
        or value.get("receipt_size_limit_bytes") != MAX_RECEIPT_BYTES
        or type(value.get("receipt_size_limit_bytes")) is not int
        or not _bounded_int(value.get("receipt_size_bytes"), upper=MAX_RECEIPT_BYTES)
        or value["receipt_size_bytes"]
        != len((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        or not isinstance(value.get("sha256"), str)
        or _DIGEST.fullmatch(value["sha256"]) is None
    ):
        raise JobsError("prod8 terminal probe receipt fields changed")
    scan = value.get("scan")
    if not isinstance(scan, dict) or set(scan) != {"attempts", "stable"}:
        raise JobsError("prod8 terminal probe scan schema changed")
    attempts, stable = scan.get("attempts"), scan.get("stable")
    if (
        not _bounded_int(attempts, upper=MAX_SCAN_ATTEMPTS)
        or attempts < 1
        or type(stable) is not bool
    ):
        raise JobsError("prod8 terminal probe scan values changed")
    if (stable and value.get("status") != "inspected") or (
        not stable and (value.get("status") != "unaccepted" or attempts != MAX_SCAN_ATTEMPTS)
    ):
        raise JobsError("prod8 terminal probe scan status changed")
    root_exists, root_direct = value.get("root_exists"), value.get("root_direct")
    if (
        type(root_exists) is not bool
        or type(root_direct) is not bool
        or (root_direct and not root_exists)
    ):
        raise JobsError("prod8 terminal probe root state changed")
    markers = _validate_root_markers(value.get("root_markers"), stable=stable)
    _validate_batch_inventory(value.get("batch_inventory"), stable=stable)
    _validate_checkpoint_inventory(value.get("checkpoint_inventory"), stable=stable)
    classification = _terminal_classification(
        root_exists=root_exists,
        root_direct=root_direct,
        stable=stable,
        markers=markers,
    )
    if value.get("terminal_classification") not in TERMINAL_CLASSIFICATIONS or (
        value["terminal_classification"] != classification
    ):
        raise JobsError("prod8 terminal probe classification conflicts with evidence")
    return value


def receipt_execution_accepted(value: object, *, target: str = TARGET) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        validate_receipt(value, target=target)
    except JobsError:
        return False
    return value["status"] == "inspected"


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
        value = validate_receipt(parse_receipt_json(args.validate_receipt.read_bytes()))
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
