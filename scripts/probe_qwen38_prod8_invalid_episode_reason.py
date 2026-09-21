#!/usr/bin/env python3
"""Prepare a read-only, reason-code-only probe for the terminated prod8 run.

The normal terminal probe intentionally omits per-episode files because they
can contain private task evidence.  This separate, bounded diagnostic reads no
logs, prompts, tool calls, rewards, recordings, or traces.  It emits only a
predefined ``InvalidEpisode`` reason already deliberately sanitized by
``training.rl_episode._failure``.

It is a preview/receipt tool only.  This module never creates a Kubernetes
object or calls a Fleet API.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest
from training import skyrl_reward_rayjob as direct

NAME = "chris-q38-prod8-invalid-episode-reason-v1"
TARGET = "/mnt/sfs/jobs/chris-q38-rlreward-prod8"
TARGET_BINDING = "qwen38_prod8_terminal_evidence_v1"
TRAINING_PLAN_SHA256 = "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
    "sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RECEIPT_SCHEMA = "cyber_qwen38_prod8_invalid_episode_reason_probe_receipt_v1"
RENDER_VALIDATION_SCHEMA = "cyber_qwen38_prod8_invalid_episode_reason_probe_render_validation_v1"
SERVER_PREVIEW_SCHEMA = "cyber_qwen38_prod8_invalid_episode_reason_probe_server_preview_v1"
MAX_RECEIPT_BYTES = 2_048
_RFC3339 = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

# This is an exact copy of the static ``InvalidEpisode`` reason set that
# ``training.rl_episode._failure`` already permits into sanitized receipts.
# It is intentionally local: touching that runtime module would invalidate
# source-pinned historical launch packets. The regression test parses the
# runtime source and fails if either allowlist changes independently.
SAFE_REASON_CODES = (
    "bash_timeout_maximum_unresolved",
    "generation_finish_invalid",
    "generation_incomplete",
    "generation_incomplete_aborted",
    "generation_incomplete_context_full",
    "generation_incomplete_length",
    "generation_incomplete_nontext",
    "generation_invalid_json",
    "generation_transport_failure",
    "instance_release_unconfirmed",
    "non_text_tool_result",
    "response_budget_exhausted",
    "tool_error_status_missing",
    "tool_parser_contract_invalid",
    "tool_result_exceeds_budget",
    "tool_timeout_below_advertised_budget",
    "turn_budget_exhausted",
    "turn_response_budget_exhausted",
)


def _runtime() -> str:
    """Return the immutable zero-GPU probe program with its explicit allowlist."""
    allowed = json.dumps(SAFE_REASON_CODES, separators=(",", ":"))
    return f"""import datetime,hashlib,json,math,os,pathlib,re,stat

ROOT=pathlib.Path(os.environ["PROBE_TARGET"])
PLAN=os.environ["PROBE_TRAINING_PLAN_SHA256"]
SCHEMA=os.environ["PROBE_RECEIPT_SCHEMA"]
RECEIPT=pathlib.Path(os.environ.get("PROBE_RECEIPT_PATH","/dev/termination-log"))
SAFE_REASONS=frozenset({allowed})
MAX_NATIVE_FAILURE_BYTES=131072
MAX_FAILURE_BYTES=65536
MAX_BATCHES=8
MAX_EPISODES=8
MAX_RECEIPT_BYTES={MAX_RECEIPT_BYTES}
MAX_NATIVE_LIST_ITEMS=4096
MAX_LOCAL_FRAMES=20
MAX_FAILURE_CAUSES=16
MAX_FAILURE_FRAMES=10
MAX_TEXT_LENGTH=256
MISSING=object()
BATCH=re.compile(r"^[0-9a-f]{{24}}$")
EPISODE=re.compile(r"^episode-[0-7]$")
DIGEST=re.compile(r"^(?:sha256:)?[0-9a-f]{{64}}$")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def reject_duplicates(pairs):
    result={{}}
    for key,value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key]=value
    return result

def reject_constant(value):
    raise ValueError("non-finite JSON value")

def finite_float(value):
    parsed=float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed

def read_direct(parent,name,limit):
    try:
        fd=os.open(name,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0),dir_fd=parent)
    except FileNotFoundError:
        return MISSING
    except OSError:
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
        (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
        return None
    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError,ValueError,RecursionError,OverflowError):
        return None

def open_directory(parent,name):
    try:
        fd=os.open(name,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0),dir_fd=parent)
    except (FileNotFoundError,OSError):
        return None
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    return fd

def child_directories(parent,pattern,limit):
    rows=[]
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if not pattern.fullmatch(entry.name):
                    continue
                try:
                    info=os.stat(entry.name,dir_fd=parent,follow_symlinks=False)
                except OSError:
                    return None
                if stat.S_ISDIR(info.st_mode):
                    rows.append(entry.name)
    except OSError:
        return None
    rows=sorted(rows)
    return rows if len(rows)<=limit else None

def text(value,limit=MAX_TEXT_LENGTH):
    return isinstance(value,str) and 0<len(value)<=limit

def frame(value):
    return (
        isinstance(value,dict)
        and set(value)=={{"file","line","function"}}
        and text(value["file"])
        and type(value["line"]) is int
        and 0<value["line"]<=10000000
        and text(value["function"])
    )

def frames(value,limit):
    return isinstance(value,list) and len(value)<=limit and all(frame(item) for item in value)

def strings(value,limit):
    return isinstance(value,list) and len(value)<=limit and all(text(item) for item in value)

def native_failure_is_bound(value):
    if not isinstance(value,dict) or set(value)!={{"plan_sha256","causes","sha256"}}:
        return False
    if (
        value.get("plan_sha256")!=PLAN
        or not isinstance(value.get("causes"),list)
        or not 1<=len(value["causes"])<=8
        or not isinstance(value.get("sha256"),str)
        or DIGEST.fullmatch(value["sha256"]) is None
    ):
        return False
    invalid=False
    for cause in value["causes"]:
        expected={{"error_class","actor_init_failed","remote_error_classes","remote_frames","local_frames"}}
        if (
            not isinstance(cause,dict)
            or set(cause)!=expected
            or not text(cause["error_class"])
            or type(cause["actor_init_failed"]) is not bool
            or not strings(cause["remote_error_classes"],MAX_NATIVE_LIST_ITEMS)
            or not frames(cause["remote_frames"],MAX_NATIVE_LIST_ITEMS)
            or not frames(cause["local_frames"],MAX_LOCAL_FRAMES)
        ):
            return False
        if (
            cause.get("error_class")=="InvalidEpisode"
            or "InvalidEpisode" in cause.get("remote_error_classes",[])
        ):
            invalid=True
    unsigned={{key:item for key,item in value.items() if key!="sha256"}}
    expected_sha=sha(unsigned)
    if value["sha256"] not in (expected_sha,"sha256:"+expected_sha):
        return False
    return invalid

def safe_reasons(value):
    # This accepts only the exact local sanitized-cause shape written by
    # rl_episode._failure.  All other payload values are ignored and never
    # copied to the receipt.
    expected_top={{"error_type","elapsed_seconds","run_id","phase","causes"}}
    if (
        not isinstance(value,dict)
        or set(value)!=expected_top
        or not text(value.get("error_type"))
        or not text(value.get("run_id"))
        or not text(value.get("phase"))
        or not isinstance(value.get("causes"),list)
        or len(value["causes"])>MAX_FAILURE_CAUSES
    ):
        return None
    elapsed=value["elapsed_seconds"]
    if type(elapsed) is int:
        if not 0<=elapsed<=172800:
            return None
    elif type(elapsed) is float:
        if not math.isfinite(elapsed) or not 0<=elapsed<=172800:
            return None
    else:
        return None
    result=[]
    for cause in value["causes"]:
        if not isinstance(cause,dict):
            return None
        if "reason" not in cause:
            if (
                set(cause)!={{"error_type","frames"}}
                or not text(cause.get("error_type"))
                or not frames(cause.get("frames"),MAX_FAILURE_FRAMES)
            ):
                return None
            continue
        if (
            set(cause)!={{"error_type","frames","reason"}}
            or cause.get("error_type")!="InvalidEpisode"
            or not frames(cause.get("frames"),MAX_FAILURE_FRAMES)
        ):
            return None
        reason=cause["reason"]
        if not isinstance(reason,str) or reason not in SAFE_REASONS:
            return None
        result.append(reason)
    return result

def probe():
    now=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base={{
        "schema":SCHEMA,"target_binding":"{TARGET_BINDING}","training_plan_sha256":PLAN,"checked_at":now,
        "gpus":0,"sfs_mount_read_only":True,"raw_logs_or_trajectory_payloads_read":False,
        "sanitized_failure_receipts_read":True,"root_native_failure_bound":False,
        "failure_receipts_scanned":0,"deterministic_reason_code":None,
        "safe_reason_codes":[],
    }}
    try:
        root=os.open(ROOT,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
    except OSError:
        base["status"]="unaccepted"
        return base
    try:
        native=read_direct(root,"NATIVE_FAILURE.json",MAX_NATIVE_FAILURE_BYTES)
        if native is MISSING or not native_failure_is_bound(native):
            base["status"]="unaccepted"
            return base
        base["root_native_failure_bound"]=True
        episodes=open_directory(root,"episodes")
        if episodes is None:
            base["status"]="no_safe_reason"
            return base
        try:
            batches=open_directory(episodes,"batches")
            if batches is None:
                base["status"]="no_safe_reason"
                return base
            try:
                batch_names=child_directories(batches,BATCH,MAX_BATCHES)
                if batch_names is None:
                    base["status"]="unaccepted"
                    return base
                reasons=[]
                for batch_name in batch_names:
                    batch=open_directory(batches,batch_name)
                    if batch is None:
                        base["status"]="unaccepted"
                        return base
                    try:
                        episode_names=child_directories(batch,EPISODE,MAX_EPISODES)
                        if episode_names is None:
                            base["status"]="unaccepted"
                            return base
                        for episode_name in episode_names:
                            episode=open_directory(batch,episode_name)
                            if episode is None:
                                base["status"]="unaccepted"
                                return base
                            try:
                                failure=read_direct(episode,"failure.json",MAX_FAILURE_BYTES)
                            finally:
                                os.close(episode)
                            if failure is MISSING:
                                continue
                            if failure is None:
                                base["status"]="unaccepted"
                                return base
                            base["failure_receipts_scanned"]+=1
                            found=safe_reasons(failure)
                            if found is None:
                                base["status"]="unaccepted"
                                return base
                            reasons.extend(found)
                    finally:
                        os.close(batch)
            finally:
                os.close(batches)
        finally:
            os.close(episodes)
    finally:
        os.close(root)
    codes=sorted(set(reasons))
    base["safe_reason_codes"]=codes
    if len(codes)==1:
        base["status"]="classified"
        base["deterministic_reason_code"]=codes[0]
    elif len(codes)>1:
        base["status"]="ambiguous"
    else:
        base["status"]="no_safe_reason"
    return base

body=probe()
body["receipt_size_limit_bytes"]=MAX_RECEIPT_BYTES
while True:
    body.pop("sha256",None)
    body["sha256"]="sha256:"+sha(body)
    payload=json.dumps(body,sort_keys=True,separators=(",",":"))+"\\n"
    if body.get("receipt_size_bytes")==len(payload.encode()):
        break
    body["receipt_size_bytes"]=len(payload.encode())
if len(payload.encode())>MAX_RECEIPT_BYTES:
    raise RuntimeError("reason-code receipt exceeds bound")
RECEIPT.write_text(payload)
print(json.dumps({{"status":body["status"],"deterministic_reason_code":body["deterministic_reason_code"],"sha256":body["sha256"]}},sort_keys=True))
"""


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def manifest() -> dict[str, Any]:
    """Return the exact zero-GPU, read-only Job requiring a server preview."""
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
                            "name": "reason-code-probe",
                            "image": IMAGE,
                            "command": ["python", "-c", _runtime()],
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
                            "persistentVolumeClaim": {"claimName": direct.PVC, "readOnly": True},
                        }
                    ],
                },
            },
        },
    }


def preview_evidence(rendered: dict[str, Any], context: str) -> dict[str, Any]:
    """Validate a rendered Job; this alone is not server-origin proof."""
    expected = manifest()
    direct.validate_cpu_preview(
        expected, copy.deepcopy(rendered), context=context, purpose="data_stage"
    )
    pod = rendered["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if (
        rendered["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or pod.get("priorityClassName") != "c1"
        or container.get("volumeMounts")
        != [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}]
        or pod.get("volumes")
        != [{"name": "sfs", "persistentVolumeClaim": {"claimName": direct.PVC, "readOnly": True}}]
        or "nvidia.com/gpu" in json.dumps(rendered, sort_keys=True)
    ):
        raise JobsError("prod8 reason-code probe preview changed")
    return _seal(
        {
            "schema": RENDER_VALIDATION_SCHEMA,
            "status": "validated",
            "context": context,
            "name": NAME,
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "target_binding": TARGET_BINDING,
            "training_plan_sha256": TRAINING_PLAN_SHA256,
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "sfs_mount_read_only": True,
            "submitted": False,
        }
    )


def server_preview(*, context: str, runner: Any = subprocess.run) -> dict[str, Any]:
    """Run the sole non-mutating preview path and bind its validated render."""
    if context != direct.PROD_CONTEXT:
        raise JobsError("prod8 reason-code probe requires its exact production context")
    expected = manifest()
    rendered = direct.server_dry_run(expected, context=context, runner=runner)
    validated = preview_evidence(rendered, context)
    return _seal(
        {
            "schema": SERVER_PREVIEW_SCHEMA,
            "status": "passed",
            "proof_method": "kubectl_create_server_dry_run",
            "render_validation_sha256": validated["sha256"],
            "context": context,
            "name": NAME,
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "target_binding": TARGET_BINDING,
            "training_plan_sha256": TRAINING_PLAN_SHA256,
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "sfs_mount_read_only": True,
            "submitted": False,
        }
    )


def validate_receipt(value: object) -> dict[str, Any]:
    """Validate a sealed reason-code-only receipt without accepting raw evidence."""
    required = {
        "schema",
        "status",
        "target_binding",
        "training_plan_sha256",
        "checked_at",
        "gpus",
        "sfs_mount_read_only",
        "raw_logs_or_trajectory_payloads_read",
        "sanitized_failure_receipts_read",
        "root_native_failure_bound",
        "failure_receipts_scanned",
        "deterministic_reason_code",
        "safe_reason_codes",
        "receipt_size_limit_bytes",
        "receipt_size_bytes",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != required or value != _seal(value):
        raise JobsError("prod8 reason-code probe receipt integrity changed")
    statuses = {"classified", "ambiguous", "no_safe_reason", "unaccepted"}
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") not in statuses
        or value.get("target_binding") != TARGET_BINDING
        or value.get("training_plan_sha256") != TRAINING_PLAN_SHA256
        or value.get("gpus") != 0
        or value.get("sfs_mount_read_only") is not True
        or value.get("raw_logs_or_trajectory_payloads_read") is not False
        or value.get("sanitized_failure_receipts_read") is not True
        or type(value.get("root_native_failure_bound")) is not bool
        or type(value.get("failure_receipts_scanned")) is not int
        or not 0 <= value["failure_receipts_scanned"] <= 64
        or value.get("receipt_size_limit_bytes") != MAX_RECEIPT_BYTES
        or type(value.get("receipt_size_bytes")) is not int
        or not 0 <= value["receipt_size_bytes"] <= MAX_RECEIPT_BYTES
        or not isinstance(value.get("checked_at"), str)
        or _RFC3339.fullmatch(value["checked_at"]) is None
        or value["receipt_size_bytes"]
        != len((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
    ):
        raise JobsError("prod8 reason-code probe receipt fields changed")
    codes = value.get("safe_reason_codes")
    if (
        not isinstance(codes, list)
        or codes != sorted(set(codes))
        or any(code not in SAFE_REASON_CODES for code in codes)
    ):
        raise JobsError("prod8 reason-code probe receipt codes changed")
    single = value.get("deterministic_reason_code")
    if value["status"] == "classified":
        if not value["root_native_failure_bound"] or len(codes) != 1 or single != codes[0]:
            raise JobsError("prod8 reason-code classification changed")
    elif value["status"] == "ambiguous":
        if not value["root_native_failure_bound"] or len(codes) < 2 or single is not None:
            raise JobsError("prod8 reason-code ambiguity changed")
    elif codes or single is not None:
        # ``no_safe_reason`` and ``unaccepted`` communicate that no exact
        # action is justified; neither may smuggle out a partial code.
        raise JobsError("prod8 reason-code receipt exposed an unaccepted code")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--server-preview", action="store_true")
    parser.add_argument("--context")
    parser.add_argument("--validate-receipt", type=Path)
    args = parser.parse_args()
    selected = sum((args.manifest, args.server_preview, args.validate_receipt is not None))
    if selected != 1:
        parser.error("select exactly one operation")
    if args.manifest:
        value = manifest()
    elif args.server_preview:
        if args.context != direct.PROD_CONTEXT:
            parser.error("the exact production context is required")
        value = server_preview(context=args.context)
    else:
        value = validate_receipt(json.loads(args.validate_receipt.read_bytes()))
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
