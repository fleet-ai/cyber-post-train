#!/usr/bin/env python3
"""Build and verify the one-shot read-only prod8 SFS absence probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest
from training import skyrl_reward_rayjob as direct

NAME = "chris-q38-prod8-absence-v1"
DESTINATION = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod8-v1/data"
TRAINING_PLAN_SHA256 = "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
    "sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RECEIPT_SCHEMA = "cyber_qwen38_prod8_sfs_absence_receipt_v1"
PREVIEW_SCHEMA = "cyber_qwen38_prod8_sfs_absence_preview_v1"
RUNTIME = r'''import datetime,hashlib,json,os,pathlib,sys

def digest(value):
    payload=json.dumps(
        value,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()

destination=os.environ["PROBE_DESTINATION"]
absent=not os.path.lexists(destination)
body={
    "schema":os.environ["PROBE_RECEIPT_SCHEMA"],
    "status":"absent" if absent else "present",
    "destination":destination,
    "training_plan_sha256":os.environ["PROBE_TRAINING_PLAN_SHA256"],
    "checked_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpus":0,
    "sfs_mount_read_only":True,
    "task_rows_read":0,
    "rollout_episodes":0,
    "optimizer_steps":0,
    "checkpoints":0,
}
body["sha256"]="sha256:"+digest(body)
payload=json.dumps(body,sort_keys=True,separators=(",",":"))+"\n"
pathlib.Path("/dev/termination-log").write_text(payload)
print(json.dumps({"status":body["status"],"sha256":body["sha256"]},sort_keys=True))
sys.exit(0 if absent else 1)
'''


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
                            "name": "absence-probe",
                            "image": IMAGE,
                            "command": ["python", "-c", RUNTIME],
                            "env": [
                                {"name": "PROBE_DESTINATION", "value": DESTINATION},
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
    # Reuse the exact Job/default comparison used by the prod8 CPU gates. The
    # internal purpose label is discarded; this probe emits its own truthful evidence schema.
    direct.validate_cpu_preview(expected, rendered, context=context, purpose="data_stage")
    pod = rendered["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if (
        rendered["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION)
        != FAILURE_ALERT_OFF
        or pod.get("priorityClassName") != "c1"
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
        raise JobsError("prod8 absence probe preview changed")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "context": context,
            "name": NAME,
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "destination": DESTINATION,
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "sfs_mount_read_only": True,
            "submitted": False,
        }
    )


def validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value != _seal(value)
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") != "absent"
        or value.get("destination") != DESTINATION
        or value.get("training_plan_sha256") != TRAINING_PLAN_SHA256
        or value.get("gpus") != 0
        or value.get("sfs_mount_read_only") is not True
        or any(
            value.get(key) != 0
            for key in ("task_rows_read", "rollout_episodes", "optimizer_steps", "checkpoints")
        )
    ):
        raise JobsError("prod8 SFS absence receipt was not accepted")
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
