from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime

import pytest

from cyber_post_train.jobs import JobsError, digest
from scripts import probe_qwen38_prod8_stage_absence as probe
from training import skyrl_reward_rayjob as direct


def _rendered() -> dict:
    value = copy.deepcopy(probe.manifest())
    uid = "12345678-1234-4123-8123-123456789abc"
    labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": probe.NAME,
        "controller-uid": uid,
        "job-name": probe.NAME,
    }
    value["metadata"].update(
        {
            "creationTimestamp": "2026-09-21T12:00:00Z",
            "generation": 1,
            "labels": labels,
            "uid": uid,
        }
    )
    value["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
    )
    value["spec"]["template"]["metadata"]["labels"] = labels
    pod = value["spec"]["template"]["spec"]
    pod.update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    pod["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    value["status"] = {}
    return value


def _receipt() -> dict:
    value = {
        "schema": probe.RECEIPT_SCHEMA,
        "status": "absent",
        "destination": probe.DESTINATION,
        "training_plan_sha256": probe.TRAINING_PLAN_SHA256,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpus": 0,
        "sfs_mount_read_only": True,
        "task_rows_read": 0,
        "rollout_episodes": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
    }
    value["sha256"] = "sha256:" + digest(value)
    return value


def test_manifest_is_bounded_read_only_alert_safe_and_zero_gpu() -> None:
    value = probe.manifest()
    pod = value["spec"]["template"]["spec"]
    assert value["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert value["spec"]["activeDeadlineSeconds"] == 300
    assert value["spec"]["backoffLimit"] == 0
    assert pod["priorityClassName"] == "c1"
    assert pod["automountServiceAccountToken"] is False
    assert pod["containers"][0]["volumeMounts"][0]["readOnly"] is True
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert "nvidia.com/gpu" not in json.dumps(value, sort_keys=True)


def test_preview_and_receipt_are_exactly_validated() -> None:
    proof = probe.preview_evidence(_rendered(), direct.PROD_CONTEXT)
    assert proof["sfs_mount_read_only"] is True and proof["gpus"] == 0
    receipt = _receipt()
    assert probe.validate_receipt(receipt) == receipt
    for key, replacement in (
        ("status", "present"),
        ("sfs_mount_read_only", False),
        ("optimizer_steps", 1),
    ):
        changed = copy.deepcopy(receipt)
        changed[key] = replacement
        body = {name: item for name, item in changed.items() if name != "sha256"}
        changed["sha256"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        with pytest.raises(JobsError):
            probe.validate_receipt(changed)
