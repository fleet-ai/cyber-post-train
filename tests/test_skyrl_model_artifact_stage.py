"""Contract tests for the dev-only, zero-GPU model artifact stage."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError
from training import skyrl_model_artifact_stage as stage


@pytest.fixture
def plan():
    return stage.compile_stage(stage.CONFIG_PATH)


def test_stage_plan_is_dev_only_zero_science_and_create_once(plan) -> None:
    assert plan["schema"] == stage.PLAN_SCHEMA
    assert plan["name"] == "chris-q38-modelstage-v4"
    assert plan["execution"] == stage._expected_execution()
    assert plan["execution"]["cluster_target"] == "dev"
    assert plan["execution"]["priority"] == "c1"
    assert plan["execution"]["deadline_seconds"] == 1200
    assert plan["execution"]["artifact_path"] == (
        "fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4"
    )
    assert plan["model"]["repo"] == "Qwen/Qwen3.8-27B"
    assert plan["model"]["revision"] == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert len(plan["model"]["files"]) == 28
    assert set(plan["scientific_work"].values()) == {0}


def test_stage_job_is_zero_gpu_single_models_mount_and_explicit_user(plan) -> None:
    manifest = stage.job_manifest(plan)
    assert manifest["metadata"] == {
        "name": "chris-q38-modelstage-v4",
        "namespace": "fleet-train-jobs",
    }
    assert manifest["spec"]["activeDeadlineSeconds"] == 1200
    assert manifest["spec"]["backoffLimit"] == 0
    pod = manifest["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["priorityClassName"] == "c1"
    assert pod["restartPolicy"] == "Never"
    container = pod["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "privileged": False,
        "runAsGroup": 100,
        "runAsNonRoot": True,
        "runAsUser": 1000,
    }
    mounts = {row["name"]: row for row in container["volumeMounts"]}
    assert mounts["models"] == {
        "name": "models",
        "mountPath": "/mnt/models",
        "subPath": "models",
    }
    assert pod["volumes"] == [
        {
            "name": "models",
            "persistentVolumeClaim": {"claimName": "sfs-shared"},
        },
        {"name": "runtime", "emptyDir": {}},
    ]


def _server_render(manifest: dict) -> dict:
    rendered = copy.deepcopy(manifest)
    uid = "00000000-0000-0000-0000-000000000001"
    name = manifest["metadata"]["name"]
    labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "generation": 1,
            "labels": labels,
            "uid": uid,
        }
    )
    rendered["status"] = {}
    rendered["spec"].update(
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
    template = rendered["spec"]["template"]
    template["metadata"]["labels"] = labels
    template["spec"].update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "securityContext": {},
            "terminationGracePeriodSeconds": 30,
        }
    )
    template["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    return rendered


def test_stage_server_preview_accepts_only_exact_defaults(plan) -> None:
    manifest = stage.job_manifest(plan)
    proof = stage.validate_preview(plan, manifest, _server_render(manifest))
    assert proof["schema"] == stage.PREVIEW_SCHEMA
    assert proof["gpus"] == 0
    assert proof["runtime_user"] == {"uid": 1000, "gid": 100}
    changed = _server_render(manifest)
    changed["spec"]["template"]["spec"]["containers"][0]["securityContext"][
        "runAsUser"
    ] = 0
    with pytest.raises(JobsError, match="server dry-run changed"):
        stage.validate_preview(plan, manifest, changed)


def _small_plan(plan: dict, tmp_path: Path, monkeypatch) -> tuple[dict, Path, Path]:
    source = tmp_path / "models" / "source"
    project = tmp_path / "models" / "fleetjob-dev"
    source.mkdir(parents=True)
    project.mkdir()
    payload = b"exact-model-byte"
    (source / "tokenizer.json").write_bytes(payload)
    value = copy.deepcopy(plan)
    execution = copy.deepcopy(value["execution"])
    execution.update(
        {
            "models_mount": str(tmp_path / "models"),
            "source_alias": "source",
            "artifact_path": "fleetjob-dev/artifact",
        }
    )
    value["execution"] = execution
    value["source_root"] = str(source)
    value["artifact_root"] = str(project / "artifact")
    value["model"] = {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "root": str(source),
        "files": [
            {
                "path": "tokenizer.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(stage, "_expected_execution", lambda: execution)
    monkeypatch.setattr(stage.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(stage.os, "getegid", lambda: 100)

    def rename_noreplace(first: Path, second: Path) -> None:
        if second.exists():
            raise stage.StageGateError("artifact_destination_exists")
        first.rename(second)

    monkeypatch.setattr(stage, "_rename_noreplace", rename_noreplace)
    return value, source, project / "artifact"


def test_stage_publishes_copied_exact_inventory_atomically(
    plan, tmp_path: Path, monkeypatch
) -> None:
    value, source, artifact = _small_plan(plan, tmp_path, monkeypatch)
    receipt = stage.stage(value)
    assert receipt["schema"] == stage.RECEIPT_SCHEMA
    assert receipt["status"] == "published"
    assert receipt["files"] == receipt["copies_verified"] == 1
    assert receipt["optimizer_steps"] == receipt["rollout_episodes"] == 0
    assert artifact.is_dir() and not artifact.is_symlink()
    assert not os.path.samefile(source / "tokenizer.json", artifact / "tokenizer.json")
    assert (source / "tokenizer.json").read_bytes() == (
        artifact / "tokenizer.json"
    ).read_bytes()
    saved = json.loads((artifact / ".CYBER_ARTIFACT.json").read_text())
    assert saved == receipt
    assert not any(path.name.startswith(".artifact.tmp-") for path in artifact.parent.iterdir())


def test_stage_rejects_existing_destination_without_changing_it(
    plan, tmp_path: Path, monkeypatch
) -> None:
    value, _, artifact = _small_plan(plan, tmp_path, monkeypatch)
    artifact.mkdir()
    sentinel = artifact / "owned"
    sentinel.write_text("peer")
    with pytest.raises(stage.StageGateError, match="artifact_destination_exists"):
        stage.stage(value)
    assert sentinel.read_text() == "peer"


def test_stage_rejects_digest_drift_before_creating_temp(
    plan, tmp_path: Path, monkeypatch
) -> None:
    value, source, artifact = _small_plan(plan, tmp_path, monkeypatch)
    (source / "tokenizer.json").write_text("changed")
    with pytest.raises(stage.StageGateError, match="source_file_digest_mismatch_00"):
        stage.stage(value)
    assert not artifact.exists()
    assert set(artifact.parent.iterdir()) == set()
