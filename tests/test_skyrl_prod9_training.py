"""Fresh prod9 runtime-binding tests; synthetic only and no cluster calls."""

from __future__ import annotations

import base64
import copy
import gzip
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import JobsError, digest
from scripts import prepare_qwen38_skyrl_prod9_successor as prod9_prepare
from training import skyrl_prod9_direct as prod9_direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod9_rollout as rollout
from training import skyrl_prod9_training as prod9_training
from training import skyrl_reward_rayjob as historical_direct

pytest_plugins = ("test_skyrl_training",)
ROOT = Path(__file__).resolve().parents[1]


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_RUNTIME_BUNDLE")
    if encoded is None:
        parts = [
            (key, value)
            for key, value in request["env"].items()
            if key.startswith("CYBER_RUNTIME_BUNDLE_")
        ]
        encoded = "".join(
            value for _, value in sorted(parts, key=lambda item: int(item[0].rsplit("_", 1)[1]))
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def _prod9_plan() -> tuple[dict, dict, historical_direct.RailIdentity]:
    identity = historical_direct.load_identity(
        ROOT / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
    )
    manifest = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    manifest["name"] = identity.run_name
    manifest["sha256"] = "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    run = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v9.json").read_text()
    )
    plan, request = prod9_prepare._compile(run, manifest)
    return plan, request, identity


def _source_preview(plan: dict, request: dict) -> dict:
    """A strict synthetic Jobs API preview, with no cluster interaction."""
    placeholder = request["name"] + "-00000000"
    environment = [
        {"name": key, "value": value}
        for key, value in sorted(
            {
                **request["env"],
                "RUN_DIR": request["run_dir"],
                "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
                "FLEET_RUN_NAME": placeholder,
            }.items()
        )
    ]
    resources = request["resources"]
    value = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": prod9_direct.NAMESPACE,
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": request["name"],
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/job-image": request["image"],
                "fleet.ai/run-dir": plan["output_root"],
            },
        },
        "spec": {
            "entrypoint": request["command"],
            "submissionMode": "HTTPMode",
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {},
                        "spec": {
                            "priorityClassName": "c1",
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": request["image"],
                                    "env": environment,
                                    "envFrom": [
                                        {"secretRef": {"name": "fleet-api"}},
                                        {"secretRef": {"name": "wandb-api"}},
                                        {"secretRef": {"name": placeholder + "-fleet-key"}},
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": resources["cpu_request"],
                                            "memory": resources["memory_request"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                        "limits": {
                                            "cpu": resources["cpu_limit"],
                                            "memory": resources["memory_limit"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                    },
                                }
                            ],
                            "initContainers": [
                                {
                                    "name": "sfs-init",
                                    "command": [
                                        "sh",
                                        "-c",
                                        (
                                            f"mkdir -p {plan['output_root']} && chown 1000:100 "
                                            f"{plan['output_root']}"
                                        ),
                                    ],
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    return {"name": placeholder, "warnings": [], "manifest_yaml": yaml.safe_dump(value)}


def _direct_render(value: dict) -> dict:
    rendered = copy.deepcopy(value)
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-21T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    rendered["spec"]["ttlSecondsAfterFinished"] = 0
    rendered["spec"]["rayClusterSpec"]["headGroupSpec"].update(
        {"numOfHosts": 1, "scaleStrategy": {}}
    )
    return rendered


def test_prod9_request_bundles_fresh_entrypoint_and_historical_rail_rejects(prepared):
    plan = prod9_training.compile_rl(prepared.config, relative_to=prepared.state.tmp)
    request = prod9_training.job_request(plan)
    bundle = _bundle(request)

    assert plan["schema"] == prod9_training.SCHEMA
    assert plan["runtime_sha256"] == digest(prod9_training._runtime())
    assert plan["prod9_runtime"] == prod9_training._binding()
    assert bundle["module"] == prod9_training.MODULE
    assert bundle["files"]["training/skyrl_prod9_training.py"] == (
        Path(prod9_training.__file__).read_text()
    )
    assert bundle["files"]["training/skyrl_prod9_rollout.py"] == Path(rollout.__file__).read_text()
    assert "hardening.Recorder(" in bundle["files"]["training/skyrl_prod9_rollout.py"]

    # The old direct rail starts through its historical compiler.  It must be
    # explicitly unavailable instead of rendering a fresh plan as prod8.
    assert prod9_training.reject_historical_direct_rail(plan) == {
        "status": "rejected",
        "reason": "historical_direct_rail_cannot_render_fresh_prod9_runtime",
    }


def test_prod9_fresh_direct_renderer_and_cpu_preflight_are_alert_safe() -> None:
    plan, request, identity = _prod9_plan()
    preview = _source_preview(plan, request)
    rayjob = prod9_direct.manifest(plan, request, preview, identity=identity)
    packet = prod9_direct.packet(plan, request, preview, identity=identity)
    rendered = _direct_render(rayjob)
    proof = prod9_direct.validate_preview(
        plan,
        request,
        preview,
        rayjob,
        rendered,
        context=prod9_direct.PROD_CONTEXT,
        identity=identity,
    )
    cpu_job = prod9_direct.preflight_job_manifest(plan, identity=identity)
    container = cpu_job["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item["value"] for item in container["env"]}

    assert rayjob["metadata"]["annotations"] == {
        "fleet.ai/run-id": rayjob["metadata"]["labels"]["fleet.ai/run-id"],
        "fleet.ai/job-image": request["image"],
        "fleet.ai/run-dir": plan["output_root"],
        "fleet.ai/failure-alerts": "off",
    }
    assert packet["submitted"] is False
    assert packet["failure_alerts"] == proof["failure_alerts"] == "off"
    assert proof["nodes"] == 1 and proof["gpus"] == 8
    assert cpu_job["metadata"]["annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert "nvidia.com/gpu" not in json.dumps(cpu_job, sort_keys=True)
    assert _bundle({"env": environment})["module"] == prod9_training.MODULE
    assert _bundle({"env": environment})["argv"][-3:] == [
        "--preflight",
        "--receipt",
        "/dev/termination-log",
    ]
    assert prod9_direct.live_create_is_available() is False
    assert not hasattr(hardening, "create_once")

    census = build_capacity_census(
        {"items": []},
        {"items": []},
        owner_prefixes=hardening.PROJECT_OWNER_PREFIXES,
        max_nodes=hardening.PROJECT_MAX_NODES,
        max_gpus=hardening.PROJECT_MAX_GPUS,
        planned_nodes=1,
        planned_gpus=8,
        observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    capacity = hardening.capacity_gate(
        plan,
        request,
        rayjob,
        identity=identity,
        reader=lambda _context, **_kwargs: census,
    )
    assert capacity["planned"] == {"nodes": 1, "gpus": 8}

    unannotated = copy.deepcopy(rayjob)
    unannotated["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="root failure-alert annotation"):
        hardening.capacity_gate(
            plan,
            request,
            unannotated,
            identity=identity,
            reader=lambda _context, **_kwargs: pytest.fail(
                "missing root annotation must stop before a cluster census"
            ),
        )

    rendered["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="direct RayJob"):
        prod9_direct.validate_preview(
            plan,
            request,
            preview,
            rayjob,
            rendered,
            context=prod9_direct.PROD_CONTEXT,
            identity=identity,
        )


def test_prod9_preflight_replaces_the_preexisting_termination_file(
    tmp_path: Path, monkeypatch
) -> None:
    """A Kubernetes termination message is pre-created, not create-once."""
    target = tmp_path / "termination-log"
    target.write_text("stale\n")
    monkeypatch.setattr(prod9_training, "PREFLIGHT_RECEIPT", target)
    value = {"schema": "synthetic", "status": "passed", "gpus": 0}

    prod9_training._write_preflight_receipt(target, value)

    assert json.loads(target.read_text()) == {
        **value,
        "receipt_sha256": digest(value),
    }
    with pytest.raises(ValueError, match="receipt path"):
        prod9_training._write_preflight_receipt(tmp_path / "other", value)


@pytest.mark.asyncio
async def test_prod9_generator_constructs_fresh_recorder_and_stamps_receipt(
    tmp_path: Path, monkeypatch
):
    calls = []

    class MarkerRecorder:
        def __init__(self, *args):
            calls.append(args)

    @asynccontextmanager
    async def passthrough_engine(engine, _tokenizer, _timeout):
        yield engine

    async def collect(_config, _directory, recorder, _parse, *, client):
        assert isinstance(recorder, MarkerRecorder)
        assert client is not None
        return [
            NS(
                reward=1.0,
                tokens=[1, 2],
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                metadata={"done_reason": "report_submitted"},
            )
        ]

    monkeypatch.setenv("FLEET_API_KEY", "synthetic")
    monkeypatch.setattr(rollout.hardening, "Recorder", MarkerRecorder)
    monkeypatch.setattr(rollout.skyrl_episode, "single_attempt_engine", passthrough_engine)
    monkeypatch.setattr(rollout.skyrl_episode, "_module", lambda *_args: NS(__file__="/tmp/helper"))
    monkeypatch.setattr(rollout.rl_episode, "collect", collect)
    monkeypatch.setattr(rollout.rl_episode, "validate_samples", lambda _samples: None)

    generator = object.__new__(rollout.Generator)
    generator.busy = generator.failed = False
    generator.root = tmp_path
    generator.manifest = {"sha256": "sha256:synthetic"}
    generator.tokenizer = object()
    generator.engine = object()
    generator.response_tokens = 8
    generator.concurrency = 1
    config = {"run_id": "synthetic", "rl": {"episode_seconds": 1}}
    generator._inputs = lambda _batch: (
        {"phase": "train", "global_step": 0, "trajectory_ids": [("0", 0)]},
        "synthetic-batch",
        [config],
    )
    result = await generator.generate(
        {"sampling_params": {}, "trajectory_ids": [NS(instance_id="0", repetition_id=0)]}
    )

    assert len(calls) == 1
    assert result["rewards"] == [1.0]
    collected = json.loads((tmp_path / "batches/synthetic-batch/COLLECTED.json").read_text())
    assert collected["recorder_implementation"] == rollout.RECORDER_IMPLEMENTATION


def test_prod9_preflight_fails_before_model_or_network_when_not_image_user(prepared, monkeypatch):
    plan = prod9_training.compile_rl(prepared.config, relative_to=prepared.state.tmp)
    monkeypatch.setattr(prod9_training.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prod9_training.os, "getegid", lambda: 0)
    with pytest.raises(ValueError, match="image user"):
        prod9_training.preflight(plan)
