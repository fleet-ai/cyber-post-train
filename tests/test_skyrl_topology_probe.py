"""Offline contract tests for the zero-update development topology probe."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train import cli
from cyber_post_train.jobs import JobsError, digest
from training import skyrl_topology_probe as probe

ROOT = probe.ROOT
CONFIG = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v1.json"


@pytest.fixture
def plan():
    return probe.compile_probe(CONFIG)


def test_probe_is_distinct_dev_only_bounded_and_zero_update(plan) -> None:
    request = probe.request(plan)
    assert plan["schema"] == probe.SCHEMA
    assert plan["execution"]["cluster_target"] == "dev"
    assert plan["execution"]["jobs_api_base_url"] == "https://api.ft.dev.flt.build"
    assert (request["workers"], request["gpus_per_worker"]) == (1, 8)
    assert (plan["execution"]["workers"], plan["execution"]["gpus_per_worker"]) == (
        0,
        1,
    )
    assert plan["execution"]["max_workers"] == 1
    assert plan["execution"]["gpus_on_head"] == 8
    assert request["secrets"] == []
    assert plan["deadlines"] == {
        "setup_seconds": 1200,
        "cleanup_seconds": 300,
        "total_seconds": 1500,
    }
    assert plan["scientific_work"] == {
        "task_rows": 0,
        "rollout_episodes": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
    }
    gate = plan["qualification"]["submission_gate"]
    assert gate["cpu_preflight_authorized"] is True
    assert gate["preview_authorized"] is False
    assert gate["fleetjob_preview_authorized"] is True
    assert gate["submission_authorized"] is False
    assert request["env"]["CYBER_EXPECTED_RUNTIME_UID"] == "1000"
    assert request["env"]["CYBER_EXPECTED_RUNTIME_GID"] == "100"
    assert request["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert plan["model"]["repo"] == "Qwen/Qwen3.8-27B"
    assert plan["model"]["revision"] == ("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    assert plan["model"]["root"] == ("/mnt/sfs/jobs/chris-q38-skyrl-probe-v16/models/base")
    assert plan["execution"]["model_artifact"]["path"] == (
        "fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4"
    )
    assert (
        "models/" + plan["execution"]["model_artifact"]["path"]
        == plan["execution"]["preflight_model_pvc_subpath"]
    )


def test_probe_fleetjob_is_one_eight_gpu_pod_with_zero_replica_group(plan) -> None:
    manifest = probe.fleetjob_manifest(plan)
    spec = manifest["spec"]
    assert manifest["metadata"] == {
        "name": "chris-q38-skyrl-probe-v16",
        "namespace": "fleet-train-jobs",
    }
    assert spec["fleet"] == {
        "projectName": "fleetjob-dev",
        "auth": {"secretRef": {"name": "fleet-api", "key": "FLEET_API_KEY"}},
        "mountRoot": "/mnt/sfs/jobs/chris-q38-skyrl-probe-v16",
        "models": [
            {
                "path": "fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4",
                "mountPath": "base",
                "readOnly": True,
                "required": True,
            }
        ],
        "wandb": {"mode": "disabled"},
    }
    assert spec["kueue"] == {
        "queueName": "training-lq",
        "queuePriorityClass": "q1",
        "head": {
            "queuePriorityClass": "q1",
            "topology": {"mode": "unconstrained"},
        },
        "workerGroups": {
            "gpu": {
                "queuePriorityClass": "q1",
                "topology": {"mode": "unconstrained"},
            }
        },
    }
    cluster = spec["job"]["spec"]["rayClusterSpec"]
    assert spec["job"]["spec"]["activeDeadlineSeconds"] == 1800
    assert spec["job"]["spec"]["backoffLimit"] == 0
    assert spec["job"]["spec"]["shutdownAfterJobFinishes"] is True
    assert cluster["enableInTreeAutoscaling"] is True
    head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
    workers = cluster["workerGroupSpecs"]
    assert head["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert head["resources"]["requests"]["nvidia.com/gpu"] == "8"
    assert cluster["headGroupSpec"]["rayStartParams"]["num-gpus"] == "8"
    assert cluster["headGroupSpec"]["rayStartParams"]["num-cpus"] == "64"
    assert len(workers) == 1
    assert (workers[0]["replicas"], workers[0]["minReplicas"], workers[0]["maxReplicas"]) == (
        0,
        0,
        1,
    )
    gpu = workers[0]["template"]["spec"]["containers"][0]
    assert gpu["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert gpu["resources"]["requests"]["nvidia.com/gpu"] == "1"
    expected_user = {
        "allowPrivilegeEscalation": False,
        "privileged": False,
        "runAsGroup": 100,
        "runAsNonRoot": True,
        "runAsUser": 1000,
    }
    assert head["securityContext"] == gpu["securityContext"] == expected_user
    head_env = {row["name"]: row["value"] for row in head["env"]}
    assert head_env["RUN_DIR"] == plan["output_root"]
    assert head_env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert max(map(len, head_env.values())) <= 30000
    assert not any(row["name"].startswith("CYBER_RUNTIME_BUNDLE") for row in gpu["env"])
    assert head["terminationMessagePath"] == "/dev/termination-log"


def test_probe_cpu_preflight_is_zero_gpu_exact_mount_and_explicit_user(plan) -> None:
    manifest = probe.preflight_job_manifest(plan)
    assert manifest["kind"] == "Job"
    assert manifest["metadata"] == {
        "name": probe.PREFLIGHT_NAME,
        "namespace": "fleet-train-jobs",
    }
    spec = manifest["spec"]
    assert spec["activeDeadlineSeconds"] == 1200
    assert spec["backoffLimit"] == 0
    pod = spec["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
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
    assert mounts["model"] == {
        "name": "model",
        "mountPath": plan["model"]["root"],
        "readOnly": True,
        "subPath": "models/fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4",
    }
    assert mounts["output-registry"] == {
        "name": "output-registry",
        "mountPath": "/mnt/cyber-output-registry",
        "readOnly": True,
        "subPath": "models/fleetjob-dev",
    }
    assert mounts["output"]["mountPath"] == plan["output_root"]
    assert container["terminationMessagePath"] == "/dev/termination-log"
    expected = probe.request(plan, fleetjob_transport=True, cpu_preflight=True)
    assert container["command"] == ["/bin/sh", "-lc", "exec " + expected["command"]]


def test_probe_receipt_verifier_is_zero_gpu_read_only_and_explicit_user(plan) -> None:
    manifest = probe.receipt_verify_job_manifest(plan)
    assert manifest["metadata"] == {
        "name": probe.RECEIPT_VERIFY_NAME,
        "namespace": "fleet-train-jobs",
    }
    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert container["securityContext"]["runAsUser"] == 1000
    assert container["securityContext"]["runAsGroup"] == 100
    assert container["volumeMounts"] == [
        {
            "name": "output",
            "mountPath": plan["output_root"],
            "readOnly": True,
            "subPath": "models/fleetjob-dev/" + plan["run_name"],
        },
        {"name": "runtime", "mountPath": probe.RECEIPT_VERIFY_RUN_DIR},
    ]
    assert pod["volumes"][0]["persistentVolumeClaim"] == {
        "claimName": "sfs-shared",
        "readOnly": True,
    }
    assert pod["volumes"][1] == {"name": "runtime", "emptyDir": {}}
    environment = {row["name"]: row["value"] for row in container["env"]}
    assert environment["RUN_DIR"] == probe.RECEIPT_VERIFY_RUN_DIR
    expected = probe.request(plan, fleetjob_transport=True, receipt_verify=True)
    assert container["command"] == ["/bin/sh", "-lc", "exec " + expected["command"]]


def _render_preflight(manifest: dict) -> dict:
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


def test_probe_preflight_preview_accepts_only_exact_server_defaults(plan) -> None:
    manifest = probe.preflight_job_manifest(plan)
    rendered = _render_preflight(manifest)
    proof = probe.validate_preflight_job_preview(plan, manifest, rendered)
    assert proof["schema"] == probe.PREFLIGHT_PREVIEW_SCHEMA
    assert (proof["gpus"], proof["runtime_user"]) == (
        0,
        {"uid": 1000, "gid": 100},
    )
    environment = {
        row["name"]: row["value"]
        for row in rendered["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    changed = copy.deepcopy(rendered)
    changed["spec"]["template"]["spec"]["containers"][0]["securityContext"]["runAsUser"] = 0
    with pytest.raises(JobsError, match="changed"):
        probe.validate_preflight_job_preview(plan, manifest, changed)


def test_probe_receipt_verifier_preview_accepts_only_exact_server_defaults(plan) -> None:
    manifest = probe.receipt_verify_job_manifest(plan)
    rendered = _render_preflight(manifest)
    proof = probe.validate_receipt_verify_job_preview(plan, manifest, rendered)
    assert proof["schema"] == probe.RECEIPT_VERIFY_PREVIEW_SCHEMA
    assert proof["gpus"] == 0
    assert proof["runtime_user"] == {"uid": 1000, "gid": 100}
    changed = copy.deepcopy(rendered)
    changed["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]["readOnly"] = False
    with pytest.raises(JobsError, match="changed"):
        probe.validate_receipt_verify_job_preview(plan, manifest, changed)


def test_probe_relay_error_code_uses_only_sanitized_stage_and_category() -> None:
    class FleetVllmStartupError(Exception):
        def __init__(self):
            self.stage = "engine_core_start"
            self.sanitized_cause = {
                "schema": "fleet_vllm_startup_error_v1",
                "exception_class": "CudaOutOfMemoryError",
                "frame": None,
            }

    class RayTaskError(Exception):
        def as_instanceof_cause(self):
            return FleetVllmStartupError()

    assert probe._setup_failure_code(RayTaskError("private details")) == (
        "FleetVllmStartupError_engine_core_start_CudaOutOfMemoryError"
    )
    assert probe._setup_failure_code(RuntimeError("private details")) == "RuntimeError"


def test_probe_native_sampler_fallback_is_exact(monkeypatch) -> None:
    class TopKTopPSampler:
        def forward_native(self):
            return None

        forward = forward_native

    sampler = NS(current_platform=object(), TopKTopPSampler=TopKTopPSampler)
    modules = {
        "vllm.envs": NS(VLLM_USE_FLASHINFER_SAMPLER=False),
        "vllm.v1.sample.ops.topk_topp_sampler": sampler,
    }
    monkeypatch.setattr(probe.importlib, "import_module", modules.__getitem__)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    probe._validate_vllm_sampler_fallback_contract()
    assert type(sampler.current_platform) is object

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    with pytest.raises(ValueError, match="fallback environment"):
        probe._validate_vllm_sampler_fallback_contract()


def test_probe_fleetjob_preview_accepts_only_exact_server_mutation(plan) -> None:
    manifest = probe.fleetjob_manifest(plan)
    rendered = copy.deepcopy(manifest)
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "finalizers": ["fleet.ai/fleetjob-cleanup"],
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    proof = probe.validate_fleetjob_preview(plan, manifest, rendered)
    assert proof["schema"] == probe.FLEETJOB_PREVIEW_SCHEMA
    assert proof["runtime_user"] == {"uid": 1000, "gid": 100}
    changed = copy.deepcopy(rendered)
    changed["spec"]["job"]["spec"]["backoffLimit"] = 1
    with pytest.raises(JobsError, match="changed"):
        probe.validate_fleetjob_preview(plan, manifest, changed)


def test_probe_runtime_bundle_imports_without_source_checkout(plan, tmp_path) -> None:
    bundle = tmp_path / "bundle"
    for name, content in probe._runtime().items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for directory in ("training", "cyber_post_train", "evals", "evals/fleet"):
        (bundle / directory / "__init__.py").write_text("")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,pathlib; "
                "from training import skyrl_topology_probe as p; "
                "v=json.loads(pathlib.Path('plan.json').read_text()); "
                "p.request(v,fleetjob_transport=True,cpu_preflight=True)"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(bundle)},
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_target", "prod"),
        ("jobs_api_base_url", "https://api.ft.flt.build"),
        ("submission_transport", "jobs-api"),
        ("kubernetes_context", "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"),
        ("namespace", "default"),
        ("priority", "c0"),
        ("queue_priority", "q0"),
        ("workers", 2),
        ("max_workers", 2),
        ("gpus_per_worker", 8),
        ("gpus_on_head", 4),
    ],
)
def test_probe_rejects_topology_or_route_substitution(plan, field, value) -> None:
    plan["execution"][field] = value
    with pytest.raises(ValueError, match="plan changed"):
        probe.request(plan)


def test_probe_rejects_model_mount_or_head_resource_substitution(plan) -> None:
    for mutation in (
        lambda value: value["execution"]["model_artifact"].update(path="other/model"),
        lambda value: value["execution"]["model_artifact"].update(mount_path="other"),
        lambda value: value["execution"].update(preflight_model_pvc_subpath="models/other"),
        lambda value: value["execution"]["head_resources"].update(cpu_request="2"),
        lambda value: value.update(output_root="/mnt/sfs/jobs/other/models/run"),
    ):
        changed = copy.deepcopy(plan)
        mutation(changed)
        with pytest.raises(ValueError, match="plan changed"):
            probe.fleetjob_manifest(changed)


@pytest.mark.parametrize("field", ["repo", "revision", "weight_manifest_sha256"])
def test_probe_rejects_sealed_model_binding_substitution(plan, field) -> None:
    plan["model"][field] = "changed"
    with pytest.raises(ValueError, match="plan changed"):
        probe.request(plan)


def test_probe_rejects_sealed_model_file_manifest_substitution(plan) -> None:
    plan["model"]["files"][0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="plan changed"):
        probe.request(plan)


def test_probe_verifies_every_exact_model_file(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    payload = b"exact model bytes"
    (model / "weight.safetensors").write_bytes(payload)
    plan = {
        "model": {
            "root": str(model),
            "files": [
                {
                    "path": "weight.safetensors",
                    "size": len(payload),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
    }
    probe._verify_model(plan)
    (model / "weight.safetensors").write_bytes(payload + b"changed")
    with pytest.raises(probe.ProbeGateError, match="model_file_size_mismatch"):
        probe._verify_model(plan)


def test_probe_accepts_only_resolving_digest_valid_model_symlink(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = tmp_path / "blob"
    payload = b"exact external read-only blob"
    target.write_bytes(payload)
    (model / "weight.safetensors").symlink_to(target)
    plan = {
        "model": {
            "root": str(model),
            "files": [
                {
                    "path": "weight.safetensors",
                    "size": len(payload),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
    }
    probe._verify_model(plan)
    target.unlink()
    with pytest.raises(probe.ProbeGateError, match="model_file_broken_symlink_00"):
        probe._verify_model(plan)


def test_probe_preflight_parses_engine_without_tasks_or_gpu(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(probe.os, "getegid", lambda: 100)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = NS(generator=NS(inference_engine=object()))
    monkeypatch.setattr(probe.skyrl, "diagnostic_native_config", lambda _: cfg)
    monkeypatch.setattr(probe, "_verify_model", lambda _: None)
    monkeypatch.setattr(probe, "_validate_vllm_sampler_fallback_contract", lambda: None)
    monkeypatch.setattr(probe, "_validate_destination", lambda _: Path("/unused/run"))
    monkeypatch.setattr(
        probe, "_validate_create_once_absence", lambda _: Path("/unused/registry/run")
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=lambda value: calls.append(value)),
    )
    phases = []
    receipt = probe.preflight(plan, phases.append)
    assert receipt["schema"] == "cyber_skyrl_topology_probe_cpu_preflight_v1"
    assert receipt["request_sha256"] == digest(
        probe.request(plan, fleetjob_transport=True, cpu_preflight=True)
    )
    assert calls == [cfg]
    assert receipt["task_rows_read"] == 0
    assert receipt["rollout_episodes"] == receipt["optimizer_steps"] == 0
    assert receipt["create_once_output_absent"] is True
    assert phases == [
        "runtime_identity",
        "runtime_imports",
        "zero_gpu",
        "native_sampler_fallback",
        "plan_validation",
        "sealed_bootstrap_destination",
        "create_once_destination_absence",
        "model_inventory",
        "native_engine_arguments",
        "receipt",
    ]


def _write_probe_runtime(root: Path, plan: dict) -> None:
    files = {
        **probe._runtime(),
        "training/__init__.py": "",
        "evals/__init__.py": "",
        "evals/fleet/__init__.py": "",
        "cyber_post_train/__init__.py": "",
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
    }
    runtime = root / ".runtime"
    for name, content in files.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_probe_destination_requires_only_exact_sealed_bootstrap(plan, tmp_path) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    plan = copy.deepcopy(plan)
    plan["output_root"] = str(root)
    _write_probe_runtime(root, plan)
    assert probe._validate_destination(plan) == root
    (root / "occupied").write_text("evidence")
    with pytest.raises(FileExistsError, match="sealed bootstrap"):
        probe._validate_destination(plan)


def test_probe_destination_rejects_changed_bootstrap(plan, tmp_path) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    plan = copy.deepcopy(plan)
    plan["output_root"] = str(root)
    _write_probe_runtime(root, plan)
    (root / ".runtime" / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="digest-bound runtime"):
        probe._validate_destination(plan)


def test_probe_create_once_preflight_rejects_existing_sfs_output(
    plan, tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "registry"
    target = registry / plan["run_name"] / "models" / "run"
    registry.mkdir()
    monkeypatch.setenv("CYBER_CREATE_ONCE_ROOT", str(registry))
    changed = copy.deepcopy(plan)
    changed["execution"]["output_registry_mount"] = str(registry)
    assert probe._validate_create_once_absence(changed) == target
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        probe._validate_create_once_absence(changed)


def test_cpu_preflight_rejection_is_sealed_and_exits_cleanly(
    plan, tmp_path, monkeypatch, capsys
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    written = []
    monkeypatch.setattr(
        probe,
        "preflight",
        lambda *_: (_ for _ in ()).throw(probe.ProbeGateError("model_file_missing")),
    )
    monkeypatch.setattr(
        probe,
        "_write_receipt",
        lambda path, receipt, *, exclusive: written.append((path, receipt, exclusive)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            probe.MODULE,
            "--plan",
            str(plan_path),
            "--sha256",
            digest(plan),
            "--cpu-preflight",
            "--receipt",
            probe.PREFLIGHT_RECEIPT,
        ],
    )
    probe.main()
    assert len(written) == 1
    path, receipt, exclusive = written[0]
    assert path == Path(probe.PREFLIGHT_RECEIPT) and exclusive is False
    assert receipt["schema"] == probe.PREFLIGHT_FAILURE_SCHEMA
    assert receipt["status"] == "rejected"
    assert receipt["error_code"] == "model_file_missing"
    assert receipt == probe._seal(receipt)
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"


def _preview(request, *, identity=True):
    context = {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True} if identity else {}
    container = {
        "image": request["image"],
        "securityContext": {"privileged": False, **context},
        "resources": {
            "requests": {
                "cpu": request["resources"]["cpu_request"],
                "memory": request["resources"]["memory_request"],
                "nvidia.com/gpu": 8,
            },
            "limits": {
                "cpu": request["resources"]["cpu_limit"],
                "memory": request["resources"]["memory_limit"],
                "nvidia.com/gpu": 8,
            },
        },
        "env": [
            {"name": key, "value": value}
            for key, value in {**request["env"], "RUN_DIR": request["run_dir"]}.items()
        ],
    }
    template = {
        "spec": {
            "priorityClassName": "c1",
            "containers": [container],
            "imagePullSecrets": [],
        }
    }
    manifest = {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {"fleet.ai/run-dir": request["run_dir"]},
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "entrypoint": request["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": template},
                "workerGroupSpecs": [],
            },
        },
    }
    return {"manifest_yaml": yaml.safe_dump(manifest), "warnings": []}


def test_probe_preview_requires_effective_uid_gid(plan) -> None:
    request = probe.request(plan)
    result = probe.validate_preview(plan, request, _preview(request))
    assert result["runtime_user"] == {"uid": 1000, "gid": 100}
    assert result["shutdown_after_job_finishes"] is True
    assert result["cleanup_authority"] == "kuberay_plus_uid_bound_observer"
    assert result["receipt_authority"] == "durable_sfs_receipt_verifier"
    with pytest.raises(JobsError, match="does not prove runtime user"):
        probe.validate_preview(plan, request, _preview(request, identity=False))


def test_probe_preview_requires_automatic_cluster_cleanup(plan) -> None:
    request = probe.request(plan)
    preview = _preview(request)
    rendered = yaml.safe_load(preview["manifest_yaml"])
    rendered["spec"]["shutdownAfterJobFinishes"] = False
    preview["manifest_yaml"] = yaml.safe_dump(rendered)
    with pytest.raises(JobsError, match="release its Ray cluster"):
        probe.validate_preview(plan, request, preview)

    rendered["spec"].pop("shutdownAfterJobFinishes")
    preview["manifest_yaml"] = yaml.safe_dump(rendered)
    with pytest.raises(JobsError, match="malformed topology probe"):
        probe.validate_preview(plan, request, preview)


def test_probe_failure_receipt_is_durable_and_create_once(plan, tmp_path) -> None:
    changed = copy.deepcopy(plan)
    changed["output_root"] = str(tmp_path)
    failure = probe._seal(
        {
            "schema": probe.PROBE_FAILURE_SCHEMA,
            "status": "failed",
            "phase": "gpu_topology_probe",
            "error_class": "ProbeGateError",
            "error_code": "model_file_digest_mismatch_00",
            "plan_sha256": digest(changed),
            "task_rows_read": 0,
            "rollout_episodes": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        }
    )
    probe._persist_probe_failure(changed, failure)
    assert json.loads((tmp_path / probe.FAILURE_RECEIPT).read_bytes()) == failure
    with pytest.raises(FileExistsError):
        probe._persist_probe_failure(changed, failure)


def test_probe_durable_receipt_verifier_is_cpu_only_and_plan_bound(
    plan, tmp_path, monkeypatch
) -> None:
    changed = copy.deepcopy(plan)
    changed["output_root"] = str(tmp_path)
    accepted = probe._seal(
        {
            "schema": probe.RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(changed),
            **changed["scientific_work"],
            "engines_started": 2,
            "tensor_parallel_size": 4,
            "runtime_users": {
                "driver": {
                    "uid": 1000,
                    "gid": 100,
                    "physical_node": "gpu-node-1",
                },
                "gpu_pods": [{"uid": 1000, "gid": 100, "physical_node": "gpu-node-1"}],
            },
            "ray_shutdown_called": True,
            "external_release_required": True,
        }
    )
    (tmp_path / "TOPOLOGY_PROBE.json").write_text(json.dumps(accepted))
    monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(probe.os, "getegid", lambda: 100)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        probe,
        "_validate_probe_receipt",
        lambda value, receipt: None if value is changed and receipt == accepted else 1 / 0,
    )
    proof = probe.verify_durable_receipt(changed)
    assert proof["schema"] == probe.RECEIPT_VERIFY_SCHEMA
    assert proof["receipt_sha256"] == accepted["sha256"]
    assert proof["runtime_user"] == {"uid": 1000, "gid": 100}
    assert all(proof[key] == 0 for key in changed["scientific_work"])

    (tmp_path / probe.FAILURE_RECEIPT).write_text("{}")
    with pytest.raises(probe.ProbeGateError, match="durable_failure_receipt_present"):
        probe.verify_durable_receipt(changed)


def test_release_receipt_requires_exact_terminal_absence_and_zero_gpus(plan) -> None:
    receipt = probe._seal(
        {
            "schema": probe.RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(plan),
            **plan["scientific_work"],
            "engines_started": 2,
            "tensor_parallel_size": 4,
            "runtime_users": {
                "driver": {
                    "uid": 1000,
                    "gid": 100,
                    "physical_node": "gpu-node-1",
                },
                "gpu_pods": [
                    {"uid": 1000, "gid": 100, "physical_node": "gpu-node-1"},
                ],
            },
            "ray_shutdown_called": True,
            "external_release_required": True,
        }
    )
    observation = {
        "kubernetes_context": plan["execution"]["kubernetes_context"],
        "namespace": plan["execution"]["namespace"],
        "fleetjob_name": plan["run_name"],
        "job_id": "00000000-0000-0000-0000-000000000001",
        "fleetjob_uid": "00000000-0000-0000-0000-000000000002",
        "rayjob_uid": "00000000-0000-0000-0000-000000000003",
        "workload_uid": "00000000-0000-0000-0000-000000000004",
        "raycluster_uid": "00000000-0000-0000-0000-000000000005",
        "pod_uids": [
            "00000000-0000-0000-0000-000000000006",
        ],
        "terminal_status": "Succeeded",
        "fleetjob_present": False,
        "rayjob_present": False,
        "workload_present": False,
        "raycluster_present": False,
        "pods_present": False,
        "active_gpus": 0,
        "created_at": "2026-09-20T00:00:00Z",
        "deletion_requested_at": "2026-09-20T00:29:59Z",
        "release_observed_at": "2026-09-20T00:31:00Z",
    }
    release = probe.validate_release(plan, receipt, observation)
    assert release["schema"] == probe.RELEASE_SCHEMA and release["status"] == "released"
    for field, value in (
        ("pods_present", True),
        ("active_gpus", 8),
        ("terminal_status", "RUNNING"),
    ):
        changed = copy.deepcopy(observation)
        changed[field] = value
        with pytest.raises(ValueError, match="release was not proven"):
            probe.validate_release(plan, receipt, changed)
    late = copy.deepcopy(observation)
    late["deletion_requested_at"] = "2026-09-20T00:30:01Z"
    late["release_observed_at"] = "2026-09-20T00:31:00Z"
    with pytest.raises(ValueError, match="release was not proven"):
        probe.validate_release(plan, receipt, late)
    wrong_plan = copy.deepcopy(receipt)
    wrong_plan["plan_sha256"] = "0" * 64
    wrong_plan = probe._seal(wrong_plan)
    with pytest.raises(ValueError, match="exact accepted internal cleanup"):
        probe.validate_release(plan, wrong_plan, observation)


def test_probe_cli_prepares_but_external_gate_stays_closed(tmp_path) -> None:
    output = tmp_path / "prepared"
    result = cli.app.registered_commands  # prove command registration without network
    assert result
    from typer.testing import CliRunner

    response = CliRunner().invoke(
        cli.app, ["rl-topology-probe", str(CONFIG), "--output", str(output)]
    )
    assert response.exit_code == 0, response.output
    plan, _ = cli._prepared(output)
    assert (output / "fleetjob.json").exists()
    assert (output / "FLEETJOB_PREPARED.json").exists()
    assert (output / "preflight-job.json").exists()
    assert (output / "PREFLIGHT_JOB_PREPARED.json").exists()
    assert (output / "receipt-verify-job.json").exists()
    assert (output / "RECEIPT_VERIFY_JOB_PREPARED.json").exists()
    with pytest.raises(ValueError, match="preview blocked by qualification gate"):
        cli._external_action_gate(plan, "preview")
    with pytest.raises(ValueError, match="submit blocked by qualification gate"):
        cli._external_action_gate(plan, "submit")


def test_probe_cli_server_preview_is_bound_to_exact_dev_context(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    output = tmp_path / "prepared"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli.app, ["rl-topology-probe", str(CONFIG), "--output", str(output)]
        ).exit_code
        == 0
    )
    plan, _ = cli._prepared(output)
    rendered = probe.fleetjob_manifest(plan)
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "finalizers": ["fleet.ai/fleetjob-cleanup"],
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    seen = []

    def dry_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return NS(returncode=0, stdout=json.dumps(rendered), stderr="")

    monkeypatch.setattr(cli.subprocess, "run", dry_run)
    response = runner.invoke(cli.app, ["rl-topology-probe-preview", str(output)])
    assert response.exit_code == 0, response.output
    command, options = seen[0]
    assert command[:3] == [
        "kubectl",
        "--context",
        "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
    ]
    assert "--dry-run=server" in command and "create" in command
    assert "prod" not in " ".join(command)
    assert options["timeout"] == 60
    assert (output / "FLEETJOB_PREVIEW.json").exists()


def test_probe_cli_preflight_preview_is_bound_to_exact_dev_context(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    output = tmp_path / "prepared"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli.app, ["rl-topology-probe", str(CONFIG), "--output", str(output)]
        ).exit_code
        == 0
    )
    plan, _ = cli._prepared(output)
    rendered = _render_preflight(probe.preflight_job_manifest(plan))
    seen = []

    def dry_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return NS(returncode=0, stdout=json.dumps(rendered), stderr="")

    monkeypatch.setattr(cli.subprocess, "run", dry_run)
    response = runner.invoke(cli.app, ["rl-topology-probe-preflight-preview", str(output)])
    assert response.exit_code == 0, response.output
    command, options = seen[0]
    assert command[:3] == [
        "kubectl",
        "--context",
        "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
    ]
    assert "--dry-run=server" in command and "create" in command
    assert options["timeout"] == 60
    assert (output / "PREFLIGHT_JOB_PREVIEW.json").exists()


def test_probe_cli_receipt_preview_is_bound_to_exact_dev_context(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    output = tmp_path / "prepared"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli.app, ["rl-topology-probe", str(CONFIG), "--output", str(output)]
        ).exit_code
        == 0
    )
    plan, _ = cli._prepared(output)
    rendered = _render_preflight(probe.receipt_verify_job_manifest(plan))
    seen = []

    def dry_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return NS(returncode=0, stdout=json.dumps(rendered), stderr="")

    monkeypatch.setattr(cli.subprocess, "run", dry_run)
    response = runner.invoke(cli.app, ["rl-topology-probe-receipt-preview", str(output)])
    assert response.exit_code == 0, response.output
    command, options = seen[0]
    assert command[:3] == [
        "kubectl",
        "--context",
        "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
    ]
    assert "--dry-run=server" in command and "create" in command
    assert "prod" not in " ".join(command)
    assert options["timeout"] == 60
    assert (output / "RECEIPT_VERIFY_JOB_PREVIEW.json").exists()
