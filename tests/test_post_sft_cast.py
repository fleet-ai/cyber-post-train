import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from training import post_sft_cast as cast
from training.io import digest_json
from training.post_sft_artifacts import full_file_manifest, inspect_hf_export


def _fp32_export(root: Path, tensors: dict[str, torch.Tensor] | None = None) -> dict:
    root.mkdir(parents=True)
    tensors = tensors or {
        "alpha": torch.tensor([1.0, -2.25, 3.125], dtype=torch.float32),
        "omega": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    save_file(tensors, root / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * 4 for t in tensors.values())},
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in sorted(tensors)
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    for name in (
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (root / name).write_text(f"{name}\n")
    return tensors


def _execution(source: Path, destination: Path) -> dict:
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_plan_v1",
        "namespace": cast.NAMESPACE,
        "job_name": cast.JOB_NAME,
        "config_map_name": cast.CONFIG_MAP_NAME,
        "service_account_name": cast.SERVICE_ACCOUNT_NAME,
        "container_name": cast.CONTAINER_NAME,
        "image": cast.CAST_IMAGE,
        "image_digest": cast.CAST_IMAGE.rsplit("@", 1)[1],
        "command_sha256": cast.CAST_COMMAND_SHA256,
        "source_path": str(source),
        "destination_path": str(destination),
        "policy": cast.CAST_POLICY,
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "max_shard_bytes": cast.DEFAULT_MAX_SHARD_BYTES,
        "max_source_tensor_bytes": cast.DEFAULT_MAX_SOURCE_TENSOR_BYTES,
        "config_map_code_sha256": {
            path: "sha256:" + f"{index + 1:x}" * 64
            for index, path in enumerate(cast.MOUNTED_CODE_FILES.values())
        },
    }


def test_cast_is_deterministic_and_every_tensor_is_exact_direct_bf16(tmp_path: Path):
    source = tmp_path / "source"
    tensors = _fp32_export(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    proof = cast.cast_fp32_export(source, first, max_shard_bytes=24)
    repeated = cast.cast_fp32_export(source, second, max_shard_bytes=24)

    assert proof["all_destination_bits_equal_direct_bf16_cast"] is True
    assert proof["all_source_values_finite"] is True
    assert proof["cast_rows_sha256"] == repeated["cast_rows_sha256"]
    assert proof["destination_shard_count"] == 2
    assert full_file_manifest(first)["manifest_sha256"] == full_file_manifest(second)[
        "manifest_sha256"
    ]
    index = json.loads((first / "model.safetensors.index.json").read_text())
    for key, source_tensor in tensors.items():
        with safe_open(
            str(first / index["weight_map"][key]), framework="pt", device="cpu"
        ) as handle:
            actual = handle.get_tensor(key)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(
            actual.contiguous().view(torch.uint16),
            source_tensor.to(torch.bfloat16).contiguous().view(torch.uint16),
        )


def test_cast_fails_closed_on_nonfinite_mixed_dtype_and_memory_bound(tmp_path: Path):
    nonfinite = tmp_path / "nonfinite"
    _fp32_export(nonfinite, {"weight": torch.tensor([float("nan")], dtype=torch.float32)})
    with pytest.raises(ValueError, match="non-finite"):
        cast.cast_fp32_export(nonfinite, tmp_path / "nonfinite-out")

    mixed = tmp_path / "mixed"
    _fp32_export(mixed, {"weight": torch.ones(1, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="uniformly F32"):
        cast.cast_fp32_export(mixed, tmp_path / "mixed-out")

    bounded = tmp_path / "bounded"
    _fp32_export(bounded, {"weight": torch.ones(8, dtype=torch.float32)})
    with pytest.raises(ValueError, match="memory bound"):
        cast.cast_fp32_export(
            bounded, tmp_path / "bounded-out", max_source_tensor_bytes=16
        )

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="pre-existing"):
        cast.cast_fp32_export(bounded, dangling)


def test_cast_input_binds_raw_fp32_observation_and_manifest(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    tensors = _fp32_export(source)
    monkeypatch.setattr(cast, "SOURCE_PATH", source)
    monkeypatch.setattr(cast, "DESTINATION_PATH", destination)
    monkeypatch.setattr(cast, "EVIDENCE_DIR", tmp_path / "cast-evidence")
    manifest = full_file_manifest(source)
    inspection = inspect_hf_export(
        source,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_parameter_count=sum(t.numel() for t in tensors.values()),
        require_base_sidecars=False,
        expected_dtype="F32",
    )
    observation = {
        "schema": "fleet_sft_sfs_checkpoint_observation_v1",
        "output_inspection": inspection,
        "raw_export_full_manifest_sha256": manifest["manifest_sha256"],
        "full_file_manifest_sha256": "sha256:" + "1" * 64,
    }
    observation["observation_sha256"] = digest_json(observation)
    execution = _execution(source, destination)
    plan = {
        "cast_execution": execution,
        "base_model": {"parameter_count": sum(t.numel() for t in tensors.values())},
    }
    value = cast.build_cast_input(plan, observation, manifest)
    assert value["source"]["raw_inspection"]["dtype"] == "f32"
    assert value["cast_input_sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "cast_input_sha256"}
    )
    observation["output_inspection"]["dtype"] = "bf16"
    observation["observation_sha256"] = digest_json(
        {key: item for key, item in observation.items() if key != "observation_sha256"}
    )
    with pytest.raises(ValueError, match="raw F32"):
        cast.build_cast_input(plan, observation, manifest)


def test_execute_cast_is_atomic_recoverable_and_never_accepts_incomplete_output(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    tensors = _fp32_export(source)
    monkeypatch.setattr(cast, "SOURCE_PATH", source)
    monkeypatch.setattr(cast, "DESTINATION_PATH", destination)
    evidence_dir = tmp_path / "cast-evidence"
    monkeypatch.setattr(cast, "EVIDENCE_DIR", evidence_dir)
    manifest = full_file_manifest(source)
    inspection = inspect_hf_export(
        source,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_parameter_count=sum(t.numel() for t in tensors.values()),
        require_base_sidecars=False,
        expected_dtype="F32",
    )
    value = {
        "schema": cast.CAST_INPUT_SCHEMA,
        "source": {
            "path": str(source),
            "observation_sha256": "sha256:" + "1" * 64,
            "source_checkpoint_full_manifest_sha256": "sha256:" + "2" * 64,
            "raw_full_manifest": manifest,
            "raw_inspection": inspection,
        },
        "destination": {"path": str(destination), "must_be_absent": True},
        "execution": _execution(source, destination),
        "expected_parameter_count": sum(t.numel() for t in tensors.values()),
    }
    value["cast_input_sha256"] = digest_json(value)
    real_rename = cast._rename_noreplace
    monkeypatch.setattr(
        cast,
        "_runtime_execution_provenance",
        lambda _value: {
            "schema": "cyber_sft_fp32_to_bf16_cast_execution_v1",
            "image": cast.CAST_IMAGE,
            "cast_input_sha256": value["cast_input_sha256"],
        },
    )
    drifted = json.loads(json.dumps(value))
    drifted["execution"]["max_shard_bytes"] = 1024
    drifted["cast_input_sha256"] = digest_json(
        {key: item for key, item in drifted.items() if key != "cast_input_sha256"}
    )
    with pytest.raises(ValueError, match="max_shard_bytes differs"):
        cast.execute_cast(drifted)

    def crash(_source, _destination):
        raise OSError("crash before promotion")

    monkeypatch.setattr(cast, "_rename_noreplace", crash)
    with pytest.raises(OSError, match="crash"):
        cast.execute_cast(value)
    suffix = value["cast_input_sha256"][7:19]
    partial = destination.parent / f".partial-{destination.name}-{suffix}"
    assert (partial / cast.ACCEPTANCE_RECEIPT_NAME).is_file()
    assert not destination.exists()

    destination.mkdir()
    with pytest.raises(ValueError, match="both final and partial"):
        cast.execute_cast(value)
    destination.rmdir()

    monkeypatch.setattr(cast, "_rename_noreplace", real_rename)
    receipt = cast.execute_cast(value)
    assert receipt["destination"]["atomic_promotion"] is True
    assert receipt["conversion"]["all_destination_bits_equal_direct_bf16_cast"] is True
    assert destination.is_dir()
    assert json.loads((evidence_dir / "COMPLETE.json").read_text())["status"] == "COMPLETE"
    assert cast.execute_cast(value) == receipt

    destination.rename(tmp_path / "valid")
    destination.mkdir()
    with pytest.raises(ValueError, match="terminal acceptance"):
        cast.execute_cast(value)


def test_cast_runtime_provenance_binds_live_job_pod_image_and_configmap(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    code_data = {
        "training__init__.py": "# init\n",
        "training_io.py": "# io\n",
        "training_post_sft_artifacts.py": "# artifacts\n",
        "training_post_sft_cast.py": "# cast\n",
    }
    code_hashes = {
        relative: "sha256:" + hashlib.sha256(code_data[key].encode()).hexdigest()
        for key, relative in cast.MOUNTED_CODE_FILES.items()
    }
    execution = _execution(cast.SOURCE_PATH, cast.DESTINATION_PATH)
    execution["config_map_code_sha256"] = code_hashes
    value = {
        "schema": cast.CAST_INPUT_SCHEMA,
        "source": {"binding": "exact"},
        "destination": {"path": str(cast.DESTINATION_PATH), "must_be_absent": True},
        "execution": execution,
    }
    value["cast_input_sha256"] = digest_json(value)
    for key, relative in cast.MOUNTED_CODE_FILES.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code_data[key])
    (bundle / cast.CAST_INPUT_MOUNT_PATH).write_bytes(cast.canonical_cast_input_bytes(value))
    monkeypatch.setattr(cast, "BUNDLE_ROOT", bundle)
    for name, item in {
        "POD_NAMESPACE": cast.NAMESPACE,
        "POD_NAME": "cast-pod",
        "POD_UID": "pod-uid",
        "JOB_UID": "job-uid",
    }.items():
        monkeypatch.setenv(name, item)
    pod = {
        "metadata": {
            "name": "cast-pod",
            "uid": "pod-uid",
            "resourceVersion": "2",
            "ownerReferences": [
                {"kind": "Job", "name": cast.JOB_NAME, "uid": "job-uid"}
            ],
        },
        "spec": {
            "serviceAccountName": cast.SERVICE_ACCOUNT_NAME,
            "nodeSelector": {"workload": "fleetai-training-ng-cpu"},
            "containers": [
                {
                    "name": cast.CONTAINER_NAME,
                    "image": cast.CAST_IMAGE,
                    **cast.CAST_COMMAND,
                    "resources": {
                        "requests": {"cpu": "4", "memory": "32Gi"},
                        "limits": {"cpu": "8", "memory": "64Gi"},
                    },
                    "volumeMounts": [
                        {"name": "bundle", "mountPath": "/bundle", "readOnly": True},
                        {"name": "sfs", "mountPath": "/mnt/sfs"},
                    ],
                }
            ],
        },
        "status": {
            "containerStatuses": [
                {
                    "name": cast.CONTAINER_NAME,
                    "imageID": "containerd://" + cast.CAST_IMAGE.rsplit("@", 1)[1],
                }
            ]
        },
    }
    job = {
        "metadata": {
            "name": cast.JOB_NAME,
            "uid": "job-uid",
            "resourceVersion": "1",
            "labels": {"kueue.x-k8s.io/queue-name": "training-lq"},
        },
        "spec": {"suspend": False},
    }
    config_map = {
        "metadata": {
            "name": cast.CONFIG_MAP_NAME,
            "uid": "config-uid",
            "resourceVersion": "3",
        },
        "immutable": True,
        "data": {
            **code_data,
            cast.CAST_INPUT_MOUNT_PATH: cast.canonical_cast_input_bytes(value).decode(),
        },
    }

    def get(path):
        if "/pods/" in path:
            return pod
        if "/jobs/" in path:
            return job
        return config_map

    monkeypatch.setattr(cast, "_kubernetes_get", get)
    receipt = cast._runtime_execution_provenance(value)
    assert receipt["resolved_image_digest"] == cast.CAST_IMAGE.rsplit("@", 1)[1]
    assert receipt["config_map"]["immutable"] is True

    config_map["immutable"] = False
    with pytest.raises(ValueError, match="mutable"):
        cast._runtime_execution_provenance(value)


def test_cast_job_is_queued_cpu_only_digest_pinned_and_create_only():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    assert cast.validate_local_cast_bundle(plan, root) == plan["cast_execution"][
        "config_map_code_sha256"
    ]
    manifest_path = root / "evals/post_sft/cluster/qwen36-sft-bf16-cast-job.yaml"
    documents = list(yaml.safe_load_all(manifest_path.read_text()))
    job = next(value for value in documents if value.get("kind") == "Job")
    assert job["metadata"]["name"] == cast.JOB_NAME
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    assert job["spec"]["suspend"] is True
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == cast.CAST_IMAGE
    assert "nvidia.com/gpu" not in json.dumps(job)
    assert container["command"] == cast.CAST_COMMAND["command"]
    assert container["args"] == cast.CAST_COMMAND["args"]
    script = (root / "evals/post_sft/scripts/submit_bf16_cast.sh").read_text()
    assert "kubectl apply" not in script
    assert "kubectl create --dry-run=server" in script
    assert "kubectl create -f" in script
    assert 'value["immutable"]=True' in script
    assert "require_all_absent" in script
