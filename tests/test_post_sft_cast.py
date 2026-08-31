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
from training.post_sft_artifacts import (
    QWEN36_EXACT_MTP_OMISSION_KEYS,
    full_file_manifest,
    inspect_hf_export_with_exact_auxiliary_omission,
)


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


def _bf16_base(root: Path, trained: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = {key: value.to(torch.bfloat16) for key, value in trained.items()}
    tensors.update(
        {
            key: torch.tensor([index + 0.5], dtype=torch.bfloat16)
            for index, key in enumerate(QWEN36_EXACT_MTP_OMISSION_KEYS)
        }
    )
    root.mkdir(parents=True)
    save_file(tensors, root / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * 2 for t in tensors.values())},
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in sorted(tensors)
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return tensors


def _omission(trained: dict[str, torch.Tensor]) -> dict:
    trained_count = sum(value.numel() for value in trained.values())
    rows = [
        {"key": key, "shape": [1], "dtype": "BF16", "elements": 1}
        for key in QWEN36_EXACT_MTP_OMISSION_KEYS
    ]
    return {
        "schema": "cyber_sft_exact_auxiliary_head_omission_v1",
        "role": "speculative_draft_heads",
        "base_repository": "Qwen/Qwen3.6-27B",
        "base_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "base_weights_manifest_sha256": "sha256:" + "a" * 64,
        "serving_inference_effect": "inert_without_speculative_decoding",
        "serving_registration_sha256": "sha256:" + "b" * 64,
        "restoration_policy": "copy_exact_frozen_base_bf16_tensor_bits",
        "base_tensor_count": len(trained) + len(rows),
        "base_parameter_count": trained_count + len(rows),
        "raw_export_tensor_count": len(trained),
        "raw_export_parameter_count": trained_count,
        "missing_tensor_count": len(rows),
        "missing_parameter_count": len(rows),
        "tensors": rows,
    }


def _no_speculative(omission: dict) -> dict:
    return {
        "schema": "cyber_post_sft_no_speculative_decoding_proof_v1",
        "registration_sha256": omission["serving_registration_sha256"],
        "runtime_args_sha256": "sha256:" + "c" * 64,
        "prohibited_runtime_args": ["--speculative-algorithm"],
        "prohibited_runtime_args_absent": True,
        "no_speculative_or_draft_argument": True,
    }


def _execution(source: Path, destination: Path, base: Path | None = None) -> dict:
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_plan_v2",
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
        "base_model_path": str(base or cast.BASE_MODEL_PATH),
        "base_model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "policy": cast.CAST_POLICY,
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "max_shard_bytes": cast.DEFAULT_MAX_SHARD_BYTES,
        "max_source_tensor_bytes": cast.DEFAULT_MAX_SOURCE_TENSOR_BYTES,
        "trained_tensor_count": cast.TRAINED_TENSOR_COUNT,
        "trained_parameter_count": cast.TRAINED_PARAMETER_COUNT,
        "restored_auxiliary_tensor_count": cast.RESTORED_AUXILIARY_TENSOR_COUNT,
        "restored_auxiliary_parameter_count": cast.RESTORED_AUXILIARY_PARAMETER_COUNT,
        "final_tensor_count": cast.FINAL_TENSOR_COUNT,
        "final_parameter_count": cast.FINAL_PARAMETER_COUNT,
        "config_map_code_sha256": {
            path: "sha256:" + f"{index + 1:x}" * 64
            for index, path in enumerate(cast.MOUNTED_CODE_FILES.values())
        },
    }


def test_cast_is_deterministic_and_every_tensor_is_exact_direct_bf16(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    tensors = _fp32_export(source)
    base = tmp_path / "base"
    base_tensors = _bf16_base(base, tensors)
    omission = _omission(tensors)
    first = tmp_path / "first"
    second = tmp_path / "second"
    synced = []
    real_fsync_file = cast._fsync_file

    def record_fsync(path):
        synced.append(Path(path))
        real_fsync_file(path)

    monkeypatch.setattr(cast, "_fsync_file", record_fsync)
    proof = cast.cast_fp32_export(source, base, first, omission, max_shard_bytes=128)
    repeated = cast.cast_fp32_export(source, base, second, omission, max_shard_bytes=128)

    assert proof["all_destination_bits_equal_direct_bf16_cast"] is True
    assert proof["all_source_values_finite"] is True
    assert proof["cast_rows_sha256"] == repeated["cast_rows_sha256"]
    assert proof["all_restored_auxiliary_bits_equal_frozen_base"] is True
    assert proof["restored_auxiliary_tensor_count"] == 15
    assert full_file_manifest(first)["manifest_sha256"] == full_file_manifest(second)[
        "manifest_sha256"
    ]
    assert set(first.iterdir()).issubset(set(synced))
    assert set(second.iterdir()).issubset(set(synced))
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
    for key in QWEN36_EXACT_MTP_OMISSION_KEYS:
        with safe_open(
            str(first / index["weight_map"][key]), framework="pt", device="cpu"
        ) as handle:
            actual = handle.get_tensor(key)
        assert torch.equal(actual.view(torch.uint16), base_tensors[key].view(torch.uint16))


def test_cast_fails_closed_on_nonfinite_mixed_dtype_and_memory_bound(tmp_path: Path):
    nonfinite = tmp_path / "nonfinite"
    trained = _fp32_export(
        nonfinite, {"weight": torch.tensor([float("nan")], dtype=torch.float32)}
    )
    base = tmp_path / "nonfinite-base"
    _bf16_base(base, trained)
    with pytest.raises(ValueError, match="non-finite"):
        cast.cast_fp32_export(
            nonfinite, base, tmp_path / "nonfinite-out", _omission(trained)
        )

    mixed = tmp_path / "mixed"
    mixed_tensors = _fp32_export(mixed, {"weight": torch.ones(1, dtype=torch.bfloat16)})
    mixed_base = tmp_path / "mixed-base"
    _bf16_base(mixed_base, mixed_tensors)
    with pytest.raises(ValueError, match="uniformly F32|trained tensors are not uniformly F32"):
        cast.cast_fp32_export(
            mixed, mixed_base, tmp_path / "mixed-out", _omission(mixed_tensors)
        )

    bounded = tmp_path / "bounded"
    bounded_tensors = _fp32_export(bounded, {"weight": torch.ones(8, dtype=torch.float32)})
    bounded_base = tmp_path / "bounded-base"
    _bf16_base(bounded_base, bounded_tensors)
    with pytest.raises(ValueError, match="memory bound"):
        cast.cast_fp32_export(
            bounded,
            bounded_base,
            tmp_path / "bounded-out",
            _omission(bounded_tensors),
            max_source_tensor_bytes=16,
        )

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="pre-existing"):
        cast.cast_fp32_export(
            bounded, bounded_base, dangling, _omission(bounded_tensors)
        )


def test_cast_input_binds_raw_fp32_observation_and_manifest(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    base = tmp_path / "base"
    destination = tmp_path / "destination"
    tensors = _fp32_export(source)
    _bf16_base(base, tensors)
    omission = _omission(tensors)
    omission["base_weights_manifest_sha256"] = cast._weights_manifest_sha256(
        full_file_manifest(base)
    )
    monkeypatch.setattr(cast, "SOURCE_PATH", source)
    monkeypatch.setattr(cast, "BASE_MODEL_PATH", base)
    monkeypatch.setattr(cast, "BASE_MODEL_REPOSITORY", omission["base_repository"])
    monkeypatch.setattr(cast, "BASE_MODEL_REVISION", omission["base_revision"])
    monkeypatch.setattr(
        cast, "BASE_WEIGHTS_MANIFEST_SHA256", omission["base_weights_manifest_sha256"]
    )
    monkeypatch.setattr(cast, "DESTINATION_PATH", destination)
    monkeypatch.setattr(cast, "TRAINED_TENSOR_COUNT", len(tensors))
    monkeypatch.setattr(cast, "TRAINED_PARAMETER_COUNT", sum(t.numel() for t in tensors.values()))
    monkeypatch.setattr(cast, "RESTORED_AUXILIARY_TENSOR_COUNT", 15)
    monkeypatch.setattr(cast, "RESTORED_AUXILIARY_PARAMETER_COUNT", 15)
    monkeypatch.setattr(cast, "FINAL_TENSOR_COUNT", len(tensors) + 15)
    monkeypatch.setattr(
        cast, "FINAL_PARAMETER_COUNT", sum(t.numel() for t in tensors.values()) + 15
    )
    monkeypatch.setattr(cast, "EVIDENCE_DIR", tmp_path / "cast-evidence")
    manifest = full_file_manifest(source)
    inspection, layout = inspect_hf_export_with_exact_auxiliary_omission(
        source,
        base_root=base,
        omission=omission,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        require_base_sidecars=False,
    )
    observation = {
        "schema": "fleet_sft_sfs_checkpoint_observation_v1",
        "output_inspection": inspection,
        "raw_export_full_manifest_sha256": manifest["manifest_sha256"],
        "full_file_manifest_sha256": "sha256:" + "1" * 64,
        "weight_layout_exact_auxiliary_omission": layout,
        "no_speculative_decoding_proof": _no_speculative(omission),
    }
    observation["observation_sha256"] = digest_json(observation)
    execution = _execution(source, destination, base)
    plan = {
        "cast_execution": execution,
        "base_model": {
            "repository": omission["base_repository"],
            "revision": omission["base_revision"],
            "weights_manifest_sha256": omission["base_weights_manifest_sha256"],
            "parameter_count": omission["base_parameter_count"],
        },
        "export": {"raw_export_auxiliary_head_omission": omission},
        "serving": {
            "speculative_decoding": {
                "enabled": False,
                "proof": (
                    "exact_base_registration_contains_no_speculative_decoding_or_"
                    "draft_model_argument"
                ),
                "registration_sha256": omission["serving_registration_sha256"],
                "prohibited_runtime_args": ["--speculative-algorithm"],
            }
        },
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
    base = tmp_path / "base"
    destination = tmp_path / "destination"
    tensors = _fp32_export(source)
    _bf16_base(base, tensors)
    omission = _omission(tensors)
    omission["base_weights_manifest_sha256"] = cast._weights_manifest_sha256(
        full_file_manifest(base)
    )
    monkeypatch.setattr(cast, "SOURCE_PATH", source)
    monkeypatch.setattr(cast, "BASE_MODEL_PATH", base)
    monkeypatch.setattr(cast, "BASE_MODEL_REPOSITORY", omission["base_repository"])
    monkeypatch.setattr(cast, "BASE_MODEL_REVISION", omission["base_revision"])
    monkeypatch.setattr(
        cast, "BASE_WEIGHTS_MANIFEST_SHA256", omission["base_weights_manifest_sha256"]
    )
    monkeypatch.setattr(cast, "DESTINATION_PATH", destination)
    trained_parameters = sum(t.numel() for t in tensors.values())
    monkeypatch.setattr(cast, "TRAINED_TENSOR_COUNT", len(tensors))
    monkeypatch.setattr(cast, "TRAINED_PARAMETER_COUNT", trained_parameters)
    monkeypatch.setattr(cast, "RESTORED_AUXILIARY_TENSOR_COUNT", 15)
    monkeypatch.setattr(cast, "RESTORED_AUXILIARY_PARAMETER_COUNT", 15)
    monkeypatch.setattr(cast, "FINAL_TENSOR_COUNT", len(tensors) + 15)
    monkeypatch.setattr(cast, "FINAL_PARAMETER_COUNT", trained_parameters + 15)
    evidence_dir = tmp_path / "cast-evidence"
    monkeypatch.setattr(cast, "EVIDENCE_DIR", evidence_dir)
    manifest = full_file_manifest(source)
    inspection, layout = inspect_hf_export_with_exact_auxiliary_omission(
        source,
        base_root=base,
        omission=omission,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        require_base_sidecars=False,
    )
    value = {
        "schema": cast.CAST_INPUT_SCHEMA,
        "source": {
            "path": str(source),
            "observation_sha256": "sha256:" + "1" * 64,
            "source_checkpoint_full_manifest_sha256": "sha256:" + "2" * 64,
            "raw_full_manifest": manifest,
            "raw_inspection": inspection,
            "exact_auxiliary_omission": layout,
        },
        "frozen_base_auxiliary_source": {
            "path": str(base),
            "repository": omission["base_repository"],
            "revision": omission["base_revision"],
            "weights_manifest_sha256": omission["base_weights_manifest_sha256"],
            "omission_policy": omission,
                "speculative_decoding": {
                    "enabled": False,
                    "registration_sha256": omission["serving_registration_sha256"],
                    "prohibited_runtime_args": ["--speculative-algorithm"],
                },
                "no_speculative_decoding_proof": _no_speculative(omission),
        },
        "destination": {"path": str(destination), "must_be_absent": True},
        "execution": _execution(source, destination, base),
        "expected_trained_parameter_count": trained_parameters,
        "expected_final_parameter_count": trained_parameters + 15,
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

    wrong_base = json.loads(json.dumps(value))
    wrong_base_digest = "sha256:" + "c" * 64
    wrong_base["frozen_base_auxiliary_source"]["weights_manifest_sha256"] = (
        wrong_base_digest
    )
    wrong_base["cast_input_sha256"] = digest_json(
        {key: item for key, item in wrong_base.items() if key != "cast_input_sha256"}
    )
    monkeypatch.setattr(cast, "BASE_WEIGHTS_MANIFEST_SHA256", wrong_base_digest)
    with pytest.raises(ValueError, match="signed model lock"):
        cast.execute_cast(wrong_base)
    monkeypatch.setattr(
        cast, "BASE_WEIGHTS_MANIFEST_SHA256", omission["base_weights_manifest_sha256"]
    )

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
    manifest_path = root / "evals/post_sft/cluster/qwen36-sft-bf16-cast-v2-job.yaml"
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
    script = (root / "evals/post_sft/scripts/submit_bf16_cast_v2.sh").read_text()
    assert "kubectl apply" not in script
    assert "kubectl create --dry-run=server" in script
    assert "kubectl create -f" in script
    assert 'value["immutable"]=True' in script
    assert "require_all_absent" in script
    assert "temp_dir=$(mktemp -d)" in script
    assert 'cast_input="$temp_dir/cast-input.json"' in script
    assert "cast_input=$(mktemp)" not in script
