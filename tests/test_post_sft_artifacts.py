import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from training import post_sft_artifacts as artifacts
from training.io import digest_json
from training.post_sft_artifacts import (
    QWEN36_EXACT_MTP_OMISSION_KEYS,
    compare_model_config_architecture,
    compare_safetensor_layout,
    compare_safetensor_layout_with_exact_auxiliary_omission,
    full_file_manifest,
    inspect_hf_export,
    prove_speculative_decoding_disabled,
    structural_manifest,
    structural_tsv_sha256,
)


def test_pinned_registration_programmatically_disables_speculative_decoding():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    registration = json.loads(
        (
            root
            / "evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json"
        ).read_text()
    )
    proof = prove_speculative_decoding_disabled(registration, plan["serving"])
    assert proof["prohibited_runtime_args_absent"] is True
    tampered = json.loads(json.dumps(registration))
    tampered["spec"]["runtime"]["args"].extend(
        ["--speculative-algorithm", "EAGLE"]
    )
    with pytest.raises(ValueError, match="enables speculative"):
        prove_speculative_decoding_disabled(tampered, plan["serving"])


def test_sfs_evidence_runtime_binds_live_job_pod_image_and_immutable_bundle(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"
    plan = json.loads(plan_path.read_text())
    contract = plan["evidence_execution"]["sfs_export_inspector"]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    data = {}
    for key, relative in artifacts.EVIDENCE_MOUNTED_CODE_FILES.items():
        source = root / artifacts.EVIDENCE_LOCAL_CODE_FILES[key]
        value = source.read_text()
        data[key] = value
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value)
    plan_text = plan_path.read_text()
    registration_path = root / contract["registration_path"]
    registration_text = registration_path.read_text()
    data[artifacts.EVIDENCE_PLAN_MOUNT] = plan_text
    data[artifacts.EVIDENCE_REGISTRATION_MOUNT] = registration_text
    (bundle / artifacts.EVIDENCE_PLAN_MOUNT).write_text(plan_text)
    (bundle / artifacts.EVIDENCE_REGISTRATION_MOUNT).write_text(registration_text)
    monkeypatch.setattr(artifacts, "EVIDENCE_BUNDLE_ROOT", bundle)
    monkeypatch.setenv("POD_NAMESPACE", contract["namespace"])
    monkeypatch.setenv("POD_NAME", "evidence-pod")
    monkeypatch.setenv("POD_UID", "pod-uid")
    monkeypatch.setenv("JOB_UID", "job-uid")
    job = {
        "metadata": {
            "name": contract["job_name"],
            "uid": "job-uid",
            "resourceVersion": "11",
            "labels": contract["job_labels"],
        },
        "spec": {"suspend": True},
    }
    documents = list(
        yaml.safe_load_all(
            (
                root / "evals/post_sft/cluster/qwen36-sft-evidence-v4-job.yaml"
            ).read_text()
        )
    )
    job_document = next(value for value in documents if value.get("kind") == "Job")
    reviewed_container = job_document["spec"]["template"]["spec"]["containers"][0]
    pod_spec = {
        "serviceAccountName": contract["service_account_name"],
        "nodeSelector": contract["node_selector"],
        "containers": [
            {
                "name": contract["container_name"],
                "image": contract["image"],
                "resources": contract["resources"],
                "command": reviewed_container["command"],
                "args": reviewed_container["args"],
            }
        ],
    }
    pod = {
        "metadata": {
            "name": "evidence-pod",
            "uid": "pod-uid",
            "resourceVersion": "12",
            "labels": contract["pod_labels"],
            "ownerReferences": [
                {
                    "kind": "Job",
                    "name": contract["job_name"],
                    "uid": "job-uid",
                }
            ],
        },
        "spec": pod_spec,
        "status": {
            "containerStatuses": [
                {
                    "name": contract["container_name"],
                    "imageID": "containerd://" + contract["image_digest"],
                }
            ]
        },
    }
    config_map = {
        "metadata": {
            "name": contract["config_map_name"],
            "uid": "config-uid",
            "resourceVersion": "13",
        },
        "immutable": True,
        "data": data,
    }

    def get(path):
        if "/pods/" in path:
            return pod
        if "/jobs/" in path:
            return job
        return config_map

    monkeypatch.setattr(artifacts, "_kubernetes_get", get)
    result = artifacts.collect_sfs_evidence_runtime_provenance(plan)
    assert result["job"]["uid"] == "job-uid"
    assert result["config_map"]["immutable"] is True
    assert result["no_speculative_decoding_proof"][
        "no_speculative_or_draft_argument"
    ] is True


def _write_safetensors(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir()
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, root / name)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: name for key in sorted(tensors)}})
    )


def test_manifests_are_sorted_relative_and_content_sensitive(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "nested" / "a").write_bytes(b"alpha")
    structural = structural_manifest(tmp_path)
    full = full_file_manifest(tmp_path)
    assert [row["path"] for row in structural["files"]] == ["nested/a", "z"]
    assert [row["path"] for row in full["files"]] == ["nested/a", "z"]
    assert full["total_bytes"] == 6
    assert structural_tsv_sha256(tmp_path).startswith("sha256:")
    before = full["manifest_sha256"]
    (tmp_path / "z").write_bytes(b"changed")
    assert full_file_manifest(tmp_path)["manifest_sha256"] != before


def test_hf_inspection_binds_shards_tokenizer_config_and_parameter_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Slice:
        def get_shape(self):
            return [2, 3]

        def get_dtype(self):
            return "BF16"

    class SafeFile:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def keys(self):
            return ["weight"]

        def __iter__(self):
            return iter(self.keys())

        def get_slice(self, _key):
            return Slice()

    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: SafeFile())
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    rows = []
    for name in (
        "chat_template.jinja",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (tmp_path / name).write_text(name)
        from training.post_sft_artifacts import sha256_file

        rows.append({"path": name, "sha256": sha256_file(tmp_path / name).removeprefix("sha256:")})
    (tmp_path / "config.json").write_text("config")
    result = inspect_hf_export(
        tmp_path,
        expected_tokenizer_manifest_sha256=digest_json(sorted(rows, key=lambda row: row["path"])),
        expected_chat_template_sha256=next(
            "sha256:" + row["sha256"] for row in rows if row["path"] == "chat_template.jinja"
        ),
        expected_config_sha256=(
            "sha256:b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"
        ),
        expected_parameter_count=6,
    )
    assert result["shard_count"] == 1
    assert result["parameter_count"] == 6

    with pytest.raises(ValueError, match="parameter count"):
        inspect_hf_export(
            tmp_path,
            expected_tokenizer_manifest_sha256=result["tokenizer_manifest_sha256"],
            expected_chat_template_sha256=result["chat_template_sha256"],
            expected_config_sha256=result["config_sha256"],
            expected_parameter_count=7,
        )


def test_raw_hf_inspection_records_but_can_allow_sidecar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Slice:
        def get_shape(self):
            return [2, 3]

        def get_dtype(self):
            return "BF16"

    class SafeFile:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def __iter__(self):
            return iter(["weight"])

        def get_slice(self, _key):
            return Slice()

    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: SafeFile())
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("trainer-sidecar")
    (tmp_path / "config.json").write_text("trainer-config")
    result = inspect_hf_export(
        tmp_path,
        expected_tokenizer_manifest_sha256="sha256:" + "a" * 64,
        expected_chat_template_sha256="sha256:" + "b" * 64,
        expected_config_sha256="sha256:" + "c" * 64,
        expected_parameter_count=6,
        expected_sidecar_sha256={"config.json": "sha256:" + "d" * 64},
        require_base_sidecars=False,
    )
    assert result["base_sidecars_required"] is False
    assert result["config_matches_base"] is False
    assert result["sidecar_matches_base"] == {"config.json": False}


def test_safetensor_layout_compares_keys_shapes_and_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    for root in (base, candidate):
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}})
        )

    shapes = {str(base / "model.safetensors"): [2, 3], str(candidate / "model.safetensors"): [2, 3]}

    class Slice:
        def __init__(self, shape):
            self.shape = shape

        def get_shape(self):
            return self.shape

        def get_dtype(self):
            return "BF16"

    class SafeFile:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def __iter__(self):
            return iter(["weight"])

        def get_slice(self, _key):
            return Slice(shapes[self.name])

    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda name, **_k: SafeFile(name))
    assert compare_safetensor_layout(base, candidate)["all_keys_shapes_and_dtypes_match"]
    shapes[str(candidate / "model.safetensors")] = [3, 2]
    with pytest.raises(ValueError, match="layout differs"):
        compare_safetensor_layout(base, candidate)


def test_model_config_architecture_allows_only_declared_trainer_metadata(tmp_path: Path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    base.write_text(
        json.dumps(
            {
                "architectures": ["Qwen"],
                "text_config": {"hidden_size": 64, "dtype": "bfloat16"},
                "transformers_version": "4.57.1",
            }
        )
    )
    candidate.write_text(
        json.dumps(
            {
                "architectures": ["Qwen"],
                "text_config": {"hidden_size": 64, "dtype": "float32"},
                "transformers_version": "5.8.0",
            }
        )
    )
    result = compare_model_config_architecture(base, candidate)
    assert result["all_architecture_and_vocab_fields_identical"] is True
    changed = json.loads(candidate.read_text())
    changed["text_config"]["hidden_size"] = 128
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="architectural"):
        compare_model_config_architecture(base, candidate)


def test_exact_auxiliary_omission_accepts_only_all_15_frozen_mtp_tensors(tmp_path: Path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base_tensors = {"weight": torch.ones(2, dtype=torch.bfloat16)}
    base_tensors.update(
        {key: torch.ones(1, dtype=torch.bfloat16) for key in QWEN36_EXACT_MTP_OMISSION_KEYS}
    )
    _write_safetensors(base, base_tensors)
    _write_safetensors(candidate, {"weight": torch.ones(2, dtype=torch.float32)})
    rows = [
        {"key": key, "shape": [1], "dtype": "BF16", "elements": 1}
        for key in QWEN36_EXACT_MTP_OMISSION_KEYS
    ]
    omission = {
        "schema": "cyber_sft_exact_auxiliary_head_omission_v1",
        "role": "speculative_draft_heads",
        "serving_inference_effect": "inert_without_speculative_decoding",
        "restoration_policy": "copy_exact_frozen_base_bf16_tensor_bits",
        "base_tensor_count": 16,
        "base_parameter_count": 17,
        "raw_export_tensor_count": 1,
        "raw_export_parameter_count": 2,
        "missing_tensor_count": 15,
        "missing_parameter_count": 15,
        "tensors": rows,
    }
    result = compare_safetensor_layout_with_exact_auxiliary_omission(
        base, candidate, omission
    )
    assert result["exact_allowlist_match"] is True
    assert result["missing_parameter_count"] == 15

    missing_row = json.loads(json.dumps(omission))
    missing_row["tensors"].pop()
    with pytest.raises(ValueError, match="exact frozen Qwen3.6 MTP set"):
        compare_safetensor_layout_with_exact_auxiliary_omission(
            base, candidate, missing_row
        )

    wrong_key_candidate = tmp_path / "wrong-key-candidate"
    _write_safetensors(wrong_key_candidate, {"other": torch.ones(2, dtype=torch.float32)})
    with pytest.raises(ValueError, match="missing keys differ"):
        compare_safetensor_layout_with_exact_auxiliary_omission(
            base, wrong_key_candidate, omission
        )
