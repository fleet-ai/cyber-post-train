import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from training import post_sft_staging as staging
from training.io import digest_json, file_sha256
from training.post_sft_artifacts import TOKENIZER_FILES, full_file_manifest, inspect_hf_export


def _hf_root(root: Path) -> tuple[dict[str, str], dict]:
    root.mkdir(parents=True)
    save_file({"weight": torch.ones((1,), dtype=torch.bfloat16)}, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": 2}, "weight_map": {"weight": "model.safetensors"}}
        )
        + "\n"
    )
    payloads = {
        "chat_template.jinja": "{{ messages }}\n",
        "config.json": "{}\n",
        "merges.txt": "#version: 0.2\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "vocab.json": "{}\n",
    }
    for name, payload in payloads.items():
        (root / name).write_text(payload)
    sidecars = {name: file_sha256(root / name) for name in payloads}
    tokenizer_rows = [
        {"path": name, "sha256": file_sha256(root / name).removeprefix("sha256:")}
        for name in TOKENIZER_FILES
    ]
    expected = {
        "tokenizer": digest_json(sorted(tokenizer_rows, key=lambda row: row["path"])),
        "chat": file_sha256(root / "chat_template.jinja"),
        "config": file_sha256(root / "config.json"),
    }
    return sidecars, expected


def _execution_plan(code_hashes: dict[str, str] | None = None) -> dict:
    if code_hashes is None:
        code_hashes = {
            relative: "sha256:" + f"{index + 1:x}" * 64
            for index, relative in enumerate(staging.MOUNTED_CODE_FILES.values())
        }
    return {
        "schema": "cyber_sft_inference_stage_execution_plan_v1",
        "namespace": "inference",
        "job_name": staging.JOB_NAME,
        "config_map_name": staging.CONFIG_MAP_NAME,
        "service_account_name": staging.SERVICE_ACCOUNT_NAME,
        "container_name": staging.CONTAINER_NAME,
        "image": staging.STAGING_IMAGE,
        "image_digest": staging.STAGING_IMAGE.rsplit("@", 1)[1],
        "command_sha256": staging.STAGING_COMMAND_SHA256,
        "config_map_code_sha256": code_hashes,
    }


def _test_base_surface_and_manifest() -> tuple[dict, dict]:
    sidecar_names = (
        "chat_template.jinja",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    )
    sidecars = {
        name: "sha256:" + f"{index:x}" * 64
        for index, name in enumerate(sidecar_names, start=1)
    }
    weight_rows = [
        {
            "path": f"model-{index:05d}-of-00015.safetensors",
            "size": index,
            "sha256": f"{index:x}" * 64,
        }
        for index in range(1, 16)
    ]
    index_row = {
        "path": "model.safetensors.index.json",
        "size": 3,
        "sha256": "9" * 64,
    }
    sidecar_rows = [
        {
            "path": name,
            "size": index + 20,
            "sha256": digest.removeprefix("sha256:"),
        }
        for index, (name, digest) in enumerate(sidecars.items())
    ]
    rows = sorted(
        [*weight_rows, index_row, *sidecar_rows], key=lambda row: row["path"]
    )
    surface = {
        "schema": "cyber_sft_base_inference_artifact_surface_v1",
        "policy": "exact_top_level_inference_artifacts_with_reviewed_control_exclusions_v1",
        "weight_shard_count": 15,
        "weights_manifest_sha256": digest_json(weight_rows),
        "index": {
            "path": "model.safetensors.index.json",
            "sha256": "sha256:" + "9" * 64,
        },
        "required_runtime_sidecar_sha256": sidecars,
        "allowed_non_artifact_top_level_files": {
            "README.md": "test documentation; not loaded by inference"
        },
        "excluded_non_artifact_directory_prefixes": {
            ".cache/": "test cache; not loaded by inference"
        },
        "unknown_top_level_entries": "reject",
        "symlinks": "reject",
    }
    manifest = {
        "schema": "cyber_sft_base_inference_artifact_manifest_v1",
        "root": "/base",
        "surface_sha256": digest_json(surface),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
        "weights_manifest_sha256": digest_json(weight_rows),
        "runtime_sidecar_sha256": surface["required_runtime_sidecar_sha256"],
        "excluded_non_artifact_files": [
            {
                "path": "README.md",
                "reviewed_reason": "test documentation; not loaded by inference",
            }
        ],
        "excluded_non_artifact_directory_prefixes": [
            {
                "path_prefix": ".cache/",
                "reviewed_reason": "test cache; not loaded by inference",
            }
        ],
    }
    return surface, manifest


def _tamper_base_artifact_manifest(manifest: dict, kind: str) -> None:
    rows = manifest["files"]
    if kind == "evil_path":
        rows.append({"path": "evil.bin", "size": 1, "sha256": "1" * 64})
    elif kind == "extra_path":
        rows.append({"path": "unknown.txt", "size": 1, "sha256": "4" * 64})
    elif kind == "missing_path":
        rows.pop(0)
    elif kind == "duplicate_path":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "changed_index_hash":
        next(
            row for row in rows if row["path"] == "model.safetensors.index.json"
        )["sha256"] = "1" * 64
    elif kind == "changed_sidecar_hash":
        next(row for row in rows if row["path"] == "config.json")["sha256"] = "0" * 64
    elif kind == "wrong_shard_aggregate":
        next(row for row in rows if row["path"].endswith(".safetensors"))[
            "sha256"
        ] = "3" * 64
    elif kind == "wrong_exclusions":
        manifest["excluded_non_artifact_directory_prefixes"] = []
    rows.sort(key=lambda row: row["path"])
    manifest["file_count"] = len(rows)
    manifest["total_bytes"] = sum(row["size"] for row in rows)
    manifest["manifest_sha256"] = digest_json(rows)
    if kind == "wrong_shard_aggregate":
        manifest["weights_manifest_sha256"] = digest_json(
            [row for row in rows if row["path"].endswith(".safetensors")]
        )
    if kind == "wrong_count":
        manifest["file_count"] += 1
    elif kind == "wrong_total":
        manifest["total_bytes"] += 1


def _cast_receipt_and_manifest(
    root: Path, inspection: dict, observation_sha256: str
) -> tuple[dict, dict]:
    payload = full_file_manifest(root)
    receipt = {
        "schema": "cyber_sft_fp32_to_bf16_cast_receipt_v2",
        "source": {"observation_sha256": observation_sha256},
        "frozen_base_auxiliary_source": {
            "revision": "base-revision",
            "weights_manifest_sha256": "sha256:" + "8" * 64,
            "restoration_semantics": (
                "frozen_base_auxiliary_head_restoration_not_trained_weights"
            ),
        },
        "conversion": {
            "schema": "cyber_sft_fp32_to_bf16_cast_and_restore_proof_v2",
            "trained_tensor_count": 1,
            "trained_parameter_count": 1,
            "restored_auxiliary_tensor_count": 0,
            "restored_auxiliary_parameter_count": 0,
            "final_tensor_count": 1,
            "final_parameter_count": 1,
            "cast_rows_sha256": "sha256:" + "6" * 64,
            "restoration_rows_sha256": "sha256:" + "7" * 64,
            "exact_omission_evidence_sha256": "sha256:" + "5" * 64,
            "all_destination_bits_equal_direct_bf16_cast": True,
            "all_restored_auxiliary_bits_equal_frozen_base": True,
            "final_layout_exactly_matches_frozen_base": True,
        },
        "destination": {
            "path": str(root),
            "dtype": "BF16",
            "inspection": {**inspection, "root": str(root)},
            "payload_manifest_sha256": payload["manifest_sha256"],
        },
    }
    receipt["cast_receipt_sha256"] = digest_json(receipt)
    (root / ".fleet-bf16-cast-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt, full_file_manifest(root)


def _fake_runtime_provenance(stage_input: dict) -> dict:
    execution = stage_input["execution"]
    stage_input_sha256 = stage_input["stage_input_sha256"]
    stage_input_file_sha256 = "sha256:" + hashlib.sha256(
        staging.canonical_stage_input_bytes(stage_input)
    ).hexdigest()
    mounted = {
        **execution["config_map_code_sha256"],
        staging.STAGE_INPUT_MOUNT_PATH: stage_input_file_sha256,
    }
    return {
        "schema": staging.EXECUTION_SCHEMA,
        "image": staging.STAGING_IMAGE,
        "image_id": "containerd://" + staging.STAGING_IMAGE.rsplit("@", 1)[1],
        "resolved_image_digest": staging.STAGING_IMAGE.rsplit("@", 1)[1],
        "command_sha256": staging.STAGING_COMMAND_SHA256,
        "service_account_name": staging.SERVICE_ACCOUNT_NAME,
        "container_name": staging.CONTAINER_NAME,
        "job": {
            "namespace": "inference",
            "name": staging.JOB_NAME,
            "uid": "job-uid",
            "resource_version": "11",
            "spec_sha256": "sha256:" + "a" * 64,
        },
        "pod": {
            "namespace": "inference",
            "name": "stage-pod",
            "uid": "pod-uid",
            "resource_version": "12",
            "spec_sha256": "sha256:" + "b" * 64,
        },
        "config_map": {
            "namespace": "inference",
            "name": staging.CONFIG_MAP_NAME,
            "uid": "config-uid",
            "resource_version": "123",
            "immutable": True,
            "reviewed_code_sha256": execution["config_map_code_sha256"],
            "mounted_file_sha256": mounted,
            "stage_input_file_sha256": stage_input_file_sha256,
        },
        "stage_input_sha256": stage_input_sha256,
    }


def test_manifest_verification_and_weights_only_composition(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    base = tmp_path / "base"
    _hf_root(raw)
    base_sidecars, _ = _hf_root(base)
    (raw / "config.json").write_text('{"trainer":"drift"}\n')
    manifest = full_file_manifest(raw)
    assert staging.verify_full_manifest(raw, manifest)["manifest_sha256"] == manifest[
        "manifest_sha256"
    ]

    output = tmp_path / "composed"
    synced = []
    real_fsync_file = staging._fsync_file

    def record_fsync(path):
        synced.append(Path(path))
        real_fsync_file(path)

    monkeypatch.setattr(staging, "_fsync_file", record_fsync)
    staging.compose_bundle(raw, base, output, base_sidecars)
    assert file_sha256(output / "model.safetensors") == file_sha256(raw / "model.safetensors")
    assert file_sha256(output / "config.json") == base_sidecars["config.json"]
    assert file_sha256(output / "config.json") != file_sha256(raw / "config.json")
    assert set(output.iterdir()) == set(synced)


def test_manifest_rejects_collision_and_path_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "one").write_text("1")
    manifest = full_file_manifest(root)
    manifest["files"][0]["path"] = "../escape"
    manifest["manifest_sha256"] = digest_json(manifest["files"])
    try:
        staging.verify_full_manifest(root, manifest)
    except ValueError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe manifest unexpectedly passed")


def test_stage_input_binds_observation_manifest_and_tokenizer_evidence():
    artifact_surface, artifact_manifest = _test_base_surface_and_manifest()
    assert artifact_surface["weight_shard_count"] == 15
    assert len(artifact_surface["required_runtime_sidecar_sha256"]) == 10
    assert artifact_manifest["file_count"] == 26
    omission = {
        "schema": "cyber_sft_exact_auxiliary_head_omission_v1",
        "base_revision": "base-revision",
        "base_weights_manifest_sha256": artifact_surface["weights_manifest_sha256"],
        "serving_registration_sha256": "sha256:" + "4" * 64,
        "raw_export_tensor_count": 1,
        "raw_export_parameter_count": 1,
        "missing_tensor_count": 0,
        "missing_parameter_count": 0,
        "base_tensor_count": 1,
        "base_parameter_count": 1,
        "tensors": [],
    }
    cast_receipt = {
        "schema": "cyber_sft_fp32_to_bf16_cast_receipt_v2",
        "source": {"observation_sha256": "sha256:" + "1" * 64},
        "frozen_base_auxiliary_source": {
            "revision": "base-revision",
            "weights_manifest_sha256": artifact_surface["weights_manifest_sha256"],
            "inference_artifact_surface": artifact_surface,
            "inference_artifact_surface_sha256": digest_json(artifact_surface),
            "inference_artifact_manifest_before": artifact_manifest,
            "inference_artifact_manifest_after": artifact_manifest,
            "manifest_scope": "exact_inference_artifact_surface_v1",
            "serving_registration_sha256": "sha256:" + "4" * 64,
            "omission_policy_sha256": digest_json(omission),
            "exact_auxiliary_omission_policy": omission,
            "restoration_semantics": (
                "frozen_base_auxiliary_head_restoration_not_trained_weights"
            ),
        },
        "conversion": {
            "schema": "cyber_sft_fp32_to_bf16_cast_and_restore_proof_v2",
            "trained_tensor_count": 1,
            "trained_parameter_count": 1,
            "restored_auxiliary_tensor_count": 0,
            "restored_auxiliary_parameter_count": 0,
            "final_tensor_count": 1,
            "final_parameter_count": 1,
            "cast_rows_sha256": "sha256:" + "6" * 64,
            "restoration_rows_sha256": digest_json([]),
            "restoration_rows": [],
            "exact_omission_evidence_sha256": digest_json([]),
            "exact_omission_tensors": [],
            "all_destination_bits_equal_direct_bf16_cast": True,
            "all_restored_auxiliary_bits_equal_frozen_base": True,
            "final_layout_exactly_matches_frozen_base": True,
        },
        "destination": {
            "path": "/cast",
            "dtype": "BF16",
            "inspection": {
                "root": "/cast",
                "dtype": "bf16",
                "files_manifest_sha256": digest_json(
                    [{"path": "model.safetensors", "size": 2, "sha256": "a" * 64}]
                ),
            },
            "payload_manifest_sha256": digest_json(
                [{"path": "model.safetensors", "size": 2, "sha256": "a" * 64}]
            ),
        },
    }
    cast_receipt["cast_receipt_sha256"] = digest_json(cast_receipt)
    receipt_bytes = (json.dumps(cast_receipt, indent=2, sort_keys=True) + "\n").encode()
    rows = [
        {
            "path": ".fleet-bf16-cast-acceptance.json",
            "size": len(receipt_bytes),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        {"path": "model.safetensors", "size": 2, "sha256": "a" * 64},
    ]
    cast_manifest = {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": "/cast",
        "file_count": 2,
        "total_bytes": 2 + len(receipt_bytes),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }
    observation = {
        "schema": "fleet_sft_sfs_checkpoint_observation_v1",
        "run_name": "ft-run-574bd7b3",
        "step": 318,
        "output_inspection": {
            "root": "/raw",
            "dtype": "f32",
            "files_manifest_sha256": "sha256:" + "9" * 64,
        },
    }
    observation["observation_sha256"] = digest_json(observation)
    cast_receipt["source"]["observation_sha256"] = observation["observation_sha256"]
    cast_receipt["cast_receipt_sha256"] = digest_json(
        {key: value for key, value in cast_receipt.items() if key != "cast_receipt_sha256"}
    )
    receipt_bytes = (json.dumps(cast_receipt, indent=2, sort_keys=True) + "\n").encode()
    rows[0].update(size=len(receipt_bytes), sha256=hashlib.sha256(receipt_bytes).hexdigest())
    cast_manifest.update(
        total_bytes=2 + len(receipt_bytes),
        manifest_sha256=digest_json(rows),
    )
    evidence_sha = "sha256:" + "b" * 64
    plan = {
        "cast_execution": {
            "destination_path": "/cast",
            "base_model_path": "/base",
            "base_inference_artifact_surface": artifact_surface,
        },
        "staging_execution": _execution_plan(),
        "base_model": {
            "tokenizer_equivalence_evidence": {"sha256": evidence_sha},
            "runtime_sidecar_sha256": artifact_surface[
                "required_runtime_sidecar_sha256"
            ],
            "parameter_count": 1,
            "tokenizer_manifest_sha256": "sha256:" + "d" * 64,
            "chat_template_sha256": "sha256:" + "e" * 64,
            "config_sha256": "sha256:" + "f" * 64,
        },
        "export": {"raw_export_auxiliary_head_omission": omission},
    }
    result = staging.build_stage_input(
        plan,
        observation,
        cast_receipt,
        cast_manifest,
        tokenizer_evidence_sha256=evidence_sha,
    )
    assert result["source"]["bf16_full_manifest"]["manifest_sha256"] == digest_json(rows)
    assert result["stage_input_sha256"] == digest_json(
        {key: value for key, value in result.items() if key != "stage_input_sha256"}
    )

    for mutate in (
        lambda value: value["frozen_base_auxiliary_source"].__setitem__(
            "serving_registration_sha256", "sha256:" + "3" * 64
        ),
        lambda value: value["conversion"].__setitem__(
            "restoration_rows_sha256", "sha256:" + "2" * 64
        ),
        lambda value: value["conversion"].__setitem__(
            "exact_omission_evidence_sha256", "sha256:" + "1" * 64
        ),
    ):
        tampered_receipt = copy.deepcopy(cast_receipt)
        tampered_manifest = copy.deepcopy(cast_manifest)
        mutate(tampered_receipt)
        tampered_receipt["cast_receipt_sha256"] = digest_json(
            {
                key: value
                for key, value in tampered_receipt.items()
                if key != "cast_receipt_sha256"
            }
        )
        tampered_bytes = (
            json.dumps(tampered_receipt, indent=2, sort_keys=True) + "\n"
        ).encode()
        tampered_manifest["files"][0].update(
            size=len(tampered_bytes), sha256=hashlib.sha256(tampered_bytes).hexdigest()
        )
        tampered_manifest["total_bytes"] = 2 + len(tampered_bytes)
        tampered_manifest["manifest_sha256"] = digest_json(
            tampered_manifest["files"]
        )
        try:
            staging.build_stage_input(
                plan,
                observation,
                tampered_receipt,
                tampered_manifest,
                tokenizer_evidence_sha256=evidence_sha,
            )
        except ValueError as exc:
            assert "exact trained/restored tensor provenance" in str(exc)
        else:
            raise AssertionError("re-signed cast provenance tamper unexpectedly passed")

    for kind in (
        "evil_path",
        "extra_path",
        "missing_path",
        "duplicate_path",
        "changed_index_hash",
        "changed_sidecar_hash",
        "wrong_shard_aggregate",
        "wrong_count",
        "wrong_total",
        "wrong_exclusions",
    ):
        tampered_receipt = copy.deepcopy(cast_receipt)
        base_source = tampered_receipt["frozen_base_auxiliary_source"]
        tampered_base_manifest = copy.deepcopy(
            base_source["inference_artifact_manifest_before"]
        )
        _tamper_base_artifact_manifest(tampered_base_manifest, kind)
        base_source["inference_artifact_manifest_before"] = tampered_base_manifest
        base_source["inference_artifact_manifest_after"] = copy.deepcopy(
            tampered_base_manifest
        )
        base_source["full_manifest_before_sha256"] = tampered_base_manifest[
            "manifest_sha256"
        ]
        base_source["full_manifest_after_sha256"] = tampered_base_manifest[
            "manifest_sha256"
        ]
        tampered_receipt["cast_receipt_sha256"] = digest_json(
            {
                key: value
                for key, value in tampered_receipt.items()
                if key != "cast_receipt_sha256"
            }
        )
        tampered_bytes = (
            json.dumps(tampered_receipt, indent=2, sort_keys=True) + "\n"
        ).encode()
        tampered_manifest = copy.deepcopy(cast_manifest)
        tampered_manifest["files"][0].update(
            size=len(tampered_bytes), sha256=hashlib.sha256(tampered_bytes).hexdigest()
        )
        tampered_manifest["total_bytes"] = sum(
            row["size"] for row in tampered_manifest["files"]
        )
        tampered_manifest["manifest_sha256"] = digest_json(
            tampered_manifest["files"]
        )
        with pytest.raises(ValueError, match="base inference artifact"):
            staging.build_stage_input(
                plan,
                observation,
                tampered_receipt,
                tampered_manifest,
                tokenizer_evidence_sha256=evidence_sha,
            )

    cast_manifest["files"][1]["sha256"] = "b" * 64
    cast_manifest["manifest_sha256"] = digest_json(cast_manifest["files"])
    try:
        staging.build_stage_input(
            plan,
            observation,
            cast_receipt,
            cast_manifest,
            tokenizer_evidence_sha256=evidence_sha,
        )
    except ValueError as exc:
        assert "payload manifest" in str(exc)
    else:
        raise AssertionError("observation/manifest mismatch unexpectedly passed")


def test_reviewed_plan_binds_exact_local_configmap_code(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    observed = staging.validate_local_staging_bundle(plan, root)
    assert observed == plan["staging_execution"]["config_map_code_sha256"]

    copy_root = tmp_path / "checkout"
    for relative in staging.LOCAL_CODE_FILES.values():
        destination = copy_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    (copy_root / "training/post_sft_staging.py").write_text("# drift\n")
    try:
        staging.validate_local_staging_bundle(plan, copy_root)
    except ValueError as exc:
        assert "differs from the reviewed digest" in str(exc)
    else:
        raise AssertionError("unreviewed ConfigMap code unexpectedly passed")


def test_execute_stage_streams_verifies_composes_and_promotes_atomically(tmp_path, monkeypatch):
    raw = tmp_path / "source-raw"
    base = tmp_path / "base"
    sidecars, expected = _hf_root(raw)
    base_sidecars, _ = _hf_root(base)
    raw_inspection = inspect_hf_export(
        raw,
        expected_tokenizer_manifest_sha256=expected["tokenizer"],
        expected_chat_template_sha256=expected["chat"],
        expected_config_sha256=expected["config"],
        expected_parameter_count=1,
        expected_sidecar_sha256=sidecars,
    )
    cast_receipt, raw_manifest = _cast_receipt_and_manifest(
        raw, raw_inspection, "sha256:" + "1" * 64
    )
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in raw.iterdir():
            bundle.write(path, Path("policy") / path.name)

    final = tmp_path / "models" / "final"
    monkeypatch.setattr(staging, "BASE_ROOT", str(base))
    monkeypatch.setattr(staging, "DESTINATION", str(final))

    def fake_download(_url, destination, _user):
        shutil.copyfile(archive, destination)
        data = destination.read_bytes()
        return {
            "zip_bytes": len(data),
            "zip_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }

    monkeypatch.setattr(staging, "_download_export", fake_download)
    stage_input = {
        "schema": staging.STAGE_SCHEMA,
        "source": {
            "filebrowser_url": "http://filebrowser.invalid/download",
            "bf16_full_manifest": raw_manifest,
            "bf16_inspection": raw_inspection,
            "observation_sha256": "sha256:" + "1" * 64,
            "cast_receipt_sha256": cast_receipt["cast_receipt_sha256"],
        },
        "composition": {
            "base_root": str(base),
            "runtime_sidecar_sha256": base_sidecars,
            "tokenizer_equivalence_evidence_sha256": "sha256:" + "2" * 64,
            "expected_parameter_count": 1,
            "expected_tokenizer_manifest_sha256": expected["tokenizer"],
            "expected_chat_template_sha256": expected["chat"],
            "expected_config_sha256": expected["config"],
        },
        "execution": _execution_plan(),
        "destination": {"path": str(final), "must_be_absent": True},
        "source_file_count": raw_manifest["file_count"],
    }
    stage_input["stage_input_sha256"] = digest_json(stage_input)
    monkeypatch.setattr(staging, "_runtime_execution_provenance", _fake_runtime_provenance)
    real_rename = staging._rename_noreplace

    def crash_before_commit(_source, _destination):
        raise OSError("simulated crash before atomic promotion")

    monkeypatch.setattr(staging, "_rename_noreplace", crash_before_commit)
    try:
        staging.execute_stage(
            stage_input, work_root=tmp_path / "work", forwarded_user="scientist@example.com"
        )
    except OSError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("failure injection unexpectedly committed the staging transaction")
    suffix = stage_input["stage_input_sha256"].removeprefix("sha256:")[:12]
    complete_partial = final.parent / f".partial-{final.name}-{suffix}"
    assert (complete_partial / staging.ACCEPTANCE_RECEIPT_NAME).is_file()
    assert not final.exists()

    monkeypatch.setattr(staging, "_rename_noreplace", real_rename)
    receipt = staging.execute_stage(
        stage_input, work_root=tmp_path / "work-recovery", forwarded_user="scientist@example.com"
    )
    assert receipt["destination"]["atomic_promotion"] is True
    assert receipt["execution"] == _fake_runtime_provenance(stage_input)
    assert receipt["composition"]["inspection"]["root"] == str(final)
    assert final.is_dir()
    acceptance = final / staging.ACCEPTANCE_RECEIPT_NAME
    assert acceptance.is_file()
    assert receipt["destination"]["acceptance_receipt_path"] == str(acceptance)
    assert receipt["destination"]["payload_manifest_excludes"] == [
        staging.ACCEPTANCE_RECEIPT_NAME
    ]
    assert staging._payload_manifest(final)["manifest_sha256"] == receipt["composition"][
        "inspection"
    ]["files_manifest_sha256"]
    assert not any(path.name.startswith(".partial-") for path in final.parent.iterdir())

    recovered = staging.execute_stage(
        stage_input,
        work_root=tmp_path / "work-after-commit",
        forwarded_user="scientist@example.com",
    )
    assert recovered == receipt


def test_staging_execution_identity_matches_cluster_job():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "evals/post_sft/cluster/qwen36-sft-inference-stage-job.yaml"
    ).read_text(encoding="utf-8")
    assert f"image: {staging.STAGING_IMAGE}" in manifest
    assert 'command: ["python3", "-m", "training.post_sft_staging", "execute"]' in manifest
    expected_args = 'args: ["--input", "/bundle/stage-input.json", "--work-root", "/work/transfer"]'
    assert expected_args in manifest
    assert f"serviceAccountName: {staging.SERVICE_ACCOUNT_NAME}" in manifest
    assert "resources: [pods]" in manifest and "verbs: [get]" in manifest
    assert "fieldPath: metadata.uid" in manifest
    assert "batch.kubernetes.io/controller-uid" in manifest
    assert "Job controller generates the Pod name" in manifest
    submit = (
        Path(__file__).resolve().parents[1]
        / "evals/post_sft/scripts/submit_inference_stage.sh"
    ).read_text(encoding="utf-8")
    assert "kubectl apply" not in submit
    assert "kubectl create --dry-run=server" in submit
    assert "kubectl create -f" in submit
    assert 'value["immutable"]=True' in submit
    for resource in (
        "serviceaccount",
        "role.rbac.authorization.k8s.io",
        "rolebinding.rbac.authorization.k8s.io",
        "configmap",
        "job.batch",
    ):
        assert resource in submit


def test_atomic_promotion_refuses_destination_created_after_preflight(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "model.safetensors").write_text("verified-payload")
    final = tmp_path / "final"

    # This directory represents an independent writer winning the race after the caller's
    # initial absence check but before the final rename.
    final.mkdir()
    winning_inode = final.stat().st_ino
    try:
        staging._rename_noreplace(partial, final)
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic promotion replaced a destination created after preflight")
    assert final.stat().st_ino == winning_inode
    assert list(final.iterdir()) == []
    assert (partial / "model.safetensors").read_text() == "verified-payload"


def test_runtime_provenance_binds_live_kubernetes_and_configmap_bytes(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    code_data = {
        "training__init__.py": "# package\n",
        "training_io.py": "# io\n",
        "training_post_sft_artifacts.py": "# artifacts\n",
        "training_post_sft_base_surface.py": "# base surface\n",
        "training_post_sft_staging.py": "# staging\n",
    }
    code_hashes = {
        relative: "sha256:" + hashlib.sha256(code_data[key].encode()).hexdigest()
        for key, relative in staging.MOUNTED_CODE_FILES.items()
    }
    stage_input = {
        "schema": staging.STAGE_SCHEMA,
        "source": {"binding": "exact"},
        "execution": _execution_plan(code_hashes),
    }
    stage_input["stage_input_sha256"] = digest_json(stage_input)
    data = {
        **code_data,
        staging.STAGE_INPUT_CONFIG_MAP_KEY: staging.canonical_stage_input_bytes(
            stage_input
        ).decode(),
    }
    for key, relative in staging.MOUNTED_CODE_FILES.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data[key])
    (bundle / staging.STAGE_INPUT_MOUNT_PATH).write_bytes(
        staging.canonical_stage_input_bytes(stage_input)
    )
    monkeypatch.setattr(staging, "BUNDLE_ROOT", bundle)
    for name, value in {
        "POD_NAMESPACE": "inference",
        "POD_NAME": "stage-pod",
        "POD_UID": "pod-uid",
        "JOB_UID": "job-uid",
    }.items():
        monkeypatch.setenv(name, value)
    pod = {
        "metadata": {
            "namespace": "inference",
            "name": "stage-pod",
            "uid": "pod-uid",
            "resourceVersion": "12",
            "ownerReferences": [
                {"kind": "Job", "name": staging.JOB_NAME, "uid": "job-uid"}
            ],
        },
        "spec": {
            "serviceAccountName": staging.SERVICE_ACCOUNT_NAME,
            "containers": [
                {
                    "name": staging.CONTAINER_NAME,
                    "image": staging.STAGING_IMAGE,
                    **staging.STAGING_COMMAND,
                }
            ],
            "volumes": [{"configMap": {"name": staging.CONFIG_MAP_NAME}}],
        },
        "status": {
            "containerStatuses": [
                {
                    "name": staging.CONTAINER_NAME,
                    "imageID": "docker-pullable://" + staging.STAGING_IMAGE,
                }
            ]
        },
    }
    responses = {
        "/api/v1/namespaces/inference/pods/stage-pod": pod,
        f"/apis/batch/v1/namespaces/inference/jobs/{staging.JOB_NAME}": {
            "metadata": {
                "namespace": "inference",
                "name": staging.JOB_NAME,
                "uid": "job-uid",
                "resourceVersion": "11",
            },
            "spec": {"template": {"binding": "exact"}},
        },
        f"/api/v1/namespaces/inference/configmaps/{staging.CONFIG_MAP_NAME}": {
            "metadata": {
                "namespace": "inference",
                "name": staging.CONFIG_MAP_NAME,
                "uid": "config-uid",
                "resourceVersion": "123",
            },
            "immutable": True,
            "data": data,
        },
    }
    monkeypatch.setattr(staging, "_kubernetes_get", responses.__getitem__)

    receipt = staging._runtime_execution_provenance(stage_input)
    assert receipt["job"]["uid"] == "job-uid"
    assert receipt["pod"]["uid"] == "pod-uid"
    assert receipt["config_map"]["uid"] == "config-uid"
    assert receipt["config_map"]["resource_version"] == "123"
    assert set(receipt["config_map"]["mounted_file_sha256"]) == set(
        {*staging.MOUNTED_CODE_FILES.values(), staging.STAGE_INPUT_MOUNT_PATH}
    )

    data["training_post_sft_staging.py"] = "# different live ConfigMap bytes\n"
    try:
        staging._runtime_execution_provenance(stage_input)
    except ValueError as exc:
        assert "live ConfigMap differs from the reviewed digest" in str(exc)
    else:
        raise AssertionError("runtime provenance accepted ConfigMap/mount byte drift")


def test_resolved_image_id_parser_is_exactly_anchored():
    digest = staging.STAGING_IMAGE.rsplit("@", 1)[1]
    assert staging._resolved_image_digest("containerd://" + digest) == digest
    assert staging._resolved_image_digest("docker-pullable://repo/image@" + digest) == digest
    for invalid in (digest + "-suffix", "prefix-" + digest, digest + "/extra"):
        try:
            staging._resolved_image_digest(invalid)
        except ValueError as exc:
            assert "exact digest identity" in str(exc)
        else:
            raise AssertionError(f"non-exact image ID unexpectedly passed: {invalid}")


def _minimal_stage_input(destination: Path) -> dict:
    value = {
        "schema": staging.STAGE_SCHEMA,
        "source": {},
        "composition": {},
        "destination": {"path": str(destination), "must_be_absent": True},
    }
    value["stage_input_sha256"] = digest_json(value)
    return value


def test_execute_stage_rejects_broken_symlink_at_final_destination(tmp_path, monkeypatch):
    final = tmp_path / "models" / "final"
    final.parent.mkdir()
    final.symlink_to(tmp_path / "missing-final-target")
    monkeypatch.setattr(staging, "DESTINATION", str(final))

    try:
        staging.execute_stage(
            _minimal_stage_input(final),
            work_root=tmp_path / "work",
            forwarded_user="scientist@example.com",
        )
    except ValueError as exc:
        assert "not a regular directory" in str(exc)
    else:
        raise AssertionError("broken destination symlink was treated as absent")


def test_execute_stage_rejects_broken_symlink_at_partial_destination(tmp_path, monkeypatch):
    final = tmp_path / "models" / "final"
    final.parent.mkdir()
    stage_input = _minimal_stage_input(final)
    suffix = stage_input["stage_input_sha256"].removeprefix("sha256:")[:12]
    partial = final.parent / f".partial-{final.name}-{suffix}"
    partial.symlink_to(tmp_path / "missing-partial-target")
    monkeypatch.setattr(staging, "DESTINATION", str(final))

    try:
        staging.execute_stage(
            stage_input,
            work_root=tmp_path / "work",
            forwarded_user="scientist@example.com",
        )
    except ValueError as exc:
        assert "not a regular directory" in str(exc)
    else:
        raise AssertionError("broken partial symlink was treated as absent")
