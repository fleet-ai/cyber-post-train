import hashlib
import json
import shutil
import zipfile
from pathlib import Path

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


def test_manifest_verification_and_weights_only_composition(tmp_path):
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
    staging.compose_bundle(raw, base, output, base_sidecars)
    assert file_sha256(output / "model.safetensors") == file_sha256(raw / "model.safetensors")
    assert file_sha256(output / "config.json") == base_sidecars["config.json"]
    assert file_sha256(output / "config.json") != file_sha256(raw / "config.json")


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
    rows = [{"path": "model.safetensors", "size": 2, "sha256": "a" * 64}]
    raw_manifest = {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-v1/global_step_318/policy",
        "file_count": 1,
        "total_bytes": 2,
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }
    observation = {
        "schema": "fleet_sft_sfs_checkpoint_observation_v1",
        "run_name": "ft-run-574bd7b3",
        "step": 318,
        "output_inspection": {
            "root": raw_manifest["root"],
            "files_manifest_sha256": raw_manifest["manifest_sha256"],
        },
        "raw_export_full_manifest_sha256": raw_manifest["manifest_sha256"],
        "raw_export_file_count": 1,
        "raw_export_total_bytes": 2,
    }
    observation["observation_sha256"] = digest_json(observation)
    evidence_sha = "sha256:" + "b" * 64
    plan = {
        "export": {"expected_output_path": raw_manifest["root"]},
        "base_model": {
            "tokenizer_equivalence_evidence": {"sha256": evidence_sha},
            "runtime_sidecar_sha256": {"config.json": "sha256:" + "c" * 64},
            "parameter_count": 1,
            "tokenizer_manifest_sha256": "sha256:" + "d" * 64,
            "chat_template_sha256": "sha256:" + "e" * 64,
            "config_sha256": "sha256:" + "f" * 64,
        },
    }
    result = staging.build_stage_input(
        plan, observation, raw_manifest, tokenizer_evidence_sha256=evidence_sha
    )
    assert result["source"]["raw_full_manifest"]["manifest_sha256"] == digest_json(rows)
    assert result["stage_input_sha256"] == digest_json(
        {key: value for key, value in result.items() if key != "stage_input_sha256"}
    )

    observation["raw_export_file_count"] = 2
    observation["observation_sha256"] = digest_json(
        {key: value for key, value in observation.items() if key != "observation_sha256"}
    )
    try:
        staging.build_stage_input(
            plan, observation, raw_manifest, tokenizer_evidence_sha256=evidence_sha
        )
    except ValueError as exc:
        assert "does not bind" in str(exc)
    else:
        raise AssertionError("observation/manifest mismatch unexpectedly passed")


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
    raw_manifest = full_file_manifest(raw)
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
            "raw_full_manifest": raw_manifest,
            "raw_inspection": raw_inspection,
            "observation_sha256": "sha256:" + "1" * 64,
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
        "destination": {"path": str(final), "must_be_absent": True},
        "raw_file_count": raw_manifest["file_count"],
    }
    stage_input["stage_input_sha256"] = digest_json(stage_input)
    receipt = staging.execute_stage(
        stage_input, work_root=tmp_path / "work", forwarded_user="scientist@example.com"
    )
    assert receipt["destination"]["atomic_promotion"] is True
    assert final.is_dir()
    assert (final / ".fleet-acceptance.json").is_file()
    assert not any(path.name.startswith(".partial-") for path in final.parent.iterdir())

    try:
        staging.execute_stage(
            stage_input, work_root=tmp_path / "work-two", forwarded_user="scientist@example.com"
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("occupied destination unexpectedly passed")
