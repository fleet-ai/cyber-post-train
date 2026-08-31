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
        "staging_execution": _execution_plan(),
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


def test_reviewed_plan_binds_exact_local_configmap_code(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json").read_text()
    )
    observed = staging.validate_local_staging_bundle(plan, root)
    assert observed == plan["staging_execution"]["config_map_code_sha256"]

    copy_root = tmp_path / "checkout"
    for relative in staging.MOUNTED_CODE_FILES.values():
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
        "execution": _execution_plan(),
        "destination": {"path": str(final), "must_be_absent": True},
        "raw_file_count": raw_manifest["file_count"],
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
