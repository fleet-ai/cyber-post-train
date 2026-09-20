from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import shutil
import stat
from pathlib import Path

import pytest
import yaml

from training import model_stage as stage
from training import model_stage_current_base as current_stage

ROOT = Path(__file__).resolve().parents[1]
FAILED_V1 = ROOT / "configs/qualification/qwen38-fresh75-step230-inference-stage-v1.json"
PRODUCTION = ROOT / "configs/qualification/qwen38-fresh75-step230-inference-stage-v2.json"
SELF_SFT = ROOT / "configs/qualification/qwen38-self-sft-step44-inference-stage-v1.json"
ACCEPTED_V2 = (
    ROOT / "docs/evidence/qwen38-fresh75-step230-inference-stage-v2-accepted-20260915.json"
)
SOURCE = ROOT / "training/model_stage.py"


def _receipt(value: dict) -> bytes:
    payload = {**value, "receipt_sha256": hashlib.sha256(stage.canonical(value)).hexdigest()}
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def _rehash_plan(plan: dict) -> dict:
    plan["plan_sha256"] = stage.digest_json(stage._unsigned(plan, "plan_sha256"))
    return plan


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, bytes], dict[str, str]]:
    plan = copy.deepcopy(json.loads(PRODUCTION.read_text()))
    destination = tmp_path / plan["destination"]["model_id"]
    plan["destination"]["path"] = str(destination)
    plan["destination"]["acceptance_path"] = str(destination / stage.ACCEPTANCE)
    plan["desired_registration"]["spec"]["model"]["sourcePath"] = str(destination)
    payloads = {f"payload-{index:02d}.bin": f"content-{index}".encode() for index in range(29)}
    files = {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in payloads.items()
    }
    export_binding = plan["source"]["export_receipt"]
    export = {**export_binding["required_fields"], "files": files}
    export_bytes = _receipt(export)
    export_value = json.loads(export_bytes)
    export_binding["file_sha256"] = stage._digest_bytes(export_bytes)
    export_binding["receipt_sha256"] = "sha256:" + export_value["receipt_sha256"]
    plan["source"]["payload"]["manifest_sha256"] = stage.digest_json(files)
    plan["desired_registration"]["spec"]["model"]["revision"] = stage.digest_json(files)
    gpu_binding = plan["source"]["gpu_check_receipt"]
    gpu = dict(gpu_binding["required_fields"])
    gpu["export_sha256"] = export_binding["file_sha256"].removeprefix("sha256:")
    gpu["export_receipt_sha256"] = export_value["receipt_sha256"]
    gpu_binding["required_fields"] = gpu
    gpu_bytes = _receipt(gpu)
    gpu_value = json.loads(gpu_bytes)
    gpu_binding["file_sha256"] = stage._digest_bytes(gpu_bytes)
    gpu_binding["receipt_sha256"] = "sha256:" + gpu_value["receipt_sha256"]
    plan["execution"]["runtime_source_sha256"] = stage._digest_bytes(SOURCE.read_bytes())
    _rehash_plan(plan)
    objects = {
        export_binding["sfs_path"]: export_bytes,
        gpu_binding["sfs_path"]: gpu_bytes,
        **{
            export_binding["sfs_path"].rsplit("/", 1)[0] + "/" + name: value
            for name, value in payloads.items()
        },
    }
    environment = {
        "POD_NAMESPACE": "inference",
        "POD_NAME": plan["execution"]["pod_name"],
        "POD_UID": "12345678-1234-1234-1234-123456789abc",
    }
    return plan, objects, environment


def _opener(objects: dict[str, bytes], calls: list[str] | None = None):
    @contextlib.contextmanager
    def open_source(path: str):
        if calls is not None:
            calls.append(path)
        yield io.BytesIO(objects[path])

    return open_source


def test_production_plan_is_self_digesting_and_exact_registration_clone() -> None:
    plan = stage.read_plan(PRODUCTION)
    assert plan["execution"]["runtime_source_sha256"] == stage._digest_bytes(SOURCE.read_bytes())
    assert plan["source"]["payload"] == {
        "mapping": "exact EXPORT.files object with bare per-file SHA-256 values",
        "file_count": 29,
        "manifest_sha256": (
            "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
        ),
        "maximum_file_bytes": 4 * 1024**3,
        "maximum_total_bytes": 64 * 1024**3,
    }
    desired = plan["desired_registration"]
    assert desired["id"] == plan["destination"]["model_id"]
    assert desired["spec"]["desiredState"] == "paused"
    assert desired["spec"]["scaling"]["minReplicas"] == 0
    assert desired["spec"]["placement"]["priorityClassName"] == "c1"
    assert desired["spec"]["model"]["revision"] == plan["source"]["payload"]["manifest_sha256"]


def test_self_sft_plan_clones_current_base_runtime_and_starts_paused() -> None:
    plan = current_stage.read_plan(SELF_SFT)
    source = plan["registration_source"]
    desired = plan["desired_registration"]
    source_spec = copy.deepcopy(source["spec"])
    desired_spec = desired["spec"]

    assert source["id"] == "qwen3.8-27b"
    assert source["spec_sha256"] == stage.digest_json(source_spec)
    assert plan["execution"]["runtime_source_sha256"] == stage._digest_bytes(SOURCE.read_bytes())
    assert plan["execution"]["runtime_wrapper_sha256"] == stage._digest_bytes(
        Path(current_stage.__file__).read_bytes()
    )
    assert desired["id"] == plan["destination"]["model_id"]
    assert desired_spec["desiredState"] == "paused"
    assert desired_spec["scaling"] == {"minReplicas": 0}
    assert desired_spec["placement"]["priorityClassName"] == "c1"
    assert desired_spec["model"]["dataParallelSize"] == 8
    assert desired_spec["model"]["tensorParallelSize"] == 1
    assert desired_spec["runtime"]["image"] == source_spec["runtime"]["image"]
    assert desired_spec["runtime"]["command"] == source_spec["runtime"]["command"]
    assert "--dp-size" in desired_spec["runtime"]["args"]
    assert (
        desired_spec["runtime"]["args"][desired_spec["runtime"]["args"].index("--dp-size") + 1]
        == "8"
    )
    assert (
        desired_spec["runtime"]["args"][
            desired_spec["runtime"]["args"].index("--context-length") + 1
        ]
        == "262144"
    )
    assert (
        desired_spec["runtime"]["args"][
            desired_spec["runtime"]["args"].index("--kv-cache-dtype") + 1
        ]
        == "fp8_e4m3"
    )
    assert (
        desired_spec["runtime"]["args"][
            desired_spec["runtime"]["args"].index("--reasoning-parser") + 1
        ]
        == "qwen3"
    )
    assert (
        desired_spec["runtime"]["args"][
            desired_spec["runtime"]["args"].index("--tool-call-parser") + 1
        ]
        == "qwen3_coder"
    )


def test_self_sft_registration_rejects_current_base_runtime_drift() -> None:
    plan = copy.deepcopy(json.loads(SELF_SFT.read_text()))
    plan["desired_registration"]["spec"]["runtime"]["args"].extend(
        ["--unreviewed-runtime-option", "1"]
    )
    _rehash_plan(plan)
    with pytest.raises(ValueError, match="outside the reviewed clone set"):
        current_stage.validate_plan(plan)


def test_self_sft_renderer_binds_both_runtime_sources() -> None:
    plan = current_stage.read_plan(SELF_SFT)
    wrapper = Path(current_stage.__file__)
    rendered = current_stage.render_manifest(plan, SOURCE, wrapper, SELF_SFT)
    config_map, pod = list(yaml.safe_load_all(rendered))
    assert config_map["data"]["model_stage.py"] == SOURCE.read_text()
    assert config_map["data"]["model_stage_current_base.py"] == wrapper.read_text()
    container = pod["spec"]["containers"][0]
    assert container["command"] == ["python3", "/bundle/model_stage_current_base.py"]
    assert {
        "name": "bundle",
        "mountPath": "/bundle/model_stage_current_base.py",
        "subPath": "model_stage_current_base.py",
        "readOnly": True,
    } in container["volumeMounts"]


def test_production_plan_is_exact_successor_of_preserved_failed_v1() -> None:
    failed_v1 = stage.read_plan(FAILED_V1)
    production = stage.read_plan(PRODUCTION)
    expected = copy.deepcopy(failed_v1)
    expected["execution"]["pod_name"] = "chris-q38-f75-p230-stage-v2"
    expected["execution"]["config_map_name"] = "chris-q38-f75-p230-stage-v2"
    _rehash_plan(expected)

    assert failed_v1["execution"]["pod_name"] == "chris-q38-f75-p230-stage-v1"
    assert failed_v1["execution"]["config_map_name"] == "chris-q38-f75-p230-stage-v1"
    assert production == expected


def test_v2_acceptance_evidence_is_self_digesting_and_records_one_post_then_get() -> None:
    evidence = json.loads(ACCEPTED_V2.read_text())
    plan = stage.read_plan(PRODUCTION)
    assert evidence["sha256"] == stage.digest_json(stage._unsigned(evidence, "sha256"))
    assert evidence["staging"]["plan_file_sha256"] == stage._digest_bytes(PRODUCTION.read_bytes())
    assert evidence["staging"]["plan_sha256"] == plan["plan_sha256"]
    assert evidence["staging"]["receipt_sha256"] == (
        "sha256:b1f9073be0fb3dca15847d7240d7c62fcad1f1d77c05542b0d8234c8134f7af8"
    )
    assert evidence["staging"]["payload"] == {
        "manifest_sha256": (
            "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
        ),
        "file_count": 29,
        "total_bytes": 55_586_032_099,
        "exact_size_and_sha256_verified": True,
        "create_once_atomic_promotion_verified": True,
        "post_promotion_readback_verified": True,
    }
    registration = evidence["registration"]
    assert registration["post_count"] == 1
    assert registration["second_post_performed"] is False
    assert registration["post_response_contained_exact_full_spec"] is False
    assert registration["acceptance_source"] == "exact_authenticated_get_reconciliation"
    assert registration["desired_registration_sha256"] == stage.digest_json(
        plan["desired_registration"]
    )
    assert registration["get_readback"] == {
        "full_spec_exact": True,
        "phase": "paused",
        "desired_state": "paused",
        "minimum_replicas": 0,
        "active_pods": 0,
        "priority_class": "c1",
        "model_revision": (
            "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029"
        ),
        "gpus_allocated": 0,
    }
    assert evidence["cleanup"]["gpus_held_after_cleanup"] == 0
    assert registration["serving_qualified"] is False


def test_streams_once_then_idempotently_reopens_without_network(tmp_path: Path) -> None:
    plan, objects, environment = _fixture(tmp_path)
    calls: list[str] = []
    first = stage.execute_stage(plan, open_source=_opener(objects, calls), environment=environment)
    final = Path(plan["destination"]["path"])
    assert final.is_dir() and not final.is_symlink()
    assert stat.S_IMODE(final.stat().st_mode) == 0o755
    assert len(calls) == 31
    assert first["payload"]["file_count"] == 29
    assert first["source"]["gpu_reload_verified"] is True
    assert first["source"]["optimizer_steps_executed"] == 0
    assert first["receipt_sha256"] == stage.digest_json(stage._unsigned(first, "receipt_sha256"))
    assert not any(final.parent.glob(f".partial-{final.name}-*"))

    def no_network(_: str):
        raise AssertionError("idempotent readback must not contact Filebrowser")

    second = stage.execute_stage(plan, open_source=no_network, environment={})
    assert second == first


def test_corrupt_stream_is_never_promoted(tmp_path: Path) -> None:
    plan, objects, environment = _fixture(tmp_path)
    payload = next(path for path in objects if path.endswith("payload-00.bin"))
    objects[payload] += b"corruption"
    with pytest.raises(ValueError, match="exceeds EXPORT size|differs from EXPORT"):
        stage.execute_stage(plan, open_source=_opener(objects), environment=environment)
    final = Path(plan["destination"]["path"])
    assert not final.exists()
    partials = list(final.parent.glob(f".partial-{final.name}-*"))
    assert len(partials) == 1
    shutil.rmtree(partials[0])


@pytest.mark.parametrize("collision", ["final_file", "final_directory", "partial_directory"])
def test_existing_collision_is_not_replaced(tmp_path: Path, collision: str) -> None:
    plan, objects, environment = _fixture(tmp_path)
    final = Path(plan["destination"]["path"])
    suffix = plan["plan_sha256"].removeprefix("sha256:")[:12]
    partial = final.parent / f".partial-{final.name}-{suffix}"
    target = partial if collision == "partial_directory" else final
    if collision == "final_file":
        target.write_text("owned by something else")
    else:
        target.mkdir()
        (target / "sentinel").write_text("owned by something else")
    with pytest.raises((FileNotFoundError, NotADirectoryError, ValueError)):
        stage.execute_stage(plan, open_source=_opener(objects), environment=environment)
    assert target.exists()
    sentinel = target / "sentinel" if target.is_dir() else target
    assert sentinel.read_text() == "owned by something else"


def test_symlink_destination_and_payload_are_rejected(tmp_path: Path) -> None:
    plan, objects, environment = _fixture(tmp_path)
    final = Path(plan["destination"]["path"])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    final.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(OSError):
        stage.execute_stage(plan, open_source=_opener(objects), environment=environment)
    assert not list(elsewhere.iterdir())
    final.unlink()

    stage.execute_stage(plan, open_source=_opener(objects), environment=environment)
    victim = final / "payload-00.bin"
    victim.unlink()
    external = tmp_path / "external"
    external.write_bytes(b"content-0")
    victim.symlink_to(external)
    with pytest.raises(OSError):
        stage.execute_stage(plan, open_source=_opener({}), environment={})


@pytest.mark.parametrize("defect", ["export_file", "export_receipt", "gpu_field", "manifest"])
def test_receipt_or_manifest_corruption_fails_before_payload_transfer(
    tmp_path: Path, defect: str
) -> None:
    plan, objects, environment = _fixture(tmp_path)
    export_path = plan["source"]["export_receipt"]["sfs_path"]
    gpu_path = plan["source"]["gpu_check_receipt"]["sfs_path"]
    if defect == "export_file":
        objects[export_path] += b" "
    elif defect == "export_receipt":
        value = json.loads(objects[export_path])
        value["receipt_sha256"] = "0" * 64
        objects[export_path] = json.dumps(value).encode()
        plan["source"]["export_receipt"]["file_sha256"] = stage._digest_bytes(objects[export_path])
        _rehash_plan(plan)
    elif defect == "gpu_field":
        value = json.loads(objects[gpu_path])
        value["gpu_reload_verified"] = False
        objects[gpu_path] = _receipt(stage._unsigned(value, "receipt_sha256"))
        refreshed = json.loads(objects[gpu_path])
        plan["source"]["gpu_check_receipt"]["file_sha256"] = stage._digest_bytes(objects[gpu_path])
        plan["source"]["gpu_check_receipt"]["receipt_sha256"] = (
            "sha256:" + refreshed["receipt_sha256"]
        )
        _rehash_plan(plan)
    else:
        plan["source"]["payload"]["manifest_sha256"] = "sha256:" + "0" * 64
        plan["desired_registration"]["spec"]["model"]["revision"] = "sha256:" + "0" * 64
        _rehash_plan(plan)
    calls: list[str] = []
    with pytest.raises(ValueError):
        stage.execute_stage(plan, open_source=_opener(objects, calls), environment=environment)
    assert len(calls) <= 2
    assert not Path(plan["destination"]["path"]).exists()


def test_registration_drift_is_rejected_even_with_new_plan_digest(tmp_path: Path) -> None:
    plan, _, _ = _fixture(tmp_path)
    plan["desired_registration"]["spec"]["runtime"]["args"][9] = "9999"
    _rehash_plan(plan)
    with pytest.raises(ValueError, match="outside the reviewed clone set"):
        stage.validate_plan(plan)


def test_renderer_is_an_immutable_bounded_zero_gpu_direct_pod() -> None:
    plan = stage.read_plan(PRODUCTION)
    rendered = stage.render_manifest(plan, SOURCE, PRODUCTION)
    objects = list(yaml.safe_load_all(rendered))
    assert [obj["kind"] for obj in objects] == ["ConfigMap", "Pod"]
    config_map, pod = objects
    assert config_map["immutable"] is True
    assert config_map["data"]["model_stage.py"] == SOURCE.read_text()
    assert json.loads(config_map["data"]["plan.json"]) == plan
    spec = pod["spec"]
    assert spec["priorityClassName"] == "c1"
    assert spec["restartPolicy"] == "Never"
    assert spec["activeDeadlineSeconds"] == 5400
    assert spec["automountServiceAccountToken"] is False
    assert "preemptionPolicy" not in spec
    container = spec["containers"][0]
    assert container["image"].endswith(
        "@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
    )
    assert all("nvidia.com/gpu" not in resources for resources in container["resources"].values())
    assert next(item for item in container["env"] if item["name"] == "FILEBROWSER_USER") == {
        "name": "FILEBROWSER_USER",
        "value": "christopher@fleet.so",
    }
    bundle_mounts = [mount for mount in container["volumeMounts"] if mount["name"] == "bundle"]
    assert bundle_mounts == [
        {
            "name": "bundle",
            "mountPath": "/bundle/model_stage.py",
            "subPath": "model_stage.py",
            "readOnly": True,
        },
        {
            "name": "bundle",
            "mountPath": "/bundle/plan.json",
            "subPath": "plan.json",
            "readOnly": True,
        },
    ]
    assert not any(obj["kind"] in {"Job", "RayJob", "Deployment", "Service"} for obj in objects)
    assert pod["metadata"]["namespace"] == "inference"
    assert "api_key" not in rendered.lower()
