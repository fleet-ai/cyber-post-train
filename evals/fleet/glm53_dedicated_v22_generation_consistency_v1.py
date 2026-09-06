"""One-generation authority for the GLM v22 serving/controller pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

GENERATION = 22
SCHEMA = "fleet-glm53-dedicated-generation-consistency-v1"
CANARY_SCHEMA = "fleet-glm53-dedicated-v22-generation-consistency-canary-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v22"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v22"
RELEASE_JOB = "chris-glm53-dedicated-v22-r051-a2-release-v1"
CONTROLLER_JOB = "chris-glm53-dedicated-v22-r051-a2-canary-v1"
SERVING_BLOCK = "glm-dedicated-v22-r051-v1"
LEASE_ROOT = "/mnt/sfs/endpoint-leases/opencode11827-dedicated-v22-v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def qualification_fixture() -> dict[str, Any]:
    api_run_id = "ft-run-v22qual"
    service_name = f"{api_run_id}-v22-head-svc"
    origin = f"http://{service_name}.fleet-train-jobs.svc.cluster.local:8000"
    return {
        "schema_version": SCHEMA,
        "generation": GENERATION,
        "server": {
            "title": TITLE,
            "run_dir": RUN_DIR,
            "api_run_id": api_run_id,
            "rayjob_uid": "22222222-2222-4222-8222-222222222221",
            "workload_uid": "22222222-2222-4222-8222-222222222222",
            "pod_uid": "22222222-2222-4222-8222-222222222223",
            "service_uid": "22222222-2222-4222-8222-222222222224",
            "service_name": service_name,
            "service_origin": origin,
        },
        "release": {
            "job_name": RELEASE_JOB,
            "configmap_name": f"{RELEASE_JOB}-run",
            "output_path": f"/mnt/sfs/jobs/{RELEASE_JOB}/RELEASE.json",
            "api_run_id": api_run_id,
            "service_origin": origin,
            "controller_job_name": CONTROLLER_JOB,
            "controller_configmap_name": f"{CONTROLLER_JOB}-run",
        },
        "controller": {
            "job_name": CONTROLLER_JOB,
            "configmap_name": f"{CONTROLLER_JOB}-run",
            "sfs_root": f"/mnt/sfs/jobs/{CONTROLLER_JOB}",
            "server_run_dir": RUN_DIR,
            "api_run_id": api_run_id,
            "service_origin": origin,
            "serving_block": SERVING_BLOCK,
            "lease_root": LEASE_ROOT,
            "endpoint_lease_key": api_run_id,
            "selection_rank": 51,
            "attempt": 2,
        },
        "qualification_only": True,
        "gpu_server_launch_authorized": False,
        "scoring_launch_authorized": False,
    }


def validate(value: dict[str, Any]) -> None:
    server = value.get("server") or {}
    release = value.get("release") or {}
    controller = value.get("controller") or {}
    ids = [server.get(key) for key in ("rayjob_uid", "workload_uid", "pod_uid", "service_uid")]
    try:
        ids_valid = all(uuid.UUID(str(item)).int != 0 for item in ids)
    except ValueError:
        ids_valid = False
    expected_origin = f"http://{server.get('service_name')}.fleet-train-jobs.svc.cluster.local:8000"
    current_fields = [
        server.get("title"),
        server.get("run_dir"),
        server.get("service_name"),
        server.get("service_origin"),
        release.get("job_name"),
        release.get("configmap_name"),
        release.get("output_path"),
        controller.get("job_name"),
        controller.get("configmap_name"),
        controller.get("sfs_root"),
        controller.get("server_run_dir"),
        controller.get("service_origin"),
        controller.get("serving_block"),
        controller.get("lease_root"),
    ]
    if (
        value.get("schema_version") != SCHEMA
        or value.get("generation") != GENERATION
        or server.get("title") != TITLE
        or server.get("run_dir") != RUN_DIR
        or not str(server.get("api_run_id", "")).startswith("ft-run-")
        or not ids_valid
        or server.get("service_origin") != expected_origin
        or release.get("api_run_id") != server.get("api_run_id")
        or release.get("service_origin") != expected_origin
        or release.get("job_name") != RELEASE_JOB
        or release.get("configmap_name") != f"{RELEASE_JOB}-run"
        or release.get("output_path") != f"/mnt/sfs/jobs/{RELEASE_JOB}/RELEASE.json"
        or release.get("controller_job_name") != CONTROLLER_JOB
        or release.get("controller_configmap_name") != f"{CONTROLLER_JOB}-run"
        or controller.get("job_name") != CONTROLLER_JOB
        or controller.get("configmap_name") != f"{CONTROLLER_JOB}-run"
        or controller.get("sfs_root") != f"/mnt/sfs/jobs/{CONTROLLER_JOB}"
        or controller.get("server_run_dir") != RUN_DIR
        or controller.get("api_run_id") != server.get("api_run_id")
        or controller.get("service_origin") != expected_origin
        or controller.get("serving_block") != SERVING_BLOCK
        or controller.get("lease_root") != LEASE_ROOT
        or controller.get("endpoint_lease_key") != server.get("api_run_id")
        or controller.get("selection_rank") != 51
        or controller.get("attempt") != 2
        or any("v20" in str(item) or "v21" in str(item) for item in current_fields)
        or value.get("qualification_only") is not True
        or value.get("gpu_server_launch_authorized") is not False
        or value.get("scoring_launch_authorized") is not False
    ):
        raise RuntimeError("GLM dedicated generation binding drifted")


def run_canary(fixture_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError("generation consistency canary output collision")
    raw = fixture_path.read_bytes()
    value = json.loads(raw)
    validate(value)
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    if any(uuid.UUID(item).int == 0 for item in (job_uid, pod_uid)):
        raise RuntimeError("generation consistency canary identity absent")
    # Cross the same serialized-boundary shape used by projected ConfigMaps.
    roundtrip = json.loads(canonical_json(value))
    validate(roundtrip)
    receipt = {
        "schema_version": CANARY_SCHEMA,
        "status": "PASSED_FULL_GENERATION_BINDING_PATH",
        "generation": GENERATION,
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "fixture_file_sha256": sha256(raw),
        "server_title": value["server"]["title"],
        "server_run_dir": value["server"]["run_dir"],
        "api_run_id": value["server"]["api_run_id"],
        "service_origin": value["server"]["service_origin"],
        "release_job_name": value["release"]["job_name"],
        "release_output_path": value["release"]["output_path"],
        "controller_job_name": value["controller"]["job_name"],
        "controller_configmap_name": value["controller"]["configmap_name"],
        "controller_sfs_root": value["controller"]["sfs_root"],
        "serving_block": value["controller"]["serving_block"],
        "lease_root": value["controller"]["lease_root"],
        "projected_configmap_roundtrip_valid": True,
        "single_generation_valid": True,
        "model_requests": 0,
        "claims_sessions_verifier_or_scoring_calls": 0,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    output_path.parent.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_path.write_bytes(canonical_json(receipt) + b"\n")
    return receipt
