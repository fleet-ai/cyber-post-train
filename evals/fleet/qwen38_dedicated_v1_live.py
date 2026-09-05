"""Authenticated preview and create-once submit rail for dedicated Qwen3.8 v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dedicated_v1 as v1
from evals.fleet import self_hosted

BASE_URL = shared.BASE_URL
NAMESPACE = shared.NAMESPACE
OBSERVER_POD = shared.OBSERVER_POD
STAGING_RECEIPT_SHA = "9a680f714de5c72e60dc6b08f363a8c82658dc1540bbb9aab90a3b8939445869"
MODEL_METADATA = {
    "indexed_shard_count": 18,
    "indexed_weight_bytes": 55563006776,
    "stage_marker_sha256": "2c8d9eb23fef165c20f05bc02f635a3d518fb22baa86a7690596ac4f06e8c2c5",
    "weights_index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    "indexed_shard_sizes_sha256": (
        "1d3a3e7f7c20a237b2173c5310dcdad6f80f4de4042bccde2bbb5b1619319d62"
    ),
    "config_sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
}


def _staging_identity() -> dict[str, Any]:
    program = r"""import hashlib,json
from pathlib import Path
receipt=Path("/shared/jobs/chris-cyber-evalstage-verify-glm53-q38-v4/STAGED.json")
root=Path("/shared/models/qwen3.8-27b-1d4bf0f2")
index_bytes=(root/"model.safetensors.index.json").read_bytes()
index=json.loads(index_bytes)
shards=sorted(set(index.get("weight_map",{}).values()))
sizes={name:(root/name).stat().st_size for name in shards}
print(json.dumps({
 "staging_receipt_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest(),
 "stage_marker_sha256":hashlib.sha256((root/".stage-complete").read_bytes().strip()).hexdigest(),
 "weights_index_sha256":hashlib.sha256(index_bytes).hexdigest(),
 "indexed_shard_count":len(shards),
 "indexed_weight_bytes":sum(sizes.values()),
 "indexed_shard_sizes_sha256":hashlib.sha256(json.dumps(sizes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
 "config_sha256":hashlib.sha256((root/"config.json").read_bytes()).hexdigest(),
},sort_keys=True))"""
    value = json.loads(shared._kubectl("exec", OBSERVER_POD, "--", "python3", "-c", program))
    expected = {"staging_receipt_sha256": STAGING_RECEIPT_SHA, **MODEL_METADATA}
    if value != expected:
        raise RuntimeError("exact staged Qwen metadata drifted")
    return value


def _preview_identity(manifest_yaml: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    if len(manifests) != 1 or manifests[0].get("kind") != "RayJob":
        raise RuntimeError("Jobs API preview did not render exactly one RayJob")
    rayjob = manifests[0]
    spec = rayjob.get("spec") or {}
    head = ((spec.get("rayClusterSpec") or {}).get("headGroupSpec") or {}).get("template") or {}
    pod = head.get("spec") or {}
    containers = pod.get("containers") or []
    if len(containers) != 1:
        raise RuntimeError("Jobs API preview rendered an unexpected GPU container shape")
    container = containers[0]
    env = {row.get("name"): row.get("value") for row in container.get("env") or []}
    resources = container.get("resources") or {}
    gpu_request = (resources.get("requests") or {}).get("nvidia.com/gpu")
    gpu_limit = (resources.get("limits") or {}).get("nvidia.com/gpu")
    actual = {
        "image": container.get("image"),
        "priority_class": pod.get("priorityClassName"),
        "privileged": (container.get("securityContext") or {}).get("privileged"),
        "run_dir": env.get("RUN_DIR"),
        "gpus": gpu_request if gpu_request == gpu_limit else None,
        "image_pull_secrets": [row.get("name") for row in pod.get("imagePullSecrets") or []],
        "queue": (rayjob.get("metadata", {}).get("labels") or {}).get("kueue.x-k8s.io/queue-name"),
        "suspended_for_admission": spec.get("suspend"),
        "command_sha256": self_hosted.sha256(str(spec.get("entrypoint") or "").encode()),
    }
    expected = {
        "image": payload["image"],
        "priority_class": "fleet-infra-quiet",
        "privileged": True,
        "run_dir": payload["run_dir"],
        "gpus": 1,
        "image_pull_secrets": ["ghcr-pull"],
        "queue": "training-lq",
        "suspended_for_admission": True,
        "command_sha256": self_hosted.sha256(payload["command"].encode()),
    }
    if actual != expected:
        raise RuntimeError("Jobs API rendered Qwen identity drifted")
    return actual


def live_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v1.payload(v1.spec(root), root)
    token = shared._token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60) as client:
        whoami = client.get("/v1/whoami")
        whoami.raise_for_status()
        if not (whoami.json().get("login") or whoami.json().get("email")):
            raise RuntimeError("Jobs API identity is absent")
        openapi = client.get("/v1/openapi.json")
        openapi.raise_for_status()
        paths = openapi.json().get("paths") or {}
        if not all(route in paths for route in ("/v1/runs", "/v1/runs/preview", "/v1/runs/{name}")):
            raise RuntimeError("Jobs API contract routes are absent")
        if "delete" not in paths["/v1/runs/{name}"]:
            raise RuntimeError("Jobs API release route is absent")
        runs = shared._runs(client)
        title_matches = sum(row.get("title") == v1.TITLE for row in runs)
        run_dir_matches = sum(row.get("run_dir") == v1.RUN_DIR for row in runs)
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        rendered = _preview_identity(preview.json()["manifest_yaml"], payload)
    inventory = json.loads(
        shared._kubectl(
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services",
            "-o",
            "json",
        )
    )
    serialized = json.dumps(inventory, sort_keys=True)
    kubernetes_matches = int(v1.TITLE in serialized) + int(v1.RUN_DIR in serialized)
    observer_run_dir = shared._observer_sfs_path(v1.RUN_DIR)
    absent = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            OBSERVER_POD,
            "--",
            "test",
            "!",
            "-e",
            observer_run_dir,
        ],
        check=False,
        capture_output=True,
    )
    if title_matches or run_dir_matches or kubernetes_matches or absent.returncode != 0:
        raise RuntimeError("Qwen v1 create-once identity is not absent")
    priority = json.loads(
        subprocess.run(
            ["kubectl", "get", "priorityclass", "fleet-infra-quiet", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if priority.get("preemptionPolicy") != "Never":
        raise RuntimeError("fleet-infra-quiet is not nonpreempting")
    staging = _staging_identity()
    gate = {
        "schema_version": "fleet-qwen38-dedicated-serving-v1-live-gate-v1",
        "status": "PASSED",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "jobs_api_contract_valid": True,
        "jobs_api_identity_valid": True,
        "jobs_api_title_matches": title_matches,
        "jobs_api_run_dir_matches": run_dir_matches,
        "kubernetes_identity_matches": kubernetes_matches,
        "sfs_run_dir_exists": False,
        "sfs_observer_run_dir": observer_run_dir,
        "staging_identity": staging,
        "rendered": rendered,
        "hardware_preflight_embedded": True,
        "pre_ready_exit_receipt_embedded": True,
        "preemption_policy": "Never",
        "release_route": "DELETE /v1/runs/{name}",
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    gate["receipt_sha256"] = self_hosted.digest_without(gate, "receipt_sha256")
    return payload, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("refusing existing output receipt")
    if args.command == "submit":
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise RuntimeError("submit requires a clean immutable worktree")
    payload, gate = live_gate(root)
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": "PASSED", "receipt_sha256": gate["receipt_sha256"]}))
        return 0
    token = shared._token()
    with httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    ) as client:
        response = client.post("/v1/runs", json=payload)
        response.raise_for_status()
        if response.status_code != 202:
            raise RuntimeError("Jobs API submit did not return 202")
        api_run_id = response.json().get("name")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("Jobs API response omitted run identity")
    receipt = {
        "schema_version": "fleet-qwen38-dedicated-serving-v1-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v1.TITLE,
        "run_dir": v1.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "route": "POST /v1/runs",
        "http_status": 202,
        "scored_tasks_launched": 0,
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    shared._write_once(args.output, receipt)
    print(json.dumps({"api_run_id": api_run_id, "status": "SUBMITTED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
