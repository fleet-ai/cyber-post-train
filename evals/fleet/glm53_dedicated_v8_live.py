"""Authenticated create-once preview and submit rail for GLM5.3 v8."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.fleet import glm53_dedicated_v8 as v8
from evals.fleet import self_hosted

BASE_URL = "https://api.ft.flt.build"
NAMESPACE = "fleet-train-jobs"
OBSERVER_POD = "allie-dev"
RUNTIME_SFS_PREFIX = "/mnt/sfs/"
OBSERVER_SFS_PREFIX = "/shared/"


def _token() -> str:
    value = subprocess.run(
        ["gh", "auth", "token", "--hostname", "github.com"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not value:
        raise RuntimeError("GitHub CLI returned no Jobs API credential")
    return value


def _kubectl(*args: str) -> str:
    return subprocess.run(
        ["kubectl", "-n", NAMESPACE, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.chmod(0o400)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _observer_sfs_path(runtime_path: str) -> str:
    """Translate the Jobs API mount into allie-dev's read-only SFS mount."""
    if not runtime_path.startswith(RUNTIME_SFS_PREFIX):
        raise ValueError("runtime path is outside the exact SFS mount")
    relative = runtime_path.removeprefix(RUNTIME_SFS_PREFIX)
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("runtime SFS path is unsafe")
    return OBSERVER_SFS_PREFIX + relative


def _runs(client: httpx.Client) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = client.get("/v1/runs", params={"limit": 200, "offset": offset})
        response.raise_for_status()
        page = response.json()
        rows = page.get("items") or []
        if not isinstance(rows, list):
            raise RuntimeError("Jobs API run inventory is invalid")
        items.extend(row for row in rows if isinstance(row, dict))
        if not page.get("has_more"):
            return items
        if not rows:
            raise RuntimeError("Jobs API pagination made no progress")
        offset += len(rows)


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
    expected = {
        "image": payload["image"],
        "priority_class": "fleet-infra-quiet",
        "privileged": True,
        "run_dir": payload["run_dir"],
        "gpus": 8,
        "image_pull_secrets": ["ghcr-pull"],
        "queue": "training-lq",
        "suspended_for_admission": True,
        "command_sha256": self_hosted.sha256(payload["command"].encode()),
    }
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
    if actual != expected:
        raise RuntimeError("Jobs API rendered identity drifted")
    return actual


def live_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = v8.spec(root)
    payload = v8.payload(value, root)
    token = _token()
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
        runs = _runs(client)
        title_matches = sum(row.get("title") == v8.TITLE for row in runs)
        run_dir_matches = sum(row.get("run_dir") == v8.RUN_DIR for row in runs)
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        rendered = _preview_identity(preview.json()["manifest_yaml"], payload)

    inventory = json.loads(
        _kubectl(
            "get",
            ("rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services"),
            "-o",
            "json",
        )
    )
    serialized = json.dumps(inventory, sort_keys=True)
    kubernetes_matches = int(v8.TITLE in serialized) + int(v8.RUN_DIR in serialized)
    observer_uid = _kubectl("get", "pod", OBSERVER_POD, "-o", "jsonpath={.metadata.uid}")
    observer_run_dir = _observer_sfs_path(v8.RUN_DIR)
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
        raise RuntimeError("v8 create-once identity is not absent")
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
    gate = {
        "schema_version": "fleet-glm53-dedicated-serving-v8-live-gate-v1",
        "status": "PASSED",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "jobs_api_contract_valid": True,
        "jobs_api_identity_valid": True,
        "jobs_api_title_matches": title_matches,
        "jobs_api_run_dir_matches": run_dir_matches,
        "kubernetes_identity_matches": kubernetes_matches,
        "sfs_run_dir_exists": False,
        "sfs_runtime_run_dir": v8.RUN_DIR,
        "sfs_observer_run_dir": observer_run_dir,
        "sfs_observer_pod_uid": observer_uid,
        "rendered": rendered,
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
        _write_once(args.output, gate)
        print(json.dumps({"status": "PASSED", "receipt_sha256": gate["receipt_sha256"]}))
        return 0

    token = _token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60) as client:
        response = client.post("/v1/runs", json=payload)
        response.raise_for_status()
        if response.status_code != 202:
            raise RuntimeError("Jobs API submit did not return 202")
        api_run_id = response.json().get("name")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise RuntimeError("Jobs API response omitted the exact run identity")
    receipt = {
        "schema_version": "fleet-glm53-dedicated-serving-v8-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v8.TITLE,
        "run_dir": v8.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "route": "POST /v1/runs",
        "http_status": 202,
        "scored_tasks_launched": 0,
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    _write_once(args.output, receipt)
    print(json.dumps({"api_run_id": api_run_id, "status": "SUBMITTED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
