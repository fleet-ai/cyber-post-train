"""Create-once Jobs API rail for Qwen TP1-d."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import qwen38_dedicated_v6_live as v6_live
from evals.fleet import qwen38_dedicated_v7 as v7
from evals.fleet import self_hosted


def _project_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v6_live._project_shape(rows)


def live_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v7.payload(v7.spec(root), root)
    headers = {"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"}
    with httpx.Client(base_url=shared.BASE_URL, headers=headers, timeout=60) as client:
        runs = shared._runs(client)
        if any(row.get("title") == v7.TITLE or row.get("run_dir") == v7.RUN_DIR for row in runs):
            raise RuntimeError("Qwen v7 Jobs API identity already exists")
        shape = _project_shape(runs)
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        rendered = v6_live._preview_identity(preview.json()["manifest_yaml"], payload)
    inventory = json.loads(
        shared._kubectl(
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services",
            "-o",
            "json",
        )
    )
    serialized = json.dumps(inventory, sort_keys=True)
    if v7.TITLE in serialized or v7.RUN_DIR in serialized:
        raise RuntimeError("Qwen v7 Kubernetes identity already exists")
    absent = subprocess.run(
        [
            "kubectl",
            "-n",
            shared.NAMESPACE,
            "exec",
            shared.OBSERVER_POD,
            "--",
            "test",
            "!",
            "-e",
            shared._observer_sfs_path(v7.RUN_DIR),
        ],
        check=False,
        capture_output=True,
    )
    if absent.returncode != 0:
        raise RuntimeError("Qwen v7 SFS run directory already exists")
    gate = {
        "schema_version": "fleet-qwen38-dedicated-serving-v7-live-gate-v1",
        "status": "PASSED",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "rendered": rendered,
        "project_resource_shape": shape,
        "preemption_policy": "Never",
        "traffic_idle_guard_seconds": 600,
        "fresh_actual_opencode_parity_required_before_scoring": True,
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
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("refusing existing output receipt")
    if (
        args.command == "submit"
        and subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    ):
        raise RuntimeError("submit requires a clean immutable worktree")
    payload, gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": "PASSED", "receipt_sha256": gate["receipt_sha256"]}))
        return 0
    with httpx.Client(
        base_url=shared.BASE_URL,
        headers={"Authorization": f"Bearer {shared._token()}", "Accept": "application/json"},
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
        "schema_version": "fleet-qwen38-dedicated-serving-v7-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v7.TITLE,
        "run_dir": v7.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "project_resource_shape": gate["project_resource_shape"],
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
