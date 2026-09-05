"""Authenticated preview-only gate for the held GLM5.3 TP8 v11 successor."""

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
from evals.fleet import glm53_dedicated_v11 as v11
from evals.fleet import jobs_api_topology_guard as topology
from evals.fleet import priority_preemption_guard as priority_guard
from evals.fleet import self_hosted


def require_contract(openapi: dict[str, Any]) -> None:
    paths = openapi.get("paths") or {}
    if not all(route in paths for route in ("/v1/runs", "/v1/runs/preview", "/v1/runs/{name}")):
        raise RuntimeError("Jobs API contract routes are absent")
    if "delete" not in paths["/v1/runs/{name}"]:
        raise RuntimeError("Jobs API release route is absent")
    schema = ((openapi.get("components") or {}).get("schemas") or {}).get("RLJobConfig") or {}
    properties = schema.get("properties") or {}
    mode_options = {
        option
        for branch in (properties.get("topology_mode") or {}).get("anyOf") or []
        for option in branch.get("enum") or []
    }
    if mode_options != {"required", "preferred"} or "topology_level" not in properties:
        raise RuntimeError("Jobs API topology request contract drifted")
    if "queue" in properties or "resource_flavor" in properties:
        raise RuntimeError("review the newly exposed queue or flavor selector before using it")


def preview_identity(
    manifest_yaml: str, payload: dict[str, Any], package: Any = v11
) -> dict[str, Any]:
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    if len(manifests) != 1 or manifests[0].get("kind") != "RayJob":
        raise RuntimeError("Jobs API preview did not render exactly one RayJob")
    rayjob = manifests[0]
    spec = rayjob.get("spec") or {}
    cluster = spec.get("rayClusterSpec") or {}
    if cluster.get("workerGroupSpecs"):
        raise RuntimeError("one-node v11 unexpectedly rendered a worker group")
    head = (cluster.get("headGroupSpec") or {}).get("template") or {}
    annotations = (head.get("metadata") or {}).get("annotations") or {}
    pod = head.get("spec") or {}
    containers = pod.get("containers") or []
    if len(containers) != 1:
        raise RuntimeError("Jobs API preview rendered an unexpected GPU container shape")
    container = containers[0]
    env = {row.get("name"): row.get("value") for row in container.get("env") or []}
    resources = container.get("resources") or {}
    expected = {
        "image": payload["image"],
        "priority_class": package.PRIORITY_CLASS,
        "privileged": True,
        "run_dir": payload["run_dir"],
        "gpus": 8,
        "image_pull_secrets": ["ghcr-pull"],
        "queue": package.QUEUE,
        "suspended_for_admission": True,
        "topology_mode": package.TOPOLOGY_MODE,
        "topology_level": package.TOPOLOGY_LEVEL,
        "command_sha256": self_hosted.sha256(payload["command"].encode()),
    }
    gpu_request = (resources.get("requests") or {}).get("nvidia.com/gpu")
    gpu_limit = (resources.get("limits") or {}).get("nvidia.com/gpu")
    actual = {
        "image": container.get("image"),
        "priority_class": pod.get("priorityClassName"),
        "privileged": (container.get("securityContext") or {}).get("privileged"),
        "run_dir": env.get("RUN_DIR"),
        "gpus": gpu_request if gpu_request == gpu_limit else None,
        "image_pull_secrets": [row.get("name") for row in pod.get("imagePullSecrets") or []],
        "queue": ((rayjob.get("metadata") or {}).get("labels") or {}).get(
            "kueue.x-k8s.io/queue-name"
        ),
        "suspended_for_admission": spec.get("suspend"),
        "topology_mode": "required"
        if annotations.get("kueue.x-k8s.io/podset-required-topology")
        else None,
        "topology_level": annotations.get("kueue.x-k8s.io/podset-required-topology"),
        "command_sha256": self_hosted.sha256(str(spec.get("entrypoint") or "").encode()),
    }
    if actual != expected:
        raise RuntimeError("Jobs API rendered identity drifted")
    return actual


def _get_json(kind: str, name: str | None = None) -> dict[str, Any]:
    args = ["get", kind]
    if name:
        args.append(name)
    args.extend(["-o", "json"])
    return json.loads(shared._kubectl(*args))


def live_gate(root: Path, package: Any = v11) -> dict[str, Any]:
    value = package.spec(root)
    payload = package.payload(value, root)
    token = shared._token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    with httpx.Client(base_url=shared.BASE_URL, headers=headers, timeout=60) as client:
        whoami = client.get("/v1/whoami")
        whoami.raise_for_status()
        if not (whoami.json().get("login") or whoami.json().get("email")):
            raise RuntimeError("Jobs API identity is absent")
        openapi = client.get("/v1/openapi.json")
        openapi.raise_for_status()
        require_contract(openapi.json())
        runs = shared._runs(client)
        title_matches = sum(row.get("title") == package.TITLE for row in runs)
        run_dir_matches = sum(row.get("run_dir") == package.RUN_DIR for row in runs)
        preview = client.post("/v1/runs/preview", json=payload)
        preview.raise_for_status()
        manifest_yaml = preview.json()["manifest_yaml"]
        rendered = preview_identity(manifest_yaml, payload, package)

    local_queue = _get_json("localqueue.kueue.x-k8s.io", package.QUEUE)
    cluster_queue_name = local_queue["spec"]["clusterQueue"]
    cluster_queue = json.loads(
        subprocess.run(
            ["kubectl", "get", "clusterqueue.kueue.x-k8s.io", cluster_queue_name, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    flavors_list = json.loads(
        subprocess.run(
            ["kubectl", "get", "resourceflavors.kueue.x-k8s.io", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    topologies_list = json.loads(
        subprocess.run(
            ["kubectl", "get", "topologies.kueue.x-k8s.io", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    flavors = {row["metadata"]["name"]: row for row in flavors_list.get("items") or []}
    topologies = {row["metadata"]["name"]: row for row in topologies_list.get("items") or []}
    topology_gate = topology.require_exact_topology_candidate(
        manifest_yaml,
        local_queue,
        cluster_queue,
        flavors,
        topologies,
        expected_queue=package.QUEUE,
        expected_level=package.TOPOLOGY_LEVEL,
        expected_podsets=("head",),
    )
    if [row["name"] for row in topology_gate["eligible_flavors"]] != ["b300-training"]:
        raise RuntimeError("v11 did not resolve the exact reviewed topology flavor")

    inventory = json.loads(
        shared._kubectl(
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,jobs,pods,services",
            "-o",
            "json",
        )
    )
    serialized = json.dumps(inventory, sort_keys=True)
    kubernetes_matches = int(package.TITLE in serialized) + int(package.RUN_DIR in serialized)
    observer_run_dir = shared._observer_sfs_path(package.RUN_DIR)
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
            observer_run_dir,
        ],
        check=False,
        capture_output=True,
    )
    priority_classes = json.loads(
        subprocess.run(
            ["kubectl", "get", "priorityclasses.scheduling.k8s.io", "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    selected_priority = priority_guard.select_highest_nonpreempting(
        priority_classes.get("items") or [], allowed_names=package.API_PRIORITY_CLASSES
    )
    if (
        title_matches
        or run_dir_matches
        or kubernetes_matches
        or absent.returncode != 0
        or selected_priority["name"] != package.PRIORITY_CLASS
    ):
        raise RuntimeError("v11 create-once or nonpreemption gate failed")
    body = {
        "schema_version": f"fleet-glm53-dedicated-serving-{package.VERSION}-preview-receipt-v1",
        "status": "PASSED_PREVIEW_ONLY",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "rendered": rendered,
        "topology_gate": topology_gate,
        "selected_priority": selected_priority,
        "jobs_api_title_matches": title_matches,
        "jobs_api_run_dir_matches": run_dir_matches,
        "kubernetes_identity_matches": kubernetes_matches,
        "sfs_run_dir_exists": False,
        "launch_authorized": True,
        "mutations": 0,
        "two_node_tp16_allowed": False,
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    gate = live_gate(Path.cwd())
    if args.command == "preview":
        shared._write_once(args.output, gate)
        print(json.dumps({"status": gate["status"], "receipt_sha256": gate["receipt_sha256"]}))
        return 0

    payload = v11.payload(v11.spec(Path.cwd()), Path.cwd())
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
        raise RuntimeError("Jobs API response omitted the exact run identity")
    receipt = {
        "schema_version": "fleet-glm53-dedicated-serving-v11-submission-v1",
        "status": "SUBMITTED",
        "submitted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_run_id": api_run_id,
        "title": v11.TITLE,
        "run_dir": v11.RUN_DIR,
        "request_sha256": gate["request_sha256"],
        "live_gate_receipt_sha256": gate["receipt_sha256"],
        "eligible_flavor_uids": [row["uid"] for row in gate["topology_gate"]["eligible_flavors"]],
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
