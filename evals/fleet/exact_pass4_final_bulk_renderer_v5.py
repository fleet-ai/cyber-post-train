"""Render one independently released final-v5 controller group."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import exact_pass4_bulk_release_renderer_v3 as prior_renderer
from evals.fleet import exact_pass4_final_bulk_package_v5 as package
from evals.fleet import exact_pass4_final_bulk_v5 as bulk


def _group_core_name(name: str, group: str) -> str:
    return f"{name}-{group.replace('_', '-')}"


def _dedicated_service(release: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    replica = "a" if "-a-" in release["group"] else "b"
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"chris-cyber-glm53-dedicated-{replica}-v7",
            "namespace": "fleet-train-jobs",
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": "exact100-final-v5",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": release["receipt_sha256"],
                "cyber-post-train.fleet.ai/ray-cluster-uid": parity["ray_cluster_uid"],
                "cyber-post-train.fleet.ai/service-uid": parity["service_uid"],
            },
        },
        "spec": {
            "type": "ClusterIP",
            "selector": parity["service_selector"],
            "ports": [{"name": "http", "port": 8000, "targetPort": 8000, "protocol": "TCP"}],
        },
    }


def render(
    release: dict[str, Any],
    inventory: dict[str, Any],
    prebulk_terminal: dict[str, Any],
    fresh_duplicate: dict[str, Any],
    root: Path,
    *,
    qualifier_launch_release: dict[str, Any] | None = None,
    qualifier_model: dict[str, Any] | None = None,
    qualifier_terminal: dict[str, Any] | None = None,
    qualifier_job: dict[str, Any] | None = None,
    qualifier_pods: dict[str, Any] | None = None,
    dedicated_parity: dict[str, Any] | None = None,
    dedicated_canary: dict[str, Any] | None = None,
    dedicated_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = bulk.build_release(
        release["group"],
        root,
        release["package_commit"],
        prebulk_terminal=prebulk_terminal,
        fresh_duplicate=fresh_duplicate,
        qualifier_launch_release=qualifier_launch_release,
        qualifier_model=qualifier_model,
        qualifier_terminal=qualifier_terminal,
        qualifier_job=qualifier_job,
        qualifier_pods=qualifier_pods,
        dedicated_parity=dedicated_parity,
        dedicated_canary=dedicated_canary,
        dedicated_runtime=dedicated_runtime,
    )
    if release != expected:
        raise ValueError("final v5 release drifted")
    bulk.hosted.validate_inventory_gate(inventory, root)
    built = package.build_package(root)
    group = release["group"]
    group_core_names = [_group_core_name(name, group) for name in package.CORE_NAMES]
    items: list[dict[str, Any]] = []
    for source_name, target_name in zip(package.CORE_NAMES, group_core_names, strict=True):
        configmap = copy.deepcopy(built["configmaps"][source_name])
        configmap["metadata"]["name"] = target_name
        configmap["metadata"]["annotations"].update(
            {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": release["receipt_sha256"],
            }
        )
        items.append(configmap)
    release_bytes = bulk.canonical(release) + b"\n"
    release_file_sha = bulk.sha256(release_bytes)
    runtime_plans = {}
    for controller in bulk.GROUPS[group]:
        plan = bulk.build_runtime_plan(controller, inventory, root)
        runtime_plans[controller] = plan
        name = bulk.CONTROLLERS[controller]["configmap_name"]
        configmap = copy.deepcopy(built["configmaps"][name])
        configmap["metadata"]["annotations"].update(
            {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/release-receipt-sha256": release["receipt_sha256"],
            }
        )
        configmap["data"]["runtime-plan.json"] = bulk.canonical(plan).decode() + "\n"
        configmap["data"]["release.json"] = release_bytes.decode()
        if len(package.canonical(configmap)) >= package.prior.PACKAGE_OBJECT_LIMIT:
            raise ValueError("final v5 controller ConfigMap exceeds safety budget")
        items.append(configmap)
    if group.endswith("-canary"):
        if dedicated_parity is None:
            raise ValueError("dedicated canary service requires parity")
        items.append(_dedicated_service(release, dedicated_parity))

    original_bulk, original_package = prior_renderer.bulk, prior_renderer.package
    original_core_names = package.CORE_NAMES
    try:
        prior_renderer.bulk = bulk
        prior_renderer.package = package
        package.CORE_NAMES = tuple(group_core_names)
        for controller in bulk.GROUPS[group]:
            manifest = {
                "release_receipt_sha256": release["receipt_sha256"],
                "package_aggregate_sha256": built["controller_manifests"][controller][
                    "aggregate_sha256"
                ],
                "package_commit": release["package_commit"],
                "runtime_gates": {
                    "FINAL_BULK_RELEASE_PATH": "/bootstrap/release.json",
                    "FINAL_BULK_RELEASE_FILE_SHA256": release_file_sha,
                    "FINAL_BULK_PACKAGE_COMMIT": release["package_commit"],
                },
            }
            job = prior_renderer._job(controller, manifest, runtime_plans[controller])  # noqa: SLF001
            for labels in (
                job["metadata"]["labels"],
                job["spec"]["template"]["metadata"]["labels"],
            ):
                labels["cyber-post-train.fleet.ai/experiment"] = "exact100-final-v5"
            for env in job["spec"]["template"]["spec"]["containers"][0]["env"]:
                if env["name"] == "BULK_PLAN":
                    env["name"] = "FINAL_BULK_PLAN"
                elif env["name"] == "BULK_OUTPUT_ROOT":
                    env["name"] = "FINAL_BULK_OUTPUT_ROOT"
                elif env["name"] == "FLEET_API_KEY":
                    env["valueFrom"]["secretKeyRef"]["name"] = bulk.SECRET_NAME
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/secret-uid"] = bulk.SECRET_UID
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/fleet-team-id"] = (
                bulk.FLEET_TEAM_ID
            )
            job["spec"]["template"]["metadata"].setdefault("annotations", {}).update(
                {
                    "cyber-post-train.fleet.ai/secret-uid": bulk.SECRET_UID,
                    "cyber-post-train.fleet.ai/fleet-team-id": bulk.FLEET_TEAM_ID,
                }
            )
            items.append(job)
    finally:
        package.CORE_NAMES = original_core_names
        prior_renderer.bulk = original_bulk
        prior_renderer.package = original_package
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prebulk", type=Path, required=True)
    parser.add_argument("--fresh-duplicate", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--qualifier-launch-release", type=Path)
    parser.add_argument("--qualifier-model", type=Path)
    parser.add_argument("--qualifier-terminal", type=Path)
    parser.add_argument("--qualifier-job", type=Path)
    parser.add_argument("--qualifier-pods", type=Path)
    parser.add_argument("--dedicated-parity", type=Path)
    parser.add_argument("--dedicated-canary", type=Path)
    parser.add_argument("--dedicated-runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def optional(path: Path | None) -> dict[str, Any] | None:
        return bulk.load(path) if path else None

    value = render(
        bulk.load(args.release),
        bulk.load(args.inventory),
        bulk.load(args.prebulk),
        bulk.load(args.fresh_duplicate),
        args.repo.resolve(),
        qualifier_launch_release=optional(args.qualifier_launch_release),
        qualifier_model=optional(args.qualifier_model),
        qualifier_terminal=optional(args.qualifier_terminal),
        qualifier_job=optional(args.qualifier_job),
        qualifier_pods=optional(args.qualifier_pods),
        dedicated_parity=optional(args.dedicated_parity),
        dedicated_canary=optional(args.dedicated_canary),
        dedicated_runtime=optional(args.dedicated_runtime),
    )
    text = yaml.safe_dump(value, sort_keys=False)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
