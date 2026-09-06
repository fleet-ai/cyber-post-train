"""Execute the released one-cell GLM dedicated canary."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-glm53-dedicated-v14-canary-release-v1"
BOOTSTRAP_SCHEMA = "fleet-glm53-dedicated-controller-bootstrap-v1"


def _bootstrap_only(root: Path) -> dict[str, Any]:
    """Prove the packaged controller bootstrap and stop before any live route use."""
    evidence = {}
    for key, fallback in (
        ("parity", "/bootstrap/parity.json"),
        ("binding", "/bootstrap/binding.json"),
        ("release", "/bootstrap/release.json"),
    ):
        env_key = f"DEDICATED_{key.upper()}_PATH"
        path = Path(os.environ.get(env_key, fallback))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"bootstrap evidence is not a private regular file: {key}")
        json.loads(path.read_text())
        evidence[key] = self_hosted.sha256(path.read_bytes())
    expected_modules = (
        "fixed_proxy.py",
        "glm53_dedicated_v14_scored_canary_runtime_v1.py",
        "glm53_dedicated_v14_scored_canary_v1.py",
        "hosted_glm_exact_bulk_runtime_v1.py",
        "hosted_glm_exact_bulk_v1.py",
        "opencode_train_sweep_runner.py",
    )
    for name in expected_modules:
        path = root / "evals/fleet" / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"bootstrap install closure is absent: {name}")
        compile(path.read_text(), name, "exec")
    engine.validate_bulk_adapter(canary)
    inventory_receipt = canary.load(inventory.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    rebuilt = canary.build_runtime_plan(
        canary.CONTROLLER, plan["inventory_receipt"], root
    )
    if plan != rebuilt:
        raise RuntimeError("bootstrap runtime plan rebuild drifted")
    image = os.environ.get("AGENT_HARNESS_IMAGE")
    package_sha256 = os.environ.get("DEDICATED_CONTROLLER_PACKAGE_SHA256")
    if image != "chris/opencode:1.18.27-cyber-v1":
        raise RuntimeError("bootstrap harness image drifted")
    if (
        not isinstance(package_sha256, str)
        or len(package_sha256) != 71
        or not package_sha256.startswith("sha256:")
    ):
        raise RuntimeError("bootstrap controller package digest drifted")
    job_uid = str(os.environ.get("JOB_UID", ""))
    pod_uid = str(os.environ.get("POD_UID", ""))
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    receipt = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "status": "PASSED_PRE_MODEL",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "controller_package_sha256": package_sha256,
        "harness_image": image,
        "harness_version": "1.18.27",
        "projected_evidence_copied_to_private_regular_files": True,
        "evidence_file_sha256": evidence,
        "runtime_import_closure_valid": True,
        "bulk_adapter_interface_valid": True,
        "runtime_plan_rebuilt_exactly": True,
        "runtime_plan_sha256": plan["plan_sha256"],
        "docker_build_and_version_check_completed_by_exact_run_sh": True,
        "claim_calls": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    output = Path(os.environ["DEDICATED_BOOTSTRAP_RECEIPT"])
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_bytes(self_hosted.canonical_json(receipt) + b"\n")
    return receipt


def _route_check(plan: dict[str, Any], key: str) -> None:
    origin = plan["model"]["endpoint_origin"].rstrip("/")
    parity_path = Path(os.environ.get("DEDICATED_PARITY_PATH", "/bootstrap/parity.json"))
    canary._validate_evidence(  # noqa: SLF001
        canary.load(parity_path), plan["dedicated_server_binding"], origin
    )
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
    ):
        raise RuntimeError("dedicated canary Fleet team identity drifted")
    with urlopen(origin + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("dedicated GLM service is unhealthy")
    with urlopen(origin + "/v1/models", timeout=30) as response:
        roster = json.load(response)
    if [row.get("id") for row in roster.get("data", [])] != ["glm-5.3"]:
        raise RuntimeError("dedicated GLM served model drifted")


def _runtime_gate(plan: dict[str, Any]) -> None:
    release_path = Path(os.environ.get("DEDICATED_RELEASE_PATH", "/bootstrap/release.json"))
    release = canary.load(release_path)
    item = plan["attempts"][0]
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("status") != "CLEAR"
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell_id") != item["cell_id"]
        or release.get("execution_id") != item["execution_id"]
        or release.get("all_four_rank_cells_unstarted") is not True
        or release.get("fleet_session_collisions") != 0
        or release.get("global_claim_collisions") != 0
        or release.get("kubernetes_object_collisions") != 0
        or release.get("checked_immediately_before_create") is not True
        or release.get("receipt_sha256")
        != self_hosted.digest_without(release, "receipt_sha256")
    ):
        raise RuntimeError("dedicated canary release gate drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory_receipt = canary.load(inventory.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    engine.bulk = canary
    return engine.run_controller(
        plan,
        out=canary.SFS_ROOT,
        proxy=proxy,
        route_check=_route_check,
        runtime_gate_check=_runtime_gate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("DEDICATED_BOOTSTRAP_ONLY") == "1":
        _bootstrap_only(args.repo.resolve())
        return 0
    run(args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
