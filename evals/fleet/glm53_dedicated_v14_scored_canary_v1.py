"""Held rank-51 GLM v14 scored canary plan.

The first attempt is the only executable canary cell.  All four attempts of
rank 51 are bound as one dedicated-serving block so a later release observer
can prove the complete task was wholly unstarted before any claim is written.
"""

from __future__ import annotations

import copy
import os
import re
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v14-scored-canary-plan-v1"
CONTROLLER = "glm-hosted-s3"
JOB_NAME = "chris-glm53-dedicated-v18-r051-a1-canary-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = Path(source.CLAIM_ROOT)
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v18-v1")
SELECTION_RANK = 51
CANARY_ATTEMPT = 1
SERVING_BLOCK = "glm-dedicated-v18-r051-v2"
PARITY_SCHEMA = "fleet-opencode-actual-harness-hosted-parity-v1"
MODEL_REVISION = "30333038ada1f1dacb294a93270305a890b50c14"
SERVICE_RE = re.compile(
    r"^http://[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    r"\.fleet-train-jobs\.svc\.cluster\.local:8000$"
)

CONTROLLERS = {CONTROLLER: copy.deepcopy(source.CONTROLLERS[CONTROLLER])}
SHA256_RE = source.SHA256_RE
COMMIT_RE = source.COMMIT_RE
CANARY_GATE_SCHEMA = source.predecessor.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = source.predecessor.RECONCILIATION_GATE_SCHEMA


def load(path: Path) -> dict[str, Any]:
    return source.load(path)


def validate_inventory_gate(value: dict[str, Any], root: Path) -> None:
    source.validate_inventory_gate(value, root)


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    """Expose the complete frozen universe required by the bulk adapter contract."""
    return source.validate_all(root)


def _validate_evidence(
    parity: dict[str, Any], binding: dict[str, Any], service_origin: str
) -> None:
    parity_sha = parity.get("receipt_sha256")
    binding_sha = self_hosted.sha256(self_hosted.canonical_json(binding))
    if (
        parity.get("schema_version") != PARITY_SCHEMA
        or parity.get("status") != "PASSED_NON_SCORED"
        or parity_sha != self_hosted.digest_without(parity, "receipt_sha256")
        or parity.get("execution", {}).get("task_instance_session_verifier_scoring_calls") != 0
        or parity.get("harness", {}).get("version") != "1.18.27"
        or parity.get("harness", {}).get("context_window_size") != 262144
        or parity.get("tool_contract", {}).get("names") != ["bash", "submit_report"]
        or parity.get("endpoint", {}).get("server_binding") != binding
        or parity.get("endpoint", {}).get("server_binding_sha256")
        != binding_sha
    ):
        raise ValueError("dedicated v14 parity evidence drifted")
    if (
        set(binding)
        != {
            "api_run_id", "context_length", "head_pod_uid", "model_revision",
            "rayjob_uid", "served_id", "service_uid",
        }
        or re.fullmatch(r"ft-run-[a-z0-9]+", str(binding.get("api_run_id"))) is None
        or binding.get("context_length") != 262144
        or binding.get("model_revision") != MODEL_REVISION
        or binding.get("served_id") != "glm-5.3"
    ):
        raise ValueError("dedicated v14 server binding drifted")
    for field in ("head_pod_uid", "rayjob_uid", "service_uid"):
        try:
            uuid.UUID(str(binding[field]))
        except ValueError as exc:
            raise ValueError("dedicated v14 server UID binding drifted") from exc
    if SERVICE_RE.fullmatch(service_origin) is None:
        raise ValueError("dedicated v14 service origin is not cluster-local HTTP")


def _transform(
    plan: dict[str, Any],
    *,
    parity: dict[str, Any],
    binding: dict[str, Any],
    service_origin: str,
    runtime: bool,
) -> dict[str, Any]:
    _validate_evidence(parity, binding, service_origin)
    rank = [row for row in plan["attempts"] if row["selection_rank"] == SELECTION_RANK]
    if [row["attempt"] for row in rank] != [1, 2, 3, 4]:
        raise ValueError("dedicated v14 whole-task authority drifted")
    canary = copy.deepcopy(rank[0])
    canary["ordinal"] = 1
    body = copy.deepcopy(plan)
    body.update(
        {
            "schema_version": SCHEMA,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
            "task_count": 1,
            "new_session_count": 1,
            "attempts": [canary],
            "serving_load_block": SERVING_BLOCK,
            "launch_authorized": runtime,
            "dedicated_server_binding": binding,
            "dedicated_parity_receipt_sha256": parity["receipt_sha256"],
            "whole_task_reservation": {
                "selection_rank": SELECTION_RANK,
                "cell_ids": [row["cell_id"] for row in rank],
                "execution_ids": [row["execution_id"] for row in rank],
                "all_four_unstarted_required_at_release": True,
                "remaining_attempts_require_post_canary_release": [2, 3, 4],
            },
        }
    )
    body["model"]["endpoint_origin"] = service_origin
    body["model"]["serving_kind"] = "dedicated_uid_bound_inference"
    body["execution"]["workers"] = 1
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": binding["api_run_id"],
        "maximum_streams": 1,
    }
    serving_run_dir = binding.get(
        "run_dir", "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v18"
    )
    if not str(serving_run_dir).startswith("/mnt/sfs/jobs/chris-cyber-evalserve-glm53-"):
        raise ValueError("dedicated v14 serving run directory drifted")
    body["execution"]["traffic_heartbeat_path"] = (
        f"{serving_run_dir}/lifecycle/traffic-stream-1"
    )
    body["serving"] = {
        "kind": "dedicated_uid_bound_inference",
        "serving_block": SERVING_BLOCK,
        "service_origin": service_origin,
        **binding,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "pool_with_hosted_without_review": False,
    }
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == SELECTION_RANK]
        if len(body["tasks"]) != 1:
            raise ValueError("dedicated v14 runtime task drifted")
    body.pop("plan_sha256", None)
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(
    root: Path, *, service_origin: str, parity_path: Path, binding_path: Path
) -> dict[str, Any]:
    parity = load(parity_path)
    binding = load(binding_path)
    return _transform(
        source.validate_all(root)[CONTROLLER],
        parity=parity,
        binding=binding,
        service_origin=service_origin,
        runtime=False,
    )


def build_runtime_plan(
    controller: str,
    inventory_receipt: dict[str, Any],
    root: Path,
    *,
    service_origin: str | None = None,
    parity_path: Path | None = None,
    binding_path: Path | None = None,
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown dedicated v14 canary controller")
    service_origin = service_origin or os.environ.get("DEDICATED_SERVICE_ORIGIN")
    parity_path = parity_path or Path(os.environ.get("DEDICATED_PARITY_PATH", "/bootstrap/parity.json"))
    binding_path = binding_path or Path(os.environ.get("DEDICATED_BINDING_PATH", "/bootstrap/binding.json"))
    if service_origin is None:
        raise ValueError("dedicated v14 service origin is absent")
    parity = load(parity_path)
    binding = load(binding_path)
    return _transform(
        source.build_runtime_plan(CONTROLLER, inventory_receipt, root),
        parity=parity,
        binding=binding,
        service_origin=service_origin,
        runtime=True,
    )
