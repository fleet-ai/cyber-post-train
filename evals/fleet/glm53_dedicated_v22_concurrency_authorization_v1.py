"""Post-acceptance authorization gate for the held GLM v22 score-free ramp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as qualification

SCHEMA = "fleet-glm53-dedicated-v22-concurrency-authorization-v1"
CELL_ID = "sha256:2aed501bd3b6db1176ea03687f7bc91c88d205997b00e3a0747dc6a1e1e8b5df"
EXECUTION_ID = "sha256:a9be25eab3efd65f86a832b3c9681dd992d40b51cc1d6d4046212e6797e0ec81"


class AuthorizationError(RuntimeError):
    pass


def build(root: Path, accepted: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    plan = qualification.render(root)
    if (
        accepted.get("receipt_sha256") != crypto.digest_without(accepted, "receipt_sha256")
        or accepted.get("status") != "ACCEPTED_VALIDATED"
        or accepted.get("accepted") is not True
        or accepted.get("credited") is not True
        or accepted.get("cell_id") != CELL_ID
        or accepted.get("execution_id") != EXECUTION_ID
    ):
        raise AuthorizationError("rank51_attempt2_acceptance_invalid")
    binding = qualification.load(root / qualification.BINDING)
    if (
        live.get("server") != plan["server"]
        or live.get("binding") != binding
        or live.get("rayjob_running") is not True
        or live.get("workload_admitted") is not True
        or live.get("workload_preemption_events") != 0
        or live.get("head_pod_ready") is not True
        or live.get("head_pod_restarts") != 0
        or live.get("active_scored_controller_count") != 0
        or live.get("qualification_result_root_absent") is not True
        or live.get("endpoint_lease_available") is not True
        or live.get("request_counter_watchdog_active") is not True
    ):
        raise AuthorizationError("live_score_free_boundary_invalid")
    body = {
        "schema_version": SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_ONLY",
        "rank51_attempt2_accepted_validated_receipt_sha256": accepted["receipt_sha256"],
        "server": plan["server"],
        "no_active_scored_controller": True,
        "qualification_result_root_absent": True,
        "endpoint_lease_exclusive": True,
        "workload_preemption_events": 0,
        "qualification_launch_authorized": True,
        "scored_successor_launch_authorized": False,
        "privacy": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main() -> int:
    raise SystemExit("authorization is created only by the reviewed live-state reconciler")


if __name__ == "__main__":
    main()
