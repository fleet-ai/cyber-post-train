"""Create one UID-bound drain request at an OpenCode campaign claim boundary."""

from __future__ import annotations

import argparse
import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted
from evals.fleet.opencode_train_sweep_runner import (
    ALLOWED_DRAIN_REASONS,
    DRAIN_REQUEST_SCHEMA,
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def request_drain(
    *,
    plan: dict[str, Any],
    root: Path,
    target_job_uid: str,
    target_pod_uid: str,
    reason: str,
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("campaign output root is absent or unsafe")
    frozen = load_object(root / "PLAN.json")
    if plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256") or frozen != plan:
        raise RuntimeError("campaign drain plan binding drifted")
    if not target_job_uid or not target_pod_uid or reason not in ALLOWED_DRAIN_REASONS:
        raise ValueError("campaign drain requires exact Job/Pod UIDs and a reason")

    request = {
        "schema_version": DRAIN_REQUEST_SCHEMA,
        "requested_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "plan_sha256": plan["plan_sha256"],
        "campaign_id": plan["campaign_id"],
        "target_job_uid": target_job_uid,
        "target_pod_uid": target_pod_uid,
        "reason": reason,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    request["request_sha256"] = self_hosted.digest_without(request, "request_sha256")

    gate_path = root / ".attempt-claim-gate.lock"
    with gate_path.open("a+b") as gate:
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        if (root / "ACCEPTED.json").exists() or (root / "DRAINED.json").exists():
            raise RuntimeError("campaign is already terminal")
        self_hosted.write_json_once(root / "DRAIN-REQUEST.json", request)
    return request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-job-uid", required=True)
    parser.add_argument("--target-pod-uid", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    receipt = request_drain(
        plan=load_object(args.plan),
        root=args.out_dir,
        target_job_uid=args.target_job_uid,
        target_pod_uid=args.target_pod_uid,
        reason=args.reason,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
