"""Minimal create-once Generation-12 hosted OpenCode cell runner.

Generation 12 preserves the exact G11 model/task/harness treatment but restores
the already accepted v2 smoke Pod bootstrap.  Sanitized stage receipts make a
pre-rollout failure diagnosable without reading application logs.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import opencode_train_sweep_runner as sweep
from evals.fleet import self_hosted

TEAM = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
TOMBSTONE_SCHEMA = "fleet-opencode-generation11-preclaim-preoutput-tombstone-v1"
TOMBSTONE_STATUS = "G11_FAILED_BEFORE_CLAIM_INSTANCE_MODEL_SESSION_VERIFIER"
STAGES = (
    "01-before-static-validation",
    "02-after-static-validation",
    "03-before-docker-ready",
    "04-docker-ready",
    "05-before-docker-build",
    "06-after-docker-build",
    "07-before-duplicate-check",
    "08-after-duplicate-check",
    "09-before-live-route-check",
    "10-after-live-route-check",
    "11-before-claim",
    "12-after-claim",
    "13-before-rollout",
    "14-after-rollout",
    "15-accepted",
    "99-failed",
)

EXPECTED: dict[str, dict[str, Any]] = {
    "qwen3.8-27b": {
        "rank": 4,
        "cell_id": "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a",
        "execution_id": "sha256:30d06a7e4299bbf543547e738e80889a09cccc077059a07b3db7c32e1f7e61e5",
        "job_name": "chris-q38-ac-r004-a1-g12-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g12-v1-run-v1",
        "run_id": "chris-q38-ac-g12-r004-a1-30d06a7e",
        "network": "q38-ac-g12-r004-a1-30d06a7e",
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "g11_job": "chris-q38-ac-r004-a1-g11-v1",
        "g11_execution_id": (
            "sha256:b9ca7eeb281f106ae4e47c2779f8af452b04cf123e6f0a5a863eaeb21a2958c9"
        ),
        "tombstone_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-generation11-preclaim-preoutput-tombstone-v1.json"
        ),
        "tombstone_sha256": (
            "sha256:edc71ea8c2057dc9018f9862e71861579d7839f694483af7c9b5f8f59c799e0b"
        ),
    },
    "glm-5.3": {
        "rank": 13,
        "cell_id": "sha256:905051f141077d3aa5086c1f5dc6ad015d5ee6173a1ea7515025805cf9a24b41",
        "execution_id": "sha256:eeec7e3435daa618950ad4f9a61a008c1705f328b4801d8b0cb54460284f8537",
        "job_name": "chris-glm53-ac-r013-a1-g12-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g12-v1-run-v1",
        "run_id": "chris-glm53-ac-g12-r013-a1-eeec7e34",
        "network": "glm53-ac-g12-r013-a1-eeec7e34",
        "repository": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
        "g11_job": "chris-glm53-ac-r013-a1-g11-v1",
        "g11_execution_id": (
            "sha256:28b8bac1ba9f51442d65238f0ca3c4345651c55b13e5b0062b449f687a8af215"
        ),
        "tombstone_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-glm53-generation11-preclaim-preoutput-tombstone-v1.json"
        ),
        "tombstone_sha256": (
            "sha256:5f23c984ee5b4d38236b853456001057eebaa352ea5977c205df8a4381af4d5c"
        ),
    },
}


def digest(value: dict[str, Any], field: str = "receipt_sha256") -> str:
    return self_hosted.digest_without(value, field)


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent immutable input: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != self_hosted.canonical_json(value) + b"\n":
        raise ValueError(f"non-canonical immutable input: {path}")
    return value


def _expected(spec: dict[str, Any]) -> dict[str, Any]:
    row = EXPECTED.get(str(spec.get("model")))
    if row is None:
        raise ValueError("unsupported Generation-12 model")
    return row


def validate_tombstone(value: dict[str, Any], row: dict[str, Any]) -> None:
    if any(
        (
            value.get("schema_version") != TOMBSTONE_SCHEMA,
            value.get("status") != TOMBSTONE_STATUS,
            value.get("model_called") is not False,
            value.get("task_instance_created") is not False,
            value.get("scored_outcome_created") is not False,
            value.get("execution_id") != row["g11_execution_id"],
            value.get("job", {}).get("name") != row["g11_job"],
            value.get("job", {}).get("terminal") != "Failed",
            value.get("pod", {}).get("evaluator_exit_code") != 1,
            value.get("sfs")
            != {"claim_absent": True, "output_root_absent": True, "stage_root_absent": True},
            value.get("fleet", {}).get("matching_sessions") != 0,
            value.get("fleet", {}).get("matching_verifier_records") != 0,
            value.get("receipt_sha256") != row["tombstone_sha256"],
            value.get("receipt_sha256") != digest(value),
        )
    ):
        raise ValueError("Generation-11 tombstone drifted")


def validate(spec: dict[str, Any], repo: Path) -> None:
    row = _expected(spec)
    attempt = spec.get("attempt_config") or {}
    harness = attempt.get("harness") or {}
    model = attempt.get("model") or {}
    execution = attempt.get("execution") or {}
    tombstone = load(repo / row["tombstone_path"])
    validate_tombstone(tombstone, row)
    if any(
        (
            spec.get("schema_version") != "fleet-opencode-generation12-simple-cell-v1",
            spec.get("spec_sha256") != digest(spec, "spec_sha256"),
            spec.get("source_rank") != row["rank"],
            spec.get("attempt") != 1,
            spec.get("cell_id") != row["cell_id"],
            spec.get("execution_id") != row["execution_id"],
            spec.get("execution_generation") != 12,
            spec.get("job_name") != row["job_name"],
            spec.get("configmap_name") != row["configmap_name"],
            spec.get("output_root") != f"/mnt/sfs/jobs/{row['job_name']}",
            spec.get("stage_root") != f"/mnt/sfs/jobs/{row['job_name']}-stages",
            spec.get("run_id") != row["run_id"],
            spec.get("network") != row["network"],
            spec.get("predecessor_generation11", {}).get("receipt_sha256")
            != row["tombstone_sha256"],
            attempt.get("config_sha256") != digest(attempt, "config_sha256"),
            attempt.get("run_id") != row["run_id"],
            attempt.get("campaign_id") != row["job_name"],
            attempt.get("source_job_id") != row["job_name"],
            execution.get("network") != row["network"],
            execution.get("required_task_tools") != ["bash", "submit_report"],
            model.get("repository") != row["repository"],
            model.get("revision") != row["revision"],
            harness.get("name") != "opencode",
            harness.get("version") != "1.18.27",
            harness.get("context_window_size") != 262144,
            harness.get("max_output_tokens") != 32768,
            harness.get("compaction_headroom_tokens") != 20000,
            harness.get("context_management")
            != "opencode_1.18.27_native_compaction_autocontinue_v1",
            spec.get("runtime")
            != {
                "bootstrap": "accepted_v2_smoke_dind_and_apt_docker",
                "cpu_only": True,
                "create_once": True,
                "fleet_team_id": TEAM,
                "preemption_policy": "Never",
                "priority_class": "fleet-serve-low",
                "secret_name": "chris-cyber-opencode-evals-v2",
                "secret_uid": SECRET_UID,
            },
        )
    ):
        raise ValueError("Generation-12 exact treatment drifted")
    if self_hosted.opencode_settings(attempt).get("compaction") != {
        "auto": True,
        "reserved": 20000,
    }:
        raise ValueError("OpenCode compaction treatment drifted")


def claim_path(spec: dict[str, Any], execution_id: str | None = None) -> Path:
    value = execution_id or spec["execution_id"]
    return CLAIM_ROOT / f"{value.removeprefix('sha256:')}.json"


def mark_stage(spec: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("unknown sanitized stage")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not UUID.fullmatch(job_uid) or not UUID.fullmatch(pod_uid):
        raise RuntimeError("downward API UIDs required")
    root = Path(spec["stage_root"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = {
        "schema_version": "fleet-opencode-generation12-stage-v1",
        "model": spec["model"],
        "cell_id": spec["cell_id"],
        "execution_id": spec["execution_id"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "stage": stage,
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt = {**body, "receipt_sha256": digest(body)}
    self_hosted.write_json_once(root / f"{stage}.json", receipt)
    return receipt


def require_fresh(spec: dict[str, Any], repo: Path, key: str) -> None:
    row = _expected(spec)
    validate_tombstone(load(repo / row["tombstone_path"]), row)
    paths = (
        claim_path(spec, row["g11_execution_id"]),
        Path(f"/mnt/sfs/jobs/{row['g11_job']}"),
        claim_path(spec),
        Path(spec["output_root"]),
    )
    if any(path.exists() or path.is_symlink() for path in paths):
        raise RuntimeError("predecessor or Generation-12 identity is not fresh")
    attempt = spec["attempt_config"]
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != TEAM:
            raise RuntimeError("key is not scoped to Fleet")
        persisted = self_hosted.persisted_session_model_identity(attempt)
        matches = [
            item
            for item in self_hosted._task_sessions(client, attempt["task"]["key"])
            if item.get("model") == persisted
        ]
    if matches:
        raise RuntimeError("exact task/treatment already has a persisted session")


def require_live_route(spec: dict[str, Any], key: str) -> None:
    model = spec["attempt_config"]["model"]
    response = httpx.get(
        model["endpoint_origin"].rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data") or []
    if model["served_id"] not in {row.get("id") for row in rows if isinstance(row, dict)}:
        raise RuntimeError("fresh authenticated route does not advertise exact served model")


def create_claim(spec: dict[str, Any]) -> dict[str, Any]:
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not UUID.fullmatch(job_uid) or not UUID.fullmatch(pod_uid):
        raise RuntimeError("downward API UIDs required")
    CLAIM_ROOT.mkdir(parents=True, exist_ok=True)
    with (CLAIM_ROOT / ".claim.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if claim_path(spec).exists() or Path(spec["output_root"]).exists():
            raise RuntimeError("identity claimed concurrently")
        body = {
            "schema_version": "fleet-statistical-cell-execution-claim-v12",
            "spec_sha256": spec["spec_sha256"],
            "cell_id": spec["cell_id"],
            "execution_id": spec["execution_id"],
            "execution_generation": 12,
            "run_id": spec["run_id"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "claimed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "generation11_tombstone_receipt_sha256": _expected(spec)["tombstone_sha256"],
            "automatic_retry": False,
            "immutable": True,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
        receipt = {**body, "receipt_sha256": digest(body)}
        self_hosted.write_json_once(claim_path(spec), receipt)
        return receipt


def run(spec: dict[str, Any], repo: Path, proxy: Path) -> None:
    validate(spec, repo)
    if (
        os.environ.get("JOB_NAME") != spec["job_name"]
        or os.environ.get("GENERATION12_SECRET_UID") != SECRET_UID
    ):
        raise RuntimeError("Job or Secret UID binding drifted")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY required")
    mark_stage(spec, "07-before-duplicate-check")
    require_fresh(spec, repo, key)
    mark_stage(spec, "08-after-duplicate-check")
    mark_stage(spec, "09-before-live-route-check")
    require_live_route(spec, key)
    mark_stage(spec, "10-after-live-route-check")
    mark_stage(spec, "11-before-claim")
    claim = create_claim(spec)
    mark_stage(spec, "12-after-claim")
    root = Path(spec["output_root"])
    root.mkdir(mode=0o700)
    (root / "attempts").mkdir(mode=0o700)
    self_hosted.write_json_once(root / "SPEC.json", spec)
    self_hosted.write_json_once(root / "CLAIM.json", claim)
    mark_stage(spec, "13-before-rollout")
    out = root / "attempts" / spec["run_id"]
    result = self_hosted.run(spec["attempt_config"], out, proxy)
    mark_stage(spec, "14-after-rollout")
    accepted = sweep._accept_attempt(
        out,
        spec["attempt_config"],
        rank=spec["source_rank"],
        attempt_number=1,
        claim_sha256=claim["receipt_sha256"],
    )
    terminal = {
        "schema_version": "fleet-opencode-generation12-simple-terminal-v1",
        "accepted": True,
        "spec_sha256": spec["spec_sha256"],
        "claim_sha256": claim["receipt_sha256"],
        "acceptance_receipt_sha256": accepted["receipt_sha256"],
        "run_id": spec["run_id"],
        "session_id": result["session_id"],
        "verifier_execution_id": result["verifier_execution_id"],
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    terminal["receipt_sha256"] = digest(terminal)
    self_hosted.write_json_once(root / "ACCEPTED.json", terminal)
    mark_stage(spec, "15-accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "mark-stage", "run"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--stage")
    args = parser.parse_args()
    spec = load(args.spec)
    if args.command == "validate":
        validate(spec, args.repo)
        return 0
    if args.command == "mark-stage":
        if not args.stage:
            parser.error("mark-stage requires --stage")
        mark_stage(spec, args.stage)
        return 0
    if args.proxy is None:
        parser.error("run requires --proxy")
    try:
        run(spec, args.repo, args.proxy)
    except Exception:
        with contextlib.suppress(Exception):
            mark_stage(spec, "99-failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
