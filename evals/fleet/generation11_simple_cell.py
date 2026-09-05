"""Minimal create-once Generation-11 hosted OpenCode cell runner."""

from __future__ import annotations

import argparse
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
TOMBSTONES = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/2026-09-05-qwen38-generation10-preclaim-preoutput-tombstone-v1.json",
        "sha256:c0a6c2c7e025d8273f061c10c26d20ccd49ee8c5077dcdbbb32409029780d97f",
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/2026-09-05-glm53-generation10-preclaim-preoutput-tombstone-v1.json",
        "sha256:8ae99ef75226a899895d0ca8dd896af0fac0955d1f301bdfd679bc99872449fb",
    ),
}
EXPECTED = {
    "qwen3.8-27b": (
        4,
        "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a",
        "sha256:b9ca7eeb281f106ae4e47c2779f8af452b04cf123e6f0a5a863eaeb21a2958c9",
        "chris-q38-ac-r004-a1-g11-v1",
        "chris-q38-ac-g11-r004-a1-b9ca7eeb",
        "q38-ac-g11-r004-a1-b9ca7eeb",
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ),
    "glm-5.3": (
        13,
        "sha256:905051f141077d3aa5086c1f5dc6ad015d5ee6173a1ea7515025805cf9a24b41",
        "sha256:28b8bac1ba9f51442d65238f0ca3c4345651c55b13e5b0062b449f687a8af215",
        "chris-glm53-ac-r013-a1-g11-v1",
        "chris-glm53-ac-g11-r013-a1-28b8bac1",
        "glm53-ac-g11-r013-a1-28b8bac1",
        "zai-org/GLM-5.3",
        "30333038ada1f1dacb294a93270305a890b50c14",
    ),
}


def digest(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent immutable input: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != self_hosted.canonical_json(value) + b"\n":
        raise ValueError(f"non-canonical immutable input: {path}")
    return value


def validate(spec: dict[str, Any], repo: Path) -> None:
    model = str(spec.get("model"))
    expected = EXPECTED.get(model)
    if expected is None:
        raise ValueError("unsupported Generation-11 model")
    rank, cell, execution_id, job, run_id, network, repository, revision = expected
    tombstone_path, tombstone_digest = TOMBSTONES[model]
    tombstone = load(repo / tombstone_path)
    attempt = spec.get("attempt_config") or {}
    harness = attempt.get("harness") or {}
    model_config = attempt.get("model") or {}
    execution = attempt.get("execution") or {}
    if (
        tombstone.get("receipt_sha256") != tombstone_digest
        or digest(tombstone, "receipt_sha256") != tombstone_digest
    ):
        raise ValueError("Generation-10 tombstone drifted")
    if any(
        (
            spec.get("schema_version") != "fleet-opencode-generation11-simple-cell-v1",
            spec.get("spec_sha256") != digest(spec, "spec_sha256"),
            spec.get("source_rank") != rank,
            spec.get("attempt") != 1,
            spec.get("cell_id") != cell,
            spec.get("execution_id") != execution_id,
            spec.get("execution_generation") != 11,
            spec.get("job_name") != job,
            spec.get("output_root") != f"/mnt/sfs/jobs/{job}",
            spec.get("run_id") != run_id,
            spec.get("network") != network,
            attempt.get("config_sha256") != digest(attempt, "config_sha256"),
            attempt.get("run_id") != run_id,
            attempt.get("campaign_id") != job,
            attempt.get("source_job_id") != job,
            execution.get("network") != network,
            execution.get("required_task_tools") != ["bash", "submit_report"],
            model_config.get("repository") != repository,
            model_config.get("revision") != revision,
            harness.get("name") != "opencode",
            harness.get("version") != "1.18.27",
            harness.get("context_window_size") != 262144,
            harness.get("max_output_tokens") != 32768,
            harness.get("compaction_headroom_tokens") != 20000,
            harness.get("context_management")
            != "opencode_1.18.27_native_compaction_autocontinue_v1",
            spec.get("runtime")
            != {
                "cpu_only": True,
                "priority_class": "fleet-serve-low",
                "preemption_policy": "Never",
                "create_once": True,
                "secret_name": "chris-cyber-opencode-evals-v2",
                "secret_uid": SECRET_UID,
                "fleet_team_id": TEAM,
            },
        )
    ):
        raise ValueError("Generation-11 exact treatment drifted")
    if self_hosted.opencode_settings(attempt).get("compaction") != {
        "auto": True,
        "reserved": 20000,
    }:
        raise ValueError("OpenCode compaction treatment drifted")


def claim_path(spec: dict[str, Any], execution_id: str | None = None) -> Path:
    value = execution_id or spec["execution_id"]
    return Path(spec["claim_root"]) / f"{value.removeprefix('sha256:')}.json"


def require_fresh(spec: dict[str, Any], key: str) -> None:
    old = spec["predecessor_generation10"]
    paths = [
        claim_path(spec, old["execution_id"]),
        Path(f"/mnt/sfs/jobs/{old['job_name']}"),
        claim_path(spec),
        Path(spec["output_root"]),
    ]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise RuntimeError("predecessor or Generation-11 identity is not fresh")
    attempt = spec["attempt_config"]
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=1800
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != TEAM:
            raise RuntimeError("key is not scoped to Fleet")
        persisted = self_hosted.persisted_session_model_identity(attempt)
        matches = [
            row
            for row in self_hosted._task_sessions(client, attempt["task"]["key"])
            if row.get("model") == persisted
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
    root = Path(spec["claim_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".claim.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if claim_path(spec).exists() or Path(spec["output_root"]).exists():
            raise RuntimeError("identity claimed concurrently")
        body = {
            "schema_version": "fleet-statistical-cell-execution-claim-v11",
            "spec_sha256": spec["spec_sha256"],
            "cell_id": spec["cell_id"],
            "execution_id": spec["execution_id"],
            "execution_generation": 11,
            "run_id": spec["run_id"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "claimed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "generation10_tombstone_receipt_sha256": TOMBSTONES[spec["model"]][1],
            "immutable": True,
            "automatic_retry": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        receipt = {**body, "receipt_sha256": digest(body, "receipt_sha256")}
        self_hosted.write_json_once(claim_path(spec), receipt)
        return receipt


def run(spec: dict[str, Any], repo: Path, proxy: Path) -> None:
    validate(spec, repo)
    if (
        os.environ.get("JOB_NAME") != spec["job_name"]
        or os.environ.get("GENERATION11_SECRET_UID") != SECRET_UID
    ):
        raise RuntimeError("Job or Secret UID binding drifted")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY required")
    require_fresh(spec, key)
    require_live_route(spec, key)
    claim = create_claim(spec)
    root = Path(spec["output_root"])
    root.mkdir(mode=0o700)
    (root / "attempts").mkdir(mode=0o700)
    self_hosted.write_json_once(root / "SPEC.json", spec)
    self_hosted.write_json_once(root / "CLAIM.json", claim)
    out = root / "attempts" / spec["run_id"]
    result = self_hosted.run(spec["attempt_config"], out, proxy)
    accepted = sweep._accept_attempt(
        out,
        spec["attempt_config"],
        rank=spec["source_rank"],
        attempt_number=1,
        claim_sha256=claim["receipt_sha256"],
    )
    terminal = {
        "schema_version": "fleet-opencode-generation11-simple-terminal-v1",
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
    terminal["receipt_sha256"] = digest(terminal, "receipt_sha256")
    self_hosted.write_json_once(root / "ACCEPTED.json", terminal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    spec = load(args.spec)
    if args.command == "validate":
        validate(spec, args.repo)
        return 0
    if args.proxy is None:
        parser.error("run requires --proxy")
    run(spec, args.repo, args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
