"""Retry-safe generation-2 wrappers for the two corrected-treatment canary cells."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-plan-v1"
TOMBSTONE_BUNDLE_SCHEMA = "fleet-opencode-autocontinue-generation1-tombstones-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-held-release-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-scoring-release-v1"
INCIDENT_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json"
)
INCIDENT_SHA = "sha256:fd71d3fb42d1ea4fef4734352c817c62cf0ec5fe6e3e35a28c2a9a4f47aa3f23"
TOMBSTONE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json"
)
CAMPAIGN_PATH = "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh"
MODULE_PATH = "evals/fleet/autocontinue_generation2_canary.py"
GENERATION_CLAIM_ROOT = "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1"
OLD_GLOBAL_CLAIM_ROOT = "/mnt/sfs/cell-claims/opencode11827-autocontinue-primary-v1"
INTENT_CONFIGMAP = "chris-ac-g2-canary-submit-v1"
EXPECTED = {
    "qwen3.8-27b": {
        "selection_rank": 4,
        "task_version_id": "02dd4e3f-d85d-4bf8-9976-eae2f102384d",
        "predecessor_plan_path": (
            "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
        ),
        "job_name": "chris-q38-ac-r004-a1-g2-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g2-run-v1",
        "run_id": "chris-q38-ac-g2-r004-a1-02dd4e3f",
        "network": "q38-ac-g2-r004-a1-02dd4e3f",
    },
    "glm-5.3": {
        "selection_rank": 13,
        "task_version_id": "9375a9b9-04e5-4f6f-ad47-286121278992",
        "predecessor_plan_path": (
            "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
        ),
        "job_name": "chris-glm53-ac-r013-a1-g2-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g2-run-v1",
        "run_id": "chris-glm53-ac-g2-r013-a1-9375a9b9",
        "network": "glm53-ac-g2-r013-a1-9375a9b9",
    },
}


def load(path: Path) -> dict[str, Any]:
    return exact.read_object(path)


def digest(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required package file is unsafe or absent: {path}")
    return self_hosted.sha256(path.read_bytes())


def _incident(root: Path) -> dict[str, Any]:
    value = load(root / INCIDENT_PATH)
    if (
        value.get("receipt_sha256") != INCIDENT_SHA
        or value.get("receipt_sha256") != digest(value, "receipt_sha256")
        or value.get("schema_version")
        != "fleet-opencode-autocontinue-canary-pre-model-failure-v1"
        or value.get("status") != "INFRASTRUCTURE_INVALID_PRE_MODEL"
        or value.get("failure_class") != "deterministic_plan_renderer_rejection"
        or value.get("scientific_disposition")
        != {
            "statistical_cells_consumed": 0,
            "valid_rollouts": 0,
            "retry_requires_fresh_execution_generation": True,
            "reuse_old_run_or_job_identity": False,
            "preserve_old_claims": True,
        }
        or value.get("aggregate")
        != {
            "jobs_failed": 2,
            "pods_restarted": 0,
            "attempt_directories": 0,
            "fleet_sessions": 0,
            "verifier_executions": 0,
            "accepted_rollouts": 0,
        }
        or value.get("evidence_method", {}).get("hosted_model_requests_observed") != 0
        or value.get("privacy", {}).get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("generation-1 incident is not retry-safe authority")
    rows = value.get("models")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("generation-1 incident model rows drifted")
    for row in rows:
        if (
            row.get("served_id") not in EXPECTED
            or row.get("execution_generation") != 1
            or row.get("attempt") != 1
            or row.get("task_version_id") != EXPECTED[row["served_id"]]["task_version_id"]
            or row.get("attempt_directory_count") != 0
            or row.get("fleet_session_count") != 0
            or row.get("verifier_execution_count") != 0
            or row.get("pod", {}).get("exit_code") != 1
            or row.get("pod", {}).get("restart_count") != 0
            or row.get("job", {}).get("reason") != "BackoffLimitExceeded"
            or not exact.SHA256_RE.fullmatch(str(row.get("local_claim_sha256")))
            or not exact.SHA256_RE.fullmatch(str(row.get("global_claim_receipt_sha256")))
        ):
            raise ValueError("generation-1 incident row is not retry-safe")
    if {row["served_id"] for row in rows} != set(EXPECTED):
        raise ValueError("generation-1 incident model set drifted")
    return value


def _universe(root: Path) -> dict[str, Any]:
    return exact.build_universe(load(root / CAMPAIGN_PATH), root)


def _cell(root: Path, model: str) -> dict[str, Any]:
    expected = EXPECTED[model]
    matches = [
        row
        for row in _universe(root)["cells"]
        if row["model"] == model
        and row["selection_rank"] == expected["selection_rank"]
        and row["attempt"] == 1
    ]
    if len(matches) != 1 or matches[0]["task_version_id"] != expected["task_version_id"]:
        raise ValueError("generation-2 statistical cell is not unique")
    return matches[0]


def validate_tombstones(bundle: dict[str, Any], root: Path) -> None:
    incident = _incident(root)
    rows = bundle.get("models")
    if (
        set(bundle)
        != {
            "schema_version",
            "append_only",
            "status",
            "incident_receipt",
            "models",
            "privacy",
            "receipt_sha256",
        }
        or bundle.get("schema_version") != TOMBSTONE_BUNDLE_SCHEMA
        or bundle.get("append_only") is not True
        or bundle.get("status") != "GENERATION_1_TOMBSTONED"
        or bundle.get("incident_receipt") != {"path": INCIDENT_PATH, "receipt_sha256": INCIDENT_SHA}
        or not isinstance(rows, list)
        or len(rows) != 2
        or bundle.get("privacy")
        != {"prompts_traces_flags_or_scores_included": False, "credentials_included": False}
        or bundle.get("receipt_sha256") != digest(bundle, "receipt_sha256")
    ):
        raise ValueError("generation-1 tombstone bundle drifted")
    incident_rows = {row["served_id"]: row for row in incident["models"]}
    for row in rows:
        model = row.get("served_id")
        if model not in EXPECTED:
            raise ValueError("generation-1 tombstone model drifted")
        cell = _cell(root, model)
        source = incident_rows[model]
        old_identity = {
            "context_management": exact.EXPECTED_TREATMENT["context_management"],
            "served_id": model,
            "session_model": exact.EXPECTED_MODELS[model]["session_model"],
            "task_version_id": cell["task_version_id"],
            "attempt": 1,
        }
        old_global_name = (
            self_hosted.sha256(self_hosted.canonical_json(old_identity)).removeprefix("sha256:")
            + ".json"
        )
        expected_claims = {
            "local_path": (
                f"/mnt/sfs/jobs/{source['job']['name']}/claims/{source['run_id']}.json"
            ),
            "local_claim_sha256": source["local_claim_sha256"],
            "global_path": f"{OLD_GLOBAL_CLAIM_ROOT}/{old_global_name}",
            "global_claim_receipt_sha256": source["global_claim_receipt_sha256"],
            "preserved_not_deleted_or_reused": True,
        }
        if (
            set(row) != {"served_id", "cell_id", "predecessor", "claims", "tombstone"}
            or row.get("cell_id") != cell["cell_id"]
            or row.get("predecessor") != source
            or row.get("claims") != expected_claims
        ):
            raise ValueError("generation-1 tombstone predecessor or claim binding drifted")
        exact.validate_tombstone(row.get("tombstone") or {}, cell)
        if row["tombstone"]["evidence_receipt_sha256"] != INCIDENT_SHA:
            raise ValueError("generation-1 tombstone incident binding drifted")
    if {row["served_id"] for row in rows} != set(EXPECTED):
        raise ValueError("generation-1 tombstone coverage drifted")


def _tombstones(root: Path) -> dict[str, Any]:
    value = load(root / TOMBSTONE_PATH)
    validate_tombstones(value, root)
    return value


def validate_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-2 model is unsupported")
    expected = EXPECTED[model]
    tombstones = _tombstones(root)
    tombstone = next(row["tombstone"] for row in tombstones["models"] if row["served_id"] == model)
    cell = _cell(root, model)
    execution = exact.next_execution(cell, [tombstone])
    if (
        set(spec)
        != {
            "schema_version",
            "model",
            "statistical_cell",
            "execution",
            "predecessor_plan",
            "supersession",
            "identities",
            "launch_authorized",
            "plan_sha256",
        }
        or spec.get("schema_version") != PLAN_SCHEMA
        or spec.get("statistical_cell")
        != {
            "cell_id": cell["cell_id"],
            "selection_rank": expected["selection_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        }
        or spec.get("execution") != execution
        or spec.get("predecessor_plan")
        != {
            "path": expected["predecessor_plan_path"],
            "plan_sha256": next(
                row["predecessor"]["plan_sha256"]
                for row in tombstones["models"]
                if row["served_id"] == model
            ),
        }
        or spec.get("supersession")
        != {
            "incident_receipt_sha256": INCIDENT_SHA,
            "tombstone_bundle_receipt_sha256": tombstones["receipt_sha256"],
            "prior_generation": 1,
            "new_generation": 2,
            "same_statistical_cell": True,
            "old_claims_preserved": True,
        }
        or spec.get("identities")
        != {
            "job_name": expected["job_name"],
            "configmap_name": expected["configmap_name"],
            "intent_configmap_name": INTENT_CONFIGMAP,
            "run_id": expected["run_id"],
            "network": expected["network"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
            "generation_claim_root": GENERATION_CLAIM_ROOT,
        }
        or spec.get("launch_authorized") is not False
        or spec.get("plan_sha256") != digest(spec, "plan_sha256")
    ):
        raise ValueError("generation-2 canary plan drifted")

    predecessor = load(root / expected["predecessor_plan_path"])
    if predecessor.get("plan_sha256") != spec["predecessor_plan"]["plan_sha256"]:
        raise ValueError("generation-2 predecessor plan digest drifted")
    plan = copy.deepcopy(predecessor)
    plan["campaign_id"] = expected["job_name"]
    plan["source_job_id"] = expected["job_name"]
    plan["preflight_job_name"] = expected["job_name"] + "-preflight"
    plan["scored_job_name"] = expected["job_name"]
    plan["sfs_root"] = expected["job_name"]
    plan["harness"]["compaction_headroom_tokens"] = 20000
    plan["treatment_block"]["harness"]["compaction_headroom_tokens"] = 20000
    plan["attempts"][0].update(
        {"run_id": expected["run_id"], "network": expected["network"], "execution_generation": 2}
    )
    plan["execution"].update(
        {
            "execution_generation": 2,
            "generation_claim_root": GENERATION_CLAIM_ROOT,
            "old_claims_preserved": True,
            "retry_policy": "generation2_only_after_exact_pre_model_tombstone",
            "launch_authorized": False,
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": CAMPAIGN_PATH,
        "statistical_cell_id": cell["cell_id"],
        "execution_id": execution["execution_id"],
        "tombstone_bundle_receipt_sha256": tombstones["receipt_sha256"],
        "incident_receipt_sha256": INCIDENT_SHA,
    }
    plan["plan_sha256"] = spec["plan_sha256"]
    settings = self_hosted.opencode_settings(plan)
    canonical = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical)
        or plan["harness"]["settings_file_sha256"] != self_hosted.sha256(canonical + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or plan["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("generation-2 executable treatment renderer drifted")
    return plan


def validate_held(release: dict[str, Any], specs: list[dict[str, Any]], root: Path) -> None:
    plans = [validate_plan(spec, root) for spec in specs]
    if (
        set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "observed_at_utc",
            "incident_receipt_sha256",
            "tombstone_bundle_receipt_sha256",
            "plan_sha256s",
            "jobs",
            "execution_generation",
            "statistical_cells",
            "implementation",
            "launch_authorized",
            "bulk_release_authorized",
            "dedicated_serving_authorized",
            "remaining_gates",
            "privacy",
            "receipt_sha256",
        }
        or release.get("schema_version") != HELD_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "HELD"
        or not isinstance(release.get("observed_at_utc"), str)
        or not release.get("observed_at_utc")
        or release.get("launch_authorized") is not False
        or release.get("bulk_release_authorized") is not False
        or release.get("dedicated_serving_authorized") is not False
        or release.get("incident_receipt_sha256") != INCIDENT_SHA
        or release.get("tombstone_bundle_receipt_sha256") != _tombstones(root)["receipt_sha256"]
        or release.get("plan_sha256s") != [plan["plan_sha256"] for plan in plans]
        or release.get("jobs")
        != [EXPECTED[plan["model"]["served_id"]]["job_name"] for plan in plans]
        or release.get("execution_generation") != 2
        or release.get("statistical_cells") != 2
        or release.get("implementation")
        != {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
        }
        or release.get("remaining_gates")
        != [
            "independent_package_audit",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_model_specific_scoring_releases",
            "explicit_root_launch_authorization",
        ]
        or release.get("privacy")
        != {"prompts_traces_flags_or_scores_included": False, "credentials_included": False}
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
    ):
        raise ValueError("generation-2 held release drifted")


def validate_preserved_claims(spec: dict[str, Any], repo: Path) -> None:
    """Prove generation 1 remains immutable before claiming generation 2."""
    model = spec["model"]
    row = next(row for row in _tombstones(repo)["models"] if row["served_id"] == model)
    local_path = Path(row["claims"]["local_path"])
    global_path = Path(row["claims"]["global_path"])
    local = load(local_path)
    global_claim = load(global_path)
    if (
        local.get("claim_sha256") != row["claims"]["local_claim_sha256"]
        or local.get("claim_sha256") != digest(local, "claim_sha256")
        or global_claim.get("receipt_sha256")
        != row["claims"]["global_claim_receipt_sha256"]
        or global_claim.get("receipt_sha256") != digest(global_claim, "receipt_sha256")
        or (local_path.parents[1] / "attempts" / row["predecessor"]["run_id"]).exists()
    ):
        raise RuntimeError("generation-1 claims are absent, changed, or no longer pre-model")


def claim_execution_generation(
    spec: dict[str, Any], plan: dict[str, Any], claim_root: Path | None = None
) -> dict[str, Any]:
    """Atomically claim generation 2 without deleting or reusing generation 1."""
    if not legacy._is_uuid(os.environ.get("JOB_UID")) or not legacy._is_uuid(
        os.environ.get("POD_UID")
    ):
        raise RuntimeError("generation claim requires downward API UIDs")
    root = claim_root or Path(spec["identities"]["generation_claim_root"])
    root_fd = legacy._open_directory_nofollow(root)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("generation claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            name = spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
            try:
                existing = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode):
                    raise RuntimeError("generation claim path is unsafe")
                raise RuntimeError("execution generation is already claimed")
            receipt = {
                "schema_version": "fleet-statistical-cell-execution-claim-v1",
                "cell_id": spec["statistical_cell"]["cell_id"],
                "execution_id": spec["execution"]["execution_id"],
                "execution_generation": 2,
                "plan_sha256": spec["plan_sha256"],
                "run_id": plan["attempts"][0]["run_id"],
                "job_uid": os.environ["JOB_UID"],
                "pod_uid": os.environ["POD_UID"],
                "claimed_at_utc": datetime.now(UTC).isoformat(),
                "generation_1_claims_preserved": True,
                "immutable": True,
                "automatic_retry": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            claim_fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(claim_fd).st_mode):
                    raise RuntimeError("generation claim path is unsafe")
                payload = self_hosted.canonical_json(receipt) + b"\n"
                with os.fdopen(claim_fd, "wb") as stored:
                    claim_fd = -1
                    stored.write(payload)
                    stored.flush()
                    os.fsync(stored.fileno())
            finally:
                if claim_fd >= 0:
                    os.close(claim_fd)
            os.fsync(root_fd)
            return receipt
    finally:
        os.close(root_fd)


def validate_release(
    release: dict[str, Any], spec: dict[str, Any], root: Path, package_commit: str
) -> None:
    plan = validate_plan(spec, root)
    expected = EXPECTED[spec["model"]]
    if (
        set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "released_at_utc",
            "plan_sha256",
            "cell_id",
            "execution_id",
            "incident_receipt_sha256",
            "tombstone_bundle_receipt_sha256",
            "package_commit",
            "implementation",
            "authorization",
            "route_and_inventory",
            "privacy",
            "receipt_sha256",
        }
        or release.get("schema_version") != RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or not isinstance(release.get("released_at_utc"), str)
        or not release.get("released_at_utc")
        or release.get("plan_sha256") != spec["plan_sha256"]
        or release.get("cell_id") != spec["statistical_cell"]["cell_id"]
        or release.get("execution_id") != spec["execution"]["execution_id"]
        or release.get("incident_receipt_sha256") != INCIDENT_SHA
        or release.get("tombstone_bundle_receipt_sha256") != _tombstones(root)["receipt_sha256"]
        or release.get("package_commit") != package_commit
        or release.get("implementation", {}).get("job_name") != expected["job_name"]
        or release.get("implementation", {}).get("configmap_name") != expected["configmap_name"]
        or release.get("implementation", {}).get("sfs_root")
        != spec["identities"]["sfs_root"]
        or release.get("authorization")
        != {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "statement": release.get("authorization", {}).get("statement"),
        }
        or not release.get("authorization", {}).get("statement")
        or release.get("route_and_inventory")
        != {
            "fresh_authenticated_hosted_route_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "old_claims_must_exist_and_match": True,
            "generation_2_claim_must_be_absent": True,
        }
        or release.get("privacy")
        != {"prompts_traces_flags_or_scores_included": False, "credentials_included": False}
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
        or plan["execution"]["launch_authorized"] is not False
    ):
        raise ValueError("generation-2 scoring release is not authoritative")


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
    launch_route: dict[str, Any],
    out: Path,
    proxy: Path,
    repo: Path,
    package_commit: str,
) -> dict[str, Any]:
    plan = validate_plan(spec, repo)
    validate_release(release, spec, repo, package_commit)
    jobs = tuple(row["job_name"] for row in EXPECTED.values())
    hosted_runtime.validate_live_route(
        launch_route,
        caller=hosted_runtime.LAUNCHER_CALLER,
        maximum_age_seconds=None,
        candidate_scored_job_names=jobs,
    )
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = plan["execution"]["endpoint_lease"]
    with legacy.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        runtime_route = hosted_runtime.observe_live_route(
            key,
            caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted_runtime.validate_live_route(
            runtime_route,
            caller=hosted_runtime.RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        validate_preserved_claims(spec, repo)
        hosted._validate_plan_identity_absence(plan, out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(plan, out, plan["tasks"][0], key)
        generation_claim = claim_execution_generation(spec, plan)
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        result = legacy._run_cell(plan, out, proxy, key)
        terminal = {
            "schema_version": "fleet-opencode-autocontinue-generation2-terminal-v1",
            "plan_sha256": spec["plan_sha256"],
            "cell_id": spec["statistical_cell"]["cell_id"],
            "execution_id": spec["execution"]["execution_id"],
            "execution_generation": 2,
            "generation_claim_receipt_sha256": generation_claim["receipt_sha256"],
            "job_uid": os.environ["JOB_UID"],
            "pod_uid": os.environ["POD_UID"],
            "accepted": result["accepted"],
            "quarantined": result["quarantined"],
            "retry_allowed": False,
            "bulk_release_authorized": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        terminal["receipt_sha256"] = digest(terminal, "receipt_sha256")
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate-plan", "validate-held", "validate-release", "preview", "run")
    )
    parser.add_argument("--plan", action="append", type=Path, default=[])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    specs = [load(path) for path in args.plan]
    if args.command == "validate-plan":
        if len(specs) != 1:
            parser.error("validate-plan requires exactly one --plan")
        validate_plan(specs[0], args.repo)
        return 0
    if args.command in {"validate-release", "run"}:
        if not args.release or len(specs) != 1 or not args.package_commit:
            parser.error(f"{args.command} requires one --plan, --release, and --package-commit")
        release = load(args.release)
        validate_release(release, specs[0], args.repo, args.package_commit)
        if args.command == "run":
            if not args.launch_route or not args.out_dir or not args.proxy:
                parser.error("run requires --launch-route, --out-dir, and --proxy")
            run(
                specs[0],
                release,
                load(args.launch_route),
                args.out_dir,
                args.proxy,
                args.repo,
                args.package_commit,
            )
        return 0
    if not args.release or len(specs) != 2:
        parser.error(f"{args.command} requires two --plan values and --release")
    release = load(args.release)
    validate_held(release, specs, args.repo)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "models": sorted(EXPECTED),
                    "statistical_cells": 2,
                    "execution_generation": 2,
                    "launch_authorized": False,
                    "bulk_release_authorized": False,
                    "cluster_objects_created": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
