"""Run one score-blind Qwen3.8 dedicated-TP1 statistical canary.

The canary is selection rank 2 / attempt 1 from the frozen exact-100
campaign.  It keeps the statistical cell unchanged while binding a separate
dedicated serving block.  A global execution claim is written before the
first model request, and every live duplicate check is repeated immediately
before that claim.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import httpx

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dedicated-tp1-scored-canary-v1"
ACCEPTED_SCHEMA = "fleet-qwen38-dedicated-tp1-cell-accepted-v1"
TERMINAL_SCHEMA = "fleet-qwen38-dedicated-tp1-canary-terminal-v1"
CONTROLLER = "qwen-dedicated-tp1-rank2-v1"
CAMPAIGN = Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json")
SOURCE = Path("evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json")
PARITY = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v1/parity/PARITY.json")
PARITY_SELF_DIGEST = "sha256:234c3050029adc6b3115586bfc3a34693e179f7b658d1837e9b79fe3f930c83b"
SERVICE_ORIGIN = "http://ft-run-c4c7d042-vh5wn-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SERVICE_UID = "4bb3c7a4-77cc-46cc-a3a6-e204b7f46b33"
RAYJOB_UID = "9f1ecf8a-a047-44ae-8d51-45af1f4ae9ac"
WORKLOAD_UID = "a289f7d3-c57b-4737-a533-059b1d28e8f4"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RUN_ID = "chris-cyber-q38-opencode11827-ded-tp1-r002-a1-v2"
JOB_NAME = RUN_ID
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
OUTPUT_ROOT = Path(f"/mnt/sfs/jobs/{JOB_NAME}")
TRAFFIC = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v1/lifecycle/traffic")
EXPECTED_CELL_ID = "sha256:6351b9167d5846a2a30f0725f09f9918cbc749df963c5eba63dc1fc2a835b713"
EXPECTED_EXECUTION_ID = "sha256:d84a03547e45240e9702097b47d68ee122312987c5b18fcfa6ab7ee2ab738f0f"
EXPECTED_IDENTITIES = {
    1: (EXPECTED_CELL_ID, EXPECTED_EXECUTION_ID),
    2: (
        "sha256:7f64bf75eca3485a3e9505e055aca0e61da0e0309bd9557b2b12e24dc57cf413",
        "sha256:d03e96edd079706a35ff9ada0a01dc4249f351ad3819c3491e7b969bc586415e",
    ),
    3: (
        "sha256:d0344219ca8ee093d13f8318b01058b0ed613287464dbbf4e209a6c5fc0f03bc",
        "sha256:cafce1143041a9da93564df60dc3858d2f919f193bc7ace481b6f44f592643f2",
    ),
    4: (
        "sha256:9fa6cc8fd67f925b15936d6a195c4c37a51230230e02f5eab463c2ae0a6d2ba7",
        "sha256:9b3279796be5705ebaaee52b297a3b17d06e682202ab80b62dc516c2510b9eb4",
    ),
}
CLAIM_SCHEMA = "fleet-exact-pass4-bulk-cell-execution-claim-v3"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": self_hosted.digest_without(value, "receipt_sha256")}


def _claim(plan: dict[str, Any], job_uid: str, pod_uid: str) -> dict[str, Any]:
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    item = plan["item"]
    receipt = _seal(
        {
            "schema_version": CLAIM_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "controller": CONTROLLER,
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "execution_generation": item["execution_generation"],
            "run_id": item["run_id"],
            "selection_rank": item["selection_rank"],
            "attempt": item["attempt"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "claimed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "immutable": True,
            "automatic_retry": False,
            "model_call_started_when_claim_written": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )
    name = item["execution_id"].removeprefix("sha256:") + ".json"
    try:
        self_hosted.write_json_once(CLAIM_ROOT / name, receipt)
    except FileExistsError as exc:
        raise RuntimeError("dedicated canary execution claim collision") from exc
    return receipt


def build_plan(
    root: Path,
    attempt: int = 1,
    *,
    selection_rank: int = 2,
    execution_generation: int = 1,
    run_id_override: str | None = None,
    expected_cell_id: str | None = None,
    expected_execution_id: str | None = None,
    expected_task_version_id: str | None = None,
) -> dict[str, Any]:
    if attempt not in EXPECTED_IDENTITIES:
        raise ValueError("dedicated attempt must be 1 through 4")
    run_id = run_id_override or (
        RUN_ID if attempt == 1 else
        f"chris-cyber-q38-opencode11827-ded-tp1-r002-a{attempt}-v1"
    )
    campaign = exact.read_object(root / CAMPAIGN)
    universe = exact.build_universe(campaign, root)
    selected = [
        row
        for row in universe["cells"]
        if row["model"] == "qwen3.8-27b"
        and row["selection_rank"] == selection_rank
        and row["attempt"] == attempt
    ]
    if len(selected) != 1:
        raise ValueError("dedicated canary cell is not unique")
    cell = selected[0]
    execution = exact.execution_for(cell["cell_id"], execution_generation)
    if selection_rank == 2 and execution_generation == 1:
        default_cell_id, default_execution_id = EXPECTED_IDENTITIES[attempt]
        expected_cell_id = expected_cell_id or default_cell_id
        expected_execution_id = expected_execution_id or default_execution_id
        expected_task_version_id = (
            expected_task_version_id or "09a3fea6-f691-4841-9218-d04459041a1f"
        )
    if not all((expected_cell_id, expected_execution_id, expected_task_version_id)):
        raise ValueError("non-default dedicated selection requires exact expected identities")
    if (
        cell["cell_id"] != expected_cell_id
        or execution["execution_id"] != expected_execution_id
        or cell["task_version_id"] != expected_task_version_id
    ):
        raise ValueError("dedicated canary statistical identity drifted")

    source = _load(root / SOURCE)
    tasks = [row for row in source["tasks"] if row["rank"] == selection_rank]
    if len(tasks) != 1 or tasks[0]["task"]["version_id"] != cell["task_version_id"]:
        raise ValueError("dedicated canary task hydration drifted")
    task = tasks[0]
    treatment = campaign["treatment"]
    model = campaign["models"]["qwen3.8-27b"]
    harness = {
        "name": treatment["harness"],
        "version": treatment["harness_version"],
        "release_asset_sha256": treatment["release_asset_sha256"],
        "provider_adapter": treatment["provider_adapter"],
        "context_management": treatment["context_management"],
        "context_window_size": treatment["context_window_size"],
        "compaction_headroom_tokens": treatment["compaction_headroom_tokens"],
        "max_output_tokens": treatment["max_output_tokens"],
        "max_model_requests": treatment["max_model_requests"],
        "timeout_seconds": treatment["timeout_seconds"],
    }
    config = {
        "schema_version": "fleet-hosted-opencode-task-boundary-attempt-v1",
        "run_id": run_id,
        "campaign_id": campaign["campaign_id"],
        "source_job_id": campaign["selection"]["source_job_id"],
        "task": task["task"],
        "environment": task["environment"],
        "verifier": task["verifier"],
        "authority": source["authority"],
        "model": {**model, "endpoint_origin": SERVICE_ORIGIN},
        "harness": harness,
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": run_id,
            "cell_id": cell["cell_id"],
            "execution_id": execution["execution_id"],
            "training_data_eligible": False,
            "required_task_tools": treatment["tools"],
            "required_task_tool_catalog_sha256": treatment["tool_catalog_sha256"],
        },
        "serving": {
            "kind": "dedicated_qwen_tp1_v1",
            "serving_block": "dedicated-qwen-tp1-v1",
            "service_origin": SERVICE_ORIGIN,
            "service_uid": SERVICE_UID,
            "rayjob_uid": RAYJOB_UID,
            "workload_uid": WORKLOAD_UID,
            "parity_receipt_sha256": PARITY_SELF_DIGEST,
            "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
        },
    }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    item = {
        "cell_id": cell["cell_id"],
        "execution_id": execution["execution_id"],
        "execution_generation": execution_generation,
        "run_id": run_id,
        "selection_rank": selection_rank,
        "attempt": attempt,
        "task_version_id": cell["task_version_id"],
    }
    body = {
        "schema_version": SCHEMA,
        "controller": CONTROLLER,
        "campaign_id": campaign["campaign_id"],
        "model": "qwen3.8-27b",
        "model_revision": MODEL_REVISION,
        "treatment": treatment,
        "item": item,
        "config": config,
        "claim_root": str(CLAIM_ROOT),
        "output_root": f"/mnt/sfs/jobs/{run_id}",
        "launch_authorized": True,
        "score_blind": True,
    }
    return {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}


def _validate_parity() -> dict[str, Any]:
    parity = _load(PARITY)
    if (
        parity.get("receipt_sha256") != PARITY_SELF_DIGEST
        or parity.get("receipt_sha256") != self_hosted.digest_without(parity, "receipt_sha256")
        or parity.get("status") != "PASSED"
        or parity.get("model_revision") != MODEL_REVISION
        or parity.get("service_uid") != SERVICE_UID
        or parity.get("ray_job_uid") != RAYJOB_UID
        or parity.get("context_length") != 262144
        or parity.get("opencode_version") != "1.18.27"
        or parity.get("provider_adapter") != "@ai-sdk/openai-compatible"
        or parity.get("context_management")
        != "opencode_1.18.27_native_compaction_autocontinue_v1"
        or parity.get("selected_tool_names") != ["bash", "submit_report"]
        or parity.get("structured_tool_schema_names") != ["bash", "submit_report"]
        or parity.get("task_instance_created") is not False
        or parity.get("session_created") is not False
        or parity.get("verifier_executed") is not False
        or parity.get("scoring_executed") is not False
    ):
        raise RuntimeError("dedicated Qwen parity receipt drifted")
    return parity


def _safe_receipt_scan(item: dict[str, Any]) -> None:
    needles = {item["cell_id"], item["execution_id"]}
    matches = 0
    for base in (Path("/mnt/sfs/jobs"), CLAIM_ROOT):
        for path in base.rglob("*.json"):
            relevant = (
                path.name == "ACCEPTED.json"
                or "accepted" in path.parts
                or "claim" in path.as_posix().lower()
            )
            if not relevant:
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            matches += int(any(needle in text for needle in needles))
    if matches:
        raise RuntimeError("dedicated canary cell already has accepted or claimed evidence")


def _live_checks(plan: dict[str, Any], key: str) -> None:
    config, item = plan["config"], plan["item"]
    output_root = Path(plan["output_root"])
    if output_root.exists():
        raise RuntimeError("dedicated canary output root already exists")
    if (CLAIM_ROOT / (item["execution_id"].removeprefix("sha256:") + ".json")).exists():
        raise RuntimeError("dedicated canary execution already claimed")
    _safe_receipt_scan(item)
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        sessions = self_hosted._task_sessions(client, config["task"]["key"])
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("dedicated canary Fleet team identity drifted")
    collisions = [
        row
        for row in sessions
        if (row.get("metadata") or {}).get("cell_id") == item["cell_id"]
        or (row.get("metadata") or {}).get("execution_id") == item["execution_id"]
        or (row.get("metadata") or {}).get("run_id") == item["run_id"]
    ]
    if collisions:
        raise RuntimeError("dedicated canary has an authoritative session collision")
    with urlopen(SERVICE_ORIGIN + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("dedicated Qwen service is unhealthy")
    with urlopen(SERVICE_ORIGIN + "/v1/models", timeout=30) as response:
        roster = json.load(response)
    if [row.get("id") for row in roster.get("data", [])] != ["qwen3.8-27b"]:
        raise RuntimeError("dedicated Qwen served model drifted")


def _traffic_loop(stop: threading.Event) -> None:
    last: str | None = None
    while not stop.wait(20):
        try:
            with urlopen(SERVICE_ORIGIN + "/metrics", timeout=10) as response:
                metrics = response.read(8_000_000).decode(errors="ignore")
            current = "\n".join(
                line
                for line in metrics.splitlines()
                if not line.startswith("#")
                and any(
                    token in line
                    for token in (
                        "generation_tokens_total",
                        "prompt_tokens_total",
                        "requests_total",
                    )
                )
            )
            if current and last is not None and current != last:
                TRAFFIC.touch()
            last = current
        except Exception:
            continue


def _classify(out: Path, plan: dict[str, Any], claim: dict[str, Any], key: str) -> dict[str, Any]:
    config, item = plan["config"], plan["item"]
    result = _load(out / "result.json")
    reward = _load(out / "reward-result.json")
    ingest = _load(out / "session-ingest.json")
    cleanup = _load(out / "cleanup.json")
    session_id, verifier_id = result.get("session_id"), result.get("verifier_execution_id")
    uuid.UUID(str(session_id))
    uuid.UUID(str(verifier_id))
    if not all((
        result.get("run_id") == item["run_id"],
        result.get("task_version_id") == config["task"]["version_id"],
        result.get("agent_termination") == "completed",
        type(result.get("agent_exit_code")) is int,
        result.get("session_ingest_status") == "completed",
        ingest.get("status") == "completed" and ingest.get("session_id") == session_id,
        reward.get("task_version_id") == config["task"]["version_id"],
        reward.get("verifier_execution_id") == verifier_id,
        isinstance(reward.get("reward"), (int, float))
        and not isinstance(reward.get("reward"), bool),
        cleanup == {"instance_created": True, "instance_closed": True, "containers_removed": True},
    )):
        raise RuntimeError("dedicated canary is infrastructure-incomplete")
    authoritative: list[dict[str, Any]] = []
    for _ in range(12):
        with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=1800) as client:
            authoritative = [
                row for row in self_hosted._task_sessions(client, config["task"]["key"])
                if row.get("session_id") == session_id
            ]
        if authoritative:
            break
        time.sleep(5)
    if len(authoritative) != 1:
        raise RuntimeError("dedicated canary lacks one authoritative session")
    row = authoritative[0]
    metadata = row.get("metadata")
    projected_task_version = row.get("eval_task_version_id") or row.get("task_version_id")
    expected_model = self_hosted.persisted_session_model_identity(config)
    if (
        row.get("status") != "completed"
        or row.get("task_key") != config["task"]["key"]
        or (row.get("verifier_execution") or {}).get("id") != verifier_id
        or (row.get("model") is not None and row.get("model") != expected_model)
        or (
            projected_task_version is not None
            and projected_task_version != config["task"]["version_id"]
        )
        or (
            metadata is not None
            and (
                not isinstance(metadata, dict)
                or metadata.get("run_id") != item["run_id"]
                or metadata.get("cell_id") != item["cell_id"]
                or metadata.get("execution_id") != item["execution_id"]
            )
        )
    ):
        raise RuntimeError("dedicated canary authoritative session binding drifted")
    projection_omissions = sorted(
        name
        for name, value in {
            "model": row.get("model"),
            "task_version_id": projected_task_version,
            "metadata": metadata,
        }.items()
        if value is None
    )
    return _seal({
        "schema_version": ACCEPTED_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "serving_block": "dedicated-qwen-tp1-v1",
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_version_id": config["task"]["version_id"],
        "session_id": session_id,
        "verifier_execution_id": verifier_id,
        "agent_exit_code": result["agent_exit_code"],
        "agent_process_exit_success": result["agent_exit_code"] == 0,
        "claim_sha256": claim["receipt_sha256"],
        "config_sha256": config["config_sha256"],
        "cleanup_completed": True,
        "session_ingest_completed": True,
        "authoritative_session_task_key_matched": True,
        "authoritative_projection_omissions": projection_omissions,
        "authoritative_projection_rule": "legacy_list_fields_may_be_null_but_never_mismatched_v1",
        "scores_included": False,
        "prompts_or_traces_included": False,
    })


def run(root: Path, proxy: Path, attempt: int = 1) -> dict[str, Any]:
    plan = build_plan(root, attempt)
    output_root = Path(plan["output_root"])
    key = os.environ.get("FLEET_API_KEY")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    _validate_parity()
    _live_checks(plan, key)
    output_root.mkdir(mode=0o700, parents=False)
    self_hosted.write_json_once(output_root / "PLAN.json", plan)
    claim = _claim(plan, job_uid, pod_uid)
    stop = threading.Event()
    watcher = threading.Thread(target=_traffic_loop, args=(stop,), daemon=True)
    watcher.start()
    attempt_root = output_root / "attempt"
    try:
        self_hosted.run(plan["config"], attempt_root, proxy)
        accepted = _classify(attempt_root, plan, claim, key)
        self_hosted.write_json_once(output_root / "ACCEPTED.json", accepted)
        terminal = _seal({
            "schema_version": TERMINAL_SCHEMA,
            "status": "ACCEPTED",
            "accepted": True,
            "serving_block": "dedicated-qwen-tp1-v1",
            "cell_id": plan["item"]["cell_id"],
            "execution_id": plan["item"]["execution_id"],
            "claim_sha256": claim["receipt_sha256"],
            "accepted_receipt_sha256": accepted["receipt_sha256"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "scores_included": False,
            "prompts_or_traces_included": False,
        })
        self_hosted.write_json_once(output_root / "TERMINAL.json", terminal)
        return terminal
    finally:
        stop.set()
        watcher.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=sorted(EXPECTED_IDENTITIES), default=1)
    args = parser.parse_args()
    root = Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train"))
    run(root, root / "evals/fleet/fixed_proxy.py", args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
