"""Append-only phased authority for the final exact pass@4 execution."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_hosted_c4_bulk_v4 as hosted
from evals.fleet import exact_pass4_prebulk_reconciliation_v4 as prebulk
from evals.fleet import glm53_dedicated_v7 as dedicated
from evals.fleet import hosted_concurrency4_qualification_v1 as qualifier
from evals.fleet import self_hosted

BASE_COMMIT = "91d8b201af4a048c1a578ec488293f37fc0212a5"
MODULE_PATH = "evals/fleet/exact_pass4_final_bulk_v5.py"
RUNTIME_PATH = "evals/fleet/exact_pass4_final_bulk_runtime_v5.py"
PACKAGE_PATH = "evals/fleet/exact_pass4_final_bulk_package_v5.py"
RENDER_PATH = "evals/fleet/exact_pass4_final_bulk_renderer_v5.py"
DEDICATED_EVIDENCE_PATH = "evals/fleet/exact_pass4_final_dedicated_evidence_v5.py"
DUPLICATE_OBSERVER_PATH = "evals/fleet/exact_pass4_final_duplicate_observer_v5.py"
CREATE_RELAY_PATH = "evals/fleet/kubernetes_create_relay.py"
QUALIFIER_GATHER_PATH = "evals/fleet/hosted_concurrency4_qualification_release_v3_gather.py"
QUALIFIER_PREPARE_PATH = (
    "evals/fleet/scripts/prepare_hosted_concurrency4_qualification_release_v3.sh"
)
RUN_PATH = "evals/fleet/scripts/run_exact_pass4_final_bulk_v5.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_exact_pass4_final_bulk_v5.sh"
HELD_PATH = "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-final-bulk-held-v5.json"
DOC_PATH = "docs/EXACT_PASS4_FINAL_BULK_V5.md"
MANIFEST_PATH = "evals/fleet/cluster/opencode-exact-pass4-final-bulk-held-v5.yaml"

PLAN_SCHEMA = "fleet-exact-pass4-final-controller-plan-v5"
HELD_SCHEMA = "fleet-exact-pass4-final-bulk-held-v5"
RELEASE_SCHEMA = "fleet-exact-pass4-final-bulk-release-v5"
FRESH_SCHEMA = "fleet-exact-pass4-final-bulk-fresh-duplicate-v5"
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

PREBULK_TERMINAL = Path(prebulk.OUTPUT_ROOT) / "TERMINAL.json"
INVENTORY_TERMINAL = Path(hosted.INVENTORY_TERMINAL)
QUALIFIER_ROOT = Path("/mnt/sfs/jobs/chris-cyber-hosted-c4-qualification-v1")
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"


def _controller(
    model: str,
    kind: str,
    ranks: range,
    *,
    replica: str | None = None,
    stream: int | None = None,
    canary: tuple[int, int] | None = None,
) -> dict[str, Any]:
    short = "q38" if model == "qwen3.8-27b" else "glm53"
    suffix = (
        f"dedicated-{replica.lower()}-canary"
        if canary
        else f"{kind}-{replica.lower()}-s{stream}"
        if replica
        else f"hosted-s{stream}"
    )
    key = f"{short}-{suffix}"
    serving = (
        "qwen-hosted-autocontinue-v1"
        if model == "qwen3.8-27b"
        else "glm-hosted-autocontinue-v1"
        if kind == "hosted"
        else f"glm-dedicated-{replica.lower()}-v7"
    )
    return {
        "key": key,
        "model": model,
        "serving_kind": kind,
        "serving_block": serving,
        "replica": replica,
        "stream": stream,
        "ranks": list(ranks),
        "canary": canary,
        "job_name": f"chris-cyber-exact100-{key}-v5",
        "configmap_name": f"chris-cyber-exact100-{key}-run-v5",
    }


_ROWS = [
    *[
        _controller("qwen3.8-27b", "hosted", range(first, last + 1), stream=stream)
        for stream, (first, last) in enumerate(((1, 25), (26, 50), (51, 75), (76, 100)), 1)
    ],
    *[
        _controller("glm-5.3", "hosted", range(first, last + 1), stream=stream)
        for stream, (first, last) in enumerate(((1, 13), (14, 25), (26, 38), (39, 50)), 1)
    ],
    _controller("glm-5.3", "dedicated", range(51, 52), replica="A", canary=(51, 1)),
    _controller("glm-5.3", "dedicated", range(51, 64), replica="A", stream=1),
    _controller("glm-5.3", "dedicated", range(64, 76), replica="A", stream=2),
    _controller("glm-5.3", "dedicated", range(76, 77), replica="B", canary=(76, 1)),
    _controller("glm-5.3", "dedicated", range(76, 89), replica="B", stream=1),
    _controller("glm-5.3", "dedicated", range(89, 101), replica="B", stream=2),
]
CONTROLLERS = {row["key"]: row for row in _ROWS}

GROUPS = {
    "hosted-qwen": [key for key, row in CONTROLLERS.items() if row["model"] == "qwen3.8-27b"],
    "hosted-glm": [
        key
        for key, row in CONTROLLERS.items()
        if row["model"] == "glm-5.3" and row["serving_kind"] == "hosted"
    ],
    "dedicated-a-canary": ["glm53-dedicated-a-canary"],
    "dedicated-a-bulk": ["glm53-dedicated-a-s1", "glm53-dedicated-a-s2"],
    "dedicated-b-canary": ["glm53-dedicated-b-canary"],
    "dedicated-b-bulk": ["glm53-dedicated-b-s1", "glm53-dedicated-b-s2"],
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any], field: str = "receipt_sha256") -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field}))


def load(path: Path) -> dict[str, Any]:
    return hosted.load(path)


def _selected_keys(row: dict[str, Any]) -> list[tuple[str, int, int]]:
    if row["canary"]:
        return [(row["model"], *row["canary"])]
    excluded = {("qwen3.8-27b", 4, 1), ("glm-5.3", 13, 1)}
    if row["replica"] == "A":
        excluded.add(("glm-5.3", 51, 1))
    if row["replica"] == "B":
        excluded.add(("glm-5.3", 76, 1))
    return [
        (row["model"], rank, attempt)
        for rank in row["ranks"]
        for attempt in (1, 2, 3, 4)
        if (row["model"], rank, attempt) not in excluded
    ]


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    row = CONTROLLERS[controller]
    source = hosted._predecessor_rows(root)  # noqa: SLF001 - reviewed identity authority
    cells = [copy.deepcopy(source[key]) for key in _selected_keys(row)]
    for ordinal, cell in enumerate(cells, 1):
        cell["ordinal"] = ordinal
    body = {
        "schema_version": PLAN_SCHEMA,
        "controller": controller,
        "job_name": row["job_name"],
        "configmap_name": row["configmap_name"],
        "sfs_root": f"/mnt/sfs/jobs/{row['job_name']}",
        "model": row["model"],
        "serving_kind": row["serving_kind"],
        "serving_block": row["serving_block"],
        "replica": row["replica"],
        "stream": row["stream"],
        "task_ranks": row["ranks"],
        "attempts": cells,
        "new_session_count": len(cells),
        "global_claim_root": hosted.CLAIM_ROOT,
        "endpoint_lease": {
            "lease_root": hosted.LEASE_ROOT,
            "endpoint_key": row["serving_block"],
            "maximum_streams": 4 if row["serving_kind"] == "hosted" else 2,
        },
        "resource_policy": {
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "whole_task_serving_block": True,
        "automatic_retry": False,
        "launch_authorized": False,
    }
    return {**body, "plan_sha256": digest(body, "plan_sha256")}


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {key: build_plan(key, root) for key in CONTROLLERS}
    selected: dict[tuple[str, int, int], str] = {}
    task_owner: dict[tuple[str, int], str] = {}
    for controller, plan in plans.items():
        for cell in plan["attempts"]:
            key = (plan["model"], cell["selection_rank"], cell["attempt"])
            if key in selected:
                raise ValueError("final v5 cell overlap")
            selected[key] = controller
            task = key[:2]
            block = plan["serving_block"]
            if task in task_owner and task_owner[task] != block:
                raise ValueError("final v5 task crosses serving treatment")
            task_owner[task] = block
    source = set(hosted._predecessor_rows(root))  # noqa: SLF001
    if set(selected) != source or len(selected) != 798:
        raise ValueError("final v5 partition has a gap or extra cell")
    expected = {
        "hosted-qwen": 399,
        "hosted-glm": 199,
        "dedicated-a-canary": 1,
        "dedicated-a-bulk": 99,
        "dedicated-b-canary": 1,
        "dedicated-b-bulk": 99,
    }
    if {
        group: sum(plans[key]["new_session_count"] for key in keys)
        for group, keys in GROUPS.items()
    } != expected:
        raise ValueError("final v5 group counts drifted")
    return plans


def build_runtime_plan(controller: str, inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    hosted.validate_inventory_gate(inventory, root)
    compact = validate_all(root)[controller]
    template_path = hosted.predecessor.RUNTIME_TEMPLATE_PATHS[compact["model"]]
    template = load(root / template_path)
    inventory_by_rank = {row["selection_rank"]: row for row in inventory["tasks"]}
    tasks = []
    for rank in compact["task_ranks"]:
        source = inventory_by_rank[rank]
        environment = copy.deepcopy(source["environment"])
        environment.pop("version_id_authority", None)
        environment.pop("runtime_seed_file_count", None)
        environment["ttl_seconds"] = 32400
        task = copy.deepcopy(source["task"])
        task.pop("version", None)
        tasks.append(
            {
                "rank": rank,
                "source_rank": rank,
                "environment": environment,
                "task": task,
                "verifier": copy.deepcopy(source["verifier"]),
            }
        )
    model = copy.deepcopy(template["model"])
    if compact["serving_kind"] == "dedicated":
        model["endpoint_origin"] = (
            f"http://chris-cyber-glm53-dedicated-{compact['replica'].lower()}-v7."
            "fleet-train-jobs.svc.cluster.local:8000"
        )
    execution = {
        "workers": 1,
        "same_task_max_inflight": 1,
        "attempts_per_task_sequential": True,
        "global_execution_claim_before_model_call": True,
        "claim_root": hosted.CLAIM_ROOT,
        "accepted_active_claimed_or_model_started_cells_are_nonrepeatable": True,
        "automatic_retry": False,
        "restart_resumes_only_unclaimed_cells": True,
        "endpoint_lease": compact["endpoint_lease"],
        "priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
        "cpu_only": True,
        "required_task_tools": ["bash", "submit_report"],
        "required_task_tool_catalog_sha256": template["execution"][
            "required_task_tool_catalog_sha256"
        ],
        "training_data_eligible": False,
    }
    harness = copy.deepcopy(template["harness"])
    harness["compaction_headroom_tokens"] = 20000
    body = {
        "schema_version": "fleet-exact-pass4-final-executable-plan-v5",
        "controller": controller,
        "campaign_id": compact["job_name"],
        "source_job_id": compact["job_name"],
        "sfs_root": compact["sfs_root"],
        "repo_root": ".",
        "inventory_receipt": inventory,
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "model": model,
        "harness": harness,
        "authority": copy.deepcopy(template["authority"]),
        "serving_kind": compact["serving_kind"],
        "serving_block": compact["serving_block"],
        "replica": compact["replica"],
        "stream": compact["stream"],
        "tasks": tasks,
        "attempts": [
            {
                **copy.deepcopy(cell),
                "rank": cell["selection_rank"],
                "source_rank": cell["selection_rank"],
            }
            for cell in compact["attempts"]
        ],
        "execution": execution,
        "treatment": hosted.predecessor.exact.EXPECTED_TREATMENT,
        "release_required": True,
        "launch_authorized": True,
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    plan = {**body, "plan_sha256": digest(body, "plan_sha256")}
    settings = self_hosted.opencode_settings(plan)
    canonical_settings = self_hosted.canonical_json(settings)
    harness["settings_canonical_sha256"] = self_hosted.sha256(canonical_settings)
    harness["settings_file_sha256"] = self_hosted.sha256(canonical_settings + b"\n")
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    if (
        harness["name"] != "opencode"
        or harness["version"] != "1.18.27"
        or harness["context_window_size"] != 262144
        or harness["compaction_headroom_tokens"] != 20000
        or execution["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("final v5 runtime treatment drifted")
    return plan


def identity_sha256(root: Path) -> str:
    plans = validate_all(root)
    rows = sorted(
        (
            {
                "model": plan["model"],
                "selection_rank": cell["selection_rank"],
                "attempt": cell["attempt"],
                "cell_id": cell["cell_id"],
                "execution_id": cell["execution_id"],
                "run_id": cell["run_id"],
                "serving_block": plan["serving_block"],
            }
            for plan in plans.values()
            for cell in plan["attempts"]
        ),
        key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
    )
    return sha256(canonical(rows))


def expected_held(root: Path) -> dict[str, Any]:
    plans = validate_all(root)
    body = {
        "schema_version": HELD_SCHEMA,
        "status": "HELD",
        "append_only": True,
        "launch_authorized": False,
        "objects_created": False,
        "base_commit": BASE_COMMIT,
        "total_cells": 798,
        "group_counts": {
            group: sum(plans[key]["new_session_count"] for key in keys)
            for group, keys in GROUPS.items()
        },
        "controller_count": len(plans),
        "exact_identity_sha256": identity_sha256(root),
        "hosted_independent_of_dedicated_readiness": True,
        "whole_task_serving_blocks": True,
        "hosted_endpoint_maximum_streams": 4,
        "dedicated_endpoint_maximum_streams": 2,
        "cpu_policy": {"priority_class": "fleet-serve-low", "preemption_policy": "Never"},
        "gpu_policy": {"priority_class": "fleet-infra-quiet", "preemption_policy": "Never"},
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_held(value: dict[str, Any], root: Path) -> None:
    if value != expected_held(root):
        raise ValueError("final v5 held receipt drifted")


PACKAGE_PATHS = (
    MODULE_PATH,
    RUNTIME_PATH,
    PACKAGE_PATH,
    RENDER_PATH,
    DEDICATED_EVIDENCE_PATH,
    DUPLICATE_OBSERVER_PATH,
    CREATE_RELAY_PATH,
    QUALIFIER_GATHER_PATH,
    QUALIFIER_PREPARE_PATH,
    RUN_PATH,
    SUBMIT_PATH,
    HELD_PATH,
    DOC_PATH,
    MANIFEST_PATH,
    "tests/test_exact_pass4_final_bulk_v5.py",
)


def validate_package_commit(root: Path, commit: str) -> None:
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("final v5 package commit is invalid")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *PACKAGE_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != commit or status:
        raise ValueError("final v5 package is not clean at its exact commit")
    for relative in PACKAGE_PATHS:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != (root / relative).read_bytes():
            raise ValueError(f"final v5 package byte drifted: {relative}")


def _validate_prebulk(receipt: dict[str, Any], root: Path) -> None:
    prebulk.validate_terminal(receipt, root)
    if receipt.get("status") != "CLEAR" or receipt.get("planned_execution_count") != 798:
        raise ValueError("final v5 prebulk is not clear")


def _group_identity(group: str, root: Path) -> list[dict[str, Any]]:
    plans = validate_all(root)
    return sorted(
        (
            {
                "controller": controller,
                "model": plans[controller]["model"],
                "cell_id": cell["cell_id"],
                "execution_id": cell["execution_id"],
                "run_id": cell["run_id"],
            }
            for controller in GROUPS[group]
            for cell in plans[controller]["attempts"]
        ),
        key=lambda row: (row["model"], row["cell_id"]),
    )


def validate_fresh_duplicate(
    receipt: dict[str, Any], group: str, root: Path, package_commit: str
) -> None:
    plans = validate_all(root)
    checked = [plans[key] for key in GROUPS[group]]
    observed = receipt.get("observed_at_utc")
    try:
        observed_at = datetime.strptime(str(observed), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("final v5 duplicate observation time is invalid") from exc
    if abs((datetime.now(UTC) - observed_at).total_seconds()) > 900:
        raise ValueError("final v5 duplicate observation is stale")
    if (
        receipt.get("schema_version") != FRESH_SCHEMA
        or receipt.get("status") != "CLEAR"
        or receipt.get("group") != group
        or receipt.get("planned_execution_count")
        != sum(plan["new_session_count"] for plan in checked)
        or receipt.get("exact_identity_sha256") != sha256(canonical(_group_identity(group, root)))
        or receipt.get("checked_job_names") != sorted(plan["job_name"] for plan in checked)
        or receipt.get("checked_configmap_names")
        != sorted(plan["configmap_name"] for plan in checked)
        or receipt.get("checked_output_roots") != sorted(plan["sfs_root"] for plan in checked)
        or receipt.get("collisions")
        != {
            "fleet_api": 0,
            "kubernetes_job_pod_or_configmap": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "accepted_active_or_model_started_cell": 0,
        }
        or receipt.get("methods") != ["GET"]
        or receipt.get("mutation_calls") != 0
        or receipt.get("checked_immediately_before_release") is not True
        or receipt.get("observer_job_succeeded") is not True
        or receipt.get("observer_pod_restarts") != 0
        or receipt.get("observer_package_commit") != package_commit
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("credentials_included") is not False
        or receipt.get("receipt_sha256") != digest(receipt)
    ):
        raise ValueError("final v5 fresh duplicate gate is not clear")
    uuid.UUID(str(receipt.get("observer_job_uid")))
    uuid.UUID(str(receipt.get("observer_pod_uid")))


def validate_selected_qualifier(
    model: str,
    launch_release: dict[str, Any],
    terminal: dict[str, Any],
    model_receipt: dict[str, Any],
    job: dict[str, Any],
    pods: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    hosted.qualifier_release.validate_release(launch_release, root)
    metadata, status = job.get("metadata") or {}, job.get("status") or {}
    items = pods.get("items")
    conditions = status.get("conditions") or []
    if (
        metadata.get("name") != hosted.qualifier_release.package.JOB_NAME
        or not any(
            row.get("type") == "Complete" and row.get("status") == "True" for row in conditions
        )
        or any(row.get("type") == "Failed" and row.get("status") == "True" for row in conditions)
        or status.get("active", 0) not in (0, None)
        or status.get("succeeded") != 1
        or status.get("failed", 0) not in (0, None)
        or not isinstance(items, list)
        or len(items) != 1
    ):
        raise ValueError("final v5 qualifier Job is not exclusively complete")
    pod = items[0]
    job_uid = str(uuid.UUID(metadata["uid"]))
    pod_uid = str(uuid.UUID(pod["metadata"]["uid"]))
    owners = pod["metadata"].get("ownerReferences") or []
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if (
        owners
        != [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": hosted.qualifier_release.package.JOB_NAME,
                "uid": job_uid,
                "controller": True,
            }
        ]
        or (pod.get("status") or {}).get("phase") != "Succeeded"
        or len(statuses) != 1
        or statuses[0].get("name") != "qualifier"
        or statuses[0].get("restartCount") != 0
        or ((statuses[0].get("state") or {}).get("terminated") or {}).get("exitCode") != 0
    ):
        raise ValueError("final v5 qualifier Pod did not cleanly succeed")
    runtime = terminal.get("runtime") or {}
    selected = [row for row in terminal.get("models") or [] if row.get("served_id") == model]
    # Reuse the exact model-result validator above, then additionally bind it
    # to the G7-gated launch release and exclusive Job/Pod terminal.
    expected_model = qualifier.EXPECTED_MODELS[model]
    if (
        model_receipt.get("schema_version") != qualifier.SCHEMA
        or model_receipt.get("status") != "PASSED"
        or model_receipt.get("model", {}).get("repository") != expected_model["repository"]
        or model_receipt.get("model", {}).get("revision") != expected_model["revision"]
        or model_receipt.get("receipt_sha256") != digest(model_receipt)
        or model_receipt.get("decision")
        != qualifier.evaluate_waves(model_receipt.get("waves") or [])
        or model_receipt["decision"].get("accepted") is not True
        or selected
        != [
            {
                "served_id": model,
                "status": "PASSED",
                "receipt_sha256": model_receipt["receipt_sha256"],
            }
        ]
        or terminal.get("schema_version") != qualifier.SCHEMA
        or terminal.get("status") not in {"PASSED", "REJECTED"}
        or terminal.get("classification") != "operational_gate_no_capability_claim"
        or hosted.ISO_UTC_RE.fullmatch(str(terminal.get("observed_at_utc"))) is None
        or runtime.get("job_uid") != job_uid
        or runtime.get("pod_uid") != pod_uid
        or runtime.get("package_commit") != launch_release["implementation"]["commit"]
        or runtime.get("package_sha256") != launch_release["probe_package"]["sha256"]
        or runtime.get("priority_class") != "fleet-serve-low"
        or runtime.get("preemption_policy") != "Never"
        or runtime.get("cpu_only") is not True
        or terminal.get("endpoint_lease")
        != {
            "namespace": qualifier.LEASE_NAMESPACE,
            "separate_from_scored_endpoint_leases": True,
            "released": True,
        }
        or terminal.get("scored_bulk_launch_authorized") is not False
        or terminal.get("request_counts")
        != {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 24,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        }
        or terminal.get("privacy")
        != {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        or terminal.get("receipt_sha256") != digest(terminal)
    ):
        raise ValueError("final v5 selected qualifier did not pass exact G7-gated runtime")
    return {
        "launch_release_receipt_sha256": launch_release["receipt_sha256"],
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "model_receipt_sha256": model_receipt["receipt_sha256"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
    }


def build_release(
    group: str,
    root: Path,
    package_commit: str,
    *,
    prebulk_terminal: dict[str, Any],
    fresh_duplicate: dict[str, Any],
    qualifier_launch_release: dict[str, Any] | None = None,
    qualifier_model: dict[str, Any] | None = None,
    qualifier_terminal: dict[str, Any] | None = None,
    qualifier_job: dict[str, Any] | None = None,
    qualifier_pods: dict[str, Any] | None = None,
    dedicated_parity: dict[str, Any] | None = None,
    dedicated_canary: dict[str, Any] | None = None,
    dedicated_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if group not in GROUPS:
        raise ValueError("unsupported final v5 release group")
    validate_package_commit(root, package_commit)
    _validate_prebulk(prebulk_terminal, root)
    # A structurally valid receipt is not sufficient here.  The dedicated
    # source/accept observer revalidates freshness and binds both exact
    # terminal Job/Pod UID pairs immediately before this release is rendered.
    from evals.fleet import exact_pass4_final_duplicate_observer_v5 as duplicate

    duplicate.validate_accepted_for_release(
        fresh_duplicate,
        group,
        root,
        package_commit,
    )
    plans = validate_all(root)
    gates: dict[str, Any] = {
        "prebulk_receipt_sha256": prebulk_terminal["receipt_sha256"],
        "fresh_duplicate_source_accept": {
            "receipt_sha256": fresh_duplicate["receipt_sha256"],
            "source_job_uid": fresh_duplicate["observer_job_uid"],
            "source_pod_uid": fresh_duplicate["observer_pod_uid"],
            "accept_job_uid": fresh_duplicate["acceptor_job_uid"],
            "accept_pod_uid": fresh_duplicate["acceptor_pod_uid"],
        },
    }
    if group.startswith("hosted-"):
        model = "qwen3.8-27b" if group == "hosted-qwen" else "glm-5.3"
        if any(
            value is None
            for value in (
                qualifier_launch_release,
                qualifier_model,
                qualifier_terminal,
                qualifier_job,
                qualifier_pods,
            )
        ):
            raise ValueError("hosted release requires its model-specific qualifier")
        qualifier_binding = validate_selected_qualifier(
            model,
            qualifier_launch_release,
            qualifier_terminal,
            qualifier_model,
            qualifier_job,
            qualifier_pods,
            root,
        )
        expected_model = qualifier.EXPECTED_MODELS[model]
        expected_contexts = (
            {
                "expected_context_length": 262144,
                "context_length_observable": False,
                "observed_context_length": None,
                "unobservable_roster_context_requires_existing_exact_harness_gate": True,
            },
            {
                "expected_context_length": 262144,
                "context_length_observable": True,
                "observed_context_length": 262144,
                "unobservable_roster_context_requires_existing_exact_harness_gate": True,
            },
        )
        if (
            qualifier_model.get("schema_version") != qualifier.SCHEMA
            or qualifier_model.get("status") != "PASSED"
            or qualifier_model.get("model")
            != {
                "served_id": model,
                "repository": expected_model["repository"],
                "revision": expected_model["revision"],
                "endpoint_origin": qualifier.ORIGIN,
                "response_model_exact": True,
            }
            or qualifier_model.get("context_contract") not in expected_contexts
            or qualifier_model.get("tool_contract")
            != {
                "names": ["bash", "submit_report"],
                "openai_tool_schema_sha256": qualifier.sha256(
                    qualifier.canonical_json(qualifier.TOOLS)
                ),
                "forced_selection_order_per_stream": ["bash", "submit_report"],
                "argument_shapes_valid": True,
                "tool_execution_performed": False,
            }
            or qualifier_model.get("decision")
            != qualifier.evaluate_waves(qualifier_model.get("waves") or [])
            or not qualifier_model["decision"]["accepted"]
            or qualifier_model.get("lease")
            != {"namespace": qualifier.LEASE_NAMESPACE, "exclusive": True}
            or qualifier_model.get("request_counts")
            != {
                "chat_completions": 12,
                "task_instance": 0,
                "session": 0,
                "scoring": 0,
                "verifier": 0,
            }
            or qualifier_model.get("privacy")
            != {
                "request_bodies_included": False,
                "response_bodies_included": False,
                "tool_arguments_included": False,
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            }
            or qualifier_model.get("receipt_sha256") != digest(qualifier_model)
        ):
            raise ValueError("model-specific hosted qualifier did not pass")
        terminal_models = qualifier_terminal.get("models") or []
        selected = [row for row in terminal_models if row.get("served_id") == model]
        if (
            qualifier_terminal.get("schema_version") != qualifier.SCHEMA
            or qualifier_terminal.get("status") not in {"PASSED", "REJECTED"}
            or qualifier_terminal.get("classification") != "operational_gate_no_capability_claim"
            or selected
            != [
                {
                    "served_id": model,
                    "status": "PASSED",
                    "receipt_sha256": qualifier_model["receipt_sha256"],
                }
            ]
            or qualifier_terminal.get("endpoint_lease")
            != {
                "namespace": qualifier.LEASE_NAMESPACE,
                "separate_from_scored_endpoint_leases": True,
                "released": True,
            }
            or qualifier_terminal.get("scored_bulk_launch_authorized") is not False
            or qualifier_terminal.get("receipt_sha256") != digest(qualifier_terminal)
        ):
            raise ValueError("hosted qualifier terminal does not bind the model PASS")
        gates["qualifier_model_receipt_sha256"] = qualifier_model["receipt_sha256"]
        gates["qualifier_terminal_receipt_sha256"] = qualifier_terminal["receipt_sha256"]
        gates["qualifier_binding"] = qualifier_binding
    else:
        replica = "A" if "-a-" in group else "B"
        value = dedicated.spec(root)
        if dedicated_parity is None:
            raise ValueError("dedicated release requires UID-bound parity")
        dedicated.validate_parity_gate(dedicated_parity, value, root, replica)
        gates["parity_receipt_sha256"] = dedicated_parity["receipt_sha256"]
        if group.endswith("-bulk"):
            if dedicated_canary is None or dedicated_runtime is None:
                raise ValueError("dedicated bulk requires accepted canary and runtime gate")
            dedicated.validate_runtime_gate(
                dedicated_runtime, dedicated_parity, dedicated_canary, value, root, replica
            )
            gates["canary_receipt_sha256"] = dedicated_canary["receipt_sha256"]
            gates["runtime_receipt_sha256"] = dedicated_runtime["receipt_sha256"]
    body = {
        "schema_version": RELEASE_SCHEMA,
        "status": "RELEASED",
        "append_only": True,
        "launch_authorized": True,
        "group": group,
        "package_commit": package_commit,
        "base_commit": BASE_COMMIT,
        "exact_identity_sha256": identity_sha256(root),
        "credential_authority": {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        },
        "controllers": [
            {
                "controller": key,
                "job_name": plans[key]["job_name"],
                "configmap_name": plans[key]["configmap_name"],
                "sfs_root": plans[key]["sfs_root"],
                "plan_sha256": plans[key]["plan_sha256"],
                "session_count": plans[key]["new_session_count"],
            }
            for key in GROUPS[group]
        ],
        "gates": gates,
        "fresh_duplicate_receipt_sha256": fresh_duplicate["receipt_sha256"],
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_runtime_release(release: dict[str, Any], plan: dict[str, Any], root: Path) -> None:
    group = release.get("group")
    plans = validate_all(root)
    if (
        group not in GROUPS
        or plan.get("controller") not in GROUPS[group]
        or release.get("schema_version") != RELEASE_SCHEMA
        or release.get("status") != "RELEASED"
        or release.get("launch_authorized") is not True
        or release.get("base_commit") != BASE_COMMIT
        or release.get("exact_identity_sha256") != identity_sha256(root)
        or release.get("credential_authority")
        != {
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
            "fleet_team_id": FLEET_TEAM_ID,
        }
        or release.get("controllers")
        != [
            {
                "controller": key,
                "job_name": plans[key]["job_name"],
                "configmap_name": plans[key]["configmap_name"],
                "sfs_root": plans[key]["sfs_root"],
                "plan_sha256": plans[key]["plan_sha256"],
                "session_count": plans[key]["new_session_count"],
            }
            for key in GROUPS[group]
        ]
        or release.get("receipt_sha256") != digest(release)
    ):
        raise ValueError("final v5 runtime release drifted")
    rebuilt = build_runtime_plan(plan["controller"], plan["inventory_receipt"], root)
    if plan != rebuilt:
        raise ValueError("final v5 runtime plan drifted")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "validate-held", "build-release"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--group", choices=tuple(GROUPS))
    parser.add_argument("--package-commit")
    parser.add_argument("--prebulk", type=Path)
    parser.add_argument("--fresh-duplicate", type=Path)
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
    root = args.repo.resolve()
    if args.command == "validate-held":
        validate_held(load(root / HELD_PATH), root)
        return 0
    if args.command == "build-release":
        if (
            not args.group
            or not args.package_commit
            or not args.prebulk
            or not args.fresh_duplicate
            or not args.output
        ):
            parser.error(
                "build-release requires group, package commit, prebulk, fresh duplicate, and output"
            )
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("final v5 release output already exists")

        def optional(path: Path | None) -> dict[str, Any] | None:
            return load(path.resolve(strict=True)) if path else None

        value = build_release(
            args.group,
            root,
            args.package_commit,
            prebulk_terminal=load(args.prebulk.resolve(strict=True)),
            fresh_duplicate=load(args.fresh_duplicate.resolve(strict=True)),
            qualifier_launch_release=optional(args.qualifier_launch_release),
            qualifier_model=optional(args.qualifier_model),
            qualifier_terminal=optional(args.qualifier_terminal),
            qualifier_job=optional(args.qualifier_job),
            qualifier_pods=optional(args.qualifier_pods),
            dedicated_parity=optional(args.dedicated_parity),
            dedicated_canary=optional(args.dedicated_canary),
            dedicated_runtime=optional(args.dedicated_runtime),
        )
        args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        args.output.chmod(0o400)
        return 0
    plans = validate_all(root)
    print(
        json.dumps(
            {
                "status": "HELD",
                "launch_authorized": False,
                "objects_created": False,
                "controllers": len(plans),
                "groups": {
                    group: sum(plans[key]["new_session_count"] for key in keys)
                    for group, keys in GROUPS.items()
                },
                "total_cells": 798,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
