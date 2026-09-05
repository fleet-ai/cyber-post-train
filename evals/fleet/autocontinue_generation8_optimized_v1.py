"""Held Generation-8 canary fallback with single-pass immutable validation.

This module is deliberately not a launch authorization.  It defines the
fail-closed evidence, release, and runtime mechanics that may be used only
after both Generation-7 Jobs are terminal before claim/output/session state.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation7_authority_v1 as authority7
from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation8-canary-spec-v1"
PLAN_SCHEMA = "fleet-hosted-opencode-task-boundary-shard-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation8-optimized-held-v1"
TOMBSTONE_SCHEMA = "fleet-opencode-generation7-preclaim-stop-tombstone-v1"
PRESTOP_SCHEMA = "fleet-opencode-generation7-controlled-prestop-v1"
DELETE_AUTH_SCHEMA = "fleet-opencode-generation7-foreground-delete-authorization-v1"
POSTSTOP_SCHEMA = "fleet-opencode-generation7-controlled-poststop-v1"
DELETE_RECEIPT_SCHEMA = "fleet-opencode-generation7-foreground-delete-receipt-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation8-scoring-release-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v10"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation8-terminal-v1"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")

MODULE_PATH = "evals/fleet/autocontinue_generation8_optimized_v1.py"
PACKAGE_PATH = "evals/fleet/autocontinue_generation8_package_v1.py"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation8_v1.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation8_v1.sh"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation8-held-v1.yaml"
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation8-optimized-held-v1.json"
)
DOC_PATH = "docs/GENERATION8_OPTIMIZED_FALLBACK.md"
CLAIM_ROOT = generation7.generation6.generation5.CLAIM_ROOT
NAMESPACE = "fleet-train-jobs"

MODELS = {
    "qwen3.8-27b": {
        "short": "q38",
        "rank": 4,
        "spec": "evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-v1.json",
        "plan": "evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-plan-v1.json",
        "g7_job_uid": "123c0026-0de9-43ab-8703-c92b34063fe1",
        "g7_pod_uid": "30229380-8ac8-4f95-b769-dead83397752",
        "run_ids": (
            "chris-q38-ac-canary1-v1-sr004-a1-02dd4e3f",
            "chris-q38-ac-canary1-v2-sr004-a1-02dd4e3f",
            "chris-q38-ac-g2-r004-a1-02dd4e3f",
            "chris-q38-ac-g3-r004-a1-02dd4e3f",
            "chris-q38-ac-g4-r004-a1-02dd4e3f",
            "chris-q38-ac-g5-r004-a1-02dd4e3f",
            "chris-q38-ac-g6-r004-a1-fa0c9b85",
            "chris-q38-ac-g7-r004-a1-77deaade",
            "chris-q38-ac-g8-r004-a1-88b1f006",
        ),
    },
    "glm-5.3": {
        "short": "glm53",
        "rank": 13,
        "spec": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-v1.json",
        "plan": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-plan-v1.json",
        "g7_job_uid": "2f2dbf17-3d78-49be-8969-5ab0b8280f8e",
        "g7_pod_uid": "ec18d919-9e2e-47d8-8c0a-b5e727db4921",
        "run_ids": (
            "chris-glm53-ac-canary1-v1-sr013-a1-9375a9b9",
            "chris-glm53-ac-canary1-v2-sr013-a1-9375a9b9",
            "chris-glm53-ac-g2-r013-a1-9375a9b9",
            "chris-glm53-ac-g3-r013-a1-9375a9b9",
            "chris-glm53-ac-g4-r013-a1-9375a9b9",
            "chris-glm53-ac-g5-r013-a1-9375a9b9",
            "chris-glm53-ac-g6-r013-a1-a63dbb28",
            "chris-glm53-ac-g7-r013-a1-b6338d53",
            "chris-glm53-ac-g8-r013-a1-a8b2f9b6",
        ),
    },
}
RELEASE_PATHS = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-autocontinue-generation8-scoring-release-v1.json"
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-glm53-autocontinue-generation8-scoring-release-v1.json"
    ),
}

EXPECTED_HELD_RECEIPT_SHA256 = (
    "sha256:cec21a7e7d501633bd9871ceb4a6d07a9a749884a20f0d5a0a03bbd5f046e2ec"
)
MEASURED_G7_ONE_MODEL_VALIDATE_SECONDS = 61.53
REQUIRED_VALIDATION_SPEEDUP = 10.0


def canonical(value: Any) -> bytes:
    if isinstance(value, Mapping):
        value = dict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: Mapping[str, Any], field_name: str = "receipt_sha256") -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field_name}))


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("immutable JSON root is not an object")
    return value


@dataclass
class ValidationLedger:
    """Process-local proof that every immutable input is parsed once.

    Callers pass the validated objects onward.  ``assert_reuse`` compares
    canonical bytes with the first validated bytes; it never reloads or
    revalidates a file and therefore cannot silently switch inputs.
    """

    values: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    bytes_by_label: dict[str, bytes] = field(default_factory=dict)
    read_counts: dict[str, int] = field(default_factory=dict)

    def read_once(self, label: str, path: Path) -> Mapping[str, Any]:
        if label in self.values:
            raise RuntimeError(f"immutable input requested twice: {label}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"immutable input is absent or unsafe: {label}")
        raw = path.read_bytes()
        value = _strict_json(raw)
        if raw != canonical(value) + b"\n":
            raise ValueError(f"immutable input is not canonical JSON: {label}")
        frozen = MappingProxyType(value)
        self.values[label] = frozen
        self.bytes_by_label[label] = canonical(value)
        self.read_counts[label] = 1
        return frozen

    def assert_reuse(self, label: str, value: Mapping[str, Any]) -> None:
        if label not in self.bytes_by_label or canonical(value) != self.bytes_by_label[label]:
            raise RuntimeError(f"validated immutable value changed on reuse: {label}")


@dataclass(frozen=True)
class ValidatedContext:
    model: str
    root: Path
    package_commit: str
    spec: Mapping[str, Any]
    plan: Mapping[str, Any]
    held: Mapping[str, Any]
    g7_release: Mapping[str, Any]
    release: Mapping[str, Any]
    tombstone: Mapping[str, Any]
    package_manifest: Mapping[str, Any]
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]
    ledger: ValidationLedger

    def assert_unchanged(self) -> None:
        for label, value in (
            (f"spec:{self.model}", self.spec),
            (f"plan:{self.model}", self.plan),
            ("held", self.held),
            (f"g7-release:{self.model}", self.g7_release),
            ("release", self.release),
            ("tombstone", self.tombstone),
            ("package", self.package_manifest),
        ):
            self.ledger.assert_reuse(label, value)
        for model, (spec, plan) in self.static.items():
            self.ledger.assert_reuse(f"spec:{model}", spec)
            self.ledger.assert_reuse(f"plan:{model}", plan)
            self.ledger.assert_reuse(
                f"g7-release:{model}", self.ledger.values[f"g7-release:{model}"]
            )


def _expected_names(model: str) -> dict[str, str]:
    row = MODELS[model]
    short, rank = row["short"], row["rank"]
    job = f"chris-{short}-ac-r{rank:03d}-a1-g8-v1"
    return {
        "job": job,
        "configmap": f"chris-{short}-ac-r{rank:03d}-a1-g8-run-v1",
        "output": f"/mnt/sfs/jobs/{job}",
    }


def validate_static(
    root: Path,
    model: str,
    ledger: ValidationLedger,
    *,
    held: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if model not in MODELS:
        raise ValueError("Generation-8 model is unsupported")
    row = MODELS[model]
    spec = ledger.read_once(f"spec:{model}", root / row["spec"])
    plan = ledger.read_once(f"plan:{model}", root / row["plan"])
    if held is None:
        held = ledger.read_once("held", root / HELD_PATH)
    g7_release = ledger.read_once(f"g7-release:{model}", root / authority7.RELEASE_PATHS[model])
    names = _expected_names(model)
    held_binding = held.get("model_bindings", {}).get(model)
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("model") != model
        or spec.get("generation8_spec_sha256") != digest(spec, "generation8_spec_sha256")
        or spec.get("execution", {}).get("execution_generation") != 8
        or spec.get("identities", {}).get("job_name") != names["job"]
        or spec.get("identities", {}).get("configmap_name") != names["configmap"]
        or spec.get("identities", {}).get("sfs_root") != names["output"]
        or spec.get("identities", {}).get("generation_claim_root") != CLAIM_ROOT
        or spec.get("rendered_plan_sha256") != plan.get("plan_sha256")
        or plan.get("plan_sha256") != digest(plan, "plan_sha256")
        or plan.get("model", {}).get("served_id") != model
        or plan.get("attempts", [{}])[0].get("execution_generation") != 8
        or plan.get("attempts", [{}])[0].get("run_id") != spec.get("identities", {}).get("run_id")
        or plan.get("attempts", [{}])[0].get("network") != spec.get("identities", {}).get("network")
        or plan.get("execution", {}).get("execution_generation") != 8
        or plan.get("execution", {}).get("required_task_tools") != ["bash", "submit_report"]
        or plan.get("execution", {}).get("launch_authorized") is not False
        or plan.get("harness", {}).get("name") != "opencode"
        or plan.get("harness", {}).get("version") != "1.18.27"
        or plan.get("harness", {}).get("context_window_size") != 262144
        or plan.get("harness", {}).get("compaction_headroom_tokens") != 20000
        or plan.get("authority", {}).get("scoring_mode") != "partial"
        or spec.get("statistical_cell", {}).get("cell_id")
        != plan.get("source", {}).get("statistical_cell_id")
        or held_binding
        != {
            "cell_id": spec.get("statistical_cell", {}).get("cell_id"),
            "execution_id": spec.get("execution", {}).get("execution_id"),
            "g7_job_uid": row["g7_job_uid"],
            "g7_pod_uid": row["g7_pod_uid"],
            "generation8_spec_sha256": spec.get("generation8_spec_sha256"),
            "job_name": names["job"],
            "plan_sha256": plan.get("plan_sha256"),
        }
    ):
        raise ValueError("Generation-8 spec/plan treatment drifted")
    predecessor = spec["predecessor_generation7"]
    if (
        predecessor.get("spec_path") != generation7.G7_SPEC_PATHS[model]
        or predecessor.get("job_name") != generation7.EXPECTED[model]["job_name"]
        or predecessor.get("job_uid") != row["g7_job_uid"]
        or predecessor.get("pod_uid") != row["g7_pod_uid"]
        or predecessor.get("scoring_release_path") != authority7.RELEASE_PATHS[model]
        or predecessor.get("scoring_release_receipt_sha256") != g7_release.get("receipt_sha256")
        or predecessor.get("scoring_release_file_sha256")
        != sha256(ledger.bytes_by_label[f"g7-release:{model}"] + b"\n")
        or g7_release.get("receipt_sha256") != digest(g7_release)
        or g7_release.get("model") != model
        or g7_release.get("cell_id") != spec.get("statistical_cell", {}).get("cell_id")
    ):
        raise ValueError("Generation-7 immutable predecessor binding drifted")
    settings = self_hosted.opencode_settings(copy.deepcopy(dict(plan)))
    settings_raw = self_hosted.canonical_json(settings)
    if (
        plan["harness"].get("settings_canonical_sha256") != self_hosted.sha256(settings_raw)
        or plan["harness"].get("settings_file_sha256") != self_hosted.sha256(settings_raw + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
    ):
        raise ValueError("Generation-8 OpenCode settings drifted")
    return spec, plan, held, g7_release


def validate_held(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != HELD_SCHEMA
        or value.get("status") != "HELD"
        or value.get("launch_authorized") is not False
        or value.get("objects_created") is not False
        or value.get("models") != sorted(MODELS)
        or value.get("receipt_sha256") != digest(value)
        or value.get("receipt_sha256") != EXPECTED_HELD_RECEIPT_SHA256
    ):
        raise ValueError("Generation-8 held authority drifted")


def generation_output_paths(model: str, spec: Mapping[str, Any]) -> list[str]:
    short = MODELS[model]["short"]
    rank = MODELS[model]["rank"]
    return [
        f"/mnt/sfs/jobs/chris-{short}-ac-canary1-v1",
        f"/mnt/sfs/jobs/chris-{short}-ac-canary1-v2",
        *[
            f"/mnt/sfs/jobs/chris-{short}-ac-r{rank:03d}-a1-g{generation}-v1"
            for generation in range(2, 9)
        ],
    ]


def generation_claim_paths(spec: Mapping[str, Any]) -> list[str]:
    cell = spec["statistical_cell"]["cell_id"]
    executions = [
        generation7.exact.execution_for(cell, generation)["execution_id"]
        for generation in range(1, 9)
    ]
    return [f"{CLAIM_ROOT}/{value.removeprefix('sha256:')}.json" for value in executions]


def _identity_values(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                found.add(item)
            found.update(_identity_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_identity_values(item, keys))
    return found


def _absence_row(
    model: str,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    path_exists: Callable[[str], bool],
    sessions: Callable[[str], list[dict[str, Any]]],
) -> dict[str, Any]:
    claims = generation_claim_paths(spec)
    outputs = generation_output_paths(model, spec)
    if any(path_exists(path) for path in [*claims, *outputs]):
        raise RuntimeError("Generation-1-through-8 claim/output identity exists")
    run_ids = list(MODELS[model]["run_ids"])
    if (
        plan["attempts"][0]["run_id"] != run_ids[-1]
        or generation7.EXPECTED[model]["run_id"] != run_ids[-2]
    ):
        raise RuntimeError("Generation-1-through-8 run identity binding drifted")
    expected = {
        spec["statistical_cell"]["cell_id"],
        *(
            generation7.exact.execution_for(spec["statistical_cell"]["cell_id"], generation)[
                "execution_id"
            ]
            for generation in range(1, 9)
        ),
        *run_ids,
    }
    task_key = plan["tasks"][0]["task"]["key"]
    rows = sessions(task_key)
    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        raise RuntimeError("Fleet session metadata response is invalid")
    if any(
        _identity_values(item, {"cell_id", "execution_id", "run_id"}) & expected for item in rows
    ):
        raise RuntimeError("Generation-1-through-8 authoritative session identity exists")
    return {
        "checked_claim_paths": claims,
        "checked_output_paths": outputs,
        "run_ids_checked": run_ids,
        "task_key": task_key,
        "session_rows_examined": len(rows),
        "matching_sessions": 0,
    }


def _validate_absence_row(
    row: Mapping[str, Any],
    model: str,
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> int:
    if (
        row.get("checked_claim_paths") != generation_claim_paths(spec)
        or row.get("checked_output_paths") != generation_output_paths(model, spec)
        or row.get("run_ids_checked") != list(MODELS[model]["run_ids"])
        or row.get("task_key") != plan["tasks"][0]["task"]["key"]
        or not isinstance(row.get("session_rows_examined"), int)
        or row.get("session_rows_examined") < 0
        or row.get("matching_sessions") != 0
    ):
        raise ValueError("Generation-7 stop absence binding drifted")
    return row["session_rows_examined"]


def build_prestop(
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    kubernetes: Callable[[str, str], dict[str, Any]],
    path_exists: Callable[[str], bool],
    sessions: Callable[[str], list[dict[str, Any]]],
    observed_at_utc: str,
) -> dict[str, Any]:
    """Capture the exact active G7 objects and all pre-model absence evidence."""
    models = []
    total_rows = 0
    for model in sorted(MODELS):
        spec, plan = static[model]
        expected = MODELS[model]
        job_name = generation7.EXPECTED[model]["job_name"]
        job = kubernetes("job", job_name)
        pods = kubernetes("pods", job_name).get("items", [])
        terminal = [
            item
            for item in job.get("status", {}).get("conditions", [])
            if item.get("type") in {"Complete", "Failed"} and item.get("status") == "True"
        ]
        if (
            job.get("metadata", {}).get("uid") != expected["g7_job_uid"]
            or not isinstance(job.get("metadata", {}).get("resourceVersion"), str)
            or job.get("status", {}).get("active") != 1
            or terminal
            or len(pods) != 1
            or pods[0].get("metadata", {}).get("uid") != expected["g7_pod_uid"]
            or not isinstance(pods[0].get("metadata", {}).get("name"), str)
            or not isinstance(pods[0].get("metadata", {}).get("resourceVersion"), str)
            or pods[0].get("status", {}).get("phase") not in {"Pending", "Running"}
        ):
            raise RuntimeError("Generation-7 active pre-stop identity drifted")
        absence = _absence_row(model, spec, plan, path_exists=path_exists, sessions=sessions)
        total_rows += absence["session_rows_examined"]
        models.append(
            {
                "model": model,
                "cell_id": spec["statistical_cell"]["cell_id"],
                "g7_job": {
                    "name": job_name,
                    "uid": expected["g7_job_uid"],
                    "resource_version": job["metadata"]["resourceVersion"],
                    "state": "Active",
                },
                "g7_pod": {
                    "name": pods[0]["metadata"]["name"],
                    "uid": expected["g7_pod_uid"],
                    "resource_version": pods[0]["metadata"]["resourceVersion"],
                    "phase": pods[0]["status"]["phase"],
                },
                **absence,
            }
        )
    body = {
        "schema_version": PRESTOP_SCHEMA,
        "status": "G7_ACTIVE_PRECLAIM_PREOUTPUT_PRESESSION",
        "observed_at_utc": observed_at_utc,
        "models": models,
        "total_session_rows_examined": total_rows,
        "generation_range_checked": [1, 8],
        "claims_outputs_sessions_absent": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_prestop(
    value: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "observed_at_utc",
            "models",
            "total_session_rows_examined",
            "generation_range_checked",
            "claims_outputs_sessions_absent",
            "prompts_traces_flags_or_scores_included",
            "credentials_included",
            "receipt_sha256",
        }
        or value.get("schema_version") != PRESTOP_SCHEMA
        or value.get("status") != "G7_ACTIVE_PRECLAIM_PREOUTPUT_PRESESSION"
        or value.get("generation_range_checked") != [1, 8]
        or value.get("claims_outputs_sessions_absent") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError("Generation-7 controlled pre-stop receipt drifted")
    rows = value.get("models")
    if not isinstance(rows, list) or [row.get("model") for row in rows] != sorted(MODELS):
        raise ValueError("Generation-7 controlled pre-stop model set drifted")
    total = 0
    for row in rows:
        model = row["model"]
        spec, plan = static[model]
        expected = MODELS[model]
        if (
            set(row)
            != {
                "model",
                "cell_id",
                "g7_job",
                "g7_pod",
                "checked_claim_paths",
                "checked_output_paths",
                "run_ids_checked",
                "task_key",
                "session_rows_examined",
                "matching_sessions",
            }
            or row.get("cell_id") != spec["statistical_cell"]["cell_id"]
            or row.get("g7_job", {}).get("name") != generation7.EXPECTED[model]["job_name"]
            or row.get("g7_job", {}).get("uid") != expected["g7_job_uid"]
            or row.get("g7_job", {}).get("state") != "Active"
            or not isinstance(row.get("g7_job", {}).get("resource_version"), str)
            or row.get("g7_pod", {}).get("uid") != expected["g7_pod_uid"]
            or row.get("g7_pod", {}).get("phase") not in {"Pending", "Running"}
            or not isinstance(row.get("g7_pod", {}).get("name"), str)
            or not isinstance(row.get("g7_pod", {}).get("resource_version"), str)
        ):
            raise ValueError("Generation-7 controlled pre-stop binding drifted")
        total += _validate_absence_row(row, model, spec, plan)
    if value.get("total_session_rows_examined") != total:
        raise ValueError("Generation-7 controlled pre-stop session total drifted")


def build_delete_authorization(
    prestop: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_prestop(prestop, static)
    requests = []
    for row in prestop["models"]:
        requests.append(
            {
                "method": "DELETE",
                "namespace": NAMESPACE,
                "kind": "Job",
                "name": row["g7_job"]["name"],
                "uid_precondition": row["g7_job"]["uid"],
                "propagation_policy": "Foreground",
            }
        )
    body = {
        "schema_version": DELETE_AUTH_SCHEMA,
        "status": "HELD_UID_PRECONDITIONED_FOREGROUND_DELETE",
        "prestop_receipt_sha256": prestop["receipt_sha256"],
        "requests": requests,
        "execution_command": [
            "python",
            "-m",
            "evals.fleet.autocontinue_generation8_optimized_v1",
            "execute-controlled-stop",
        ],
        "requires_global_claim_lock": True,
        "launch_or_delete_performed": False,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_delete_authorization(
    value: Mapping[str, Any],
    prestop: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    expected = build_delete_authorization(prestop, static)
    if value != expected:
        raise ValueError("Generation-7 foreground-delete authorization drifted")


def build_poststop(
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    prestop: Mapping[str, Any],
    *,
    kubernetes: Callable[[str, str], dict[str, Any]],
    path_exists: Callable[[str], bool],
    sessions: Callable[[str], list[dict[str, Any]]],
    observed_at_utc: str,
) -> dict[str, Any]:
    validate_prestop(prestop, static)
    models = []
    total_rows = 0
    for model in sorted(MODELS):
        spec, plan = static[model]
        expected = MODELS[model]
        job_name = generation7.EXPECTED[model]["job_name"]
        if kubernetes("job", job_name) or kubernetes("pods", job_name).get("items", []):
            raise RuntimeError("Generation-7 foreground deletion is not terminally absent")
        absence = _absence_row(model, spec, plan, path_exists=path_exists, sessions=sessions)
        total_rows += absence["session_rows_examined"]
        models.append(
            {
                "model": model,
                "cell_id": spec["statistical_cell"]["cell_id"],
                "g7_job": {
                    "name": job_name,
                    "uid": expected["g7_job_uid"],
                    "state": "AbsentAfterForegroundDelete",
                },
                "g7_pod": {
                    "uid": expected["g7_pod_uid"],
                    "state": "AbsentAfterForegroundDelete",
                },
                **absence,
            }
        )
    body = {
        "schema_version": POSTSTOP_SCHEMA,
        "status": "G7_FOREGROUND_DELETED_PRECLAIM_PREOUTPUT_PRESESSION",
        "observed_at_utc": observed_at_utc,
        "prestop_receipt_sha256": prestop["receipt_sha256"],
        "models": models,
        "total_session_rows_examined": total_rows,
        "generation_range_checked": [1, 8],
        "claims_outputs_sessions_absent": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_poststop(
    value: Mapping[str, Any],
    prestop: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "observed_at_utc",
            "prestop_receipt_sha256",
            "models",
            "total_session_rows_examined",
            "generation_range_checked",
            "claims_outputs_sessions_absent",
            "prompts_traces_flags_or_scores_included",
            "credentials_included",
            "receipt_sha256",
        }
        or value.get("schema_version") != POSTSTOP_SCHEMA
        or value.get("status") != "G7_FOREGROUND_DELETED_PRECLAIM_PREOUTPUT_PRESESSION"
        or value.get("prestop_receipt_sha256") != prestop["receipt_sha256"]
        or value.get("generation_range_checked") != [1, 8]
        or value.get("claims_outputs_sessions_absent") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError("Generation-7 controlled post-stop receipt drifted")
    rows = value.get("models")
    if not isinstance(rows, list) or [row.get("model") for row in rows] != sorted(MODELS):
        raise ValueError("Generation-7 controlled post-stop model set drifted")
    total = 0
    for row in rows:
        model = row["model"]
        spec, plan = static[model]
        expected = MODELS[model]
        if (
            set(row)
            != {
                "model",
                "cell_id",
                "g7_job",
                "g7_pod",
                "checked_claim_paths",
                "checked_output_paths",
                "run_ids_checked",
                "task_key",
                "session_rows_examined",
                "matching_sessions",
            }
            or row.get("cell_id") != spec["statistical_cell"]["cell_id"]
            or row.get("g7_job")
            != {
                "name": generation7.EXPECTED[model]["job_name"],
                "uid": expected["g7_job_uid"],
                "state": "AbsentAfterForegroundDelete",
            }
            or row.get("g7_pod")
            != {
                "uid": expected["g7_pod_uid"],
                "state": "AbsentAfterForegroundDelete",
            }
        ):
            raise ValueError("Generation-7 controlled post-stop binding drifted")
        total += _validate_absence_row(row, model, spec, plan)
    if value.get("total_session_rows_examined") != total:
        raise ValueError("Generation-7 controlled post-stop session total drifted")


def build_delete_receipt(
    authorization: Mapping[str, Any],
    prestop: Mapping[str, Any],
    fresh_prestop: Mapping[str, Any],
    poststop: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    started_at_utc: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    validate_prestop(prestop, static)
    validate_prestop(fresh_prestop, static)
    validate_delete_authorization(authorization, prestop, static)
    validate_poststop(poststop, fresh_prestop, static)
    original = {row["model"]: row for row in prestop["models"]}
    fresh = {row["model"]: row for row in fresh_prestop["models"]}
    for model in MODELS:
        if (
            original[model]["g7_job"] != fresh[model]["g7_job"]
            or original[model]["g7_pod"] != fresh[model]["g7_pod"]
        ):
            raise ValueError("Generation-7 changed between pre-stop and delete lock")
    body = {
        "schema_version": DELETE_RECEIPT_SCHEMA,
        "status": "FOREGROUND_DELETED_EXACT_G7_UIDS",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "prestop_receipt_sha256": prestop["receipt_sha256"],
        "fresh_prestop_receipt_sha256": fresh_prestop["receipt_sha256"],
        "poststop_receipt_sha256": poststop["receipt_sha256"],
        "deletions": [
            {
                "name": row["g7_job"]["name"],
                "uid_precondition": row["g7_job"]["uid"],
                "propagation_policy": "Foreground",
                "api_result": "Accepted",
            }
            for row in fresh_prestop["models"]
        ],
        "global_claim_lock_held_across_fresh_check_delete_and_postcheck": True,
        "post_delete_jobs_and_pods_absent": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_delete_receipt(
    value: Mapping[str, Any],
    authorization: Mapping[str, Any],
    prestop: Mapping[str, Any],
    fresh_prestop: Mapping[str, Any],
    poststop: Mapping[str, Any],
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    expected = build_delete_receipt(
        authorization,
        prestop,
        fresh_prestop,
        poststop,
        static,
        started_at_utc=str(value.get("started_at_utc")),
        completed_at_utc=str(value.get("completed_at_utc")),
    )
    if value != expected:
        raise ValueError("Generation-7 foreground-delete receipt drifted")


def build_tombstone(
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    kubernetes: Callable[[str, str], dict[str, Any]],
    path_exists: Callable[[str], bool],
    sessions: Callable[[str], list[dict[str, Any]]],
    observed_at_utc: str,
    controlled_stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if controlled_stop is not None:
        if set(controlled_stop) != {
            "prestop",
            "fresh_prestop",
            "authorization",
            "poststop",
            "deletion",
        }:
            raise ValueError("Generation-7 controlled-stop bundle shape drifted")
        prestop = controlled_stop["prestop"]
        fresh_prestop = controlled_stop["fresh_prestop"]
        authorization = controlled_stop["authorization"]
        poststop = controlled_stop["poststop"]
        deletion = controlled_stop["deletion"]
        validate_delete_receipt(
            deletion,
            authorization,
            prestop,
            fresh_prestop,
            poststop,
            static,
        )
    models = []
    total_rows = 0
    for model in sorted(MODELS):
        spec, plan = static[model]
        row = MODELS[model]
        job_name = generation7.EXPECTED[model]["job_name"]
        job = kubernetes("job", job_name)
        pods = kubernetes("pods", job_name).get("items", [])
        if controlled_stop is None:
            conditions = job.get("status", {}).get("conditions", [])
            terminal_conditions = [
                item
                for item in conditions
                if item.get("type") in {"Complete", "Failed"} and item.get("status") == "True"
            ]
            if (
                job.get("metadata", {}).get("uid") != row["g7_job_uid"]
                or job.get("status", {}).get("active", 0) not in (0, None)
                or job.get("status", {}).get("failed") != 1
                or len(terminal_conditions) != 1
                or terminal_conditions[0].get("type") != "Failed"
                or len(pods) != 1
                or pods[0].get("metadata", {}).get("uid") != row["g7_pod_uid"]
                or pods[0].get("status", {}).get("phase") != "Failed"
            ):
                raise RuntimeError("Generation-7 is not exclusively terminally stopped")
            job_evidence = {"name": job_name, "uid": row["g7_job_uid"], "terminal": "Failed"}
            pod_evidence = {"uid": row["g7_pod_uid"], "phase": "Failed"}
        else:
            if job or pods:
                raise RuntimeError("Generation-7 controlled stop is not terminally absent")
            job_evidence = {
                "name": job_name,
                "uid": row["g7_job_uid"],
                "terminal": "ForegroundDeleted",
            }
            pod_evidence = {"uid": row["g7_pod_uid"], "phase": "Absent"}
        absence = _absence_row(model, spec, plan, path_exists=path_exists, sessions=sessions)
        total_rows += absence["session_rows_examined"]
        models.append(
            {
                "model": model,
                "cell_id": spec["statistical_cell"]["cell_id"],
                "g7_job": job_evidence,
                "g7_pod": pod_evidence,
                **absence,
            }
        )
    body = {
        "schema_version": TOMBSTONE_SCHEMA,
        "status": "G7_TERMINAL_PRECLAIM_PREOUTPUT_PRESESSION",
        "stop_mode": (
            "natural_failed_preserved"
            if controlled_stop is None
            else "uid_preconditioned_foreground_delete"
        ),
        "observed_at_utc": observed_at_utc,
        "controlled_stop": copy.deepcopy(dict(controlled_stop)) if controlled_stop else None,
        "models": models,
        "total_session_rows_examined": total_rows,
        "generation_range_checked": [1, 8],
        "claims_outputs_sessions_absent": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_tombstone(
    value: Mapping[str, Any], static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> None:
    expected_top_keys = {
        "schema_version",
        "status",
        "stop_mode",
        "observed_at_utc",
        "controlled_stop",
        "models",
        "total_session_rows_examined",
        "generation_range_checked",
        "claims_outputs_sessions_absent",
        "prompts_traces_flags_or_scores_included",
        "credentials_included",
        "receipt_sha256",
    }
    if (
        set(value) != expected_top_keys
        or value.get("schema_version") != TOMBSTONE_SCHEMA
        or value.get("status") != "G7_TERMINAL_PRECLAIM_PREOUTPUT_PRESESSION"
        or value.get("generation_range_checked") != [1, 8]
        or value.get("claims_outputs_sessions_absent") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError("Generation-7 preclaim stop tombstone drifted")
    rows = value.get("models")
    if not isinstance(rows, list) or [row.get("model") for row in rows] != sorted(MODELS):
        raise ValueError("Generation-7 tombstone model set drifted")
    total_session_rows = 0
    stop_mode = value.get("stop_mode")
    controlled_stop = value.get("controlled_stop")
    if stop_mode == "natural_failed_preserved":
        if controlled_stop is not None:
            raise ValueError("Generation-7 natural stop unexpectedly has delete evidence")
    elif stop_mode == "uid_preconditioned_foreground_delete":
        if not isinstance(controlled_stop, dict):
            raise ValueError("Generation-7 controlled stop evidence is absent")
        validate_delete_receipt(
            controlled_stop.get("deletion", {}),
            controlled_stop.get("authorization", {}),
            controlled_stop.get("prestop", {}),
            controlled_stop.get("fresh_prestop", {}),
            controlled_stop.get("poststop", {}),
            static,
        )
    else:
        raise ValueError("Generation-7 stop mode drifted")
    for row in rows:
        if set(row) != {
            "model",
            "cell_id",
            "g7_job",
            "g7_pod",
            "checked_claim_paths",
            "checked_output_paths",
            "run_ids_checked",
            "task_key",
            "session_rows_examined",
            "matching_sessions",
        }:
            raise ValueError("Generation-7 tombstone row shape drifted")
        model = row["model"]
        spec, plan = static[model]
        expected = MODELS[model]
        if row.get("cell_id") != spec["statistical_cell"]["cell_id"]:
            raise ValueError("Generation-7 tombstone binding drifted")
        expected_job = {
            "name": generation7.EXPECTED[model]["job_name"],
            "uid": expected["g7_job_uid"],
            "terminal": (
                "Failed" if stop_mode == "natural_failed_preserved" else "ForegroundDeleted"
            ),
        }
        expected_pod = {
            "uid": expected["g7_pod_uid"],
            "phase": "Failed" if stop_mode == "natural_failed_preserved" else "Absent",
        }
        if row.get("g7_job") != expected_job or row.get("g7_pod") != expected_pod:
            raise ValueError("Generation-7 tombstone terminal binding drifted")
        total_session_rows += _validate_absence_row(row, model, spec, plan)
    if value.get("total_session_rows_examined") != total_session_rows:
        raise ValueError("Generation-7 tombstone session total drifted")


def build_release(
    model: str,
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    held: Mapping[str, Any],
    tombstone: Mapping[str, Any],
    package: Mapping[str, Any],
    package_commit: str,
) -> dict[str, Any]:
    validate_tombstone(tombstone, static)
    return _build_release_from_validated(model, static, held, tombstone, package, package_commit)


def _build_release_from_validated(
    model: str,
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    held: Mapping[str, Any],
    tombstone: Mapping[str, Any],
    package: Mapping[str, Any],
    package_commit: str,
) -> dict[str, Any]:
    """Build from values already validated in the current process."""
    if COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("Generation-8 package commit is invalid")
    if set(static) != set(MODELS) or model not in static:
        raise ValueError("Generation-8 release requires both model bindings")
    spec, plan = static[model]
    if held.get("launch_authorized") is not False or package.get("launch_authorized") is not False:
        raise ValueError("Generation-8 held/package authority drifted")
    body = {
        "schema_version": RELEASE_SCHEMA,
        "status": "RELEASED",
        "append_only": True,
        "model": model,
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation8_spec_sha256": spec["generation8_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "package_commit": package_commit,
        "package_aggregate_sha256": package["aggregate_sha256"],
        "held_receipt_sha256": held["receipt_sha256"],
        "g7_preclaim_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
        "g7_scoring_release": {
            "receipt_sha256": spec["predecessor_generation7"]["scoring_release_receipt_sha256"],
            "file_sha256": spec["predecessor_generation7"]["scoring_release_file_sha256"],
        },
        "authorization": {
            "execution_generation": 8,
            "same_statistical_cell": True,
            "hosted_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "create_once": True,
            "bulk_release_authorized": False,
        },
        "launch_authorized": True,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_package_manifest(
    value: Mapping[str, Any], model: str, spec: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if (
        set(value)
        != {
            "schema_version",
            "model",
            "generation8_spec_sha256",
            "rendered_plan_sha256",
            "held_receipt_sha256",
            "objects",
            "aggregate_sha256",
            "release_included",
            "launch_authorized",
        }
        or value.get("schema_version") != "fleet-opencode-autocontinue-generation8-split-package-v1"
        or value.get("model") != model
        or value.get("generation8_spec_sha256") != spec["generation8_spec_sha256"]
        or value.get("rendered_plan_sha256") != plan["plan_sha256"]
        or value.get("held_receipt_sha256") != EXPECTED_HELD_RECEIPT_SHA256
        or value.get("release_included") is not False
        or value.get("launch_authorized") is not False
        or value.get("aggregate_sha256") != sha256(canonical(value.get("objects")))
    ):
        raise ValueError("Generation-8 package manifest drifted")


def validate_release(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if value != expected or value.get("receipt_sha256") != digest(value):
        raise ValueError("Generation-8 scoring release drifted")


def validate_context(
    root: Path,
    model: str,
    package_commit: str,
    *,
    release_path: Path,
    tombstone_path: Path,
    package_manifest_path: Path,
) -> ValidatedContext:
    ledger = ValidationLedger()
    held = ledger.read_once("held", root / HELD_PATH)
    validate_held(held)
    static: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    selected: (
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None
    ) = None
    for candidate in MODELS:
        row = validate_static(root, candidate, ledger, held=held)
        static[candidate] = (row[0], row[1])
        if candidate == model:
            selected = row
    if selected is None:
        raise ValueError("Generation-8 model is unsupported")
    spec, plan, _, g7_release = selected
    tombstone = ledger.read_once("tombstone", tombstone_path)
    validate_tombstone(tombstone, static)
    package = ledger.read_once("package", package_manifest_path)
    validate_package_manifest(package, model, spec, plan)
    release = ledger.read_once("release", release_path)
    expected = _build_release_from_validated(
        model, static, held, tombstone, package, package_commit
    )
    validate_release(release, expected)
    context = ValidatedContext(
        model,
        root,
        package_commit,
        spec,
        plan,
        held,
        g7_release,
        release,
        tombstone,
        package,
        MappingProxyType(static),
        ledger,
    )
    context.assert_unchanged()
    if any(count != 1 for count in ledger.read_counts.values()):
        raise RuntimeError("immutable Generation-8 input validation count drifted")
    return context


def _claim_body(
    context: ValidatedContext, *, claimed_at_utc: str, job_uid: str, pod_uid: str
) -> dict[str, Any]:
    body = {
        "schema_version": CLAIM_SCHEMA,
        "generation8_spec_sha256": context.spec["generation8_spec_sha256"],
        "plan_sha256": context.plan["plan_sha256"],
        "cell_id": context.spec["statistical_cell"]["cell_id"],
        "execution_id": context.spec["execution"]["execution_id"],
        "execution_generation": 8,
        "run_id": context.plan["attempts"][0]["run_id"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "claimed_at_utc": claimed_at_utc,
        "scoring_release_receipt_sha256": context.release["receipt_sha256"],
        "g7_preclaim_stop_tombstone_receipt_sha256": context.tombstone["receipt_sha256"],
        "prior_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def claim_execution(context: ValidatedContext, *, claim_root: Path | None = None) -> dict[str, Any]:
    context.assert_unchanged()
    job_uid, pod_uid = os.environ.get("JOB_UID"), os.environ.get("POD_UID")
    if not legacy._is_uuid(job_uid) or not legacy._is_uuid(pod_uid):
        raise RuntimeError("Generation-8 claim requires downward API UIDs")
    claim_dir = claim_root or Path(CLAIM_ROOT)
    root_fd = legacy._open_directory_nofollow(claim_dir)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for path in generation_claim_paths(context.spec):
                name = Path(path).name
                try:
                    os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise RuntimeError("Generation-8 statistical cell already has a generation claim")
            receipt = _claim_body(
                context,
                claimed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                job_uid=str(job_uid),
                pod_uid=str(pod_uid),
            )
            name = Path(generation_claim_paths(context.spec)[-1]).name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            with os.fdopen(fd, "wb") as stored:
                stored.write(canonical(receipt) + b"\n")
                stored.flush()
                os.fsync(stored.fileno())
            os.fsync(root_fd)
            return receipt
    finally:
        os.close(root_fd)


def terminal_receipt(
    context: ValidatedContext, claim: Mapping[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    context.assert_unchanged()
    expected_claim = _claim_body(
        context,
        claimed_at_utc=str(claim.get("claimed_at_utc")),
        job_uid=str(claim.get("job_uid")),
        pod_uid=str(claim.get("pod_uid")),
    )
    if claim != expected_claim:
        raise ValueError("Generation-8 claim drifted before terminal write")
    body = {
        "schema_version": TERMINAL_SCHEMA,
        "generation8_spec_sha256": context.spec["generation8_spec_sha256"],
        "plan_sha256": context.plan["plan_sha256"],
        "cell_id": context.spec["statistical_cell"]["cell_id"],
        "execution_id": context.spec["execution"]["execution_id"],
        "execution_generation": 8,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scoring_release_receipt_sha256": context.release["receipt_sha256"],
        "result": generation2._validated_result(result, copy.deepcopy(dict(context.plan))),
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Kubernetes redirect forbidden")


def _kubernetes(kind: str, name: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host or kind not in {"job", "pods"}:
        raise RuntimeError("in-cluster Kubernetes reader unavailable")
    quoted = urllib.parse.quote(name, safe="")
    path = (
        f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{quoted}"
        if kind == "job"
        else f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector="
        + urllib.parse.quote(f"job-name={name}", safe="")
    )
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    request = urllib.request.Request(
        f"https://{host}:{port}{path}",
        method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    context = ssl.create_default_context(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    opener = urllib.request.build_opener(
        _RejectRedirects(), urllib.request.HTTPSHandler(context=context)
    )
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"items": []} if kind == "pods" else {}
        raise RuntimeError("Kubernetes evidence query failed") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("Kubernetes evidence response too large")
    return _strict_json(raw)


def _delete_kubernetes_job(name: str, uid: str) -> None:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host or not legacy._is_uuid(uid):
        raise RuntimeError("in-cluster UID-preconditioned delete is unavailable")
    body = canonical(
        {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {"uid": uid},
        }
    )
    quoted = urllib.parse.quote(name, safe="")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    request = urllib.request.Request(
        f"https://{host}:{port}/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{quoted}",
        data=body,
        method="DELETE",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    context = ssl.create_default_context(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    opener = urllib.request.build_opener(
        _RejectRedirects(), urllib.request.HTTPSHandler(context=context)
    )
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("UID-preconditioned foreground delete failed") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("Kubernetes delete response too large")
    value = _strict_json(raw)
    response_uid = value.get("metadata", {}).get("uid")
    status_uid = value.get("details", {}).get("uid")
    if response_uid != uid and status_uid != uid:
        raise RuntimeError("Kubernetes delete response UID drifted")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp is not canonical UTC")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    if parsed.tzinfo != UTC:
        raise ValueError("timestamp is not UTC")
    return parsed


def execute_controlled_stop(
    static: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    prestop: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol_dir: Path,
    *,
    kubernetes: Callable[[str, str], dict[str, Any]] = _kubernetes,
    delete_job: Callable[[str, str], None] = _delete_kubernetes_job,
    path_exists: Callable[[str], bool] | None = None,
    sessions: Callable[[str], list[dict[str, Any]]],
    claim_root: Path = Path(CLAIM_ROOT),
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    wait: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Stop both exact G7 Jobs under the shared claim lock and freeze evidence."""
    validate_prestop(prestop, static)
    validate_delete_authorization(authorization, prestop, static)
    current = now()
    age = (current - _parse_utc(prestop["observed_at_utc"])).total_seconds()
    if age < 0 or age > 900:
        raise RuntimeError("Generation-7 controlled pre-stop snapshot is stale")
    if protocol_dir.exists() or protocol_dir.is_symlink():
        raise RuntimeError("Generation-7 stop protocol output already exists")
    protocol_dir.mkdir(mode=0o700)
    self_hosted.write_json_once(protocol_dir / "PRESTOP.json", copy.deepcopy(dict(prestop)))
    self_hosted.write_json_once(
        protocol_dir / "DELETE-AUTHORIZATION.json", copy.deepcopy(dict(authorization))
    )
    exists = path_exists or (lambda path: Path(path).exists() or Path(path).is_symlink())
    root_fd = legacy._open_directory_nofollow(claim_root)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            fresh = build_prestop(
                static,
                kubernetes=kubernetes,
                path_exists=exists,
                sessions=sessions,
                observed_at_utc=now().isoformat().replace("+00:00", "Z"),
            )
            original_rows = {row["model"]: row for row in prestop["models"]}
            fresh_rows = {row["model"]: row for row in fresh["models"]}
            for model in MODELS:
                if (
                    original_rows[model]["g7_job"] != fresh_rows[model]["g7_job"]
                    or original_rows[model]["g7_pod"] != fresh_rows[model]["g7_pod"]
                ):
                    raise RuntimeError("Generation-7 UID/resourceVersion changed before delete")
            self_hosted.write_json_once(protocol_dir / "FRESH-PRESTOP.json", fresh)
            started = now().isoformat().replace("+00:00", "Z")
            for row in fresh["models"]:
                delete_job(row["g7_job"]["name"], row["g7_job"]["uid"])
            for _ in range(180):
                if all(
                    not kubernetes("job", generation7.EXPECTED[model]["job_name"])
                    and not kubernetes("pods", generation7.EXPECTED[model]["job_name"]).get(
                        "items", []
                    )
                    for model in MODELS
                ):
                    break
                wait(1)
            else:
                raise RuntimeError("Generation-7 foreground deletion did not reach absence")
            poststop = build_poststop(
                static,
                fresh,
                kubernetes=kubernetes,
                path_exists=exists,
                sessions=sessions,
                observed_at_utc=now().isoformat().replace("+00:00", "Z"),
            )
            deletion = build_delete_receipt(
                authorization,
                prestop,
                fresh,
                poststop,
                static,
                started_at_utc=started,
                completed_at_utc=now().isoformat().replace("+00:00", "Z"),
            )
            self_hosted.write_json_once(protocol_dir / "POSTSTOP.json", poststop)
            self_hosted.write_json_once(protocol_dir / "DELETION.json", deletion)
            return {
                "prestop": copy.deepcopy(dict(prestop)),
                "fresh_prestop": fresh,
                "authorization": copy.deepcopy(dict(authorization)),
                "poststop": poststop,
                "deletion": deletion,
            }
    finally:
        os.close(root_fd)


def load_controlled_stop(protocol_dir: Path) -> dict[str, Any]:
    names = {
        "prestop": "PRESTOP.json",
        "fresh_prestop": "FRESH-PRESTOP.json",
        "authorization": "DELETE-AUTHORIZATION.json",
        "poststop": "POSTSTOP.json",
        "deletion": "DELETION.json",
    }
    return {key: _strict_json((protocol_dir / name).read_bytes()) for key, name in names.items()}


def observe_live_tombstone(context: ValidatedContext, key: str) -> dict[str, Any]:
    from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

    controlled_stop = None
    if context.tombstone.get("stop_mode") == "uid_preconditioned_foreground_delete":
        controlled_stop = context.tombstone.get("controlled_stop")
    return build_tombstone(
        context.static,
        kubernetes=_kubernetes,
        path_exists=lambda path: Path(path).exists() or Path(path).is_symlink(),
        sessions=lambda task_key: reconciliation._default_sessions(  # noqa: SLF001
            task_key, key
        ),
        observed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        controlled_stop=controlled_stop,
    )


def run(
    context: ValidatedContext,
    launch_route: Mapping[str, Any],
    out: Path,
    proxy: Path,
) -> dict[str, Any]:
    """Execute only after one context validation and a fresh live stop proof."""
    context.assert_unchanged()
    jobs = tuple(generation7.EXPECTED[model]["job_name"] for model in MODELS)
    hosted_runtime.validate_live_route(
        dict(launch_route),
        caller=hosted_runtime.LAUNCHER_CALLER,
        maximum_age_seconds=None,
        candidate_scored_job_names=jobs,
    )
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = context.plan["execution"]["endpoint_lease"]
    with legacy.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        live_route = hosted_runtime.observe_live_route(
            key,
            caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted_runtime.validate_live_route(
            live_route,
            caller=hosted_runtime.RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted._validate_plan_identity_absence(copy.deepcopy(dict(context.plan)), out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(
            copy.deepcopy(dict(context.plan)),
            out,
            copy.deepcopy(context.plan["tasks"][0]),
            key,
        )
        # This fresh observation occurs immediately before the global O_EXCL
        # claim.  It cannot write an output and checks both cells, G1-G8.
        fresh = observe_live_tombstone(context, key)
        validate_tombstone(fresh, context.static)
        claim = claim_execution(context)
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", copy.deepcopy(dict(context.plan)))
        self_hosted.write_json_once(
            out / "SCORING-RELEASE.json", copy.deepcopy(dict(context.release))
        )
        self_hosted.write_json_once(
            out / "G7-PRECLAIM-STOP.json", copy.deepcopy(dict(context.tombstone))
        )
        result = legacy._run_cell(copy.deepcopy(dict(context.plan)), out, proxy, key)
        terminal = terminal_receipt(context, claim, result)
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def held_preview(root: Path) -> dict[str, Any]:
    models = {}
    ledger = ValidationLedger()
    held = ledger.read_once("held", root / HELD_PATH)
    validate_held(held)
    for model in MODELS:
        spec, plan, _, _ = validate_static(root, model, ledger, held=held)
        models[model] = {
            "cell_id": spec["statistical_cell"]["cell_id"],
            "execution_id": spec["execution"]["execution_id"],
            "plan_sha256": plan["plan_sha256"],
            "job_name": spec["identities"]["job_name"],
        }
    return {
        "status": "HELD",
        "launch_authorized": False,
        "objects_created": False,
        "models": models,
        "required_g7_state": "terminal_preclaim_preoutput_presession",
        "validation_scope": "process_local_exactly_once",
    }


def _all_static(
    root: Path,
) -> tuple[
    ValidationLedger, Mapping[str, Any], dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]
]:
    ledger = ValidationLedger()
    held = ledger.read_once("held", root / HELD_PATH)
    validate_held(held)
    static = {}
    for model in MODELS:
        spec, plan, _, _ = validate_static(root, model, ledger, held=held)
        static[model] = (spec, plan)
    return ledger, held, static


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preview",
            "validate-static",
            "observe-prestop",
            "render-delete-authorization",
            "execute-controlled-stop",
            "observe-tombstone",
            "render-release",
            "run",
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--package-commit")
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--tombstone", type=Path)
    parser.add_argument("--prestop", type=Path)
    parser.add_argument("--delete-authorization", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.command == "preview":
        print(json.dumps(held_preview(root), sort_keys=True))
        return 0
    ledger, held, static = _all_static(root)
    if args.command == "validate-static":
        if any(count != 1 for count in ledger.read_counts.values()):
            raise RuntimeError("static validation was not exactly once")
        return 0
    if args.command == "observe-prestop":
        if not args.output:
            parser.error("observe-prestop requires an unused output")
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

        value = build_prestop(
            static,
            kubernetes=_kubernetes,
            path_exists=lambda path: Path(path).exists() or Path(path).is_symlink(),
            sessions=lambda task_key: reconciliation._default_sessions(  # noqa: SLF001
                task_key, key
            ),
            observed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "render-delete-authorization":
        if not args.prestop or not args.output:
            parser.error("render-delete-authorization requires prestop and unused output")
        prestop = _strict_json(args.prestop.read_bytes())
        self_hosted.write_json_once(args.output, build_delete_authorization(prestop, static))
        return 0
    if args.command == "execute-controlled-stop":
        if not args.prestop or not args.delete_authorization or not args.protocol_dir:
            parser.error(
                "execute-controlled-stop requires prestop, delete authorization, and protocol dir"
            )
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

        prestop = _strict_json(args.prestop.read_bytes())
        authorization = _strict_json(args.delete_authorization.read_bytes())
        execute_controlled_stop(
            static,
            prestop,
            authorization,
            args.protocol_dir,
            sessions=lambda task_key: reconciliation._default_sessions(  # noqa: SLF001
                task_key, key
            ),
        )
        return 0
    if args.command == "observe-tombstone":
        if not args.output:
            parser.error("observe-tombstone requires an unused output")
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

        controlled_stop = load_controlled_stop(args.protocol_dir) if args.protocol_dir else None
        value = build_tombstone(
            static,
            kubernetes=_kubernetes,
            path_exists=lambda path: Path(path).exists() or Path(path).is_symlink(),
            sessions=lambda task_key: reconciliation._default_sessions(  # noqa: SLF001
                task_key, key
            ),
            observed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            controlled_stop=controlled_stop,
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    required = (
        args.model,
        args.package_commit,
        args.package_manifest,
        args.tombstone,
        args.release,
    )
    if args.command == "render-release":
        if (
            not args.model
            or not args.package_commit
            or not args.package_manifest
            or not args.tombstone
            or not args.output
        ):
            parser.error(
                "render-release requires model, package commit/manifest, tombstone, and output"
            )
        tombstone = _strict_json(args.tombstone.read_bytes())
        validate_tombstone(tombstone, static)
        package = _strict_json(args.package_manifest.read_bytes())
        spec, plan = static[args.model]
        validate_package_manifest(package, args.model, spec, plan)
        from evals.fleet import immutable_submission_snapshot as snapshot

        snapshot.assert_stable(root, args.package_commit)
        if args.output.resolve() != (root / RELEASE_PATHS[args.model]).resolve():
            parser.error("render-release output must be the model's exact release path")
        self_hosted.write_json_once(
            args.output,
            _build_release_from_validated(
                args.model, static, held, tombstone, package, args.package_commit
            ),
        )
        return 0
    if (
        any(item is None for item in required)
        or not args.launch_route
        or not args.out_dir
        or not args.proxy
    ):
        parser.error(
            "run requires model, package commit/manifest, tombstone, release, route, out, and proxy"
        )
    context = validate_context(
        root,
        args.model,
        args.package_commit,
        release_path=args.release,
        tombstone_path=args.tombstone,
        package_manifest_path=args.package_manifest,
    )
    run(context, _strict_json(args.launch_route.read_bytes()), args.out_dir, args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
