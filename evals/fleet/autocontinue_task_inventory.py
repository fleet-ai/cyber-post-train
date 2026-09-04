"""Prepare and observe the immutable task bindings for the held campaign.

The live observer performs authenticated GETs only. Task payload content is
hashed in memory and is never written to its receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

FLEET_ORIGIN = "https://orchestrator.fleetai.com"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
CAMPAIGN_PATH = Path("evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json")
MAPPING_PATH = Path("evals/fleet/configs/q38-glm53-primary-scientific-mapping-v2.json")
BASE_SELECTION_PATH = Path("evals/fleet/configs/opencode-easiest-train100-selection-v2.json")
R114_HYDRATION_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-b-r114-hydration-v1.json"
)
R114_PLAN_PATH = Path(
    "evals/fleet/configs/glm53-opencode-dedicated-b-v5-r114-replacement-pass4-v1.json"
)
R112_R113_HYDRATION_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-a-r112-r113-hydration-v1.json"
)
R114_PLAN_SHA256 = "sha256:12453fb47b7c1350b2435ef944d9d14979fb5c9366f4a4f6b1a749446f778505"
R114_HYDRATION_SHA256 = "sha256:47ece81ef595625048ae08f0960b4e6aca425b45e9fa7cd8c6e2fe54e0fa93aa"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
EXPECTED_SCHEMA = "fleet-opencode-autocontinue-task-inventory-expected-v1"
RECEIPT_SCHEMA = "fleet-opencode-autocontinue-task-inventory-terminal-v1"


class GateError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GateError("duplicate_json_key")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GateError("expected_file_absent_or_unsafe")
    value = json.loads(path.read_text(), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise GateError("expected_file_invalid_shape")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def _receipt_row_for_rank(receipt: dict[str, Any], rank: int) -> dict[str, Any] | None:
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("replacement_rank") == rank and value.get("task_version_id"):
                found.append(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(receipt)
    unique = {(row.get("task_key"), row.get("task_version_id")): row for row in found}
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _validate_r114(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    hydration = load_json(root / R114_HYDRATION_PATH)
    plan = load_json(root / R114_PLAN_PATH)
    if hydration.get("receipt_sha256") != R114_HYDRATION_SHA256 or hydration.get(
        "receipt_sha256"
    ) != digest_without(hydration, "receipt_sha256"):
        raise GateError("r114_hydration_digest_mismatch")
    if plan.get("plan_sha256") != R114_PLAN_SHA256 or plan.get("plan_sha256") != digest_without(
        plan, "plan_sha256"
    ):
        raise GateError("r114_plan_digest_mismatch")
    if (
        plan.get("task_count") != 1
        or plan.get("pass_k") != 4
        or plan.get("new_session_count") != 4
        or plan.get("execution", {}).get("launch_authorized") is not False
        or plan.get("execution", {}).get("required_priority_class") != "fleet-train-high"
        or plan.get("source", {}).get("replacement_hydration_receipt_sha256")
        != R114_HYDRATION_SHA256
        or [row.get("source_rank") for row in plan.get("tasks") or []] != [114]
        or {(row.get("source_rank"), row.get("attempt")) for row in plan.get("attempts") or []}
        != {(114, attempt) for attempt in range(1, 5)}
    ):
        raise GateError("r114_execution_binding_invalid")
    hydrated = (hydration.get("tasks") or [None])[0]
    task = (plan.get("tasks") or [None])[0]
    if (
        not isinstance(hydrated, dict)
        or not isinstance(task, dict)
        or {
            "task": hydrated.get("task"),
            "environment": hydrated.get("environment"),
            "verifier": hydrated.get("verifier"),
        }
        != {
            "task": task.get("task"),
            "environment": task.get("environment"),
            "verifier": task.get("verifier"),
        }
    ):
        raise GateError("r114_hydration_plan_binding_mismatch")
    return hydration, plan


def prepare_expected(root: Path) -> dict[str, Any]:
    # Import locally so the cluster observer has no repository dependency.
    from evals.fleet import autocontinue_campaign, campaign_supervisor

    campaign = load_json(root / CAMPAIGN_PATH)
    campaign_summary = autocontinue_campaign.validate_campaign(campaign, root=root)
    mapping = load_json(root / MAPPING_PATH)
    mapping_summary = campaign_supervisor.validate_scientific_mapping(mapping, root=root)
    hydration114, plan114 = _validate_r114(root)
    hydration_by_authority = {
        "sha256:1ebba23918ef3357f6430261841eca2163b632a1d4d67430ae2465e023e719ed": load_json(
            root / R112_R113_HYDRATION_PATH
        ),
        "sha256:3faf062cd63315708c5468ab396fb4a7591c7266c297bb946e0ada904409c2c0": hydration114,
    }
    base = load_json(root / BASE_SELECTION_PATH)
    if (
        base.get("selection_sha256")
        != "sha256:38bd544c74f4e45cb67b271849d657f49e356ccf41f97acca9fb6cd7eb7f56b8"
    ):
        raise GateError("base_selection_digest_mismatch")
    base_by_rank = {int(row["rank"]): row for row in base.get("tasks") or []}
    replacement_map = {
        (row["model"], int(row["replacement_source_rank"])): row
        for row in mapping.get("replacement_mappings") or []
    }
    rows: list[dict[str, Any]] = []
    authority_digests: set[str] = set()
    for component in mapping.get("components") or []:
        model = component["model"]
        for source in component.get("scientific_task_sources") or []:
            if source.get("kind") == "plan":
                plan = load_json(root / Path(source["repo_plan_path"]))
                if plan.get("plan_sha256") != source.get("plan_sha256") or plan.get(
                    "plan_sha256"
                ) != digest_without(plan, "plan_sha256"):
                    raise GateError("scientific_source_plan_digest_mismatch")
                indexed = {int(row["source_rank"]): row for row in plan.get("tasks") or []}
                selected = [indexed[int(rank)] for rank in source.get("source_ranks") or []]
            else:
                hydration = hydration_by_authority.get(source.get("authority_receipt_sha256"))
                if hydration is None or hydration.get("receipt_sha256") != digest_without(
                    hydration, "receipt_sha256"
                ):
                    raise GateError("inline_hydration_authority_invalid")
                indexed = {
                    int(row["replacement_rank"]): {"source_rank": row["replacement_rank"], **row}
                    for row in hydration.get("tasks") or []
                }
                selected = [indexed[int(row["source_rank"])] for row in source.get("tasks") or []]
            for task_row in selected:
                rank = int(task_row["source_rank"])
                task = task_row["task"]
                environment = task_row["environment"]
                verifier = task_row["verifier"]
                authority = base_by_rank.get(rank)
                authority_digest = base["selection_sha256"]
                authority_path = str(BASE_SELECTION_PATH)
                if authority is None or (
                    authority.get("task_key"),
                    authority.get("task_version_id"),
                ) != (task.get("key"), task.get("version_id")):
                    replacement = replacement_map.get((model, rank))
                    if replacement is None:
                        raise GateError("task_split_authority_absent")
                    authority_digest = replacement["selection_receipt_sha256"]
                    authority_path = str(
                        campaign_supervisor.REPLACEMENT_SELECTION_EVIDENCE[authority_digest]
                    )
                    receipt = load_json(root / Path(authority_path))
                    if receipt.get("receipt_sha256") != authority_digest or receipt.get(
                        "receipt_sha256"
                    ) != digest_without(receipt, "receipt_sha256"):
                        raise GateError("task_split_authority_digest_mismatch")
                    authority = _receipt_row_for_rank(receipt, rank)
                if authority is None or any(
                    (
                        authority.get("task_key") != task.get("key"),
                        authority.get("task_version_id") != task.get("version_id"),
                        authority.get("split") != "train",
                        authority.get("environment_version_id") != environment.get("version_id"),
                        authority.get("env_key") != environment.get("id"),
                        authority.get("env_version") != environment.get("version"),
                        authority.get("data_key") != environment.get("data_id"),
                        authority.get("data_version") != environment.get("data_version"),
                    )
                ):
                    raise GateError("frozen_task_authority_mismatch")
                authority_digests.add(authority_digest)
                rows.append(
                    {
                        "model": model,
                        "component_id": component["id"],
                        "source_rank": rank,
                        "task_key": task["key"],
                        "task_version_id": task["version_id"],
                        "split": "train",
                        "split_and_environment_uuid_authority": {
                            "path": authority_path,
                            "digest": authority_digest,
                        },
                        "task_content_sha256": {
                            "prompt": task["prompt_sha256"],
                            "env_variables": task["env_variables_sha256"],
                            "output_json_schema": task["output_json_schema_sha256"],
                        },
                        "cyber_contract": task["cyber_contract"],
                        "environment": environment,
                        "verifier": verifier,
                    }
                )
    rows.sort(key=lambda row: (row["model"], row["component_id"], row["source_rank"]))
    if (
        len(rows) != 150
        or len({(row["model"], row["task_version_id"]) for row in rows}) != 150
        or sum(row["model"] == "qwen3.8-27b" for row in rows) != 50
        or sum(row["model"] == "glm-5.3" for row in rows) != 100
        or mapping_summary.get("cells") != 600
        or campaign_summary.get("task_identity_sha256") != campaign.get("task_identity_sha256")
    ):
        raise GateError("expected_inventory_denominator_mismatch")
    expected = {
        "schema_version": EXPECTED_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "scientific_mapping_sha256": mapping["mapping_sha256"],
        "task_identity_sha256": campaign["task_identity_sha256"],
        "selection_counts": {"qwen3.8-27b": 50, "glm-5.3": 100, "total": 150},
        "cell_counts": {"qwen3.8-27b": 200, "glm-5.3": 400, "total": 600},
        "split_counts": {"train": 150, "dev": 0, "test": 0},
        "environment_version_id_fresh_api_observable": False,
        "r114_execution_binding": {
            "plan_path": str(R114_PLAN_PATH),
            "plan_sha256": plan114["plan_sha256"],
            "hydration_path": str(R114_HYDRATION_PATH),
            "hydration_receipt_sha256": hydration114["receipt_sha256"],
            "attempts": 4,
            "launch_authorized": False,
        },
        "authority_digests": sorted(authority_digests),
        "tasks": rows,
        "privacy": {
            "task_content_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    expected["expected_sha256"] = digest_without(expected, "expected_sha256")
    return expected


def fetch_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise GateError("fleet_get_http_status")
            payload = response.read(8 * 1024 * 1024 + 1)
    except GateError:
        raise
    except Exception as exc:
        raise GateError("fleet_get_failed") from exc
    if len(payload) > 8 * 1024 * 1024:
        raise GateError("fleet_get_response_too_large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("fleet_get_invalid_json") from exc
    if not isinstance(value, dict):
        raise GateError("fleet_get_invalid_shape")
    return value


def _live_comparable(task: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata") or {}
    runtime_seed = metadata.get("runtime_seed_manifest") or {}
    verifier = task.get("verifier") or {}
    files = runtime_seed.get("files")
    if not isinstance(files, list) or not files:
        raise GateError("runtime_seed_manifest_incomplete")
    return {
        "task_key": task.get("key"),
        "task_version_id": expected["task_version_id"],
        "task_content_sha256": {
            "prompt": sha256((task.get("prompt") or "").encode()),
            "env_variables": sha256(canonical_json(task.get("env_variables") or {})),
            "output_json_schema": sha256(canonical_json(task.get("output_json_schema"))),
        },
        "cyber_contract": metadata.get("cyber_contract"),
        "environment": {
            "id": task.get("environment_id"),
            "version": task.get("version"),
            "data_id": task.get("data_id"),
            "data_version": task.get("data_version"),
            "runtime_seed_content_sha256": runtime_seed.get("content_sha256"),
            "runtime_seed_file_count_positive": True,
        },
        "verifier": {
            "id": task.get("verifier_id"),
            "version_id": verifier.get("verifier_version_id"),
            "version": verifier.get("version"),
            "sha256": verifier.get("sha256"),
            "function_name": verifier.get("function_name") or "verify",
        },
    }


def _expected_comparable(row: dict[str, Any]) -> dict[str, Any]:
    environment = row["environment"]
    return {
        "task_key": row["task_key"],
        "task_version_id": row["task_version_id"],
        "task_content_sha256": row["task_content_sha256"],
        "cyber_contract": row["cyber_contract"],
        "environment": {
            "id": environment["id"],
            "version": environment["version"],
            "data_id": environment["data_id"],
            "data_version": environment["data_version"],
            "runtime_seed_content_sha256": environment["runtime_seed_content_sha256"],
            "runtime_seed_file_count_positive": True,
        },
        "verifier": row["verifier"],
    }


def observe(
    expected: dict[str, Any], api_key: str, *, job_uid: str, pod_uid: str
) -> dict[str, Any]:
    if expected.get("expected_sha256") != digest_without(expected, "expected_sha256"):
        raise GateError("expected_inventory_digest_mismatch")
    account = fetch_json(f"{FLEET_ORIGIN}/v1/account", api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise GateError("fleet_team_identity_mismatch")
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    observed_rows: list[dict[str, Any]] = []
    for row in expected.get("tasks") or []:
        identity = (row["task_key"], row["task_version_id"])
        if identity not in cache:
            path = (
                f"/v1/tasks/{quote(identity[0], safe='')}?{urlencode({'version_id': identity[1]})}"
            )
            cache[identity] = fetch_json(FLEET_ORIGIN + path, api_key)
        comparable = _live_comparable(cache[identity], row)
        if comparable != _expected_comparable(row):
            raise GateError("live_task_binding_mismatch")
        observed_rows.append(
            {
                "model": row["model"],
                "component_id": row["component_id"],
                "source_rank": row["source_rank"],
                **comparable,
                "split": row["split"],
                "environment_version_id": row["environment"]["version_id"],
                "environment_version_id_authority_digest": row[
                    "split_and_environment_uuid_authority"
                ]["digest"],
            }
        )
    if len(observed_rows) != 150:
        raise GateError("live_task_count_mismatch")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASSED",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign_sha256": expected["campaign_sha256"],
        "scientific_mapping_sha256": expected["scientific_mapping_sha256"],
        "task_identity_sha256": expected["task_identity_sha256"],
        "expected_inventory_sha256": expected["expected_sha256"],
        "expected_bindings_sha256": sha256(canonical_json(expected["tasks"])),
        "observed_bindings_sha256": sha256(canonical_json(observed_rows)),
        "selection_counts": expected["selection_counts"],
        "cell_counts": expected["cell_counts"],
        "split_counts": expected["split_counts"],
        "binding_mismatch_count": 0,
        "environment_version_id_fresh_api_observable": False,
        "environment_version_uuid_authority": "frozen_immutable_selection_and_replacement_receipts",
        "environment_version_uuid_authority_digests": expected["authority_digests"],
        "r114_execution_binding": expected["r114_execution_binding"],
        "fleet_account": {"team_name": "fleet", "team_id": FLEET_TEAM_ID},
        "request_counts": {
            "account_get": 1,
            "task_version_get": len(cache),
            "post_put_patch_delete": 0,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
        },
        "runtime": {
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "secret_name": SECRET_NAME,
            "secret_uid": SECRET_UID,
        },
        "privacy": {
            "task_payloads_persisted": False,
            "task_content_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.prepare:
        args.prepare.write_bytes(canonical_json(prepare_expected(args.root)) + b"\n")
        return 0
    if not args.expected or not args.out_dir:
        parser.error("observer requires --expected and --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        receipt = observe(
            load_json(args.expected),
            os.environ["FLEET_API_KEY"],
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
        )
        write_once(args.out_dir / "TERMINAL.json", receipt)
        print("immutable task inventory passed")
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, GateError) else "unexpected_failure"
        failure = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "FAILED",
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "failure_code": code,
            "runtime": {
                "job_uid": os.environ.get("JOB_UID"),
                "pod_uid": os.environ.get("POD_UID"),
                "secret_name": SECRET_NAME,
                "secret_uid": SECRET_UID,
            },
            "request_policy": {"method": "GET", "model_or_scoring_calls": 0, "session_calls": 0},
            "task_payloads_persisted": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        failure["receipt_sha256"] = digest_without(failure, "receipt_sha256")
        write_once(args.out_dir / "TERMINAL.json", failure)
        print("immutable task inventory failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
