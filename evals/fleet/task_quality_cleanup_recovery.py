"""Seal exact absence evidence for an interrupted task-quality cleanup.

This recovery issuer is deliberately narrower than the qualification controller.
It never provisions, retries, or deletes an environment.  It can resolve a cell
only when the durable create-request claim is absent and two authoritative list
reads for the cell's exact deterministic run ID are both empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import task_quality_qualification as qualification

RECOVERY_EVIDENCE_SCHEMA = "cyber_task_quality_independent_absence_evidence_v1"
RECOVERY_AGGREGATE_SCHEMA = "cyber_task_quality_empty_run_recovery_aggregate_v1"
RECOVERY_AGGREGATE_FILE = "EMPTY_RUN_RECOVERY_AGGREGATE.json"
RECOVERY_EVIDENCE_FILE = "INDEPENDENT_ABSENCE_EVIDENCE.json"
RECOVERY_RESOLUTION = "independent_claim_absent_and_exact_run_empty"
SHA1 = re.compile(r"[0-9a-f]{40}")
CANONICAL_REMOTE = "https://github.com/fleet-ai/cyber-post-train.git"


class RecoveryError(RuntimeError):
    """The historical operation cannot be resolved by the narrow absence proof."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fresh_canonical_main(root: Path) -> str:
    if _git(root, "remote", "get-url", "origin") != CANONICAL_REMOTE:
        raise RecoveryError("origin is not the canonical cyber-post-train repository")
    subprocess.run(
        ["git", "-C", str(root), "fetch", "--quiet", "--no-tags", "origin", "main"],
        check=True,
        capture_output=True,
    )
    return _git(root, "rev-parse", "FETCH_HEAD")


def recovery_source_provenance() -> dict[str, Any]:
    """Require this issuer to be the exact clean, freshly fetched main commit."""
    root = Path(__file__).resolve().parents[2]
    fetched_main = _fresh_canonical_main(root)
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RecoveryError("recovery source worktree must be clean")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != fetched_main:
        raise RecoveryError("recovery source must equal freshly fetched origin/main")
    return {
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "origin_main_commit": fetched_main,
        "canonical_remote": CANONICAL_REMOTE,
        "module_path": "evals/fleet/task_quality_cleanup_recovery.py",
        "module_file_sha256": qualification.file_digest(Path(__file__).resolve()),
    }


def verify_historical_source(source: object, *, current_commit: str) -> dict[str, Any]:
    """Prove the frozen controller bytes exist in current main's ancestry."""
    if not isinstance(source, dict) or source.get("merged_to_origin_main") is not True:
        raise RecoveryError("qualification plan source was not merged")
    commit = source.get("git_commit")
    tree = source.get("git_tree")
    controller_path = source.get("controller_path")
    controller_sha256 = source.get("controller_file_sha256")
    if (
        not isinstance(commit, str)
        or SHA1.fullmatch(commit) is None
        or not isinstance(tree, str)
        or SHA1.fullmatch(tree) is None
        or controller_path != "evals/fleet/task_quality_qualification.py"
        or not isinstance(controller_sha256, str)
    ):
        raise RecoveryError("qualification plan source identity is invalid")
    root = Path(__file__).resolve().parents[2]
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, current_commit],
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise RecoveryError("qualification source is not in recovery source ancestry")
    if _git(root, "rev-parse", f"{commit}^{{tree}}") != tree:
        raise RecoveryError("qualification source tree differs from the frozen plan")
    controller = _git_bytes(root, "show", f"{commit}:{controller_path}")
    if _sha256_bytes(controller) != controller_sha256:
        raise RecoveryError("historical qualification controller bytes differ")
    return {
        "git_commit": commit,
        "git_tree": tree,
        "controller_path": controller_path,
        "controller_file_sha256": controller_sha256,
    }


def _read_sealed(path: Path, schema: str, label: str) -> dict[str, Any]:
    value = qualification._read(path, label)  # noqa: SLF001
    qualification._sealed(value, schema, label)  # noqa: SLF001
    return value


def _validate_operation_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise RecoveryError("qualification operation root must be a real directory")
    plan = _read_sealed(root / "PLAN.json", qualification.PLAN_SCHEMA, "qualification plan")
    run_intent = _read_sealed(
        root / "RUN_INTENT.json", qualification.RUN_INTENT_SCHEMA, "run intent"
    )
    expected_intent = {
        "plan_sha256": plan["sha256"],
        "wave_id": plan["wave_id"],
        "task_versions": len(plan["tasks"]),
        "concurrency": plan["execution"]["concurrency"],
        "one_shot": True,
    }
    if any(run_intent.get(key) != value for key, value in expected_intent.items()):
        raise RecoveryError("run intent differs from the exact qualification plan")
    aggregate = _read_sealed(
        root / "AGGREGATE_RECEIPT.json",
        qualification.AGGREGATE_SCHEMA,
        "qualification aggregate",
    )
    if (
        aggregate.get("plan_sha256") != plan["sha256"]
        or aggregate.get("planned_task_versions") != len(plan["tasks"])
        or aggregate.get("model_calls") != 0
    ):
        raise RecoveryError("qualification aggregate differs from the exact plan")
    return plan, run_intent


def _cell_inputs(
    root: Path, plan: dict[str, Any], index: int, binding: dict[str, Any]
) -> tuple[Path, dict[str, Any], str, str]:
    directory = root / "cells" / f"cell-{index:03d}"
    if directory.is_symlink() or not directory.is_dir():
        raise RecoveryError("qualification cell directory is missing")
    cell_intent = _read_sealed(
        directory / "CELL_INTENT.json",
        "cyber_task_quality_qualification_cell_intent_v1",
        "cell intent",
    )
    terminal = _read_sealed(
        directory / "CELL_TERMINAL.json",
        qualification.CELL_TERMINAL_SCHEMA,
        "cell terminal",
    )
    config = qualification._config(binding, plan["wave_id"])  # noqa: SLF001
    request_id = self_hosted.provisioning_request_id(config)
    provision_intent = _read_sealed(
        directory / "PROVISION_INTENT.json",
        "cyber_task_quality_provision_intent_v1",
        "provision intent",
    )
    if (
        cell_intent.get("binding_sha256") != binding["binding_sha256"]
        or cell_intent.get("wave_id") != plan["wave_id"]
        or cell_intent.get("run_id") != config["run_id"]
        or terminal.get("binding_sha256") != binding["binding_sha256"]
        or terminal.get("wave_id") != plan["wave_id"]
        or provision_intent.get("run_id") != config["run_id"]
        or provision_intent.get("task_version_id") != binding["task_version_id"]
        or provision_intent.get("request_id") != request_id
    ):
        raise RecoveryError("cell evidence differs from its frozen binding")
    if any(
        (directory / artifact).exists()
        for artifact in (
            "PROVISION_RESPONSE.json",
            "INSTANCE_BINDING.json",
            "PROVISION_RECEIPT.json",
        )
    ):
        raise RecoveryError("cells with provision evidence require the normal cleanup path")
    return directory, config, request_id, terminal["sha256"]


def _exact_absence_read(client: httpx.Client, *, request_id: str, run_id: str) -> dict[str, Any]:
    claim = client.get(f"{self_hosted.ORCHESTRATOR}/v1/env/instances/create-requests/{request_id}")
    if claim.status_code != 404:
        raise RecoveryError("durable create-request claim is not absent")
    instances = self_hosted._request(  # noqa: SLF001
        client,
        "GET",
        "/v1/env/instances",
        params={"run_id": run_id, "limit": 2, "offset": 0},
    )
    if not isinstance(instances, list) or instances:
        raise RecoveryError("exact run-id instance listing is not empty")
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "claim_http_status": 404,
        "exact_run_instance_count": 0,
    }


def _validated_existing_evidence(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    observed = _read_sealed(path, RECOVERY_EVIDENCE_SCHEMA, RECOVERY_EVIDENCE_FILE)
    immutable_keys = set(expected) - {"observations", "sha256"}
    if any(observed.get(key) != expected.get(key) for key in immutable_keys):
        raise RecoveryError("existing independent absence evidence differs")
    observations = observed.get("observations")
    if not isinstance(observations, list) or len(observations) != 2:
        raise RecoveryError("existing independent absence observations are invalid")
    for observation in observations:
        if (
            not isinstance(observation, dict)
            or observation.get("claim_http_status") != 404
            or observation.get("exact_run_instance_count") != 0
        ):
            raise RecoveryError("existing independent absence observations differ")
        try:
            timestamp = datetime.fromisoformat(observation.get("observed_at_utc", ""))
        except (TypeError, ValueError) as error:
            raise RecoveryError("absence observation timestamp is invalid") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise RecoveryError("absence observation timestamp is not UTC")
    return observed


def _write_cell_resolution(
    directory: Path,
    *,
    plan: dict[str, Any],
    binding: dict[str, Any],
    request_id: str,
    run_id: str,
    terminal_sha256: str,
    historical_source: dict[str, Any],
    recovery_source: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_body = {
        "schema": RECOVERY_EVIDENCE_SCHEMA,
        "plan_sha256": plan["sha256"],
        "binding_sha256": binding["binding_sha256"],
        "terminal_sha256": terminal_sha256,
        "request_id": request_id,
        "run_id": run_id,
        "fleet_team_id": qualification.EXPECTED_TEAM_ID,
        "historical_source": historical_source,
        "recovery_source": recovery_source,
        "observations": observations,
        "external_mutations": 0,
        "qualification_replayed": False,
        "private_content_read": False,
    }
    evidence = qualification.sealed(evidence_body)
    evidence_path = directory / RECOVERY_EVIDENCE_FILE
    existing_evidence = _validated_existing_evidence(evidence_path, evidence)
    if existing_evidence is None:
        qualification._write_once_atomic(evidence_path, evidence)  # noqa: SLF001
    else:
        evidence = existing_evidence

    resolution = qualification.sealed(
        {
            "schema": qualification.CLEANUP_RESOLUTION_SCHEMA,
            "binding_sha256": binding["binding_sha256"],
            "request_id": request_id,
            "instance_id": None,
            "resolution": RECOVERY_RESOLUTION,
            "instance_live_after": False,
            "resumable_exact_target_only": True,
            "independent_absence_evidence_sha256": evidence["sha256"],
        }
    )
    resolution_path = directory / "CLEANUP_RESOLUTION.json"
    if not resolution_path.exists():
        qualification._write_once_atomic(resolution_path, resolution)  # noqa: SLF001
    else:
        observed = qualification._read(resolution_path, resolution_path.name)  # noqa: SLF001
        if observed != resolution:
            raise RecoveryError("existing cleanup resolution differs")
    return resolution


def recover(root: Path, *, api_key: str, settle_seconds: float = 1.0) -> dict[str, Any]:
    """Resolve only claim-absent, exact-run-empty historical cells."""
    plan, _ = _validate_operation_root(root)
    recovery_source = recovery_source_provenance()
    historical_source = verify_historical_source(
        plan.get("source"), current_commit=recovery_source["git_commit"]
    )
    prepared = [
        _cell_inputs(root, plan, index, binding) for index, binding in enumerate(plan["tasks"])
    ]
    with qualification._client(api_key) as client:  # noqa: SLF001
        qualification._account(client)  # noqa: SLF001
        qualification._assert_create_claim_routes_deployed(client)  # noqa: SLF001
        first_pass = [
            _exact_absence_read(client, request_id=request_id, run_id=config["run_id"])
            for _directory, config, request_id, _terminal_sha256 in prepared
        ]
        time.sleep(settle_seconds)
        second_pass = [
            _exact_absence_read(client, request_id=request_id, run_id=config["run_id"])
            for _directory, config, request_id, _terminal_sha256 in prepared
        ]
    observation_sets = [list(pair) for pair in zip(first_pass, second_pass, strict=True)]

    # Do not write any resolution until every selected task has passed both
    # absence observations.  A mismatch in a later cell therefore cannot leave
    # a partially asserted wave.
    resolution_digests: list[str] = []
    for binding, (directory, config, request_id, terminal_sha256), observations in zip(
        plan["tasks"], prepared, observation_sets, strict=True
    ):
        resolution = _write_cell_resolution(
            directory,
            plan=plan,
            binding=binding,
            request_id=request_id,
            run_id=config["run_id"],
            terminal_sha256=terminal_sha256,
            historical_source=historical_source,
            recovery_source=recovery_source,
            observations=observations,
        )
        resolution_digests.append(resolution["sha256"])
    aggregate = qualification.sealed(
        {
            "schema": RECOVERY_AGGREGATE_SCHEMA,
            "plan_sha256": plan["sha256"],
            "planned_task_versions": len(plan["tasks"]),
            "resolution_receipts": len(resolution_digests),
            "resolution_receipts_sha256": qualification.digest(sorted(resolution_digests)),
            "claim_reads_per_task_version": 2,
            "exact_run_list_reads_per_task_version": 2,
            "all_claims_absent": True,
            "all_exact_run_instance_lists_empty": True,
            "external_mutations": 0,
            "qualification_replayed": False,
            "logs_prompts_traces_scores_or_credentials_read": False,
            "recovery_source": recovery_source,
        }
    )
    path = root / RECOVERY_AGGREGATE_FILE
    if path.is_file():
        observed = qualification._read(path, path.name)  # noqa: SLF001
        if observed != aggregate:
            raise RecoveryError("existing recovery aggregate differs")
        return observed
    qualification._write_once(path, aggregate)  # noqa: SLF001
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = recover(args.root, api_key=os.environ.get("FLEET_API_KEY", ""))
    public = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "recovery_source",
        }
    }
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
