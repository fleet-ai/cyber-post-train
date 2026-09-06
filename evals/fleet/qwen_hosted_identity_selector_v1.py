"""Bounded score-free selector for an identity-complete hosted-Qwen task."""

from __future__ import annotations

import argparse
import contextlib
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
    from evals.fleet import score_blind_session_inventory_v1 as stream
except ModuleNotFoundError:  # projected ConfigMap runtime
    import qwen_hosted_rank17_g22_release_observer_v1 as base  # type: ignore[no-redef]
    import score_blind_session_inventory_v1 as stream  # type: ignore[no-redef]

SCHEMA = "fleet-qwen38-hosted-identity-selector-observation-v1"
BINDING_SCHEMA = "fleet-qwen38-hosted-identity-selector-binding-v1"
PACKAGE_SCHEMA = "fleet-qwen38-hosted-identity-selector-package-v1"
FAILURE_SCHEMA = "fleet-qwen38-hosted-identity-selector-failure-v1"
SELECTION_PATH = "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
SELECTION_FILE_SHA256 = (
    "sha256:9f5e83123cfe7d8a3f4cc0b791ac1dc9032a3d5faf5eba2a431c7246069255df"
)
SELECTION_SHA256 = "sha256:38bd544c74f4e45cb67b271849d657f49e356ccf41f97acca9fb6cd7eb7f56b8"
ROSTER_SHA256 = "sha256:8c5d320091bc6a82a8ee88ce18b3a4a8b6e8b174d69fa70b5b951529973ce45f"
EXPECTED_TALLY = {"accepted": 51, "active": 0, "blocked_nonrepeatable": 9, "unstarted": 340}
EXPECTED_MODEL = base.EXPECTED_SESSION_MODEL
MAX_PAGES = 10_000
MAX_ROWS = 1_000_000
SAFE_FAILURE_CODES = base.SAFE_FAILURE_CODES | {
    "identity_selector_binding_invalid",
    "identity_selector_inventory_ambiguous",
    "identity_selector_package_invalid",
    "identity_selector_package_file_drifted",
    "identity_selector_pagination_stalled",
    "session_inventory_invalid",
    "session_inventory_too_large",
}


def validate_binding(value: Any) -> None:
    if not isinstance(value, dict):
        raise base.GateError("identity_selector_binding_invalid")
    roster = value.get("task_roster")
    if (
        set(value)
        != {
            "schema_version",
            "selection_path",
            "selection_file_sha256",
            "selection_sha256",
            "roster_sha256",
            "task_count",
            "task_roster",
            "session_model",
            "authoritative_tally",
            "output_root",
            "binding_sha256",
        }
        or value.get("schema_version") != BINDING_SCHEMA
        or value.get("selection_path") != SELECTION_PATH
        or value.get("selection_file_sha256") != SELECTION_FILE_SHA256
        or value.get("selection_sha256") != SELECTION_SHA256
        or value.get("roster_sha256") != ROSTER_SHA256
        or value.get("task_count") != 100
        or not isinstance(roster, list)
        or len(roster) != 100
        or roster != sorted(roster, key=lambda row: row.get("rank", -1))
        or [row.get("rank") for row in roster] != list(range(1, 101))
        or any(set(row) != {"rank", "task_key"} for row in roster)
        or any(not isinstance(row.get("task_key"), str) or not row["task_key"] for row in roster)
        or len({row["task_key"] for row in roster}) != 100
        or base.sha256(base.canonical(roster)) != ROSTER_SHA256
        or value.get("session_model") != EXPECTED_MODEL
        or value.get("authoritative_tally") != EXPECTED_TALLY
        or value.get("output_root") != "/mnt/sfs/jobs/chris-q38-hosted-identity-selector-v1"
        or value.get("binding_sha256") != base.binding_digest(value)
    ):
        raise base.GateError("identity_selector_binding_invalid")


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise base.GateError("redirect_forbidden")


def _response_chunks(response: Any) -> Any:
    while True:
        block = response.read(64 * 1024)
        if not block:
            return
        yield block


def _session_page(api_key: str, *, offset: int) -> stream.SessionPage:
    query = urllib.parse.urlencode({"limit": 500, "offset": offset})
    request = urllib.request.Request(
        f"{base.ORCHESTRATOR}/v1/sessions?{query}",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(
        _RejectRedirects(), urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    try:
        with opener.open(request, timeout=60) as response:
            return stream.parse_session_page(_response_chunks(response))
    except urllib.error.HTTPError as exc:
        raise base.GateError("read_request_failed") from exc


def select(
    binding: dict[str, Any], api_key: str, *, job_uid: str, pod_uid: str
) -> dict[str, Any]:
    validate_binding(binding)
    account = base._fleet_get("/v1/account", api_key)  # noqa: SLF001
    if account.get("team_name") != "fleet" or account.get("team_id") != base.FLEET_TEAM_ID:
        raise base.GateError("fleet_team_identity_invalid")

    ranks = {row["task_key"]: row["rank"] for row in binding["task_roster"]}
    states = {rank: "IDENTITY_CLEAR" for rank in range(1, 101)}
    session_ids: set[str] = set()
    pages = rows_examined = offset = 0
    while True:
        pages += 1
        if pages > MAX_PAGES:
            raise base.GateError("identity_selector_pagination_stalled")
        page = _session_page(api_key, offset=offset)
        if page.offset != offset or page.limit != 500 or len(page.sessions) > 500:
            raise base.GateError("identity_selector_pagination_stalled")
        for row in page.sessions:
            session_id = row.get("session_id")
            eval_task_id = row.get("eval_task_id")
            task_key = row.get("task_key")
            status = row.get("status")
            if (
                not isinstance(session_id, str)
                or not session_id
                or session_id in session_ids
                or not isinstance(eval_task_id, str)
                or not eval_task_id
                or not isinstance(task_key, str)
                or not task_key
                or (status is not None and not isinstance(status, str))
            ):
                raise base.GateError("identity_selector_inventory_ambiguous")
            session_ids.add(session_id)
            rows_examined += 1
            if rows_examined > MAX_ROWS:
                raise base.GateError("identity_selector_pagination_stalled")
            rank = ranks.get(task_key)
            if rank is None:
                continue
            model = row.get("model")
            if not isinstance(model, str) or not model:
                states[rank] = "AMBIGUOUS_BLOCK"
            elif model == binding["session_model"] and states[rank] != "AMBIGUOUS_BLOCK":
                states[rank] = "EXACT_MODEL_COLLISION"
        if not page.has_more:
            break
        if not page.sessions:
            raise base.GateError("identity_selector_pagination_stalled")
        offset += len(page.sessions)

    counts = {
        name: sum(state == name for state in states.values())
        for name in ("IDENTITY_CLEAR", "EXACT_MODEL_COLLISION", "AMBIGUOUS_BLOCK")
    }
    clear = [rank for rank, state in states.items() if state == "IDENTITY_CLEAR"]
    return base.seal(
        {
            "schema_version": SCHEMA,
            "status": "CLEAR_CANDIDATE_IDENTIFIED" if clear else "NO_CLEAR_CANDIDATE",
            "selection_file_sha256": SELECTION_FILE_SHA256,
            "selection_sha256": SELECTION_SHA256,
            "roster_sha256": ROSTER_SHA256,
            "binding_sha256": binding["binding_sha256"],
            "authoritative_tally": EXPECTED_TALLY,
            "task_count": 100,
            "classification_counts": counts,
            "earliest_clear_rank": min(clear) if clear else None,
            "session_pages_read": pages,
            "session_rows_examined": rows_examined,
            "runtime": {
                "job_uid": base._uuid(job_uid, "observer_job_uid"),  # noqa: SLF001
                "pod_uid": base._uuid(pod_uid, "observer_pod_uid"),  # noqa: SLF001
            },
            "methods": ["GET"],
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "protected_values_materialized": False,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )


def validate_observation(value: Any, binding: dict[str, Any]) -> None:
    validate_binding(binding)
    if not isinstance(value, dict) or value.get("receipt_sha256") != base.digest(value):
        raise base.GateError("release_observation_invalid")
    counts = value.get("classification_counts")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "selection_file_sha256",
            "selection_sha256",
            "roster_sha256",
            "binding_sha256",
            "authoritative_tally",
            "task_count",
            "classification_counts",
            "earliest_clear_rank",
            "session_pages_read",
            "session_rows_examined",
            "runtime",
            "methods",
            "model_calls",
            "task_calls",
            "session_mutations",
            "verifier_calls",
            "scoring_calls",
            "api_mutations",
            "protected_values_materialized",
            "scores_included",
            "prompts_traces_flags_included",
            "credentials_included",
            "receipt_sha256",
        }
        or value.get("schema_version") != SCHEMA
        or value.get("selection_file_sha256") != SELECTION_FILE_SHA256
        or value.get("selection_sha256") != SELECTION_SHA256
        or value.get("roster_sha256") != ROSTER_SHA256
        or value.get("binding_sha256") != binding["binding_sha256"]
        or value.get("authoritative_tally") != EXPECTED_TALLY
        or value.get("task_count") != 100
        or not isinstance(counts, dict)
        or set(counts) != {"IDENTITY_CLEAR", "EXACT_MODEL_COLLISION", "AMBIGUOUS_BLOCK"}
        or any(type(count) is not int or count < 0 for count in counts.values())
        or sum(counts.values()) != 100
        or value.get("status")
        not in {"CLEAR_CANDIDATE_IDENTIFIED", "NO_CLEAR_CANDIDATE"}
        or (value["status"] == "CLEAR_CANDIDATE_IDENTIFIED" and counts["IDENTITY_CLEAR"] < 1)
        or (value["status"] == "NO_CLEAR_CANDIDATE" and counts["IDENTITY_CLEAR"] != 0)
        or (value["status"] == "CLEAR_CANDIDATE_IDENTIFIED")
        != (type(value.get("earliest_clear_rank")) is int)
        or (
            type(value.get("earliest_clear_rank")) is int
            and not 1 <= value["earliest_clear_rank"] <= 100
        )
        or type(value.get("session_pages_read")) is not int
        or value["session_pages_read"] < 1
        or type(value.get("session_rows_examined")) is not int
        or value["session_rows_examined"] < 0
        or not isinstance(value.get("runtime"), dict)
        or set(value["runtime"]) != {"job_uid", "pod_uid"}
        or value["runtime"].get("job_uid")
        != base._uuid(value["runtime"].get("job_uid"), "observer_job_uid")  # noqa: SLF001
        or value["runtime"].get("pod_uid")
        != base._uuid(value["runtime"].get("pod_uid"), "observer_pod_uid")  # noqa: SLF001
        or value.get("methods") != ["GET"]
        or any(
            value.get(key) != 0
            for key in (
                "model_calls",
                "task_calls",
                "session_mutations",
                "verifier_calls",
                "scoring_calls",
                "api_mutations",
            )
        )
        or value.get("protected_values_materialized") is not False
        or value.get("scores_included") is not False
        or value.get("prompts_traces_flags_included") is not False
        or value.get("credentials_included") is not False
    ):
        raise base.GateError("release_observation_invalid")


def validate_package_source(path: Path, root: Path) -> None:
    value = base.load_projected(path, root)
    files = value.get("files")
    expected = {
        "binding.json",
        "qwen_hosted_identity_selector_v1.py",
        "qwen_hosted_rank17_g22_release_observer_v1.py",
        "score_blind_session_inventory_v1.py",
    }
    if (
        set(value) != {"schema_version", "files", "file_count", "receipt_sha256"}
        or value.get("schema_version") != PACKAGE_SCHEMA
        or not isinstance(files, dict)
        or set(files) != expected
        or value.get("file_count") != len(expected)
        or value.get("receipt_sha256") != base.digest(value)
    ):
        raise base.GateError("identity_selector_package_invalid")
    for name, expected_digest in files.items():
        try:
            target = (root / name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise base.GateError("identity_selector_package_file_drifted") from exc
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or target.stat().st_size > base.MAX_JSON_BYTES
            or base.sha256(target.read_bytes()) != expected_digest
        ):
            raise base.GateError("identity_selector_package_file_drifted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = "runtime-identity"
    job_uid: str | None = None
    pod_uid: str | None = None
    try:
        job_uid = base._uuid(os.environ.get("JOB_UID"), "observer_job_uid")  # noqa: SLF001
        pod_uid = base._uuid(os.environ.get("POD_UID"), "observer_pod_uid")  # noqa: SLF001
        stage = "package-source"
        validate_package_source(args.package_source, args.package_source.parent)
        stage = "binding"
        binding = base.load_projected(args.binding, args.package_source.parent)
        validate_binding(binding)
        api_key = os.environ.get("FLEET_API_KEY")
        if not api_key:
            raise base.GateError("fleet_api_key_absent")
        stage = "collect"
        observation = select(binding, api_key, job_uid=job_uid, pod_uid=pod_uid)
        validate_observation(observation, binding)
        stage = "write"
        base.write_once(args.output, observation)
    except Exception as exc:
        code = str(exc)
        failure = base.seal(
            {
                "schema_version": FAILURE_SCHEMA,
                "status": "FAILED",
                "last_stage": stage,
                "failure_code": code if code in SAFE_FAILURE_CODES else "redacted",
                "job_uid": job_uid,
                "pod_uid": pod_uid,
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scores_included": False,
                "prompts_traces_flags_included": False,
                "credentials_included": False,
            }
        )
        with contextlib.suppress(Exception):
            base.write_once(args.output.with_name("FAILED.json"), failure)
        raise base.GateError("observer_failed_safely") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
