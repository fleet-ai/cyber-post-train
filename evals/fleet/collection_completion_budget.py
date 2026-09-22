"""Collection-only completion-budget runtime overlay.

This module is a successor input, not an edit to the frozen v1/v2 collection
runtime.  It mounts a private evidence directory into the model proxy, permits
the configured number of chat completions in addition to bounded discovery
traffic, and translates only a cryptographically valid exhaustion receipt into
an ``output_limit`` termination.  Every other non-zero OpenCode exit remains a
process error.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest
from evals.fleet import collection_fixed_proxy_v2 as proxy
from evals.fleet import opencode_self_hosted as harness
from evals.fleet import rollout_worker as worker
from evals.fleet import visible_action_collection as visible_action_v1

PROXY_FILE = "collection_fixed_proxy_v2.py"
BUDGET_DIRECTORY = "completion-budget-evidence"
BUDGET_RECEIPT_FILE = "COMPLETION_BUDGET_EXHAUSTED.json"
TOTAL_REQUEST_HEADROOM = 30

_BASE_RUN = harness.run
_BASE_DOCKER = harness._docker  # noqa: SLF001
_BASE_TERMINATION = harness.opencode_termination
_BASE_ACCEPTED = worker._accepted_receipt  # noqa: SLF001
_STATE = threading.local()


class CompletionBudgetOutputLimit(RuntimeError):
    """One exact configured completion budget ended the agent process."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_path(out_dir: Path) -> Path:
    return out_dir / BUDGET_DIRECTORY / BUDGET_RECEIPT_FILE


def validate_receipt(path: Path, *, maximum_completions: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("completion budget receipt is not a regular file")
    value = json.loads(path.read_text())
    fields = {
        "schema",
        "reason",
        "maximum_total_requests",
        "completed_total_requests",
        "maximum_completions",
        "completed_completions",
        "rejected_completion_number",
        "observed_request_number",
        "prompts_responses_scores_or_credentials_included",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("completion budget receipt has unknown or missing fields")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value["sha256"] != "sha256:" + digest(unsigned):
        raise ValueError("completion budget receipt digest changed")
    if (
        value["schema"] != proxy.BUDGET_RECEIPT_SCHEMA
        or value["reason"] != "exact_completion_budget_exhausted"
        or value["maximum_total_requests"] != maximum_completions + TOTAL_REQUEST_HEADROOM
        or type(value["completed_total_requests"]) is not int
        or value["completed_total_requests"] != value["observed_request_number"] - 1
        or value["maximum_completions"] != maximum_completions
        or value["completed_completions"] != maximum_completions
        or value["rejected_completion_number"] != maximum_completions + 1
        or type(value["observed_request_number"]) is not int
        or not maximum_completions + 1
        <= value["observed_request_number"]
        <= maximum_completions + TOTAL_REQUEST_HEADROOM
        or value["prompts_responses_scores_or_credentials_included"] is not False
    ):
        raise ValueError("completion budget receipt does not prove exact exhaustion")
    return value


def _budget_docker(*arguments: Any, **kwargs: Any) -> Any:
    rendered = visible_action_v1.collection_docker_args(arguments)
    environment = kwargs.get("env")
    if isinstance(environment, dict) and environment.get("FIXED_ALLOWED_PATHS") == (
        "/v1/chat/completions,/v1/models"
    ):
        out_dir = getattr(_STATE, "out_dir", None)
        maximum = getattr(_STATE, "maximum_completions", None)
        if not isinstance(out_dir, Path) or type(maximum) is not int or maximum < 1:
            raise RuntimeError("completion budget runtime state is unavailable")
        if environment.get("FIXED_MAX_REQUESTS") != str(maximum):
            raise RuntimeError("historical proxy request ceiling changed unexpectedly")
        budget_directory = out_dir / BUDGET_DIRECTORY
        budget_directory.mkdir(mode=0o700)
        environment = dict(environment)
        environment.update(
            {
                "FIXED_MAX_REQUESTS": str(maximum + TOTAL_REQUEST_HEADROOM),
                "FIXED_MAX_COMPLETIONS": str(maximum),
                "FIXED_BUDGET_EVIDENCE_PATH": f"/budget/{BUDGET_RECEIPT_FILE}",
            }
        )
        kwargs["env"] = environment
        try:
            command_index = rendered.index("python")
        except ValueError as error:
            raise RuntimeError("fixed model proxy command is missing") from error
        image_index = command_index - 1
        rendered = (
            *rendered[:image_index],
            "-e",
            "FIXED_MAX_COMPLETIONS",
            "-e",
            "FIXED_BUDGET_EVIDENCE_PATH",
            "-v",
            f"{budget_directory.resolve()}:/budget",
            *rendered[image_index:],
        )
    return _BASE_DOCKER(*rendered, **kwargs)


def _budget_run(
    config: dict[str, Any],
    out_dir: Path,
    proxy_script: Path,
    *,
    safe_scoring_intent_sink=None,
) -> dict[str, Any]:
    expected_proxy = Path(__file__).with_name(PROXY_FILE).resolve()
    if proxy_script.resolve() != expected_proxy:
        raise RuntimeError("completion-budget runtime requires its exact successor proxy")
    maximum = config.get("harness", {}).get("max_model_requests")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("positive exact completion budget is required")
    if hasattr(_STATE, "out_dir"):
        raise RuntimeError("nested completion-budget runtime is forbidden")
    _STATE.out_dir = out_dir
    _STATE.maximum_completions = maximum
    try:
        return _BASE_RUN(
            config,
            out_dir,
            proxy_script,
            safe_scoring_intent_sink=safe_scoring_intent_sink,
        )
    finally:
        del _STATE.out_dir
        del _STATE.maximum_completions


def _budget_termination(
    events: list[dict[str, Any]], *, malformed_lines: int, exit_code: int, timed_out: bool
) -> str:
    base = _BASE_TERMINATION(
        events,
        malformed_lines=malformed_lines,
        exit_code=exit_code,
        timed_out=timed_out,
    )
    if base != "process_error":
        return base
    out_dir = getattr(_STATE, "out_dir", None)
    maximum = getattr(_STATE, "maximum_completions", None)
    if not isinstance(out_dir, Path) or type(maximum) is not int:
        return base
    try:
        validate_receipt(receipt_path(out_dir), maximum_completions=maximum)
    except (OSError, ValueError, json.JSONDecodeError):
        return base
    return "output_limit"


def _budget_accepted(
    client: Any,
    config: dict[str, Any],
    ledger_cell: dict[str, Any],
    result: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if result.get("agent_termination") == "output_limit":
        try:
            validate_receipt(
                receipt_path(out_dir),
                maximum_completions=config["harness"]["max_model_requests"],
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # A native per-response length stop is also called ``output_limit``
            # by the base harness.  It is not the collection-wide exhaustion
            # condition and must retain the historical acceptance failure.
            pass
        else:
            raise CompletionBudgetOutputLimit("exact completion budget exhausted")
    return _BASE_ACCEPTED(client, config, ledger_cell, result, out_dir)


def activate_runtime() -> None:
    """Install exact collection-only overrides once before worker threads."""
    allowed_settings = {
        visible_action_v1._BASE_OPENCODE_SETTINGS,  # noqa: SLF001
        visible_action_v1.nonthinking_opencode_settings,
    }
    if harness.opencode_settings not in allowed_settings:
        raise RuntimeError("OpenCode settings were already modified by another runtime")
    if harness._docker not in {_BASE_DOCKER, _budget_docker}:  # noqa: SLF001
        raise RuntimeError("Docker launcher was already modified by another runtime")
    if harness.run not in {_BASE_RUN, _budget_run}:
        raise RuntimeError("self-hosted runner was already modified by another runtime")
    if harness.opencode_termination not in {_BASE_TERMINATION, _budget_termination}:
        raise RuntimeError("OpenCode termination was already modified by another runtime")
    if worker._accepted_receipt not in {_BASE_ACCEPTED, _budget_accepted}:  # noqa: SLF001
        raise RuntimeError("rollout acceptance was already modified by another runtime")
    harness.opencode_settings = visible_action_v1.nonthinking_opencode_settings
    harness._docker = _budget_docker  # type: ignore[attr-defined]  # noqa: SLF001
    harness.run = _budget_run
    harness.opencode_termination = _budget_termination
    worker._accepted_receipt = _budget_accepted  # type: ignore[attr-defined]  # noqa: SLF001
