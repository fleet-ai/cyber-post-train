"""GLM whole-task engine adapter with a process-scoped atomic claim hook."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as base

UUID_RE = base.UUID_RE
bulk = base.bulk
claim_cell = base.claim_cell
claim_filename = base.claim_filename
_assert_run_absent = base._assert_run_absent
_attempt_config = base._attempt_config
_task_for_item = base._task_for_item
_validate_preserved_claim = base._validate_preserved_claim
_write_once = base._write_once


def run_controller(
    plan: dict[str, Any],
    *,
    claim_provider: Callable[
        [dict[str, Any], dict[str, Any], Path, str, str], dict[str, Any] | None
    ],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the byte-pinned base engine with one local atomic claim provider."""
    prior_bulk = base.bulk
    prior_claim_cell = base.claim_cell
    try:
        base.bulk = bulk
        base.claim_cell = claim_provider
        return base.run_controller(plan, **kwargs)
    finally:
        base.claim_cell = prior_claim_cell
        base.bulk = prior_bulk
