"""Small, fail-closed Kubernetes identity queries for create-once launchers.

Fetching every object in a busy shared namespace is neither a reliable nor a
necessary way to prove that one reviewed run identity is absent.  The Fleet
launch contract binds a canonical SFS output directory to one root
``fleet.ai/run-name`` label, and binds a fresh UUID to ``fleet.ai/run-id``.
This module turns those already-required bindings into server-side selectors.

It intentionally does not offer an unfiltered list helper. Root create-once
callers use all three bounded selectors below; fixed-name CPU helper Jobs use
the exact-name selector. Every response must be readable and empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from .jobs import JobsError

SFS_JOBS_ROOT = "/mnt/sfs/jobs"
_DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?\Z")


@dataclass(frozen=True)
class IdentityQuery:
    """One server-side exact-identity selector, never an unfiltered census."""

    flag: str
    value: str
    purpose: str

    def kubectl_args(self) -> list[str]:
        return [self.flag, self.value]


def require_canonical_output_root(run_name: object, output_root: object) -> None:
    """Bind output ownership to the mandatory Fleet run-name label.

    Jobs API history remains the authority for generic API runs.  For the
    direct rails, this exact mapping makes a run-name selector an output-owner
    selector as well, while the runtime's exclusive output creation remains the
    final concurrent-writer guard.
    """

    if not isinstance(run_name, str) or _DNS_LABEL.fullmatch(run_name) is None:
        raise JobsError("Kubernetes identity run name is invalid")
    if output_root != f"{SFS_JOBS_ROOT}/{run_name}":
        raise JobsError("direct launch output root must equal its canonical run-name path")


def identity_queries(
    run_name: object, run_id: object, exact_name: object
) -> tuple[IdentityQuery, ...]:
    """Return the complete bounded query set for one direct-launch identity."""

    if not isinstance(run_name, str) or _DNS_LABEL.fullmatch(run_name) is None:
        raise JobsError("Kubernetes identity run name is invalid")
    if not isinstance(exact_name, str) or _DNS_LABEL.fullmatch(exact_name) is None:
        raise JobsError("Kubernetes identity object name is invalid")
    try:
        canonical_run_id = str(UUID(str(run_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise JobsError("Kubernetes identity run ID is invalid") from exc
    if canonical_run_id != run_id:
        raise JobsError("Kubernetes identity run ID is not canonical")
    return (
        IdentityQuery("--selector", f"fleet.ai/run-name={run_name}", "run-name/output"),
        IdentityQuery("--selector", f"fleet.ai/run-id={canonical_run_id}", "run-id"),
        IdentityQuery("--field-selector", f"metadata.name={exact_name}", "exact-name"),
    )


def exact_name_query(name: object) -> IdentityQuery:
    """Build the one bounded query used by create-once CPU helper Jobs."""

    if not isinstance(name, str) or _DNS_LABEL.fullmatch(name) is None:
        raise JobsError("Kubernetes identity object name is invalid")
    return IdentityQuery("--field-selector", f"metadata.name={name}", "exact-name")


def require_empty_identity_list(payload: object, query: IdentityQuery) -> int:
    """Validate one selector response and reject every possible owner.

    The Kubernetes API returns a ``List`` for both CRDs and built-ins.  We do
    not accept a partial, non-list, or malformed answer as absence.
    """

    if not isinstance(payload, dict):
        raise JobsError("Kubernetes scoped duplicate response is not an object")
    kind, items = payload.get("kind"), payload.get("items")
    if not isinstance(kind, str) or not kind.endswith("List") or not isinstance(items, list):
        raise JobsError("Kubernetes scoped duplicate response is incomplete")
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
            raise JobsError("Kubernetes scoped duplicate response contains an invalid object")
    if items:
        raise JobsError(f"a Kubernetes object already owns the direct {query.purpose} identity")
    return 1
