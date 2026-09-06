"""Rank-30 runtime-plan adapter with distinct static and runtime bindings."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_v3 as prior
from evals.fleet import self_hosted

CONTROLLER = prior.CONTROLLER
CONTROLLERS = prior.CONTROLLERS
RELEASE_SCHEMA = prior.RELEASE_SCHEMA
RELEASE_PATH = prior.RELEASE_PATH
SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CLAIM_ROOT = prior.CLAIM_ROOT
JOBS_ROOT = prior.JOBS_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
build_plan = prior.build_plan
validate_all = prior.validate_all
release_projection = prior.release_projection
validate_release = prior.validate_release


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("rank-30 single-slot controller drifted")
    source_plan = prior.prior.whole.source.build_runtime_plan(
        prior.prior.whole.SOURCE_CONTROLLER, inventory_receipt, root
    )
    runtime_source_sha = source_plan.get("plan_sha256")
    if runtime_source_sha != self_hosted.digest_without(source_plan, "plan_sha256"):
        raise ValueError("hosted GLM runtime source plan digest drifted")

    # The historical transformer correctly projects the plan body but compares
    # every input against the reviewed static-plan digest. Normalize only that
    # comparison boundary, then preserve both authorities explicitly.
    normalized = copy.deepcopy(source_plan)
    normalized["plan_sha256"] = prior.prior.whole.SOURCE_PLAN_SHA256
    projected = prior.prior.whole._transform(  # noqa: SLF001
        normalized, controller, runtime=True
    )
    body = copy.deepcopy(projected)
    body.pop("plan_sha256")
    body["static_predecessor_plan_sha256"] = prior.prior.whole.SOURCE_PLAN_SHA256
    body["runtime_predecessor_plan_sha256"] = runtime_source_sha
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body
