"""Immutable data models for Fleet baseline evaluation plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
OTS_PROJECT_ID = "63d6fda8-48c4-4726-9ec3-d1028f2c47f5"
DEFAULT_SALES_PRODUCT_ID = "e4084858-f272-570f-82bc-a74e1ad60b2e"
DEFAULT_SOURCE_JOB_ID = "a62dd51f-a52b-4941-8207-4679e4b25b51"
DEFAULT_GATEWAY_MODEL = "glm-5.2-fp8"
DEFAULT_EXPERIMENT = "glm52-fleet-blackbox-baseline-v1"
DEFAULT_RUNTIME_LABEL = "fleet-agent-runtime-v1"
# Historical Agent Runtime examples used ``fleet-glm/glm-5.2-fp8``. New
# tool-use submissions advertise this canonical harness model identifier.
DEFAULT_RUNTIME_MODEL = "z-ai/glm-5.2"
DEFAULT_SMOKE_TASK_KEY = "cysec1-2-current-gen_blackbox-9afe9e08da314948b573657e__blackbox_ctf_v1"
MAX_SESSIONS_PER_JOB = 6
LEGACY_SOURCE_TASK_KEYS = {
    "fakelook_blackbox-e1fabce2cfffde31d1e5ecba__blackbox_ctf_v1",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class JobBatch:
    """One bounded, idempotent Fleet job."""

    index: int
    task_keys: tuple[str, ...]
    runtime_model: str
    pass_k: int
    max_steps: int
    max_duration_minutes: int
    run_name: str
    source_job_id: str
    harness: str | None = None
    experiment: str = DEFAULT_EXPERIMENT
    gateway_model: str = DEFAULT_GATEWAY_MODEL
    runtime_label: str = DEFAULT_RUNTIME_LABEL

    @property
    def planned_sessions(self) -> int:
        return len(self.task_keys) * self.pass_k

    @property
    def name(self) -> str:
        return f"{self.run_name}-b{self.index:03d}"

    @property
    def idempotency_key(self) -> str:
        logical_request = {
            "agent_runtime": True,
            "max_duration_minutes": self.max_duration_minutes,
            "max_steps": self.max_steps,
            "model": self.runtime_model,
            "name": self.name,
            "pass_k": self.pass_k,
            "sales_product_id": DEFAULT_SALES_PRODUCT_ID,
            "task_keys": self.task_keys,
        }
        # Preserve legacy GLM idempotency keys while binding every treatment
        # selector for explicit alternate-harness arms.
        if self.harness is not None:
            logical_request.update(
                {
                    "harness": self.harness,
                    "experiment": self.experiment,
                    "gateway_model": self.gateway_model,
                    "runtime_label": self.runtime_label,
                    "source_job_id": self.source_job_id,
                }
            )
        return str(uuid5(NAMESPACE_URL, _canonical_json(logical_request)))

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "task_keys": list(self.task_keys),
            "models": [self.runtime_model],
            "pass_k": self.pass_k,
            "mode": "tool-use",
            "cost_team_id": FLEET_TEAM_ID,
            "project_id": OTS_PROJECT_ID,
            "sales_product_id": DEFAULT_SALES_PRODUCT_ID,
            "capture_db_diff": True,
            "job_type": "evaluation",
            "agent_runtime": True,
            "max_steps": self.max_steps,
            "max_duration_minutes": self.max_duration_minutes,
            "metadata": {
                "experiment": self.experiment,
                "source_job_id": self.source_job_id,
                "gateway_model": self.gateway_model,
                "runtime": self.runtime_label,
            },
        }
        if self.harness is not None:
            payload["harness"] = self.harness
        return payload

    def sanitized_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["idempotency_key"] = self.idempotency_key
        payload["planned_sessions"] = self.planned_sessions
        return payload


@dataclass(frozen=True)
class EvalPlan:
    """A reproducible sequence of capped Fleet jobs."""

    source_job_id: str
    source_task_digest: str
    batches: tuple[JobBatch, ...]

    @property
    def task_count(self) -> int:
        return sum(len(batch.task_keys) for batch in self.batches)

    @property
    def planned_sessions(self) -> int:
        return sum(batch.planned_sessions for batch in self.batches)

    def sanitized_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fleet_agent_runtime_eval_plan_v1",
            "source_job_id": self.source_job_id,
            "source_task_digest": self.source_task_digest,
            "task_count": self.task_count,
            "planned_sessions": self.planned_sessions,
            "batches": [batch.sanitized_dict() for batch in self.batches],
        }


def validate_task_keys(task_keys: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate, deduplicate and sort registered cyber task keys."""

    normalized = tuple(sorted(set(task_keys)))
    if not normalized:
        raise ValueError("no task keys were found")
    invalid = [
        key
        for key in normalized
        if (not key.startswith("cysec1-2-") or not key.endswith("__blackbox_ctf_v1"))
        and key not in LEGACY_SOURCE_TASK_KEYS
    ]
    if invalid:
        sample = ", ".join(invalid[:3])
        raise ValueError(f"found non-canonical Fleet blackbox task keys: {sample}")
    return normalized


def task_digest(task_keys: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(task_keys) + "\n").encode()).hexdigest()


def build_plan(
    task_keys: list[str] | tuple[str, ...],
    *,
    source_job_id: str = DEFAULT_SOURCE_JOB_ID,
    runtime_model: str = DEFAULT_RUNTIME_MODEL,
    pass_k: int = 1,
    max_steps: int = 300,
    max_duration_minutes: int = 120,
    batch_session_cap: int = MAX_SESSIONS_PER_JOB,
    run_name: str = "glm52-fleet-blackbox-baseline-v1",
    harness: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    gateway_model: str = DEFAULT_GATEWAY_MODEL,
    runtime_label: str = DEFAULT_RUNTIME_LABEL,
) -> EvalPlan:
    keys = validate_task_keys(task_keys)
    if pass_k < 1:
        raise ValueError("pass_k must be positive")
    if not 1 <= batch_session_cap <= MAX_SESSIONS_PER_JOB:
        raise ValueError(f"batch_session_cap must be between 1 and {MAX_SESSIONS_PER_JOB}")
    tasks_per_batch = batch_session_cap // pass_k
    if tasks_per_batch < 1:
        raise ValueError("pass_k exceeds the six-session per-job safety cap")

    batches = tuple(
        JobBatch(
            index=index,
            task_keys=keys[offset : offset + tasks_per_batch],
            runtime_model=runtime_model,
            pass_k=pass_k,
            max_steps=max_steps,
            max_duration_minutes=max_duration_minutes,
            run_name=run_name,
            source_job_id=source_job_id,
            harness=harness,
            experiment=experiment,
            gateway_model=gateway_model,
            runtime_label=runtime_label,
        )
        for index, offset in enumerate(range(0, len(keys), tasks_per_batch), start=1)
    )
    return EvalPlan(
        source_job_id=source_job_id,
        source_task_digest=task_digest(keys),
        batches=batches,
    )
