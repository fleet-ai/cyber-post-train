"""Typed, duplicate-safe client for Fleet's Nebius training Jobs API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .rl_config import read_json

DEFAULT_BASE_URL = "https://api.ft.flt.build"
SUPPORTED_KINDS = {"rl", "sft"}


class JobsAPIError(ValueError):
    """A request or response violated the safe launch contract."""


def load_run_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    kind = config.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise JobsAPIError(f"run config kind must be one of {sorted(SUPPORTED_KINDS)}")
    title = config.get("title")
    if not isinstance(title, str) or not title.strip():
        raise JobsAPIError("run config requires a non-empty title")
    return config


class TrainingJobsClient:
    """Small API boundary; bearer credentials are held only in request headers."""

    def __init__(
        self,
        bearer: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not bearer:
            raise JobsAPIError("FLEET_TRAINING_API_TOKEN must be non-empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=60,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TrainingJobsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500]
            raise JobsAPIError(
                f"Jobs API {method} {path} returned HTTP {response.status_code}: {detail}"
            ) from exc
        try:
            return response.json()
        except ValueError as exc:
            raise JobsAPIError(f"Jobs API {method} {path} returned non-JSON") from exc

    def preview(self, config: dict[str, Any]) -> dict[str, Any]:
        kind = str(config.get("kind") or "")
        if kind not in SUPPORTED_KINDS:
            raise JobsAPIError(f"unsupported run kind {kind!r}")
        value = self._json("POST", f"/v1/{kind}/run/preview", json=config)
        if not isinstance(value, dict):
            raise JobsAPIError("Jobs API preview returned a non-object")
        return value

    def list_runs(self, *, limit: int = 200) -> list[dict[str, Any]]:
        value = self._json("GET", "/v1/rl/run", params={"limit": limit})
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise JobsAPIError("Jobs API run list returned an invalid payload")
        return value

    def runs_with_title(self, title: str) -> list[dict[str, Any]]:
        return [row for row in self.list_runs() if row.get("title") == title]

    def submit(self, config: dict[str, Any]) -> dict[str, Any]:
        kind = str(config.get("kind") or "")
        title = str(config.get("title") or "")
        duplicates = self.runs_with_title(title)
        if duplicates:
            names = ", ".join(str(row.get("name")) for row in duplicates)
            raise JobsAPIError(f"refusing duplicate run title {title!r}; existing: {names}")
        preview = self.preview(config)
        errors = preview.get("errors") or []
        if errors:
            raise JobsAPIError(f"refusing preview with errors: {errors}")
        value = self._json("POST", f"/v1/{kind}/run", json=config)
        if not isinstance(value, dict) or not value.get("name"):
            raise JobsAPIError("Jobs API submit returned an invalid receipt")
        return value

    def status(self, name: str) -> dict[str, Any]:
        value = self._json("GET", f"/v1/rl/run/{name}")
        if not isinstance(value, dict):
            raise JobsAPIError("Jobs API status returned a non-object")
        return value


def concise_status(run: dict[str, Any]) -> dict[str, Any]:
    """Return operational evidence without copying manifests or task payloads."""
    sessions = run.get("sessions") if isinstance(run.get("sessions"), dict) else {}
    return {
        "name": run.get("name"),
        "kind": run.get("kind"),
        "title": run.get("title"),
        "status": run.get("status"),
        "status_detail": run.get("status_detail"),
        "failure_message": run.get("failure_message"),
        "failure_signature": run.get("failure_signature"),
        "trainer_version_id": run.get("trainer_version_id"),
        "ray_job_id": run.get("ray_job_id"),
        "steps": len(run.get("steps") or []),
        "step_metrics": len(run.get("step_metrics") or []),
        "checkpoints": len(run.get("checkpoints") or []),
        "training_sessions": len(sessions.get("training") or []),
        "eval_sessions": len(sessions.get("eval") or []),
    }
