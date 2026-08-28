"""Read-only Fleet transcript export client.

Only public, authenticated read routes are used.  The exporter never downloads
grader source/stdout and never writes back to Fleet.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


class FleetExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionRef:
    job_id: str
    job_name: str | None
    task_key: str
    session_id: str
    session: Mapping[str, Any]
    task_binding: Mapping[str, Any]


ROSTER_TASK_BINDING_FIELDS = {
    "key",
    "env_id",
    "created_at",
    "version",
    "eval_task_version_id",
    "data_id",
    "data_version",
    "multi_app_seed_versions",
    "verifier_id",
    "verifier_sha",
    "output_json_schema",
}


class FleetClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://orchestrator.fleetai.com",
        timeout: float = 120,
    ):
        if not api_key:
            raise ValueError("FLEET_API_KEY is required")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        last: BaseException | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise FleetExportError(f"GET {path} returned non-object JSON")
                return payload
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    detail = exc.read(1000).decode("utf-8", "replace")
                    raise FleetExportError(f"GET {path} failed ({exc.code}): {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            if attempt < 4:
                time.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.2)
        raise FleetExportError(f"GET {path} failed after retries: {last}")

    def job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}")

    def job_sessions(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/v1/sessions/job/{urllib.parse.quote(job_id, safe='')}")

    def transcript(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/transcript")

    def completed_jobs(self, *, created_after: dt.datetime) -> Iterator[dict[str, Any]]:
        """Page the inclusive `created_before` cursor without duplicating rows."""
        cursor: dt.datetime | None = None
        seen: set[str] = set()
        while True:
            payload = self._get(
                "/v1/jobs",
                {"limit": 1000, "created_before": cursor.isoformat() if cursor else None},
            )
            jobs = payload.get("jobs") or []
            if not isinstance(jobs, list):
                raise FleetExportError("GET /v1/jobs omitted jobs[]")
            fresh = [
                job for job in jobs if isinstance(job, dict) and str(job.get("id")) not in seen
            ]
            for job in fresh:
                seen.add(str(job.get("id")))
                created = _parse_time(job.get("created_at"))
                if created and created < created_after:
                    return
                if str(job.get("status") or "").lower() in {"completed", "succeeded", "success"}:
                    yield job
            times = [_parse_time(job.get("created_at")) for job in fresh]
            times = [value for value in times if value]
            if len(jobs) < 1000 or not fresh or not times:
                return
            cursor = min(times) - dt.timedelta(microseconds=1)


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def roster_session_refs(
    job_id: str, job_name: str | None, roster: Mapping[str, Any]
) -> list[SessionRef]:
    refs: list[SessionRef] = []
    for task_group in roster.get("tasks") or []:
        if not isinstance(task_group, Mapping):
            continue
        task = task_group.get("task") if isinstance(task_group.get("task"), Mapping) else {}
        task_key = str(
            task.get("key") or task_group.get("task_key") or task_group.get("task_id") or ""
        )
        task_binding = {
            key: task[key]
            for key in sorted(ROSTER_TASK_BINDING_FIELDS)
            if key in task and task[key] is not None
        }
        for session in task_group.get("sessions") or []:
            if not isinstance(session, Mapping):
                continue
            session_id = session.get("session_id") or session.get("id")
            if isinstance(session_id, str) and session_id:
                refs.append(
                    SessionRef(job_id, job_name, task_key, session_id, session, task_binding)
                )
    return refs


def is_blackbox_task_key(task_key: str) -> bool:
    return "blackbox" in task_key.lower() or "__blackbox_ctf_v" in task_key.lower()


def export_sessions(
    client: FleetClient,
    refs: Iterable[SessionRef],
    *,
    workers: int = 16,
) -> Iterator[dict[str, Any]]:
    refs = list(refs)
    if workers < 1 or workers > 64:
        raise ValueError("workers must be in 1..64")

    def fetch(ref: SessionRef) -> tuple[SessionRef, dict[str, Any]]:
        return ref, client.transcript(ref.session_id)

    # executor.map preserves roster order, making manifests reproducible.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for ref, transcript in executor.map(fetch, refs):
            yield {
                "export_schema": "fleet_session_export_v1",
                "source": {
                    "job_id": ref.job_id,
                    "job_name": ref.job_name,
                    "session_id": ref.session_id,
                    "task_key_from_roster": ref.task_key,
                    "roster_task_binding": dict(ref.task_binding),
                },
                "session": dict(ref.session),
                "transcript_envelope": transcript,
            }
