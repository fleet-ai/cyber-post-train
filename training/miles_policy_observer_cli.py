"""Single-POST operator rail for the dev3 Miles policy observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, Jobs, digest

from . import miles_event_evidence as events
from . import miles_policy_observer as observer
from .miles_acceptance import NAMESPACE, NAMESPACE_UID, _json_snapshot

_RESOURCES = (
    "rayjobs.ray.io",
    "workloads.kueue.x-k8s.io",
    "rayclusters.ray.io",
    "pods",
)
AMBIGUOUS_SUBMISSION = "AMBIGUOUS_SUBMISSION.json"


def _append_recovered_response(journal: Path, response: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(journal, flags)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps({"state": "POST_RESPONSE", **response}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _recover_created_run(
    jobs: Jobs,
    request: dict[str, Any],
    journal: Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Recover one lost create response by reads only; never issue another POST."""

    try:
        rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ambiguous Jobs API POST has no readable intent journal") from error
    intent = rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else None
    if (
        intent is None
        or intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != API_URLS["dev"]
        or intent.get("request_sha256") != digest(request)
    ):
        raise ValueError("ambiguous Jobs API POST has no exact dev intent to reconcile")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise ValueError("Jobs API recovery timeout must be numeric")
    if timeout_seconds < 0 or timeout_seconds > 43200:
        raise ValueError("Jobs API recovery timeout must fit the configured watch deadline")
    expected_name = re.compile(re.escape(request["name"]) + r"-[a-f0-9]{8}")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            history = jobs.all_runs()
        except Exception:
            history = []
        candidates = [
            row
            for row in history
            if isinstance(row, dict)
            and row.get("run_dir") == request["run_dir"]
            and expected_name.fullmatch(str(row.get("name", ""))) is not None
        ]
        if len(candidates) > 1:
            raise ValueError("ambiguous Jobs API POST matches multiple name/output records")
        if candidates:
            candidate_name = str(candidates[0]["name"])
            try:
                observed = jobs.status(candidate_name)
            except Exception:
                observed = None
            if observed is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "ambiguous Jobs API POST remained unreadable through watch deadline"
                    ) from None
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
                continue
            run_id = str(observed.get("job_id", ""))
            try:
                parsed_id = uuid.UUID(run_id)
            except ValueError as error:
                raise ValueError("recovered Jobs API run ID is not a UUID") from error
            recovered_name = request["name"] + "-" + run_id[:8]
            status = observed.get("status")
            created_at, finished_at = observed.get("created_at"), observed.get("finished_at")
            try:
                created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                finished = (
                    None
                    if finished_at is None
                    else datetime.fromisoformat(str(finished_at).replace("Z", "+00:00"))
                )
            except ValueError as error:
                raise ValueError("recovered Jobs API timestamps are invalid") from error
            if (
                set(observed)
                != {"name", "job_id", "run_dir", "status", "created_at", "finished_at"}
                or str(parsed_id) != run_id
                or observed.get("name") != candidate_name
                or observed.get("name") != recovered_name
                or observed.get("run_dir") != request["run_dir"]
                or expected_name.fullmatch(recovered_name) is None
                or created.tzinfo is None
                or (finished is not None and finished.tzinfo is None)
                or status
                not in {"queued", "PENDING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"}
                or (status in {"queued", "PENDING", "QUEUED", "RUNNING"} and finished is not None)
                or (status in {"SUCCEEDED", "FAILED", "STOPPED"} and finished is None)
                or (finished is not None and finished < created)
            ):
                raise ValueError("recovered Jobs API run differs from exact name/output")
            _append_recovered_response(journal, observed)
            return observed
        if time.monotonic() >= deadline:
            raise TimeoutError("ambiguous Jobs API POST was not visible during read-only recovery")
        time.sleep(0.5)


def submit_once_or_reconcile(
    jobs: Jobs,
    request: dict[str, Any],
    journal: Path,
    *,
    recovery_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    try:
        return jobs.submit_once(request, journal)
    except Exception:
        if not journal.is_file() or journal.is_symlink():
            raise
        return _recover_created_run(
            jobs,
            request,
            journal,
            timeout_seconds=recovery_timeout_seconds,
        )


def _json(path: Path) -> dict[str, Any]:
    value, _ = _json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("operator input must be a JSON object")
    return value


def _objects(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode:
        raise ValueError("exact dev3 kubectl preflight failed")
    try:
        value = json.loads(process.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exact dev3 kubectl preflight returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("exact dev3 kubectl preflight returned a non-object")
    return value


def _preflight_submission(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finish every deterministic failure-prone check before opening the POST rail."""
    if not os.environ.get("FLEET_API_KEY"):
        raise ValueError("FLEET_API_KEY is required")
    if not 1800 <= args.watch_timeout_seconds <= 43200:
        raise ValueError("watch timeout must be between 30 minutes and 12 hours")
    for path in (args.journal, args.output, args.watch_directory):
        if path.exists() or path.is_symlink():
            raise FileExistsError("observer create-once output already exists")
    plan, request = _json(args.plan), _json(args.request)
    observer._validate_plan(plan, check_files=True, require_current_runtime=True)
    observer._request_projection(
        plan,
        request,
        args.source_commit,
        Path(__file__).resolve().parents[1],
    )
    return plan, request


class WatchBuffer:
    """Open gap-free resource-version watches before the only mutating POST."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen[str]] = []
        self.events: queue.Queue[tuple[float, dict[str, Any]] | BaseException] = queue.Queue()

    def start(self) -> None:
        namespace = _objects(
            [
                "kubectl", "--context", events.DEV_KUBE_CONTEXT,
                "get", "namespace", NAMESPACE, "--output", "json",
            ]
        )
        if namespace.get("metadata", {}).get("uid") != NAMESPACE_UID:
            raise ValueError("kubectl is not bound to the exact dev3 namespace UID")
        for resource in _RESOURCES:
            listing = _objects(
                [
                    "kubectl", "--context", events.DEV_KUBE_CONTEXT,
                    "get", resource, "--namespace", NAMESPACE, "--output", "json",
                ]
            )
            version = listing.get("metadata", {}).get("resourceVersion")
            if not isinstance(version, str) or not version:
                raise ValueError("Kubernetes list omitted its watch resource version")
            process = subprocess.Popen(
                [
                    "kubectl", "--context", events.DEV_KUBE_CONTEXT,
                    "get", resource, "--namespace", NAMESPACE,
                    "--watch-only", "--output-watch-events",
                    "--resource-version", version, "--output", "json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if process.stdout is None:
                raise ValueError("kubectl watch has no readable stream")
            self.processes.append(process)
            threading.Thread(target=self._read, args=(process,), daemon=True).start()
        time.sleep(0.2)
        if any(process.poll() is not None for process in self.processes):
            raise ValueError("a pre-POST Kubernetes watch exited before submission")

    def _read(self, process: subprocess.Popen[str]) -> None:
        decoder, buffer = json.JSONDecoder(), ""
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(4096):
                buffer += chunk
                while buffer.strip():
                    stripped = buffer.lstrip()
                    try:
                        value, end = decoder.raw_decode(stripped)
                    except json.JSONDecodeError:
                        break
                    buffer = stripped[end:]
                    if not isinstance(value, dict):
                        raise ValueError("kubectl watch emitted a non-object")
                    self.events.put((time.time(), value))
            if buffer.strip():
                raise ValueError("kubectl watch ended with incomplete JSON")
        except BaseException as error:
            self.events.put(error)

    @staticmethod
    def _owner(raw: dict[str, Any]) -> tuple[str, str, str] | None:
        owners = raw.get("object", {}).get("metadata", {}).get("ownerReferences") or []
        rows = [row for row in owners if isinstance(row, dict) and row.get("controller") is True]
        if len(rows) != 1:
            return None
        row = rows[0]
        return str(row.get("kind")), str(row.get("name")), str(row.get("uid"))

    @staticmethod
    def _pod_projectable(raw: dict[str, Any]) -> bool:
        obj = raw.get("object", {})
        containers = obj.get("spec", {}).get("containers")
        statuses = obj.get("status", {}).get("containerStatuses")
        if not isinstance(containers, list) or len(containers) != 1:
            return False
        if not isinstance(statuses, list):
            return False
        name = containers[0].get("name") if isinstance(containers[0], dict) else None
        return sum(
            isinstance(row, dict) and row.get("name") == name for row in statuses
        ) == 1

    def preserve_ambiguous(
        self,
        directory: Path,
        request: dict[str, Any],
        journal: Path,
        error: BaseException,
    ) -> dict[str, Any]:
        """Persist value-free candidate ownership if API identity never becomes readable."""

        name_pattern = re.compile(re.escape(request["name"]) + r"-[a-f0-9]{8}")
        with self.events.mutex:
            buffered = list(self.events.queue)
        raws = [
            item[1]
            for item in buffered
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict)
        ]
        rayjobs: set[tuple[str, str]] = set()
        for raw in raws:
            obj = raw.get("object") or {}
            metadata = obj.get("metadata") or {}
            owner = self._owner(raw)
            if obj.get("kind") == "RayJob" and name_pattern.fullmatch(
                str(metadata.get("name", ""))
            ):
                rayjobs.add((str(metadata.get("name")), str(metadata.get("uid"))))
            if owner and owner[0] == "RayJob" and name_pattern.fullmatch(owner[1]):
                rayjobs.add((owner[1], owner[2]))
        clusters = {
            (str((raw.get("object") or {}).get("metadata", {}).get("name")),
             str((raw.get("object") or {}).get("metadata", {}).get("uid")))
            for raw in raws
            if (raw.get("object") or {}).get("kind") == "RayCluster"
            and self._owner(raw) is not None
            and self._owner(raw)[0] == "RayJob"
            and (self._owner(raw)[1], self._owner(raw)[2]) in rayjobs
        }
        projected: list[dict[str, Any]] = []
        for raw in raws:
            obj = raw.get("object") or {}
            metadata = obj.get("metadata") or {}
            kind, owner = obj.get("kind"), self._owner(raw)
            accepted = (
                kind == "RayJob"
                and (str(metadata.get("name")), str(metadata.get("uid"))) in rayjobs
            ) or (
                kind in {"Workload", "RayCluster"}
                and owner is not None
                and owner[0] == "RayJob"
                and (owner[1], owner[2]) in rayjobs
            ) or (
                kind == "Pod"
                and owner is not None
                and owner[0] == "RayCluster"
                and (owner[1], owner[2]) in clusters
            )
            if not accepted:
                continue
            projected.append(
                {
                    "event_type": raw.get("type"),
                    "kind": kind,
                    "name": metadata.get("name"),
                    "uid": metadata.get("uid"),
                    "resource_version": metadata.get("resourceVersion"),
                    "creation_timestamp": metadata.get("creationTimestamp"),
                    "deletion_timestamp": metadata.get("deletionTimestamp"),
                    "owner": None
                    if owner is None
                    else {"kind": owner[0], "name": owner[1], "uid": owner[2]},
                    "controller_status": (obj.get("status") or {}).get("jobStatus")
                    if kind == "RayJob"
                    else None,
                    "phase": (obj.get("status") or {}).get("phase")
                    if kind == "Pod"
                    else None,
                }
            )
        try:
            journal_sha256 = hashlib.sha256(journal.read_bytes()).hexdigest()
        except OSError:
            journal_sha256 = None
        value = {
            "schema": "cyber_miles_ambiguous_submission_v1",
            "status": "possible_active_resource_leak",
            "api_base_url": API_URLS["dev"],
            "request_sha256": digest(request),
            "request_name_prefix": request["name"],
            "run_dir": request["run_dir"],
            "submission_journal_sha256": journal_sha256,
            "failure_type": type(error).__name__,
            "watches_closed_after_deadline": True,
            "candidate_events": projected,
            "private_logs_included": False,
            "environment_values_included": False,
            "task_content_included": False,
        }
        value["sha256"] = digest(value)
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = directory / AMBIGUOUS_SUBMISSION
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return value

    def record(self, directory: Path, submission: dict[str, Any], timeout: int = 43200) -> None:
        run = submission["api"]
        rayjob_uid = raycluster_identity = None
        pending: list[tuple[float, dict[str, Any]]] = []
        terminal_job = terminal_pod = failed_job = False
        seen: set[str] = set()
        sequence, last_recorded_at = 0, 0.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self.events.get(timeout=1)
            except queue.Empty:
                if any(process.poll() is not None for process in self.processes):
                    raise ValueError("a Kubernetes lifecycle watch ended") from None
                continue
            if isinstance(item, BaseException):
                raise ValueError("a Kubernetes lifecycle watch failed") from item
            pending.append(item)
            progressed = True
            while progressed:
                progressed = False
                for _observed_at, raw in list(pending):
                    obj = raw.get("object", {})
                    kind = obj.get("kind")
                    metadata = obj.get("metadata", {})
                    owner = self._owner(raw)
                    accepted = False
                    if kind == "RayJob":
                        labels = metadata.get("labels") or {}
                        accepted = (
                            metadata.get("name") == run["run_name"]
                            and labels.get("fleet.ai/run-id") == run["run_id"]
                        )
                        if accepted:
                            rayjob_uid = metadata.get("uid")
                            status = obj.get("status", {}).get("jobStatus")
                            terminal_job |= status == "SUCCEEDED"
                            failed_job |= status in {"FAILED", "STOPPED"}
                    elif kind in {"Workload", "RayCluster"} and rayjob_uid:
                        accepted = owner == ("RayJob", run["run_name"], rayjob_uid)
                        if accepted and kind == "RayCluster":
                            raycluster_identity = (metadata.get("name"), metadata.get("uid"))
                    elif kind == "Pod" and raycluster_identity:
                        accepted = owner == ("RayCluster", *raycluster_identity)
                        if accepted:
                            if not self._pod_projectable(raw):
                                pending.remove((_observed_at, raw))
                                progressed = True
                                continue
                            statuses = obj.get("status", {}).get("containerStatuses") or []
                            terminal_pod |= (
                                obj.get("status", {}).get("phase") == "Succeeded"
                                and any(
                                    row.get("state", {}).get("terminated", {}).get("exitCode") == 0
                                    and row.get("state", {}).get("terminated", {}).get("reason")
                                    == "Completed"
                                    for row in statuses
                                    if isinstance(row, dict)
                                )
                            )
                    if accepted:
                        recorded_at = max(time.time(), last_recorded_at)
                        events.record_event(
                            directory,
                            raw,
                            observed_at=recorded_at,
                            sequence=sequence,
                        )
                        sequence += 1
                        last_recorded_at = recorded_at
                        seen.add(str(kind))
                        pending.remove((_observed_at, raw))
                        progressed = True
            if failed_job:
                raise RuntimeError("observer RayJob reached terminal failure on dev")
            if terminal_job and terminal_pod and seen == {
                "RayJob",
                "Workload",
                "RayCluster",
                "Pod",
            }:
                return
        raise TimeoutError("observer watch did not see terminal RayJob and Pod success")

    def close(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def submit(args: argparse.Namespace) -> None:
    plan, request = _preflight_submission(args)
    token = os.environ.get("FLEET_API_KEY", "")
    watcher = WatchBuffer()
    observer.start_capture_intent(
        plan,
        request,
        namespace_uid=NAMESPACE_UID,
        started_at=time.time(),
        directory=args.watch_directory,
        kube_context=events.DEV_KUBE_CONTEXT,
    )
    try:
        watcher.start()
        with Jobs(token, base_url=API_URLS["dev"]) as jobs:
            try:
                submit_once_or_reconcile(
                    jobs,
                    request,
                    args.journal,
                    recovery_timeout_seconds=args.watch_timeout_seconds,
                )
            except Exception as error:
                if args.journal.is_file() and not args.journal.is_symlink():
                    watcher.preserve_ambiguous(
                        args.watch_directory,
                        request,
                        args.journal,
                        error,
                    )
                raise
        submission = observer.compile_submission_binding(
            plan_path=args.plan,
            request_path=args.request,
            source_commit=args.source_commit,
            submission_journal_path=args.journal,
            output=args.output,
        )
        observer.start_capture(
            plan,
            submission,
            intent_path=args.watch_directory / "CAPTURE_INTENT.json",
        )
        watcher.record(
            args.watch_directory,
            submission,
            timeout=args.watch_timeout_seconds,
        )
    finally:
        watcher.close()


def controller(args: argparse.Namespace) -> None:
    plan, submission = _json(args.plan), _json(args.submission)
    with Jobs(os.environ.get("FLEET_API_KEY", ""), base_url=API_URLS["dev"]) as jobs:
        status = jobs.status(submission["api"]["run_name"])
    observer.compile_controller(
        plan,
        submission,
        directory=args.watch_directory,
        api_status=status["status"],
        observed_at=time.time(),
        output=args.output,
    )


def release(args: argparse.Namespace) -> None:
    plan, submission = _json(args.plan), _json(args.submission)
    with Jobs(os.environ.get("FLEET_API_KEY", ""), base_url=API_URLS["dev"]) as jobs:
        observer.collect_release_query(
            plan,
            submission,
            controller_path=args.controller,
            jobs=jobs,
            output=args.query_output,
        )
    observer.compile_release(
        plan,
        submission,
        controller_path=args.controller,
        release_query_path=args.query_output,
        output=args.output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    for name in ("plan", "request", "journal", "output", "watch_directory"):
        submit_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    submit_parser.add_argument("--source-commit", required=True)
    submit_parser.add_argument(
        "--watch-timeout-seconds",
        type=int,
        default=43200,
    )
    submit_parser.set_defaults(function=submit)
    controller_parser = commands.add_parser("controller")
    for name in ("plan", "submission", "watch_directory", "output"):
        controller_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    controller_parser.set_defaults(function=controller)
    release_parser = commands.add_parser("release")
    for name in ("plan", "submission", "controller", "query_output", "output"):
        release_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    release_parser.set_defaults(function=release)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
