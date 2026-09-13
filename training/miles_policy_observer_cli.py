"""Single-POST operator rail for the dev3 Miles policy observer."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, Jobs

from . import miles_event_evidence as events
from . import miles_policy_observer as observer
from .miles_acceptance import NAMESPACE, NAMESPACE_UID, _json_snapshot

_RESOURCES = (
    "rayjobs.ray.io",
    "workloads.kueue.x-k8s.io",
    "rayclusters.ray.io",
    "pods",
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
            jobs.submit_once(request, args.journal)
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
