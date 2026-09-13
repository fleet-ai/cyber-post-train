"""Pre-watched operator rail for dev3 Miles HF export and reload.

Export is one zero-GPU Kubernetes batch Job; reload is one one-GPU Jobs API
RayJob.  Both paths open LIST-resource-version watches before their only create
and retain exact-UID deletion events through TTL-zero cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, Jobs

from . import miles_event_evidence as events
from . import miles_hf_export_job as job
from .miles_acceptance import _time
from .miles_policy_observer_cli import WatchBuffer, submit_once_or_reconcile

WATCH_DIRECTORY = "HF_UID_EVENTS"
PREFLIGHT_FILE = "PREFLIGHT.json"
SUBMISSION_JOURNAL = "SUBMISSION.jsonl"
SUBMISSION_BINDING = "SUBMITTED.json"
EXPORT_RESOURCES = ("jobs.batch", "workloads.kueue.x-k8s.io", "pods")


class KubernetesJobWatchBuffer(WatchBuffer):
    """Reuse the generic gap-free watches for one direct batch Job owner chain."""

    def __init__(self) -> None:
        super().__init__(EXPORT_RESOURCES)

    def record(
        self,
        directory: Path,
        submission: dict[str, Any],
        timeout: int = 43200,
        *,
        require_deletions: bool = True,
    ) -> None:
        if require_deletions is not True:
            raise ValueError("direct Kubernetes Job watches must retain every deletion")
        run = submission["api"]
        job_uid: str | None = None
        pending: list[tuple[float, dict[str, Any]]] = []
        terminal_job = terminal_pod = failed_job = False
        expected = {"Job", "Workload", "Pod"}
        seen: set[str] = set()
        deleted: set[str] = set()
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
                for buffered in list(pending):
                    _received_at, raw = buffered
                    obj = raw.get("object", {})
                    metadata = obj.get("metadata", {})
                    kind = obj.get("kind")
                    accepted = False
                    if kind == "Job":
                        accepted = (
                            metadata.get("name") == run["run_name"]
                            and metadata.get("uid") == run["run_id"]
                        )
                        if accepted:
                            job_uid = str(metadata["uid"])
                            conditions = obj.get("status", {}).get("conditions") or []
                            terminal_job |= any(
                                row.get("type") == "Complete"
                                and str(row.get("status")).lower() == "true"
                                for row in conditions
                                if isinstance(row, dict)
                            )
                            failed_job |= any(
                                row.get("type") == "Failed"
                                and str(row.get("status")).lower() == "true"
                                for row in conditions
                                if isinstance(row, dict)
                            )
                    elif kind in {"Workload", "Pod"} and job_uid:
                        accepted = self._owner(raw) == ("Job", run["run_name"], job_uid)
                        if accepted and kind == "Pod":
                            if not self._pod_projectable(raw):
                                pending.remove(buffered)
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
                        if raw.get("type") == "DELETED":
                            deleted.add(str(kind))
                        pending.remove(buffered)
                        progressed = True
            if failed_job:
                raise RuntimeError("Miles HF export Job reached terminal failure on dev")
            if terminal_job and terminal_pod and seen == expected and deleted == expected:
                return
        raise TimeoutError("Miles HF export watch missed terminal exact-UID deletions")


def _json(path: Path) -> dict[str, Any]:
    value, _ = job._snapshot(path)
    return value


def _paths(directory: Path) -> dict[str, Path]:
    root = directory.resolve()
    if job._sfs(str(root), "HF prepared evidence root") != str(root):
        raise ValueError("HF prepared evidence root must be durable SFS")
    return {
        "root": root,
        "plan": root / "plan.json",
        "request": root / "request.json",
        "preflight": root / PREFLIGHT_FILE,
        "journal": root / SUBMISSION_JOURNAL,
        "submission": root / SUBMISSION_BINDING,
        "watch": root / WATCH_DIRECTORY,
    }


def _preflight(
    directory: Path,
    source_commit: str,
    *,
    require_unsubmitted: bool,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Reopen every deterministic rail before the single mutating operation."""

    paths = _paths(directory)
    plan = _json(paths["plan"])
    request = _json(paths["request"])
    proof = _json(paths["preflight"])
    job.validate_plan(plan, check_files=True)
    if request != job.job_request(plan):
        raise ValueError("HF prepared request differs from its exact plan")
    job.validate_preflight(plan, request, proof)
    job._request_projection(
        plan,
        request,
        source_commit=source_commit,
        repo_root=Path(__file__).resolve().parents[1],
    )
    if require_unsubmitted:
        if any(
            path.exists() or path.is_symlink()
            for path in (paths["journal"], paths["submission"], paths["watch"])
        ):
            raise FileExistsError("HF submission/watch evidence already exists; reconcile it")
        output_root = Path(plan["output_root"])
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError("HF runtime output root already exists; never repeat POST")
    return paths, plan, request


def _submission(
    paths: dict[str, Path], plan: dict[str, Any], *, check_files: bool
) -> dict[str, Any]:
    value = _json(paths["submission"])
    job.validate_submission_binding(value, plan, check_files=check_files)
    return value


def _terminal_api(jobs: Jobs, plan: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    """Read and minimize one exact terminal Jobs API response."""

    run = jobs.status(submission["api"]["run_name"])
    expected_fields = {
        "name",
        "job_id",
        "run_dir",
        "status",
        "created_at",
        "finished_at",
    }
    if (
        set(run) != expected_fields
        or run.get("name") != submission["api"]["run_name"]
        or run.get("job_id") != submission["api"]["run_id"]
        or run.get("run_dir") != plan["output_root"]
        or run.get("status") != "SUCCEEDED"
        or run.get("created_at") != submission["submitted_at"]
    ):
        raise ValueError("HF Jobs API terminal response differs from the exact submission")
    created = _time(run.get("created_at"), "HF Jobs API creation time")
    finished = _time(run.get("finished_at"), "HF Jobs API finish time")
    if finished < created:
        raise ValueError("HF Jobs API terminal response predates submission")
    return run


def _kubectl_json(argv: list[str], *, stdin: bytes | None = None) -> dict[str, Any]:
    process = subprocess.run(
        ["kubectl", "--context", job.DEV_KUBE_CONTEXT, *argv],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode:
        raise ValueError("exact dev Kubernetes request failed")
    try:
        value = json.loads(process.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exact dev Kubernetes request returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("exact dev Kubernetes request returned a non-object")
    return value


def _manifest_projection(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata") or {}
    spec = value.get("spec") or {}
    template = spec.get("template") or {}
    pod = template.get("spec") or {}
    containers = pod.get("containers") or []
    if len(containers) != 1 or not isinstance(containers[0], dict):
        raise ValueError("Kubernetes Job has no unique export container")
    container = containers[0]
    label_names = {
        "kueue.x-k8s.io/queue-name",
        "kueue.x-k8s.io/priority-class",
        "cyber-post-train.fleet.ai/owner",
        "cyber-post-train.fleet.ai/component",
    }
    annotation_names = {
        "cyber-post-train.fleet.ai/source-plan-sha256",
        "cyber-post-train.fleet.ai/runtime-source-sha256",
        "cyber-post-train.fleet.ai/runtime-bundle-sha256",
    }
    labels = metadata.get("labels") or {}
    template_labels = (template.get("metadata") or {}).get("labels") or {}
    annotations = metadata.get("annotations") or {}
    return {
        "apiVersion": value.get("apiVersion"),
        "kind": value.get("kind"),
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "labels": {name: labels.get(name) for name in sorted(label_names)},
        "annotations": {name: annotations.get(name) for name in sorted(annotation_names)},
        "suspend": spec.get("suspend"),
        "backoffLimit": spec.get("backoffLimit"),
        "completions": spec.get("completions"),
        "parallelism": spec.get("parallelism"),
        "activeDeadlineSeconds": spec.get("activeDeadlineSeconds"),
        "ttlSecondsAfterFinished": spec.get("ttlSecondsAfterFinished"),
        "template_labels": {
            name: template_labels.get(name) for name in sorted(label_names)
        },
        "restartPolicy": pod.get("restartPolicy"),
        "automountServiceAccountToken": pod.get("automountServiceAccountToken"),
        "priorityClassName": pod.get("priorityClassName"),
        "nodeSelector": pod.get("nodeSelector"),
        "tolerations": pod.get("tolerations"),
        "volumes": pod.get("volumes"),
        "container": {
            key: container.get(key)
            for key in (
                "name",
                "image",
                "imagePullPolicy",
                "command",
                "env",
                "resources",
                "volumeMounts",
                "terminationMessagePolicy",
            )
        },
    }


def _server_dry_run(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    observed = _kubectl_json(
        ["create", "--dry-run=server", "--filename=-", "--output=json"],
        stdin=payload,
    )
    if _manifest_projection(observed) != _manifest_projection(request):
        raise ValueError("Kubernetes server dry-run changed the immutable Job manifest")
    return hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _append(fd: int, value: dict[str, Any]) -> None:
    os.write(fd, json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    os.fsync(fd)


def _validated_created_job(
    response: dict[str, Any], plan: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Minimize only an exact immutable Job create/readback response."""

    metadata = response.get("metadata") or {}
    if (
        _manifest_projection(response) != _manifest_projection(request)
        or response.get("apiVersion") != "batch/v1"
        or response.get("kind") != "Job"
        or metadata.get("name") != plan["run_name"]
        or metadata.get("namespace") != job.DEV_NAMESPACE
        or job._UUID.fullmatch(str(metadata.get("uid", ""))) is None
        or not isinstance(metadata.get("resourceVersion"), str)
        or not metadata["resourceVersion"]
    ):
        raise ValueError("created Kubernetes Job differs from the exact immutable request")
    _time(metadata.get("creationTimestamp"), "Kubernetes Job creation time")
    return {
        "state": "KUBERNETES_POST_RESPONSE",
        "api_version": response["apiVersion"],
        "kind": response["kind"],
        "name": metadata["name"],
        "namespace": metadata["namespace"],
        "uid": metadata["uid"],
        "resource_version": metadata["resourceVersion"],
        "creation_timestamp": metadata["creationTimestamp"],
    }


def _direct_create_once(
    paths: dict[str, Path],
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    server_dry_run_sha256: str,
    recovery_timeout_seconds: int,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(paths["journal"], flags, 0o600)
    try:
        _append(
            fd,
            {
                "state": "KUBERNETES_POST_INTENT_DO_NOT_RETRY",
                "api_base_url": job.DEV_KUBERNETES_API,
                "kube_context": job.DEV_KUBE_CONTEXT,
                "namespace": job.DEV_NAMESPACE,
                "namespace_uid": job.DEV_NAMESPACE_UID,
                "request_sha256": job.digest(request),
                "server_dry_run_sha256": server_dry_run_sha256,
                "name": plan["run_name"],
                "image": job.miles.IMAGE,
                "gpus": 0,
            },
        )
        payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        try:
            response = _kubectl_json(
                ["create", "--filename=-", "--output=json"],
                stdin=payload,
            )
        except Exception as create_error:
            # The server may have persisted the Job before the client lost its
            # response. Reconcile by exact-name GET while pre-opened watches stay
            # live; never issue a second create.
            deadline = time.monotonic() + recovery_timeout_seconds
            while True:
                try:
                    response = _kubectl_json(
                        [
                            "get",
                            "jobs.batch",
                            plan["run_name"],
                            "--namespace",
                            job.DEV_NAMESPACE,
                            "--output=json",
                        ]
                    )
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Kubernetes create response remained ambiguous through watch deadline; "
                            "the durable intent may own a live Job and must not be retried"
                        ) from create_error
                    time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        _append(fd, _validated_created_job(response, plan, request))
        job._kubernetes_submission_journal(paths["journal"], plan, request)
    finally:
        os.close(fd)


def _kubectl_exact_uid_absent(resource: str, name: str, uid: str) -> bool:
    process = subprocess.run(
        [
            "kubectl",
            "--context",
            job.DEV_KUBE_CONTEXT,
            "get",
            resource,
            name,
            "--namespace",
            job.DEV_NAMESPACE,
            "--ignore-not-found",
            "--output=json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode:
        raise ValueError("exact dev Kubernetes release query failed")
    if not process.stdout.strip():
        return True
    try:
        value = json.loads(process.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exact dev Kubernetes release query returned invalid JSON") from error
    return value.get("metadata", {}).get("uid") != uid


def _export_absence(controller_value: dict[str, Any]) -> dict[str, Any]:
    namespace = _kubectl_json(["get", "namespace", job.DEV_NAMESPACE, "--output=json"])
    if namespace.get("metadata", {}).get("uid") != job.DEV_NAMESPACE_UID:
        raise ValueError("Kubernetes release query is not in the exact dev namespace")
    rows = (
        ("jobs.batch", controller_value["job_name"], controller_value["job_uid"]),
        (
            "workloads.kueue.x-k8s.io",
            controller_value["workload_name"],
            controller_value["workload_uid"],
        ),
        ("pods", controller_value["pods"][0]["name"], controller_value["pods"][0]["uid"]),
    )
    if not all(_kubectl_exact_uid_absent(*row) for row in rows):
        raise ValueError("an exact Miles HF export Kubernetes UID is still present")
    return {
        "job_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
    }


def _reload_absence(controller_value: dict[str, Any]) -> dict[str, Any]:
    namespace = _kubectl_json(["get", "namespace", job.DEV_NAMESPACE, "--output=json"])
    if namespace.get("metadata", {}).get("uid") != job.DEV_NAMESPACE_UID:
        raise ValueError("Kubernetes release query is not in the exact dev namespace")
    rows = (
        ("rayjobs.ray.io", controller_value["rayjob_name"], controller_value["rayjob_uid"]),
        (
            "workloads.kueue.x-k8s.io",
            controller_value["workload_name"],
            controller_value["workload_uid"],
        ),
        (
            "rayclusters.ray.io",
            controller_value["raycluster_name"],
            controller_value["raycluster_uid"],
        ),
        ("pods", controller_value["pods"][0]["name"], controller_value["pods"][0]["uid"]),
    )
    if not all(_kubectl_exact_uid_absent(*row) for row in rows):
        raise ValueError("an exact Miles HF reload Kubernetes UID is still present")
    return {
        "raycluster_present": False,
        "rayjob_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
    }


def submit(directory: Path, source_commit: str, *, watch_timeout_seconds: int) -> dict[str, Any]:
    """Open all watches, perform exactly one dev POST, and retain all UID deletions."""

    if not 1800 <= watch_timeout_seconds <= 43200:
        raise ValueError("watch timeout must be between 30 minutes and 12 hours")
    paths, plan, request = _preflight(directory, source_commit, require_unsubmitted=True)
    if plan["stage"] == "reload" and not os.environ.get("FLEET_API_KEY"):
        raise ValueError("FLEET_API_KEY is required for the one-GPU reload")
    server_dry_run_sha256 = _server_dry_run(request) if plan["stage"] == "export" else None
    watcher = KubernetesJobWatchBuffer() if plan["stage"] == "export" else WatchBuffer()
    try:
        watcher.start()
        started_at = time.time()
        if plan["stage"] == "export":
            assert server_dry_run_sha256 is not None
            _direct_create_once(
                paths,
                plan,
                request,
                server_dry_run_sha256=server_dry_run_sha256,
                recovery_timeout_seconds=watch_timeout_seconds,
            )
        else:
            with Jobs(os.environ["FLEET_API_KEY"], base_url=API_URLS["dev"]) as jobs:
                try:
                    submit_once_or_reconcile(
                        jobs,
                        request,
                        paths["journal"],
                        recovery_timeout_seconds=watch_timeout_seconds,
                    )
                except Exception as error:
                    if paths["journal"].is_file() and not paths["journal"].is_symlink():
                        watcher.preserve_ambiguous(
                            paths["watch"],
                            request,
                            paths["journal"],
                            error,
                        )
                    raise
        submission = job.compile_submission_binding(
            plan_path=paths["plan"],
            request_path=paths["request"],
            submission_journal_path=paths["journal"],
            source_commit=source_commit,
            output=paths["submission"],
        )
        events.start_capture(
            plan,
            submission,
            namespace_uid=job.DEV_NAMESPACE_UID,
            started_at=started_at,
            directory=paths["watch"],
            kube_context=job.DEV_KUBE_CONTEXT,
        )
        watcher.record(
            paths["watch"],
            submission,
            timeout=watch_timeout_seconds,
            require_deletions=True,
        )
    finally:
        watcher.close()
    return {
        "status": "submitted_and_uid_deletions_recorded",
        "stage": plan["stage"],
        "api_run_id": submission["api"]["run_id"],
        "api_run_name": submission["api"]["run_name"],
        "submission_binding_sha256": submission["sha256"],
        "watch_directory": str(paths["watch"]),
        "controller_compiled": False,
        "release_compiled": False,
    }


def controller(directory: Path) -> dict[str, Any]:
    """Compile the exact terminal API/controller projection from the UID journal."""

    paths = _paths(directory)
    plan = _json(paths["plan"])
    job.validate_plan(plan, check_files=True)
    submission = _submission(paths, plan, check_files=True)
    if plan["stage"] == "export":
        api_status = "SUCCEEDED"
    else:
        if not os.environ.get("FLEET_API_KEY"):
            raise ValueError("FLEET_API_KEY is required")
        with Jobs(os.environ["FLEET_API_KEY"], base_url=API_URLS["dev"]) as jobs:
            api_status = _terminal_api(jobs, plan, submission)["status"]
    output = Path(plan["output_root"]) / (
        "HF_EXPORT_CONTROLLER_TERMINAL.json"
        if plan["stage"] == "export"
        else "HF_RELOAD_CONTROLLER_TERMINAL.json"
    )
    value = events.compile_controller(
        plan,
        submission,
        directory=paths["watch"],
        api_status=api_status,
        observed_at=time.time(),
        output=output,
    )
    return {"status": value["status"], "path": str(output), "sha256": value["sha256"]}


def release(directory: Path) -> dict[str, Any]:
    """Compile release only from durable deletion events for all exact UIDs."""

    paths = _paths(directory)
    plan = _json(paths["plan"])
    job.validate_plan(plan, check_files=True)
    submission = _submission(paths, plan, check_files=True)
    root = Path(plan["output_root"])
    if plan["stage"] == "export":
        controller_path = root / "HF_EXPORT_CONTROLLER_TERMINAL.json"
        output = root / "HF_EXPORT_RELEASE.json"
        controller_value = _json(controller_path)
        absence = _export_absence(controller_value)
    else:
        if not os.environ.get("FLEET_API_KEY"):
            raise ValueError("FLEET_API_KEY is required")
        with Jobs(os.environ["FLEET_API_KEY"], base_url=API_URLS["dev"]) as jobs:
            _terminal_api(jobs, plan, submission)
        controller_path = root / "HF_RELOAD_CONTROLLER_TERMINAL.json"
        output = root / "HF_RELOAD_RELEASE.json"
        absence = _reload_absence(_json(controller_path))
    value = events.compile_release(
        plan,
        submission,
        controller_path=controller_path,
        absence=absence,
        observed_at=time.time(),
        output=output,
    )
    return {"status": value["status"], "path": str(output), "sha256": value["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--directory", type=Path, required=True)
    submit_parser.add_argument("--source-commit", required=True)
    submit_parser.add_argument("--watch-timeout-seconds", type=int, default=43200)
    for name in ("controller", "release"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "submit":
        result = submit(
            args.directory,
            args.source_commit,
            watch_timeout_seconds=args.watch_timeout_seconds,
        )
    elif args.command == "controller":
        result = controller(args.directory)
    else:
        result = release(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
