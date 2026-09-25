"""Create-once Fleet pass@4 CPU Jobs for a sealed, matched OpenCode protocol.

The evaluator image and checkpoint/route proof are external inputs, not claims
made by this launcher. ``preview`` is read-only (including Kubernetes server
dry-run); ``launch_once`` has one Job-create boundary and never retries it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from evals.fleet import validate_protocol

SCHEMA = "fleet_paired_launch_v1"
NAMESPACE = "fleet-train-jobs"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
ANNOTATION = "cyber-post-train.fleet.ai"
ARM_ENV = {
    "EVAL_ARM", "EVAL_MODEL_REVISION", "EVAL_WEIGHTS_SHA256",
    "EVAL_CHECKPOINT_SHA256", "EVAL_SERVED_ID", "EVAL_OUTPUT",
}


class LaunchError(ValueError):
    """A pre-create gate failed. An uncertain create must never be replayed."""


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _fields(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise LaunchError(f"{label} has missing or unknown fields")
    return value


def _families(plan: dict) -> None:
    roles = _fields(plan["family_roles"], {"train", "dev", "final"}, "family roles")
    seen: set[tuple[str, str]] = set()
    for role, rows in roles.items():
        if not isinstance(rows, list) or (role != "train" and not rows):
            raise LaunchError("frozen development and final family rosters are required")
        for row in rows:
            if not isinstance(row, list) or len(row) != 2 or any(
                not isinstance(x, str) or not x for x in row
            ):
                raise LaunchError("invalid reviewed family identity")
            pair = tuple(row)
            if pair in seen:
                raise LaunchError("train/development/final family overlap or duplicate")
            seen.add(pair)
    protocol = plan["protocol"]
    role = "final" if protocol["role"] == "final" else "dev"
    selected = {(x["application"], x["family_id"]) for x in protocol["tasks"]}
    if selected != {tuple(x) for x in roles[role]}:
        raise LaunchError("protocol tasks differ from frozen family roster")


def _identity(protocol: dict) -> str:
    """Ignore display-only study ID so renamed duplicate studies collide."""
    return digest({key: protocol[key] for key in (
        "role", "tasks", "common", "arms", "final_selection_sha256",
    )})


def _env(container: dict) -> dict[str, Any]:
    rows = container.get("env", [])
    if not isinstance(rows, list) or any(not isinstance(x, dict) or "name" not in x for x in rows):
        raise LaunchError("evaluator environment is invalid")
    # Kubernetes omits a deliberately empty ``value`` after server defaulting.
    result = {x["name"]: ({**x, "value": ""} if "value" not in x and "valueFrom" not in x else x)
              for x in rows}
    if len(result) != len(rows):
        raise LaunchError("duplicate evaluator environment name")
    return result


def _job(plan: dict, arm: str, identity: str) -> dict:
    protocol = plan["protocol"]
    job = plan["jobs"][arm]
    if not isinstance(job, dict) or job.get("apiVersion") != "batch/v1" or job.get("kind") != "Job":
        raise LaunchError("one batch/v1 Job per arm is required")
    meta, spec = job.get("metadata"), job.get("spec")
    if not isinstance(meta, dict) or not isinstance(spec, dict):
        raise LaunchError("Job metadata/spec missing")
    expected_name = f"chris-q38-fleet-{identity[7:19]}-{arm}"
    if meta.get("namespace") != NAMESPACE or meta.get("name") != expected_name:
        raise LaunchError("Job name/namespace differs from create-once identity")
    annotations = meta.get("annotations") or {}
    if any(annotations.get(key) != value for key, value in {
        "fleet.ai/failure-alerts": "off",
        f"{ANNOTATION}/protocol-sha256": protocol["sha256"],
        f"{ANNOTATION}/identity-sha256": identity,
        f"{ANNOTATION}/arm": arm,
    }.items()):
        raise LaunchError("root Job annotations do not bind alert opt-out and protocol")
    if (meta.get("labels") or {}).get(f"{ANNOTATION}/identity") != identity[7:55]:
        raise LaunchError("Job lacks its immutable scientific-identity selector")
    if spec.get("backoffLimit") != 0 or type(spec.get("activeDeadlineSeconds")) is not int or spec["activeDeadlineSeconds"] <= 0:
        raise LaunchError("Job must have zero retry and a bounded deadline")
    template = spec.get("template") or {}
    pod = template.get("spec") or {}
    if pod.get("priorityClassName") != "c1" or pod.get("restartPolicy") != "Never":
        raise LaunchError("Job must use c1 and never restart a failed Pod")
    containers = pod.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise LaunchError("one pinned evaluator container is required")
    if (not isinstance(containers[0].get("command"), list) or not containers[0]["command"]
            or not isinstance(containers[0].get("args"), list) or not containers[0]["args"]):
        raise LaunchError("evaluator entrypoint must be explicit and pinned")
    init_containers = pod.get("initContainers", [])
    if not isinstance(init_containers, list):
        raise LaunchError("initContainers must be a list")
    all_containers = containers + init_containers
    for container in all_containers:
        if not isinstance(container, dict) or not isinstance(container.get("image"), str) or not IMAGE.fullmatch(container["image"]):
            raise LaunchError("every Job image must be immutable")
        for resources in (container.get("resources") or {}).values():
            if not isinstance(resources, dict) or any("gpu" in key.lower() for key in resources):
                raise LaunchError("Fleet held-out evaluator must request zero GPUs")
    values = _env(containers[0])
    artifact = protocol["arms"][arm]
    wanted = {
        "EVAL_HARNESS": "opencode",
        "EVAL_PROTOCOL_SHA256": protocol["sha256"],
        "EVAL_ARM": arm,
        "EVAL_MODEL_REVISION": artifact["model_revision"],
        "EVAL_WEIGHTS_SHA256": artifact["weights_sha256"],
        "EVAL_CHECKPOINT_SHA256": artifact.get("checkpoint_sha256", ""),
        "EVAL_TASK_ROSTER_SHA256": digest(protocol["tasks"]),
        "EVAL_OUTPUT": plan["output_roots"][arm],
    }
    if any(values.get(key) != {"name": key, "value": value} for key, value in wanted.items()):
        raise LaunchError("evaluator environment differs from sealed protocol")
    if not isinstance(values.get("EVAL_SERVED_ID", {}).get("value"), str) or not values["EVAL_SERVED_ID"]["value"]:
        raise LaunchError("exact serving route ID is required")
    return job


def _treatment(job: dict) -> dict:
    """Only model/route/output env values may differ between paired Jobs."""
    spec = copy.deepcopy(job["spec"])
    pod = spec["template"]["spec"]
    for row in pod["containers"][0]["env"]:
        if row["name"] in ARM_ENV:
            row["value"] = "<arm>"
    return spec


def validate_plan(plan: dict) -> str:
    _fields(plan, {"schema", "harness", "protocol", "family_roles", "task_response_sha256",
                   "task_qualification_sha256", "readiness_sha256", "jobs", "output_roots"},
            "launch plan")
    if plan["schema"] != SCHEMA:
        raise LaunchError("unsupported launch schema")
    if plan["harness"] != "opencode":
        raise LaunchError("only matched OpenCode evaluations are supported")
    protocol = validate_protocol(plan["protocol"])
    _families(plan)
    versions = {x["task_version_id"] for x in protocol["tasks"]}
    proofs = _fields(plan["task_response_sha256"], versions, "task API digests")
    if not all(map(_sha, proofs.values())):
        raise LaunchError("exact-version Fleet task GET digests are required")
    qualification = _fields(plan["task_qualification_sha256"], versions,
                            "task qualification receipts")
    if not all(map(_sha, qualification.values())):
        raise LaunchError("independent current-runtime task qualification is required")
    readiness = _fields(plan["readiness_sha256"], {
        "candidate_export", "candidate_reload", "base_registration",
        "candidate_registration", "live_parity",
    }, "independent readiness receipts")
    if not all(map(_sha, readiness.values())):
        raise LaunchError("export, reload, registration and live parity gates are required")
    _fields(plan["jobs"], {"base", "candidate"}, "paired Jobs")
    roots = _fields(plan["output_roots"], {"base", "candidate"}, "output roots")
    if (roots["base"] == roots["candidate"] or any(
        not isinstance(x, str) or not x.startswith("/mnt/sfs/jobs/chris-")
        or ".." in PurePosixPath(x).parts or "//" in x
        for x in roots.values()
    )):
        raise LaunchError("two distinct project-owned immutable output roots are required")
    identity = _identity(protocol)
    jobs = {arm: _job(plan, arm, identity) for arm in ("base", "candidate")}
    if _treatment(jobs["base"]) != _treatment(jobs["candidate"]):
        raise LaunchError("paired evaluator Pods differ beyond the model intervention")
    return identity


def _fleet_get(path: str, query: dict[str, str] | None = None) -> dict:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise LaunchError("Fleet API credential unavailable")
    url = "https://orchestrator.fleetai.com" + path
    if query:
        url += "?" + urlencode(query)
    request = Request(url, headers={"Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def fleet_account_get() -> dict:
    return _fleet_get("/v1/account")


def fleet_task_get(task_key: str, version_id: str) -> dict:
    """Only exact GETs; never print or persist task prompts, flags or verifier code."""
    return _fleet_get("/v1/tasks/" + quote(task_key, safe=""), {"version_id": version_id})


def _stable_job(job: dict) -> dict:
    result = copy.deepcopy(job)
    result.pop("status", None)
    for key in ("creationTimestamp", "generation", "managedFields", "resourceVersion", "uid"):
        result.get("metadata", {}).pop(key, None)
    result.get("spec", {}).pop("selector", None)
    labels = result.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
    for key in ("batch.kubernetes.io/controller-uid", "controller-uid"):
        labels.pop(key, None)
    return result


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            (k in actual and _contains(actual[k], v)) or (k not in actual and v == "")
            for k, v in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _contains(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


class Cluster(Protocol):
    def jobs_for_identity(self, identity: str) -> list[dict]: ...
    def server_preview(self, job: dict) -> dict: ...
    def create_once(self, job: dict) -> dict: ...


def preview(plan: dict, arm: str, *, cluster: Cluster,
            account_get: Callable[[], dict],
            task_get: Callable[[str, str], dict],
            readiness_check: Callable[[dict, dict], dict[str, str]],
            qualification_check: Callable[[dict, dict], dict[str, str]],
            output_exists: Callable[[str], bool],
            capacity_check: Callable[[], bool]) -> str:
    """Return a stable server-rendered digest, or fail before paid work."""
    identity = validate_plan(plan)
    if arm not in ("base", "candidate"):
        raise LaunchError("unknown comparison arm")
    protocol = plan["protocol"]
    account = account_get()
    if account.get("team_id") != TEAM_ID or account.get("team_name") != "fleet":
        raise LaunchError("Fleet-team account identity is required")
    if readiness_check(protocol, plan["readiness_sha256"]) != plan["readiness_sha256"]:
        raise LaunchError("independent checkpoint/export/serving proof did not match")
    if qualification_check(protocol, plan["task_qualification_sha256"]) != plan["task_qualification_sha256"]:
        raise LaunchError("current-runtime task qualification proof did not match")
    for task in protocol["tasks"]:
        live = task_get(task["task_key"], task["task_version_id"])
        verifier = live.get("verifier") or {}
        metadata = live.get("metadata") or {}
        if (live.get("team_id") != TEAM_ID
                or live.get("key") != task["task_key"]
                or live.get("eval_task_version_id") != task["task_version_id"]
                or live.get("environment_version_id") != task["environment_version_id"]
                or live.get("data_version") != task["data_version"]
                or "sha256:" + str(verifier.get("sha256")) != task["verifier_sha256"]
                or not verifier.get("verifier_version_id")
                or not (metadata.get("runtime_seed_manifest") or {}).get("content_sha256")
                or metadata.get("projection_id") != "blackbox_ctf_v1"
                or live.get("task_lifecycle_status") != "production"
                or not live.get("seed_config")
                or digest(live) != plan["task_response_sha256"][task["task_version_id"]]):
            raise LaunchError("live exact-version task/runtime/verifier binding changed")
    existing = cluster.jobs_for_identity(identity)
    names = {plan["jobs"][x]["metadata"]["name"]: x for x in ("base", "candidate")}
    for job in existing:
        meta = job.get("metadata") or {}
        old_arm = names.get(meta.get("name"))
        annotations = meta.get("annotations") or {}
        if old_arm is None or annotations.get(f"{ANNOTATION}/protocol-sha256") != protocol["sha256"]:
            raise LaunchError("conflicting scientific-identity Job exists")
        if old_arm == arm:
            raise LaunchError("exact arm already exists; never replay")
    if output_exists(plan["output_roots"][arm]):
        raise LaunchError("evaluation output already exists; never overwrite")
    if capacity_check() is not True:
        raise LaunchError("project queue/capacity gate did not pass")
    expected = plan["jobs"][arm]
    rendered = []
    for _ in range(2):
        job = cluster.server_preview(expected)
        if not _contains(_stable_job(job), _stable_job(expected)):
            raise LaunchError("server-rendered Job changed the sealed evaluator")
        # Validate the *rendered root*, not only the request's Pod template.
        rendered_plan = {**plan, "jobs": {**plan["jobs"], arm: job}}
        _job(rendered_plan, arm, identity)
        rendered.append(digest(_stable_job(job)))
    if rendered[0] != rendered[1]:
        raise LaunchError("two server dry-runs disagreed")
    return rendered[0]


def launch_once(plan: dict, arm: str, *, cluster: Cluster,
                account_get: Callable[[], dict],
                task_get: Callable[[str, str], dict],
                readiness_check: Callable[[dict, dict], dict[str, str]],
                qualification_check: Callable[[dict, dict], dict[str, str]],
                output_exists: Callable[[str], bool],
                capacity_check: Callable[[], bool],
                capacity_reserve: Callable[[str, str], str],
                capacity_release: Callable[[str], None], journal_dir: Path) -> dict:
    """Create exactly one arm after duplicate/digest checks; uncertain = stop."""
    identity = validate_plan(plan)
    journal = journal_dir / f"{identity[7:]}-{arm}.jsonl"
    if journal.exists() or journal.is_symlink():
        raise LaunchError("create intent already exists; reconcile, never retry")
    first = preview(plan, arm, cluster=cluster, account_get=account_get, task_get=task_get,
                    readiness_check=readiness_check, qualification_check=qualification_check,
                    output_exists=output_exists,
                    capacity_check=capacity_check)
    second = preview(plan, arm, cluster=cluster, account_get=account_get, task_get=task_get,
                     readiness_check=readiness_check, qualification_check=qualification_check,
                     output_exists=output_exists,
                     capacity_check=capacity_check)
    if first != second:
        raise LaunchError("pre-create previews changed")
    # This provider must atomically count live jobs plus *all* outstanding
    # project reservations. A read-only capacity snapshot cannot prevent races.
    lease = capacity_reserve(identity, arm)
    if not isinstance(lease, str) or not lease:
        raise LaunchError("shared capacity reservation was not acquired")
    if not journal_dir.is_dir() or journal_dir.is_symlink():
        capacity_release(lease)
        raise LaunchError("durable journal directory is unavailable")
    try:
        fd = os.open(journal, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        capacity_release(lease)
        raise LaunchError("concurrent create intent already exists") from exc
    except OSError as exc:
        capacity_release(lease)
        raise LaunchError("could not create durable create intent") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"state": "CREATE_INTENT_DO_NOT_RETRY", "plan_sha256": digest(plan),
                       "arm": arm, "server_preview_sha256": first,
                       "capacity_lease_sha256": digest(lease)}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        capacity_release(lease)
        raise LaunchError("create intent could not be durably written") from exc
    try:
        dir_fd = os.open(journal_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        capacity_release(lease)
        raise LaunchError("create intent could not be durably synchronized") from exc
    try:
        created = cluster.create_once(plan["jobs"][arm])
        meta = created.get("metadata") or {}
        if meta.get("name") != plan["jobs"][arm]["metadata"]["name"] or not meta.get("uid"):
            raise LaunchError("create response lacks exact Job identity; reconcile")
        _job({**plan, "jobs": {**plan["jobs"], arm: created}}, arm, identity)
        result = {"job_name": meta["name"], "job_uid": meta["uid"],
                  "arm": arm, "protocol_sha256": plan["protocol"]["sha256"]}
        with journal.open("a") as stream:
            json.dump({"state": "CREATED", **result}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
    except Exception as exc:
        raise LaunchError("Job create response uncertain; inspect exact name/UID and never retry") from exc


class Kubectl:
    """Explicit-context Kubernetes boundary. No context-default submission."""

    def __init__(self, context: str):
        if not context:
            raise LaunchError("explicit Kubernetes context required")
        self.context = context

    def _run(self, args: list[str], payload: dict | None = None) -> dict:
        result = subprocess.run(["kubectl", "--context", self.context, *args],
                                input=None if payload is None else json.dumps(payload),
                                text=True, capture_output=True, check=False, timeout=60)
        if result.returncode:
            raise LaunchError("Kubernetes request failed; inspect cluster access separately")
        return json.loads(result.stdout)

    def jobs_for_identity(self, identity: str) -> list[dict]:
        value = self._run(["-n", NAMESPACE, "get", "jobs", "-l",
                           f"{ANNOTATION}/identity={identity[7:55]}", "-o", "json"])
        return value.get("items", [])

    def server_preview(self, job: dict) -> dict:
        return self._run(["-n", NAMESPACE, "create", "--dry-run=server", "-f", "-", "-o", "json"], job)

    def create_once(self, job: dict) -> dict:
        return self._run(["-n", NAMESPACE, "create", "-f", "-", "-o", "json"], job)
