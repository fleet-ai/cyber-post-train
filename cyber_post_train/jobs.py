"""Small generic Jobs API client shared by training backends.

This is a transport/resource boundary, not a model-qualification certificate.
Callers validate/stage their scientific inputs and check the live ownership
budget before submitting. The journal must be shared by every submitter for an
experiment: the server generates a new run ID on every POST, not an idempotency
key. An uncertain POST is recorded and must be reconciled, never replayed.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import fcntl
import gzip
import hashlib
import json
import os
import re
import shlex
import stat
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
import yaml

API_URL = "https://api.ft.flt.build"
API_URLS = {
    "dev": "https://api.ft.dev.flt.build",
    "prod": API_URL,
}
FAILURE_ALERT_REQUEST_FIELD = "failureAlerts"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
FAILURE_ALERT_OFF = "off"
PRIVILEGED_WHOLE_NODE_WARNING = (
    "privileged: the GPU containers run with full device access and seccomp Unconfined. "
    "The schema allows it only on a whole-node pod, so the blast radius is this job -- "
    "but a compromised or buggy image can now reset the node's GPUs and read anything "
    "mounted on it"
)
JOBS_RUN_FIELDS = {
    "image",
    "job_id",
    "message",
    "name",
    "priority_class",
    "priority_reason",
    "queue_priority_class",
    "requeueIfPreempted",
    "run_dir",
    "status",
    "submitted_by",
    "submitted_by_profile_id",
}


class JobsError(ValueError):
    pass


def plan_api_target(plan: dict | None) -> tuple[str, str]:
    """Return the immutable Jobs API route carried by a prepared plan.

    Older plans predate explicit cluster routing and remain production-bound.
    Once either routing field is present, both are mandatory and must match the
    closed target table.  There is deliberately no environment or CLI override:
    changing cluster is a new plan, request digest, preflight and launch.
    """
    if plan is None:
        return "prod", API_URLS["prod"]
    if not isinstance(plan, dict):
        raise JobsError("prepared plan must be an object")
    execution = plan.get("execution", {})
    if not isinstance(execution, dict):
        raise JobsError("prepared plan execution binding is malformed")
    target = execution.get("cluster_target")
    base_url = execution.get("jobs_api_base_url")
    if target is None and base_url is None:
        return "prod", API_URLS["prod"]
    if target not in API_URLS or base_url != API_URLS[target]:
        raise JobsError("prepared plan Jobs API target is incomplete or mismatched")
    return target, base_url


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def quantity(value: Any) -> Decimal:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(m|Ki|Mi|Gi|Ti)?", str(value))
    if not match:
        raise JobsError("unsupported resource quantity")
    factors = {
        None: 1,
        "m": Decimal(".001"),
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
    }
    return Decimal(match[1]) * factors[match[2]]


def canonical_gzip(payload: bytes) -> bytes:
    """Python 3.12 delegates mtime=0 headers to zlib: macOS=19, Linux=3.

    Normalize the descriptive OS byte so CPU validation can reproduce a request
    prepared on the desktop. This does not change the decompressed payload.
    """
    blob = gzip.compress(payload, mtime=0)
    return blob[:9] + b"\xff" + blob[10:]


def bundled_request(
    request: dict,
    files: dict[str, str],
    module: str,
    argv: list[str],
    *,
    transport_split_threshold: int = 120000,
    transport_chunk_size: int = 48000,
) -> dict:
    """Embed small, public runtime inputs in a create-once Jobs API entrypoint."""
    if module.replace(".", "/") + ".py" not in files or not files:
        raise JobsError("entry module is absent from runtime bundle")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise JobsError("runtime bundle path escapes its directory")
    if any(not isinstance(t, str) for t in [*files.values(), *argv]):
        raise JobsError("runtime bundle must contain text")
    blob = canonical_gzip(
        json.dumps({"files": files, "module": module, "argv": argv}, sort_keys=True).encode()
    )
    encoded = base64.b64encode(blob).decode()
    if len(encoded) > 512000 or any(
        key.startswith("CYBER_RUNTIME_BUNDLE") for key in request.get("env", {})
    ):
        raise JobsError("runtime bundle is too large or overrides reserved transport fields")
    if not (1 <= transport_chunk_size <= transport_split_threshold <= 120000):
        raise JobsError("runtime bundle transport limits are invalid")
    transport = {"CYBER_RUNTIME_BUNDLE": encoded}
    expression = "os.environ.pop('CYBER_RUNTIME_BUNDLE')"
    if len(encoded) > transport_split_threshold:
        # Linux limits EACH argument/env value, even when total ARG_MAX is free.
        parts = [
            encoded[i : i + transport_chunk_size]
            for i in range(0, len(encoded), transport_chunk_size)
        ]
        transport = {f"CYBER_RUNTIME_BUNDLE_{i}": part for i, part in enumerate(parts)}
        expression = (
            f"''.join(os.environ.pop('CYBER_RUNTIME_BUNDLE_'+str(i)) for i in range({len(parts)}))"
        )
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        f"b=base64.b64decode({expression},validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={hashlib.sha256(blob).hexdigest()!r};"
        "v=json.loads(gzip.decompress(b));"
        "p=pathlib.Path(os.environ['RUN_DIR'])/'.runtime';p.mkdir(parents=True,mode=0o700);"
        "[((p/n).parent.mkdir(parents=True,exist_ok=True),(p/n).write_text(t)) "
        "for n,t in v['files'].items()];"
        "os.chdir(p);sys.path.insert(0,str(p));importlib.invalidate_caches();"
        "sys.argv=[v['module']]+v['argv'];"
        "runpy.run_module(v['module'],run_name='__main__',alter_sys=True)"
    )
    result = {
        **request,
        "command": "python -c " + shlex.quote(bootstrap),
        "env": {**request.get("env", {}), **transport},
    }
    validate_request(result)
    return result


def validate_request(config: dict) -> None:
    """Check our reproducible GPU-job subset before any network request."""
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", config.get("name", "")):
        raise JobsError("name must be a DNS label of at most 31 characters")
    if config["name"] == "ft-run":
        raise JobsError("ft-run is a reserved name")
    if not isinstance(config.get("title"), str) or not config["title"].strip():
        raise JobsError("a nonempty title is required for duplicate reconciliation")
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", config.get("image", "")):
        raise JobsError("image must be pinned by immutable digest")
    if not isinstance(config.get("command"), str) or not config["command"].strip():
        raise JobsError("command is required")
    root = PurePosixPath(config.get("run_dir", ""))
    if (
        root.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(root.parts) < 5
        or ".." in root.parts
        or str(root) != config["run_dir"]
    ):
        raise JobsError("run_dir must be a canonical per-run path under /mnt/sfs/jobs/")
    # Per-request ceiling, not an experiment-wide admission controller. Operators
    # must also count other active allocations before submitting a new request.
    for key, upper in (("workers", 8), ("gpus_per_worker", 8)):
        if type(config.get(key)) is not int or not 1 <= config[key] <= upper:
            raise JobsError(f"{key} must be a positive integer no greater than {upper}")
    if config.get("priority_class") not in {"c1", "c2"}:
        raise JobsError(
            "use c1 (normal/high training) or c2 (backfill); c0 needs separate approval"
        )
    if config.get("queue_priority_class") is not None:
        raise JobsError("queue priority is derived by the platform; omit queue_priority_class")
    if config.get("requeueIfPreempted") is not False:
        raise JobsError("automatic requeue must be disabled; resume is an explicit decision")
    if config.get(FAILURE_ALERT_REQUEST_FIELD) is not False:
        raise JobsError("failed-job alerts must be explicitly disabled for every project job")
    if type(config.get("privileged", False)) is not bool:
        raise JobsError("privileged must be a boolean")
    if config.get("privileged") and config["gpus_per_worker"] != 8:
        raise JobsError("privileged jobs must own every GPU on each node")
    if config.get("models") or config.get("entrypoint_wrapper"):
        raise JobsError("stage weights off-GPU; bind the complete entrypoint explicitly")
    env = config.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(v, str) for v in env.values()):
        raise JobsError("env must contain string values")
    if any(len(str(k).encode()) + len(v.encode()) + 2 > 131072 for k, v in env.items()):
        raise JobsError("one environment value exceeds the Linux process-start limit")
    if any(
        not isinstance(k, str)
        or re.search(r"(?:^|_)(?:TOKEN|PASSWORD|CREDENTIALS|SECRET|API_KEY|ACCESS_KEY)(?:_|$)", k)
        for k in env
    ):
        raise JobsError("credentials belong in cluster Secrets, not the saved request env")
    for key in ("secrets", "image_pull_secrets"):
        values = config.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(v, str) or not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?", v)
            for v in values
        ):
            raise JobsError(f"{key} must contain Kubernetes Secret names, not values")
    resources = config.get("resources", {})
    if set(resources) != {"cpu_request", "cpu_limit", "memory_request", "memory_limit"}:
        raise JobsError("explicit CPU/memory requests and limits are required")
    for kind in ("cpu", "memory"):
        if not 0 < quantity(resources[f"{kind}_request"]) <= quantity(resources[f"{kind}_limit"]):
            raise JobsError("resource requests must be positive and no greater than limits")


def validate_preview(config: dict, preview: dict) -> dict:
    validate_request(config)
    privileged_whole_node = (
        config.get("privileged") is True
        and config.get("workers") == 1
        and config.get("gpus_per_worker") == 8
    )
    if privileged_whole_node:
        if preview.get("errors", object()) is not None or preview.get("warnings") != [
            PRIVILEGED_WHOLE_NODE_WARNING
        ]:
            raise JobsError("preview reported an unreviewed whole-node warning policy")
    elif preview.get("errors") or preview.get("warnings"):
        raise JobsError("preview reported errors/warnings; review before submission")
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        meta, spec = obj["metadata"], obj["spec"]
        if obj["kind"] != "RayJob" or meta["namespace"] != "fleet-train-jobs":
            raise JobsError("preview kind/namespace mismatch")
        if meta["labels"].get("kueue.x-k8s.io/queue-name") != "training-lq":
            raise JobsError("preview must use normal training-lq admission")
        if (
            meta["labels"].get("kueue.x-k8s.io/priority-class")
            != "q" + config["priority_class"][1:]
        ):
            raise JobsError("preview queue priority differs from requested pod priority")
        if meta["labels"].get("fleet.ai/requeue-if-preempted") != "false":
            raise JobsError("preview requeue policy drift")
        annotations = meta["annotations"]
        if annotations.get("fleet.ai/run-dir") != config["run_dir"]:
            raise JobsError("preview output directory drift")
        if annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF:
            raise JobsError("preview does not disable failed-job alerts")
        if not (spec["suspend"] is True and spec["shutdownAfterJobFinishes"] is True):
            raise JobsError("preview must queue normally and release on exit")
        if type(spec.get("backoffLimit")) is not int or spec["backoffLimit"] != 0:
            raise JobsError("preview must disable controller retries")
        if spec["entrypoint"] != config["command"]:
            raise JobsError("preview entrypoint drift")
        cluster = spec["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (g["replicas"], g["template"]) for g in cluster.get("workerGroupSpecs", [])
        ]
        nodes = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid preview replica count")
            if not replicas:
                continue
            pod = template["spec"]
            if pod.get("restartPolicy") != "Never":
                raise JobsError("preview Pod restart policy drift")
            if pod.get("nodeName") or pod.get("priorityClassName") != config["priority_class"]:
                raise JobsError("preview node assignment/priority bypass")
            gpu_containers = [
                c
                for c in pod["containers"]
                if quantity(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)) > 0
            ]
            if len(gpu_containers) != 1:
                raise JobsError("expected exactly one GPU container per worker node")
            c = gpu_containers[0]
            if c["image"] != config["image"]:
                raise JobsError("preview image drift")
            if c.get("securityContext", {}).get("privileged", False) != config.get(
                "privileged", False
            ):
                raise JobsError("preview privilege drift")
            for k in ("requests", "limits"):
                resources = c["resources"][k]
                if quantity(resources.get("nvidia.com/gpu", 0)) != config["gpus_per_worker"]:
                    raise JobsError("preview GPU count drift")
                for name in ("cpu", "memory"):
                    expected = config["resources"][f"{name}_{k[:-1]}"]
                    if quantity(resources[name]) != quantity(expected):
                        raise JobsError("preview CPU/memory resource drift")
            entries = c.get("env", [])
            env = {v["name"]: v.get("value") for v in entries}
            if len(env) != len(entries):
                raise JobsError("preview has duplicate runtime environment names")
            if env.get("RUN_DIR") != config["run_dir"] or any(
                env.get(k) != v for k, v in config.get("env", {}).items()
            ):
                raise JobsError("preview runtime environment drift")
            for name in config.get("secrets", []):
                refs = [
                    v for v in c.get("envFrom", []) if v.get("secretRef", {}).get("name") == name
                ]
                if (
                    len(refs) != 1
                    or refs[0].get("prefix", "") != ""
                    or refs[0]["secretRef"].get("optional", False) is not False
                ):
                    raise JobsError(
                        "preview workload Secret must be unique, required and unprefixed"
                    )
            pulls = {v["name"] for v in pod.get("imagePullSecrets", [])}
            if not set(config.get("image_pull_secrets", [])).issubset(pulls):
                raise JobsError("preview image-pull Secret reference missing")
            nodes += replicas
        if nodes != config["workers"]:
            raise JobsError("preview worker count drift")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed Jobs API preview") from exc
    canonical = copy.deepcopy(obj)
    metadata = canonical.get("metadata", {})
    for field in ("creationTimestamp", "generation", "managedFields", "resourceVersion", "uid"):
        metadata.pop(field, None)
    canonical.pop("status", None)
    return {
        "manifest_sha256": digest(canonical),
        "nodes": nodes,
        "gpus": nodes * config["gpus_per_worker"],
        "image": config["image"],
    }


def safe_status(run: dict) -> dict:
    """Never return failure messages: Ray may embed private trainer output."""
    return {
        k: run.get(k) for k in ("name", "job_id", "run_dir", "status", "created_at", "finished_at")
    }


def validate_creator_response(config: dict, value: object) -> dict:
    """Bind the deployed flat creator row to one exact reviewed request."""
    if not isinstance(value, dict) or set(value) != JOBS_RUN_FIELDS:
        raise JobsError("Jobs creator response fields changed; reconcile, never repeat POST")
    try:
        job_id = str(UUID(str(value["job_id"])))
    except (KeyError, TypeError, ValueError):
        raise JobsError("Jobs creator identity is invalid; reconcile, never repeat POST") from None
    generated_name = re.fullmatch(
        re.escape(config["name"]) + r"-[a-f0-9]{8}", str(value.get("name", ""))
    )
    optional_strings = ("message", "priority_reason", "submitted_by", "submitted_by_profile_id")
    if (
        generated_name is None
        or job_id != value["job_id"]
        or value.get("run_dir") != config["run_dir"]
        or value.get("image") != config["image"]
        or value.get("priority_class") != config["priority_class"]
        or value.get("queue_priority_class") != "q" + config["priority_class"][1:]
        or value.get("requeueIfPreempted") is not False
        or not isinstance(value.get("status"), str)
        or not value["status"]
        or any(
            value.get(key) is not None and not isinstance(value[key], str)
            for key in optional_strings
        )
    ):
        raise JobsError("Jobs creator response differs from the exact request; reconcile")
    return {
        "name": value["name"],
        "job_id": job_id,
        "run_dir": value["run_dir"],
        "status": value["status"],
    }


def _validate_journal_response(config: dict, value: object) -> dict:
    fields = {"state", "name", "job_id", "run_dir", "status"}
    if not isinstance(value, dict) or set(value) != fields or value.get("state") != "POST_RESPONSE":
        raise JobsError("submission response journal record is malformed")
    try:
        job_id = str(UUID(str(value["job_id"])))
    except (KeyError, TypeError, ValueError):
        raise JobsError("submission response journal job ID is invalid") from None
    if (
        re.fullmatch(re.escape(config["name"]) + r"-[a-f0-9]{8}", str(value.get("name", "")))
        is None
        or job_id != value["job_id"]
        or value.get("run_dir") != config["run_dir"]
        or not isinstance(value.get("status"), str)
        or not value["status"]
    ):
        raise JobsError("submission response journal identity changed")
    return {key: value[key] for key in ("name", "job_id", "run_dir", "status")}


def read_submission_journal(
    path: Path,
    config: dict,
    *,
    expected_manifest_sha256: str,
    expected_intent_evidence_file_sha256: str | None,
    expected_response: dict | None = None,
) -> tuple[dict, dict | None, str]:
    """Read one durable intent and an optional complete response without replay."""
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise JobsError("submission journal publisher is still active") from None
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not 0 < metadata.st_size <= 64 * 1024
        ):
            raise JobsError("submission journal is not one bounded mode-0600 regular file")
        chunks = []
        remaining = 64 * 1024 + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > 64 * 1024:
            raise JobsError("submission journal changed while it was read")
    except OSError as exc:
        raise JobsError("submission journal could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    parts = payload.split(b"\n")
    tail = b"" if parts[-1] == b"" else parts[-1]
    complete = parts[:-1]
    if not complete or len(complete) > 2 or any(not row for row in complete):
        raise JobsError("submission journal has malformed complete records")
    if len(complete) == 2 and tail:
        raise JobsError("submission journal has an unexpected third record")
    if tail:
        if expected_response is None:
            raise JobsError("submission journal has an unverified torn response")
        immutable = {key: expected_response[key] for key in ("name", "job_id", "run_dir")}
        canonical_prefix = (
            json.dumps({"state": "POST_RESPONSE", **immutable})[:-1] + ', "status": '
        ).encode()
        if not (
            canonical_prefix.startswith(tail)
            or (tail.startswith(canonical_prefix) and len(tail) <= len(canonical_prefix) + 1024)
        ):
            raise JobsError("submission journal torn response differs from exact Jobs history")
    try:
        rows = [json.loads(row) for row in complete]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JobsError("submission journal has a malformed complete record") from exc
    intent = rows[0]
    expected = {
        "state": "POST_INTENT_DO_NOT_RETRY",
        "intent_created_at_epoch": intent.get("intent_created_at_epoch"),
        "intent_evidence_file_sha256": expected_intent_evidence_file_sha256,
        "request_sha256": digest(config),
        "manifest_sha256": expected_manifest_sha256,
        "nodes": config["workers"],
        "gpus": config["workers"] * config["gpus_per_worker"],
        "image": config["image"],
    }
    if type(intent.get("intent_created_at_epoch")) is not int or intent != expected:
        raise JobsError("submission intent differs from the reviewed request and preview")
    response = _validate_journal_response(config, rows[1]) if len(rows) == 2 else None
    return intent, response, "sha256:" + hashlib.sha256(payload).hexdigest()


class Jobs:
    def __init__(self, token: str, *, base_url: str = API_URL, transport=None):
        if not token:
            raise JobsError("a Jobs API token is required")
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=60,
            headers={"Authorization": "Bearer " + token},
            transport=transport,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.client.close()

    def request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise JobsError(f"Jobs API {method} returned HTTP {exc.response.status_code}") from None
        except httpx.RequestError:
            raise JobsError(
                f"Jobs API {method} transport failed; reconcile any submitted intent"
            ) from None
        try:
            value = response.json()
        except ValueError:
            raise JobsError("Jobs API returned invalid JSON") from None
        if not isinstance(value, dict):
            raise JobsError("Jobs API returned a non-object")
        return value

    def all_runs(self) -> list[dict]:
        rows, seen = [], set()
        while True:
            page = self.request("GET", "/v1/runs", params={"limit": 200, "offset": len(rows)})
            items, more = page.get("items"), page.get("has_more")
            if not isinstance(items, list) or type(more) is not bool or (more and not items):
                raise JobsError("invalid/nonadvancing run pagination")
            for row in items:
                if not isinstance(row, dict) or not row.get("name") or row["name"] in seen:
                    raise JobsError("run history changed during pagination; repeat read-only check")
                seen.add(row["name"])
                rows.append(row)
            if not more:
                return rows

    def preview(self, config: dict) -> dict:
        validate_request(config)
        result = self.request("POST", "/v1/runs/preview", json=config)
        validate_preview(config, result)
        return result

    def raw_preview(self, config: dict) -> dict:
        """Return the live render without declaring it submission-qualified.

        The maintained direct-SFT fallback consumes this response and performs
        its own stricter, transform-aware validation.  All normal Jobs API
        submissions must continue to use :meth:`preview`.
        """
        validate_request(config)
        return self.request("POST", "/v1/runs/preview", json=config)

    def status(self, name: str) -> dict:
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise JobsError("invalid run name")
        return safe_status(self.request("GET", "/v1/runs/" + quote(name, safe="")))

    def status_exact(self, config: dict, name: str, job_id: str) -> dict:
        """Resolve status from the deployed flat list contract, not single-GET shape."""
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise JobsError("invalid run name")
        matches = [
            validate_creator_response(config, row)
            for row in self.all_runs()
            if row.get("name") == name or row.get("job_id") == job_id
        ]
        if len(matches) != 1:
            raise JobsError("Jobs status did not resolve one exact creator identity")
        result = matches[0]
        if result["name"] != name or result["job_id"] != job_id:
            raise JobsError("Jobs status identity differs from the creator receipt")
        return result

    def reconcile_submission(self, config: dict) -> dict:
        """Resolve one uncertain POST by exact history reads; never submit."""
        matches = []
        for row in self.all_runs():
            name = row.get("name")
            if row.get("run_dir") == config["run_dir"] or (
                isinstance(name, str) and name.startswith(config["name"] + "-")
            ):
                matches.append(validate_creator_response(config, row))
        if len(matches) != 1:
            raise JobsError("uncertain submission did not resolve to exactly one creator identity")
        return matches[0]

    def delete(self, name: str) -> dict:
        """Release one exact Jobs API run; success is exactly HTTP 204.

        DELETE is deliberately not routed through :meth:`request`: the API's
        successful response has no JSON body.  A transport error is uncertain,
        so callers must reconcile GET/Kubernetes state instead of repeating it.
        """
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise JobsError("invalid run name")
        try:
            response = self.client.request("DELETE", "/v1/runs/" + quote(name, safe=""))
        except httpx.RequestError:
            raise JobsError(
                "Jobs API DELETE transport failed; reconcile before any further action"
            ) from None
        if response.status_code != 204:
            raise JobsError(
                f"Jobs API DELETE returned HTTP {response.status_code}; reconcile before retry"
            )
        if response.content:
            raise JobsError("Jobs API DELETE returned an unexpected response body")
        return {"name": name, "deleted": True, "http_status": 204}

    def submit_once(
        self,
        config: dict,
        journal: Path,
        *,
        expected_preview_manifest_sha256: str | None = None,
        intent_evidence_file_sha256: str | None = None,
        before_intent: Callable[[dict], str] | None = None,
        not_after_epoch: int | None = None,
    ) -> dict:
        validate_request(config)
        if before_intent is not None and intent_evidence_file_sha256 is not None:
            raise JobsError("submission intent evidence has two competing producers")
        if journal.exists() or journal.is_symlink():
            raise JobsError("submission journal already exists; reconcile, never repeat POST")
        for row in self.all_runs():
            name = row["name"]
            if (
                row.get("run_dir") == config["run_dir"]
                or name == config["name"]
                or name.startswith(config["name"] + "-")
                or (config.get("title") is not None and row.get("title") == config["title"])
            ):
                raise JobsError("a recorded run already owns this name/title/output; reconcile it")
        proof = validate_preview(config, self.preview(config))
        if (
            expected_preview_manifest_sha256 is not None
            and proof["manifest_sha256"] != expected_preview_manifest_sha256
        ):
            raise JobsError("final server preview differs from the reviewed manifest")
        if (
            intent_evidence_file_sha256 is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", intent_evidence_file_sha256) is None
        ):
            raise JobsError("submission intent evidence digest is invalid")
        for row in self.all_runs():
            name = row["name"]
            if (
                row.get("run_dir") == config["run_dir"]
                or name == config["name"]
                or name.startswith(config["name"] + "-")
                or (config.get("title") is not None and row.get("title") == config["title"])
            ):
                raise JobsError(
                    "a recorded run appeared after preview; reconcile it before submission"
                )
        if not_after_epoch is not None and (
            type(not_after_epoch) is not int or time.time() + 30 > not_after_epoch
        ):
            raise JobsError("submission authority expired during final duplicate checks")
        if journal.name in {"", ".", ".."} or journal.parent.is_symlink():
            raise JobsError("submission journal path is unsafe")
        directory_fd = -1
        journal_fd = -1
        temporary_name = f".{journal.name}.{uuid4().hex}.tmp"

        def require_attached_journal_directory(*, durable_intent: bool) -> None:
            try:
                opened = os.fstat(directory_fd)
                named = os.stat(journal.parent, follow_symlinks=False)
            except OSError as exc:
                suffix = (
                    " after durable intent; reconcile, never repeat POST"
                    if durable_intent
                    else " before durable intent"
                )
                raise JobsError("submission journal directory became unavailable" + suffix) from exc
            if (
                not stat.S_ISDIR(named.st_mode)
                or named.st_dev != opened.st_dev
                or named.st_ino != opened.st_ino
            ):
                suffix = (
                    " after durable intent; reconcile, never repeat POST"
                    if durable_intent
                    else " before durable intent"
                )
                raise JobsError("submission journal directory changed" + suffix)

        try:
            directory_fd = os.open(
                journal.parent,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            journal_fd = os.open(
                temporary_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(journal_fd, 0o600)
            fcntl.flock(journal_fd, fcntl.LOCK_EX)
            if before_intent is not None:
                request_sha256 = digest(config)
                proof_sha256 = digest(proof)
                intent_evidence_file_sha256 = before_intent(copy.deepcopy(proof))
                if digest(config) != request_sha256 or digest(proof) != proof_sha256:
                    raise JobsError("submission callback mutated the reviewed request or preview")
            if (
                intent_evidence_file_sha256 is not None
                and re.fullmatch(r"sha256:[0-9a-f]{64}", intent_evidence_file_sha256) is None
            ):
                raise JobsError("submission intent evidence digest is invalid")
            intent_payload = (
                json.dumps(
                    {
                        "state": "POST_INTENT_DO_NOT_RETRY",
                        "intent_created_at_epoch": int(time.time()),
                        "intent_evidence_file_sha256": intent_evidence_file_sha256,
                        "request_sha256": digest(config),
                        **proof,
                    }
                )
                + "\n"
            ).encode()
            written = 0
            while written < len(intent_payload):
                count = os.write(journal_fd, intent_payload[written:])
                if count <= 0:
                    raise OSError("submission intent write made no progress")
                written += count
            os.fsync(journal_fd)
            require_attached_journal_directory(durable_intent=False)
            os.link(
                temporary_name,
                journal.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.fsync(directory_fd)
        except OSError as exc:
            if directory_fd >= 0:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_name, dir_fd=directory_fd)
            if journal_fd >= 0:
                os.close(journal_fd)
                journal_fd = -1
            if directory_fd >= 0:
                os.close(directory_fd)
                directory_fd = -1
            raise JobsError("submission intent could not be durably published") from exc
        except Exception:
            if directory_fd >= 0:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_name, dir_fd=directory_fd)
            if journal_fd >= 0:
                os.close(journal_fd)
                journal_fd = -1
            if directory_fd >= 0:
                os.close(directory_fd)
                directory_fd = -1
            raise
        try:
            if not_after_epoch is not None and time.time() > not_after_epoch:
                raise JobsError(
                    "submission authority expired after durable intent; "
                    "reconcile, never repeat POST"
                )
            require_attached_journal_directory(durable_intent=True)
            # Do not wrap this POST in retry logic, even for a timeout or HTTP error.
            response = self.request("POST", "/v1/runs", json=config)
            result = validate_creator_response(config, response)
            response_payload = (json.dumps({"state": "POST_RESPONSE", **result}) + "\n").encode()
            os.lseek(journal_fd, 0, os.SEEK_END)
            written = 0
            while written < len(response_payload):
                count = os.write(journal_fd, response_payload[written:])
                if count <= 0:
                    raise OSError("submission response write made no progress")
                written += count
            os.fsync(journal_fd)
            return result
        except OSError as exc:
            raise JobsError(
                "submission response could not be durably recorded; reconcile, never repeat POST"
            ) from exc
        finally:
            if journal_fd >= 0:
                os.close(journal_fd)
                journal_fd = -1
            if directory_fd >= 0:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_name, dir_fd=directory_fd)
                with contextlib.suppress(OSError):
                    os.fsync(directory_fd)
                os.close(directory_fd)
