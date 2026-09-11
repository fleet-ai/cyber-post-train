"""Small generic Jobs API client shared by training backends.

This is a transport/resource boundary, not a model-qualification certificate.
Callers validate/stage their scientific inputs and check the live ownership
budget before submitting. The journal must be shared by every submitter for an
experiment: the server generates a new run ID on every POST, not an idempotency
key. An uncertain POST is recorded and must be reconciled, never replayed.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import shlex
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx
import yaml

API_URL = "https://api.ft.flt.build"


class JobsError(ValueError):
    pass


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


def bundled_request(request: dict, files: dict[str, str], module: str, argv: list[str]) -> dict:
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
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        "b=base64.b64decode(os.environ.pop('CYBER_RUNTIME_BUNDLE'),validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={hashlib.sha256(blob).hexdigest()!r};"
        "v=json.loads(gzip.decompress(b));"
        "p=pathlib.Path(os.environ['RUN_DIR'])/'.runtime';p.mkdir(mode=0o700);"
        "[((p/n).parent.mkdir(parents=True,exist_ok=True),(p/n).write_text(t)) "
        "for n,t in v['files'].items()];"
        "os.chdir(p);sys.path.insert(0,str(p));importlib.invalidate_caches();"
        "sys.argv=[v['module']]+v['argv'];runpy.run_module(v['module'],run_name='__main__')"
    )
    result = {
        **request,
        "command": "python -c " + shlex.quote(bootstrap),
        "env": {**request.get("env", {}), "CYBER_RUNTIME_BUNDLE": base64.b64encode(blob).decode()},
    }
    validate_request(result)
    return result


def validate_request(config: dict) -> None:
    """Check our reproducible GPU-job subset before any network request."""
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", config.get("name", "")):
        raise JobsError("name must be a DNS label of at most 31 characters")
    if config["name"] == "ft-run":
        raise JobsError("ft-run is a reserved name")
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
    for key, upper in (("workers", 4), ("gpus_per_worker", 8)):
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
    if preview.get("errors") or preview.get("warnings"):
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
        if meta["annotations"].get("fleet.ai/run-dir") != config["run_dir"]:
            raise JobsError("preview output directory drift")
        if not (spec["suspend"] is True and spec["shutdownAfterJobFinishes"] is True):
            raise JobsError("preview must queue normally and release on exit")
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
            env = {v["name"]: v.get("value") for v in c.get("env", [])}
            if env.get("RUN_DIR") != config["run_dir"] or any(
                env.get(k) != v for k, v in config.get("env", {}).items()
            ):
                raise JobsError("preview runtime environment drift")
            refs = {v.get("secretRef", {}).get("name") for v in c.get("envFrom", [])}
            if not set(config.get("secrets", [])).issubset(refs):
                raise JobsError("preview workload Secret reference missing")
            pulls = {v["name"] for v in pod.get("imagePullSecrets", [])}
            if not set(config.get("image_pull_secrets", [])).issubset(pulls):
                raise JobsError("preview image-pull Secret reference missing")
            nodes += replicas
        if nodes != config["workers"]:
            raise JobsError("preview worker count drift")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed Jobs API preview") from exc
    return {
        "manifest_sha256": digest(obj),
        "nodes": nodes,
        "gpus": nodes * config["gpus_per_worker"],
        "image": config["image"],
    }


def safe_status(run: dict) -> dict:
    """Never return failure messages: Ray may embed private trainer output."""
    return {
        k: run.get(k) for k in ("name", "job_id", "run_dir", "status", "created_at", "finished_at")
    }


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

    def status(self, name: str) -> dict:
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise JobsError("invalid run name")
        return safe_status(self.request("GET", "/v1/runs/" + quote(name, safe="")))

    def submit_once(self, config: dict, journal: Path) -> dict:
        validate_request(config)
        if journal.exists() or journal.is_symlink():
            raise JobsError("submission journal already exists; reconcile, never repeat POST")
        for row in self.all_runs():
            name = row["name"]
            if (
                row.get("run_dir") == config["run_dir"]
                or name == config["name"]
                or name.startswith(config["name"] + "-")
            ):
                raise JobsError("a recorded run already owns this name/output; reconcile it")
        proof = validate_preview(config, self.preview(config))
        journal.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(
                json.dumps(
                    {"state": "POST_INTENT_DO_NOT_RETRY", "request_sha256": digest(config), **proof}
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Do not wrap this POST in retry logic, even for a timeout or HTTP error.
        response = self.request("POST", "/v1/runs", json=config)
        if not re.fullmatch(re.escape(config["name"]) + r"-[a-f0-9]{8}", response.get("name", "")):
            raise JobsError("ambiguous submit response; reconcile journal, never repeat POST")
        if response.get("run_dir") not in (None, config["run_dir"]):
            raise JobsError("submitted output differs; reconcile resource ownership immediately")
        result = safe_status(response)
        with journal.open("a") as stream:
            stream.write(json.dumps({"state": "POST_RESPONSE", **result}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
