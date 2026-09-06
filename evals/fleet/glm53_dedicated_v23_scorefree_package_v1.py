"""Immutable held package renderer for the GLM v23 score-free qualifier."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_package_v1 as prior
from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as qualifier

SCHEMA = "fleet-glm53-dedicated-v23-scorefree-package-v1"
CONFIGMAP_NAME = qualifier.JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = qualifier.JOB_NAME + "-authorization"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FILES = tuple(
    path
    for path in prior.FILES
    if "glm53_dedicated_v22_concurrency_authorization_v1.py" not in path
    and "glm53_dedicated_v22_concurrency_qualification_v1.py" not in path
) + (
    "evals/fleet/glm53_dedicated_v22_concurrency_qualification_v1.py",
    "evals/fleet/glm53_dedicated_v23_scorefree_qualifier_v1.py",
    "evals/fleet/glm53_dedicated_v23_request_counter_watchdog_v1.py",
)
RUN = "evals/fleet/scripts/run_glm53_dedicated_v23_scorefree_qualification_v1.sh"
OPERATOR_FILES = (
    "evals/fleet/glm53_dedicated_v23_scorefree_gpu_observer_v1.py",
    "evals/fleet/scripts/observe_glm53_dedicated_v23_scorefree_gpu_v1.sh",
)


class PackageError(RuntimeError):
    """The immutable v23 package cannot be rendered."""


def _source(root: Path, commit: str, path: str) -> bytes:
    if COMMIT_RE.fullmatch(commit) is None:
        raise PackageError("package_commit_invalid")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError(f"package_source_absent:{path}") from exc


def _key(path: str) -> str:
    return path.replace("/", "__SLASH__")


def build_configmap(root: Path, commit: str) -> dict[str, Any]:
    data: dict[str, str] = {}
    manifest: dict[str, str] = {}
    for path in FILES:
        raw = _source(root, commit, path)
        data[_key(path)] = raw.decode()
        manifest[path] = crypto.sha256(raw)
    run = _source(root, commit, RUN)
    data["run.sh"] = run.decode()
    manifest[RUN] = crypto.sha256(run)
    package = {
        "schema_version": SCHEMA,
        "package_commit": commit,
        "files": manifest,
        "job_name": qualifier.JOB_NAME,
        "output_root": str(qualifier.RESULT_ROOT),
        "concurrency_waves": list(qualifier.CONCURRENCY),
        "score_free": True,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "external_uid_bound_operator_files": {
            path: crypto.sha256(_source(root, commit, path)) for path in OPERATOR_FILES
        },
    }
    package["package_sha256"] = crypto.digest_without(package, "package_sha256")
    data["package.json"] = json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    data["package_commit"] = commit
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": prior.NAMESPACE},
        "immutable": True,
        "data": data,
    }


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    if (
        authorization.get("schema_version") != qualifier.AUTH_SCHEMA
        or authorization.get("qualification_launch_authorized") is not True
        or authorization.get("scored_launch_authorized") is not False
        or authorization.get("receipt_sha256")
        != crypto.digest_without(authorization, "receipt_sha256")
    ):
        raise PackageError("v23_authorization_invalid")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": AUTHORIZATION_CONFIGMAP_NAME, "namespace": prior.NAMESPACE},
        "immutable": True,
        "data": {
            "authorization.json": json.dumps(authorization, sort_keys=True, separators=(",", ":"))
            + "\n"
        },
    }


def build_job(configmap: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    seed = {
        "qualification_launch_authorized": True,
        "scored_successor_launch_authorized": False,
    }
    seed["receipt_sha256"] = crypto.digest_without(seed, "receipt_sha256")
    old_configmap = {
        "data": {
            "package.json": json.dumps({"package_commit": "0" * 40}),
        }
    }
    job = copy.deepcopy(prior.build_job(old_configmap, seed))
    job["metadata"]["name"] = qualifier.JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = qualifier.JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/authorization-receipt-sha256"] = (
        authorization["receipt_sha256"]
    )
    job["spec"]["activeDeadlineSeconds"] = 1800
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = qualifier.JOB_NAME
    package_commit = json.loads(configmap["data"]["package.json"])["package_commit"]
    for row in pod["spec"]["containers"][0]["env"]:
        if row["name"] == "QUALIFICATION_PACKAGE_COMMIT":
            row["value"] = package_commit
        elif row["name"] == "QUALIFICATION_OUTPUT_ROOT":
            row["value"] = str(qualifier.RESULT_ROOT)
    pod["spec"]["containers"][0]["env"].append(
        {
            "name": "DEDICATED_SERVICE_ORIGIN",
            "value": authorization["server_binding"]["service_origin"],
        }
    )
    for volume in pod["spec"]["volumes"]:
        if volume["name"] == "package":
            volume["configMap"]["name"] = CONFIGMAP_NAME
        elif volume["name"] == "authorization":
            volume["configMap"]["name"] = AUTHORIZATION_CONFIGMAP_NAME
    return job


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    configmap = build_configmap(root, commit)
    auth = build_authorization_configmap(authorization)
    job = build_job(configmap, authorization)
    objects = {"apiVersion": "v1", "kind": "List", "items": [configmap, auth, job]}
    return {
        "objects": objects,
        "package_sha256": json.loads(configmap["data"]["package.json"])["package_sha256"],
        "server_launch_authorized": False,
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
    }
