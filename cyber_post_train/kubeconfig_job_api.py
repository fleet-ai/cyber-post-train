"""Typed Kubernetes API transport for the prod10 zero-GPU controller Job.

This adapter talks directly to the API endpoint selected by one exact
kubeconfig context.  It never invokes ``kubectl`` and exposes only the scoped
Job create/read and terminal-evidence reads required by the reviewed caller.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import ssl
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import UUID

import httpx
import yaml

from .jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError
from .skyrl_controller_job import (
    NAME,
    NAMESPACE,
    PROD_CONTEXT,
    ControllerJobPackage,
    validate_controller_job_package,
)

FIELD_MANAGER = "cyber-post-train-prod10-controller-v1"
_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


def _one_named(rows: object, name: str, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise JobsError(f"prod10 kubeconfig {label} inventory is malformed")
    matches = [
        row.get(label)
        for row in rows
        if isinstance(row, dict) and row.get("name") == name and isinstance(row.get(label), dict)
    ]
    if len(matches) != 1:
        raise JobsError(f"prod10 kubeconfig {label} binding is not unique")
    return matches[0]


class KubeconfigControllerJobApi:
    """Exact prod-context batch Job API with exec-credential authentication."""

    def __init__(
        self,
        package: ControllerJobPackage,
        *,
        kubeconfig: Path | None = None,
        context: str = PROD_CONTEXT,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        client: httpx.Client | None = None,
    ) -> None:
        validate_controller_job_package(package)
        self._manifest = copy.deepcopy(package.job)
        if context != PROD_CONTEXT:
            raise JobsError("prod10 controller API requires the exact production context")
        configured = os.environ.get("KUBECONFIG")
        if kubeconfig is None:
            if configured and os.pathsep in configured:
                raise JobsError("prod10 controller does not merge kubeconfig files")
            kubeconfig = (
                Path(configured).expanduser() if configured else Path.home() / ".kube/config"
            )
        path = Path(kubeconfig).expanduser()
        try:
            identity = path.lstat()
            raw = path.read_bytes()
        except OSError:
            raise JobsError("prod10 controller kubeconfig is unreadable") from None
        if not stat.S_ISREG(identity.st_mode) or path.is_symlink():
            raise JobsError("prod10 controller kubeconfig must be one regular file")
        try:
            config = yaml.safe_load(raw)
        except yaml.YAMLError:
            raise JobsError("prod10 controller kubeconfig is invalid YAML") from None
        if not isinstance(config, dict) or config.get("current-context") != context:
            raise JobsError("prod10 controller kubeconfig current context changed")
        context_row = _one_named(config.get("contexts"), context, "context")
        cluster_name = context_row.get("cluster")
        user_name = context_row.get("user")
        if (
            not isinstance(cluster_name, str)
            or not cluster_name
            or not isinstance(user_name, str)
            or not user_name
            or set(context_row) - {"cluster", "user", "namespace"}
            or context_row.get("namespace", NAMESPACE) != NAMESPACE
        ):
            raise JobsError("prod10 controller kubeconfig context binding changed")
        cluster = _one_named(config.get("clusters"), cluster_name, "cluster")
        user = _one_named(config.get("users"), user_name, "user")
        server = cluster.get("server")
        ca_data = cluster.get("certificate-authority-data")
        parsed = urlsplit(server) if isinstance(server, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not parsed.netloc
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or set(cluster) != {"server", "certificate-authority-data"}
            or not isinstance(ca_data, str)
            or not ca_data
        ):
            raise JobsError("prod10 controller kubeconfig cluster trust changed")
        try:
            ca_pem = base64.b64decode(ca_data, validate=True).decode()
        except (ValueError, UnicodeDecodeError):
            raise JobsError("prod10 controller kubeconfig CA data is invalid") from None
        execution = user.get("exec") if isinstance(user, dict) else None
        if not isinstance(execution, dict) or set(user) != {"exec"}:
            raise JobsError("prod10 controller requires one exec-credential user")
        if (
            set(execution)
            != {
                "apiVersion",
                "args",
                "command",
                "env",
                "interactiveMode",
                "provideClusterInfo",
            }
            or execution.get("apiVersion")
            not in {
                "client.authentication.k8s.io/v1",
                "client.authentication.k8s.io/v1beta1",
            }
            or execution.get("interactiveMode") not in {"Never", "IfAvailable"}
            or execution.get("provideClusterInfo") is not False
            or not isinstance(execution.get("command"), str)
            or not Path(execution["command"]).is_absolute()
            or not isinstance(execution.get("args"), list)
            or not all(isinstance(item, str) and item for item in execution["args"])
            or execution.get("env") not in (None, [])
        ):
            raise JobsError("prod10 controller exec-credential contract changed")
        command = Path(execution["command"])
        try:
            command_identity = command.lstat()
        except OSError:
            raise JobsError("prod10 controller exec-credential command is unavailable") from None
        if not stat.S_ISREG(command_identity.st_mode) or not os.access(command, os.X_OK):
            raise JobsError("prod10 controller exec-credential command is not executable")
        self.server = server.rstrip("/")
        self.context = context
        self._exec = execution
        self._runner = runner
        self._owns_client = client is None
        if client is None:
            try:
                tls = ssl.create_default_context(cadata=ca_pem)
            except ssl.SSLError:
                raise JobsError("prod10 controller kubeconfig CA cannot create TLS trust") from None
            client = httpx.Client(timeout=60, verify=tls)
        self._client = client

    def __enter__(self) -> KubeconfigControllerJobApi:
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_client:
            self._client.close()

    def _token(self) -> str:
        try:
            result = self._runner(
                [self._exec["command"], *self._exec["args"]],
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                check=False,
                env={
                    **os.environ,
                    "KUBERNETES_EXEC_INFO": json.dumps(
                        {
                            "apiVersion": self._exec["apiVersion"],
                            "kind": "ExecCredential",
                            "spec": {"interactive": False},
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            )
        except (OSError, subprocess.SubprocessError):
            raise JobsError("prod10 controller exec-credential command failed") from None
        if result.returncode:
            raise JobsError("prod10 controller exec-credential command failed")
        try:
            value = json.loads(result.stdout)
            status = value["status"]
            token = status["token"]
        except (KeyError, TypeError, json.JSONDecodeError):
            raise JobsError("prod10 controller exec-credential response is malformed") from None
        if (
            not isinstance(value, dict)
            or value.get("kind") != "ExecCredential"
            or value.get("apiVersion") != self._exec["apiVersion"]
            or not isinstance(status, dict)
            or not isinstance(token, str)
            or not token
        ):
            raise JobsError("prod10 controller exec-credential response changed")
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        value: dict[str, Any] | None = None,
        allow_missing: bool = False,
        expect_text: bool = False,
    ) -> dict[str, Any] | str | None:
        try:
            response = self._client.request(
                method,
                self.server + path,
                params=params,
                json=value,
                headers={
                    "Authorization": "Bearer " + self._token(),
                    "Accept": "text/plain" if expect_text else "application/json",
                    "Content-Type": "application/json",
                },
            )
        except httpx.RequestError:
            raise JobsError("prod10 controller Kubernetes transport failed; reconcile") from None
        if allow_missing and response.status_code == 404:
            return None
        if response.status_code < 200 or response.status_code >= 300:
            raise JobsError(
                f"prod10 controller Kubernetes {method} returned HTTP {response.status_code}"
            )
        if expect_text:
            return response.text
        try:
            result = response.json()
        except ValueError:
            raise JobsError("prod10 controller Kubernetes response is not JSON") from None
        if not isinstance(result, dict):
            raise JobsError("prod10 controller Kubernetes response is not an object")
        return result

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
        if (
            manifest != self._manifest
            or manifest.get("apiVersion") != "batch/v1"
            or manifest.get("kind") != "Job"
            or metadata.get("namespace") != NAMESPACE
            or metadata.get("name") != NAME
            or metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        ):
            raise JobsError("prod10 controller API manifest root binding changed")

    def get_job(self, namespace: str, name: str) -> dict[str, Any] | None:
        if namespace != NAMESPACE or name != NAME:
            raise JobsError("prod10 controller Job read is outside the exact identity")
        result = self._request(
            "GET",
            f"/apis/batch/v1/namespaces/{quote(namespace, safe='')}/jobs/{quote(name, safe='')}",
            allow_missing=True,
        )
        return result if isinstance(result, dict) else None

    def server_dry_run_job(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._validate_manifest(manifest)
        result = self._request(
            "POST",
            f"/apis/batch/v1/namespaces/{quote(NAMESPACE, safe='')}/jobs",
            params={
                "dryRun": "All",
                "fieldManager": FIELD_MANAGER,
                "fieldValidation": "Strict",
            },
            value=manifest,
        )
        assert isinstance(result, dict)
        return result

    def create_job_once(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._validate_manifest(manifest)
        result = self._request(
            "POST",
            f"/apis/batch/v1/namespaces/{quote(NAMESPACE, safe='')}/jobs",
            params={"fieldManager": FIELD_MANAGER, "fieldValidation": "Strict"},
            value=manifest,
        )
        assert isinstance(result, dict)
        return result

    def terminal_evidence(
        self,
        *,
        job_uid: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
        try:
            UUID(job_uid)
        except (TypeError, ValueError):
            raise JobsError("prod10 controller terminal Job UID is invalid") from None
        job = self.get_job(NAMESPACE, NAME)
        if job is None or job.get("metadata", {}).get("uid") != job_uid:
            raise JobsError("prod10 controller terminal Job UID changed")
        workloads = self._request(
            "GET",
            f"/apis/kueue.x-k8s.io/v1beta2/namespaces/{quote(NAMESPACE, safe='')}/workloads",
            params={"labelSelector": "kueue.x-k8s.io/job-uid=" + job_uid},
        )
        pods = self._request(
            "GET",
            f"/api/v1/namespaces/{quote(NAMESPACE, safe='')}/pods",
            params={"labelSelector": "batch.kubernetes.io/job-name=" + NAME},
        )
        service_account = self._request(
            "GET",
            f"/api/v1/namespaces/{quote(NAMESPACE, safe='')}/serviceaccounts/default",
        )
        if not all(isinstance(item, dict) for item in (workloads, pods, service_account)):
            raise JobsError("prod10 controller terminal evidence is incomplete")
        items = pods.get("items")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise JobsError("prod10 controller terminal Pod identity is ambiguous")
        pod_name = items[0].get("metadata", {}).get("name")
        if not isinstance(pod_name, str) or _NAME.fullmatch(pod_name) is None:
            raise JobsError("prod10 controller terminal Pod name is invalid")
        logs = self._request(
            "GET",
            f"/api/v1/namespaces/{quote(NAMESPACE, safe='')}/pods/{quote(pod_name, safe='')}/log",
            params={"container": "controller", "limitBytes": 8192, "timestamps": "false"},
            expect_text=True,
        )
        if not isinstance(logs, str):
            raise JobsError("prod10 controller sanitized Pod log is unavailable")
        return job, workloads, pods, service_account, logs
