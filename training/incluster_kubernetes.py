"""Narrow in-cluster Kubernetes transport for the prod10 SkyRL operator.

The existing create and cleanup gates accept a ``subprocess.run``-shaped
callable because their normal operator boundary is ``kubectl``.  The bounded
prod10 coordinator runs inside Kubernetes and must not carry a kubeconfig or a
kubectl binary.  This module translates only the exact command subset used by
those reviewed gates to the mounted service-account API.  Every other command,
context, namespace, resource, verb, or flag fails closed.
"""

from __future__ import annotations

import json
import os
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import dev_cleanup_observer as cleanup

_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
_RESOURCE_PATHS = {
    "job": ("/apis/batch/v1", "jobs"),
    "jobs": ("/apis/batch/v1", "jobs"),
    "jobs.batch": ("/apis/batch/v1", "jobs"),
    "pod": ("/api/v1", "pods"),
    "pods": ("/api/v1", "pods"),
    "rayjob": ("/apis/ray.io/v1", "rayjobs"),
    "rayjobs": ("/apis/ray.io/v1", "rayjobs"),
    "rayjobs.ray.io": ("/apis/ray.io/v1", "rayjobs"),
    "raycluster": ("/apis/ray.io/v1", "rayclusters"),
    "rayclusters": ("/apis/ray.io/v1", "rayclusters"),
    "rayclusters.ray.io": ("/apis/ray.io/v1", "rayclusters"),
    "workload": ("/apis/kueue.x-k8s.io/v1beta2", "workloads"),
    "workloads": ("/apis/kueue.x-k8s.io/v1beta2", "workloads"),
    "workloads.kueue.x-k8s.io": ("/apis/kueue.x-k8s.io/v1beta2", "workloads"),
}
_CREATE_PATHS = {
    ("batch/v1", "Job"): "/apis/batch/v1/namespaces/{namespace}/jobs",
}


class InClusterKubernetesError(RuntimeError):
    """A sanitized refusal or service-account transport failure."""


def _default_request(
    method: str, path: str, body: bytes | None, headers: dict[str, str]
) -> tuple[int, bytes]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443").strip()
    if not host or not port.isdecimal():
        raise InClusterKubernetesError("in-cluster Kubernetes endpoint is unavailable")
    try:
        token = _TOKEN.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise InClusterKubernetesError("service-account token is unavailable") from exc
    if not token:
        raise InClusterKubernetesError("service-account token is empty")
    request = urllib.request.Request(
        f"https://{host}:{port}{path}",
        method=method,
        data=body,
        headers={"Authorization": f"Bearer {token}", **headers},
    )
    context = ssl.create_default_context(cafile=str(_CA))
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (OSError, urllib.error.URLError) as exc:
        raise InClusterKubernetesError("Kubernetes API transport failed") from exc


class InClusterKubernetesRunner:
    """A ``subprocess.run``-compatible adapter for one production namespace."""

    def __init__(
        self,
        *,
        request: Callable[
            [str, str, bytes | None, dict[str, str]], tuple[int, bytes]
        ] = _default_request,
    ) -> None:
        self._request = request

    @staticmethod
    def _completed(
        command: list[str], returncode: int, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    @staticmethod
    def _prefix(command: list[str]) -> tuple[str, str, list[str]]:
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise InClusterKubernetesError("Kubernetes command is malformed")
        if not command or command[0] != "kubectl":
            raise InClusterKubernetesError("only the typed kubectl compatibility surface is valid")
        arguments = command[1:]
        context = namespace = ""
        while arguments and arguments[0].startswith("--"):
            flag = arguments.pop(0)
            if flag == "--context" and arguments:
                context = arguments.pop(0)
            elif flag == "--namespace" and arguments:
                namespace = arguments.pop(0)
            elif flag.startswith("--request-timeout="):
                continue
            else:
                raise InClusterKubernetesError("Kubernetes command prefix is not reviewed")
        if context != cleanup.PROD_CONTEXT or namespace != cleanup.NAMESPACE:
            raise InClusterKubernetesError("in-cluster runner is production-namespace only")
        if not arguments:
            raise InClusterKubernetesError("Kubernetes command has no verb")
        return context, namespace, arguments

    @staticmethod
    def _get_path(namespace: str, resource: str, name: str | None) -> str:
        try:
            base, plural = _RESOURCE_PATHS[resource]
        except KeyError as exc:
            raise InClusterKubernetesError("Kubernetes read resource is not reviewed") from exc
        path = f"{base}/namespaces/{urllib.parse.quote(namespace, safe='')}/{plural}"
        if name is not None:
            path += "/" + urllib.parse.quote(name, safe="")
        return path

    def _get(
        self, command: list[str], namespace: str, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        if len(arguments) < 2:
            raise InClusterKubernetesError("Kubernetes get command is incomplete")
        resource = arguments[1]
        index = 2
        name: str | None = None
        if index < len(arguments) and not arguments[index].startswith("-"):
            name = arguments[index]
            index += 1
        ignore_not_found = False
        selector: str | None = None
        while index < len(arguments):
            flag = arguments[index]
            index += 1
            if flag in {"--output", "-o"} and index < len(arguments):
                if arguments[index] != "json":
                    raise InClusterKubernetesError("Kubernetes get output is not reviewed")
                index += 1
            elif flag in {"--output=json", "-o=json"}:
                continue
            elif flag == "--ignore-not-found":
                ignore_not_found = True
            elif flag == "--selector" and index < len(arguments):
                selector = arguments[index]
                index += 1
            elif flag.startswith("--selector="):
                selector = flag.removeprefix("--selector=")
            else:
                raise InClusterKubernetesError("Kubernetes get flag is not reviewed")
        if name is not None and selector is not None:
            raise InClusterKubernetesError("Kubernetes get cannot mix name and selector")
        path = self._get_path(namespace, resource, name)
        if selector is not None:
            path += "?" + urllib.parse.urlencode({"labelSelector": selector})
        status, payload = self._request("GET", path, None, {"Accept": "application/json"})
        if status == 404 and ignore_not_found:
            return self._completed(command, 0)
        if status != 200:
            return self._completed(command, 1, stderr="Kubernetes read failed")
        return self._completed(command, 0, stdout=payload.decode("utf-8"))

    def _create(
        self,
        command: list[str],
        namespace: str,
        arguments: list[str],
        input_text: str | None,
    ) -> subprocess.CompletedProcess[str]:
        flags = arguments[1:]
        dry_run = False
        index = 0
        while index < len(flags):
            flag = flags[index]
            index += 1
            if flag in {"-f", "--filename"} and index < len(flags):
                if flags[index] != "-":
                    raise InClusterKubernetesError("Kubernetes create source is not stdin")
                index += 1
            elif flag in {"-o", "--output"} and index < len(flags):
                if flags[index] != "json":
                    raise InClusterKubernetesError("Kubernetes create output is not reviewed")
                index += 1
            elif flag in {"-o=json", "--output=json"}:
                continue
            elif flag in {"--dry-run=server", "--dry-run", "--dry-run=all"}:
                dry_run = True
            else:
                raise InClusterKubernetesError("Kubernetes create flag is not reviewed")
        if not isinstance(input_text, str):
            raise InClusterKubernetesError("Kubernetes create manifest is absent")
        try:
            value = json.loads(input_text)
        except ValueError as exc:
            raise InClusterKubernetesError("Kubernetes create manifest is invalid") from exc
        if not isinstance(value, dict):
            raise InClusterKubernetesError("Kubernetes create manifest is not an object")
        key = (value.get("apiVersion"), value.get("kind"))
        try:
            template = _CREATE_PATHS[key]
        except KeyError as exc:
            raise InClusterKubernetesError("Kubernetes create kind is not reviewed") from exc
        metadata = value.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("namespace") != namespace:
            raise InClusterKubernetesError("Kubernetes create namespace changed")
        path = template.format(namespace=urllib.parse.quote(namespace, safe=""))
        if dry_run:
            path += "?" + urllib.parse.urlencode(
                {"dryRun": "All", "fieldManager": "cyber-post-train-prod10"}
            )
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        status, payload = self._request(
            "POST",
            path,
            body,
            {"Accept": "application/json", "Content-Type": "application/json"},
        )
        if status not in {200, 201}:
            return self._completed(command, 1, stderr="Kubernetes create failed")
        return self._completed(command, 0, stdout=payload.decode("utf-8"))

    def _delete_raw(
        self, command: list[str], arguments: list[str], input_text: str | None
    ) -> subprocess.CompletedProcess[str]:
        if arguments[:2] != ["delete", "--raw"] or len(arguments) != 5:
            raise InClusterKubernetesError("Kubernetes delete is not the UID-CAS raw route")
        path, source_flag, source = arguments[2:]
        if source_flag != "-f" or source != "-" or not path.startswith("/apis/"):
            raise InClusterKubernetesError("Kubernetes delete route is not reviewed")
        if not isinstance(input_text, str):
            raise InClusterKubernetesError("Kubernetes UID-CAS delete body is absent")
        try:
            body = json.loads(input_text)
        except ValueError as exc:
            raise InClusterKubernetesError("Kubernetes UID-CAS delete body is invalid") from exc
        if (
            not isinstance(body, dict)
            or body.get("apiVersion") != "v1"
            or body.get("kind") != "DeleteOptions"
            or body.get("propagationPolicy") != "Foreground"
            or not isinstance(body.get("preconditions"), dict)
            or set(body["preconditions"]) != {"uid"}
        ):
            raise InClusterKubernetesError("Kubernetes delete is not exact-UID conditioned")
        status, payload = self._request(
            "DELETE",
            path,
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
            {"Accept": "application/json", "Content-Type": "application/json"},
        )
        if status not in {200, 202}:
            return self._completed(command, 1, stderr="Kubernetes UID-CAS delete failed")
        return self._completed(command, 0, stdout=payload.decode("utf-8"))

    def __call__(
        self,
        command: list[str],
        *,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: int | float | None = None,
        **unexpected: Any,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        if unexpected or capture_output is not True or text is not True:
            raise InClusterKubernetesError("Kubernetes runner invocation is not reviewed")
        _, namespace, arguments = self._prefix(list(command))
        if arguments[0] == "get":
            if input is not None:
                raise InClusterKubernetesError("Kubernetes get received stdin")
            return self._get(command, namespace, arguments)
        if arguments[0] == "create":
            return self._create(command, namespace, arguments, input)
        if arguments[0] == "delete":
            return self._delete_raw(command, arguments, input)
        raise InClusterKubernetesError("Kubernetes verb is not reviewed")
