"""Bounded create-only Kubernetes relay for reviewed cyber-eval objects.

The local caller supplies both rendered objects and an independently frozen
allowlist of their canonical digests.  The relay can use the current kubectl
identity when it has create authority, or execute this same file inside the
exact ``allie-dev`` Pod and use that Pod's projected service-account identity.
The service-account token never crosses the Pod boundary or enters a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

NAMESPACE = "fleet-train-jobs"
ALLIE_NAME = "allie-dev"
ALLIE_UID = "73dabe56-60f8-4879-be9f-365196c502e3"
ALLOWLIST_SCHEMA = "fleet-kubernetes-create-relay-allowlist-v1"
ENVELOPE_SCHEMA = "fleet-kubernetes-create-relay-envelope-v1"
RECEIPT_SCHEMA = "fleet-kubernetes-create-relay-receipt-v1"
ALLOWED_IDENTITIES = {("v1", "ConfigMap"), ("batch/v1", "Job"), ("v1", "Service")}
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
NAME_RE = re.compile(r"chris-[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
MAX_ENVELOPE_BYTES = 4 * 1024 * 1024


class RelayError(RuntimeError):
    """A fail-closed relay validation or create error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def strict_json(raw: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise RelayError("duplicate_json_key")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelayError("invalid_json") from exc


def _metadata(value: dict[str, Any]) -> tuple[str, str]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise RelayError("object_metadata_absent")
    forbidden = {
        "uid",
        "resourceVersion",
        "generation",
        "creationTimestamp",
        "deletionTimestamp",
        "managedFields",
        "generateName",
        "ownerReferences",
    }
    if forbidden.intersection(metadata):
        raise RelayError("server_or_generated_metadata_forbidden")
    name = metadata.get("name")
    namespace = metadata.get("namespace")
    if not isinstance(name, str) or NAME_RE.fullmatch(name) is None:
        raise RelayError("object_name_not_allowlisted_shape")
    if namespace != NAMESPACE:
        raise RelayError("object_namespace_invalid")
    return name, namespace


def _resources_have_gpu(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "gpu" in str(key).lower() or _resources_have_gpu(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_resources_have_gpu(item) for item in value)
    return False


def validate_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RelayError("object_not_mapping")
    api_version, kind = value.get("apiVersion"), value.get("kind")
    if (api_version, kind) not in ALLOWED_IDENTITIES:
        raise RelayError("object_kind_not_allowed")
    _metadata(value)
    if kind == "ConfigMap":
        if value.get("immutable") is not True or "binaryData" in value:
            raise RelayError("configmap_must_be_immutable_text_only")
    elif kind == "Job":
        spec = value.get("spec") or {}
        pod = (spec.get("template") or {}).get("spec") or {}
        containers = pod.get("containers")
        if (
            spec.get("backoffLimit") != 0
            or pod.get("restartPolicy") != "Never"
            or pod.get("preemptionPolicy") != "Never"
            or pod.get("priorityClassName") != "fleet-serve-low"
            or not isinstance(containers, list)
            or not containers
            or any(not isinstance(container, dict) for container in containers)
            or _resources_have_gpu(
                [container.get("resources", {}) for container in containers or []]
            )
        ):
            raise RelayError("job_safety_contract_invalid")
    else:
        spec = value.get("spec") or {}
        if (
            spec.get("type", "ClusterIP") != "ClusterIP"
            or any(
                key in spec
                for key in (
                    "externalIPs",
                    "externalName",
                    "loadBalancerIP",
                    "loadBalancerClass",
                )
            )
            or any("nodePort" in port for port in spec.get("ports", []))
        ):
            raise RelayError("service_must_be_internal_clusterip")
    return value


def load_objects(path: Path) -> list[dict[str, Any]]:
    # Kept local so the in-cluster relay path remains standard-library only.
    import yaml

    documents = list(yaml.safe_load_all(path.read_text()))
    objects: list[Any] = []
    for document in documents:
        if document is None:
            continue
        if isinstance(document, dict) and document.get("kind") == "List":
            objects.extend(document.get("items") or [])
        else:
            objects.append(document)
    if not objects:
        raise RelayError("rendered_object_set_empty")
    validated = [validate_object(item) for item in objects]
    identities = [
        (item["apiVersion"], item["kind"], item["metadata"]["namespace"], item["metadata"]["name"])
        for item in validated
    ]
    if len(identities) != len(set(identities)):
        raise RelayError("rendered_object_identity_duplicate")
    return validated


def build_envelope(objects: list[dict[str, Any]], allowlist: dict[str, Any]) -> dict[str, Any]:
    rows = allowlist.get("objects")
    if (
        allowlist.get("schema_version") != ALLOWLIST_SCHEMA
        or allowlist.get("allowlist_sha256") != digest_without(allowlist, "allowlist_sha256")
        or allowlist.get("namespace") != NAMESPACE
        or allowlist.get("allie_dev") != {"name": ALLIE_NAME, "uid": ALLIE_UID}
        or not isinstance(rows, list)
    ):
        raise RelayError("allowlist_header_invalid")
    expected: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "api_version",
            "kind",
            "namespace",
            "name",
            "sha256",
        }:
            raise RelayError("allowlist_row_invalid")
        identity = (
            row.get("api_version"),
            row.get("kind"),
            row.get("namespace"),
            row.get("name"),
        )
        digest = row.get("sha256")
        if (
            identity[:2] not in ALLOWED_IDENTITIES
            or identity[2] != NAMESPACE
            or not isinstance(identity[3], str)
            or NAME_RE.fullmatch(identity[3]) is None
            or not isinstance(digest, str)
            or SHA_RE.fullmatch(digest) is None
            or identity in expected
        ):
            raise RelayError("allowlist_row_invalid")
        expected[identity] = digest
    observed: dict[tuple[str, str, str, str], str] = {}
    for item in objects:
        validate_object(item)
        identity = (
            item["apiVersion"],
            item["kind"],
            item["metadata"]["namespace"],
            item["metadata"]["name"],
        )
        observed[identity] = sha256(canonical_json(item))
    if expected != observed:
        raise RelayError("rendered_objects_do_not_match_frozen_allowlist")
    body = {
        "schema_version": ENVELOPE_SCHEMA,
        "namespace": NAMESPACE,
        "allie_dev": {"name": ALLIE_NAME, "uid": ALLIE_UID},
        "allowlist_sha256": allowlist["allowlist_sha256"],
        "objects": objects,
        "object_digests": rows,
    }
    body["envelope_sha256"] = digest_without(body, "envelope_sha256")
    if len(canonical_json(body)) > MAX_ENVELOPE_BYTES:
        raise RelayError("relay_envelope_too_large")
    return body


def validate_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RelayError("relay_envelope_invalid")
    if (
        value.get("schema_version") != ENVELOPE_SCHEMA
        or value.get("namespace") != NAMESPACE
        or value.get("allie_dev") != {"name": ALLIE_NAME, "uid": ALLIE_UID}
        or value.get("envelope_sha256") != digest_without(value, "envelope_sha256")
    ):
        raise RelayError("relay_envelope_invalid")
    allowlist = {
        "schema_version": ALLOWLIST_SCHEMA,
        "namespace": NAMESPACE,
        "allie_dev": value["allie_dev"],
        "objects": value.get("object_digests"),
    }
    allowlist["allowlist_sha256"] = digest_without(allowlist, "allowlist_sha256")
    if value.get("allowlist_sha256") != allowlist["allowlist_sha256"]:
        raise RelayError("relay_envelope_allowlist_digest_invalid")
    expected = build_envelope(value.get("objects") or [], allowlist)
    if expected != value:
        raise RelayError("relay_envelope_drifted")
    return value


def validate_allie_pod(value: dict[str, Any]) -> None:
    status = value.get("status") or {}
    statuses = status.get("containerStatuses") or []
    if (
        (value.get("metadata") or {}).get("name") != ALLIE_NAME
        or (value.get("metadata") or {}).get("namespace") != NAMESPACE
        or (value.get("metadata") or {}).get("uid") != ALLIE_UID
        or (value.get("spec") or {}).get("serviceAccountName") != "default"
        or status.get("phase") != "Running"
        or not statuses
        or any(row.get("restartCount") != 0 or row.get("ready") is not True for row in statuses)
    ):
        raise RelayError("allie_dev_identity_or_freshness_invalid")


def _resource_path(item: dict[str, Any], *, collection: bool) -> str:
    name = urllib.parse.quote(item["metadata"]["name"], safe="")
    if item["kind"] == "ConfigMap":
        base = f"/api/v1/namespaces/{NAMESPACE}/configmaps"
    elif item["kind"] == "Service":
        base = f"/api/v1/namespaces/{NAMESPACE}/services"
    else:
        base = f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs"
    return base if collection else f"{base}/{name}"


class InClusterClient:
    """Minimal API client whose projected credential never leaves the Pod."""

    def __init__(self) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise RelayError("in_cluster_service_unavailable")
        # This credential is consumed only inside this process and is never
        # returned, logged, persisted, or included in an exception.
        token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
        self.base = f"https://{host}:{port}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.context = ssl.create_default_context(
            cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        )

    def request(
        self, method: str, path: str, body: bytes | None = None
    ) -> tuple[int, dict[str, Any]]:
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path, method=method, headers=headers, data=body
        )
        try:
            with urllib.request.urlopen(request, timeout=30, context=self.context) as response:
                raw = response.read(MAX_ENVELOPE_BYTES + 1)
                status_code = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read(64 * 1024 + 1)
            status_code = exc.code
        if len(raw) > MAX_ENVELOPE_BYTES:
            raise RelayError("kubernetes_response_too_large")
        parsed = strict_json(raw)
        if not isinstance(parsed, dict):
            raise RelayError("kubernetes_response_invalid")
        return status_code, parsed


def _uid(value: Any) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise RelayError("kubernetes_uid_invalid") from exc
    if parsed.int == 0:
        raise RelayError("kubernetes_uid_invalid")
    return str(parsed)


def _validate_created_identity(value: dict[str, Any], item: dict[str, Any]) -> str:
    metadata = value.get("metadata") or {}
    if (
        value.get("apiVersion") != item["apiVersion"]
        or value.get("kind") != item["kind"]
        or metadata.get("name") != item["metadata"]["name"]
        or metadata.get("namespace") != NAMESPACE
    ):
        raise RelayError("kubernetes_create_identity_mismatch")
    return _uid(metadata.get("uid"))


def create_in_cluster(
    envelope: dict[str, Any], *, client: InClusterClient | Any | None = None
) -> dict[str, Any]:
    envelope = validate_envelope(envelope)
    client = client or InClusterClient()
    allie_status, allie = client.request("GET", f"/api/v1/namespaces/{NAMESPACE}/pods/{ALLIE_NAME}")
    if allie_status != 200:
        raise RelayError("allie_dev_readback_failed")
    validate_allie_pod(allie)
    created = []
    digests = {
        (row["api_version"], row["kind"], row["namespace"], row["name"]): row["sha256"]
        for row in envelope["object_digests"]
    }
    for item in envelope["objects"]:
        identity = (
            item["apiVersion"],
            item["kind"],
            item["metadata"]["namespace"],
            item["metadata"]["name"],
        )
        status_code, _ = client.request("GET", _resource_path(item, collection=False))
        if status_code != 404:
            raise RelayError("create_once_object_not_absent")
        status_code, response = client.request(
            "POST", _resource_path(item, collection=True), canonical_json(item)
        )
        if status_code != 201:
            raise RelayError("kubernetes_create_failed_partial_objects_preserved")
        uid = _validate_created_identity(response, item)
        get_status, readback = client.request("GET", _resource_path(item, collection=False))
        if get_status != 200 or _validate_created_identity(readback, item) != uid:
            raise RelayError("kubernetes_uid_readback_failed_partial_objects_preserved")
        created.append(
            {
                "api_version": identity[0],
                "kind": identity[1],
                "namespace": identity[2],
                "name": identity[3],
                "uid": uid,
                "source_sha256": digests[identity],
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CREATED",
        "transport": "in_cluster_service_account",
        "allie_dev": {"name": ALLIE_NAME, "uid": ALLIE_UID, "fresh": True},
        "envelope_sha256": envelope["envelope_sha256"],
        "allowlist_sha256": envelope["allowlist_sha256"],
        "objects": created,
        "create_only": True,
        "uids_read_back": True,
        "partial_failure_policy": "preserve_and_reconcile_never_repeat",
        "credentials_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def _run_json(argv: list[str], *, input_value: dict[str, Any] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        argv,
        input=None if input_value is None else canonical_json(input_value),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RelayError("relay_transport_failed")
    value = strict_json(result.stdout)
    if not isinstance(value, dict):
        raise RelayError("relay_transport_response_invalid")
    return value


def local_can_create(objects: list[dict[str, Any]]) -> bool:
    try:
        for kind in sorted({item["kind"].lower() for item in objects}):
            result = subprocess.run(
                ["kubectl", "auth", "can-i", "create", kind, "-n", NAMESPACE],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if result.returncode != 0 or result.stdout.strip() != "yes":
                return False
    except OSError:
        return False
    return True


def create_local(envelope: dict[str, Any]) -> dict[str, Any]:
    envelope = validate_envelope(envelope)
    created = []
    digests = {
        (row["api_version"], row["kind"], row["namespace"], row["name"]): row["sha256"]
        for row in envelope["object_digests"]
    }
    for item in envelope["objects"]:
        name = item["metadata"]["name"]
        kind = item["kind"].lower()
        absent = subprocess.run(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                kind,
                name,
                "--ignore-not-found",
                "-o",
                "name",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if absent.returncode != 0:
            raise RelayError("create_once_absence_check_failed")
        if absent.stdout.strip():
            raise RelayError("create_once_object_not_absent")
        response = _run_json(
            ["kubectl", "-n", NAMESPACE, "create", "-f", "-", "-o", "json"],
            input_value=item,
        )
        uid = _validate_created_identity(response, item)
        readback = _run_json(["kubectl", "-n", NAMESPACE, "get", kind, name, "-o", "json"])
        if _validate_created_identity(readback, item) != uid:
            raise RelayError("kubernetes_uid_readback_failed_partial_objects_preserved")
        identity = (item["apiVersion"], item["kind"], NAMESPACE, name)
        created.append(
            {
                "api_version": identity[0],
                "kind": identity[1],
                "namespace": NAMESPACE,
                "name": name,
                "uid": uid,
                "source_sha256": digests[identity],
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CREATED",
        "transport": "local_kubectl",
        "allie_dev": None,
        "envelope_sha256": envelope["envelope_sha256"],
        "allowlist_sha256": envelope["allowlist_sha256"],
        "objects": created,
        "create_only": True,
        "uids_read_back": True,
        "partial_failure_policy": "preserve_and_reconcile_never_repeat",
        "credentials_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def create_via_allie(envelope: dict[str, Any]) -> dict[str, Any]:
    pod = _run_json(["kubectl", "-n", NAMESPACE, "get", "pod", ALLIE_NAME, "-o", "json"])
    validate_allie_pod(pod)
    source = Path(__file__).read_text()
    return _run_json(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            "-i",
            ALLIE_NAME,
            "--",
            "python3",
            "-c",
            source,
            "in-cluster-create",
        ],
        input_value=envelope,
    )


def validate_receipt(
    value: dict[str, Any], envelope: dict[str, Any], *, expected_transport: str
) -> dict[str, Any]:
    expected = {
        (row["api_version"], row["kind"], row["namespace"], row["name"]): row["sha256"]
        for row in envelope["object_digests"]
    }
    observed: dict[tuple[str, str, str, str], str] = {}
    rows = value.get("objects")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "transport",
            "allie_dev",
            "envelope_sha256",
            "allowlist_sha256",
            "objects",
            "create_only",
            "uids_read_back",
            "partial_failure_policy",
            "credentials_included",
            "receipt_sha256",
        }
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("status") != "CREATED"
        or value.get("transport") != expected_transport
        or value.get("envelope_sha256") != envelope["envelope_sha256"]
        or value.get("allowlist_sha256") != envelope["allowlist_sha256"]
        or value.get("create_only") is not True
        or value.get("uids_read_back") is not True
        or value.get("partial_failure_policy") != "preserve_and_reconcile_never_repeat"
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest_without(value, "receipt_sha256")
        or not isinstance(rows, list)
    ):
        raise RelayError("relay_receipt_invalid")
    if expected_transport == "in_cluster_service_account":
        if value.get("allie_dev") != {"name": ALLIE_NAME, "uid": ALLIE_UID, "fresh": True}:
            raise RelayError("relay_receipt_allie_binding_invalid")
    elif value.get("allie_dev") is not None:
        raise RelayError("local_receipt_allie_binding_invalid")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "api_version",
            "kind",
            "namespace",
            "name",
            "uid",
            "source_sha256",
        }:
            raise RelayError("relay_receipt_object_invalid")
        identity = (row["api_version"], row["kind"], row["namespace"], row["name"])
        if identity in observed:
            raise RelayError("relay_receipt_object_duplicate")
        _uid(row["uid"])
        observed[identity] = row["source_sha256"]
    if observed != expected:
        raise RelayError("relay_receipt_objects_mismatch")
    return value


def create_with_fallback(
    envelope: dict[str, Any],
    *,
    transport: str = "auto",
    can_create: Callable[[list[dict[str, Any]]], bool] = local_can_create,
    local_creator: Callable[[dict[str, Any]], dict[str, Any]] = create_local,
    relay_creator: Callable[[dict[str, Any]], dict[str, Any]] = create_via_allie,
) -> dict[str, Any]:
    envelope = validate_envelope(envelope)
    if transport not in {"auto", "local", "allie"}:
        raise RelayError("relay_transport_invalid")
    if transport == "local" or (transport == "auto" and can_create(envelope["objects"])):
        return validate_receipt(
            local_creator(envelope), envelope, expected_transport="local_kubectl"
        )
    return validate_receipt(
        relay_creator(envelope), envelope, expected_transport="in_cluster_service_account"
    )


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        raw = canonical_json(value) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise RelayError("receipt_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "validate", "in-cluster-create"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--transport", choices=("auto", "local", "allie"), default="auto")
    args = parser.parse_args(argv)
    if args.command == "in-cluster-create":
        raw = sys.stdin.buffer.read(MAX_ENVELOPE_BYTES + 1)
        if len(raw) > MAX_ENVELOPE_BYTES:
            raise RelayError("relay_envelope_too_large")
        print(json.dumps(create_in_cluster(validate_envelope(strict_json(raw))), sort_keys=True))
        return 0
    if args.manifest is None or args.allowlist is None:
        parser.error(f"{args.command} requires --manifest and --allowlist")
    allowlist = strict_json(args.allowlist.read_bytes())
    envelope = build_envelope(load_objects(args.manifest), allowlist)
    if args.command == "validate":
        print(json.dumps({"status": "VALID", "envelope_sha256": envelope["envelope_sha256"]}))
        return 0
    if args.receipt is None:
        parser.error("create requires --receipt")
    receipt = create_with_fallback(envelope, transport=args.transport)
    write_once(args.receipt, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "transport": receipt["transport"],
                "object_count": len(receipt["objects"]),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
