from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import kubernetes_create_relay as relay


def _objects() -> list[dict]:
    return [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "chris-relay-config", "namespace": relay.NAMESPACE},
            "immutable": True,
            "data": {"plan": "reviewed"},
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": "chris-relay-job", "namespace": relay.NAMESPACE},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "priorityClassName": "fleet-serve-low",
                        "preemptionPolicy": "Never",
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "worker",
                                "image": "example.invalid/pinned@sha256:" + "1" * 64,
                            }
                        ],
                    }
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "chris-relay-service", "namespace": relay.NAMESPACE},
            "spec": {
                "type": "ClusterIP",
                "selector": {"app": "chris-relay"},
                "ports": [{"port": 80, "targetPort": 8080}],
            },
        },
    ]


def _allowlist(objects: list[dict]) -> dict:
    value = {
        "schema_version": relay.ALLOWLIST_SCHEMA,
        "namespace": relay.NAMESPACE,
        "allie_dev": {"name": relay.ALLIE_NAME, "uid": relay.ALLIE_UID},
        "objects": [
            {
                "api_version": item["apiVersion"],
                "kind": item["kind"],
                "namespace": relay.NAMESPACE,
                "name": item["metadata"]["name"],
                "sha256": relay.sha256(relay.canonical_json(item)),
            }
            for item in objects
        ],
    }
    value["allowlist_sha256"] = relay.digest_without(value, "allowlist_sha256")
    return value


def _allie() -> dict:
    return {
        "metadata": {
            "name": relay.ALLIE_NAME,
            "namespace": relay.NAMESPACE,
            "uid": relay.ALLIE_UID,
        },
        "spec": {"serviceAccountName": "default"},
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "main", "ready": True, "restartCount": 0}],
        },
    }


class FakeClient:
    def __init__(self) -> None:
        self.created: dict[str, dict] = {}

    def request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
        if path.endswith("/pods/allie-dev"):
            return 200, _allie()
        if method == "GET":
            return (200, self.created[path]) if path in self.created else (404, {"kind": "Status"})
        assert method == "POST" and body is not None
        item = json.loads(body)
        uid = {
            "ConfigMap": "11111111-1111-4111-8111-111111111111",
            "Job": "22222222-2222-4222-8222-222222222222",
            "Service": "33333333-3333-4333-8333-333333333333",
        }[item["kind"]]
        response = json.loads(body)
        response["metadata"]["uid"] = uid
        readback_path = relay._resource_path(item, collection=False)
        self.created[readback_path] = response
        return 201, response


def test_allowlist_and_envelope_are_independently_digest_bound() -> None:
    objects = _objects()
    envelope = relay.build_envelope(objects, _allowlist(objects))
    assert relay.validate_envelope(envelope) == envelope
    changed = json.loads(json.dumps(objects))
    changed[0]["data"]["plan"] = "drifted"
    with pytest.raises(relay.RelayError, match="frozen_allowlist"):
        relay.build_envelope(changed, _allowlist(objects))
    allowlist = _allowlist(objects)
    allowlist["objects"].reverse()
    with pytest.raises(relay.RelayError, match="allowlist_header_invalid"):
        relay.build_envelope(objects, allowlist)


def test_in_cluster_create_is_absence_gated_and_uid_read_back() -> None:
    objects = _objects()
    envelope = relay.build_envelope(objects, _allowlist(objects))
    client = FakeClient()
    receipt = relay.create_in_cluster(envelope, client=client)
    relay.validate_receipt(receipt, envelope, expected_transport="in_cluster_service_account")
    assert [row["kind"] for row in receipt["objects"]] == ["ConfigMap", "Job", "Service"]
    with pytest.raises(relay.RelayError, match="not_absent"):
        relay.create_in_cluster(envelope, client=client)


def test_unsafe_job_and_external_service_are_rejected() -> None:
    objects = _objects()
    objects[1]["spec"]["template"]["spec"]["containers"][0]["resources"] = {
        "limits": {"nvidia.com/gpu": 1}
    }
    with pytest.raises(relay.RelayError, match="job_safety_contract"):
        relay.validate_object(objects[1])
    objects = _objects()
    objects[2]["spec"]["ports"][0]["nodePort"] = 30001
    with pytest.raises(relay.RelayError, match="internal_clusterip"):
        relay.validate_object(objects[2])


def test_auto_transport_falls_back_to_exact_allie_relay() -> None:
    objects = _objects()
    envelope = relay.build_envelope(objects, _allowlist(objects))
    expected = relay.create_in_cluster(envelope, client=FakeClient())
    calls: list[str] = []

    def local(_: dict) -> dict:
        calls.append("local")
        raise AssertionError("local creator must not run")

    def allie(_: dict) -> dict:
        calls.append("allie")
        return expected

    receipt = relay.create_with_fallback(
        envelope,
        can_create=lambda _: False,
        local_creator=local,
        relay_creator=allie,
    )
    assert receipt == expected
    assert calls == ["allie"]


def test_receipt_writer_is_create_once(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    relay.write_once(target, {"safe": True})
    with pytest.raises(FileExistsError):
        relay.write_once(target, {"safe": False})
