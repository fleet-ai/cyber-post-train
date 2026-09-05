from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evals/fleet/autocontinue_hosted_health.py"
SPEC = importlib.util.spec_from_file_location("autocontinue_hosted_health", MODULE_PATH)
assert SPEC and SPEC.loader
health = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(health)


class Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self.payload = json.dumps(payload).encode()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        assert limit == 4 * 1024 * 1024 + 1
        return self.payload


def opener_for(account: dict[str, Any], models: dict[str, Any]):
    calls: list[tuple[str, str, int]] = []

    def opener(request: Any, *, timeout: int) -> Response:
        calls.append((request.method, request.full_url, timeout))
        if request.full_url == health.ACCOUNT_URL:
            return Response(account)
        if request.full_url == health.MODELS_URL:
            return Response(models)
        raise AssertionError(request.full_url)

    return opener, calls


def valid_account() -> dict[str, Any]:
    return {"team_name": "fleet", "team_id": health.FLEET_TEAM_ID}


def valid_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "qwen3.8-27b", "max_model_len": 262144},
            {"id": "glm-5.3"},
            {"id": "unrelated-model", "context_length": 1000},
        ],
    }


def test_qualify_is_two_gets_and_score_blind() -> None:
    opener, calls = opener_for(valid_account(), valid_models())
    receipt = health.qualify(
        "not-recorded",
        job_uid="job-uid",
        pod_uid="pod-uid",
        secret_uid="secret-uid",
        opener=opener,
    )

    assert calls == [
        ("GET", health.ACCOUNT_URL, 30),
        ("GET", health.MODELS_URL, 30),
    ]
    assert receipt["status"] == "PASSED"
    assert receipt["fleet_account"] == {
        "team_name": "fleet",
        "team_id": health.FLEET_TEAM_ID,
        "authenticated_get_succeeded": True,
    }
    assert receipt["hosted_inference"]["models"] == {
        "qwen3.8-27b": {
            "available": True,
            "context_length_observable": True,
            "context_length": 262144,
        },
        "glm-5.3": {
            "available": True,
            "context_length_observable": False,
            "context_length": None,
        },
    }
    assert receipt["request_counts"]["chat_completions"] == 0
    assert receipt["request_counts"]["fleet_task_or_scoring"] == 0
    assert receipt["credentials_included"] is False
    assert "not-recorded" not in json.dumps(receipt)
    assert receipt["receipt_sha256"] == health.digest_without(receipt, "receipt_sha256")


@pytest.mark.parametrize(
    ("account", "models", "code"),
    [
        (
            {"team_name": "fleet", "team_id": "wrong"},
            valid_models(),
            "fleet_team_identity_mismatch",
        ),
        (
            valid_account(),
            {"data": [{"id": "qwen3.8-27b"}]},
            "glm-5.3_availability_mismatch",
        ),
        (
            valid_account(),
            {
                "data": [
                    {"id": "qwen3.8-27b", "context_length": 131072},
                    {"id": "glm-5.3"},
                ]
            },
            "qwen3.8-27b_context_length_mismatch",
        ),
        (
            valid_account(),
            {
                "data": [
                    {"id": "qwen3.8-27b"},
                    {"id": "glm-5.3", "metadata": {"context_window": 131072}},
                ]
            },
            "glm-5.3_context_length_mismatch",
        ),
    ],
)
def test_qualify_fails_closed(account: dict[str, Any], models: dict[str, Any], code: str) -> None:
    opener, _calls = opener_for(account, models)
    with pytest.raises(health.GateError, match=code):
        health.qualify(
            "secret-value",
            job_uid="job-uid",
            pod_uid="pod-uid",
            secret_uid="secret-uid",
            opener=opener,
        )


def test_duplicate_model_row_fails_closed() -> None:
    models = valid_models()
    models["data"].append(copy.deepcopy(models["data"][0]))
    opener, _calls = opener_for(valid_account(), models)
    with pytest.raises(health.GateError, match="qwen3.8-27b_availability_mismatch"):
        health.qualify(
            "secret-value",
            job_uid="job-uid",
            pod_uid="pod-uid",
            secret_uid="secret-uid",
            opener=opener,
        )


def test_manifest_is_low_cost_high_priority_and_uid_bound() -> None:
    path = ROOT / "evals/fleet/cluster/opencode-autocontinue-hosted-health-v1.yaml"
    document = yaml.safe_load(path.read_text())
    assert document["kind"] == "Job"
    assert document["metadata"] == {
        "name": "chris-cyber-opencode-ac-hosted-health-v1",
        "namespace": "fleet-train-jobs",
        "labels": {
            "cyber-post-train.fleet.ai/experiment": "opencode-ac-hosted-health-v1",
            "cyber-post-train.fleet.ai/owner": "chris",
            "kueue.x-k8s.io/queue-name": "training-lq",
        },
    }
    spec = document["spec"]
    assert spec["backoffLimit"] == 0
    pod = spec["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-train-high"
    assert "preemptionPolicy" not in pod
    container = pod["containers"][0]
    assert container["resources"] == {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "500m", "memory": "256Mi"},
    }
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["FLEET_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "chris-cyber-opencode-evals-v2",
        "key": "FLEET_API_KEY",
    }
    assert env["EXPECTED_SECRET_UID"]["value"] == "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
    assert env["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['batch.kubernetes.io/controller-uid']"
    )
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"


def test_submitter_never_reads_secret_value_and_uses_server_preview() -> None:
    script = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_hosted_health_v1.sh"
    ).read_text()
    assert "get secret \"$SECRET\" -o jsonpath='{.metadata.uid}'" in script
    assert ".data" not in script
    assert "base64" not in script
    assert "--from-file=probe.py=" in script
    assert script.count("--dry-run=server") == 2
    assert 'git -C "$ROOT" status --porcelain' in script


def test_canonical_self_digest() -> None:
    value = {"z": 1, "a": [True, None]}
    expected = "sha256:" + hashlib.sha256(b'{"a":[true,null],"z":1}').hexdigest()
    assert health.digest_without(value, "receipt_sha256") == expected
