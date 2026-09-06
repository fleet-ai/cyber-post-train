from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import production_blackbox_tool_catalog_v1 as production_tools

ROOT = Path(__file__).resolve().parents[1]


def test_treatment_is_exact_for_both_models() -> None:
    for model in parity.MODEL_KEYS:
        config = parity.treatment_config(model)
        assert config["model"] == exact.EXPECTED_MODELS[model]
        assert config["harness"] == {
            "name": "opencode",
            "version": "1.18.27",
            "release_asset_sha256": exact.EXPECTED_TREATMENT["release_asset_sha256"],
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v1",
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
        }


def test_mcp_catalog_is_exact_frozen_tool_schema() -> None:
    expected = production_tools.load(parity.REPO_ROOT)
    assert parity.mcp_tools() == expected
    assert [row["name"] for row in parity.mcp_tools()] == ["bash", "submit_report"]
    assert crypto.sha256(crypto.canonical_json(expected)) == (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    )
    assert set(expected[0]["inputSchema"]["properties"]) == {"script", "timeoutMs"}
    assert set(expected[1]["inputSchema"]["properties"]) == {
        "flag",
        "flags",
        "verdict",
        "explanation",
    }


def test_openai_catalog_is_exact_opencode_mcp_projection() -> None:
    expected = []
    for row in parity.mcp_tools():
        parameters = {**row["inputSchema"], "additionalProperties": False}
        expected.append(
            {
                "type": "function",
                "function": {
                    "name": f"fleet_{row['name']}",
                    "description": row["description"],
                    "parameters": parameters,
                },
            },
        )
    assert parity.expected_openai_tools() == expected
    assert crypto.sha256(crypto.canonical_json(expected)) == (
        "sha256:585574ec1a459141a2e79f4945d140864876224ebef1260be65f06c6d237610f"
    )


def test_actual_opencode_sequence_allows_only_terminal_no_tool_request() -> None:
    exact_request = {
        "model": "qwen3.8-27b",
        "tools": parity.expected_openai_tools(),
    }
    observations = [
        parity.classify_model_request_catalog(exact_request, "qwen3.8-27b") for _ in range(3)
    ]
    observations.append(
        parity.classify_model_request_catalog({"model": "qwen3.8-27b"}, "qwen3.8-27b")
    )
    assert all(row["catalog_valid"] for row in observations[:3])
    assert observations[3] == {
        "classification": "NO_TOOLS_TERMINAL_REQUEST",
        "catalog_sha256": "absent",
    }
    drifted = copy.deepcopy(exact_request)
    drifted["tools"][0]["function"]["parameters"]["required"] = []
    assert parity.classify_model_request_catalog(drifted, "qwen3.8-27b")["catalog_valid"] is False


def _server_binding() -> dict:
    return {
        "api_run_id": "ft-run-12345678",
        "rayjob_uid": "7c6bcfb9-c4a7-497c-b1d5-9f31f7a838cf",
        "head_pod_uid": "9781ee19-cd71-44df-9013-89cf351b4269",
        "service_uid": "6c244802-37ca-4607-aacd-95a37bc303b0",
        "served_id": "qwen3.8-27b",
        "model_revision": exact.EXPECTED_MODELS["qwen3.8-27b"]["revision"],
        "context_length": 262144,
    }


def test_dedicated_binding_requires_exact_nonzero_uids_and_model() -> None:
    binding = _server_binding()
    assert parity.validate_server_binding(binding, "qwen3.8-27b") == binding
    for field in ("rayjob_uid", "head_pod_uid", "service_uid"):
        changed = {**binding, field: "00000000-0000-0000-0000-000000000000"}
        with pytest.raises(parity.ActualHarnessParityError):
            parity.validate_server_binding(changed, "qwen3.8-27b")
    changed = {**binding, "served_id": "glm-5.3"}
    with pytest.raises(parity.ActualHarnessParityError):
        parity.validate_server_binding(changed, "qwen3.8-27b")


def test_rendered_settings_preserve_exact_runtime_and_only_repoint_transport() -> None:
    settings = parity.render_settings("qwen3.8-27b", 18080, 18081)
    provider = settings["provider"]["fleet-cluster"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://host.docker.internal:18080/v1"
    assert provider["models"]["qwen3.8-27b"]["limit"] == {
        "context": 262144,
        "input": 229376,
        "output": 32768,
    }
    assert settings["compaction"] == {"auto": True, "reserved": 20000}
    assert settings["mcp"]["fleet"] == {
        "type": "remote",
        "url": "http://host.docker.internal:18081/mcp",
        "enabled": True,
    }
    assert settings["permission"] == {"*": "deny", "fleet_*": "allow"}
    assert all(value is False for value in settings["tools"].values())


def test_local_image_inspection_requires_exact_immutable_amd64_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {
        "Id": parity.IMAGE_ID,
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {"User": "node", "WorkingDir": "/workspace"},
    }

    class Result:
        returncode = 0
        stdout = parity.json.dumps(observed)

    monkeypatch.setattr(parity.subprocess, "run", lambda *_args, **_kwargs: Result())
    assert parity.inspect_local_image() == {
        "image": parity.IMAGE,
        "image_id": parity.IMAGE_ID,
        "os": "linux",
        "architecture": "amd64",
        "user": "node",
        "working_dir": "/workspace",
    }

    observed["Architecture"] = "arm64"
    Result.stdout = parity.json.dumps(observed)
    with pytest.raises(parity.ActualHarnessParityError, match="opencode_image_identity_drift"):
        parity.inspect_local_image()
