from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.external_ctf import opencode_scored
from evals.external_ctf.protocol import load_protocol


def test_settings_use_only_the_internal_proxy_and_null_seed_policy() -> None:
    settings = opencode_scored._settings("served-model", 262144, 32768)  # noqa: SLF001
    provider = settings["provider"]["external-ctf"]

    assert provider["options"] == {
        "baseURL": "http://model-proxy:8877/v1",
        "apiKey": "local-proxy-only",
        "timeout": False,
        "chunkTimeout": 300000,
    }
    assert provider["models"]["served-model"]["limit"] == {
        "context": 262144,
        "output": 32768,
        "input": 229376,
    }
    assert settings["compaction"] == {"auto": True, "reserved": 20000}


def test_proxy_is_the_only_container_given_the_provider_credential(monkeypatch) -> None:
    created: list[str] = []

    class Sandbox:
        network = "private-network"

        def create(self, _role: str, *args: str) -> str:
            created.extend(args)
            return "proxy"

    monkeypatch.setattr(
        opencode_scored.rt,
        "image_lock",
        lambda *_args, **_kwargs: {"repository_digest": opencode_scored.PROXY_IMAGE},
    )
    monkeypatch.setattr(
        opencode_scored.rt,
        "docker",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    assert (
        opencode_scored._proxy(  # noqa: SLF001
            Sandbox(),  # type: ignore[arg-type]
            credential="secret-sentinel",
            model="served-model",
            requests=150,
            output=32768,
        )
        == "proxy"
    )
    assert "FIXED_AUTH_VALUE=Bearer secret-sentinel" in created
    assert not any(item.startswith("FLEET_API_KEY=") for item in created)
    assert "FIXED_ALLOWED_PATHS=/v1/chat/completions,/v1/models" in created
    assert "--read-only" in created
    assert "/var/run/docker.sock" not in created


@pytest.mark.parametrize(
    ("runner", "first_boundary"),
    [
        ("run_nyu", "load_task"),
        ("run_cybench", "expected_remote_runtime_evidence"),
    ],
)
def test_provider_credential_is_removed_before_benchmark_runtime(
    monkeypatch, tmp_path: Path, runner: str, first_boundary: str
) -> None:
    protocol = load_protocol()
    monkeypatch.setenv("FLEET_API_KEY", "secret-sentinel")

    def stopped(*_args, **_kwargs):
        assert "FLEET_API_KEY" not in opencode_scored.os.environ
        raise RuntimeError("boundary-observed")

    if first_boundary == "load_task":
        monkeypatch.setattr(opencode_scored.nyu_adapter, first_boundary, stopped)
        task_id = protocol["benchmarks"]["nyu_ctf_web_test"]["task_ids"][6]
        qualification = {}
    else:
        monkeypatch.setattr(opencode_scored.cy_source, first_boundary, stopped)
        task_id = protocol["benchmarks"]["cybench_web"]["task_ids"][2]
        qualification = {}

    with pytest.raises(RuntimeError, match="boundary-observed"):
        getattr(opencode_scored, runner)(protocol, tmp_path, task_id, "base", qualification)


def test_source_identity_binds_executor_and_proxy_bytes() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", opencode_scored.source_sha256())
