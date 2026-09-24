"""Versioned evaluator tests use synthetic identities and make no live calls."""

from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.fleet import cluster_entry_v2, evaluate_v2
from evals.fleet import evaluate as evaluate_v1

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "configs/evaluation/fleet-opencode-evaluator-runtime-v2.json"


def _plan() -> dict:
    return {
        "images": {
            "agent": "registry.invalid/agent@sha256:" + "d" * 64,
            "proxy": "registry.invalid/proxy@sha256:" + "e" * 64,
        },
        "treatment": {
            "harness_version": "1.18.27",
            "release_asset_sha256": "sha256:" + "b" * 64,
        },
    }


def _inspect(args: list[str]) -> SimpleNamespace:
    image = args[3]
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "Id": image if image.startswith("sha256:") else "sha256:" + "f" * 64,
                "RepoDigests": [image],
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {"Labels": {"cyber.opencode.release-sha256": "sha256:" + "b" * 64}},
            }
        ),
    )


def test_v1_source_bytes_remain_frozen_and_v2_closure_is_distinct() -> None:
    v1_path = Path(evaluate_v1.__file__)
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == (
        "33b88cf783bffdfa7fc194e8107afbd6721e762cb5d7020da9057c8d797e761e"
    )
    runtime = evaluate_v2.runtime_identity()
    assert runtime["evaluate.py"] == hashlib.sha256(v1_path.read_bytes()).hexdigest()
    assert (
        runtime["evaluate_v2.py"]
        == hashlib.sha256(Path(evaluate_v2.__file__).read_bytes()).hexdigest()
    )
    assert runtime != evaluate_v1.runtime_identity()


def test_v2_probe_checks_semantic_startup_without_database_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_BIND_ROOT", str(tmp_path))

    def run(args: list[str], **kwargs) -> SimpleNamespace:
        if args[1:3] == ["image", "inspect"]:
            return _inspect(args)
        assert args[:3] == ["docker", "run", "--rm"]
        assert args[-4:-1] == ["bash", "-ceu", "--"]
        assert args[-1] == evaluate_v2.AGENT_STARTUP_PROBE
        assert "opencode db path" not in args[-1]
        assert kwargs["timeout"] == evaluate_v2.AGENT_STARTUP_PROBE_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0, stdout="1.18.27\n")

    monkeypatch.setattr(evaluate_v2.subprocess, "run", run)
    evaluate_v2.check_images(_plan())


def test_v2_probe_timeout_is_content_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_BIND_ROOT", str(tmp_path))

    def run(args: list[str], **kwargs) -> SimpleNamespace:
        if args[1:3] == ["image", "inspect"]:
            return _inspect(args)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"], output="private-output")

    monkeypatch.setattr(evaluate_v2.subprocess, "run", run)
    with pytest.raises(RuntimeError) as raised:
        evaluate_v2.check_images(_plan())
    assert str(raised.value) == "harness startup probe exceeded its safety deadline"
    assert "private-output" not in str(raised.value)


def test_v2_cluster_entry_scopes_runtime_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    original_runtime_identity = cluster_entry_v2.v1.evaluate.runtime_identity
    original_check_images = cluster_entry_v2.v1.evaluate.check_images

    def execute(args: Namespace) -> dict:
        assert args.marker == "synthetic"
        assert cluster_entry_v2.v1.evaluate is evaluate_v1
        assert cluster_entry_v2.v1.evaluate.runtime_identity is evaluate_v2.runtime_identity
        assert cluster_entry_v2.v1.evaluate.check_images is evaluate_v2.check_images
        return {"status": "passed"}

    monkeypatch.setattr(cluster_entry_v2.v1, "execute", execute)
    assert cluster_entry_v2.execute(Namespace(marker="synthetic")) == {"status": "passed"}
    assert cluster_entry_v2.v1.evaluate.runtime_identity is original_runtime_identity
    assert cluster_entry_v2.v1.evaluate.check_images is original_check_images


def test_v2_runtime_binding_is_self_digesting_and_exact() -> None:
    value = json.loads(BINDING.read_text())
    body = {key: item for key, item in value.items() if key != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert value["sha256"] == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert value["entrypoint"] == "evals.fleet.cluster_entry_v2"
    assert value["runtime_files_sha256"] == {
        f"evals/fleet/{name}": digest for name, digest in evaluate_v2.runtime_identity().items()
    }
    assert value["entrypoint_files_sha256"] == {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in ("evals/fleet/cluster_entry.py", "evals/fleet/cluster_entry_v2.py")
    }
    evidence = json.loads((ROOT / value["qualification_evidence"]["path"]).read_text())
    evidence_body = {key: item for key, item in evidence.items() if key != "sha256"}
    evidence_canonical = json.dumps(evidence_body, sort_keys=True, separators=(",", ":")).encode()
    assert evidence["sha256"] == hashlib.sha256(evidence_canonical).hexdigest()
    assert value["qualification_evidence"]["receipt_sha256"] == "sha256:" + evidence["sha256"]
    assert value["effects"] == {
        "cluster_mutations": 0,
        "eval_launches": 0,
        "model_calls": 0,
    }
