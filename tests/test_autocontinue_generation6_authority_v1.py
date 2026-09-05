from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation5_canary as generation5
from evals.fleet import autocontinue_generation6_authority_package_v1 as package
from evals.fleet import autocontinue_generation6_authority_v1 as authority
from evals.fleet import autocontinue_generation6_canary as generation6

ROOT = Path.cwd()


@pytest.fixture(scope="module")
def built() -> dict:
    return package.build_package(ROOT)


def test_generation5_cli_failure_is_sealed_and_content_free() -> None:
    receipt = generation6.load(ROOT / generation6.FAILURE_PATH)
    generation6.validate_failure(receipt, ROOT)
    assert receipt["side_effects"] == {
        "model_calls": 0,
        "sessions": 0,
        "verifier_executions": 0,
        "generation5_claims": 0,
        "output_roots": 0,
    }
    assert receipt["root_cause"]["logs_used"] is False
    assert receipt["privacy"] == {
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }


def test_generation6_preserves_cells_and_uses_fresh_executions() -> None:
    for model, path in generation6.G6_SPEC_PATHS.items():
        spec = generation6.load(ROOT / path)
        plan = generation6.validate_spec(spec, ROOT)
        old = generation5.load(ROOT / generation5.G5_SPEC_PATHS[model])
        assert spec["statistical_cell"] == old["statistical_cell"]
        assert spec["execution"]["execution_generation"] == 6
        assert spec["execution"]["execution_id"] != old["execution"]["execution_id"]
        assert plan["attempts"][0]["run_id"] == generation6.EXPECTED[model]["run_id"]
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["docker_cli_source_image"] == generation6.DOCKER_IMAGE


def test_tampered_failure_receipt_is_rejected() -> None:
    receipt = generation6.load(ROOT / generation6.FAILURE_PATH)
    changed = copy.deepcopy(receipt)
    changed["side_effects"]["sessions"] = 1
    changed["receipt_sha256"] = generation6.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="failure tombstone"):
        generation6.validate_failure(changed, ROOT)


def test_root_authorization_is_exact() -> None:
    receipt = authority.load(ROOT / authority.AUTH_PATH)
    plans = authority.validate_root_authorization(receipt, ROOT)
    assert len(plans) == 2
    assert receipt["scope"]["bulk_release_authorized"] is False


def test_package_contains_successor_dependencies_and_fits(built: dict) -> None:
    paths = {
        entry["source_path"]
        for obj in built["object_manifests"].values()
        for entry in obj["entries"]
    }
    assert set(package.G6_PATHS) <= paths
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.PACKAGE_OBJECT_LIMIT


def test_manifest_copies_exact_pinned_cli_before_evaluator() -> None:
    manifest = yaml.safe_load((ROOT / authority.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for job in manifest["items"]:
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        cli, dind = pod["initContainers"]
        assert cli["name"] == "docker-cli"
        assert cli["image"] == generation6.DOCKER_IMAGE
        assert dind["name"] == "dind"
        assert dind["image"] == generation6.DOCKER_IMAGE
        evaluator = pod["containers"][0]
        env = {row["name"]: row for row in evaluator["env"]}
        assert env["PATH"]["value"].startswith("/docker-cli/bin:")
        assert env["DOCKER_CONFIG"]["value"] == "/workspace/docker-config"
        cli_mounts = [
            row for row in evaluator["volumeMounts"] if row["name"] == "docker-cli"
        ]
        assert len(cli_mounts) == 1
        assert cli_mounts[0]["mountPath"] == "/docker-cli"
        assert all(row["readOnly"] is True for row in cli_mounts)
        assert all("apt" not in str(container) for container in pod["initContainers"])


def test_runner_proves_cli_before_output_or_claim() -> None:
    run = (ROOT / authority.RUN_PATH).read_text()
    cli = 'test "$(command -v docker)" = /docker-cli/bin/docker'
    claim = "-m evals.fleet.autocontinue_generation6_authority_v1 run"
    assert run.index(cli) < run.index("docker info") < run.index(claim)
    assert "apt-get" not in run and "curl" not in run


def test_claim_is_create_once_and_binds_generation6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = "qwen3.8-27b"
    spec = generation6.load(ROOT / generation6.G6_SPEC_PATHS[model])
    plan = generation6.validate_spec(spec, ROOT)
    root_authority = authority.load(ROOT / authority.AUTH_PATH)
    fake_release = {"receipt_sha256": "sha256:" + "1" * 64}
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(authority, "validate_release", lambda *args: None)
    monkeypatch.setattr(
        authority,
        "authority_binding",
        lambda *args: {
            "path": authority.AUTH_PATH,
            "file_sha256": "sha256:" + "2" * 64,
            "receipt_sha256": root_authority["receipt_sha256"],
        },
    )
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    claim = authority.claim_execution(
        spec,
        plan,
        fake_release,
        "sha256:" + "3" * 64,
        root_authority,
        ROOT,
        "a" * 40,
        claim_root=claim_root,
    )
    assert claim["execution_generation"] == 6
    assert claim["execution_id"] == spec["execution"]["execution_id"]
    with pytest.raises(RuntimeError, match="already claimed"):
        authority.claim_execution(
            spec,
            plan,
            fake_release,
            "sha256:" + "3" * 64,
            root_authority,
            ROOT,
            "a" * 40,
            claim_root=claim_root,
        )


def test_specs_are_canonical_json() -> None:
    for path in generation6.G6_SPEC_PATHS.values():
        raw = (ROOT / path).read_bytes()
        expected = json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":")
        ).encode() + b"\n"
        assert raw == expected
