from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation6_canary as generation6
from evals.fleet import autocontinue_generation7_authority_package_v1 as package
from evals.fleet import autocontinue_generation7_authority_v1 as authority
from evals.fleet import autocontinue_generation7_canary as generation7

ROOT = Path.cwd()


@pytest.fixture(scope="module")
def built() -> dict:
    return package.build_package(ROOT)


def test_generation6_cli_volume_failure_is_sealed_and_content_free() -> None:
    receipt = generation7.load(ROOT / generation7.FAILURE_PATH)
    generation7.validate_failure(receipt, ROOT)
    assert receipt["side_effects"] == {
        "model_calls": 0,
        "sessions": 0,
        "verifier_executions": 0,
        "generation6_claims": 0,
        "output_roots": 0,
    }
    assert receipt["root_cause"]["logs_used"] is False
    assert receipt["root_cause"]["exact_copied_binary_bytes"] == 105_594_160
    assert receipt["root_cause"]["successor_size_limit_bytes"] == 256 * 1024 * 1024
    assert {(row["job_uid"], row["pod_uid"]) for row in receipt["jobs"]} == {
        (
            "674ac8ec-0d20-49e0-a260-b77d96e9cd81",
            "2a2e94ed-c86d-4159-80d5-45ef94f89bf8",
        ),
        (
            "859378a9-b2ce-4edc-9b25-18d701201e5f",
            "ea288ab8-aea8-45b4-84b8-7d011a2a5e8f",
        ),
    }
    assert all(
        row["generation6_output_root_absent"] is True
        and row["generation6_execution_claim_absent"] is True
        and row["exact_treatment_sessions"] == 0
        for row in receipt["jobs"]
    )
    assert receipt["privacy"] == {
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }


def test_generation7_preserves_cells_and_uses_fresh_executions() -> None:
    for model, path in generation7.G7_SPEC_PATHS.items():
        spec = generation7.load(ROOT / path)
        plan = generation7.validate_spec(spec, ROOT)
        old = generation6.load(ROOT / generation6.G6_SPEC_PATHS[model])
        assert spec["statistical_cell"] == old["statistical_cell"]
        assert spec["execution"]["execution_generation"] == 7
        assert spec["execution"]["execution_id"] != old["execution"]["execution_id"]
        assert plan["attempts"][0]["run_id"] == generation7.EXPECTED[model]["run_id"]
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["docker_cli_source_image"] == generation7.DOCKER_IMAGE
        assert plan["execution"]["docker_cli_copied_binary_bytes"] == 105_594_160
        assert plan["execution"]["docker_cli_emptydir_size_limit_bytes"] == 256 * 1024 * 1024


def test_tampered_failure_receipt_is_rejected() -> None:
    receipt = generation7.load(ROOT / generation7.FAILURE_PATH)
    changed = copy.deepcopy(receipt)
    changed["side_effects"]["sessions"] = 1
    changed["receipt_sha256"] = generation7.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="failure tombstone"):
        generation7.validate_failure(changed, ROOT)


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
    assert set(package.G7_PATHS) <= paths
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
        assert cli["image"] == generation7.DOCKER_IMAGE
        assert "105594160" in cli["args"][0]
        assert dind["name"] == "dind"
        assert dind["image"] == generation7.DOCKER_IMAGE
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
        volume = next(row for row in pod["volumes"] if row["name"] == "docker-cli")
        assert volume["emptyDir"]["sizeLimit"] == "256Mi"
        limit_bytes = int(volume["emptyDir"]["sizeLimit"].removesuffix("Mi")) * 1024 * 1024
        assert limit_bytes >= 2 * 105_594_160


def test_submitter_rechecks_exact_g6_uid_owners_and_absence() -> None:
    submitter = (ROOT / authority.SUBMIT_PATH).read_text()
    for value in (
        "674ac8ec-0d20-49e0-a260-b77d96e9cd81",
        "2a2e94ed-c86d-4159-80d5-45ef94f89bf8",
        "859378a9-b2ce-4edc-9b25-18d701201e5f",
        "ea288ab8-aea8-45b4-84b8-7d011a2a5e8f",
        '.uid==$job_uid',
        'status.reason=="Evicted"',
        'generation6_canary as generation6',
        "fa0c9b8529643267ad12a3ac4e817ea62a2bc78b83b9161d0f5b2183344288c1",
        "a63dbb284779822f4beef416581c99be91fdfb3406ec77b7095a6334b034d07d",
    ):
        assert value in submitter


def test_runner_proves_cli_before_output_or_claim() -> None:
    run = (ROOT / authority.RUN_PATH).read_text()
    cli = 'test "$(command -v docker)" = /docker-cli/bin/docker'
    claim = "-m evals.fleet.autocontinue_generation7_authority_v1 run"
    assert run.index(cli) < run.index("docker info") < run.index(claim)
    assert "apt-get" not in run and "curl" not in run


def test_claim_is_create_once_and_binds_generation7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = "qwen3.8-27b"
    spec = generation7.load(ROOT / generation7.G7_SPEC_PATHS[model])
    plan = generation7.validate_spec(spec, ROOT)
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
    assert claim["execution_generation"] == 7
    assert claim["execution_id"] == spec["execution"]["execution_id"]
    with pytest.raises(RuntimeError, match="existing generation claim"):
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


def test_claim_rechecks_predecessor_generation_under_same_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = "qwen3.8-27b"
    spec = generation7.load(ROOT / generation7.G7_SPEC_PATHS[model])
    plan = generation7.validate_spec(spec, ROOT)
    root_authority = authority.load(ROOT / authority.AUTH_PATH)
    fake_release = {"receipt_sha256": "sha256:" + "1" * 64}
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(authority, "validate_release", lambda *args: None)
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    predecessor = generation7.exact.execution_for(spec["statistical_cell"]["cell_id"], 6)
    (claim_root / (predecessor["execution_id"].removeprefix("sha256:") + ".json")).write_text(
        "preserved predecessor\n"
    )
    with pytest.raises(RuntimeError, match="existing generation claim"):
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


def test_runtime_rechecks_both_generation6_output_roots(tmp_path: Path) -> None:
    authority.validate_generation6_output_absence(tmp_path)
    predecessor = next(iter(generation6.EXPECTED.values()))
    appeared = tmp_path / predecessor["job_name"]
    appeared.symlink_to(tmp_path / "missing-target")
    with pytest.raises(RuntimeError, match="output root appeared"):
        authority.validate_generation6_output_absence(tmp_path)


def test_specs_are_canonical_json() -> None:
    for path in generation7.G7_SPEC_PATHS.values():
        raw = (ROOT / path).read_bytes()
        expected = json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":")
        ).encode() + b"\n"
        assert raw == expected
