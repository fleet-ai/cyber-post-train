from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import autocontinue_generation3_executable_package as package

ROOT = Path.cwd()


def test_split_configmaps_are_immutable_and_below_each_byte_limit() -> None:
    built = package.build_package(ROOT)
    assert set(built["configmaps"]) == {
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        *package.MODEL_NAMES.values(),
    }
    assert built["launch_authorized"] is False
    for name, configmap in built["configmaps"].items():
        size = package.configmap_size(configmap)
        assert configmap["immutable"] is True
        assert size == built["object_json_bytes"][name]
        assert size < package.PACKAGE_OBJECT_LIMIT
        assert size < package.KUBERNETES_OBJECT_LIMIT


@pytest.mark.parametrize("model", ["qwen3.8-27b", "glm-5.3"])
def test_projected_payload_verifies_and_reconstructs_exact_sources(
    model: str, tmp_path: Path
) -> None:
    built = package.build_package(ROOT)
    bootstrap = tmp_path / "bootstrap"
    destination = tmp_path / "workspace"
    bootstrap.mkdir()
    names = (package.CORE_A_NAME, package.CORE_B_NAME, package.MODEL_NAMES[model])
    for name in names:
        for key, value in built["configmaps"][name]["data"].items():
            (bootstrap / key).write_text(value)
    manifest_path = bootstrap / "package-manifest.json"
    manifest = package.verify_mounted(manifest_path, bootstrap)
    assert manifest["aggregate_sha256"] == built["model_manifests"][model]["aggregate_sha256"]
    package.install_mounted(manifest, bootstrap, destination)
    for obj in manifest["objects"]:
        for entry in obj["entries"]:
            assert (destination / entry["source_path"]).read_bytes() == (
                ROOT / entry["source_path"]
            ).read_bytes()


def test_projected_payload_rejects_tampering(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    model = "qwen3.8-27b"
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for name in (package.CORE_A_NAME, package.CORE_B_NAME, package.MODEL_NAMES[model]):
        for key, value in built["configmaps"][name]["data"].items():
            (bootstrap / key).write_text(value)
    first = built["model_manifests"][model]["objects"][0]["entries"][0]
    (bootstrap / first["data_key"]).write_text("tampered")
    with pytest.raises(ValueError, match="payload"):
        package.verify_mounted(bootstrap / "package-manifest.json", bootstrap)


def test_held_receipt_and_projected_manifest_are_nonlaunching() -> None:
    held = json.loads((ROOT / package.HELD_PATH).read_text())
    package.validate_held(held, ROOT)
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for item in manifest["items"]:
        pod = item["spec"]["template"]["spec"]
        assert (
            item["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
            == "false"
        )
        assert item["metadata"]["annotations"]["cyber-post-train.fleet.ai/preview-only"] == "true"
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert len(pod["volumes"][0]["projected"]["sources"]) == 3
    submit = (ROOT / package.SUBMIT_PATH).read_text()
    assert '[[ "$MODE" == preview ]]' in submit
    assert "create --dry-run=server" in submit
    assert "kubectl create -f" not in submit
    assert (ROOT / package.SUBMIT_PATH).stat().st_mode & stat.S_IXUSR


def test_persisted_generation3_claim_reloads_and_binds_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = generation3.load(ROOT / generation3.G3_SPEC_PATHS["qwen3.8-27b"])
    plan = generation3.validate_spec(spec, ROOT)
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    created = generation3.claim_execution_generation(spec, plan, root=ROOT, claim_root=claim_root)
    persisted_paths = list(claim_root.glob("*.json"))
    assert len(persisted_paths) == 1
    persisted = generation3.load(persisted_paths[0])
    assert persisted == created
    generation3.validate_claim(persisted, spec, plan, root=ROOT)
    attempt, config = generation2._expected_cell_bindings(plan)
    terminal = generation3.terminal_receipt(
        spec,
        plan,
        persisted,
        {
            "accepted": False,
            "quarantined": True,
            "claim_sha256": attempt,
            "attempt_config_sha256": config,
        },
        root=ROOT,
        terminal_at_utc="2026-09-05T06:00:00Z",
    )
    generation3.validate_terminal(terminal, spec, plan, persisted, root=ROOT)
