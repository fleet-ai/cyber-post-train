from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import autocontinue_generation3_executable_package as prior
from evals.fleet import autocontinue_generation3_executable_package_v2 as package

ROOT = Path.cwd()


def _copy_package_tree(destination: Path) -> None:
    paths = set(
        package.CORE_A_PATHS
        + package.CORE_B_PATHS
        + tuple(path for values in package.MODEL_PATHS.values() for path in values)
        + (
            package.MANIFEST_PATH,
            package.SUBMIT_PATH,
            package.HELD_PATH,
            prior.HELD_PATH,
        )
    )
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_semantic_split_package_validates_full_generation3_chain() -> None:
    held = json.loads((ROOT / package.HELD_PATH).read_text())
    built = package.validate_held(held, ROOT)
    assert built["launch_authorized"] is False
    assert set(built["configmaps"]) == {
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        *package.MODEL_NAMES.values(),
    }
    for name, value in built["configmaps"].items():
        assert value["immutable"] is True
        assert package.prior.configmap_size(value) == built["object_json_bytes"][name]
        assert built["object_json_bytes"][name] < package.PACKAGE_OBJECT_LIMIT
    for manifest in built["model_manifests"].values():
        assert manifest["incident_receipt_sha256"] == generation3.INCIDENT_SHA
        assert manifest["tombstone_receipt_sha256"] == generation3.TOMBSTONE_SHA
        assert manifest["launch_authorized"] is False


@pytest.mark.parametrize("field", ["sessions", "model_calls"])
def test_resealed_side_effect_drift_rejects_at_semantic_layer(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_package_tree(tmp_path)
    incident_path = tmp_path / generation3.INCIDENT_PATH
    incident = json.loads(incident_path.read_text())
    incident["side_effects"][field] = 1
    incident["receipt_sha256"] = generation3.digest(incident, "receipt_sha256")
    incident_path.write_text(json.dumps(incident, indent=2) + "\n")
    monkeypatch.setattr(
        generation3, "INCIDENT_FILE_SHA", generation3.file_sha256(incident_path)
    )
    monkeypatch.setattr(generation3, "INCIDENT_SHA", incident["receipt_sha256"])
    with pytest.raises(ValueError, match="incident"):
        package.build_package(tmp_path)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (
            generation3.G3_SPEC_PATHS["qwen3.8-27b"],
            ("predecessor_generation2", "job_uid"),
        ),
        (
            generation3.TOMBSTONE_PATH,
            ("models", 0, "generation2_execution_id"),
        ),
    ],
)
def test_resealed_identity_drift_rejects(
    path: str, field: tuple[object, ...], tmp_path: Path
) -> None:
    _copy_package_tree(tmp_path)
    target = tmp_path / path
    value = json.loads(target.read_text())
    cursor = value
    for part in field[:-1]:
        cursor = cursor[part]
    cursor[field[-1]] = "11111111-1111-4111-8111-111111111111"
    digest_field = (
        "generation3_spec_sha256"
        if path.endswith("generation3-v1.json")
        else "receipt_sha256"
    )
    value[digest_field] = generation3.digest(value, digest_field)
    target.write_text(json.dumps(value, indent=2) + "\n")
    with pytest.raises(ValueError):
        package.build_package(tmp_path)


def test_projected_payload_has_no_collisions_and_verifies(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    for model, model_name in package.MODEL_NAMES.items():
        bootstrap = tmp_path / model
        bootstrap.mkdir()
        seen: set[str] = set()
        for name in (package.CORE_A_NAME, package.CORE_B_NAME, model_name):
            for key, value in built["configmaps"][name]["data"].items():
                assert key not in seen
                seen.add(key)
                (bootstrap / key).write_text(value)
        manifest = package.verify_mounted(
            bootstrap / "package-manifest.json", bootstrap
        )
        assert manifest["model"] == model


def test_manifest_and_submitter_remain_held_only() -> None:
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for item in manifest["items"]:
        pod = item["spec"]["template"]["spec"]
        assert item["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/package-layout": "semantic-split-configmap-v3",
        }
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert item["spec"]["backoffLimit"] == 0
    submit = (ROOT / package.SUBMIT_PATH).read_text()
    assert '[[ "$MODE" == preview ]]' in submit
    assert "create --dry-run=server" in submit
    assert "kubectl create -f" not in submit


def test_held_receipt_rejects_resealed_semantic_binding_drift() -> None:
    held = json.loads((ROOT / package.HELD_PATH).read_text())
    changed = copy.deepcopy(held)
    changed["semantic_evidence"]["both_specs_and_rendered_plans_validated"] = False
    unsigned = {key: value for key, value in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = package.sha256(package.canonical(unsigned))
    with pytest.raises(ValueError, match="held"):
        package.validate_held(changed, ROOT)
