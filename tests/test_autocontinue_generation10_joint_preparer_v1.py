from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from evals.fleet import autocontinue_generation10_joint_preparer_v1 as generation10

ROOT = Path(__file__).resolve().parents[1]


def test_joint_g10_package_is_held_and_glm_requires_exact_tombstone() -> None:
    generation10.assert_route_gate_implementation()
    generation10.assert_optimized_runtime_implementation(ROOT)
    package = generation10.prepare(ROOT)
    generation10.validate_held_receipt(ROOT, package)
    assert package["status"] == "HELD"
    assert package["launch_authorized"] is False
    assert package["objects_created"] is False
    assert package["models"]["qwen3.8-27b"]["status"] == (
        "HELD_RELEASE_AND_DUPLICATE_PREFLIGHT_REQUIRED"
    )
    assert package["models"]["glm-5.3"] == {
        "status": "HELD_PREINSTANCE_TOMBSTONE_REQUIRED",
        "tombstone_path": generation10.MODELS["glm-5.3"]["tombstone"],
    }
    qwen = package["models"]["qwen3.8-27b"]
    assert qwen["spec"]["execution"]["execution_generation"] == 10
    assert qwen["plan"]["execution"]["launch_authorized"] is False
    assert qwen["spec"]["required_authority_gate"] == {
        "mode": "behavioral_method_not_allowed",
        "method": "GET",
        "statuses": {"provisioning": 405, "scoring": 405},
        "response_body_read": False,
        "diagnosis_path": generation10.DIAGNOSIS_PATH,
        "diagnosis_receipt_sha256": (
            "sha256:d52f9d9bbbcd9bc2dbc8d5dbe15cd36a4c65018cf186029109a86063408f349c"
        ),
    }
    assert qwen["spec"]["required_optimized_runtime"] == generation10.OPTIMIZED_RUNTIME
    assert qwen["plan"]["source"]["required_optimized_runtime"] == (generation10.OPTIMIZED_RUNTIME)


def test_g10_fresh_identities_do_not_reuse_g7_g8_or_qwen_g9() -> None:
    package = generation10.prepare(ROOT)
    qwen_g9 = "sha256:b2991e6062643b6a7875d3dd8d677a63e0f848323c2faedf3022b16ea1616d98"
    for model, row in generation10.MODELS.items():
        assert len({row["g7_execution_id"], row["g8_execution_id"], row["g10_execution_id"]}) == 3
        assert row["g10_execution_id"] != qwen_g9
        if package["models"][model]["status"].startswith("HELD_RELEASE"):
            assert (
                package["models"][model]["spec"]["execution"]["execution_id"]
                == row["g10_execution_id"]
            )


def test_corrected_authority_diagnosis_supersedes_openapi_only_false_negative() -> None:
    diagnosis = generation10.read_canonical(ROOT / generation10.DIAGNOSIS_PATH)
    generation10.validate_authority_diagnosis(diagnosis)
    assert diagnosis["behavioral_probe"]["provisioning_status"] == 405
    assert diagnosis["behavioral_probe"]["scoring_status"] == 405
    assert diagnosis["behavioral_probe"]["response_bodies_read"] is False
    assert diagnosis["g9_authority_diagnosis_status"] == ("SUPERSEDED_HELD_DO_NOT_USE_FOR_RELEASE")
    assert diagnosis["startup_cost_assessment"]["classification"] == (
        "SOURCE_INSPECTION_INFERENCE_NOT_SEPARATELY_TIMED"
    )
    superseded = ROOT / diagnosis["supersedes"]["path"]
    assert (
        "sha256:" + hashlib.sha256(superseded.read_bytes()).hexdigest()
        == (diagnosis["supersedes"]["file_sha256"])
    )


def test_preinstance_tombstone_rejects_any_authoritative_activity() -> None:
    generation10.load_static(ROOT)
    path = ROOT / generation10.MODELS["qwen3.8-27b"]["tombstone"]
    tombstone = generation10.read_canonical(path)
    generation10.validate_preinstance_tombstone("qwen3.8-27b", tombstone)
    for field, value in (
        ("model_called", True),
        ("scored_outcome_created", True),
    ):
        changed = copy.deepcopy(tombstone)
        changed[field] = value
        changed["receipt_sha256"] = generation10.digest(changed)
        with pytest.raises(ValueError, match="tombstone drifted"):
            generation10.validate_preinstance_tombstone("qwen3.8-27b", changed)


def test_preinstance_tombstone_rejects_identity_or_digest_drift() -> None:
    generation10.load_static(ROOT)
    path = ROOT / generation10.MODELS["qwen3.8-27b"]["tombstone"]
    tombstone = generation10.read_canonical(path)
    changed = copy.deepcopy(tombstone)
    changed["generation7_job"]["uid"] = "00000000-0000-0000-0000-000000000001"
    changed["receipt_sha256"] = generation10.digest(changed)
    with pytest.raises(ValueError, match="tombstone drifted"):
        generation10.validate_preinstance_tombstone("qwen3.8-27b", changed)

    changed = copy.deepcopy(tombstone)
    changed["receipt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="tombstone drifted"):
        generation10.validate_preinstance_tombstone("qwen3.8-27b", changed)


def test_prepare_is_deterministic_and_side_effect_free() -> None:
    before = sorted(path.relative_to(ROOT) for path in ROOT.rglob("*"))
    first = generation10.prepare(ROOT)
    second = generation10.prepare(ROOT)
    after = sorted(path.relative_to(ROOT) for path in ROOT.rglob("*"))
    assert first == second
    assert first["package_sha256"] == generation10.digest(first, "package_sha256")
    assert before == after
