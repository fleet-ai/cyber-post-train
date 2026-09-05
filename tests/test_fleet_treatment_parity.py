from __future__ import annotations

import copy

import pytest

from evals.fleet.treatment_parity import (
    TreatmentParityError,
    assert_local_image_matches_host,
    assert_poolable_treatments,
)


def _receipt() -> dict:
    return {
        "schema": "fleet_eval_model_route_evidence_v1",
        "model": {"repository": "org/model", "revision": "a" * 40},
        "route": {
            "provider": "fleet-dedicated",
            "endpoint": "https://inference.example.test/v1",
            "revision": "b" * 40,
        },
        "harness": {
            "name": "opencode",
            "version": "1.18.27",
            "image_digest": "sha256:" + "c" * 64,
            "context_window": 262144,
            "compaction_policy": "native-post-compaction-autocontinue",
        },
        "tools": {"schema_digest": "sha256:" + "d" * 64},
        "tasks": {"binding": "exact_eval_task_version_id"},
        "observed_at": "2026-09-05T00:00:00Z",
        "evidence_source": "credential-free deployment readback",
    }


def test_exact_immutable_treatments_can_pool() -> None:
    assert_poolable_treatments(_receipt(), _receipt())


def test_generic_catalog_alias_without_checkpoint_revision_cannot_pool() -> None:
    candidate = _receipt()
    candidate["model"] = {"catalog_id": "z-ai/glm-5.3"}
    with pytest.raises(TreatmentParityError, match=r"model\.repository"):
        assert_poolable_treatments(_receipt(), candidate)


def test_openrouter_route_is_a_separate_treatment_even_with_same_model_claim() -> None:
    candidate = copy.deepcopy(_receipt())
    candidate["route"]["provider"] = "openrouter"
    with pytest.raises(TreatmentParityError, match=r"route\.provider"):
        assert_poolable_treatments(_receipt(), candidate)


def test_mutable_provider_route_label_is_not_immutable_evidence() -> None:
    candidate = copy.deepcopy(_receipt())
    candidate["route"]["revision"] = "current"
    with pytest.raises(TreatmentParityError, match="immutable"):
        assert_poolable_treatments(_receipt(), candidate)


def test_mutable_task_keys_cannot_enter_exact_version_block() -> None:
    candidate = copy.deepcopy(_receipt())
    candidate["tasks"]["binding"] = "task_keys"
    with pytest.raises(TreatmentParityError, match="exact eval_task_version_id"):
        assert_poolable_treatments(_receipt(), candidate)


@pytest.mark.parametrize(
    ("host", "image"), [("x86_64", "amd64"), ("aarch64", "arm64")]
)
def test_local_image_accepts_native_architecture_aliases(host: str, image: str) -> None:
    assert_local_image_matches_host(host, image)


def test_local_image_rejects_arm64_image_on_amd64_host() -> None:
    with pytest.raises(TreatmentParityError, match="does not match"):
        assert_local_image_matches_host("x86_64", "arm64")


def test_local_image_rejects_amd64_image_on_arm64_host() -> None:
    with pytest.raises(TreatmentParityError, match="does not match"):
        assert_local_image_matches_host("arm64", "amd64")
