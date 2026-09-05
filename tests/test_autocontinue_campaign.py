from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import autocontinue_campaign

ROOT = Path(__file__).parents[1]
CAMPAIGN_PATH = (
    ROOT / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
)
RELEASE_PATH = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-primary-campaign-release-preview-v1.json"
)


def load_bundle() -> tuple[dict, dict]:
    return (
        autocontinue_campaign.load_object(CAMPAIGN_PATH),
        autocontinue_campaign.load_object(RELEASE_PATH),
    )


def reseal(value: dict, field: str) -> None:
    value[field] = autocontinue_campaign.digest_without(value, field)


def test_autocontinue_campaign_is_exact_600_cell_zero_credit_preview() -> None:
    campaign, release = load_bundle()
    summary = autocontinue_campaign.validate_release_preview(release, campaign, root=ROOT)
    assert summary == {
        "tasks": 150,
        "cells": 600,
        "models": {"glm-5.3": 100, "qwen3.8-27b": 50},
        "partitions": 10,
        "canary_cells": 2,
        "bulk_cells": 598,
        "task_identity_sha256": (
            "sha256:c4eca5c58aa7e43eaf85ffb94179a57dbaa320a54a5e530c71795a1467e50fc7"
        ),
        "launch_authorized": False,
    }
    assert campaign["scientific_mapping"]["legacy_credit"] == 0
    assert release["cluster_objects_created"] is False


def test_autocontinue_partition_arithmetic_and_endpoint_caps() -> None:
    campaign, _ = load_bundle()
    by_id = {row["id"]: row for row in campaign["partitions"]}
    assert [
        by_id[key]["planned_cell_count"]
        for key in (
            "qwen-canary1",
            "qwen-hosted-a99",
            "qwen-hosted-b100",
        )
    ] == [1, 99, 100]
    assert [
        by_id[key]["planned_cell_count"]
        for key in (
            "glm-canary1",
            "glm-hosted-a91",
            "glm-hosted-b92",
            "glm-dedicated-a52",
            "glm-dedicated-a56",
            "glm-dedicated-b52",
            "glm-dedicated-b56",
        )
    ] == [1, 91, 92, 52, 56, 52, 56]
    assert campaign["gates"]["maximum_streams_per_endpoint"] == 2
    assert campaign["gates"]["atomic_endpoint_lease_required"] is True


def test_persisted_session_models_are_treatment_distinct() -> None:
    campaign, _ = load_bundle()
    old = {
        "qwen3.8-27b": "fleet-cluster-opencode-1.18.27/qwen3.8-27b",
        "glm-5.3": "fleet-cluster-opencode-1.18.27/glm-5.3",
    }
    for model_name, model in campaign["models"].items():
        new_config = {"model": model, "harness": {"name": "opencode"}}
        old_config = {
            "model": {"served_id": model_name, "session_model": old[model_name]},
            "harness": {"name": "opencode"},
        }
        new_identity = autocontinue_campaign.self_hosted.persisted_session_model_identity(
            new_config
        )
        old_identity = autocontinue_campaign.self_hosted.persisted_session_model_identity(
            old_config
        )
        assert new_identity != old_identity
        assert new_identity == f"{model_name}-opencode11827-autocontinue-v1"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda campaign: campaign["models"]["qwen3.8-27b"].__setitem__(
                "session_model", "fleet-cluster-opencode-1.18.27/qwen3.8-27b"
            ),
            "model identity",
        ),
        (
            lambda campaign: campaign["models"]["glm-5.3"].__setitem__(
                "context_management",
                "opencode_1.18.27_native_compaction_no_autocontinue",
            ),
            "context treatment",
        ),
        (
            lambda campaign: campaign["partitions"][1].__setitem__(
                "priority_class", "fleet-infra-quiet"
            ),
            "scheduling",
        ),
        (
            lambda campaign: campaign["partitions"][2]["full_source_ranks"].append(30),
            "full-task partition",
        ),
        (
            lambda campaign: campaign["gates"]["drain_protocol"].__setitem__(
                "checked_before_every_attempt_claim", False
            ),
            "safety gates",
        ),
        (
            lambda campaign: campaign["dedicated_runtime_gates"][
                "glm-dedicated-a-v5-autocontinue-v1"
            ].__setitem__("service_uid", "00000000-0000-0000-0000-000000000000"),
            "UID-bound parity",
        ),
        (
            lambda campaign: campaign.__setitem__("campaign_id", "replacement-campaign"),
            "campaign envelope",
        ),
        (
            lambda campaign: campaign["partitions"][0].__setitem__(
                "preflight_job_name", "replacement-preflight"
            ),
            "create-once identity",
        ),
        (
            lambda campaign: campaign["partitions"][1].__setitem__(
                "scored_job_name", "replacement-scored"
            ),
            "create-once identity",
        ),
        (
            lambda campaign: campaign["partitions"][2].__setitem__("sfs_root", "replacement-root"),
            "create-once identity",
        ),
    ],
)
def test_autocontinue_campaign_rejects_resealed_scientific_drift(mutate, message: str) -> None:
    campaign, _ = load_bundle()
    campaign = copy.deepcopy(campaign)
    mutate(campaign)
    reseal(campaign, "campaign_sha256")
    with pytest.raises(ValueError, match=message):
        autocontinue_campaign.validate_campaign(campaign, root=ROOT)


def test_autocontinue_release_cannot_authorize_or_hide_runtime_gates() -> None:
    campaign, release = load_bundle()
    for field in ("canary_launch_authorized", "bulk_launch_authorized"):
        changed = copy.deepcopy(release)
        changed[field] = True
        reseal(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="release preview"):
            autocontinue_campaign.validate_release_preview(changed, campaign, root=ROOT)
    changed = copy.deepcopy(release)
    changed["remaining_runtime_gates"].pop()
    reseal(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="release preview"):
        autocontinue_campaign.validate_release_preview(changed, campaign, root=ROOT)


def test_preview_script_has_no_submit_mode() -> None:
    script = (ROOT / "evals/fleet/scripts/preview_opencode_autocontinue_primary_v1.sh").read_text()
    assert "!= preview" in script
    assert "kubectl create" not in script
    assert "--submit" not in script
