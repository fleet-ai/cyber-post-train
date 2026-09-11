import copy
import json
from pathlib import Path

import pytest

from evals.fleet import base_control_certification_v2 as certification
from training.io import digest_json

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs/evaluation/qwen38-blackbox-fleet-dev-a-base-certification-v2.json"
PAYLOAD_PATH = (
    ROOT / "docs/evidence/inference/2026-09-11-qwen38-shared-base-payload-readback-v1.json"
)
ROUTE_PATH = (
    ROOT / "docs/evidence/inference/2026-09-11-qwen38-shared-base-route-component-v2.json"
)
PROXY_PATH = (
    ROOT / "docs/evidence/inference/2026-09-11-qwen38-fixed-proxy-dev-qualification-v1.json"
)
HARNESS_PATH = ROOT / "docs/evidence/inference/2026-09-11-qwen38-harness-component-v2.json"
PUBLICATION_PATH = ROOT / "configs/evaluation/opencode11827-agent-image-publication-plan-v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _reseal(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("sha256", None)
    return {**value, "sha256": digest_json(value)}


def test_v2_plan_and_measured_base_components_validate() -> None:
    plan = _read(PLAN_PATH)
    certification.validate_plan(plan, root=ROOT)
    certification.validate_route_component(
        plan,
        _read(PAYLOAD_PATH),
        _read(ROUTE_PATH),
        root=ROOT,
    )
    assert "inference_model_resource_version" not in plan["stable_candidate_base_route"][
        "object_identity"
    ]
    assert plan["launchable"] is False


def test_status_resource_version_must_still_be_freshly_observed() -> None:
    route = _read(ROUTE_PATH)
    route["registration"]["status_observation"]["inference_model_resource_version"] = None
    with pytest.raises(ValueError, match="status observation"):
        certification.validate_route_component(
            _read(PLAN_PATH),
            _read(PAYLOAD_PATH),
            _reseal(route),
            root=ROOT,
        )


def test_stable_uid_or_generation_drift_is_rejected() -> None:
    route = _read(ROUTE_PATH)
    route["registration"]["stable_route"]["object_identity"]["pod_uid"] = "changed"
    route["registration"]["stable_route_sha256"] = digest_json(
        route["registration"]["stable_route"]
    )
    with pytest.raises(ValueError, match="stable candidate"):
        certification.validate_route_component(
            _read(PLAN_PATH),
            _read(PAYLOAD_PATH),
            _reseal(route),
            root=ROOT,
        )


def test_component_receipt_cannot_open_scored_work() -> None:
    route = _read(ROUTE_PATH)
    route["launchable"] = True
    with pytest.raises(ValueError, match="launch certificate"):
        certification.validate_route_component(
            _read(PLAN_PATH),
            _read(PAYLOAD_PATH),
            _reseal(route),
            root=ROOT,
        )


def test_agent_publication_plan_and_harness_partial_stay_blocked() -> None:
    certification.validate_agent_publication_plan(_read(PUBLICATION_PATH), root=ROOT)
    certification.validate_harness_partial(
        _read(PLAN_PATH),
        _read(PROXY_PATH),
        _read(HARNESS_PATH),
        root=ROOT,
    )
    assert _read(HARNESS_PATH)["full_harness_receipt_present"] is False


def test_local_only_agent_identity_cannot_be_presented_as_qualified() -> None:
    harness = _read(HARNESS_PATH)
    harness["agent"]["dev_pod_created"] = True
    with pytest.raises(ValueError, match="agent-image blocker"):
        certification.validate_harness_partial(
            _read(PLAN_PATH),
            _read(PROXY_PATH),
            _reseal(harness),
            root=ROOT,
        )
