from __future__ import annotations

import pytest

from cyber_post_train.jobs import API_URLS
from training.miles_cluster import cluster_profile, plan_cluster_target


def test_cluster_profiles_are_exact_and_separate() -> None:
    dev = cluster_profile("dev")
    prod = cluster_profile("prod")

    assert dev.api_base_url == API_URLS["dev"]
    assert prod.api_base_url == API_URLS["prod"]
    assert dev.kube_context != prod.kube_context
    assert dev.namespace_uid != prod.namespace_uid
    assert dev.namespace == prod.namespace == "fleet-train-jobs"


def test_plan_target_preserves_only_legacy_dev_default() -> None:
    assert plan_cluster_target({"execution": {}}) == "dev"
    assert plan_cluster_target({"execution": {"cluster_target": "prod"}}) == "prod"
    with pytest.raises(ValueError, match="exactly dev or prod"):
        plan_cluster_target({"execution": {"cluster_target": "staging"}})
