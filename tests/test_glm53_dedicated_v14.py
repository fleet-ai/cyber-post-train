from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v13 as v13
from evals.fleet import glm53_dedicated_v14 as v14

ROOT = Path(__file__).resolve().parents[1]


def test_v14_is_fresh_and_changes_only_schedulable_requests() -> None:
    old = v13.payload(v13.spec(ROOT), ROOT)
    new = v14.payload(v14.spec(ROOT), ROOT)
    assert new["title"] == v14.TITLE
    assert new["run_dir"] == v14.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v14.RUN_DIR}
    assert new["resources"] == {
        **old["resources"],
        "cpu_request": "64",
        "memory_request": "768Gi",
    }
    for field in set(old) - {"title", "run_dir", "env", "resources"}:
        assert new[field] == old[field]


def test_v14_preserves_single_pod_tp8_runtime_and_idle_contract() -> None:
    request = v14.payload(v14.spec(ROOT), ROOT)
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["topology_mode"] == "required"
    assert v14.IMAGE == v13.IMAGE
    assert v14.MODEL_REVISION == v13.MODEL_REVISION
    assert v14.SERVER_ARGUMENTS == v13.SERVER_ARGUMENTS
    assert "IDLE_SECONDS=600" in request["command"]


def test_v14_rejects_request_shape_or_identity_drift() -> None:
    value = v14.spec(ROOT)
    value["resources"]["cpu_request"] = "63"
    with pytest.raises(ValueError, match="resource shape drifted"):
        v14.validate(value, ROOT)
    value = v14.spec(ROOT)
    value["request_shape_change"]["reason"] = "different"
    with pytest.raises(ValueError, match="rationale drifted"):
        v14.validate(value, ROOT)
    value = v14.spec(ROOT)
    value["run_dir"] = v13.RUN_DIR
    with pytest.raises(ValueError, match="identity drifted"):
        v14.validate(value, ROOT)
