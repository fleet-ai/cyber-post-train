from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v11 as v11
from evals.fleet import glm53_dedicated_v12 as v12

ROOT = Path(__file__).resolve().parents[1]


def test_v12_differs_from_v11_only_in_identity_and_priority() -> None:
    old = v11.spec(ROOT)
    new = v12.spec(ROOT)
    old_request = v11.payload(old, ROOT)
    new_request = v12.payload(new, ROOT)
    assert new["status"] == "AUTHORIZED"
    assert new["scope"] == "one_non_scored_dedicated_server_only"
    assert new_request["priority_class"] == "fleet-serve-low"
    assert new_request["topology_mode"] == "required"
    assert "topology_level" not in new_request
    for field in set(old_request) - {"title", "run_dir", "env", "priority_class"}:
        assert new_request[field] == old_request[field]
    assert new_request["env"] == {
        **old_request["env"],
        "GLM53_RUN_DIR": v12.RUN_DIR,
    }


def test_v12_preserves_exact_serving_treatment() -> None:
    assert v12.IMAGE == v11.IMAGE
    assert v12.MODEL_REVISION == v11.MODEL_REVISION
    assert v12.SERVER_ARGUMENTS == v11.SERVER_ARGUMENTS
    assert v12.SERVER_ARGUMENTS_SHA256 == v11.SERVER_ARGUMENTS_SHA256
    assert "--context-length" in v12.SERVER_ARGUMENTS
    assert v12.SERVER_ARGUMENTS[v12.SERVER_ARGUMENTS.index("--context-length") + 1] == "262144"
    assert v12.SERVER_ARGUMENTS[v12.SERVER_ARGUMENTS.index("--reasoning-parser") + 1] == "glm45"
    assert v12.SERVER_ARGUMENTS[v12.SERVER_ARGUMENTS.index("--tool-call-parser") + 1] == "glm47"


def test_v12_rejects_lower_nonpreempting_priority() -> None:
    value = v12.spec(ROOT)
    value["resources"]["priority_class"] = "fleet-infra-quiet"
    with pytest.raises(ValueError, match="priority drifted"):
        v12.validate(value, ROOT)
