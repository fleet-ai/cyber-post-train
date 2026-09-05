from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v11 as v11
from evals.fleet import glm53_dedicated_v13 as v13
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_v13_is_fresh_but_preserves_v11_serving_treatment() -> None:
    old = v11.payload(v11.spec(ROOT), ROOT)
    new = v13.payload(v13.spec(ROOT), ROOT)
    assert new["priority_class"] == "fleet-infra-quiet"
    assert new["topology_mode"] == "required"
    assert "topology_level" not in new
    for field in set(old) - {"title", "run_dir", "env"}:
        assert new[field] == old[field]
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v13.RUN_DIR}
    assert v13.spec(ROOT)["scope"] == "one_non_scored_dedicated_server_only"


def test_v13_preserves_exact_model_context_parsers_and_idle_contract() -> None:
    assert v13.IMAGE == v11.IMAGE
    assert v13.MODEL_REVISION == v11.MODEL_REVISION
    assert v13.SERVER_ARGUMENTS == v11.SERVER_ARGUMENTS
    args = v13.SERVER_ARGUMENTS
    assert args[args.index("--context-length") + 1] == "262144"
    assert args[args.index("--reasoning-parser") + 1] == "glm45"
    assert args[args.index("--tool-call-parser") + 1] == "glm47"
    assert "IDLE_SECONDS=600" in v13.payload(v13.spec(ROOT), ROOT)["command"]


def test_v13_rejects_priority_or_identity_drift() -> None:
    value = v13.spec(ROOT)
    value["resources"]["priority_class"] = "fleet-train-high"
    with pytest.raises(ValueError, match="priority drifted"):
        v13.validate(value, ROOT)
    value = v13.spec(ROOT)
    value["run_dir"] = v11.RUN_DIR
    with pytest.raises(ValueError, match="identity drifted"):
        v13.validate(value, ROOT)


@pytest.mark.parametrize(
    "relative_path,expected_status",
    [
        (
            "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v11-withdrawn.json",
            "WITHDRAWN_ZERO_SIDE_EFFECT",
        ),
        (
            "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v12-priority-preview-rejected.json",
            "PREVIEW_REJECTED",
        ),
    ],
)
def test_predecessor_terminal_receipts_are_self_digest_valid(
    relative_path: str, expected_status: str
) -> None:
    value = v11.load(ROOT / relative_path)
    assert value["status"] == expected_status
    assert value["scored_task_or_session_mutations"] == 0
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
