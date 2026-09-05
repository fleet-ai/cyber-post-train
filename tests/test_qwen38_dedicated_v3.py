from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_v2 as v2
from evals.fleet import qwen38_dedicated_v3 as v3
from evals.fleet import qwen38_dedicated_v3_live as live

ROOT = Path(__file__).resolve().parents[1]


def test_v3_is_fresh_but_preserves_exact_v2_serving_treatment() -> None:
    old = v2.payload(v2.spec(ROOT), ROOT)
    new = v3.payload(v3.spec(ROOT), ROOT)
    for key in set(old) - {"title", "run_dir", "env"}:
        assert new[key] == old[key]
    assert new["env"] == {**old["env"], "QWEN38_RUN_DIR": v3.RUN_DIR}
    assert new["priority_class"] == "fleet-infra-quiet"
    assert v3.spec(ROOT)["evaluation"]["fresh_actual_opencode_parity_required_before_scoring"]


def test_v3_resource_ceiling_allows_only_exact_glm_v13_peer() -> None:
    shape = live._project_shape(
        [
            {
                "name": "ft-run-peer",
                "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v13",
                "status": "Suspended",
            }
        ]
    )
    assert shape["planned_nodes"] == 2
    assert shape["planned_gpus"] == 9
    with pytest.raises(RuntimeError, match="unknown active"):
        live._project_shape(
            [
                {
                    "name": "ft-run-unknown",
                    "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-unknown",
                    "status": "Running",
                }
            ]
        )


def test_v3_rejects_exact_duplicate_and_identity_drift() -> None:
    with pytest.raises(RuntimeError, match="already exists"):
        live._project_shape(
            [{"name": "ft-run-duplicate", "run_dir": v3.RUN_DIR, "status": "Suspended"}]
        )
    value = v3.spec(ROOT)
    value["run_dir"] = v2.RUN_DIR
    with pytest.raises(ValueError, match="identity drifted"):
        v3.validate(value, ROOT)
