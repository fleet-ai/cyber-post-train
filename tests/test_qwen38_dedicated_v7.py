from pathlib import Path

import httpx
import pytest

from evals.fleet import qwen38_dedicated_v6 as v6
from evals.fleet import qwen38_dedicated_v7 as v7
from evals.fleet import qwen38_dedicated_v7_live as live

ROOT = Path(__file__).resolve().parents[1]


def test_v7_preserves_exact_tp1_treatment_with_fresh_identity() -> None:
    old = v6.payload(v6.spec(ROOT), ROOT)
    new = v7.payload(v7.spec(ROOT), ROOT)
    for key in set(old) - {"title", "run_dir", "env"}:
        assert new[key] == old[key]
    assert new["title"] == v7.TITLE
    assert new["run_dir"] == v7.RUN_DIR
    assert new["env"] == {**old["env"], "QWEN38_RUN_DIR": v7.RUN_DIR}


def test_v7_resource_ceiling_allows_only_live_tp1_b_peer() -> None:
    value = live._project_shape(
        [
            {
                "name": "ft-run-live-b",
                "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
                "status": "Running",
            }
        ]
    )
    assert value["planned_nodes"] == 2
    assert value["planned_gpus"] == 2
    with pytest.raises(RuntimeError, match="unknown active"):
        live._project_shape(
            [
                {
                    "name": "ft-run-other",
                    "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-other",
                    "status": "Running",
                }
            ]
        )


def test_v7_filters_stale_list_rows_through_exact_get() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ft-run-stale"):
            return httpx.Response(404)
        return httpx.Response(200, json={"status": "RUNNING"})

    rows = [
        {
            "name": "ft-run-stale",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-c-v1",
            "status": "submitted",
        },
        {
            "name": "ft-run-live-b",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
            "status": "RUNNING",
        },
    ]
    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as client:
        filtered = live._live_rows(client, rows)
    assert [row["name"] for row in filtered] == ["ft-run-live-b"]
    assert live._project_shape(filtered)["planned_nodes"] == 2
