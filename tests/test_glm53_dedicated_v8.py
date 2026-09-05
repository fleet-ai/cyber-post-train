from pathlib import Path

from evals.fleet import glm53_dedicated_v8 as v8
from evals.fleet import glm53_dedicated_v8_live as live

ROOT = Path.cwd()


def test_v8_is_one_authorized_non_scored_server() -> None:
    value = v8.spec(ROOT)
    request = v8.payload(value, ROOT)
    assert value["launch_authorized"] is True
    assert value["create_once"]["replica_b_allowed"] is False
    assert value["evaluation"]["scored_tasks_allowed"] is False
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["privileged"] is True


def test_v8_exact_model_runtime_scheduler_and_idle_release() -> None:
    value = v8.spec(ROOT)
    request = v8.payload(value, ROOT)
    assert value["model"]["revision"] == v8.MODEL_REVISION
    assert value["model"]["context_length"] == 262144
    assert request["image"] == v8.IMAGE
    assert request["priority_class"] == "fleet-infra-quiet"
    assert "IDLE_SECONDS=600" in request["command"]
    assert request["run_dir"] == v8.RUN_DIR
    assert request["title"] == v8.TITLE
    assert "--context-length 262144" in request["command"]
    assert "--model-path /mnt/sfs/models/glm-5.3-30333038" in request["command"]


def test_v8_live_rail_is_create_once_and_has_release_contract() -> None:
    assert live.BASE_URL == "https://api.ft.flt.build"
    assert live.NAMESPACE == "fleet-train-jobs"
    source = (ROOT / "evals/fleet/glm53_dedicated_v8_live.py").read_text()
    assert 'client.post("/v1/runs", json=payload)' in source
    assert '"DELETE /v1/runs/{name}"' in source
    assert '"scored_tasks_launched": 0' in source
