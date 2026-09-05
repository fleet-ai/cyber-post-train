from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v8 as v8
from evals.fleet import glm53_dedicated_v8_live as live
from evals.fleet import self_hosted

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


def test_v8_sfs_duplicate_check_translates_the_observer_mount() -> None:
    assert live._observer_sfs_path(v8.RUN_DIR) == (
        "/shared/jobs/chris-cyber-evalserve-glm53-tp8-a-v8"
    )
    with pytest.raises(ValueError, match="outside the exact SFS mount"):
        live._observer_sfs_path("/shared/jobs/not-a-runtime-path")
    with pytest.raises(ValueError, match="unsafe"):
        live._observer_sfs_path("/mnt/sfs/../escape")


def test_v8_terminal_and_diagnosis_receipts_are_digest_valid_and_non_scored() -> None:
    terminal = v8.load(
        ROOT
        / "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v8-terminal.json"
    )
    diagnosis = v8.load(
        ROOT
        / "docs/evidence/qwen38-study/2026-09-05-glm53-dedicated-serving-v8-diagnosis.json"
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert diagnosis["receipt_sha256"] == self_hosted.digest_without(
        diagnosis, "receipt_sha256"
    )
    assert terminal["evidence"]["scored_tasks_launched"] == 0
    assert diagnosis["classification"]["relaunch_allowed"] is False
