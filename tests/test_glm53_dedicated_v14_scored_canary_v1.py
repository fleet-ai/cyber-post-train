from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as package
from evals.fleet import glm53_dedicated_v14_scored_canary_preflight_package_v1 as preflight_package
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json"
BINDING = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-server-binding.json"
ORIGIN = "http://glm-v14-head-svc.fleet-train-jobs.svc.cluster.local:8000"


def test_canary_reserves_whole_rank_but_executes_only_first_cell() -> None:
    plan = canary.build_plan(
        ROOT, service_origin=ORIGIN, parity_path=PARITY, binding_path=BINDING
    )
    assert plan["launch_authorized"] is False
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [(51, 1)]
    reservation = plan["whole_task_reservation"]
    assert reservation["selection_rank"] == 51
    assert len(reservation["cell_ids"]) == len(set(reservation["cell_ids"])) == 4
    assert len(reservation["execution_ids"]) == len(set(reservation["execution_ids"])) == 4
    assert reservation["remaining_attempts_require_post_canary_release"] == [2, 3, 4]
    assert plan["serving_load_block"] == canary.SERVING_BLOCK
    assert plan["execution"]["traffic_heartbeat_path"].endswith("traffic-stream-1")
    assert isinstance(canary.CLAIM_ROOT, Path)


def test_canary_rejects_non_cluster_service_origin() -> None:
    with pytest.raises(ValueError, match="cluster-local"):
        canary.build_plan(
            ROOT,
            service_origin="http://127.0.0.1:8000",
            parity_path=PARITY,
            binding_path=BINDING,
        )


def test_canary_source_package_is_closed_but_held() -> None:
    built = package.render(ROOT)
    assert built["status"] == "READY_HELD"
    assert built["launch_authorized"] is False
    assert built["job_name"] == canary.JOB_NAME
    assert built["dynamic_evidence_required"] == [
        "fresh_server_binding",
        "fresh_non_scored_parity",
        "fresh_duplicate_release",
    ]
    source, job = built["objects"]["items"]
    assert source["metadata"]["name"] == package.SOURCE_CONFIGMAP
    assert source["immutable"] is True
    assert "dedicated_canary.py" in source["data"]
    assert "dedicated_runtime.py" in source["data"]
    assert job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] == "false"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    preview = {key: value for key, value in built.items() if key != "objects"}
    assert preview["package_sha256"] == self_hosted.digest_without(
        preview, "package_sha256"
    )


def test_canary_source_package_rejects_missing_install_mapping(monkeypatch) -> None:
    script = ROOT / "evals/fleet/scripts/run_glm53_dedicated_v14_scored_canary_v1.sh"
    text = script.read_text().replace(
        "dedicated_runtime.py:glm53_dedicated_v14_scored_canary_runtime_v1.py \\\n",
        "",
    )
    original = Path.read_text

    def read_text(path: Path, *args, **kwargs):
        if path == script:
            return text
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="install closure omits dedicated_runtime.py"):
        package._source_data(ROOT)  # noqa: SLF001


def test_preflight_package_binds_exact_controller_package() -> None:
    controller = package.render(ROOT)
    built = preflight_package.render(ROOT)
    assert built["launch_authorized"] is True
    assert built["controller_package_sha256"] == controller["package_sha256"]
    configmap, job = built["objects"]["items"]
    embedded = configmap["data"]["controller-package.json"]
    assert controller["package_sha256"] in embedded
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    run_script = configmap["data"]["run.sh"]
    assert "predecessor.py:exact_pass4_bulk_v3.py" in run_script
    assert "engine.py:exact_pass4_bulk_runtime_v3.py" in run_script
    assert "bulk.py:exact_pass4_bulk_v3.py" not in run_script
