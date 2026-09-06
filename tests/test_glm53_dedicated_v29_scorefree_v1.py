import json
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as v23
from evals.fleet import glm53_dedicated_v29_scorefree_package_v1 as package
from evals.fleet import glm53_dedicated_v29_scorefree_qualifier_v1 as qualifier

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0a0d029f0edee3170b879c11e46b338106fb8f6a"


def binding() -> dict[str, object]:
    return {
        "server_title": qualifier.SERVER_TITLE,
        "server_run_dir": qualifier.SERVER_RUN_DIR,
        "api_run_id": "ft-run-1dd00ca3",
        "rayjob_uid": "784eab08-d34c-4f18-9f2f-afa259c12161",
        "workload_uid": "62f0588b-715a-4236-b1b1-b9bf05ef66a5",
        "head_pod_uid": "474fa83d-c0bd-4b98-b8de-a7f16682fc88",
        "service_uid": "20b6b71b-75d7-47e2-b0f8-453aefd5f1d9",
        "service_origin": "http://ft-run-1dd00ca3-v2sqd-head-svc.fleet-train-jobs.svc:8000",
        "served_id": qualifier.SERVED_ID,
        "model_revision": qualifier.MODEL_REVISION,
        "context_length": qualifier.CONTEXT_LENGTH,
    }


def test_v29_binding_and_globals_are_generation_correct_and_restored() -> None:
    qualifier._validate_binding(binding())  # noqa: SLF001
    wrong = binding()
    wrong["server_title"] = v23.SERVER_TITLE
    with pytest.raises(qualifier.QualificationError, match="v23_server_binding_invalid"):
        qualifier._validate_binding(wrong)  # noqa: SLF001
    assert v23.SERVER_TITLE.endswith("v23")
    assert v23.RESULT_ROOT.name.endswith("v23-scorefree-qualification-v1")


def test_v29_package_is_fresh_and_executes_exact_adapter() -> None:
    configmap = package.build_configmap(ROOT, COMMIT)
    manifest = json.loads(configmap["data"]["package.json"])
    assert configmap["metadata"]["name"] == package.CONFIGMAP_NAME
    assert manifest["schema_version"] == package.SCHEMA
    assert manifest["job_name"] == qualifier.JOB_NAME
    assert manifest["output_root"] == str(qualifier.RESULT_ROOT)
    assert package.RUN in manifest["files"]
    assert "glm53_dedicated_v29_scorefree_qualifier_v1.py" in " ".join(manifest["files"])
    assert "glm53_dedicated_v29_scorefree_qualifier_v1" in configmap["data"]["run.sh"]
    assert v23.JOB_NAME != qualifier.JOB_NAME


def test_v29_held_contract_is_score_free_and_not_launch_authorized() -> None:
    held = package.build_held(COMMIT)
    assert held["qualification_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
    assert held["actual_opencode_bash_submit_report_required"] is True
    assert held["fleet_task_instance_calls"] == 0
    assert held["fleet_session_calls"] == 0
    assert held["verifier_calls"] == 0
    assert held["scoring_calls"] == 0
    evidence = (
        ROOT
        / "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v29-scorefree-package-held-v1.json"
    )
    assert json.loads(evidence.read_text()) == held
