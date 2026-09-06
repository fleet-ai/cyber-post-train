import json
from pathlib import Path

from evals.fleet import hosted_glm_rank29_a3a4_c2_release_package_v2 as package
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v2 as release

ROOT = Path(__file__).resolve().parents[1]


def test_global_evidence_normalizes_string_roots(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    plan = release.successor.build_plan(ROOT)
    result = release._global_evidence(  # noqa: SLF001
        plan, jobs_root=str(jobs), claim_root=str(claims)
    )
    assert result == {
        "global_claim_files_examined": 0,
        "global_accepted_files_examined": 0,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
    }


def test_fresh_release_package_is_immutable_held_and_score_free() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    assert rendered["model_calls_authorized"] is False
    assert rendered["task_session_verifier_calls_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == release.CONFIGMAP_NAME
    assert job["metadata"]["name"] == release.JOB_NAME
    assert "jobs_root = Path(jobs_root)" in configmap["data"]["release.py"]
    assert "claim_root = Path(claim_root)" in configmap["data"]["release.py"]
    assert "hosted_glm_rank29_a3a4_c2_release_v2" in configmap["data"]["run.sh"]
    assert "kubectl create" not in json.dumps(configmap["data"])


def test_v3_diagnostic_terminal_is_digest_valid() -> None:
    value = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-hosted-rank29-release-diagnostic-v3-terminal.json"
        ).read_text()
    )
    assert value["status"] == "PASSED_TO_NETWORK_BOUNDARY"
    assert value["completed_phases"] == [
        "01-source-bindings",
        "02-failed-release-terminal",
        "03-predecessors-terminal",
        "04-peer-acceptances",
        "05-endpoint-lease-capacity",
        "06-rank29-history",
        "07-runtime-plan",
        "08-global-evidence",
        "09-scored-collisions",
    ]
    assert value["model_calls"] == value["scoring_calls"] == 0
    assert value["receipt_sha256"] == release.self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_v2_held_receipt_binds_exact_package_and_stays_closed() -> None:
    value = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-hosted-rank29-a3a4-release-held-v2.json"
        ).read_text()
    )
    assert value["status"] == "HELD_FOR_INDEPENDENT_REVIEW"
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert value["receipt_sha256"] == release.self_hosted.digest_without(
        value, "receipt_sha256"
    )
