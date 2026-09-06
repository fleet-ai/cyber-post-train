import json
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_package_v3 as package
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v3 as diagnostic
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    return diagnostic.successor.build_plan(ROOT)


def _uids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")


def test_string_claim_root_is_normalized_before_scan(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    state = {}
    evidence = diagnostic._scan_global_evidence(  # noqa: SLF001
        _plan(), state, jobs_root=jobs, claim_root=str(claims)
    )
    assert evidence["global_claim_files_examined"] == 0
    assert evidence["global_accepted_files_examined"] == 0
    assert state["global_scan"] == {
        "subphase": "complete",
        "path_category": None,
        "claim_files_examined_before_terminal": 0,
        "accepted_files_examined_before_terminal": 0,
        "output_run_ids_examined_before_terminal": 4,
    }


@pytest.mark.parametrize("category", ["claim", "accepted"])
def test_scan_failure_records_category_and_counts_without_path_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, category: str
) -> None:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    jobs.mkdir()
    claims.mkdir()
    target = claims / "safe.json"
    if category == "accepted":
        target = jobs / "run" / "accepted" / "safe.json"
        target.parent.mkdir(parents=True)
    target.write_text("{}")
    monkeypatch.setattr(
        diagnostic.release,
        "_safe_receipt",
        lambda _path: (_ for _ in ()).throw(AttributeError("sanitized")),
    )
    state = {}
    with pytest.raises(AttributeError):
        diagnostic._scan_global_evidence(  # noqa: SLF001
            _plan(), state, jobs_root=jobs, claim_root=str(claims)
        )
    scan = state["global_scan"]
    assert scan["path_category"] == category
    assert "path" not in scan
    expected = 1 if category == "claim" else 0
    assert scan["claim_files_examined_before_terminal"] == expected
    expected = 1 if category == "accepted" else 0
    assert scan["accepted_files_examined_before_terminal"] == expected


@pytest.mark.parametrize(
    ("error", "classification"),
    [
        (AttributeError("x"), "attribute-error"),
        (KeyError("x"), "key-error"),
        (TypeError("x"), "type-error"),
        (UnicodeDecodeError("utf-8", b"x", 0, 1, "x"), "unicode-decode-error"),
        (RuntimeError("x"), "runtime-error"),
        (ValueError("x"), "value-error"),
        (OSError("x"), "filesystem-error"),
    ],
)
def test_error_type_class_is_allowlisted(error: Exception, classification: str) -> None:
    assert diagnostic._safe_error_type_class(error) == classification  # noqa: SLF001


def test_phase08_receipt_contains_only_sanitized_scan_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _uids(monkeypatch)

    def phase(index: int):
        def run(_root: Path, state: dict) -> None:
            if index == 7:
                state["global_scan"] = {
                    "subphase": "claim-scan",
                    "path_category": "claim",
                    "claim_files_examined_before_terminal": 7,
                    "accepted_files_examined_before_terminal": 0,
                    "output_run_ids_examined_before_terminal": 0,
                }
                raise AttributeError("must-not-be-persisted")

        return run

    phases = tuple(
        (name, phase(index))
        for index, (name, _function) in enumerate(diagnostic.PHASES)
    )
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 1
    value = json.loads(output.read_text())
    assert value["failed_phase"] == "08-global-evidence"
    assert value["error_type_class"] == "attribute-error"
    assert value["global_scan_subphase"] == "claim-scan"
    assert value["global_scan_path_category"] == "claim"
    assert value["claim_files_examined_before_terminal"] == 7
    assert "must-not-be-persisted" not in output.read_text()
    assert "path_name" not in value
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["model_calls"] == value["scoring_calls"] == 0
    assert value["api_mutation_calls"] == 0


def test_v3_package_is_fresh_immutable_held_and_credential_free() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert {"diagnostic_v1.py", "diagnostic_v2.py", "diagnostic.py"}.issubset(
        configmap["data"]
    )
    env_names = {
        row["name"]
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "FLEET_API_KEY" not in env_names
    assert "kubectl create" not in json.dumps(configmap["data"])


def test_v2_terminal_and_v3_held_receipts_are_self_digesting() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-diagnostic-v2-terminal.json"
        ).read_text()
    )
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-diagnostic-v3-held.json"
        ).read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["retry_same_identity"] is False
    assert terminal["failure_cause_deterministic"] is True
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["launch_authorized"] is False
