from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_scan_diagnostic_package_v2 as package
from evals.fleet import qwen_hosted_rank17_g22_scan_diagnostic_v2 as diagnostic

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({}, (None, None)),
        ({"schema_version": "x"}, ("x", None)),
        (
            {"schema_version": "x", "receipt_sha256": "sha256:" + "0" * 64},
            ("x", "sha256:" + "0" * 64),
        ),
        (
            {
                "nested": {"schema_version": "not-top-level", "secret": "must-not-decode"},
                "schema_version": "top-level",
                "receipt_sha256": "sha256:" + "1" * 64,
            },
            ("top-level", "sha256:" + "1" * 64),
        ),
    ],
)
def test_envelope_extracts_only_selected_top_level_strings(
    tmp_path: Path, value: dict[str, object], expected: tuple[str | None, str | None]
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(value))
    assert diagnostic._envelope(path) == expected  # noqa: SLF001


@pytest.mark.parametrize("raw", ["[]", "{", '{"schema_version":'])
def test_envelope_rejects_invalid_root_or_syntax(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(raw)
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic._envelope(path)  # noqa: SLF001


def test_collect_emits_only_hashed_paths_and_bounded_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claims = tmp_path / "claims"
    jobs = tmp_path / "jobs"
    claims.mkdir()
    (jobs / "run").mkdir(parents=True)
    (claims / "claim.json").write_text("{}")
    (jobs / "run" / "ACCEPTED.json").write_text(
        json.dumps({"schema_version": "accepted", "receipt_sha256": "sha256:" + "0" * 64})
    )
    monkeypatch.setattr(diagnostic, "CLAIM_ROOT", claims)
    monkeypatch.setattr(diagnostic, "JOBS_ROOT", jobs)
    receipt = diagnostic.collect(path_hash_salt="sha256:" + "a" * 64)
    encoded = json.dumps(receipt)
    assert receipt["candidate_counts_by_class"] == {"accepted_leaf": 1, "canonical_claim": 1}
    assert receipt["envelope_status_counts"] == {
        "envelope_present_digest_unverified": 1,
        "schema_missing_or_invalid": 1,
    }
    assert len(receipt["bounded_anomalies"]) == 1
    assert "claim.json" not in encoded
    assert str(tmp_path) not in encoded
    assert "must-not-decode" not in encoded
    assert receipt["payload_values_decoded"] is False
    assert receipt["receipt_sha256"] == diagnostic.digest(receipt)


def test_manifest_is_fresh_create_once_score_free_and_nonpreempting() -> None:
    rendered = package.render(ROOT)
    configmap, job = rendered["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == package.CONFIGMAP_NAME
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert not job["spec"]["template"]["spec"]["containers"][0]["env"]


def test_projected_package_binds_exact_source_bytes(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    diagnostic.validate_package_source(tmp_path / "package-source.json", tmp_path)
    (tmp_path / "qwen_hosted_rank17_g22_scan_diagnostic_v2.py").write_text("drift")
    with pytest.raises(diagnostic.DiagnosticError, match="package_file_drifted"):
        diagnostic.validate_package_source(tmp_path / "package-source.json", tmp_path)


def test_held_receipt_is_exact_self_digested_and_zero_effect() -> None:
    expected = package.held(ROOT)
    actual = json.loads((ROOT / package.HELD_PATH).read_text())
    assert actual == expected
    assert actual["launch_authorized"] is False
    assert actual["scoring_authorized"] is False
    assert set(actual["side_effects"].values()) == {0}
    assert actual["receipt_sha256"] == diagnostic.digest(actual)
