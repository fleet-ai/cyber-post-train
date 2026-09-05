from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import hosted_concurrency4_qualification_release_v3_gather as gather


def _allie() -> dict:
    return {
        "metadata": {
            "name": gather.ALLIE_NAME,
            "namespace": gather.NAMESPACE,
            "uid": gather.ALLIE_UID,
        },
        "spec": {"serviceAccountName": "default"},
        "status": {
            "phase": "Running",
            "containerStatuses": [{"name": "main", "ready": True, "restartCount": 0}],
        },
    }


class FakeReader:
    def request(self, method: str, path: str) -> tuple[int, dict]:
        assert method == "GET"
        if path.endswith("/pods/allie-dev"):
            return 200, _allie()
        if "/jobs/" in path:
            return 200, {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": path.rsplit("/", 1)[-1]},
            }
        return 200, {"apiVersion": "v1", "kind": "PodList", "items": []}


def _write_sources(root: Path) -> None:
    for binding in gather.G7.values():
        terminal = {
            "schema_version": gather.TERMINAL_SCHEMA,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        claim = {"schema_version": gather.CLAIM_SCHEMA, "scores_included": False}
        terminal_path = root / "jobs" / binding["job"] / "CANARY-TERMINAL.json"
        claim_path = (
            root
            / "cell-execution-claims"
            / "opencode11827-autocontinue-v1"
            / f"{binding['execution']}.json"
        )
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        terminal_path.write_text(json.dumps(terminal))
        claim_path.write_text(json.dumps(claim))


def test_in_cluster_gather_reads_only_exact_allowlisted_sources(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    bundle = gather.gather_in_cluster(reader=FakeReader(), shared_root=tmp_path)
    gather.validate_bundle(bundle)
    assert bundle["allie_dev"] == {
        "name": gather.ALLIE_NAME,
        "uid": gather.ALLIE_UID,
        "fresh": True,
    }
    assert bundle["prompts_traces_flags_or_scores_included"] is False
    output = tmp_path / "materialized"
    digests = gather.materialize_bundle(bundle, output)
    assert set(digests) == set(bundle["files"])
    assert not (output / "EVIDENCE.json").exists()
    with pytest.raises(FileExistsError):
        gather.materialize_bundle(bundle, output)


def test_protected_keys_and_symlinks_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "terminal.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": gather.TERMINAL_SCHEMA,
                "scores_included": False,
                "prompts_or_traces_included": False,
                "trace": "forbidden",
            }
        )
    )
    with pytest.raises(gather.GatherError, match="protected_evidence_key"):
        gather._read_exact(target, schema=gather.TERMINAL_SCHEMA)
    safe = tmp_path / "safe.json"
    safe.write_text(json.dumps({"schema_version": gather.CLAIM_SCHEMA}))
    link = tmp_path / "link.json"
    link.symlink_to(safe)
    with pytest.raises(gather.GatherError, match="symlink"):
        gather._read_exact(link, schema=gather.CLAIM_SCHEMA)


def test_bundle_freshness_and_self_digest_are_enforced(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    bundle = gather.gather_in_cluster(reader=FakeReader(), shared_root=tmp_path)
    bundle["collected_at_utc"] = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    bundle["receipt_sha256"] = gather.digest_without(bundle, "receipt_sha256")
    with pytest.raises(gather.GatherError, match="stale"):
        gather.validate_bundle(bundle)
    bundle["allie_dev"]["uid"] = "wrong"
    with pytest.raises(gather.GatherError, match="bundle_invalid"):
        gather.validate_bundle(bundle, maximum_age_seconds=10**9)


def test_gather_validates_before_writing_final_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    _write_sources(source)
    bundle = gather.gather_in_cluster(reader=FakeReader(), shared_root=source)
    monkeypatch.setattr(gather, "gather_via_allie", lambda: bundle)

    from evals.fleet import hosted_concurrency4_qualification_release_v3 as release

    monkeypatch.setattr(
        release,
        "generation7_acceptance_binding",
        lambda root, evidence: {"binding_sha256": "sha256:" + "7" * 64},
    )
    output = tmp_path / "out"
    evidence = gather.gather_and_validate(tmp_path, output)
    assert evidence["status"] == "VALIDATED"
    assert evidence["generation7_acceptance_binding_sha256"] == "sha256:" + "7" * 64
    assert json.loads((output / "EVIDENCE.json").read_text()) == evidence


def test_prepare_wrapper_is_append_only_and_score_blind() -> None:
    root = Path(__file__).parents[1]
    script = (
        root / "evals/fleet/scripts/prepare_hosted_concurrency4_qualification_release_v3.sh"
    ).read_text()
    assert "render-release" in script
    assert 'test ! -e "$release" && test ! -L "$release"' in script
    assert "hosted_concurrency4_qualification_release_v3_gather gather" in script
    assert "validate-live-binding" in script
    assert "kubectl logs" not in script
