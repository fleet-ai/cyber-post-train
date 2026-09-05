from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from urllib.error import URLError

import pytest

from evals.fleet import autocontinue_generation7_authority_package_v1 as g7_package
from evals.fleet import autocontinue_generation7_authority_v1 as g7_authority
from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import hosted_concurrency4_qualification_release_v3 as release
from evals.fleet import hosted_concurrency4_qualification_v1 as qualification

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_g7_gated_authority_is_held_and_self_digesting() -> None:
    receipt = release.load(ROOT / release.HELD_PATH)
    release.validate_held(receipt)
    assert receipt == release.expected_held()
    assert receipt["launch_authorized"] is False
    assert receipt["objects_created"] is False
    assert receipt["preserved_non_scoring_contract"] == {
        "chat_completion_requests": 24,
        "task_instance_session_scoring_or_verifier_calls": 0,
        "maximum_concurrent_requests_per_model": 4,
        "models_run_sequentially": True,
        "request_retries": 0,
    }
    assert receipt["preserved_acceptance_thresholds"] == {
        "maximum_request_latency_seconds": qualification.MAX_REQUEST_LATENCY_SECONDS,
        "maximum_wave_elapsed_seconds": qualification.MAX_WAVE_ELAPSED_SECONDS,
        "minimum_concurrency4_over_concurrency2_throughput_ratio": (
            qualification.MIN_C4_THROUGHPUT_GAIN
        ),
        "maximum_concurrency4_over_concurrency2_p95_stream_latency_ratio": (
            qualification.MAX_C4_TO_C2_P95_LATENCY_RATIO
        ),
        "zero_errors_required": True,
    }


def test_exact_g7_source_package_is_immutable_and_authoritative() -> None:
    binding = release._g7_source_binding(ROOT, release.G7_PACKAGE_COMMIT)
    assert binding["commit"] == release.G7_PACKAGE_COMMIT
    assert {row["path"] for row in binding["files"]} == {
        str(path) for path in release.G7_IMMUTABLE_PATHS
    }
    authority = g7_authority.load(ROOT / g7_authority.AUTH_PATH)
    g7_authority.validate_root_authorization(authority, ROOT)


def test_preserved_probe_is_exactly_24_non_scoring_requests_without_retries() -> None:
    assert list(qualification.EXPECTED_MODELS) == ["qwen3.8-27b", "glm-5.3"]
    assert qualification.WAVES == (2, 4)
    assert qualification.REQUESTS_PER_STREAM == 2
    assert (
        len(qualification.EXPECTED_MODELS)
        * sum(qualification.WAVES)
        * qualification.REQUESTS_PER_STREAM
        == 24
    )
    assert qualification.COMPLETIONS_URL.endswith("/v1/chat/completions")
    assert [row["function"]["name"] for row in qualification.TOOLS] == [
        "bash",
        "submit_report",
    ]
    package = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{release.PROBE_PACKAGE_COMMIT}:docs/evidence/qwen38-study/"
                "2026-09-05-hosted-concurrency4-qualification-held-v1.json",
            ],
            cwd=ROOT,
            text=True,
        )
    )
    assert package["launch_authorized"] is False
    assert package["chat_completion_requests_if_released"] == 24
    assert package["task_instance_session_scoring_or_verifier_calls"] == 0
    assert {
        name: value for name, value in vars(qualification).items() if name.endswith("_URL")
    } == {
        "ACCOUNT_URL": "https://orchestrator.fleetai.com/v1/account",
        "MODELS_URL": "https://inference.flt.build/v1/models",
        "COMPLETIONS_URL": "https://inference.flt.build/v1/chat/completions",
    }


def test_probe_does_not_retry_a_failed_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise URLError("synthetic")

    monkeypatch.setattr(qualification, "urlopen", fail)
    with pytest.raises(qualification.QualificationError, match="completion_request_failed"):
        qualification.post_completion("qwen3.8-27b", "bash", "synthetic-key")
    assert calls == 1


def _job(*, complete: bool = True, failed: bool = False, active: int = 0) -> dict:
    conditions = []
    if complete:
        conditions.append({"type": "Complete", "status": "True"})
    if failed:
        conditions.append({"type": "Failed", "status": "True"})
    return {
        "metadata": {"name": "g7", "uid": JOB_UID},
        "status": {
            "conditions": conditions,
            "active": active,
            "succeeded": 1 if complete else 0,
            "failed": 1 if failed else 0,
        },
    }


def _pods(
    *,
    phase: str = "Succeeded",
    restarts: int = 0,
    exit_code: int = 0,
    owner_uid: str = JOB_UID,
) -> dict:
    return {
        "items": [
            {
                "metadata": {
                    "uid": POD_UID,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": "g7",
                            "uid": owner_uid,
                            "controller": True,
                        }
                    ],
                },
                "status": {
                    "phase": phase,
                    "containerStatuses": [
                        {
                            "name": "evaluator",
                            "restartCount": restarts,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }
                    ],
                    "initContainerStatuses": [
                        {
                            "name": name,
                            "restartCount": restarts,
                            "state": {"terminated": {"exitCode": exit_code}},
                        }
                        for name in ("docker-cli", "dind")
                    ],
                },
            }
        ]
    }


def _job_with_name(name: str, uid: str) -> dict:
    value = _job()
    value["metadata"] = {"name": name, "uid": uid}
    return value


def _pods_with_owner(name: str, job_uid: str, pod_uid: str) -> dict:
    value = _pods(owner_uid=job_uid)
    value["items"][0]["metadata"]["uid"] = pod_uid
    value["items"][0]["metadata"]["ownerReferences"][0]["name"] = name
    return value


def test_generation7_job_requires_exact_uid_owner_and_clean_completion() -> None:
    assert release.validate_terminal_job(_job(), _pods(), job_name="g7") == (
        JOB_UID,
        POD_UID,
    )
    for job, pods in (
        (_job(complete=False), _pods()),
        (_job(failed=True), _pods()),
        (_job(active=1), _pods()),
        (_job(), _pods(phase="Running")),
        (_job(), _pods(restarts=1)),
        (_job(), _pods(exit_code=1)),
        (_job(), _pods(owner_uid="33333333-3333-4333-8333-333333333333")),
    ):
        with pytest.raises(release.ReleaseError):
            release.validate_terminal_job(job, pods, job_name="g7")


def test_release_writer_is_create_once(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    release.write_json_once(path, {"safe": True})
    with pytest.raises(FileExistsError):
        release.write_json_once(path, {"safe": False})


def test_submitter_is_held_and_has_exact_g7_and_bulk_gates() -> None:
    script = (ROOT / release.SUBMIT_PATH).read_text()
    assert "HELD: append-only G7-gated v3 launch release has not been rendered" in script
    assert 'kubectl -n "$namespace" create -f "$work/bundle.yaml"' in script
    assert "validate-generation7-evidence" in script
    assert "validate-live-binding" in script
    assert "CANARY-TERMINAL.json" in script
    assert "cell-execution-claims/opencode11827-autocontinue-v1" in script
    assert "validate_identity" in script
    assert "task_instance_session_scoring_or_verifier_calls:0" in script
    for value in (
        release.G7_PACKAGE_COMMIT,
        release.G7["qwen3.8-27b"]["execution"],
        release.G7["glm-5.3"]["execution"],
        *release.PRIOR_RELEASE_INTENTS,
        "chris-q38-ac-exact100-bulk-a199-v1",
        "chris-q38-ac-exact100-bulk-b200-v1",
        "chris-glm53-ac-exact100-bulk-a199-v1",
        "chris-glm53-ac-exact100-bulk-b200-v1",
    ):
        assert value in script


def test_malformed_g7_acceptance_binding_is_rejected() -> None:
    with pytest.raises(release.ReleaseError, match="acceptance_binding_invalid"):
        release.validate_acceptance_binding(
            {
                "schema_version": release.ACCEPTANCE_SCHEMA,
                "generation7_package_commit": release.G7_PACKAGE_COMMIT,
                "models": [],
                "all_exclusively_complete_and_accepted": True,
                "scores_included": False,
                "prompts_or_traces_included": False,
                "binding_sha256": "sha256:" + "0" * 64,
            },
            ROOT,
        )


def test_exact_g7_acceptance_binding_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = g7_authority.load(ROOT / g7_authority.AUTH_PATH)
    # Validate the immutable inherited authority once, then reuse those exact
    # plans.  Its validators recursively rebuild and serialize the same large
    # 798-cell plan; repeating that identical proof at every nested helper made
    # this focused round-trip spend more than 50 minutes in JSON encoding.
    exact_specs = g7_authority.specs(ROOT)
    exact_plans = g7_authority.validate_root_authorization(authority, ROOT)
    plan_by_spec = {
        spec["generation7_spec_sha256"]: plan
        for spec, plan in zip(exact_specs, exact_plans, strict=True)
    }
    spec_by_digest = {spec["generation7_spec_sha256"]: spec for spec in exact_specs}

    def cached_root(candidate: dict, root: Path) -> list[dict]:
        assert candidate == authority
        assert root.resolve() == ROOT.resolve()
        return exact_plans

    def cached_spec(candidate: dict, root: Path) -> dict:
        assert root.resolve() == ROOT.resolve()
        digest = candidate["generation7_spec_sha256"]
        assert candidate == spec_by_digest[digest]
        return plan_by_spec[digest]

    monkeypatch.setattr(g7_authority, "validate_root_authorization", cached_root)
    monkeypatch.setattr(generation7, "validate_spec", cached_spec)
    built = g7_package.build_package(ROOT)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for index, (model, source) in enumerate(release.G7.items(), start=1):
        spec = g7_authority.load(ROOT / source["spec"])
        plan = generation7.validate_spec(spec, ROOT)
        scoring_release = g7_authority._release_receipt(
            spec,
            authority,
            ROOT,
            release.G7_PACKAGE_COMMIT,
            built=built,
        )
        release_path = tmp_path / f"{model}-scoring-release.json"
        release_path.write_bytes(release.canonical_json(scoring_release) + b"\n")
        monkeypatch.setitem(source, "release", str(release_path))
        job_uid = f"{index}1111111-1111-4111-8111-111111111111"
        pod_uid = f"{index}2222222-2222-4222-8222-222222222222"
        claim = g7_authority._claim_receipt(
            spec,
            plan,
            scoring_release,
            g7_authority.file_sha256(release_path),
            authority,
            ROOT,
            release.G7_PACKAGE_COMMIT,
            claimed_at_utc="2026-09-05T12:00:00Z",
            job_uid=job_uid,
            pod_uid=pod_uid,
        )
        expected_claim, expected_config = g7_authority.generation2._expected_cell_bindings(plan)
        result = {
            "accepted": True,
            "quarantined": False,
            "claim_sha256": expected_claim,
            "attempt_config_sha256": expected_config,
            "acceptance_receipt_sha256": (
                "sha256:" + hashlib.sha256(f"{model}-accepted".encode()).hexdigest()
            ),
            "session_id": f"{index}3333333-3333-4333-8333-333333333333",
            "verifier_execution_id": f"{index}4444444-4444-4444-8444-444444444444",
            "session_ingest_completed": True,
            "cleanup_completed": True,
        }
        terminal = g7_authority.terminal_receipt(
            spec,
            plan,
            claim,
            result,
            scoring_release,
            g7_authority.file_sha256(release_path),
            authority,
            ROOT,
            release.G7_PACKAGE_COMMIT,
        )
        (evidence / f"{model}-job.json").write_text(
            json.dumps(_job_with_name(source["job"], job_uid))
        )
        (evidence / f"{model}-pods.json").write_text(
            json.dumps(_pods_with_owner(source["job"], job_uid, pod_uid))
        )
        (evidence / f"{model}-terminal.json").write_bytes(release.canonical_json(terminal) + b"\n")
        (evidence / f"{model}-claim.json").write_bytes(release.canonical_json(claim) + b"\n")
    binding = release.generation7_acceptance_binding(ROOT, evidence)
    release.validate_acceptance_binding(binding, ROOT)
    assert binding["all_exclusively_complete_and_accepted"] is True
    assert [row["model"] for row in binding["models"]] == list(release.G7)
    assert all(row["accepted_result"]["cleanup_completed"] for row in binding["models"])


def test_launcher_is_held_without_append_only_release() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / release.SUBMIT_PATH), "submit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "HELD" in result.stderr
