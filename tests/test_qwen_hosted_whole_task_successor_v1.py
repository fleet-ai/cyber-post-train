from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_v4 as source
from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
from evals.fleet import qwen_hosted_whole_task_successor_v1_package as package
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]


def _release(plans: dict[str, dict]) -> dict:
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": "2026-09-06T08:40:00Z",
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 2,
        "endpoint_maximum_streams": 2,
        "controllers": successor.release_projection(plans),
        "ledger_snapshot_path": successor.LEDGER_PATH,
        "ledger_snapshot_receipt_sha256": successor.LEDGER_SELF_SHA256,
        "ledger_snapshot_file_sha256": successor.LEDGER_FILE_SHA256,
        "predecessor_tombstones": successor.PREDECESSOR_TOMBSTONES,
        "predecessor_disposition": {
            "retry_forbidden_selection_ranks": [13, 14],
            "prior_job_uids": [
                "515c370a-ecb4-4c41-a349-46a328c8fa68",
                "f23805b5-516b-4828-a3b5-82673a1b3e2f",
            ],
            "prior_pod_uids": [
                "4d86f0c5-7cef-4f8a-a798-ea5c725e4955",
                "d5d7b7cf-d639-4ad6-b012-3001c4935b2d",
            ],
            "prior_jobs_terminal": True,
            "prior_pods_absent": True,
            "endpoint_lease_files_absent": True,
        },
        "fresh_collision_reconciliation": {
            "checked_immediately_before_release": True,
            "checked_from_uid_bound_sfs_pod": True,
            "observer_pod_uid": "33333333-3333-4333-8333-333333333333",
            "observed_cells": 8,
            "canonical_claim_collisions": 0,
            "authoritative_session_collisions": 0,
            "accepted_evidence_collisions": 0,
            "output_root_collisions": 0,
            "api_mutations": 0,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    return successor._seal(body)  # noqa: SLF001


def test_plans_select_only_two_untouched_complete_tasks() -> None:
    plans = successor.build_plans(ROOT)
    predecessors = source.validate_all(ROOT)
    assert [(key, len(plan["attempts"])) for key, plan in plans.items()] == [
        ("qwen-a", 4),
        ("qwen-b", 4),
    ]
    for controller, plan in plans.items():
        rank = successor.CONTROLLERS[controller]["rank"]
        expected = [
            row for row in predecessors[controller]["attempts"] if row["selection_rank"] == rank
        ]
        assert plan["attempts"] == expected
        assert plan["model"] == predecessors[controller]["model"]
        assert plan["harness"] == predecessors[controller]["harness"]
        assert plan["treatment"] == predecessors[controller]["treatment"]
        assert plan["authority"] == predecessors[controller]["authority"]
        assert plan["execution"] == predecessors[controller]["execution"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert {row["selection_rank"] for plan in plans.values() for row in plan["attempts"]} == {
        15,
        16,
    }


def test_held_evidence_is_digest_valid_and_never_launches() -> None:
    plans = successor.build_plans(ROOT)
    held = successor.load(ROOT / successor.HELD_PATH)
    successor.validate_held(held, plans)
    assert held["receipt_sha256"] == self_hosted.digest_without(held, "receipt_sha256")
    assert held["launch_authorized"] is False
    assert held["scoring_authorized"] is False
    assert held["retry_forbidden_selection_ranks"] == [13, 14]


def test_release_validator_fails_closed_on_every_collision_class() -> None:
    plans = successor.build_plans(ROOT)
    release = _release(plans)
    successor.validate_release(release, plans)
    for field in (
        "canonical_claim_collisions",
        "authoritative_session_collisions",
        "accepted_evidence_collisions",
        "output_root_collisions",
    ):
        bad = copy.deepcopy(release)
        bad["fresh_collision_reconciliation"][field] = 1
        bad["receipt_sha256"] = self_hosted.digest_without(bad, "receipt_sha256")
        with pytest.raises(RuntimeError, match="release drifted"):
            successor.validate_release(bad, plans)
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(successor.load(ROOT / successor.HELD_PATH), plans)
    extra = copy.deepcopy(release)
    extra["score"] = 0
    extra["receipt_sha256"] = self_hosted.digest_without(extra, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(extra, plans)


def test_atomic_provider_publishes_and_validates_all_four_before_return(
    tmp_path: Path,
) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    (tmp_path / "jobs").mkdir()
    checked: list[str] = []
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda config, _key: checked.append(config["run_id"]),
    )
    first = provider(
        plan,
        plan["attempts"][0],
        claim_root,
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    assert first == provider.claims[plan["attempts"][0]["run_id"]]
    assert checked == [row["run_id"] for row in plan["attempts"]]
    assert len(list(claim_root.glob("*.json"))) == 4
    reservation = successor.load(out / "RESERVATION.json")
    assert reservation["all_claims_before_model_call"] is True
    assert reservation["all_claims_byte_validated"] is True
    assert reservation["claim_sha256s"] == [
        provider.claims[row["run_id"]]["receipt_sha256"] for row in plan["attempts"]
    ]
    assert reservation["receipt_sha256"] == self_hosted.digest_without(
        reservation, "receipt_sha256"
    )


def test_partial_claim_publication_rolls_back_without_touching_existing_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = successor.build_plans(ROOT)["qwen-b"]
    out = tmp_path / "out"
    out.mkdir()
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    unrelated = claim_root / "unrelated.json"
    unrelated.write_text("{}\n")
    (tmp_path / "jobs").mkdir()
    real = engine.claim_cell
    calls = 0

    def collide_on_second(*args: object, **kwargs: object) -> dict | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "claim_cell", collide_on_second)
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="transaction collided"):
        provider(
            plan,
            plan["attempts"][0],
            claim_root,
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert list(claim_root.glob("*.json")) == [unrelated]
    assert not (out / "RESERVATION.json").exists()


def test_missing_canonical_mount_fails_before_claim(tmp_path: Path) -> None:
    plan = successor.build_plans(ROOT)["qwen-a"]
    out = tmp_path / "out"
    out.mkdir()
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "missing-jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="canonical jobs root is unavailable"):
        provider(
            plan,
            plan["attempts"][0],
            tmp_path / "missing-claims",
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        )
    assert not (tmp_path / "missing-claims").exists()


def test_engine_reaches_model_boundary_only_after_four_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = copy.deepcopy(successor.build_plans(ROOT)["qwen-a"])
    plan["execution"]["endpoint_lease"]["lease_root"] = str(tmp_path / "leases")
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    monkeypatch.setattr(successor, "build_runtime_plan", lambda *_args: plan)
    out = tmp_path / "out"
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    (tmp_path / "jobs").mkdir()
    provider = successor.AtomicWholeTaskClaims(
        plan,
        out,
        key="test-only",
        jobs_root=tmp_path / "jobs",
        reservation_root=tmp_path / "reservations",
        session_check=lambda *_args: None,
    )

    def stop_at_model(*_args: object) -> dict:
        assert len(list(claim_root.glob("*.json"))) == 4
        assert (out / "RESERVATION.json").is_file()
        raise SystemExit("model boundary reached")

    prior = engine.bulk
    engine.bulk = successor
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    try:
        with pytest.raises(SystemExit, match="model boundary"):
            engine.run_controller(
                plan,
                out=out,
                proxy=tmp_path / "proxy.py",
                claim_root=claim_root,
                model_runner=stop_at_model,
                classifier=lambda *_args: {},
                check_run_absent=lambda *_args: None,
                route_check=lambda *_args: None,
                runtime_gate_check=lambda *_args: None,
                claim_provider=provider,
            )
    finally:
        engine.bulk = prior


def test_held_package_is_closed_create_once_and_cap_two(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    assert [item["kind"] for item in rendered["items"]] == [
        "ConfigMap",
        "Job",
        "ConfigMap",
        "Job",
    ]
    for cm, job in zip(rendered["items"][::2], rendered["items"][1::2], strict=True):
        assert job["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        assert job["spec"]["backoffLimit"] == 0
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"
        assert job["spec"]["activeDeadlineSeconds"] >= 4 * 28_800
        plan = json.loads(cm["data"]["plan.json"])
        assert len(plan["attempts"]) == 4
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        assert [row["attempt"] for row in plan["attempts"]] == [1, 2, 3, 4]
        assert {row["selection_rank"] for row in plan["attempts"]}.isdisjoint({13, 14})

        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        config_root = module_root / "configs"
        config_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        (config_root / "qwen-hosted-generation19-qwen-a-v4.json").write_text(
            cm["data"]["source-plan-v4-a.json"]
        )
        (config_root / "qwen-hosted-generation19-qwen-b-v4.json").write_text(
            cm["data"]["source-plan-v4-b.json"]
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
                    from evals.fleet import qwen_hosted_whole_task_successor_v1_runtime
                    plans = successor.build_plans(Path.cwd())
                    assert {len(plan['attempts']) for plan in plans.values()} == {4}
                    """
                ),
            ],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


def test_released_render_binds_exact_release_without_changing_plans(tmp_path: Path) -> None:
    plans = successor.build_plans(ROOT)
    release = _release(plans)
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    rendered = package.render(ROOT, release_path=release_path)
    for cm, job in zip(rendered["items"][::2], rendered["items"][1::2], strict=True):
        assert (
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "true"
        )
        assert json.loads(cm["data"]["release.json"]) == release
        assert json.loads(cm["data"]["plan.json"])["launch_authorized"] is False
