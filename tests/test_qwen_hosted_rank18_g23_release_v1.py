from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
from evals.fleet import qwen_hosted_rank18_g23_held_v1 as held
from evals.fleet import qwen_hosted_rank18_g23_release_observer_v1 as observer
from evals.fleet import qwen_hosted_rank18_g23_release_package_v1 as package
from evals.fleet import score_blind_session_inventory_v1 as stream

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_plan_is_exact_fresh_rank18_whole_task() -> None:
    plan = held.build_plan(ROOT)
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    held.validate_plan(plan, source)
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (18, 1),
        (18, 2),
        (18, 3),
        (18, 4),
    ]
    assert all(row["execution_generation"] == 23 for row in plan["attempts"])
    assert len({row["execution_id"] for row in plan["attempts"]}) == 4
    assert plan["model"] == held.prior.EXPECTED_MODEL
    assert plan["harness"] == held.prior.EXPECTED_HARNESS
    assert plan["treatment"]["tools"] == ["bash", "submit_report"]
    assert plan["launch_authorized"] is False


def test_held_plan_rejects_rehashed_mutation() -> None:
    source = json.loads((ROOT / held.SOURCE_PATH).read_text())
    plan = held.build_plan(ROOT)
    plan["attempts"][0]["run_id"] = "reused"
    plan["plan_sha256"] = held.self_hosted.digest_without(plan, "plan_sha256")
    with pytest.raises(ValueError, match="plan drifted"):
        held.validate_plan(plan, source)


def test_binding_and_finite_paths_are_exact() -> None:
    binding = package.build_binding(ROOT)
    observer.validate_binding(binding)
    assert binding["cells"] == observer.EXPECTED_CELLS
    paths = observer.finite_paths(binding)
    assert set(paths) == {
        "canonical_claim",
        "legacy_per_run_accepted",
        "engine_accepted_registry",
    }
    assert all(len(rows) == 4 and len(set(rows)) == 4 for rows in paths.values())
    package._prove_engine_paths(ROOT)  # noqa: SLF001


@pytest.mark.parametrize(
    "path",
    [
        ("task_key",),
        ("task_version_id",),
        ("cells", 0, "execution_id"),
        ("fresh_object", "job_name"),
        ("plan_sha256",),
    ],
)
def test_binding_rejects_rehashed_mutation(path: tuple[str | int, ...]) -> None:
    binding = package.build_binding(ROOT)
    target: object = binding
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = "drift"  # type: ignore[index]
    binding["binding_sha256"] = base.binding_digest(binding)
    with pytest.raises(base.GateError, match="release_binding_invalid"):
        observer.validate_binding(binding)


def _page(rows: list[dict[str, object]], *, offset: int = 0) -> stream.SessionPage:
    return stream.SessionPage(sessions=tuple(rows), limit=500, offset=offset, has_more=False)


def _row(model: object) -> dict[str, object]:
    return {
        "session_id": "session-1",
        "eval_task_id": "task-id-1",
        "task_key": observer.EXPECTED_TASK_KEY,
        "model": model,
        "status": "completed",
    }


def test_any_exact_model_row_collides_regardless_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observer,
        "_session_page",
        lambda *_args: _page([_row(observer.EXPECTED_SESSION_MODEL)]),
    )
    assert observer._session_collisions(package.build_binding(ROOT), "unused") == (  # noqa: SLF001
        1,
        1,
        1,
    )


def test_different_nonempty_model_is_non_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observer, "_session_page", lambda *_args: _page([_row("other")]))
    assert observer._session_collisions(package.build_binding(ROOT), "unused") == (  # noqa: SLF001
        1,
        1,
        0,
    )


@pytest.mark.parametrize("model", [None, "", 7])
def test_missing_or_malformed_model_fails_closed(
    monkeypatch: pytest.MonkeyPatch, model: object
) -> None:
    monkeypatch.setattr(observer, "_session_page", lambda *_args: _page([_row(model)]))
    with pytest.raises(base.GateError, match="fleet_session_identity_ambiguous"):
        observer._session_collisions(package.build_binding(ROOT), "unused")  # noqa: SLF001


def _mock_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base,
        "_fleet_get",
        lambda *_args, **_kwargs: {"team_name": "fleet", "team_id": base.FLEET_TEAM_ID},
    )
    monkeypatch.setattr(base, "_fresh_objects_absent", lambda _binding: 3)
    monkeypatch.setattr(base, "_sfs_roots_clear", lambda _binding: None)
    monkeypatch.setattr(
        observer,
        "_finite_paths_absent",
        lambda _binding: {
            "canonical_claim": 4,
            "legacy_per_run_accepted": 4,
            "engine_accepted_registry": 4,
        },
    )
    monkeypatch.setattr(observer, "_session_collisions", lambda *_args: (3, 1, 0))
    monkeypatch.setattr(base, "_lease_slots_clear", lambda _binding: 2)


def test_collect_clear_receipt_is_exact_zero_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_clear(monkeypatch)
    binding = package.build_binding(ROOT)
    receipt = observer.collect(
        binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="not-persisted"
    )
    observer.validate_observation(receipt, binding=binding)
    assert receipt["session_collision_policy"].startswith("any_exact_model")
    assert receipt["session_inventory_parser"] == "bounded_streaming_allowlist_v1"
    assert receipt["request_counts"]["protected_value_gets"] == 0
    assert set(
        receipt[field]
        for field in (
            "model_calls",
            "task_calls",
            "session_mutations",
            "verifier_calls",
            "scoring_calls",
            "api_mutations",
        )
    ) == {0}


def test_observation_rejects_rehashed_extra_or_privacy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_clear(monkeypatch)
    binding = package.build_binding(ROOT)
    valid = observer.collect(binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="unused")
    for mutate in (
        lambda value: value.__setitem__("scores_included", True),
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value["request_counts"].__setitem__("protected_value_gets", 1),
    ):
        changed = copy.deepcopy(valid)
        mutate(changed)
        changed["receipt_sha256"] = base.digest(changed)
        with pytest.raises(base.GateError, match="release_observation_invalid"):
            observer.validate_observation(changed, binding=binding)


def test_manifest_is_immutable_create_once_score_free_and_held() -> None:
    configmap, job = package.render(ROOT)["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_projected_package_validates_and_drift_fails(
    tmp_path: Path,
) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    observer.validate_package_source(tmp_path / "package-source.json", tmp_path)
    (tmp_path / "score_blind_session_inventory_v1.py").write_text("drift")
    with pytest.raises(base.GateError, match="release_package_file_drifted"):
        observer.validate_package_source(tmp_path / "package-source.json", tmp_path)


def test_held_receipts_are_exact_and_zero_effect() -> None:
    held_value = json.loads((ROOT / held.HELD_PATH).read_text())
    held.validate_held(held_value, ROOT)
    release = json.loads((ROOT / package.HELD_RELEASE_PATH).read_text())
    assert release == package.held_release(ROOT)
    assert release["launch_authorized"] is False
    assert release["scoring_authorized"] is False
    assert set(release["side_effects"].values()) == {0}
    assert release["privacy"]["verifier_execution_values_materialized"] is False


def _main_args(tmp_path: Path) -> tuple[list[str], Path]:
    output = tmp_path / "out" / "OBSERVATION.json"
    return (
        [
            "--binding",
            str(tmp_path / "binding.json"),
            "--package-source",
            str(tmp_path / "package-source.json"),
            "--output",
            str(output),
        ],
        output,
    )


def test_invalid_runtime_uid_is_redacted_before_any_fallible_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk_live_secret123"
    monkeypatch.setenv("JOB_UID", secret)
    monkeypatch.setenv("POD_UID", POD_UID)
    monkeypatch.setattr(
        observer,
        "validate_package_source",
        lambda *_args: pytest.fail("package validation must not run"),
    )
    args, output = _main_args(tmp_path)
    with pytest.raises(base.GateError, match="observer_failed_safely") as caught:
        observer.main(args)
    failure_text = output.with_name("FAILED.json").read_text()
    failure = json.loads(failure_text)
    assert secret not in failure_text
    assert secret not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert failure["last_stage"] == "runtime-identity"
    assert failure["failure_code"] == "observer_job_uid_invalid"
    assert failure["job_uid"] is None
    assert failure["pod_uid"] is None


@pytest.mark.parametrize("failure_stage", ["package-source", "binding", "collect"])
def test_early_failures_emit_only_canonical_validated_runtime_uids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    raw_job_uid = "{AAAAAAAA-1111-4111-8111-111111111111}"
    canonical_job_uid = "aaaaaaaa-1111-4111-8111-111111111111"
    monkeypatch.setenv("JOB_UID", raw_job_uid)
    monkeypatch.setenv("POD_UID", POD_UID)
    monkeypatch.setenv("FLEET_API_KEY", "not-persisted")

    if failure_stage == "package-source":
        monkeypatch.setattr(
            observer,
            "validate_package_source",
            lambda *_args: (_ for _ in ()).throw(base.GateError("release_package_source_invalid")),
        )
    else:
        monkeypatch.setattr(observer, "validate_package_source", lambda *_args: None)
        if failure_stage == "binding":
            monkeypatch.setattr(
                base,
                "load_projected",
                lambda *_args: (_ for _ in ()).throw(base.GateError("input_path_unsafe")),
            )
        else:
            monkeypatch.setattr(base, "load_projected", lambda *_args: package.build_binding(ROOT))
            monkeypatch.setattr(
                observer,
                "collect",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    base.GateError("fleet_session_identity_ambiguous")
                ),
            )

    args, output = _main_args(tmp_path)
    with pytest.raises(base.GateError, match="observer_failed_safely"):
        observer.main(args)
    failure = json.loads(output.with_name("FAILED.json").read_text())
    assert failure["last_stage"] == failure_stage
    assert raw_job_uid not in json.dumps(failure)
    assert failure["job_uid"] == canonical_job_uid
    assert failure["pod_uid"] == POD_UID
    assert failure["credentials_included"] is False
