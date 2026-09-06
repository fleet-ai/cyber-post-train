from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as observer_v1
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v3 as observer
from evals.fleet import qwen_hosted_rank17_g22_release_package_v3 as package

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_finite_paths_are_exact_exhaustive_engine_contract() -> None:
    binding = package.build_binding(ROOT)
    paths = observer.finite_paths(binding)
    run_ids = [row["run_id"] for row in binding["cells"]]
    assert paths == {
        "canonical_claim": binding["planned_claim_paths"],
        "legacy_per_run_accepted": [
            f"/mnt/sfs/jobs/{run_id}/ACCEPTED.json" for run_id in run_ids
        ],
        "engine_accepted_registry": [
            f"{observer.SCORED_ROOT}/accepted/{run_id}.json" for run_id in run_ids
        ],
    }
    assert all(len(rows) == 4 and len(rows) == len(set(rows)) for rows in paths.values())
    package._prove_engine_paths(ROOT)  # noqa: SLF001


@pytest.mark.parametrize("receipt_class", list(observer.finite_paths(package.build_binding(ROOT))))
def test_any_exact_finite_path_existence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt_class: str
) -> None:
    binding = package.build_binding(ROOT)
    path = tmp_path / "malformed.json"
    monkeypatch.setattr(observer, "finite_paths", lambda _binding: {receipt_class: [str(path)]})
    path.write_text("malformed content is still a collision")
    with pytest.raises(observer_v1.GateError, match="planned_statistical_cell_collision"):
        observer._finite_paths_absent(binding)  # noqa: SLF001


def _mock(monkeypatch: pytest.MonkeyPatch, *, sessions: int = 0) -> None:
    monkeypatch.setattr(
        observer_v1,
        "_fleet_get",
        lambda path, _key, _params=None: (
            {"team_name": "fleet", "team_id": observer_v1.FLEET_TEAM_ID}
            if path == "/v1/account"
            else {"sessions": [], "has_more": False}
        ),
    )
    monkeypatch.setattr(observer_v1, "_fresh_objects_absent", lambda _binding: 3)
    monkeypatch.setattr(observer_v1, "_sfs_roots_clear", lambda _binding: None)
    monkeypatch.setattr(observer, "_finite_paths_absent", lambda _binding: {
        "canonical_claim": 4,
        "legacy_per_run_accepted": 4,
        "engine_accepted_registry": 4,
    })
    monkeypatch.setattr(observer, "_session_collisions", lambda *_args: (5, 1, sessions))
    monkeypatch.setattr(observer_v1, "_lease_slots_clear", lambda _binding: 2)


def test_collect_is_score_blind_and_never_uses_recursive_receipt_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock(monkeypatch)
    monkeypatch.setattr(
        observer_v1,
        "_file_identity_collisions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not scan")),
    )
    binding = package.build_binding(ROOT)
    receipt = observer.collect(
        binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="not-persisted"
    )
    observer.validate_observation(receipt, binding=binding)
    assert receipt["observed_aggregates"] == {
        "canonical_claim_paths_absent": 4,
        "legacy_per_run_accepted_paths_absent": 4,
        "engine_accepted_registry_paths_absent": 4,
        "authoritative_session_rows_examined": 5,
        "fresh_object_sets_absent": 1,
        "checked_sfs_roots_absent": 2,
        "endpoint_lease_slots_simultaneously_free": 2,
    }
    assert set(receipt["collisions"].values()) == {0}
    assert receipt["request_counts"]["transcript_prompt_task_verifier_or_score_gets"] == 0
    assert receipt["scores_included"] is False
    assert receipt["prompts_traces_flags_included"] is False


def test_collect_rejects_authoritative_session_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock(monkeypatch, sessions=1)
    with pytest.raises(observer_v1.GateError, match="planned_statistical_cell_collision"):
        observer.collect(
            package.build_binding(ROOT),
            job_uid=JOB_UID,
            pod_uid=POD_UID,
            api_key="not-persisted",
        )


@pytest.mark.parametrize(
    "row",
    [
        {
            "session_id": "one",
            "model": observer_v1.EXPECTED_SESSION_MODEL,
            "eval_task_version_id": "11111111-1111-4111-8111-111111111111",
            "task_version_id": observer_v1.EXPECTED_TASK_VERSION_ID,
        },
        {
            "session_id": "one",
            "model": observer_v1.EXPECTED_SESSION_MODEL,
            "eval_task_version_id": "",
            "task_version_id": observer_v1.EXPECTED_TASK_VERSION_ID,
        },
        {
            "session_id": "one",
            "model": observer_v1.EXPECTED_SESSION_MODEL,
            "eval_task_version_id": 7,
            "task_version_id": observer_v1.EXPECTED_TASK_VERSION_ID,
        },
    ],
)
def test_session_version_alias_disagreement_or_invalid_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, object]
) -> None:
    monkeypatch.setattr(
        observer_v1,
        "_fleet_get",
        lambda *_args, **_kwargs: {"sessions": [row], "has_more": False},
    )
    with pytest.raises(observer_v1.GateError, match="fleet_session_identity_ambiguous"):
        observer._session_collisions(  # noqa: SLF001
            package.build_binding(ROOT), "unused", set()
        )


def test_equal_session_version_aliases_collide_for_exact_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "session_id": "one",
        "model": observer_v1.EXPECTED_SESSION_MODEL,
        "eval_task_version_id": observer_v1.EXPECTED_TASK_VERSION_ID,
        "task_version_id": observer_v1.EXPECTED_TASK_VERSION_ID,
    }
    monkeypatch.setattr(
        observer_v1,
        "_fleet_get",
        lambda *_args, **_kwargs: {"sessions": [row], "has_more": False},
    )
    assert observer._session_collisions(  # noqa: SLF001
        package.build_binding(ROOT), "unused", set()
    ) == (1, 1, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["observed_aggregates"].__setitem__(
            "engine_accepted_registry_paths_absent", 3
        ),
        lambda value: value["collisions"].__setitem__("authoritative_sessions", 1),
        lambda value: value.__setitem__("scores_included", True),
        lambda value: value.__setitem__("unexpected", True),
    ],
)
def test_observation_rejects_rehashed_mutations(
    monkeypatch: pytest.MonkeyPatch, mutate: object
) -> None:
    _mock(monkeypatch)
    binding = package.build_binding(ROOT)
    valid = observer.collect(binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="unused")
    mutated = copy.deepcopy(valid)
    mutate(mutated)  # type: ignore[operator]
    mutated["receipt_sha256"] = observer_v1.digest(mutated)
    with pytest.raises(observer_v1.GateError, match="release_observation_invalid"):
        observer.validate_observation(mutated, binding=binding)


def test_projected_package_binds_v1_v3_and_exact_binding(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    observer.validate_package_source(tmp_path / "package-source.json", tmp_path)
    (tmp_path / "qwen_hosted_rank17_g22_release_observer_v3.py").write_text("drift")
    with pytest.raises(observer_v1.GateError, match="release_package_file_drifted"):
        observer.validate_package_source(tmp_path / "package-source.json", tmp_path)


def test_manifest_is_fresh_create_once_immutable_score_free_and_nonpreempting() -> None:
    configmap, job = package.render(ROOT)["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == package.CONFIGMAP_NAME
    assert job["metadata"]["name"] == package.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/score-free"] == "true"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_failure_boundary_has_no_dynamic_exception_text_or_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "OBSERVATION.json"
    monkeypatch.setenv("JOB_UID", JOB_UID)
    monkeypatch.setenv("POD_UID", POD_UID)
    monkeypatch.setattr(
        observer,
        "validate_package_source",
        lambda *_args: (_ for _ in ()).throw(
            type("sk_live_secret123", (RuntimeError,), {})("sk_live_secret123")
        ),
    )
    with pytest.raises(observer_v1.GateError, match="observer_failed_safely") as caught:
        observer.main(
            [
                "--binding",
                str(tmp_path / "binding.json"),
                "--package-source",
                str(tmp_path / "package-source.json"),
                "--output",
                str(output),
            ]
        )
    formatted = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    failure_text = output.with_name("FAILED.json").read_text()
    assert "sk_live_secret123" not in formatted
    assert "sk_live_secret123" not in failure_text
    assert "error_type" not in failure_text
    assert "error_sha256" not in failure_text


def test_held_release_is_exact_zero_effect_and_self_digested() -> None:
    actual = json.loads((ROOT / package.HELD_PATH).read_text())
    assert actual == package.held(ROOT)
    assert actual["status"] == "HELD_PENDING_INDEPENDENT_REVIEW"
    assert actual["launch_authorized"] is False
    assert actual["scoring_authorized"] is False
    assert set(actual["side_effects"].values()) == {0}
    assert actual["finite_path_authority"]["recursive_historical_jobs_scan"] is False
    assert actual["receipt_sha256"] == observer_v1.digest(actual)
