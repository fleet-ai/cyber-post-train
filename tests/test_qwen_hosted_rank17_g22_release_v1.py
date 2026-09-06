from __future__ import annotations

import copy
import fcntl
import json
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as observer
from evals.fleet import qwen_hosted_rank17_g22_release_package_v1 as package

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_binding_exactly_binds_plan_commit_tally_and_four_cells() -> None:
    binding = package.build_binding(ROOT)
    observer.validate_binding(binding)
    assert binding["plan_commit"] == "e7f1772ee7727e348523735387f2a7861308553e"
    assert binding["authoritative_tally"] == {
        "accepted": 51,
        "active": 0,
        "blocked_nonrepeatable": 8,
        "unstarted": 341,
    }
    assert binding["statistical_cell_count"] == 4
    assert [row["attempt"] for row in binding["cells"]] == [1, 2, 3, 4]
    assert len(binding["identity_values"]) == 12
    assert binding["fresh_object"] == {
        "job_name": "chris-q38-hosted-r017-whole-task-g22-v1",
        "configmap_name": "chris-q38-hosted-r017-whole-task-g22-package-v1",
    }
    assert binding["task_version_id"] == observer.EXPECTED_TASK_VERSION_ID
    assert binding["session_model"] == observer.EXPECTED_SESSION_MODEL
    assert binding["planned_claim_paths"] == [
        f"{binding['claim_root']}/{row['execution_id'].removeprefix('sha256:')}.json"
        for row in binding["cells"]
    ]
    assert binding["planned_accepted_paths"] == [
        f"/mnt/sfs/jobs/{row['run_id']}/ACCEPTED.json" for row in binding["cells"]
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["authoritative_tally"].__setitem__("accepted", 52),
        lambda value: value.__setitem__("plan_commit", "0" * 40),
        lambda value: value["cells"][0].__setitem__("cell_id", "sha256:" + "0" * 64),
        lambda value: value.__setitem__("task_version_id", "0" * 36),
        lambda value: value.__setitem__("session_model", "other-model"),
        lambda value: value["planned_claim_paths"].__setitem__(0, "/tmp/other.json"),
        lambda value: value["planned_accepted_paths"].__setitem__(0, "/tmp/ACCEPTED.json"),
        lambda value: value["fresh_object"].__setitem__("job_name", "reused"),
        lambda value: value.__setitem__("unexpected", True),
    ],
)
def test_binding_rejects_rehashed_mutation(mutate: object) -> None:
    binding = package.build_binding(ROOT)
    mutate(binding)  # type: ignore[operator]
    binding["binding_sha256"] = observer.binding_digest(binding)
    with pytest.raises(observer.GateError, match="release_binding_invalid"):
        observer.validate_binding(binding)


def test_manifest_is_create_once_immutable_score_free_and_nonpreempting() -> None:
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
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    binding_text = configmap["data"]["binding.json"]
    assert "prompt" not in binding_text
    assert "trace" not in binding_text
    assert "flag" not in binding_text
    assert "credential" not in binding_text
    assert re.search(r"sk_[A-Za-z0-9]{12,}", json.dumps(rendered)) is None


def test_package_source_binds_exact_projected_bytes(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    observer.validate_package_source(tmp_path / "package-source.json", tmp_path)
    assert observer.load_projected(
        tmp_path / "binding.json", tmp_path
    ) == package.build_binding(ROOT)
    (tmp_path / "binding.json").write_text("{}")
    with pytest.raises(observer.GateError, match="release_package_file_drifted"):
        observer.validate_package_source(tmp_path / "package-source.json", tmp_path)


def test_package_source_allows_kubernetes_two_hop_projection_and_rejects_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bootstrap"
    revision = root / "..2026_09_06"
    revision.mkdir(parents=True)
    (root / "..data").symlink_to(revision.name)
    files = {"binding.json": "{}\n", "qwen_hosted_rank17_g22_release_observer_v1.py": "# source\n"}
    body = {
        "schema_version": observer.PACKAGE_SCHEMA,
        "files": {name: observer.sha256(value.encode()) for name, value in files.items()},
        "file_count": 2,
    }
    for name, value in files.items():
        (revision / name).write_text(value)
        (root / name).symlink_to(Path("..data") / name)
    (revision / "package-source.json").write_text(json.dumps(observer.seal(body)))
    source = root / "package-source.json"
    source.symlink_to(Path("..data") / "package-source.json")
    observer.validate_package_source(source, root)
    assert observer.load_projected(root / "binding.json", root) == {}

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(observer.seal(body)))
    escaped = root / "escaped.json"
    escaped.symlink_to(outside)
    with pytest.raises(observer.GateError, match="release_package_source_escape"):
        observer.validate_package_source(escaped, root)


def test_file_collision_and_two_slot_lease_gates(tmp_path: Path) -> None:
    claims = tmp_path / "claims"
    jobs = tmp_path / "jobs"
    lease = tmp_path / "leases" / "endpoint"
    claims.mkdir()
    jobs.mkdir()
    lease.mkdir(parents=True)
    for slot in (1, 2):
        (lease / f"slot-{slot}.lock").write_bytes(b"")
    expected = {"sha256:" + "1" * 64}
    (claims / "unrelated.json").write_text(
        json.dumps(
            observer.seal(
                {
                    "schema_version": "test-sanitized-claim-v1",
                    "cell_id": "sha256:" + "2" * 64,
                }
            )
        )
    )
    assert observer._file_identity_collisions(claims, expected, accepted_only=False) == (1, 0)  # noqa: SLF001
    assert observer._file_identity_collisions(jobs, expected, accepted_only=True) == (0, 0)  # noqa: SLF001
    binding = {"lease_root": str(tmp_path / "leases"), "endpoint_key": "endpoint"}
    assert observer._lease_slots_clear(binding) == 2  # noqa: SLF001
    with (lease / "slot-1.lock").open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(observer.GateError, match="endpoint_lease_slot_held"):
            observer._lease_slots_clear(binding)  # noqa: SLF001


@pytest.mark.parametrize("body", [b"not-json", b"{}"])
def test_scanners_reject_malformed_or_unsealed_other_receipts(
    tmp_path: Path, body: bytes
) -> None:
    claims = tmp_path / "claims"
    jobs = tmp_path / "jobs" / "other-run"
    claims.mkdir()
    jobs.mkdir(parents=True)
    (claims / "other.json").write_bytes(body)
    with pytest.raises(observer.GateError, match="invalid_json|scan_receipt_invalid"):
        observer._file_identity_collisions(claims, set(), accepted_only=False)  # noqa: SLF001
    (claims / "other.json").unlink()
    (jobs / "ACCEPTED.json").write_bytes(body)
    with pytest.raises(observer.GateError, match="invalid_json|scan_receipt_invalid"):
        observer._file_identity_collisions(tmp_path / "jobs", set(), accepted_only=True)  # noqa: SLF001


@pytest.mark.parametrize("kind", ["claim", "accepted"])
@pytest.mark.parametrize("body", [b"", b"not-json", b"{}"])
def test_exact_planned_path_existence_is_collision_regardless_body(
    tmp_path: Path, kind: str, body: bytes
) -> None:
    path = tmp_path / kind / "planned.json"
    path.parent.mkdir()
    path.write_bytes(body)
    code = f"planned_{kind}_path_collision"
    with pytest.raises(observer.GateError, match=code):
        observer._planned_paths_absent([str(path)], failure_code=code)  # noqa: SLF001


@pytest.mark.parametrize(
    "page",
    [
        {},
        {"sessions": None, "has_more": False},
        {"sessions": [], "has_more": None},
        {"sessions": [], "has_more": "false"},
    ],
)
def test_session_inventory_requires_typed_page(
    monkeypatch: pytest.MonkeyPatch, page: dict[str, object]
) -> None:
    monkeypatch.setattr(observer, "_fleet_get", lambda *_args, **_kwargs: page)
    with pytest.raises(observer.GateError, match="fleet_session_inventory_invalid"):
        observer._session_collisions(  # noqa: SLF001
            {
                "task_key": observer.EXPECTED_TASK_KEY,
                "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                "session_model": observer.EXPECTED_SESSION_MODEL,
            },
            "unused",
            set(),
        )


@pytest.mark.parametrize(
    "row",
    [
        {"session_id": "one"},
        {"session_id": "one", "model": observer.EXPECTED_SESSION_MODEL},
    ],
)
def test_target_session_missing_or_ambiguous_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, object]
) -> None:
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {"sessions": [row], "has_more": False},
    )
    with pytest.raises(observer.GateError, match="fleet_session_identity_ambiguous"):
        observer._session_collisions(  # noqa: SLF001
            {
                "task_key": observer.EXPECTED_TASK_KEY,
                "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                "session_model": observer.EXPECTED_SESSION_MODEL,
            },
            "unused",
            {"planned"},
        )


@pytest.mark.parametrize("metadata", [None, {}, {"run_id": "unknown-treatment-identity"}])
def test_exact_target_model_and_version_always_collides(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, object] | None
) -> None:
    row = {
        "session_id": "target",
        "model": observer.EXPECTED_SESSION_MODEL,
        "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
    }
    if metadata is not None:
        row["metadata"] = metadata
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {"sessions": [row], "has_more": False},
    )
    assert observer._session_collisions(  # noqa: SLF001
        {
            "task_key": observer.EXPECTED_TASK_KEY,
            "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
            "session_model": observer.EXPECTED_SESSION_MODEL,
        },
        "unused",
        {"planned"},
    ) == (1, 1, 1)


def test_exact_model_with_different_version_is_positive_non_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [
                {
                    "session_id": "other-version",
                    "model": observer.EXPECTED_SESSION_MODEL,
                    "task_version_id": "11111111-1111-4111-8111-111111111111",
                }
            ],
            "has_more": False,
        },
    )
    assert observer._session_collisions(  # noqa: SLF001
        {
            "task_key": observer.EXPECTED_TASK_KEY,
            "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
            "session_model": observer.EXPECTED_SESSION_MODEL,
        },
        "unused",
        {"planned"},
    ) == (1, 1, 0)


def test_target_session_exact_identity_collides_and_other_model_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"session_id": "other", "model": "other-model"},
        {
            "session_id": "target",
            "model": observer.EXPECTED_SESSION_MODEL,
            "metadata": {"run_id": "planned"},
            "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
        },
    ]
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {"sessions": rows, "has_more": False},
    )
    assert observer._session_collisions(  # noqa: SLF001
        {
            "task_key": observer.EXPECTED_TASK_KEY,
            "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
            "session_model": observer.EXPECTED_SESSION_MODEL,
        },
        "unused",
        {"planned"},
    ) == (2, 1, 1)


def test_repeated_nonempty_session_page_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = {
        "sessions": [{"session_id": "repeated", "model": "other-model"}],
        "has_more": True,
    }
    monkeypatch.setattr(observer, "_fleet_get", lambda *_args, **_kwargs: page)
    with pytest.raises(observer.GateError, match="fleet_session_pagination_stalled"):
        observer._session_collisions(  # noqa: SLF001
            {
                "task_key": observer.EXPECTED_TASK_KEY,
                "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                "session_model": observer.EXPECTED_SESSION_MODEL,
            },
            "unused",
            set(),
        )


@pytest.mark.parametrize("limit_name", ["MAX_SESSION_PAGES", "MAX_SESSION_ROWS"])
def test_session_inventory_has_fixed_page_and_row_ceilings(
    monkeypatch: pytest.MonkeyPatch, limit_name: str
) -> None:
    monkeypatch.setattr(observer, limit_name, 0)
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda *_args, **_kwargs: {
            "sessions": [{"session_id": "one", "model": "other-model"}],
            "has_more": False,
        },
    )
    with pytest.raises(observer.GateError, match="fleet_session_pagination_stalled"):
        observer._session_collisions(  # noqa: SLF001
            {
                "task_key": observer.EXPECTED_TASK_KEY,
                "task_version_id": observer.EXPECTED_TASK_VERSION_ID,
                "session_model": observer.EXPECTED_SESSION_MODEL,
            },
            "unused",
            set(),
        )


def _mock_collect_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observer,
        "_fleet_get",
        lambda path, _key, _params=None: (
            {"team_name": "fleet", "team_id": observer.FLEET_TEAM_ID}
            if path == "/v1/account"
            else {"sessions": [], "has_more": False}
        ),
    )
    monkeypatch.setattr(observer, "_fresh_objects_absent", lambda _binding: 3)
    monkeypatch.setattr(observer, "_sfs_roots_clear", lambda _binding: None)
    monkeypatch.setattr(observer, "_file_identity_collisions", lambda *_args, **_kwargs: (7, 0))
    monkeypatch.setattr(observer, "_session_collisions", lambda *_args: (0, 1, 0))
    monkeypatch.setattr(observer, "_lease_slots_clear", lambda _binding: 2)


def test_collect_emits_only_score_blind_clear_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_collect_dependencies(monkeypatch)
    binding = package.build_binding(ROOT)
    receipt = observer.collect(binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="not-persisted")
    observer.validate_observation(receipt, binding=binding)
    assert receipt["authoritative_tally"] == observer.EXPECTED_TALLY
    assert receipt["all_planned_cells_observed_unstarted"] is True
    assert set(receipt["collisions"].values()) == {0}
    assert receipt["methods"] == ["GET"]
    assert receipt["request_counts"]["transcript_prompt_task_verifier_or_score_gets"] == 0
    assert receipt["scores_included"] is False
    assert receipt["prompts_traces_flags_included"] is False


def test_collect_fails_on_any_planned_cell_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_collect_dependencies(monkeypatch)
    monkeypatch.setattr(observer, "_file_identity_collisions", lambda *_args, **_kwargs: (7, 1))
    with pytest.raises(observer.GateError, match="planned_statistical_cell_collision"):
        observer.collect(
            package.build_binding(ROOT), job_uid=JOB_UID, pod_uid=POD_UID, api_key="not-persisted"
        )


def test_observation_rejects_rehashed_tally_or_effect_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_collect_dependencies(monkeypatch)
    binding = package.build_binding(ROOT)
    valid = observer.collect(binding, job_uid=JOB_UID, pod_uid=POD_UID, api_key="not-persisted")
    for mutate in (
        lambda value: value["authoritative_tally"].__setitem__("accepted", 52),
        lambda value: value.__setitem__("all_planned_cells_observed_unstarted", False),
        lambda value: value.__setitem__("model_calls", 1),
        lambda value: value.__setitem__("unexpected", True),
    ):
        receipt = copy.deepcopy(valid)
        mutate(receipt)
        receipt["receipt_sha256"] = observer.digest(receipt)
        with pytest.raises(observer.GateError, match="release_observation_invalid"):
            observer.validate_observation(receipt, binding=binding)


def test_failure_receipt_redacts_unexpected_text_and_raises_log_safe_terminal(
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
    with pytest.raises(observer.GateError, match="observer_failed_safely") as caught:
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
    assert "sk_live_secret123" not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    failure_text = output.with_name("FAILED.json").read_text()
    failure = json.loads(failure_text)
    assert failure["failure_code"] == "redacted"
    assert failure["failure_category"] == "unexpected_failure"
    assert "error_type" not in failure
    assert "error_sha256" not in failure
    assert "sk_live_secret123" not in failure_text
    assert failure["receipt_sha256"] == observer.digest(failure)

    collision_output = tmp_path / "collision" / "OBSERVATION.json"
    collision_output.parent.mkdir()
    collision_output.with_name("FAILED.json").write_text("existing")
    with pytest.raises(observer.GateError, match="observer_failed_safely") as collision:
        observer.main(
            [
                "--binding",
                str(tmp_path / "binding.json"),
                "--package-source",
                str(tmp_path / "package-source.json"),
                "--output",
                str(collision_output),
            ]
        )
    assert "sk_live_secret123" not in "".join(
        traceback.format_exception(collision.type, collision.value, collision.tb)
    )


def test_held_release_is_exact_nonexecuting_and_self_digested() -> None:
    value = json.loads((ROOT / package.HELD_RELEASE_PATH).read_text())
    package.validate_held_release(value, ROOT)
    assert value["status"] == "HELD_PENDING_SCORE_BLIND_OBSERVER"
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["authoritative_tally"] == observer.EXPECTED_TALLY
    assert set(value["side_effects"].values()) == {0}
    assert value["receipt_sha256"] == observer.digest(value)


def test_observed_timestamp_contract_is_second_precision_utc() -> None:
    assert observer.UTC_RE.fullmatch(datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
