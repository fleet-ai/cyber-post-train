from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from evals.fleet import qwen_hosted_identity_selector_package_v2 as package
from evals.fleet import qwen_hosted_identity_selector_v1 as v1
from evals.fleet import qwen_hosted_identity_selector_v2 as selector
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
from evals.fleet import score_blind_session_inventory_v1 as stream

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def _page(
    rows: list[dict[str, object]], *, offset: int = 0, has_more: bool = False
) -> stream.SessionPage:
    return stream.SessionPage(
        sessions=tuple(rows), limit=500, offset=offset, has_more=has_more
    )


def _row(task_key: str, model: object, suffix: str) -> dict[str, object]:
    return {
        "session_id": f"session-{suffix}",
        "eval_task_id": f"eval-task-{suffix}",
        "task_key": task_key,
        "model": model,
        "status": "completed",
    }


def _account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base,
        "_fleet_get",
        lambda *_args, **_kwargs: {"team_name": "fleet", "team_id": base.FLEET_TEAM_ID},
    )


def test_v1_terminal_receipt_and_tombstone_are_exact_and_nonrepeatable() -> None:
    failed = json.loads(
        (
            ROOT
            / "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-identity-selector-v1-failed.json"
        ).read_text()
    )
    terminal = json.loads(
        (
            ROOT
            / "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-identity-selector-v1-terminal.json"
        ).read_text()
    )
    assert failed["receipt_sha256"] == base.digest(failed)
    assert failed["failure_code"] == "read_request_failed"
    assert terminal["receipt_sha256"] == base.digest(terminal)
    assert terminal["failed_receipt_sha256"] == failed["receipt_sha256"]
    assert terminal["retry_v1_authorized"] is False
    assert terminal["scored_successor_authorized"] is False
    assert terminal["deterministic_cause"] == {
        "public_route": "GET /v1/sessions",
        "required_scope": "exactly_one_of_task_key_or_eval_task_id",
        "successor_scope": "one_bound_task_key_per_query",
        "v1_scope": "absent",
    }
    assert all(
        terminal[key] == 0
        for key in (
            "model_calls",
            "task_calls",
            "session_mutations",
            "verifier_calls",
            "scoring_calls",
            "api_mutations",
        )
    )


def test_binding_is_fresh_and_preserves_exact_authority() -> None:
    binding = package.build_binding(ROOT)
    selector.validate_binding(binding)
    assert binding["schema_version"] == selector.BINDING_SCHEMA
    assert binding["output_root"] == selector.OUTPUT_ROOT
    assert binding["selection_file_sha256"] == v1.SELECTION_FILE_SHA256
    assert binding["selection_sha256"] == v1.SELECTION_SHA256
    assert binding["roster_sha256"] == v1.ROSTER_SHA256
    assert binding["authoritative_tally"] == v1.EXPECTED_TALLY


def test_binding_rejects_rehashed_v1_identity_or_authority_drift() -> None:
    for mutate in (
        lambda value: value.__setitem__("schema_version", v1.BINDING_SCHEMA),
        lambda value: value.__setitem__(
            "output_root", "/mnt/sfs/jobs/chris-q38-hosted-identity-selector-v1"
        ),
        lambda value: value["task_roster"][0].__setitem__("task_key", "drift"),
        lambda value: value.__setitem__("protected", True),
    ):
        binding = package.build_binding(ROOT)
        mutate(binding)
        binding["binding_sha256"] = base.binding_digest(binding)
        with pytest.raises(base.GateError, match="identity_selector_binding_invalid"):
            selector.validate_binding(binding)


def test_session_request_always_includes_exact_task_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, list[str]]] = []

    class Response:
        def __init__(self) -> None:
            self.done = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            if self.done:
                return b""
            self.done = True
            return b'{"sessions":[],"limit":500,"offset":0,"has_more":false}'

    class Opener:
        def open(self, request: object, timeout: int) -> Response:
            assert timeout == 60
            query = parse_qs(urlparse(request.full_url).query)  # type: ignore[attr-defined]
            seen.append(query)
            return Response()

    monkeypatch.setattr(selector.urllib.request, "build_opener", lambda *_args: Opener())
    page = selector._session_page("not-persisted", task_key="exact-task", offset=0)  # noqa: SLF001
    assert page.sessions == ()
    assert seen == [{"task_key": ["exact-task"], "limit": ["500"], "offset": ["0"]}]


def test_one_bounded_scan_queries_each_bound_task_and_emits_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = package.build_binding(ROOT)
    rank1, rank2, rank3 = [row["task_key"] for row in binding["task_roster"][:3]]
    calls: list[tuple[str, int]] = []
    marker = "SECRET_SCORE_PROMPT_VERIFIER"

    def page(_api_key: str, *, task_key: str, offset: int) -> stream.SessionPage:
        calls.append((task_key, offset))
        rows: list[dict[str, object]] = []
        if task_key == rank1:
            rows = [_row(task_key, selector.v1.EXPECTED_MODEL, "target")]
        elif task_key == rank2:
            rows = [_row(task_key, "different-model", "other")]
        elif task_key == rank3:
            rows = [_row(task_key, None, "ambiguous")]
        return _page(rows)

    _account(monkeypatch)
    monkeypatch.setattr(selector, "_session_page", page)
    value = selector.select(binding, "not-persisted", job_uid=JOB_UID, pod_uid=POD_UID)
    selector.validate_observation(value, binding)
    assert calls == [(row["task_key"], 0) for row in binding["task_roster"]]
    assert value["classification_counts"] == {
        "IDENTITY_CLEAR": 98,
        "EXACT_MODEL_COLLISION": 1,
        "AMBIGUOUS_BLOCK": 1,
    }
    assert value["earliest_clear_rank"] == 2
    raw = json.dumps(value)
    assert marker not in raw
    assert "session-" not in raw
    assert "eval-task-" not in raw


def test_task_scoped_pagination_advances_within_same_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = package.build_binding(ROOT)
    first = binding["task_roster"][0]["task_key"]
    calls: list[tuple[str, int]] = []

    def page(_api_key: str, *, task_key: str, offset: int) -> stream.SessionPage:
        calls.append((task_key, offset))
        if task_key == first and offset == 0:
            return _page([_row(task_key, "different-model", "one")], has_more=True)
        if task_key == first and offset == 1:
            return _page(
                [_row(task_key, selector.v1.EXPECTED_MODEL, "two")], offset=1
            )
        return _page([])

    _account(monkeypatch)
    monkeypatch.setattr(selector, "_session_page", page)
    value = selector.select(binding, "unused", job_uid=JOB_UID, pod_uid=POD_UID)
    assert calls[:2] == [(first, 0), (first, 1)]
    assert value["classification_counts"]["EXACT_MODEL_COLLISION"] == 1
    assert value["session_pages_read"] == 101


def test_task_scope_mismatch_and_global_duplicate_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = package.build_binding(ROOT)
    first, second = [row["task_key"] for row in binding["task_roster"][:2]]
    _account(monkeypatch)

    def mismatched(_api_key: str, *, task_key: str, offset: int) -> stream.SessionPage:
        if task_key == first:
            return _page([_row(second, "different-model", "mismatch")], offset=offset)
        return _page([], offset=offset)

    monkeypatch.setattr(selector, "_session_page", mismatched)
    with pytest.raises(base.GateError, match="identity_selector_inventory_ambiguous"):
        selector.select(binding, "unused", job_uid=JOB_UID, pod_uid=POD_UID)

    def duplicate(_api_key: str, *, task_key: str, offset: int) -> stream.SessionPage:
        if task_key in {first, second}:
            return _page([_row(task_key, "different-model", "duplicate")], offset=offset)
        return _page([], offset=offset)

    monkeypatch.setattr(selector, "_session_page", duplicate)
    with pytest.raises(base.GateError, match="identity_selector_inventory_ambiguous"):
        selector.select(binding, "unused", job_uid=JOB_UID, pod_uid=POD_UID)


def test_observation_rejects_rehashed_v1_schema_extra_or_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = package.build_binding(ROOT)
    _account(monkeypatch)
    monkeypatch.setattr(
        selector,
        "_session_page",
        lambda *_args, **_kwargs: _page([], offset=_kwargs["offset"]),
    )
    value = selector.select(binding, "unused", job_uid=JOB_UID, pod_uid=POD_UID)
    selector.validate_observation(value, binding)
    for mutate in (
        lambda row: row.__setitem__("schema_version", v1.SCHEMA),
        lambda row: row.__setitem__("protected", "SECRET"),
        lambda row: row.__setitem__("scoring_calls", 1),
    ):
        changed = copy.deepcopy(value)
        mutate(changed)
        changed["receipt_sha256"] = base.digest(changed)
        with pytest.raises(base.GateError, match="release_observation_invalid"):
            selector.validate_observation(changed, binding)


def test_manifest_is_fresh_immutable_create_once_score_free_and_held() -> None:
    configmap, job = package.render(ROOT)["items"]
    assert configmap["metadata"]["name"] == package.CONFIGMAP_NAME
    assert configmap["immutable"] is True
    assert package.JOB_NAME.endswith("-v2")
    assert package.OUTPUT_ROOT.endswith("-v2")
    assert "identity-selector-v1" not in package.JOB_NAME
    assert job["metadata"]["annotations"] | {
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
        "cyber-post-train.fleet.ai/score-free": "true",
    } == job["metadata"]["annotations"]
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 900
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["spec"]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ].get("nvidia.com/gpu") is None
    held = package.held(ROOT)
    persisted = json.loads((ROOT / package.HELD_PATH).read_text())
    assert persisted == held
    assert held["launch_authorized"] is held["scoring_authorized"] is False
    assert held["task_scoped_session_inventory"] is True
    assert set(held["side_effects"].values()) == {0}


def test_projected_package_validates_and_source_drift_fails(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    selector.validate_package_source(tmp_path / "package-source.json", tmp_path)
    (tmp_path / "qwen_hosted_identity_selector_v2.py").write_text("drift")
    with pytest.raises(base.GateError, match="identity_selector_package_file_drifted"):
        selector.validate_package_source(tmp_path / "package-source.json", tmp_path)
