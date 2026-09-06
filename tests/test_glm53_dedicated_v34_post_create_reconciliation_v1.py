import json
import os
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v34_create_v1 as server

RUN_ID = "glm53-tp8-v34-fdaa27df"
ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status: int = 202, body: bytes = b"") -> None:
        self.status = status
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def row(
    api_run_id: str = RUN_ID,
    *,
    run_dir: str = server.RUN_DIR,
    title: str | None = None,
    status: str = "submitted",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": api_run_id,
        "run_dir": run_dir,
        "status": status,
    }
    if title is not None:
        value["config"] = {"title": title, "run_dir": run_dir}
    return value


class FakeBackend:
    def __init__(
        self,
        snapshots: list[list[dict[str, Any]]],
        *,
        exact: dict[str, dict[str, Any] | None] | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.exact = dict(exact or {})
        self.released: list[str] = []
        self.list_calls = 0

    def list_runs(self) -> tuple[list[dict[str, Any]], int]:
        index = min(self.list_calls, len(self.snapshots) - 1)
        self.list_calls += 1
        return self.snapshots[index], 1

    def get_run(self, api_run_id: str) -> dict[str, Any] | None:
        return self.exact.get(api_run_id)

    def release_run(self, api_run_id: str) -> int:
        self.released.append(api_run_id)
        self.exact[api_run_id] = None
        return 204


@pytest.fixture(autouse=True)
def authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "validate_authorization", lambda _value: None)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")


def create(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    response: FakeResponse | None = None,
) -> dict[str, Any]:
    return server.create_once(
        {"receipt_sha256": "sha256:authorization"},
        result_path=tmp_path / "CREATED.json",
        opener=lambda *_args, **_kwargs: response or FakeResponse(),
        backend=backend,
        sleep=lambda _seconds: None,
    )


@pytest.mark.parametrize("body", [b"", b"not-json", b'{"wrong":"shape"}'])
def test_empty_or_unspecified_202_body_is_never_identity_authority(
    tmp_path: Path, body: bytes
) -> None:
    created = row()
    backend = FakeBackend([[], [created]], exact={RUN_ID: created})
    result = create(tmp_path, backend, response=FakeResponse(body=body))
    assert result["api_run_id"] == RUN_ID
    assert result["create_response_body_authoritative"] is False
    assert result["pre_create_jobs_api_pages"] == 1
    assert result["post_create_jobs_api_pages"] == 1
    assert backend.released == []
    assert json.loads((tmp_path / "CREATED.json").read_text()) == result


def test_post_create_exact_get_may_expose_exact_title(tmp_path: Path) -> None:
    listed = row()
    exact = row(title=server.TITLE, status="running")
    result = create(tmp_path, FakeBackend([[], [listed]], exact={RUN_ID: exact}))
    assert result["api_run_id"] == RUN_ID


def test_exhaustive_immediate_preflight_blocks_any_fresh_identity_collision(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([[row()]])
    called = False

    def opener(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal called
        called = True
        return FakeResponse()

    with pytest.raises(server.CreateError, match="preflight_collision"):
        server.create_once(
            {"receipt_sha256": "sha256:authorization"},
            result_path=tmp_path / "CREATED.json",
            opener=opener,
            backend=backend,
        )
    assert called is False
    assert backend.released == []


def test_two_post_create_candidates_are_both_released(tmp_path: Path) -> None:
    other = "glm53-tp8-v34-deadbeef"
    rows = [row(), row(other)]
    backend = FakeBackend([[], rows], exact={RUN_ID: rows[0], other: rows[1]})
    with pytest.raises(server.CreateError, match="ambiguous_released"):
        create(tmp_path, backend)
    assert backend.released == sorted([RUN_ID, other])
    assert not (tmp_path / "CREATED.json").exists()


@pytest.mark.parametrize(
    "candidate",
    [
        row(run_dir="/mnt/sfs/jobs/wrong", title=server.TITLE),
        row(run_dir=server.RUN_DIR, title="wrong-title"),
        row(api_run_id="ft-run-fdaa27df"),
    ],
)
def test_disagreeing_post_create_identity_is_released(
    tmp_path: Path, candidate: dict[str, Any]
) -> None:
    api_run_id = candidate["name"]
    backend = FakeBackend([[], [candidate]], exact={api_run_id: candidate})
    with pytest.raises(server.CreateError, match="ambiguous_released"):
        create(tmp_path, backend)
    assert backend.released == [api_run_id]


@pytest.mark.parametrize(
    "opener",
    [
        lambda *_args, **_kwargs: FakeResponse(status=500),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("lost response")),
    ],
)
def test_non_202_or_lost_response_reconciles_then_releases(tmp_path: Path, opener: Any) -> None:
    created = row()
    backend = FakeBackend([[], [created]], exact={RUN_ID: created})
    with pytest.raises(server.CreateError, match="http_status_invalid_released"):
        server.create_once(
            {"receipt_sha256": "sha256:authorization"},
            result_path=tmp_path / "CREATED.json",
            opener=opener,
            backend=backend,
            sleep=lambda _seconds: None,
        )
    assert backend.released == [RUN_ID]


def test_exact_get_disagreement_releases_listed_candidate(tmp_path: Path) -> None:
    listed = row()
    exact = row(run_dir="/mnt/sfs/jobs/wrong")
    backend = FakeBackend([[], [listed]], exact={RUN_ID: exact})
    with pytest.raises(server.CreateError, match="ambiguous_released"):
        create(tmp_path, backend)
    assert backend.released == [RUN_ID]


def test_no_post_create_candidate_fails_without_fabricating_identity(tmp_path: Path) -> None:
    backend = FakeBackend([[], []])
    with pytest.raises(server.CreateError, match="identity_absent_reconcile_do_not_retry"):
        create(tmp_path, backend)
    assert backend.list_calls == 1 + server.POST_CREATE_ATTEMPTS
    assert backend.released == []
    assert not (tmp_path / "CREATED.json").exists()


def test_v34_generation_identities_are_fresh_and_score_free() -> None:
    from evals.fleet import glm53_dedicated_v34_controller_package_v1 as package
    from evals.fleet import glm53_dedicated_v34_incluster_parity_v1 as parity
    from evals.fleet import glm53_dedicated_v34_watchdog_package_v1 as watchdog

    assert server.payload()["name"] == server.API_NAME == "glm53-tp8-v34"
    assert server.TITLE.endswith("-v34")
    assert server.RUN_DIR.endswith("-v34")
    assert watchdog.JOB_NAME.endswith("v34-request-watchdog-v1")
    assert parity.JOB_NAME.endswith("v34-incluster-actual-parity-v1")
    assert package.JOB_NAME.endswith("v34-create-watchdog-parity-controller-v1")
    assert server.build_held()["response_body_identity_required"] is False
    assert server.build_held()["scored_launch_authorized"] is False
    assert "FLEET_API_KEY" not in str(server.build_held())
    assert os.environ["FLEET_API_KEY"] == "test-only"


def test_v33_incident_and_release_are_sanitized_and_digest_valid() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    for name in (
        "2026-09-06-glm53-dedicated-v33-empty-create-response-incident-v1.json",
        "2026-09-06-glm53-dedicated-v33-empty-create-response-release-v1.json",
    ):
        value = json.loads((evidence / name).read_text())
        assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
        assert value["protected_content_included"] is False
        assert value["fleet_task_instance_calls"] == 0
        assert value["fleet_session_calls"] == 0
        assert value["verifier_calls"] == 0
        assert value["scoring_calls"] == 0
