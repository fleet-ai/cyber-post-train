import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v34_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v34_create_v1 as server
from evals.fleet import glm53_dedicated_v34_launch_v1 as launch

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
        list_failures: set[int] | None = None,
        release_failures: set[str] | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.exact = dict(exact or {})
        self.list_failures = set(list_failures or set())
        self.release_failures = set(release_failures or set())
        self.released: list[str] = []
        self.release_attempts: list[str] = []
        self.list_calls = 0

    def list_runs(self) -> tuple[list[dict[str, Any]], int]:
        call = self.list_calls
        self.list_calls += 1
        if call in self.list_failures:
            raise OSError("dynamic detail must not escape")
        index = min(call, len(self.snapshots) - 1)
        return self.snapshots[index], 1

    def get_run(self, api_run_id: str) -> dict[str, Any] | None:
        return self.exact.get(api_run_id)

    def release_run(self, api_run_id: str) -> int:
        self.release_attempts.append(api_run_id)
        if api_run_id in self.release_failures:
            raise OSError("dynamic detail must not escape")
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
    with pytest.raises(server.CreateError, match="ambiguous.*server_released"):
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
    with pytest.raises(server.CreateError, match="ambiguous.*server_released"):
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
    with pytest.raises(server.CreateError, match="http_status_invalid_server_released"):
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
    with pytest.raises(server.CreateError, match="ambiguous.*server_released"):
        create(tmp_path, backend)
    assert backend.released == [RUN_ID]


def test_no_post_create_candidate_fails_without_fabricating_identity(tmp_path: Path) -> None:
    backend = FakeBackend([[], []])
    with pytest.raises(server.CreateError, match="cleanup_incomplete_do_not_retry"):
        create(tmp_path, backend)
    assert backend.list_calls == (
        1 + server.POST_CREATE_ATTEMPTS + server.POST_CREATE_CLEANUP_ATTEMPTS
    )
    assert backend.released == []
    assert not (tmp_path / "CREATED.json").exists()


def test_transient_post_create_inventory_errors_reconcile_without_duplicate_post(
    tmp_path: Path,
) -> None:
    created = row()
    backend = FakeBackend(
        [[], [created]],
        exact={RUN_ID: created},
        list_failures={1, 2},
    )
    result = create(tmp_path, backend)
    assert result["api_run_id"] == RUN_ID
    assert backend.list_calls == 4
    assert backend.release_attempts == []


def test_post_create_inventory_failure_runs_cleanup_and_releases_discovered_server(
    tmp_path: Path,
) -> None:
    created = row()
    backend = FakeBackend(
        [[], [created]],
        exact={RUN_ID: created},
        list_failures={1, 2, 3},
    )
    with pytest.raises(server.CreateError, match="inventory_unavailable.*server_released"):
        create(tmp_path, backend)
    assert backend.released == [RUN_ID]
    assert not (tmp_path / "CREATED.json").exists()


def test_post_create_exact_get_failure_runs_cleanup_and_releases_server(
    tmp_path: Path,
) -> None:
    created = row()

    class ExactGetFailureBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__([[], [created]], exact={RUN_ID: created})
            self.exact_get_calls = 0

        def get_run(self, api_run_id: str) -> dict[str, Any] | None:
            self.exact_get_calls += 1
            if self.exact_get_calls <= server.POST_CREATE_TRANSIENT_ERRORS:
                raise OSError("dynamic detail must not escape")
            return super().get_run(api_run_id)

    backend = ExactGetFailureBackend()
    with pytest.raises(server.CreateError, match="exact_get_unavailable.*server_released"):
        create(tmp_path, backend)
    assert backend.released == [RUN_ID]
    assert not (tmp_path / "CREATED.json").exists()


def test_unexpected_post_exception_still_reconciles_and_releases_server(
    tmp_path: Path,
) -> None:
    created = row()
    backend = FakeBackend([[], [created]], exact={RUN_ID: created})

    def unexpected(*_args: Any, **_kwargs: Any) -> FakeResponse:
        raise ValueError("dynamic detail must not escape")

    with pytest.raises(server.CreateError, match="post_create_failure_server_released"):
        server.create_once(
            {"receipt_sha256": "sha256:authorization"},
            result_path=tmp_path / "CREATED.json",
            opener=unexpected,
            backend=backend,
            sleep=lambda _seconds: None,
        )
    assert backend.released == [RUN_ID]


def test_result_parent_creation_failure_releases_reconciled_server(tmp_path: Path) -> None:
    created = row()
    backend = FakeBackend([[], [created]], exact={RUN_ID: created})
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("regular file")
    with pytest.raises(server.CreateError, match="post_create_failure_server_released"):
        server.create_once(
            {"receipt_sha256": "sha256:authorization"},
            result_path=blocked_parent / "CREATED.json",
            opener=lambda *_args, **_kwargs: FakeResponse(),
            backend=backend,
            sleep=lambda _seconds: None,
        )
    assert backend.released == [RUN_ID]


def test_result_write_failure_releases_reconciled_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = row()
    backend = FakeBackend([[], [created]], exact={RUN_ID: created})
    result_path = tmp_path / "CREATED.json"
    original_open = Path.open

    def fail_target(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == result_path:
            raise OSError("dynamic detail must not escape")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target)
    with pytest.raises(server.CreateError, match="post_create_failure_server_released"):
        server.create_once(
            {"receipt_sha256": "sha256:authorization"},
            result_path=result_path,
            opener=lambda *_args, **_kwargs: FakeResponse(),
            backend=backend,
            sleep=lambda _seconds: None,
        )
    assert backend.released == [RUN_ID]


def test_release_attempts_every_safe_candidate_when_first_delete_fails() -> None:
    other = "glm53-tp8-v34-deadbeef"
    rows = [row(), row(other)]
    backend = FakeBackend(
        [rows],
        exact={RUN_ID: rows[0], other: rows[1]},
        release_failures={RUN_ID},
    )
    with pytest.raises(server.CreateError, match="release_incomplete_do_not_retry") as error:
        server._release_candidates(backend, rows)  # noqa: SLF001
    assert str(error.value) == "v34_post_create_release_incomplete_do_not_retry"
    assert backend.release_attempts == sorted([RUN_ID, other])
    assert backend.released == [other]


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


def test_v34_held_launch_receipt_binds_frozen_package_and_zero_effects() -> None:
    evidence = (
        ROOT / "docs/evidence/glm53-study" / "2026-09-06-glm53-dedicated-v34-launch-held-v1.json"
    )
    value = json.loads(evidence.read_text())
    assert value == launch.build_held(ROOT)
    assert value["package_commit"] == launch.FROZEN_PACKAGE_COMMIT
    assert value["package_sha256"] == launch.FROZEN_PACKAGE_SHA256
    assert value["response_body_identity_required"] is False
    assert value["exhaustive_pre_post_jobs_api_reconciliation_required"] is True
    assert value["ambiguous_post_create_identity_release_required"] is True
    assert value["post_create_transient_error_limit"] == 3
    assert value["post_create_cleanup_attempt_limit"] == 3
    assert value["all_safe_candidates_release_attempted"] is True
    assert value["result_parent_and_write_failure_release_required"] is True
    assert value["api_mutation_calls"] == 0
    assert value["kubernetes_mutation_calls"] == 0
    assert value["server_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False


def test_v34_materialized_controller_package_has_complete_import_closure(
    tmp_path: Path,
) -> None:
    configmap = package.build_source_configmap(ROOT, launch.FROZEN_PACKAGE_COMMIT)
    manifest = json.loads(configmap["data"]["package.json"])
    assert manifest["package_sha256"] == launch.FROZEN_PACKAGE_SHA256
    projected = tmp_path / "work"
    for key, source in configmap["data"].items():
        if "__SLASH__" not in key:
            continue
        target = projected / key.replace("__SLASH__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from evals.fleet import "
                "glm53_dedicated_v34_controller_v1 as controller; "
                "from evals.fleet import glm53_dedicated_v34_create_v1 as server; "
                "assert server.API_RUN_ID_RE.fullmatch('glm53-tp8-v34-e4888aa7'); "
                "assert not server.API_RUN_ID_RE.fullmatch('ft-run-e4888aa7'); "
                "assert controller.RESULT_SCHEMA.endswith('v34-controller-result-v1')"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(projected)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    command = package.build_job(launch.FROZEN_PACKAGE_COMMIT)["spec"]["template"]["spec"][
        "containers"
    ][0]["command"][-1]
    assert command.count("glm53_dedicated_v34_controller_v1") == 1
    assert "glm53_dedicated_v33_controller_v1" not in command


class FakeKubernetes:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.token_reads = 0
        self.dry_runs = 0
        self.creates = 0

    def load_fleet_token(self) -> str:
        self.token_reads += 1
        return "test-token"

    def get(self, kind: str, name: str) -> dict[str, Any] | None:
        return self.objects.get((kind, name))

    def dry_run(self, _objects: dict[str, Any]) -> None:
        self.dry_runs += 1

    def create(self, _objects: dict[str, Any]) -> None:
        self.creates += 1
        for kind, name in launch._identities():  # noqa: SLF001
            self.objects[(kind, name)] = {
                "apiVersion": "v1" if kind == "configmap" else "batch/v1",
                "kind": "ConfigMap" if kind == "configmap" else "Job",
                "metadata": {
                    "name": name,
                    "namespace": launch.engine.NAMESPACE,
                    "uid": str(uuid.uuid4()),
                },
            }


def test_v34_launcher_creates_only_fresh_controller_objects_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reconciliation = {"receipt_sha256": "sha256:stale"}
    authorization = {"receipt_sha256": "sha256:authorization"}

    class StaleBackend:
        def list_runs(self) -> tuple[list[dict[str, Any]], int]:
            return [], 1

    monkeypatch.setattr(
        launch.engine.stale,
        "build_reconciliation",
        lambda **_kwargs: reconciliation,
    )
    monkeypatch.setattr(
        launch.engine.stale, "validate_reconciliation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(launch.live, "build_live_authorization", lambda **_kwargs: authorization)
    monkeypatch.setattr(server, "validate_authorization", lambda _value: None)
    monkeypatch.setattr(server, "payload", lambda: {"priority_class": "fleet-infra-quiet"})
    monkeypatch.setattr(server, "request_sha256", lambda: "sha256:request")
    monkeypatch.setattr(
        package,
        "render",
        lambda *_args: {
            "objects": {
                "apiVersion": "v1",
                "kind": "List",
                "items": [
                    {
                        "data": {
                            "package.json": json.dumps(
                                {"package_sha256": launch.FROZEN_PACKAGE_SHA256}
                            )
                        }
                    },
                    {},
                    {},
                ],
            }
        },
    )
    kubernetes = FakeKubernetes()
    prior = os.environ.pop("FLEET_API_KEY", None)
    try:
        result = launch.execute(
            root=tmp_path,
            output=tmp_path / "SUBMITTED.json",
            kubernetes=kubernetes,
            stale_backend_factory=StaleBackend,
            live_backend_factory=lambda: object(),
            now=lambda: 123.0,
        )
    finally:
        if prior is not None:
            os.environ["FLEET_API_KEY"] = prior
    assert result["status"] == "CONTROLLER_SUBMITTED_CREATE_ONCE"
    assert result["schema_version"] == launch.RESULT_SCHEMA
    assert result["controller_job_name"] == package.JOB_NAME
    assert kubernetes.token_reads == 1
    assert kubernetes.dry_runs == 1
    assert kubernetes.creates == 1
    assert os.environ.get("FLEET_API_KEY") == prior
