import copy
import json
import time
import urllib.error
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_release
from evals.fleet import glm53_dedicated_v25_create_v1 as create

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, value: object, status: int = 202) -> None:
        self.status = status
        self.raw = json.dumps(value).encode()

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.raw


def authorization(now: float) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": create.SCHEMA,
        "status": "PASSED_LIVE_CREATE_GATES",
        "observed_at_epoch": now,
        "server_title": create.TITLE,
        "server_run_dir": create.RUN_DIR,
        "request_sha256": create.request_sha256(),
        "preview_http_status": 200,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_or_remnant_matches": 0,
        "sfs_run_dir_absent": True,
        "control_result_absent": True,
        "active_dedicated_nodes": 1,
        "active_dedicated_gpus": 1,
        "planned_nodes_after_create": 2,
        "planned_gpus_after_create": 9,
        "priority_class": v24.PRIORITY_CLASS,
        "preemption_policy": v24.PREEMPTION_POLICY,
        "server_launch_authorized": True,
        "watchdog_handoff_required_immediately": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def rehash(value: dict[str, object]) -> None:
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")


def test_payload_has_fresh_identity_and_exact_corrected_shell() -> None:
    value = create.payload()
    create.validate_payload(value)
    assert value["title"] == create.TITLE
    assert value["run_dir"] == create.RUN_DIR
    assert value["env"]["GLM53_RUN_DIR"] == create.RUN_DIR
    assert v24.TITLE not in crypto.canonical_json(value).decode()
    assert v24.RUN_DIR not in crypto.canonical_json(value).decode()
    assert value["command"].startswith("bash -lc ")


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("status", "HELD"),
        ("preview_http_status", 503),
        ("jobs_api_title_matches", 1),
        ("jobs_api_run_dir_matches", 1),
        ("kubernetes_identity_or_remnant_matches", 1),
        ("sfs_run_dir_absent", False),
        ("control_result_absent", False),
        ("planned_nodes_after_create", 3),
        ("planned_gpus_after_create", 17),
        ("priority_class", "fleet-train-high"),
        ("preemption_policy", "PreemptLowerPriority"),
        ("watchdog_handoff_required_immediately", False),
        ("qualification_launch_authorized", True),
        ("scored_launch_authorized", True),
        ("protected_content_included", True),
    ],
)
def test_authorization_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object
) -> None:
    now = 2_000_000_000.0
    monkeypatch.setattr(create.time, "time", lambda: now)
    value = authorization(now)
    value[field] = bad
    rehash(value)
    with pytest.raises(create.CreateError, match="authorization_invalid"):
        create.validate_authorization(value)


def test_authorization_staleness_and_extra_fields_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 2_000_000_000.0
    monkeypatch.setattr(create.time, "time", lambda: now)
    stale = authorization(now - 61)
    with pytest.raises(create.CreateError, match="authorization_invalid"):
        create.validate_authorization(stale)
    extra = authorization(now)
    extra["unexpected"] = False
    rehash(extra)
    with pytest.raises(create.CreateError, match="authorization_invalid"):
        create.validate_authorization(extra)


def test_create_consumes_name_identity_parser_and_emits_sanitized_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    now = 2_000_000_000.0
    monkeypatch.setattr(create.time, "time", lambda: now)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    seen: dict[str, object] = {}

    def opener(request: object, timeout: int) -> Response:
        seen["request"] = request
        seen["timeout"] = timeout
        return Response({"name": "ft-run-0123abcd", "status": "pending"})

    result_path = tmp_path / "CREATED.json"
    result = create.create_once(authorization(now), result_path=result_path, opener=opener)
    assert result["api_run_id"] == "ft-run-0123abcd"
    assert result["http_status"] == 202
    assert result["watchdog_handoff_complete"] is False
    assert result["qualification_launch_authorized"] is False
    assert result["scored_launch_authorized"] is False
    assert result["receipt_sha256"] == crypto.digest_without(result, "receipt_sha256")
    assert json.loads(result_path.read_text()) == result
    request = seen["request"]
    assert request.full_url == create.API_URL
    assert request.method == "POST"
    assert json.loads(request.data) == create.payload()
    assert seen["timeout"] == 30


def test_create_calls_the_reviewed_exact_response_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    now = time.time()
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    called: dict[str, object] = {}

    def parser(value: dict[str, object]) -> str:
        called["value"] = value
        return "ft-run-deadbeef"

    monkeypatch.setattr(live_release, "extract_jobs_api_run_id", parser)
    result = create.create_once(
        authorization(now),
        result_path=tmp_path / "CREATED.json",
        opener=lambda *_args, **_kwargs: Response({"name": "ft-run-deadbeef"}),
    )
    assert called["value"] == {"name": "ft-run-deadbeef"}
    assert result["api_run_id"] == "ft-run-deadbeef"


def test_unknown_create_response_requires_reconciliation_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    now = time.time()
    monkeypatch.setenv("FLEET_API_KEY", "test-only")

    def opener(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("connection lost after POST")

    with pytest.raises(create.CreateError, match="reconcile_do_not_retry"):
        create.create_once(
            authorization(now), result_path=tmp_path / "CREATED.json", opener=opener
        )


@pytest.mark.parametrize("status", [200, 201, 204])
def test_only_deployed_202_create_success_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    now = time.time()
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    with pytest.raises(create.CreateError, match="http_status_invalid_reconcile_do_not_retry"):
        create.create_once(
            authorization(now),
            result_path=tmp_path / f"CREATED-{status}.json",
            opener=lambda *_args, **_kwargs: Response(
                {"name": "ft-run-deadbeef"}, status=status
            ),
        )


def test_missing_or_ambiguous_identity_never_authorizes_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    now = time.time()
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    for response in (
        {"name": create.TITLE},
        {"name": "ft-run-0123abcd", "run_id": "ft-run-deadbeef"},
    ):
        with pytest.raises(live_release.LiveReleaseError):
            create.create_once(
                authorization(now),
                result_path=tmp_path / (response["name"] + ".json"),
                opener=lambda *_args, value=response, **_kwargs: Response(value),
            )


def test_existing_result_blocks_post_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = time.time()
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    result_path = tmp_path / "CREATED.json"
    result_path.write_text("already attempted\n")
    called = False

    def opener(*_args: object, **_kwargs: object) -> Response:
        nonlocal called
        called = True
        return Response({"name": "ft-run-deadbeef"})

    with pytest.raises(create.CreateError, match="already_exists"):
        create.create_once(authorization(now), result_path=result_path, opener=opener)
    assert called is False


def test_cli_is_fixed_to_v25_sfs_paths() -> None:
    assert create.CONTROL_DIR != create.RUN_DIR
    assert Path(create.AUTHORIZATION_PATH).parent != Path(create.RUN_DIR)
    assert not create.AUTHORIZATION_PATH.startswith(create.RUN_DIR + "/")
    assert not create.RESULT_PATH.startswith(create.RUN_DIR + "/")
    with pytest.raises(create.CreateError, match="cli_sfs_identity_invalid"):
        create.main(["--authorization", "/tmp/auth.json"])
    with pytest.raises(create.CreateError, match="cli_sfs_identity_invalid"):
        create.main(["--result-path", "/tmp/CREATED.json"])


def test_cli_requires_exact_authorization_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "is_file", lambda _self: False)
    with pytest.raises(create.CreateError, match="cli_authorization_absent"):
        create.main([])


def test_materializing_control_authorization_does_not_create_target_run_dir(
    tmp_path: Path,
) -> None:
    target = tmp_path / Path(create.RUN_DIR).name
    control = tmp_path / Path(create.CONTROL_DIR).name
    control.mkdir()
    (control / Path(create.AUTHORIZATION_PATH).name).write_text("{}\n")
    assert control.is_dir()
    assert not target.exists()


def test_held_receipt_is_digest_valid_and_non_authorizing() -> None:
    value = create.build_held()
    assert value["create_response_name_field_supported"] is True
    assert value["expected_create_http_status"] == 202
    assert value["unknown_response_requires_reconciliation_without_retry"] is True
    assert value["cli_authorization_path"] == create.AUTHORIZATION_PATH
    assert value["cli_result_path"] == create.RESULT_PATH
    assert value["control_result_absence_required"] is True
    assert value["fresh_live_authorization_required"] is True
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
    tracked = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v25-create-wrapper-held-v1.json"
        ).read_text()
    )
    assert tracked == value


def test_payload_mutation_fails_closed() -> None:
    value = copy.deepcopy(create.payload())
    value["title"] += "-other"
    with pytest.raises(create.CreateError, match="payload_identity_invalid"):
        create.validate_payload(value)
