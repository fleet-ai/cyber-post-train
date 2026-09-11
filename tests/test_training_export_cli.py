"""Read-only export boundaries; synthetic HTTP and private temporary files only."""

import datetime as dt
import json
from types import SimpleNamespace

import httpx
import pytest

from training import cli, fleet, secrets
from training.io import file_sha256


@pytest.fixture
def http(monkeypatch):
    calls, replies, sleeps = [], [], []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(fleet.httpx, "get", get)
    monkeypatch.setattr(fleet.time, "sleep", sleeps.append)
    return calls, replies, sleeps


def response(value, status=200):
    return httpx.Response(status, json=value)


@pytest.mark.parametrize(
    "base",
    [
        "http://example.test",
        "https:///",
        "https://u@example.test",
        "https://:p@example.test",
        "https://example.test?token=private",
        "https://example.test#private",
    ],
)
def test_export_rejects_unsafe_base(base, http):
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        fleet.FleetClient("synthetic-control-secret", base_url=base)
    assert not http[0]


def test_missing_key_and_encoded_read_routes(http):
    calls, replies, _ = http
    with pytest.raises(ValueError, match="required"):
        fleet.FleetClient("")
    replies.extend([response({}), response({}), response({}), response({})])
    client = fleet.FleetClient("synthetic-control-secret", base_url="https://example.test/")
    assert client.job("a/b") == {}
    assert client.job_sessions("a/b") == {}
    assert client.transcript("a/b") == {}
    client._get("/v1/jobs", {"limit": 3, "absent": None})
    assert [url for url, _ in calls] == [
        "https://example.test/v1/jobs/a%2Fb",
        "https://example.test/v1/sessions/job/a%2Fb",
        "https://example.test/v1/sessions/a%2Fb/transcript",
        "https://example.test/v1/jobs",
    ]
    for _, kwargs in calls:
        assert kwargs["headers"]["Authorization"] == "Bearer synthetic-control-secret"
        assert kwargs["follow_redirects"] is False
        assert kwargs["timeout"] == 120
    assert calls[-1][1]["params"] == {"limit": 3}


@pytest.mark.parametrize("status", [301, 302, 307, 308, 400, 401, 403, 404, 422])
def test_redirects_and_permanent_errors_never_retry_or_echo_body(status, http):
    calls, replies, sleeps = http
    replies.append(response({"detail": "PRIVATE_BODY_AND_TOKEN"}, status))
    with pytest.raises(fleet.FleetExportError) as error:
        fleet.FleetClient("synthetic-control-secret").transcript("session")
    assert str(error.value) == f"export GET failed (HTTP {status})"
    assert len(calls) == 1 and not sleeps


@pytest.mark.parametrize("failure", [429, 500, 502, 503, 504, httpx.ConnectError("PRIVATE_URL")])
def test_read_retries_are_bounded_and_sanitized(failure, http):
    calls, replies, sleeps = http
    reply = response({"detail": "PRIVATE_BODY"}, failure) if isinstance(failure, int) else failure
    replies.extend([reply] * 5)
    with pytest.raises(fleet.FleetExportError, match="five read-only attempts"):
        fleet.FleetClient("synthetic-control-secret").job("job")
    assert len(calls) == 5 and len(sleeps) == 4
    assert all(0.5 <= delay < 8.2 for delay in sleeps)


def test_transient_error_can_recover_and_json_shape_is_checked(http):
    calls, replies, sleeps = http
    replies.extend([response({}, 503), response({"ok": True}), response([])])
    client = fleet.FleetClient("synthetic-control-secret")
    assert client.job("job") == {"ok": True}
    with pytest.raises(fleet.FleetExportError, match="non-object JSON"):
        client.job("job")
    assert len(calls) == 3 and len(sleeps) == 1


def test_paging_filters_old_jobs_and_terminates_repeated_page(http):
    calls, replies, _ = http
    now = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    page = [
        {"id": str(i), "status": "completed", "created_at": now.isoformat()} for i in range(1000)
    ]
    replies.extend([response({"jobs": page}), response({"jobs": page})])
    client = fleet.FleetClient("synthetic-control-secret")
    assert len(list(client.completed_jobs(created_after=now))) == 1000
    assert (
        calls[1][1]["params"]["created_before"] == (now - dt.timedelta(microseconds=1)).isoformat()
    )
    replies.append(response({"jobs": [{"id": "old", "created_at": "2026-01-01T00:00:00Z"}]}))
    assert list(client.completed_jobs(created_after=now)) == []


@pytest.mark.parametrize(
    "value", [{}, {"jobs": None}, {"jobs": [None]}, {"jobs": [{"id": "x", "status": "running"}]}]
)
def test_empty_or_noncompleted_discovery(value, http):
    http[1].append(response(value))
    assert (
        list(
            fleet.FleetClient("synthetic-control-secret").completed_jobs(
                created_after=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
            )
        )
        == []
    )


def test_invalid_roster_and_discovery_shape(http):
    http[1].append(response({"jobs": {"not": "a list"}}))
    with pytest.raises(fleet.FleetExportError, match="jobs"):
        list(
            fleet.FleetClient("synthetic-control-secret").completed_jobs(
                created_after=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
            )
        )
    roster = {"tasks": [None, {"task_key": "fallback", "sessions": [None, {}, {"id": "s"}]}]}
    refs = fleet.roster_session_refs("j", None, roster)
    assert len(refs) == 1 and refs[0].task_key == "fallback" and refs[0].session_id == "s"
    assert refs[0].task_binding == {}


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("invalid", None),
        ("2026-01-01", dt.datetime(2026, 1, 1, tzinfo=dt.UTC)),
        ("2026-01-01T00:00:00Z", dt.datetime(2026, 1, 1, tzinfo=dt.UTC)),
    ],
)
def test_parse_time(value, expected):
    assert fleet._parse_time(value) == expected


def roster(task="synthetic__blackbox_ctf_v1"):
    return {
        "tasks": [
            {
                "task": {"key": task},
                "sessions": [
                    {"id": "done", "status": "completed"},
                    {"id": "running", "status": "running"},
                ],
            }
        ]
    }


def test_export_cli_filters_and_writes_private_digest_manifest(tmp_path, http, monkeypatch, capsys):
    calls, replies, _ = http
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-control-secret")
    out = tmp_path / "sessions.jsonl"
    replies.extend([response(roster()), response({"private_transcript": "SYNTHETIC_TASK_TEXT"})])
    args = ["export", "--job-id", "j", "--output", str(out)]
    assert cli.main(args) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text())
    assert len(rows) == manifest["sessions"] == 1
    assert rows[0]["source"]["session_id"] == "done"
    assert manifest["sha256"] == file_sha256(out)
    assert out.stat().st_mode & 0o777 == 0o600
    assert out.with_suffix(".jsonl.manifest.json").stat().st_mode & 0o777 == 0o600
    assert len(calls) == 2
    output = capsys.readouterr()
    assert not output.err and "SYNTHETIC_TASK_TEXT" not in output.out
    replies.append(response(roster()))
    assert cli.main([*args, "--dry-run", "--include-incomplete"]) == 0
    assert json.loads(capsys.readouterr().out)["eligible_sessions"] == 2
    assert len(calls) == 3


def test_explicit_ids_and_score_blind_discovery(tmp_path, monkeypatch):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\na\n\nb\n")
    args = cli.parser().parse_args(
        [
            "export",
            "--job-id",
            "a",
            "--job-ids-file",
            str(ids_file),
            "--discover-completed-cyber",
            "--created-after",
            "2026-01-01",
            "--output",
            "unused",
        ]
    )
    observed = []

    def completed_jobs(*, created_after):
        observed.append(created_after)
        return [{"id": "b"}, {"id": "c"}, {"id": "other"}]

    client = SimpleNamespace(
        completed_jobs=completed_jobs,
        job_sessions=lambda job: roster("unrelated" if job == "other" else "x__blackbox_ctf_v1"),
    )
    assert cli._job_ids(args, client) == ["a", "b", "c"]
    assert observed == [dt.datetime(2026, 1, 1, tzinfo=dt.UTC)]
    args.created_after = "2026-01-01T00:00:00Z"
    assert cli._job_ids(args, client) == ["a", "b", "c"]
    args.created_after = None
    with pytest.raises(ValueError, match="requires"):
        cli._job_ids(args, client)


@pytest.mark.parametrize("workers", [0, 65])
def test_export_worker_bounds(workers):
    with pytest.raises(ValueError, match="1..64"):
        list(fleet.export_sessions(None, [], workers=workers))


def test_cli_errors_hide_raw_body_record_and_filesystem_details(
    tmp_path, monkeypatch, http, capsys
):
    args = ["export", "--output", str(tmp_path / "out")]
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    assert cli.main(args) == 2
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-control-secret")
    assert cli.main(args) == 2
    http[1].append(response({"private": "HIDDEN_RESPONSE_BODY"}, 403))
    assert cli.main([*args, "--job-id", "j"]) == 2
    for error in (ValueError("HIDDEN_RECORD"), OSError("HIDDEN_FILE_SECRET")):

        def fail(*args, error=error, **kwargs):
            raise error

        monkeypatch.setattr(cli, "build_datasets", fail)
        assert cli.main(["normalize", "--input", "unused", "--output-dir", "unused"]) == 2
    out = capsys.readouterr()
    assert not out.out
    assert "HTTP 403" in out.err and "ValueError" in out.err and "OSError" in out.err
    assert "HIDDEN" not in out.err and "synthetic-control-secret" not in out.err


def test_normalization_receives_all_control_plane_secrets(monkeypatch, capsys):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-session-token")
    seen = []
    monkeypatch.setattr(
        cli,
        "build_datasets",
        lambda src, dest, *, secrets: (
            seen.append((str(src), str(dest), secrets)) or {"counts": {"records": 2}}
        ),
    )
    assert cli.main(["normalize", "--input", "in", "--output-dir", "out"]) == 0
    assert seen[0][:2] == ("in", "out")
    assert "synthetic-session-token" in seen[0][2]
    assert json.loads(capsys.readouterr().out) == {"records": 2}
    assert secrets.redact_exact({"nested": ["synthetic-session-token"]}, seen[0][2]) == {
        "nested": [secrets.REDACTED]
    }
    assert secrets.redact_exact(7, [""]) == 7
    assert secrets.contains_secret({"nested": ["synthetic-session-token"]}, seen[0][2])
    assert not secrets.contains_secret("unrelated", ["", "synthetic-session-token"])
