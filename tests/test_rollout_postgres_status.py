from evals.fleet import rollout_postgres, rollout_postgres_status


def test_status_redacts_connection_error(monkeypatch, capsys):
    monkeypatch.setenv("SYNTHETIC_DATABASE", "postgresql://synthetic-secret@localhost/test")

    def fail(dsn):
        raise RuntimeError(dsn)

    monkeypatch.setattr(rollout_postgres, "summary", fail)
    assert rollout_postgres_status.main(["--dsn-env", "SYNTHETIC_DATABASE", "summary"]) == 1
    output = capsys.readouterr()
    assert "read_only_observation_failed" in output.out
    assert "synthetic-secret" not in output.out + output.err


def test_missing_status_credentials_are_sanitized(monkeypatch, capsys):
    monkeypatch.delenv("SYNTHETIC_DATABASE", raising=False)
    assert rollout_postgres_status.main(["--dsn-env", "SYNTHETIC_DATABASE", "summary"]) == 1
    assert "database_environment_missing" in capsys.readouterr().out


def test_retry_review_status_uses_only_grouped_observation(monkeypatch, capsys):
    monkeypatch.setenv("SYNTHETIC_DATABASE", "postgresql://synthetic-secret@localhost/test")
    observed = {
        "retry_review": 1,
        "with_local_result": 1,
        "groups": [
            {
                "failure_code": "authoritative_scoring_started.runtimeerror",
                "result_class": "infrastructure_invalid",
                "agent_termination": "output_limit",
                "agent_exit_code": 0,
                "session_ingest_status": "completed",
                "has_local_result": True,
                "has_session": True,
                "count": 1,
            }
        ],
    }
    monkeypatch.setattr(rollout_postgres_status, "retry_review_summary", lambda _dsn: observed)

    assert rollout_postgres_status.main(["--dsn-env", "SYNTHETIC_DATABASE", "retry-review"]) == 0
    output = capsys.readouterr().out
    assert '"view": "retry-review"' in output
    assert '"agent_termination": "output_limit"' in output
    assert "synthetic-secret" not in output
