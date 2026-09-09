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
