"""Exercise the public workflow without credentials or a paid submission."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import sft

RUNNER = CliRunner()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n")
    output = tmp_path / "prepared"
    plan = {"model": {"repo": "synthetic"}, "recipe": {"max_steps": 2}}
    request = {
        "name": "synthetic",
        "run_dir": "/mnt/sfs/jobs/synthetic",
        "image": "registry/image@sha256:" + "a" * 64,
        "command": "python run.py",
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "resources": {
            "cpu_request": "4",
            "cpu_limit": "8",
            "memory_request": "32Gi",
            "memory_limit": "48Gi",
        },
    }
    monkeypatch.setattr(sft, "compile_sft", lambda value, relative_to: plan)
    monkeypatch.setattr(sft, "job_request", lambda value: request)
    assert RUNNER.invoke(cli.app, ["train", str(config), "--output", str(output)]).exit_code == 0
    return output, plan, request, config


def record_preflight(output, plan, request):
    proof = {
        "schema": "cyber_sft_cpu_preflight_v1",
        "gpus": 0,
        "status": "passed",
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    cli._write(output / "PREFLIGHT.json", {**proof, "sha256": digest(proof)})


def test_prepare_is_create_once_and_never_networks(prepared, monkeypatch):
    output, plan, request, config = prepared
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("offline prepare used network"))
    assert cli._prepared(output) == (plan, request)
    assert (output / "plan.json").stat().st_mode & 0o777 == 0o600
    second = RUNNER.invoke(cli.app, ["train", str(config), "--output", str(output)])
    assert second.exit_code == 2
    assert cli._prepared(output) == (plan, request)


@pytest.mark.parametrize("name", ["plan.json", "request.json", "PREPARED.json"])
def test_mutated_prepared_files_are_rejected(prepared, monkeypatch, name):
    output, *_ = prepared
    (output / name).write_text("{}")
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("invalid preparation reached network"))
    for action in ("preview", "preflight", "submit"):
        assert RUNNER.invoke(cli.app, [action, str(output)]).exit_code == 2


def test_preflight_records_actual_checker_result_once(prepared, monkeypatch):
    output, plan, request, _ = prepared
    calls = []

    def check(value):
        calls.append(value)
        return {
            "schema": "cyber_sft_cpu_preflight_v1",
            "request_sha256": digest(request),
            "plan_sha256": digest(plan),
            "gpus": 0,
            "status": "passed",
        }

    monkeypatch.setattr(sft, "preflight", check)
    assert RUNNER.invoke(cli.app, ["preflight", str(output)]).exit_code == 0
    assert RUNNER.invoke(cli.app, ["preflight", str(output)]).exit_code == 2
    assert calls == [plan]


def test_submit_requires_bound_cpu_proof(prepared, monkeypatch):
    output, plan, request, _ = prepared
    monkeypatch.setattr(
        cli, "_client", lambda: pytest.fail("unqualified submission reached network")
    )
    assert RUNNER.invoke(cli.app, ["submit", str(output)]).exit_code == 2
    record_preflight(output, plan, request)
    value = cli._read(output / "PREFLIGHT.json")
    value["plan_sha256"] = "a" * 64
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    (output / "PREFLIGHT.json").write_text(json.dumps(value))
    assert RUNNER.invoke(cli.app, ["submit", str(output)]).exit_code == 2


def test_submit_uses_shared_boundary_and_journal(prepared, monkeypatch):
    output, plan, request, _ = prepared
    record_preflight(output, plan, request)
    calls = []

    def submit(config, journal):
        calls.append((config, journal))
        return {"name": "synthetic-12345678", "status": "pending"}

    monkeypatch.setattr(cli, "_client", lambda: nullcontext(SimpleNamespace(submit_once=submit)))
    result = RUNNER.invoke(cli.app, ["submit", str(output)])
    assert result.exit_code == 0
    assert calls == [(request, output / "SUBMISSION.jsonl")]
    assert json.loads(result.stdout)["status"] == "pending"


def test_preview_and_status_are_read_only(prepared, monkeypatch):
    output, _, request, _ = prepared
    fake = SimpleNamespace(
        preview=lambda config: {"synthetic": config},
        status=lambda name: {"name": name, "status": "RUNNING"},
    )
    monkeypatch.setattr(cli, "_client", lambda: nullcontext(fake))
    monkeypatch.setattr(
        cli, "validate_preview", lambda config, result: {"nodes": config["workers"]}
    )
    result = RUNNER.invoke(cli.app, ["preview", str(output)])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"submitted": False, "nodes": request["workers"]}
    assert not (output / "SUBMISSION.jsonl").exists()
    result = RUNNER.invoke(cli.app, ["status", "synthetic-12345678"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "RUNNING"


def test_missing_auth_and_sdk_errors_never_print_sensitive_contents(monkeypatch):
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    result = RUNNER.invoke(cli.app, ["status", "synthetic"])
    assert result.exit_code == 2 and "token is required" in result.stderr

    def broken():
        raise RuntimeError("private SDK response")

    monkeypatch.setattr(cli, "_client", broken)
    result = RUNNER.invoke(cli.app, ["status", "synthetic"])
    assert result.exit_code == 2 and "private SDK response" not in result.output


def test_jobs_use_standard_fleet_identity_not_a_second_token(monkeypatch):
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-operator-key")
    monkeypatch.setattr(cli, "Jobs", lambda token: {"received": token})
    assert cli._client() == {"received": "synthetic-operator-key"}


def test_conversion_uses_same_prepare_preflight_submit_rail(prepared, monkeypatch, tmp_path):
    from training import miles_conversion

    _, _, request, config = prepared
    output = tmp_path / "conversion"
    plan = {"schema": miles_conversion.SCHEMA, "optimizer_steps": 0}
    monkeypatch.setattr(miles_conversion, "compile_conversion", lambda *a, **k: plan)
    monkeypatch.setattr(miles_conversion, "job_request", lambda _: request)
    result = RUNNER.invoke(cli.app, ["miles-convert", str(config), "--output", str(output)])
    assert result.exit_code == 0 and json.loads(result.stdout)["optimizer_steps"] == 0
    assert cli._prepared(output) == (plan, request)
    assert (
        RUNNER.invoke(cli.app, ["miles-convert", str(config), "--output", str(output)]).exit_code
        == 2
    )
    proof = {
        "schema": "cyber_miles_conversion_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    monkeypatch.setattr(miles_conversion, "preflight", lambda _: proof)
    assert RUNNER.invoke(cli.app, ["preflight", str(output)]).exit_code == 0
    calls = []
    monkeypatch.setattr(
        cli,
        "_client",
        lambda: nullcontext(
            SimpleNamespace(
                submit_once=lambda *args: calls.append(args) or {"name": "synthetic-12345678"},
            )
        ),
    )
    assert RUNNER.invoke(cli.app, ["submit", str(output)]).exit_code == 0
    assert calls == [(request, output / "SUBMISSION.jsonl")]
    monkeypatch.setattr(
        miles_conversion, "seal", lambda *a: {"sha256": "a" * 64, "files": [{}, {}]}
    )
    result = RUNNER.invoke(
        cli.app, ["miles-seal", str(output), "--output", str(tmp_path / "seal.json")]
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["files"] == 2

    def fail(*args):
        raise ValueError("private error")

    monkeypatch.setattr(miles_conversion, "seal", fail)
    result = RUNNER.invoke(
        cli.app, ["miles-seal", str(output), "--output", str(tmp_path / "seal.json")]
    )
    assert result.exit_code == 2 and "private error" not in result.output


@pytest.mark.parametrize("auth", ["synthetic-key", ""])
def test_rl_data_command_does_not_submit_or_expose_private_content(tmp_path, monkeypatch, auth):
    from training import rl_data

    calls = []

    def build(config, *, relative_to, client):
        calls.append(config)
        assert relative_to == tmp_path and not client.follow_redirects
        assert client.headers["Authorization"] == "Bearer synthetic-key"
        return {"submitted": False, "files": {"train": {"rows": 2}}}

    monkeypatch.setattr(rl_data, "build", build)
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("data preparation used Jobs API"))
    monkeypatch.setenv("FLEET_API_KEY", auth)
    config = tmp_path / "data.yaml"
    config.write_text("name: synthetic-rl\n")
    result = RUNNER.invoke(cli.app, ["rl-data", str(config)])
    assert result.exit_code == (0 if auth else 2)
    assert "synthetic-key" not in result.output
    assert len(calls) == (1 if auth else 0)


def test_doctor_checks_installation_without_claiming_cluster_readiness(monkeypatch):
    result = RUNNER.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["cluster_checked"] is False
    assert json.loads(result.stdout)["model_qualified"] is False
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: None)
    assert RUNNER.invoke(cli.app, ["doctor"]).exit_code == 2


def test_model_lock_is_create_once_and_not_qualification(tmp_path, monkeypatch):
    from training import models

    calls = []
    lock, weights = {"weights": {"shards": 2}}, {"files": []}

    def freeze(repo, revision, client):
        calls.append((repo, revision))
        return lock, weights

    monkeypatch.setattr(models, "freeze", freeze)
    output = tmp_path / "model"
    command = ["model-lock", "Example/Model", "a" * 40, "--output", str(output)]
    result = RUNNER.invoke(cli.app, command)
    assert result.exit_code == 0
    assert json.loads(result.stdout)["model_qualified"] is False
    assert cli._read(output / "model.lock.json") == lock
    assert cli._read(output / "model.weights.json") == weights
    assert cli._read(output / "COMPLETE.json") == {
        "lock_sha256": digest(lock),
        "weights_sha256": digest(weights),
    }
    assert RUNNER.invoke(cli.app, command).exit_code == 2
    assert calls == [("Example/Model", "a" * 40)]


def test_checkpoint_command_uses_original_bound_plan(prepared, monkeypatch, tmp_path):
    from training import checkpoints

    directory, plan, *_ = prepared
    output = tmp_path / "checkpoint.json"
    calls = []

    def seal(value, step, path):
        calls.append((value, step, path))
        return {"optimizer_step": step, "total_bytes": 17, "receipt_sha256": "a" * 64}

    monkeypatch.setattr(checkpoints, "seal", seal)
    command = ["checkpoint-seal", str(directory), "1", "--output", str(output)]
    result = RUNNER.invoke(cli.app, command)
    assert result.exit_code == 0
    assert calls == [(plan, 1, output)]
    assert json.loads(result.stdout)["total_bytes"] == 17
    (directory / "plan.json").write_text("{}")
    assert RUNNER.invoke(cli.app, command).exit_code == 2
    assert len(calls) == 1


@pytest.mark.parametrize("command", ["jobs-run", "jobs-status", "rl-snapshot", "rl-config"])
def test_retired_submission_commands_are_not_exposed(command):
    from training.cli import parser

    with pytest.raises(SystemExit) as exc:
        parser().parse_args([command])
    assert exc.value.code == 2


def test_export_command_requires_explicit_manifest_identity(tmp_path, monkeypatch):
    from training import export as module

    manifest, out, sha = tmp_path / "sealed.json", tmp_path / "export", "a" * 64
    calls = []

    def export(path, expected, destination):
        calls.append((path, expected, destination))
        return {
            "output_root": str(destination),
            "optimizer_step": 2,
            "tensor_bytes": 16,
            "receipt_sha256": "b" * 64,
            "gpu_reload_verified": False,
        }

    monkeypatch.setattr(module, "export", export)
    base = ["checkpoint-export", str(manifest), "--output", str(out)]
    assert RUNNER.invoke(cli.app, base).exit_code == 2
    assert not calls
    result = RUNNER.invoke(cli.app, [*base, "--sha256", sha])
    assert result.exit_code == 0
    assert calls == [(manifest, sha, out)]
    assert json.loads(result.stdout)["gpu_reload_verified"] is False

    def failed(*args):
        raise RuntimeError("private trainer internals")

    monkeypatch.setattr(module, "export", failed)
    result = RUNNER.invoke(cli.app, [*base, "--sha256", sha])
    assert result.exit_code == 2 and "private trainer internals" not in result.output


def test_checkpoint_check_explicit_gpu_and_redacted_failure(tmp_path, monkeypatch):
    from training import export_check

    calls = []

    def check(*args, **kwargs):
        calls.append((args, kwargs))
        return {"gpu_reload_verified": kwargs["gpu"]}

    monkeypatch.setattr(export_check, "check", check)
    args = [
        "checkpoint-check",
        str(tmp_path / "EXPORT.json"),
        "--sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "CHECK.json"),
    ]
    assert RUNNER.invoke(cli.app, args).exit_code == 0 and calls[-1][1] == {"gpu": False}
    result = RUNNER.invoke(cli.app, [*args, "--gpu"])
    assert result.exit_code == 0 and json.loads(result.stdout)["gpu_reload_verified"]

    def fail(*args, **kwargs):
        raise RuntimeError("private details")

    monkeypatch.setattr(export_check, "check", fail)
    result = RUNNER.invoke(cli.app, args)
    assert result.exit_code == 2 and "private details" not in result.output


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["data"],
        ["train"],
        ["preflight"],
        ["preview"],
        ["submit"],
        ["status"],
        ["checkpoint-seal"],
        ["checkpoint-export"],
        ["checkpoint-check"],
    ],
)
def test_help_is_accessible(command):
    result = RUNNER.invoke(cli.app, [*command, "--help"])
    assert result.exit_code == 0
