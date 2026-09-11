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
    monkeypatch.delenv("FLEET_TRAINING_API_TOKEN", raising=False)
    result = RUNNER.invoke(cli.app, ["status", "synthetic"])
    assert result.exit_code == 2 and "token is required" in result.stderr

    def broken():
        raise RuntimeError("private SDK response")

    monkeypatch.setattr(cli, "_client", broken)
    result = RUNNER.invoke(cli.app, ["status", "synthetic"])
    assert result.exit_code == 2 and "private SDK response" not in result.output


def test_doctor_checks_installation_without_claiming_cluster_readiness(monkeypatch):
    result = RUNNER.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["cluster_checked"] is False
    assert json.loads(result.stdout)["model_qualified"] is False
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: None)
    assert RUNNER.invoke(cli.app, ["doctor"]).exit_code == 2


@pytest.mark.parametrize(
    "command", [[], ["data"], ["train"], ["preflight"], ["preview"], ["submit"], ["status"]]
)
def test_help_is_accessible(command):
    result = RUNNER.invoke(cli.app, [*command, "--help"])
    assert result.exit_code == 0
