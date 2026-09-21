"""Exercise the public workflow without credentials or a paid submission."""

import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import sft

RUNNER = CliRunner()


def test_data_rechunk_dispatches_cpu_only_builder(tmp_path, monkeypatch):
    from training import dense_rechunk

    config = tmp_path / "rechunk.json"
    config.write_text("{}")
    calls = []

    def build(value, *, relative_to):
        calls.append((value, relative_to))
        return {"submitted": False, "supervised_tokens": 20_000_000}

    monkeypatch.setattr(dense_rechunk, "build", build)
    result = RUNNER.invoke(cli.app, ["data-rechunk", str(config)])
    assert result.exit_code == 0
    assert calls == [({}, config.parent.resolve())]
    assert '"submitted": false' in result.stdout


def test_data_fleet_admit_dispatches_metadata_handoff_only(tmp_path, monkeypatch):
    from training import fleet_collection_admission

    config = tmp_path / "admission.json"
    config.write_text("{}")
    calls = []

    def build(value, *, relative_to):
        calls.append((value, relative_to))
        return {
            "submitted": False,
            "artifact_kind": "metadata_evidence_handoff_only",
            "trainable_corpus_created": False,
            "parquet_created": False,
            "source_text_read": False,
        }

    monkeypatch.setattr(fleet_collection_admission, "build", build)
    result = RUNNER.invoke(cli.app, ["data-fleet-admit", str(config)])
    assert result.exit_code == 0
    assert calls == [({}, config.parent.resolve())]
    assert '"artifact_kind": "metadata_evidence_handoff_only"' in result.stdout
    assert '"trainable_corpus_created": false' in result.stdout
    assert '"parquet_created": false' in result.stdout


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n")
    output = tmp_path / "prepared"
    plan = {"model": {"repo": "synthetic"}, "recipe": {"max_steps": 2}}
    request = {
        "name": "synthetic",
        "title": "Synthetic fixture",
        "run_dir": "/mnt/sfs/jobs/synthetic",
        "image": "registry/image@sha256:" + "a" * 64,
        "command": "python run.py",
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "resources": {
            "cpu_request": "4",
            "cpu_limit": "8",
            "memory_request": "32Gi",
            "memory_limit": "48Gi",
        },
    }
    monkeypatch.setattr(sft, "compile_sft", lambda value, relative_to: plan)
    monkeypatch.setattr(sft, "job_request", lambda value: request)
    # Most unit tests do not mount SFS. Dedicated tests below exercise the real
    # absence checker; workflow tests replace only that external mount boundary.
    monkeypatch.setattr(cli, "_require_output_absent", lambda _: None)
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
@pytest.mark.parametrize("contents", ["{}", "[]", "null"])
def test_mutated_prepared_files_are_rejected(prepared, monkeypatch, name, contents):
    output, *_ = prepared
    (output / name).write_text(contents)
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


def test_output_absence_check_fails_closed_without_mount_or_with_existing_output(
    tmp_path,
):
    mount = tmp_path / "jobs"
    request = {"run_dir": str(mount / "new-run")}
    with pytest.raises(ValueError, match="mount is unavailable"):
        cli._require_output_absent(request, jobs_root=mount)
    mount.mkdir()
    cli._require_output_absent(request, jobs_root=mount)
    (mount / "new-run").mkdir()
    with pytest.raises(ValueError, match="output already exists"):
        cli._require_output_absent(request, jobs_root=mount)


def test_preflight_and_submit_repeat_output_absence_check(prepared, monkeypatch):
    output, plan, request, _ = prepared
    checks = []
    monkeypatch.setattr(cli, "_require_output_absent", lambda value: checks.append(value))
    monkeypatch.setattr(
        sft,
        "preflight",
        lambda value: {
            "schema": "cyber_sft_cpu_preflight_v1",
            "request_sha256": digest(request),
            "plan_sha256": digest(plan),
            "gpus": 0,
            "status": "passed",
        },
    )
    assert RUNNER.invoke(cli.app, ["preflight", str(output)]).exit_code == 0
    monkeypatch.setattr(
        cli,
        "_client",
        lambda: nullcontext(
            SimpleNamespace(
                submit_once=lambda *args: {
                    "name": "synthetic-12345678",
                    "status": "queued",
                }
            )
        ),
    )
    assert RUNNER.invoke(cli.app, ["submit", str(output)]).exit_code == 0
    assert checks == [request, request]


@pytest.mark.parametrize("backend", ["miles", "skyrl"])
def test_rl_preflight_dispatch_does_not_use_sft_checker(prepared, monkeypatch, backend):
    import importlib

    module = importlib.import_module(f"training.{backend}_training")

    output, plan, request, _ = prepared
    plan["schema"] = f"cyber_{backend}_training_v1"
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    calls = []
    monkeypatch.setattr(sft, "preflight", lambda _: pytest.fail("wrong backend"))

    def check(value):
        calls.append(value)
        return {"status": "passed", "plan_sha256": digest(value), "gpus": 0}

    monkeypatch.setattr(module, "preflight", check)
    result = RUNNER.invoke(cli.app, ["preflight", str(output)])
    assert result.exit_code == 0 and calls == [plan]
    receipt = cli._read(output / "PREFLIGHT.json")
    assert receipt["sha256"] == digest({k: v for k, v in receipt.items() if k != "sha256"})


def test_module_entrypoint_exposes_public_help(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["cyber_post_train.cli", "--help"])
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as exc,
    ):
        runpy.run_module("cyber_post_train.cli", run_name="__main__")
    assert exc.value.code == 0
    assert "checkpoint-seal" in capsys.readouterr().out


def test_package_entrypoint_exposes_the_same_public_help(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["cyber_post_train", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("cyber_post_train", run_name="__main__")
    assert exc.value.code == 0
    assert "checkpoint-seal" in capsys.readouterr().out


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

    monkeypatch.setattr(
        cli, "_client", lambda _plan=None: nullcontext(SimpleNamespace(submit_once=submit))
    )
    result = RUNNER.invoke(cli.app, ["submit", str(output)])
    assert result.exit_code == 0
    assert calls == [(request, output / "SUBMISSION.jsonl")]
    assert json.loads(result.stdout)["status"] == "pending"


def test_direct_sft_submit_reuses_preflight_and_has_a_separate_journal(prepared, monkeypatch):
    from cyber_post_train import direct_submit

    output, plan, request, _ = prepared
    record_preflight(output, plan, request)
    plan["schema"] = "cyber_sft_runtime_dense_v1"
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_require_preflight", lambda *args: None)
    monkeypatch.setattr(cli, "_client", lambda: nullcontext("jobs-client"))
    calls = []

    class SyntheticKubectl:
        def __init__(self, context):
            self.context = context

    def submit(**kwargs):
        calls.append(kwargs)
        return {"name": "synthetic-12345678", "submitted": True}

    monkeypatch.setattr(direct_submit, "Kubectl", SyntheticKubectl)
    monkeypatch.setattr(direct_submit, "direct_submit_sft_once", submit)
    result = RUNNER.invoke(cli.app, ["direct-submit-sft", str(output), "--context", "prod-context"])
    assert result.exit_code == 0
    assert calls[0]["plan"] == plan and calls[0]["request"] == request
    assert calls[0]["jobs"] == "jobs-client"
    assert calls[0]["kubectl"].context == "prod-context"
    assert calls[0]["journal"] == output / "DIRECT_SUBMISSION.jsonl"
    assert calls[0]["output_absence_receipt"] is None


def test_sfs_output_receipt_is_create_once_and_bound_to_the_prepared_run(prepared, monkeypatch):
    output, plan, request, _ = prepared
    record_preflight(output, plan, request)
    receipt_path = output / "OUTPUT_ABSENT.json"
    proof = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 123,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "run_name": request["name"],
        "run_dir": request["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**proof, "sha256": digest(proof)}
    calls = []

    def build(plan_value, request_value):
        calls.append((plan_value, request_value))
        return receipt

    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_external_action_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_require_preflight", lambda *args: None)
    monkeypatch.setattr(cli, "build_output_absence_receipt", build)
    monkeypatch.setattr(
        cli, "_prepared", lambda _: ({**plan, "schema": "cyber_sft_runtime_dense_v1"}, request)
    )
    current_plan = {**plan, "schema": "cyber_sft_runtime_dense_v1"}
    receipt["plan_sha256"] = digest(current_plan)
    receipt["sha256"] = digest({key: value for key, value in receipt.items() if key != "sha256"})

    result = RUNNER.invoke(
        cli.app,
        ["sfs-output-receipt", str(output), "--output", str(receipt_path)],
    )
    assert result.exit_code == 0
    assert calls == [(current_plan, request)]
    assert cli._read(receipt_path) == receipt
    assert (
        RUNNER.invoke(
            cli.app,
            ["sfs-output-receipt", str(output), "--output", str(receipt_path)],
        ).exit_code
        == 2
    )


def test_sfs_output_job_commands_use_exact_prepared_binding(prepared, monkeypatch):
    from cyber_post_train import direct_submit

    output, plan, request, _ = prepared
    current_plan = {**plan, "schema": "cyber_sft_runtime_dense_v1"}
    record_preflight(output, current_plan, request)
    monkeypatch.setattr(cli, "_prepared", lambda _: (current_plan, request))
    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_external_action_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_require_preflight", lambda *args: None)

    class SyntheticKubectl:
        def __init__(self, context):
            self.context = context

    calls = []
    receipt = {"schema": "cyber_sft_output_absence_v1", "status": "passed"}

    def create(**kwargs):
        calls.append(("create", kwargs))
        return {"submitted": True, "name": "synthetic-sfs-a02", "gpus": 0}

    def collect(**kwargs):
        calls.append(("collect", kwargs))
        return receipt

    monkeypatch.setattr(direct_submit, "Kubectl", SyntheticKubectl)
    monkeypatch.setattr(direct_submit, "create_sfs_output_check_once", create)
    monkeypatch.setattr(direct_submit, "collect_sfs_output_check", collect)
    result = RUNNER.invoke(
        cli.app,
        ["sfs-output-job-create", str(output), "--context", "prod", "--attempt", "2"],
    )
    assert result.exit_code == 0
    assert calls[0][0] == "create"
    assert calls[0][1] == {
        "plan": current_plan,
        "request": request,
        "attempt": 2,
        "kubectl": calls[0][1]["kubectl"],
        "journal": output / "SFS_OUTPUT_CHECK_A02.jsonl",
    }
    assert calls[0][1]["kubectl"].context == "prod"

    receipt_path = output / "OUTPUT_ABSENT.json"
    result = RUNNER.invoke(
        cli.app,
        [
            "sfs-output-job-collect",
            str(output),
            "--context",
            "prod",
            "--attempt",
            "2",
            "--output",
            str(receipt_path),
        ],
    )
    assert result.exit_code == 0
    assert calls[1][0] == "collect"
    assert calls[1][1]["plan"] == current_plan
    assert calls[1][1]["request"] == request
    assert calls[1][1]["attempt"] == 2
    assert calls[1][1]["kubectl"].context == "prod"
    assert cli._read(receipt_path) == receipt


def test_sft_cpu_preflight_job_commands_use_clean_source_and_exact_prepared_binding(
    prepared, monkeypatch
):
    from cyber_post_train import direct_submit

    output, plan, request, _ = prepared
    current_plan = {**plan, "schema": "cyber_sft_runtime_dense_v1"}
    source_commit = "a" * 40
    monkeypatch.setattr(cli, "_prepared", lambda _: (current_plan, request))
    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_external_action_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_clean_source_commit", lambda: source_commit)

    class SyntheticKubectl:
        def __init__(self, context):
            self.context = context

    calls = []
    receipt = {
        "schema": "cyber_sft_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(current_plan),
        "request_sha256": digest(request),
    }
    receipt["sha256"] = digest(receipt)

    def create(**kwargs):
        calls.append(("create", kwargs))
        return {"submitted": True, "name": "synthetic-pre-a02", "gpus": 0}

    def collect(**kwargs):
        calls.append(("collect", kwargs))
        return receipt

    monkeypatch.setattr(direct_submit, "Kubectl", SyntheticKubectl)
    monkeypatch.setattr(direct_submit, "create_sft_cpu_preflight_once", create)
    monkeypatch.setattr(direct_submit, "collect_sft_cpu_preflight", collect)
    result = RUNNER.invoke(
        cli.app,
        ["sft-cpu-preflight-job-create", str(output), "--context", "prod", "--attempt", "2"],
    )
    assert result.exit_code == 0
    assert calls[0] == (
        "create",
        {
            "directory": output,
            "source_commit": source_commit,
            "attempt": 2,
            "kubectl": calls[0][1]["kubectl"],
            "journal": output / "SFT_CPU_PREFLIGHT_A02.jsonl",
        },
    )
    assert calls[0][1]["kubectl"].context == "prod"

    result = RUNNER.invoke(
        cli.app,
        ["sft-cpu-preflight-job-collect", str(output), "--context", "prod", "--attempt", "2"],
    )
    assert result.exit_code == 0
    assert calls[1] == (
        "collect",
        {
            "directory": output,
            "source_commit": source_commit,
            "attempt": 2,
            "kubectl": calls[1][1]["kubectl"],
        },
    )
    assert calls[1][1]["kubectl"].context == "prod"
    assert cli._read(output / "PREFLIGHT.json") == receipt


def test_lr30_one_off_prepare_preflight_and_direct_submit_are_exact(tmp_path, monkeypatch):
    from cyber_post_train import direct_submit
    from training import qwen38_lr30_step76_gate as gate

    plan_file = (
        Path(__file__).resolve().parents[1]
        / "configs/qualification/qwen38-lr30-step76-gpu-reload-v1.json"
    )
    output = tmp_path / "lr30"
    result = RUNNER.invoke(
        cli.app,
        ["lr30-step76-prepare", str(plan_file), "--output", str(output)],
    )
    assert result.exit_code == 0
    plan, request = cli._prepared(output)
    assert plan["schema"] == gate.PLAN_SCHEMA
    assert request == gate.job_request()

    monkeypatch.setattr(cli, "_require_output_absent", lambda _: None)
    calls = []

    def preflight(plan_value, request_value):
        calls.append((plan_value, request_value))
        return {
            "schema": gate.PREFLIGHT_SCHEMA,
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(plan_value),
            "request_sha256": digest(request_value),
            "checked_at_epoch": time.time(),
            "source_receipt_sha256": gate.EXPORT_RECEIPT_SHA256,
            "source_manifest_file_sha256": gate.SOURCE_MANIFEST_FILE_SHA256,
            "output_absent": True,
        }

    monkeypatch.setattr(gate, "preflight", preflight)
    assert RUNNER.invoke(cli.app, ["preflight", str(output)]).exit_code == 0
    assert calls == [(plan, request)]
    cli._require_preflight(output, plan, request)

    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_client", lambda: nullcontext("jobs-client"))
    direct_calls = []

    class SyntheticKubectl:
        def __init__(self, context):
            self.context = context

    def submit(**kwargs):
        direct_calls.append(kwargs)
        return {"name": "chris-q38-lr30-s76-gpu-v1-12345678", "submitted": True}

    monkeypatch.setattr(direct_submit, "Kubectl", SyntheticKubectl)
    monkeypatch.setattr(direct_submit, "direct_submit_lr30_qualification_once", submit)
    result = RUNNER.invoke(
        cli.app,
        ["direct-submit-lr30-step76", str(output), "--context", "prod-context"],
    )
    assert result.exit_code == 0
    assert direct_calls[0]["plan"] == plan and direct_calls[0]["request"] == request
    assert direct_calls[0]["jobs"] == "jobs-client"
    assert direct_calls[0]["kubectl"].context == "prod-context"
    assert direct_calls[0]["journal"] == output / "DIRECT_SUBMISSION.jsonl"


def test_lr30_preflight_expires_before_any_network(tmp_path, monkeypatch):
    from training import qwen38_lr30_step76_gate as gate

    plan = {"schema": gate.PLAN_SCHEMA}
    request = {"immutable": "synthetic"}
    output = tmp_path / "prepared"
    output.mkdir()
    proof = {
        "schema": gate.PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checked_at_epoch": time.time() - 1801,
        "output_absent": True,
    }
    cli._write(output / "PREFLIGHT.json", {**proof, "sha256": digest(proof)})
    with pytest.raises(ValueError, match="stale or incomplete"):
        cli._require_preflight(output, plan, request)


def test_submit_rejects_pre_gate_preparation_before_network(prepared, monkeypatch):
    output, plan, request, _ = prepared
    record_preflight(output, plan, request)
    (output / "PREPARED.json").write_text(
        json.dumps(
            {
                "plan_sha256": digest(plan),
                "request_sha256": digest(request),
            }
        )
    )
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("stale gate reached network"))
    assert RUNNER.invoke(cli.app, ["submit", str(output)]).exit_code == 2


def test_submit_regenerates_request_with_current_source_and_gates(prepared, monkeypatch):
    output, plan, request, _ = prepared
    record_preflight(output, plan, request)
    changed = {**request, "command": request["command"] + " --new-source"}
    monkeypatch.setattr(sft, "job_request", lambda _: changed)
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("stale request reached network"))
    result = RUNNER.invoke(cli.app, ["submit", str(output)])
    assert result.exit_code == 2
    assert "No automatic retry" in result.stderr


def test_preview_and_status_are_read_only(prepared, monkeypatch):
    output, _, request, _ = prepared
    fake = SimpleNamespace(
        preview=lambda config: {"synthetic": config},
        status=lambda name: {"name": name, "status": "RUNNING"},
    )
    monkeypatch.setattr(cli, "_client", lambda _plan=None: nullcontext(fake))
    monkeypatch.setattr(
        cli, "validate_preview", lambda config, result: {"nodes": config["workers"]}
    )
    result = RUNNER.invoke(cli.app, ["preview", str(output)])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "submitted": False,
        "nodes": request["workers"],
    }
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
    monkeypatch.setattr(
        cli, "Jobs", lambda token, *, base_url: {"received": token, "base_url": base_url}
    )
    assert cli._client() == {
        "received": "synthetic-operator-key",
        "base_url": "https://api.ft.flt.build",
    }


def test_client_route_comes_only_from_immutable_plan(monkeypatch):
    calls = []
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-operator-key")
    monkeypatch.setattr(
        cli,
        "Jobs",
        lambda token, *, base_url: calls.append((token, base_url)) or {"base_url": base_url},
    )
    plan = {
        "execution": {
            "cluster_target": "dev",
            "jobs_api_base_url": "https://api.ft.dev.flt.build",
        }
    }
    assert cli._client(plan) == {"base_url": "https://api.ft.dev.flt.build"}
    assert calls == [("synthetic-operator-key", "https://api.ft.dev.flt.build")]
    plan["execution"]["jobs_api_base_url"] = "https://api.ft.flt.build"
    with pytest.raises(Exception, match="incomplete or mismatched"):
        cli._client(plan)
    assert len(calls) == 1


def test_development_status_requires_and_uses_prepared_route(prepared, monkeypatch):
    output, plan, _, _ = prepared
    plan["run_name"] = "synthetic"
    plan["execution"] = {
        "cluster_target": "dev",
        "jobs_api_base_url": "https://api.ft.dev.flt.build",
    }
    request = cli._read(output / "request.json")
    # Replace the create-once fixture in a separate prepared directory so its
    # receipt remains digest-bound rather than mutating a launch in place.
    dev = output.parent / "dev-prepared"
    cli._prepare(dev, plan, request)
    seen = []
    fake = SimpleNamespace(status=lambda name: seen.append(name) or {"name": name})
    monkeypatch.setattr(cli, "_client", lambda selected=None: nullcontext(fake))
    result = RUNNER.invoke(cli.app, ["status", "synthetic", "--prepared", str(dev)])
    assert result.exit_code == 0 and seen == ["synthetic"]
    assert RUNNER.invoke(cli.app, ["status", "other", "--prepared", str(dev)]).exit_code == 2


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
        lambda _plan=None: nullcontext(
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


@pytest.mark.parametrize("fails", [False, True])
def test_sft_data_command_preserves_numeric_configuration(tmp_path, monkeypatch, fails):
    from training import corpus

    path = tmp_path / "data.json"
    path.write_text(json.dumps({"temperature": 1e-6}))
    calls = []

    def build(value, *, relative_to):
        calls.append((value, relative_to))
        if fails:
            raise ValueError("private data record")
        return {"train_rows": 2}

    monkeypatch.setattr(corpus, "build", build)
    result = RUNNER.invoke(cli.app, ["data", str(path)])
    assert result.exit_code == (2 if fails else 0)
    assert calls == [({"temperature": 1e-6}, tmp_path)]
    assert "private data record" not in result.output


def test_unknown_rl_backend_does_not_create_output(tmp_path):
    path, output = tmp_path / "config.json", tmp_path / "prepared"
    path.write_text(json.dumps({"backend": "unknown"}))
    result = RUNNER.invoke(cli.app, ["rl", str(path), "--output", str(output)])
    assert result.exit_code == 2 and not output.exists()


@pytest.mark.parametrize("fails", [False, True])
def test_eval_commands_dispatch_without_exposing_private_errors(tmp_path, monkeypatch, fails):
    from evals.fleet import evaluate, rollout_postgres

    calls = []
    dsn = "postgresql://synthetic.invalid/isolated-test"
    monkeypatch.setenv("ROLLOUT_DATABASE_URL", dsn)
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"temperature": 1e-6}))

    def handler(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            if fails:
                raise RuntimeError("private prompt or credential")
            return result

        return call

    for module, name, result in (
        (evaluate, "prepare", {"submitted": False}),
        (evaluate, "preflight", {"gpus": 0}),
        (evaluate, "checked_preflight", {}),
        (evaluate, "run", {"results": []}),
        (rollout_postgres, "initialize", {"pending": 1}),
        (rollout_postgres, "summary", {"accepted": 0}),
    ):
        monkeypatch.setattr(module, name, handler(name, result))
    commands = [
        ["prepare", str(source), "--output", str(tmp_path)],
        ["preflight", str(tmp_path)],
        ["init", str(tmp_path)],
        ["run", str(tmp_path), "synthetic-route", "worker-001", "--limit", "2"],
        ["status"],
    ]
    for args in commands:
        result = RUNNER.invoke(cli.app, ["eval", *args])
        assert result.exit_code == (2 if fails else 0), result.output
        assert "private prompt or credential" not in result.output
    if fails:
        assert "initialize" not in [name for name, _, _ in calls]
    else:
        assert calls == [
            ("prepare", ({"temperature": 1e-6}, tmp_path), {"relative_to": tmp_path}),
            ("preflight", (tmp_path,), {}),
            ("checked_preflight", (tmp_path,), {}),
            ("initialize", (dsn, tmp_path / "plan.csv"), {}),
            (
                "run",
                (tmp_path,),
                {
                    "dsn": dsn,
                    "route": "synthetic-route",
                    "worker_id": "worker-001",
                    "limit": 2,
                },
            ),
            ("summary", (dsn,), {}),
        ]


def test_eval_controller_failure_has_nonzero_exit_without_automatic_retry(tmp_path, monkeypatch):
    from evals.fleet import evaluate

    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"results": [{"controller_failure_code": "synthetic-failure"}]}

    monkeypatch.setenv("ROLLOUT_DATABASE_URL", "synthetic")
    monkeypatch.setattr(evaluate, "run", run)
    result = RUNNER.invoke(cli.app, ["eval", "run", str(tmp_path), "route", "worker"])
    assert result.exit_code == 1 and len(calls) == 1
    assert json.loads(result.stdout)["results"][0]["controller_failure_code"] == "synthetic-failure"


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
        ["rl-checkpoint-seal"],
        ["rl-checkpoint-export"],
        ["checkpoint-check"],
    ],
)
def test_help_is_accessible(command):
    result = RUNNER.invoke(cli.app, [*command, "--help"])
    assert result.exit_code == 0
