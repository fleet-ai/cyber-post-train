"""Configurable evaluation tests use synthetic metadata, never benchmark content."""

import copy
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from evals.fleet import evaluate as evaluation
from evals.fleet import rollout_worker

VERSION = "11111111-1111-4111-8111-111111111111"
VERSION2 = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def configuration(tmp_path):
    tasks = [
        {
            "task_key": "synthetic-" + version[:8],
            "task_version_id": version,
            "env_key": "synthetic-env",
            "env_version": "v1",
            "environment_version_id": version,
            "data_key": "synthetic-data",
            "data_version": "v1",
        }
        for version in (VERSION, VERSION2)
    ]
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": tasks}))
    return {
        "name": "synthetic-eval",
        "task_set": "tasks.json",
        "pass_k": 2,
        "concurrency": 2,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "models": {
            "student": {
                "repository": "synthetic/model",
                "revision": "a" * 40,
                "session_model": "synthetic/student",
            }
        },
        "routes": {
            "shared": {
                "model": "student",
                "served_id": "synthetic-student",
                "task_versions": [VERSION, VERSION2],
                "endpoint_origin": "https://inference.flt.build",
                "catalog": {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 1},
                "model_info": {
                    "model_path": "/model",
                    "model_type": "synthetic",
                    "architectures": ["Synthetic"],
                },
                "server_info": {
                    "model_path": "/model",
                    "context_length": 262144,
                    "tp_size": 1,
                    "quantization": None,
                    "kv_cache_dtype": "fp8_e4m3",
                    "reasoning_parser": "qwen3",
                    "tool_call_parser": "qwen3_coder",
                },
            }
        },
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "release_asset_sha256": "sha256:" + "b" * 64,
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v1",
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": "sha256:" + "c" * 64,
        },
        "images": {
            "agent": "registry/agent@sha256:" + "d" * 64,
            "proxy": "registry/proxy@sha256:" + "e" * 64,
        },
    }


def test_offline_plan_any_task_count_and_pass_k(configuration, tmp_path):
    directory = tmp_path / "prepared"
    receipt = evaluation.prepare(configuration, directory, relative_to=tmp_path)
    assert receipt["sessions"] == 4 and receipt["submitted"] is False
    plan = evaluation.load(directory)
    assert plan["training_data_eligible"] is False
    assert plan["automatic_retry"] is False
    assert len(evaluation.ledger._plan_rows(directory / "plan.csv")) == 4
    with pytest.raises(FileExistsError):
        evaluation.prepare(configuration, directory, relative_to=tmp_path)


def test_complete_task_partition_not_attempt_partition(configuration, tmp_path):
    configuration["routes"]["dedicated"] = copy.deepcopy(configuration["routes"]["shared"])
    configuration["routes"]["shared"]["task_versions"] = [VERSION]
    configuration["routes"]["dedicated"]["task_versions"] = [VERSION2]
    plan = evaluation.compile_eval(configuration, relative_to=tmp_path)
    rows = evaluation.plan_rows(plan)
    assert len(rows) == 4
    assert {(r["task_version_id"], r["serving_block"]) for r in rows} == {
        (VERSION, "shared"),
        (VERSION2, "dedicated"),
    }
    configuration["routes"]["dedicated"]["task_versions"].append(VERSION)
    with pytest.raises(ValueError, match="split"):
        evaluation.compile_eval(configuration, relative_to=tmp_path)


def test_model_alias_accepts_version_dots(configuration, tmp_path):
    configuration["models"]["qwen3.8-27b"] = configuration["models"].pop("student")
    configuration["routes"]["shared"]["model"] = "qwen3.8-27b"
    plan = evaluation.compile_eval(configuration, relative_to=tmp_path)
    assert all(row["model_id"] == "qwen3.8-27b" for row in evaluation.plan_rows(plan))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(pass_k=True),
        lambda c: c.update(concurrency=0),
        lambda c: c.update(unknown="no"),
        lambda c: c.update(training_data_eligible="yes"),
        lambda c: c["images"].update(agent="registry/agent:latest"),
        lambda c: c["models"]["student"].update(revision="main"),
        lambda c: c["harness"].update(tools=["bash"]),
        lambda c: c["harness"].update(provider_adapter="different-provider"),
        lambda c: c["harness"].update(timeout_seconds=28801),
        lambda c: c["routes"]["shared"].update(endpoint_origin="https://example.com"),
        lambda c: c["routes"]["shared"].update(task_versions=[VERSION]),
        lambda c: c["routes"]["shared"].update(task_versions=[VERSION, VERSION]),
        lambda c: c["routes"]["shared"]["server_info"].update(context_length=32768),
    ],
)
def test_invalid_config_has_no_output(configuration, tmp_path, mutate):
    mutate(configuration)
    with pytest.raises((ValueError, RuntimeError)):
        evaluation.prepare(configuration, tmp_path / "out", relative_to=tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("file", ["plan.csv", "EVAL.json"])
def test_prepared_tamper_detected(configuration, tmp_path, file):
    directory = tmp_path / "out"
    evaluation.prepare(configuration, directory, relative_to=tmp_path)
    path = directory / file
    path.write_text(path.read_text().replace("synthetic-eval", "different-eval"))
    with pytest.raises(ValueError):
        evaluation.load(directory)


def endpoint_client(route, model, *, drift=None, status=200):
    payloads = {
        "/fleet/v1/model-catalog": {
            "data": [
                {
                    **route["catalog"],
                    "id": route["served_id"],
                    "model_revision": model["revision"],
                    "routed": True,
                    "status": "ready",
                    "ready_replicas": 1,
                    "capabilities": ["tool_calling"],
                }
            ]
        },
        "/model_info": copy.deepcopy(route["model_info"]),
        "/server_info": {**route["server_info"], "served_model_name": route["served_id"]},
    }
    if drift:
        payloads["/server_info"][drift] = "changed"

    def respond(request):
        assert request.method == "GET" and request.headers["X-Fleet-Model"] == route["served_id"]
        return httpx.Response(status, json=payloads[request.url.path])

    return httpx.Client(transport=httpx.MockTransport(respond))


@pytest.mark.parametrize(
    "drift,status", [(None, 200), ("tp_size", 200), ("served_model_name", 200), (None, 503)]
)
def test_live_route_projection(configuration, drift, status):
    route, model = configuration["routes"]["shared"], configuration["models"]["student"]
    with endpoint_client(route, model, drift=drift, status=status) as client:
        if drift or status != 200:
            with pytest.raises(RuntimeError):
                evaluation.check_route(route, model, client)
        else:
            assert evaluation.check_route(route, model, client)["ready"] is True


@pytest.mark.parametrize("drift", [None, "dp_size", "load_balance_method"])
def test_data_parallel_profile_is_bound_without_reinterpreting_catalog(
    configuration, tmp_path, drift
):
    route, model = configuration["routes"]["shared"], configuration["models"]["student"]
    # Some catalog entries describe allocated GPUs here, not the runtime TP.
    # Preserve both observed values; never silently equate them.
    route["catalog"]["tensor_parallel_size"] = 8
    route["server_info"].update(dp_size=8, load_balance_method="total_tokens")
    plan = evaluation.compile_eval(configuration, relative_to=tmp_path)
    assert plan["routes"]["shared"]["server_info"]["tp_size"] == 1
    with endpoint_client(route, model, drift=drift) as client:
        if drift:
            with pytest.raises(RuntimeError, match="serving runtime drift"):
                evaluation.check_route(route, model, client)
        else:
            assert evaluation.check_route(route, model, client)["ready"]


@pytest.mark.parametrize("fault", ["missing_required", "unknown"])
def test_optional_parallel_profile_does_not_relax_required_fields(configuration, tmp_path, fault):
    profile = configuration["routes"]["shared"]["server_info"]
    profile["dp_size"] = 8
    if fault == "missing_required":
        profile.pop("tp_size")
    else:
        profile["unreviewed"] = "value"
    with pytest.raises(ValueError, match="every required runtime field"):
        evaluation.compile_eval(configuration, relative_to=tmp_path)


@pytest.fixture
def prepared(configuration, tmp_path, monkeypatch):
    directory = tmp_path / "prepared"
    evaluation.prepare(configuration, directory, relative_to=tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-not-a-secret")
    monkeypatch.setattr(
        evaluation.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "1.18.27\n/home/node/.local/share/opencode/opencode.db\n"
                if args[1] == "run"
                else json.dumps(
                    {
                        "RepoDigests": [args[3]],
                        "Os": "linux",
                        "Architecture": "amd64",
                        "Config": {
                            "Labels": {"cyber.opencode.release-sha256": "sha256:" + "b" * 64}
                        },
                    }
                )
            ),
        ),
    )
    monkeypatch.setattr(evaluation.httpx, "Client", lambda **kwargs: nullcontext(object()))
    monkeypatch.setattr(
        evaluation.harness,
        "_request",
        lambda *args: {"team_name": "fleet", "team_id": evaluation.harness.FLEET_TEAM_ID},
    )
    monkeypatch.setattr(evaluation, "check_route", lambda *args: {"ready": True})
    monkeypatch.setattr(
        rollout_worker,
        "_task_binding",
        lambda *args: (
            {"cyber_contract": rollout_worker.AUTHORITY["required_cyber_contract"]},
            {},
            {},
        ),
    )
    assert evaluation.preflight(directory)["scored_sessions"] == 0
    return directory


def test_preflight_is_once_and_requires_all_task_bindings(prepared):
    with pytest.raises(FileExistsError):
        evaluation.preflight(prepared)
    path = prepared / "EVAL_PREFLIGHT.json"
    proof = json.loads(path.read_text())
    proof["task_bindings"].pop(VERSION)
    proof["sha256"] = digest({k: v for k, v in proof.items() if k != "sha256"})
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="incomplete"):
        evaluation.checked_preflight(prepared)


def test_preflight_rejects_historical_missing_seed_using_only_gets(
    configuration, tmp_path, monkeypatch
):
    directory = tmp_path / "prepared"
    evaluation.prepare(configuration, directory, relative_to=tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-not-a-secret")
    monkeypatch.setattr(evaluation, "check_images", lambda *_: None)
    monkeypatch.setattr(evaluation, "check_route", lambda *_: {"ready": True})
    calls = []

    def respond(request):
        calls.append(request)
        assert request.method == "GET"
        if request.url.path == "/v1/account":
            return httpx.Response(
                200,
                json={
                    "team_name": "fleet",
                    "team_id": evaluation.harness.FLEET_TEAM_ID,
                },
            )
        assert request.url.params["version_id"] == VERSION
        return httpx.Response(
            200,
            json={
                "key": "synthetic-" + VERSION[:8],
                "eval_task_version_id": VERSION,
                "environment_id": "synthetic-env",
                "version": "v1",
                "environment_version_id": VERSION,
                "data_id": None,
                "data_version": None,
                "seed_config": None,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(evaluation.httpx, "Client", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="no recorded starting-data binding"):
        evaluation.preflight(directory)
    assert len(calls) == 2
    assert not (directory / "EVAL_PREFLIGHT.json").exists()


@pytest.mark.parametrize("root", [False, True])
def test_agent_uid_always_has_explicit_home(monkeypatch, root):
    monkeypatch.setattr(os, "geteuid", lambda: 0 if root else 501)
    monkeypatch.setattr(os, "getuid", lambda: 501)
    monkeypatch.setattr(os, "getgid", lambda: 20)
    expected = ["-e", "HOME=/home/node"]
    if not root:
        expected += ["--user", "501:20"]
    assert evaluation.harness.agent_container_user_args() == expected


def test_native_image_initializes_home_offline_as_actual_controller_user():
    image = os.environ.get("CYBER_TEST_OPENCODE_IMAGE")
    if not image:
        pytest.skip("set CYBER_TEST_OPENCODE_IMAGE to a staged pinned Docker image")
    evaluation.check_images(
        {
            "images": {"agent": image},
            "treatment": {
                "harness_version": "1.18.27",
                "release_asset_sha256": (
                    "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
                ),
            },
        }
    )


def test_bounded_worker_does_not_initialize_or_repeat(prepared, monkeypatch):
    monkeypatch.setattr(evaluation.postgres, "verify_plan", Mock())
    monkeypatch.setattr(
        evaluation.postgres, "initialize", lambda *a: pytest.fail("worker initialized")
    )
    run_one = Mock(return_value={"claimed": True, "accepted": True})
    monkeypatch.setattr(rollout_worker, "run_one", run_one)
    result = evaluation.run(prepared, dsn="synthetic", route="shared", worker_id="worker", limit=2)
    assert result["accepted"] == 2 and run_one.call_count == 2
    kwargs = run_one.call_args.kwargs
    assert len(kwargs["universe_index"]) == 4
    assert kwargs["campaign"]["training_data_eligible"] is False
    with pytest.raises(FileExistsError):
        evaluation.run(prepared, dsn="synthetic", route="shared", worker_id="worker", limit=2)
    assert run_one.call_count == 2


def test_real_worker_reaches_harness_with_durable_claim(prepared, monkeypatch):
    """Exercise the filesystem boundary hidden by the earlier run_one stub."""
    plan = evaluation.load(prepared)
    row = evaluation.ledger._plan_rows(prepared / "plan.csv")[0]
    cell = {**row, "worker_id": "worker", "claim_id": "claim", "cell_id": "cell"}
    monkeypatch.setattr(evaluation.postgres, "verify_plan", Mock())
    monkeypatch.setattr(evaluation.postgres, "claim", Mock(return_value=cell))
    heartbeat = SimpleNamespace(check=lambda: None, close=lambda: None)
    monkeypatch.setattr(rollout_worker, "_Heartbeat", lambda *a: nullcontext(heartbeat))
    monkeypatch.setattr(rollout_worker, "_record_local_result", Mock())
    monkeypatch.setattr(
        rollout_worker, "_accepted_receipt", lambda *a: {"receipt_sha256": "digest"}
    )
    for method in ("start", "mark_grading", "accept"):
        monkeypatch.setattr(evaluation.postgres, method, Mock())

    def harness_run(config, out, proxy):
        claims = list((prepared / "claims").glob("*.json"))
        assert len(claims) == 1
        claim = json.loads(claims[0].read_text())
        assert claim["model_call_started_when_claim_written"] is False
        assert claim["campaign_id"] == plan["campaign_id"]
        out.mkdir()  # Parent must have been prepared, not supplied by a test fixture.
        return {"session_id": "synthetic-session"}

    monkeypatch.setattr(rollout_worker.self_hosted, "run", harness_run)
    result = evaluation.run(prepared, dsn="synthetic", route="shared", worker_id="real", limit=1)
    assert result["accepted"] == 1


def test_endpoint_failure_preserves_terminal_without_claim(prepared, monkeypatch):
    monkeypatch.setattr(evaluation.postgres, "verify_plan", Mock())
    monkeypatch.setattr(evaluation, "check_route", Mock(side_effect=RuntimeError("private")))
    monkeypatch.setattr(rollout_worker, "run_one", lambda **k: pytest.fail("claimed after drift"))
    result = evaluation.run(prepared, dsn="synthetic", route="shared", worker_id="worker", limit=1)
    assert result["results"][0]["controller_failure_code"] == "runtimeerror"
    assert (prepared / "TERMINAL-worker.json").exists()
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("defect", ["missing", "platform", "digest", "release", "version", "home"])
def test_execution_host_images_checked_before_claim(prepared, monkeypatch, defect):
    monkeypatch.setattr(evaluation.postgres, "verify_plan", Mock())
    monkeypatch.setattr(rollout_worker, "run_one", lambda **k: pytest.fail("claimed after drift"))

    def inspect(args, **kwargs):
        if args[1] == "run":
            assert args[args.index("-e") : args.index("-e") + 2] == ["-e", "HOME=/home/node"]
            assert args[-1] == "opencode --version && opencode db path"
            return SimpleNamespace(returncode=int(defect == "home"), stdout="different-version")
        info = {
            "RepoDigests": [args[3]],
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {"Labels": {"cyber.opencode.release-sha256": "sha256:" + "b" * 64}},
        }
        if defect == "platform":
            info["Architecture"] = "arm64"
        if defect == "digest":
            info["RepoDigests"] = []
        if defect == "release":
            info["Config"] = {}
        return SimpleNamespace(returncode=int(defect == "missing"), stdout=json.dumps(info))

    monkeypatch.setattr(evaluation.subprocess, "run", inspect)
    with pytest.raises(RuntimeError):
        evaluation.run(prepared, dsn="synthetic", route="shared", worker_id="bad", limit=1)
    assert not (prepared / "STARTED-bad.json").exists()


def test_eval_cli_prepare_and_status(configuration, tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["eval", "prepare", str(path), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["sessions"] == 4
    monkeypatch.setenv("ROLLOUT_DATABASE_URL", "synthetic")
    monkeypatch.setattr(evaluation.postgres, "summary", lambda dsn: {"pending": 4})
    assert json.loads(runner.invoke(cli.app, ["eval", "status"]).output) == {"pending": 4}
