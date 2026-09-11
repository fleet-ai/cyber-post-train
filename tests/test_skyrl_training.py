"""Offline launch/lifecycle tests; synthetic task data and no paid requests."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_rl_data import build, setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import rl_data, sft_runtime
from training import skyrl_training as train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(data_setup):  # noqa: F811
    lock = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
    data_setup.lock.update(json.loads(lock.read_bytes()))
    build(data_setup)
    config = {
        "backend": "skyrl",
        "name": "synthetic-rl",
        "output_root": "/mnt/sfs/jobs/synthetic-rl",
        "model": {
            "lock": str(lock),
            "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
            "root": data_setup.config["model_root"],
        },
        "data": {"manifest": "out/manifest.json", "root": "/mnt/sfs/data/synthetic-rl"},
        "recipe": {"groups": 1, "samples_per_prompt": 8, "steps": 2, "lr": 1e-6},
        "wandb": {"entity": "synthetic", "project": "synthetic", "run_id": "synthetic-rl"},
    }
    plan = train.compile_rl(config, relative_to=data_setup.tmp)
    return NS(config=config, plan=plan, state=data_setup)


def test_prepare_cli_and_portable_runtime_are_offline(prepared, monkeypatch):
    from training.sft import IMAGE

    tmp, plan = prepared.state.tmp, prepared.plan
    request = train.job_request(plan)
    assert IMAGE == train.IMAGE == request["image"]
    assert request["priority_class"] == "c1" and not request["requeueIfPreempted"]
    assert request["workers"] * request["gpus_per_worker"] == 8
    assert request["secrets"] == ["fleet-api", "wandb-api"]
    assert "API_KEY" not in str(request["env"])
    assert plan["arguments"]["steps"] == plan["native_overrides"]["trainer.max_training_steps"] == 2
    source = tmp / "launch.json"
    source.write_text(json.dumps(prepared.config))
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("offline command used network"))
    # A staged SkyRL runtime does not contain the unrelated Miles launcher.
    monkeypatch.setitem(sys.modules, "training.miles_training", None)
    result = CliRunner().invoke(cli.app, ["rl", str(source), "--output", str(tmp / "prepared")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["submitted"] is False
    assert cli._prepared(tmp / "prepared") == (plan, request)
    assert (
        CliRunner()
        .invoke(cli.app, ["rl", str(source), "--output", str(tmp / "prepared")])
        .exit_code
        == 2
    )
    # The immutable GPU bundle must be importable without the source checkout.
    bundle = tmp / "bundle"
    for name, content in train._runtime().items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for directory in ("training", "cyber_post_train", "evals", "evals/fleet"):
        (bundle / directory / "__init__.py").write_text("")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from training import skyrl_training as s; from training.rl_data import selection; "
            "assert s.MODULE == 'training.skyrl_training'",
        ],
        cwd=tmp,
        env={**os.environ, "PYTHONPATH": str(bundle)},
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "backend",
        "checkpoint",
        "priority",
        "resource",
        "groups",
        "model",
        "file",
        "data_name",
        "digest",
        "runtime",
        "native",
        "output",
    ],
)
def test_invalid_launch_stops_before_gpu_import(prepared, fault):
    config, tmp = prepared.config, prepared.state.tmp
    if fault == "unknown":
        config["extra"] = True
    elif fault == "backend":
        config["backend"] = "miles"
    elif fault == "checkpoint":
        config["checkpoint"] = {"root": "unreviewed"}
    elif fault == "priority":
        config["cluster"] = {"priority": "c0"}
    elif fault == "resource":
        config["cluster"] = {"resources": {"memory_request": "2Gi"}}
    elif fault == "groups":
        config["recipe"]["groups"] = 2
    elif fault in {"model", "file", "data_name", "digest"}:
        path = tmp / "out/manifest.json"
        value = json.loads(path.read_bytes())
        if fault == "model":
            value["tokenizer"]["repo"] = "zai-org/GLM-5.3"
        elif fault == "file":
            value["files"]["train"]["path"] = "../train.jsonl"
        elif fault == "data_name":
            value["name"] = "other-run"
        value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
        if fault == "digest":
            value["sha256"] = "wrong"
        path.write_text(json.dumps(value))
    else:
        plan = prepared.plan
        if fault == "runtime":
            plan["runtime_sha256"] = "a" * 64
        elif fault == "native":
            plan["native_overrides"]["trainer.max_training_steps"] = 999
        else:
            plan["output_root"] = "/mnt/sfs/jobs/other"
        with pytest.raises(ValueError):
            train.job_request(plan)
        return
    with pytest.raises(ValueError):
        train.compile_rl(config, relative_to=tmp)


@pytest.fixture
def artifacts(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp / "out"
    monkeypatch.setattr(train, "check_inputs", lambda _: None)
    plan["arguments"]["data_manifest"] = str(root / "manifest.json")
    for split in ("train", "dev"):
        plan["arguments"][split + "_data"] = str(root / (split + ".jsonl"))
    return plan, root


def test_artifact_and_dataset_boundary_preserve_rows(artifacts, monkeypatch):
    plan, root = artifacts
    before = {p: p.read_bytes() for p in root.iterdir()}
    rows = train.check_artifacts(plan)
    assert {k: len(v) for k, v in rows.items()} == {"train": 1, "dev": 1}
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    for split in rows:
        assert len(train.dataset(plan, tokenizer, split, rows[split])) == 1
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "bytes",
        "split_digest",
        "count",
        "prompt",
        "split",
        "run",
        "env",
        "template",
        "duplicate",
        "task",
    ],
)
def test_artifact_drift_is_fatal(artifacts, fault):
    plan, root = artifacts
    if fault == "manifest":
        (root / "manifest.json").write_text("{}")
    elif fault == "bytes":
        (root / "train.jsonl").write_text("changed")
    elif fault == "split_digest":
        split = json.loads((root / "split.json").read_bytes())
        split["sha256"] = "changed"
        (root / "split.json").write_text(json.dumps(split))
    elif fault == "count":
        plan["data"]["files"]["train"]["rows"] = 2
        (root / "manifest.json").write_text(json.dumps(plan["data"]))
    else:
        path = root / "train.jsonl"
        row = json.loads(path.read_bytes())
        cfg = json.loads(row["cyber_config_json"])
        if fault == "prompt":
            row["prompt"][0]["content"] = "changed"
        elif fault == "split":
            row["split"] = "dev"
        elif fault == "run":
            cfg["run_id"] = "changed"
        elif fault == "env":
            row["env_class"] = "changed"
        elif fault == "template":
            cfg["model"]["runtime_chat_template_sha256"] = "changed"
        elif fault == "task":
            cfg["task"]["key"] = "changed"
        cfg["config_sha256"] = fleet.digest_without(cfg, "config_sha256")
        row["cyber_config_json"] = json.dumps(cfg)
        path.write_text((json.dumps(row) + "\n") * (2 if fault == "duplicate" else 1))
        item = plan["data"]["files"]["train"]
        item["sha256"] = "sha256:" + train._hash(path)
        item["rows"] = 2 if fault == "duplicate" else 1
        (root / "manifest.json").write_text(json.dumps(plan["data"]))
    with pytest.raises(ValueError):
        train.check_artifacts(plan)


@pytest.mark.parametrize("drift", ["drop", "prompt", "binding", "env"])
def test_native_dataset_cannot_filter_or_edit(artifacts, prepared, monkeypatch, drift):
    plan, _ = artifacts
    rows = train.check_artifacts(plan)
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    prepared.state.drift = drift
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    with pytest.raises(ValueError, match="silently"):
        train.dataset(plan, tokenizer, "train", rows["train"])


@pytest.mark.parametrize("fault", [None, "gpu", "output", "template"])
def test_cpu_preflight_dispatch_never_starts_ray(artifacts, monkeypatch, fault):
    plan, root = artifacts
    plan["output_root"] = str(root / "new-output")
    if fault == "output":
        Path(plan["output_root"]).mkdir()
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    if fault == "template":
        tokenizer.chat_template = "changed"
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: fault == "gpu")))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: tokenizer)),
    )
    monkeypatch.setattr(train, "job_request", lambda _: {"fixture": True})
    monkeypatch.setattr(train, "native_source", lambda: {})
    monkeypatch.setattr(train.skyrl, "native_config", lambda _: NS())
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    if fault:
        with pytest.raises((ValueError, FileExistsError)):
            train.preflight(plan)
    else:
        proof = train.preflight(plan)
        assert proof["status"] == "passed" and proof["gpus"] == 0
        assert proof["native_parser_checked"] and not proof["rl_qualified"]


@pytest.fixture
def completed(prepared):
    plan, root = prepared.plan, prepared.state.tmp / "result"
    plan["output_root"] = str(root)
    args = plan["arguments"]
    for phase, step in (("eval", 0), ("train", 1), ("eval", 1), ("train", 2), ("eval", 2)):
        directory = root / f"episodes/batches/{phase}-{step}"
        directory.mkdir(parents=True)
        value = {
            "schema": "cyber_skyrl_batch_v1",
            "phase": phase,
            "global_step": step,
            "data_sha256": plan["data"]["sha256"],
        }
        value["sha256"] = "sha256:" + digest(value)
        (directory / "COLLECTED.json").write_text(json.dumps(value))
    checkpoint = root / f"checkpoints/global_step_{args['steps']}"
    (checkpoint / "policy").mkdir(parents=True)
    for name in ("data.pt", "trainer_state.pt", "policy/synthetic.distcp"):
        (checkpoint / name).write_bytes(b"synthetic state, not a checkpoint qualification")
    (root / "checkpoints/latest_ckpt_global_step.txt").write_text(str(args["steps"]))
    return plan, root


@pytest.mark.parametrize(
    "fault", [None, "failed", "batch", "digest", "missing", "pointer", "sampler", "policy"]
)
def test_terminal_checks_do_not_fabricate_acceptance(completed, fault):
    plan, root = completed
    path = root / "episodes/batches/train-1"
    if fault == "failed":
        (path / "FAILED.json").write_text("{}")
    elif fault in {"batch", "digest"}:
        value = json.loads((path / "COLLECTED.json").read_bytes())
        value["global_step"] = 3
        if fault == "batch":
            value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
        (path / "COLLECTED.json").write_text(json.dumps(value))
    elif fault == "missing":
        (path / "COLLECTED.json").unlink()
        path.rmdir()
    elif fault == "pointer":
        (root / "checkpoints/latest_ckpt_global_step.txt").write_text("1")
    elif fault == "sampler":
        (root / "checkpoints/global_step_2/data.pt").unlink()
    elif fault == "policy":
        (root / "checkpoints/global_step_2/policy/synthetic.distcp").unlink()
    if fault:
        with pytest.raises(ValueError):
            train.native_result(plan)
    else:
        proof = train.native_result(plan)
        assert proof["checkpoint_global_step"] == 2 and proof["completed_batches"] == 5
        assert (
            not proof["optimizer_update_independently_verified"]
            and not proof["checkpoint_reload_verified"]
        )
    assert not (root / "ACCEPTED.json").exists()


def test_scalar_tracking_does_not_upload_private_exceptions(prepared, monkeypatch):
    plan = prepared.plan
    plan["output_root"] = str(prepared.state.tmp)
    calls = []
    monkeypatch.setenv("WANDB_API_KEY", "synthetic")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        NS(
            init=lambda **kw: (
                calls.append(kw),
                NS(id="synthetic-rl", entity="synthetic", project="synthetic"),
            )[1],
            log=lambda **kw: calls.append(kw),
            finish=lambda **kw: pytest.fail("premature finish"),
        ),
    )
    tracker = train.ScalarTracking(plan)
    tracker.log({"policy/loss": 0.5, "eval/reward": 0.0}, step=1, commit=True)
    tracker.log_exception(RuntimeError("private secret task text"), step=1)
    tracker.finish()
    assert len(calls) == 2 and "private secret" not in str(calls)
    assert os.environ["WANDB_CONSOLE"] == "off" and os.environ["WANDB_RESUME"] == "never"
    for value in (True, "text", {}, [0.5], float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tracker.log({"value": value}, 2)
    with pytest.raises(ValueError):
        tracker.log({"bad\nkey": 0.0}, 2)
    with pytest.raises(ValueError):
        tracker.log({}, -1)
    with pytest.raises(ValueError):
        tracker.log_samples_to_table("text", [], [], 1)
    sys.modules["wandb"].init = lambda **kw: NS(id="other", entity="synthetic", project="synthetic")
    with pytest.raises(ValueError, match="identity mismatch"):
        train.ScalarTracking(plan)


def test_native_sources_are_checked_without_inventing_a_driver(monkeypatch):
    calls = []
    monkeypatch.setattr(sft_runtime, "validate_runtime_sources", lambda: calls.append("runtime"))
    monkeypatch.setattr(train, "_module", lambda name, sha: (calls.append((name, sha)), name)[1])
    assert train.native_source() == {name: name for name in train.NATIVE}
    assert calls == ["runtime", *train.NATIVE.items()]


@pytest.mark.parametrize(
    "fault", [None, "setup", "native", "step", "checkpoint", "tracking", "no_tracking"]
)
def test_native_wrapper_keeps_native_loop_and_truthful_finalization(prepared, monkeypatch, fault):
    from training import skyrl_rollout

    plan, calls = prepared.plan, []
    result_rows = {"train": ["train"], "dev": ["dev"]}
    monkeypatch.setattr(train, "check_artifacts", lambda _: result_rows)
    monkeypatch.setattr(train.skyrl, "native_config", lambda _: "native-config")
    monkeypatch.setattr(
        train, "dataset", lambda p, t, split, rows: (calls.append((split, rows)), rows)[1]
    )
    monkeypatch.setattr(train, "ScalarTracking", lambda p: "scalar-tracker")
    monkeypatch.setattr(
        skyrl_rollout, "Generator", lambda *a, **kw: (calls.append((a, kw)), "generator")[1]
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        NS(init=lambda **kw: calls.append(("ray", kw)), shutdown=lambda: calls.append("shutdown")),
    )

    def finish(**kw):
        calls.append(("finish", kw))
        if fault == "tracking":
            raise RuntimeError("synthetic tracking failure")

    monkeypatch.setitem(
        sys.modules, "wandb", NS(run=None if fault == "no_tracking" else NS(), finish=finish)
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.utils.ppo_utils",
        NS(sync_registries=lambda: calls.append("sync")),
    )

    class Native:
        def __init__(self, cfg):
            assert cfg == "native-config"
            if fault == "setup":
                raise RuntimeError("synthetic setup failure")
            self.tokenizer = "native-tokenizer"
            assert self.get_train_dataset() == ["train"]
            assert self.get_eval_dataset() == ["dev"]
            assert self.get_generator(cfg, self.tokenizer, "engine") == "generator"
            assert self.get_tracker() == "scalar-tracker"
            assert self.get_trajectory_logger() is None
            self.trainer = NS(global_step=1 if fault == "step" else 2)

        def run(self):
            calls.append("native-loop")
            if fault == "native":
                raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(
        train,
        "native_source",
        lambda: {
            "skyrl.train.entrypoints.main_base": NS(BasePPOExp=Native),
            "skyrl.train.utils.utils": NS(
                prepare_runtime_environment=lambda cfg: {"PINNED": "yes"}
            ),
        },
    )

    def result(p):
        assert p == plan
        calls.append("checkpoint-check")
        if fault == "checkpoint":
            raise ValueError("synthetic missing checkpoint")

    monkeypatch.setattr(train, "native_result", result)
    if fault and fault != "no_tracking":
        with pytest.raises((RuntimeError, ValueError)):
            train._native(plan)
    else:
        train._native(plan)
    assert calls[-1] == "shutdown"
    if fault == "no_tracking":
        assert not any(isinstance(call, tuple) and call[0] == "finish" for call in calls)
    else:
        assert ("finish", {"exit_code": 0 if fault in (None, "tracking") else 1}) in calls
    assert calls[0][0] == "ray" and calls[0][1]["address"] == "auto"
    assert not calls[0][1]["log_to_driver"]
    assert calls[0][1]["runtime_env"]["env_vars"]["PINNED"] == "yes"


@pytest.mark.parametrize("mode", ["parent", "native", "digest", "runtime", "execution"])
def test_main_dispatch_is_digest_bound_and_sanitized(prepared, monkeypatch, capsys, mode):
    plan, tmp = prepared.plan, prepared.state.tmp
    path = tmp / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []
    argv = [
        train.MODULE,
        "--plan",
        str(path),
        "--sha256",
        "wrong" if mode == "digest" else digest(plan),
    ]
    if mode == "native":
        argv.append("--native")
    monkeypatch.setattr(sys, "argv", argv)

    def request(p):
        if mode == "runtime":
            raise ValueError("synthetic secret not for stdout")

    def run(p, path):
        calls.append("parent")
        if mode == "execution":
            raise RuntimeError("synthetic private exception")
        return {"status": "native_loop_returned", "sha256": "synthetic"}

    monkeypatch.setattr(train, "job_request", request)
    monkeypatch.setattr(train, "run", run)
    monkeypatch.setattr(train, "_native", lambda p: calls.append("native"))
    if mode in {"digest", "runtime", "execution"}:
        with pytest.raises(SystemExit) as error:
            train.main()
        assert error.value.code == 1
    else:
        train.main()
        assert calls == [mode]
    output = capsys.readouterr().out
    assert "secret" not in output and "private" not in output


def test_shared_supervisor_uses_importable_skyrl_module(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp
    plan["output_root"] = str(root)
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(train, "native_source", lambda: None)
    monkeypatch.setattr(train, "native_result", lambda p: {"status": "native_loop_returned"})
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    calls = []

    def spawn(argv, **kw):
        calls.append(argv)
        return NS(pid=12345, poll=lambda: 0, returncode=0, wait=lambda **kw: 0)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = train.run(plan, root / "plan.json")
    assert result["status"] == "native_loop_returned"
    assert calls[0][2] == "training.skyrl_training" and calls[0][-1] == "--native"
    assert (root / "private-skyrl.log").stat().st_mode & 0o777 == 0o600


def test_module_entry_rejects_missing_plan_without_private_path(tmp_path, monkeypatch, capsys):
    import runpy

    monkeypatch.delitem(sys.modules, train.MODULE)
    monkeypatch.setattr(
        sys, "argv", [train.MODULE, "--plan", str(tmp_path / "private-plan"), "--sha256", "0" * 64]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module(train.MODULE, run_name="__main__")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "error_class": "FileNotFoundError",
    }


def test_skyrl_preflight_cli_and_submission_proof_dispatch(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp / "prepared"
    request = train.job_request(plan)
    cli._prepare(root, plan, request)
    monkeypatch.setattr(
        train,
        "preflight",
        lambda p: {
            "schema": "cyber_skyrl_training_cpu_preflight_v1",
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(p),
            "request_sha256": digest(request),
        },
    )
    result = CliRunner().invoke(cli.app, ["preflight", str(root)])
    assert result.exit_code == 0, result.output
    proof = json.loads((root / "PREFLIGHT.json").read_bytes())
    assert proof["schema"] == "cyber_skyrl_training_cpu_preflight_v1"
    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit_once(self, value, path):
            calls.append((value, path))
            return {"status": "synthetic-only"}

    monkeypatch.setattr(cli, "_client", Client)
    assert CliRunner().invoke(cli.app, ["submit", str(root)]).exit_code == 0
    assert calls == [(request, root / "SUBMISSION.jsonl")]
    proof["schema"] = "cyber_miles_training_cpu_preflight_v1"
    proof["sha256"] = digest({k: v for k, v in proof.items() if k != "sha256"})
    (root / "PREFLIGHT.json").write_text(json.dumps(proof))
    assert CliRunner().invoke(cli.app, ["submit", str(root)]).exit_code == 2
    assert len(calls) == 1


@pytest.mark.skipif(importlib.util.find_spec("skyrl") is None, reason="pinned SkyRL image only")
def test_real_native_driver_and_dataset_source_identities():
    modules = train.native_source()
    assert modules["skyrl.train.entrypoints.main_base"].BasePPOExp is not None
