"""Native conversion boundaries with synthetic files; no GPUs or environment creates."""

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from training import miles_conversion as convert

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return {
        "name": "synthetic-conversion",
        "output_root": "/mnt/sfs/jobs/synthetic-conversion",
        "model": {
            "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
            "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
            "root": "/mnt/sfs/models/synthetic-base",
        },
    }


@pytest.fixture
def plan(config):
    return convert.compile_conversion(config, relative_to=ROOT)


@pytest.fixture
def tiny(plan, tmp_path, monkeypatch):
    # Source data are test bytes, never model weights or private task text.
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights").write_bytes(b"synthetic")
    plan["model"]["root"] = str(source)
    plan["model"]["files"] = [
        {"path": p.name, "sha256": convert._hash(p)} for p in sorted(source.iterdir())
    ]
    output = tmp_path / "run"
    plan["output_root"] = str(output)
    monkeypatch.setenv("RUN_DIR", str(output))
    monkeypatch.setattr(convert, "native_arguments", lambda _: ["synthetic"])
    monkeypatch.setattr(convert, "job_request", lambda _: {"synthetic": True})
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 8))
    )
    return plan


def materialize(root):
    (root / "release").mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("release\n")
    (root / "release/.metadata").write_bytes(b"synthetic-metadata")
    (root / "release/__0_0.distcp").write_bytes(b"synthetic-checkpoint")


def test_conversion_compiles_only_native_loading_not_training(plan):
    request = convert.job_request(plan)
    assert request == convert.job_request(plan)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request["requeueIfPreempted"] is False
    assert request["secrets"] == [] and request["env"]["HF_HUB_OFFLINE"] == "1"
    assert len(plan["model"]["files"]) == 28
    assert plan["optimizer_steps"] == 0 and plan["deadline_seconds"] == 1800


@pytest.mark.parametrize(
    "defect", ["unknown", "glm", "overlap", "ancestor", "resources", "priority"]
)
def test_configuration_rejects_unsafe_changes(config, defect):
    if defect == "unknown":
        config["extra_args"] = "--arbitrary-override"
    elif defect == "glm":
        base = ROOT / "configs/models/glm53-30333038"
        config["model"].update(
            lock=str(base / "model.lock.json"), weights=str(base / "model.weights.json")
        )
    elif defect == "overlap":
        config["output_root"] = config["model"]["root"]
    elif defect == "ancestor":
        config["output_root"] = config["model"]["root"] + "/child"
    elif defect == "resources":
        config["cluster"] = {"resources": {"memory_request": "32Gi"}}
    else:
        config["cluster"] = {"priority": "c0"}
    with pytest.raises(ValueError):
        convert.compile_conversion(config, relative_to=ROOT)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "runtime_sha256",
        "native_converter_sha256",
        "optimizer_steps",
        "deadline_seconds",
        "image",
    ],
)
def test_request_rejects_plan_runtime_drift(plan, field):
    if field == "image":
        plan["execution"]["image"] = "different"
    else:
        plan[field] = "different"
    with pytest.raises(ValueError):
        convert.job_request(plan)


def test_hash_and_input_integrity(tiny, tmp_path, monkeypatch):
    convert.check_inputs(tiny)
    path = Path(tiny["model"]["root"]) / "weights"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        convert.check_inputs(tiny)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        convert._hash(link)
    native = hashlib.file_digest

    def race(stream, algorithm):
        result = native(stream, algorithm)
        path.write_bytes(b"race")
        return result

    monkeypatch.setattr(hashlib, "file_digest", race)
    with pytest.raises(ValueError, match="changed during"):
        convert._hash(path)


def test_remote_model_code_is_not_accepted(tiny):
    path = Path(tiny["model"]["root"]) / "config.json"
    path.write_text('{"text_config":{"auto_map":{"AutoModel":"remote.Code"}}}')
    tiny["model"]["files"][0]["sha256"] = convert._hash(path)
    with pytest.raises(ValueError, match="remote model code"):
        convert.check_inputs(tiny)


@pytest.fixture
def native_boundary(tmp_path, monkeypatch):
    source = tmp_path / "tools/convert_hf_to_torch_dist.py"
    source.parent.mkdir()
    source.write_text("synthetic native source")
    profile = SimpleNamespace(backend="megatron", vision=False, megatron_model_type="qwen3.8-27B")
    modules = {
        "fti.trainers.miles.run_fleet": SimpleNamespace(_RECIPES={"qwen3.8-27b": profile}),
        "miles.utils.external_utils.command_utils": SimpleNamespace(repo_base_dir=tmp_path),
        "miles.utils.external_utils.model_args_utils": SimpleNamespace(
            load_model_args=lambda _: "--bf16 --num-layers 64"
        ),
    }
    for k, v in modules.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(convert, "CONVERTER_SHA256", convert._hash(source))
    return source, profile


@pytest.mark.parametrize("fault", [None, "source", "backend", "vision", "model"])
def test_native_argument_identity_and_auto_pp(plan, native_boundary, fault):
    source, profile = native_boundary
    if fault == "source":
        source.write_text("drift")
    elif fault:
        setattr(
            profile,
            {"backend": "backend", "vision": "vision", "model": "megatron_model_type"}[fault],
            True,
        )
    if fault:
        with pytest.raises(ValueError):
            convert.native_arguments(plan)
    else:
        argv = convert.native_arguments(plan)
        assert argv[:6] == [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=8",
        ]
        assert argv[6] == str(source)
        assert "--optimizer" not in argv and "--pipeline-model-parallel-size" not in argv
        assert argv[-2:] == ["--save", plan["output_root"] + "/torch-dist"]


@pytest.fixture
def cpu(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return cuda


@pytest.mark.parametrize("fault", [None, "gpu", "exists", "architecture"])
def test_cpu_preflight(tiny, cpu, monkeypatch, fault):
    text = SimpleNamespace(num_hidden_layers=64, hidden_size=5120, vocab_size=248320)
    if fault == "architecture":
        text.num_hidden_layers = 2
    if fault == "gpu":
        cpu.is_available = lambda: True
    if fault == "exists":
        Path(tiny["output_root"]).mkdir()

    def load(root, **kwargs):
        assert root == tiny["model"]["root"] and kwargs == {
            "trust_remote_code": False,
            "local_files_only": True,
        }
        return SimpleNamespace(get_text_config=lambda: text)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=load)),
    )
    if fault:
        with pytest.raises((ValueError, FileExistsError)):
            convert.preflight(tiny)
    else:
        receipt = convert.preflight(tiny)
        assert receipt["gpus"] == 0 and receipt["gpu_conversion_verified"] is False
        assert receipt["plan_sha256"] == digest(tiny)


@pytest.mark.parametrize("fault", [None, "failed", "timeout", "kill", "startup", "disappeared"])
def test_child_group_cleanup_and_private_logs(tmp_path, monkeypatch, fault):
    waited, killed = [], []
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "/native/tools/convert.py",
    ]

    def wait(timeout):
        waited.append(timeout)
        if (fault in {"timeout", "kill", "disappeared"} and len(waited) == 1) or (
            fault == "kill" and len(waited) == 2
        ):
            raise subprocess.TimeoutExpired("synthetic", timeout)
        return 1 if fault == "failed" else 0

    def popen(actual, **kwargs):
        assert actual == argv and kwargs["start_new_session"] is True
        assert kwargs["env"]["PYTHONPATH"] == "/native:/root/Megatron-LM"
        assert "CONVERT_KEEP_PP1" not in kwargs["env"]
        return SimpleNamespace(
            pid=123,
            wait=wait,
            poll=lambda: None if fault in {"timeout", "kill", "disappeared"} else 0,
        )

    monkeypatch.setattr(subprocess, "Popen", popen)

    def kill(pid, sig):
        killed.append((pid, sig))
        if fault == "disappeared":
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", kill)
    log = tmp_path / "private.log"
    monkeypatch.setenv("CONVERT_KEEP_PP1", "1")
    if fault:
        with pytest.raises((RuntimeError, TimeoutError, subprocess.TimeoutExpired)):
            convert._child(argv, log, 0 if fault == "startup" else 1)
    else:
        convert._child(argv, log, 1)
    if fault in {"timeout", "kill"}:
        assert killed[0] == (123, signal.SIGTERM)
    if fault == "kill":
        assert killed[1] == (123, signal.SIGKILL)
    if fault != "startup":
        assert log.stat().st_mode & 0o777 == 0o600


def test_real_owned_child_timeout_is_reaped(tmp_path):
    pidfile = tmp_path / "pid"
    argv = [
        sys.executable,
        "-c",
        f"import os,pathlib,time;pathlib.Path({str(pidfile)!r})"
        ".write_text(str(os.getpid()));time.sleep(60)",
        "a",
        "b",
        "c",
        "/native/tools/convert.py",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        convert._child(argv, tmp_path / "private.log", 0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_already_exited_owned_group_is_clean(tmp_path):
    argv = [sys.executable, "-c", "pass", "a", "b", "c", "/native/tools/convert.py"]
    convert._child(argv, tmp_path / "private.log", 3)


@pytest.mark.parametrize("fault", [None, "tracker", "metadata", "empty", "symlink", "root-link"])
def test_checkpoint_inventory(tmp_path, fault):
    root = tmp_path / "checkpoint"
    root.mkdir()
    materialize(root)
    if fault == "tracker":
        (root / "latest_checkpointed_iteration.txt").write_text("1")
    elif fault == "metadata":
        (root / "release/.metadata").unlink()
    elif fault == "empty":
        (root / "release/__0_0.distcp").write_bytes(b"")
    elif fault == "symlink":
        (root / "indirect").symlink_to(root / "release/.metadata")
    elif fault == "root-link":
        (tmp_path / "link").symlink_to(root, target_is_directory=True)
        root = tmp_path / "link"
    if fault:
        with pytest.raises(ValueError):
            convert.inventory(root)
    else:
        assert len(convert.inventory(root)) == 3


@pytest.fixture
def completed(tiny, monkeypatch):
    root = Path(tiny["output_root"])
    root.mkdir()
    monkeypatch.setattr(convert, "_child", lambda *args: materialize(root / "torch-dist"))
    receipt = convert.run(tiny)
    assert receipt["optimizer_steps"] == 0
    assert receipt["checkpoint_sha256_verified"] is False
    with pytest.raises(FileExistsError):
        convert.run(tiny)
    return tiny


def test_run_binding_and_failure_preserve_evidence(tiny, monkeypatch):
    root = Path(tiny["output_root"])
    root.mkdir()
    monkeypatch.delenv("RUN_DIR")
    with pytest.raises(ValueError):
        convert.run(tiny)
    monkeypatch.setenv("RUN_DIR", str(root))

    def fail(*args):
        raise RuntimeError("sensitive private output")

    monkeypatch.setattr(convert, "_child", fail)
    with pytest.raises(RuntimeError, match="no automatic retry") as exc:
        convert.run(tiny)
    assert "sensitive" not in str(exc.value)
    assert json.loads((root / "FAILED.json").read_text())["error_class"] == "RuntimeError"


def test_run_rejects_driver_without_its_gpu_claim(tiny, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
    )
    with pytest.raises(ValueError, match="eight visible"):
        convert.run(tiny)


def test_deadline_covers_input_io_and_restores_handlers(tiny, monkeypatch):
    root = Path(tiny["output_root"])
    root.mkdir()
    before = signal.getsignal(signal.SIGALRM)

    def interrupted_io(plan):
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)

    monkeypatch.setattr(convert, "check_inputs", interrupted_io)
    with pytest.raises(RuntimeError):
        convert.run(tiny)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.alarm(0) == 0
    assert json.loads((root / "FAILED.json").read_text())["error_class"] == "TimeoutError"


@pytest.mark.parametrize("fault", [None, "gpu", "failed", "receipt", "files", "race", "inside"])
def test_cpu_seal(completed, cpu, tmp_path, monkeypatch, fault):
    root = Path(completed["output_root"])
    output = tmp_path / "sealed.json"
    if fault == "gpu":
        cpu.is_available = lambda: True
    elif fault == "failed":
        (root / "FAILED.json").write_text("{}")
    elif fault == "receipt":
        value = json.loads((root / "CONVERSION_COMPLETE.json").read_text())
        value["optimizer_steps"] = 1
        (root / "CONVERSION_COMPLETE.json").write_text(json.dumps(value))
    elif fault == "files":
        (root / "torch-dist/release/__0_0.distcp").write_bytes(b"changed-size")
    elif fault == "inside":
        output = root / "torch-dist/seal.json"
    elif fault == "race":
        original = convert.inventory
        calls = []

        def race(path):
            calls.append(path)
            return original(path) if len(calls) == 1 else []

        monkeypatch.setattr(convert, "inventory", race)
    if fault:
        with pytest.raises(ValueError):
            convert.seal(completed, output)
        assert not output.exists()
    else:
        result = convert.seal(completed, output)
        assert result["gpu_reload_verified"] is False and result["optimizer_steps"] == 0
        assert len(result["files"]) == 3
        assert all(len(f["sha256"]) == 64 for f in result["files"])
        assert output.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            convert.seal(completed, output)


@pytest.mark.parametrize("fault", [False, True])
def test_main_does_not_print_private_exception(tiny, tmp_path, monkeypatch, capsys, fault):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tiny))
    monkeypatch.setattr(
        sys, "argv", ["convert", "--plan", str(path), "--sha256", "bad" if fault else digest(tiny)]
    )
    calls = []

    def init(**kwargs):
        assert kwargs["address"] == "auto" and kwargs["log_to_driver"] is False
        calls.append("attached")

    def remote(**kwargs):
        assert kwargs == {"num_cpus": 1, "num_gpus": 8}
        return lambda fn: SimpleNamespace(remote=lambda plan: "owned-ref")

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=init,
            remote=remote,
            is_initialized=lambda: True,
            get=lambda ref, timeout: {
                "status": "native_conversion_complete",
                "optimizer_steps": 0,
                "sha256": "a" * 64,
            },
            cancel=lambda ref, force: calls.append((ref, force)),
            shutdown=lambda: calls.append("disconnected"),
        ),
    )
    if fault:
        with pytest.raises(SystemExit) as exc:
            convert.main()
        assert exc.value.code == 1
    else:
        convert.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == ("failed" if fault else "native_conversion_complete")
    assert calls == ([] if fault else ["attached", ("owned-ref", True), "disconnected"])


def test_exact_image_native_conversion_argument_contract(config):
    import ast
    import logging

    pytest.importorskip("fti.trainers.miles.run_fleet")
    # Actual native source, recipe, and command, but no CUDA initialization.
    plan = convert.compile_conversion(config, relative_to=ROOT)
    argv = convert.native_arguments(plan)
    assert "--fp16" not in argv and "--num-layers" in argv
    assert argv[argv.index("--num-layers") + 1] == "64"
    assert argv[6].endswith("/tools/convert_hf_to_torch_dist.py")
    assert "--optimizer" not in argv and "--no-save-optim" not in argv
    # The converter calls this exact native default function before validation;
    # BF16 is derived from not-fp16, not emitted as a model-profile flag. Execute
    # that source unchanged without importing GPU-only Megatron dependencies.
    source = Path(argv[6]).parents[1] / "miles/backends/megatron_utils/arguments.py"
    assert (
        convert._hash(source) == "07725785aed8545096eaa6c66bbc8c7c5ec51c0dd68b7d7953e22674798d24ed"
    )
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "set_default_megatron_args"
    )
    namespace = {"os": os, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(
        optimizer=None,
        fp16=False,
        seq_length=None,
        vocab_size=248320,
        padded_vocab_size=248320,
        tokenizer_model=plan["model"]["root"],
        tokenizer_type="HuggingFaceTokenizer",
    )
    namespace["set_default_megatron_args"](args)
    assert args.bf16 is True and args.seq_length == 4096


@pytest.mark.parametrize("initialized", [False, True])
def test_main_attach_failure_does_not_start_another_cluster(
    tiny, tmp_path, monkeypatch, capsys, initialized
):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tiny))
    monkeypatch.setattr(sys, "argv", ["convert", "--plan", str(path), "--sha256", digest(tiny)])
    calls = []

    def fail(**kwargs):
        assert kwargs["address"] == "auto"
        raise RuntimeError("private Ray failure")

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=fail,
            is_initialized=lambda: initialized,
            shutdown=lambda: calls.append("disconnected"),
        ),
    )
    with pytest.raises(SystemExit) as result:
        convert.main()
    assert result.value.code == 1
    assert "private" not in capsys.readouterr().out
    assert calls == (["disconnected"] if initialized else [])


def test_module_entrypoint_rejects_tampering(tiny, tmp_path, monkeypatch):
    import runpy

    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tiny))
    monkeypatch.setattr(sys, "argv", ["convert", "--plan", str(path), "--sha256", "bad"])
    monkeypatch.delitem(sys.modules, "training.miles_conversion")
    with pytest.raises(SystemExit) as result:
        runpy.run_module("training.miles_conversion", run_name="__main__")
    assert result.value.code == 1
