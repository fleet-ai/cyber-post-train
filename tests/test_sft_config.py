"""Offline compiler and actual embedded-bootstrap tests: no GPU or network."""

import base64
import copy
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from training import sft

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", [".json", ".yaml", ""])
def test_json_numbers_survive_mapping_readback(tmp_path, suffix):
    value = {"lr": 1e-6, "large": 1e20, "count": 1, "enabled": True, "text": "1e-6"}
    path = tmp_path / ("configuration" + suffix)
    path.write_text(json.dumps(value))
    actual = sft.read_mapping(path)
    assert actual == value
    assert {k: type(v) for k, v in actual.items()} == {k: type(v) for k, v in value.items()}


def test_mapping_still_accepts_yaml_and_rejects_non_objects(tmp_path):
    path = tmp_path / "configuration.yaml"
    path.write_text("# Human-authored configuration\nrecipe:\n  lr: 1.0e-6\n  nodes: 1\n")
    assert sft.read_mapping(path) == {"recipe": {"lr": 1e-6, "nodes": 1}}
    for text in ("[]", "null", "- list\n- not-object"):
        path.write_text(text)
        with pytest.raises(ValueError, match="must be a mapping"):
            sft.read_mapping(path)


@pytest.fixture
def config(tmp_path):
    manifest = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "files": {
            "train": {
                "path": "train.parquet",
                "rows": 17,
                "sha256": "b" * 64,
                "task_keys": ["train"],
            },
            "dev": {
                "path": "dev.parquet",
                "rows": 2,
                "sha256": "c" * 64,
                "task_keys": ["dev-a", "dev-b"],
            },
        },
    }

    def save(value):
        value = {k: v for k, v in value.items() if k != "sha256"}
        value["sha256"] = "sha256:" + digest(value)
        (tmp_path / "corpus.json").write_text(json.dumps(value))

    save(manifest)
    return (
        {
            "name": "sft-compiler-test",
            "output_root": "/mnt/sfs/jobs/sft-compiler-test",
            "model": {
                "lock": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
                "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
                "root": "/mnt/sfs/models/test-base",
            },
            "data": {"manifest": "corpus.json", "root": "/mnt/sfs/datasets/test-corpus"},
            "wandb": {
                "entity": "test-team",
                "project": "test-project",
                "group": "test",
                "run_id": "sft-compiler-test",
                "name": "test",
                "tags": ["synthetic"],
            },
        },
        manifest,
        save,
    )


def test_compile_uses_exact_model_manifest_and_complete_epochs(config, tmp_path):
    source, _, _ = config
    source["recipe"] = {"epochs": 3, "batch_size": 16, "lr": 2e-6}
    plan = sft.compile_sft(source, relative_to=tmp_path)
    assert plan["recipe"]["max_steps"] == 6
    assert plan["recipe"]["lr"] == 2e-6
    assert len(plan["model"]["files"]) == 28
    assert plan["datasets"]["train"]["path"] == "/mnt/sfs/datasets/test-corpus/train.parquet"
    request = sft.job_request(plan)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert "queue_priority_class" not in request
    assert request["secrets"] == ["wandb-api"]
    assert request == sft.job_request(plan)
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert json.loads(content["plan"]) == plan
    assert hashlib.sha256(content["runtime"].encode()).hexdigest() == plan["runtime_sha256"]


def test_compiler_binds_planned_pause_without_shortening_recipe(config, tmp_path):
    source, _, _ = config
    original = sft.compile_sft(source, relative_to=tmp_path)
    source["pause_after_step"] = 1
    paused = sft.compile_sft(source, relative_to=tmp_path)
    assert paused["recipe"] == original["recipe"]
    assert paused["pause_after_step"] == 1
    request = sft.job_request(paused)
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert json.loads(content["plan"])["pause_after_step"] == 1
    assert request["requeueIfPreempted"] is False


@pytest.fixture
def glm_config(config):
    source, manifest, save = config
    model_dir = ROOT / "configs/models/glm53-30333038"
    source["model"].update(
        lock=str(model_dir / "model.lock.json"),
        weights=str(model_dir / "model.weights.json"),
        root="/mnt/sfs/models/glm-5.3-30333038",
    )
    manifest["tokenizer"] = {
        "repo": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
    }
    save(manifest)
    source["lora"] = {"rank": 16, "alpha": 32}
    source["recipe"] = {"nodes": 2, "batch_size": 16}
    source["cluster"] = {"resources": {"memory_request": "2048Gi", "memory_limit": "2304Gi"}}
    return source


def test_glm_compiler_binds_full_base_adapter_and_worker_bundle(glm_config, tmp_path):
    plan = sft.compile_sft(glm_config, relative_to=tmp_path)
    request = sft.job_request(plan)
    assert plan["model"]["repo"] == "zai-org/GLM-5.3"
    assert plan["lora"] == {"rank": 16, "alpha": 32}
    assert request["workers"] == 2 and request["gpus_per_worker"] == 8
    content = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert content["extra_files"]["training/__init__.py"] == ""
    helper = content["extra_files"]["training/glm_runtime.py"]
    assert hashlib.sha256(helper.encode()).hexdigest() == plan["glm_runtime_sha256"]
    assert request["env"]["PYTHONPATH"] == plan["output_root"] + "/.runtime"
    plan["glm_runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="GLM runtime changed"):
        sft.job_request(plan)


@pytest.mark.parametrize("defect", ["rank", "alpha", "missing", "extra", "memory", "nodes", "cpu"])
def test_glm_rejects_unsafe_or_ambiguous_configuration(glm_config, tmp_path, defect):
    if defect == "missing":
        glm_config.pop("lora")
    elif defect == "memory":
        glm_config.pop("cluster")
    elif defect == "nodes":
        glm_config["recipe"]["nodes"] = 1
    elif defect == "cpu":
        glm_config["cluster"]["resources"]["cpu_request"] = "15999m"
    else:
        glm_config["lora"][defect] = 0
    with pytest.raises(ValueError):
        sft.compile_sft(glm_config, relative_to=tmp_path)


def test_glm_bootstrap_imports_worker_in_child_without_checkout(glm_config, tmp_path, monkeypatch):
    plan = sft.compile_sft(glm_config, relative_to=tmp_path)
    helper = Path(sft.__file__).with_name("glm_runtime.py").read_text()
    (tmp_path / "glm_runtime.py").write_text(helper)
    runtime = tmp_path / "sft_runtime.py"
    runtime.write_text(
        "import os,subprocess,sys\nfrom training import glm_runtime\n"
        "assert glm_runtime.__file__.startswith(os.environ['RUN_DIR']), glm_runtime.__file__\n"
        "subprocess.run([sys.executable,'-S','-c',"
        '"from training import glm_runtime; '
        "assert glm_runtime.TARGETS[0]=='q_a_proj'\"],check=True)\n"
    )
    monkeypatch.setattr(sft, "__file__", str(tmp_path / "sft.py"))
    plan["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    request = sft.job_request(plan)
    run = tmp_path / "run"
    run.mkdir()
    argv = shlex.split(request["command"])
    argv[0] = sys.executable
    # Exclude this desktop's editable-install import hook as well as the cwd.
    argv.insert(1, "-S")
    env = {**os.environ, **request["env"], "RUN_DIR": str(run), "PYTHONPATH": str(run / ".runtime")}
    result = subprocess.run(argv, cwd=run, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (run / ".runtime/training/glm_runtime.py").read_text() == helper


def test_recovery_bootstrap_imports_complete_package_without_checkout(
    config, tmp_path, monkeypatch
):
    from training import recovery

    plan = sft.compile_sft(config[0], relative_to=tmp_path)
    # Bootstrap import boundary only; manifest validity has independent tests.
    plan["recovery"] = {"mode": "validate"}
    plan["recovery_runtime_sha256"] = recovery.digest(Path(recovery.__file__))
    request = sft.job_request(plan)
    files = json.loads(gzip.decompress(base64.b64decode(request["env"]["CYBER_SFT_BUNDLE"])))
    assert files["extra_files"]["training/sft_runtime.py"] == files["runtime"]
    # Replace only the executable entrypoint, retaining the real recovery,
    # checkpoint and runtime modules for both parent and fresh child imports.
    entry = (
        "import subprocess,sys\nfrom training import recovery\n"
        "assert callable(recovery.worker_class)\n"
        "subprocess.run([sys.executable,'-S','-c',"
        "'from training import recovery; assert callable(recovery.load)'],check=True)\n"
    )
    original = Path.read_bytes
    runtime_file = Path(sft.__file__).with_name("sft_runtime.py")
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: entry.encode() if p == runtime_file else original(p)
    )
    plan["runtime_sha256"] = hashlib.sha256(entry.encode()).hexdigest()
    request = sft.job_request(plan)
    run = tmp_path / "run"
    run.mkdir()
    argv = shlex.split(request["command"])
    argv[:1] = [sys.executable, "-S"]
    env = {**os.environ, **request["env"], "RUN_DIR": str(run), "PYTHONPATH": str(run / ".runtime")}
    result = subprocess.run(argv, cwd=run, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan["recovery_runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recovery runtime changed"):
        sft.job_request(plan)


@pytest.mark.parametrize(
    "section,key",
    [
        (None, "typo"),
        ("recipe", "learning_rate"),
        ("cluster", "queue_priority_class"),
        ("model", "revision"),
        ("data", "taskset"),
    ],
)
def test_unknown_fields_fail_instead_of_silently_ignoring_overrides(config, tmp_path, section, key):
    source, _, _ = config
    target = source if section is None else source.setdefault(section, {})
    target[key] = "unexpected"
    with pytest.raises(ValueError, match="unknown fields"):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 0),
        ("epochs", 1.5),
        ("nodes", True),
        ("batch_size", 7),
        ("lr", float("nan")),
        ("lr", 1),
        ("checkpoint_interval", 10),
        ("keep_checkpoints", 0),
    ],
)
def test_bad_recipe_fails_before_network(config, tmp_path, key, value):
    source, _, _ = config
    source["recipe"] = {key: value}
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        "unsigned",
        "tokenizer",
        "task_overlap",
        "path_escape",
        "same_file",
        "zero_rows",
        "missing_identity",
    ],
)
def test_corpus_identity_and_split_gates(config, tmp_path, defect):
    source, manifest, save = config
    if defect == "unsigned":
        manifest["sha256"] = "wrong"
        (tmp_path / "corpus.json").write_text(json.dumps(manifest))
    else:
        if defect == "tokenizer":
            manifest["tokenizer"]["revision"] = "a" * 40
        elif defect == "task_overlap":
            manifest["files"]["dev"]["task_keys"] = ["train"]
        elif defect == "path_escape":
            manifest["files"]["train"]["path"] = "../outside"
        elif defect == "same_file":
            manifest["files"]["dev"]["path"] = "train.parquet"
        elif defect == "zero_rows":
            manifest["files"]["train"]["rows"] = 0
        else:
            manifest["files"]["dev"]["sha256"] = ""
        save(manifest)
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "defect", ["revision", "shard", "sidecar", "duplicate", "unsupported_model"]
)
def test_model_must_match_full_file_inventory(config, tmp_path, defect):
    source, manifest, save = config
    lock = sft.read_mapping(Path(source["model"]["lock"]))
    if defect == "revision":
        lock["revision"] = "a" * 40
    elif defect == "shard":
        lock["weights"]["shards"] -= 1
    elif defect == "sidecar":
        lock["configuration"]["config_sha256"] = ""
    elif defect == "duplicate":
        lock["tokenizer"]["files"].append(copy.deepcopy(lock["tokenizer"]["files"][0]))
    else:
        lock["repo"] = "zai-org/GLM-5.3-Flash"
        manifest["tokenizer"]["repo"] = lock["repo"]
        save(manifest)
    (tmp_path / "model.json").write_text(json.dumps(lock))
    source["model"]["lock"] = "model.json"
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


@pytest.mark.parametrize(
    "section,root",
    [
        ("model", "/mnt/sfs"),
        ("data", "/tmp/data"),
        ("data", "/mnt/sfs/datasets/../test"),
        ("model", "/mnt/sfs//models/test"),
    ],
)
def test_roots_are_explicit_and_canonical(config, tmp_path, section, root):
    source, _, _ = config
    source[section]["root"] = root
    with pytest.raises(ValueError):
        sft.compile_sft(source, relative_to=tmp_path)


def test_bootstrap_executes_bound_bytes_and_never_overwrites(config, tmp_path, monkeypatch):
    source, _, _ = config
    plan = sft.compile_sft(source, relative_to=tmp_path)
    fake_module = tmp_path / "compiler.py"
    runtime = tmp_path / "sft_runtime.py"
    runtime.write_text("import sys; print('synthetic-bootstrap', sys.argv[4])\n")
    monkeypatch.setattr(sft, "__file__", str(fake_module))
    with pytest.raises(ValueError, match="runtime changed"):
        sft.job_request(plan)
    plan["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    request = sft.job_request(plan)
    argv = shlex.split(request["command"])
    argv[0] = sys.executable
    output = tmp_path / "run"
    output.mkdir()
    env = {**os.environ, **request["env"], "RUN_DIR": str(output)}
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "synthetic-bootstrap" in result.stdout
    assert (output / ".runtime/sft_runtime.py").read_bytes() == runtime.read_bytes()
    assert subprocess.run(argv, env=env, capture_output=True).returncode != 0
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    env.update(RUN_DIR=str(tampered), CYBER_SFT_BUNDLE=base64.b64encode(b"tampered").decode())
    assert subprocess.run(argv, env=env, capture_output=True).returncode != 0
    assert not (tampered / ".runtime").exists()


@pytest.mark.parametrize("defect", [None, "gpu", "native", "file", "tokens"])
def test_cpu_preflight_checks_files_and_actual_target_accounting(
    config, tmp_path, monkeypatch, defect
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import transformers

    from training import sft_runtime

    source, _, _ = config
    plan = sft.compile_sft(source, relative_to=tmp_path)
    root = tmp_path / "model"
    root.mkdir()
    plan["model"]["root"] = str(root)
    for item in plan["model"]["files"]:
        (root / item["path"]).write_bytes(b"synthetic-file")
        item["sha256"] = hashlib.sha256(b"synthetic-file").hexdigest()
    for split, spec in plan["datasets"].items():
        rows = [
            {
                "messages": [{"role": "assistant", "content": "synthetic"}],
                "task_key": spec["task_keys"][i % len(spec["task_keys"])],
                "window_id": str(i),
                "token_count": 4 if defect != "tokens" else 5,
            }
            for i in range(spec["rows"])
        ]
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        spec.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []

    def native():
        calls.append("native")
        if defect == "native":
            raise ValueError("native source mismatch")

    def tokenize(row, tokenizer, max_length):
        assert max_length is None  # silently truncating is never acceptable
        return {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1] * 4,
            "num_actions": 2,
            "loss_mask": [1, 1],
        }

    monkeypatch.setitem(
        sys.modules, "skyrl.train.sft_trainer", SimpleNamespace(tokenize_chat_example=tokenize)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: defect == "gpu")
    monkeypatch.setattr(sft_runtime, "validate_runtime_sources", native)
    monkeypatch.setattr(sft_runtime, "build_runtime_configs", lambda p: calls.append("config"))

    def local_loader(path, **kwargs):
        assert path == str(root)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        calls.append("local_model")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", local_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", local_loader)
    if defect == "file":
        (root / "config.json").write_bytes(b"tampered")
    if defect:
        with pytest.raises(ValueError):
            sft.preflight(plan)
        if defect in {"gpu", "file"}:
            assert calls == []
    else:
        receipt = sft.preflight(plan)
        assert calls == ["native", "config", "local_model", "local_model"]
        assert receipt["plan_sha256"] == digest(plan)
        assert receipt["request_sha256"] == digest(sft.job_request(plan))
        assert receipt["counts"] == {
            "train": {"rows": 17, "tasks": 1, "supervised_tokens": 34},
            "dev": {"rows": 2, "tasks": 2, "supervised_tokens": 4},
        }
        assert receipt["status"] == "passed" and receipt["gpus"] == 0
