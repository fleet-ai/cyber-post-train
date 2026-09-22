"""Focused native-save-return telemetry and immutable-runtime regressions."""

import base64
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import sft, skyrl_training
from training import sft_checkpoint_telemetry_v1 as telemetry_compiler
from training import sft_runtime_checkpoint_telemetry_v1 as telemetry

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = ROOT / "configs/runs"
BASE_RUNTIME = ROOT / "training/sft_runtime.py"
BASE_COMPILER = ROOT / "training/sft.py"

SEALED_SOURCE_DIGESTS = {
    BASE_COMPILER: "447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d",
    BASE_RUNTIME: "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
}
SEALED_RL_RUNTIME_DIGEST = "a88870f1624adc0ae0328bd804b353cbbbd5ee05aaf59209e85d5c56107cc468"

LEGACY_BROAD_DIGESTS = {
    "qwen38-teacher3k-32k-full-b8-lr3e6-v3.json": (
        "8488f03a65e3d158a46697eefda2751736681e3a503c26ee9c39c63fb130679c",
        "fab27e52d445ddb7ebafeb591c7b33a65222f67e88c7c6fc63d578bac2d104e3",
    ),
    "qwen38-teacher3k-32k-full-b8-lr1e6-v3.json": (
        "ebc385578874a3acd5142020fa800eef4cf3fd66a4724c9846e984e3182858e5",
        "fa397b300e8e76eab94650a3eee1f357c4ebb5647e92391aaea09d0f2a34a2fd",
    ),
    "qwen38-teacher3k-32k-full-b16-lr3e6-v3.json": (
        "f97a18dbbfff030dcfad581984e4fec32960a8f3594c95ef11bb19abce351fe3",
        "80ff4379a9b59c7cdfa9533db743b22dcba0c6c71734ee864fc2453c0f70425e",
    ),
    "qwen38-teacher3k-64k-full-b8-lr3e6-v5.json": (
        "9b2e930486e199a16b9b4f6f47960f5655e803f07d2c01a89b9ed474c073fdc0",
        "d8957b7da151ad86fcfd5296754729aedc7eac5ffd00a710560246ebc2cfaae9",
    ),
}


def test_sealed_compiler_and_runtime_sources_remain_byte_identical():
    for path, expected in SEALED_SOURCE_DIGESTS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert digest(skyrl_training._runtime()) == SEALED_RL_RUNTIME_DIGEST


def _compile_config(name: str, *, successor: bool = False) -> dict:
    path = RUN_CONFIGS / name
    config = sft.read_mapping(path)
    if successor:
        run_name = "synthetic-native-save-return-v1"
        config["name"] = run_name
        config["output_root"] = f"/mnt/sfs/jobs/{run_name}"
        config["runtime_variant"] = telemetry.RUNTIME_VARIANT
        config["wandb"] = {
            **config["wandb"],
            "run_id": run_name,
            "name": run_name,
            "tags": [*config["wandb"]["tags"], telemetry.RUNTIME_VARIANT],
        }
    compiler = telemetry_compiler.compile_sft if successor else sft.compile_sft
    return compiler(config, relative_to=path.parent)


def _bundle(request: dict) -> dict:
    environment = request["env"]
    encoded = environment.get("CYBER_SFT_BUNDLE")
    if encoded is None:
        parts = sorted(
            (
                (int(name.rsplit("_", 1)[1]), value)
                for name, value in environment.items()
                if name.startswith("CYBER_SFT_BUNDLE_")
            )
        )
        encoded = "".join(value for _, value in parts)
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


@pytest.mark.parametrize(("name", "expected"), LEGACY_BROAD_DIGESTS.items())
def test_existing_broad_plans_and_requests_keep_their_exact_digests(name, expected):
    plan = _compile_config(name)

    assert "runtime_variant" not in plan
    assert (digest(plan), digest(sft.job_request(plan))) == expected
    assert plan["runtime_sha256"] == hashlib.sha256(BASE_RUNTIME.read_bytes()).hexdigest()


def test_successor_binds_and_stages_new_wrapper_without_rewriting_base_runtime():
    plan = _compile_config(
        "qwen38-teacher3k-32k-full-b8-lr3e6-v3.json",
        successor=True,
    )
    request = telemetry_compiler.job_request(plan)
    bundle = _bundle(request)
    binding = plan["runtime_variant"]
    wrapper = ROOT / "training/sft_runtime_checkpoint_telemetry_v1.py"

    assert binding == telemetry.runtime_binding(BASE_RUNTIME)
    assert plan["runtime_sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest()
    assert plan["runtime_sha256"] != binding["base_runtime_sha256"]
    assert hashlib.sha256(bundle["runtime"].encode()).hexdigest() == plan["runtime_sha256"]
    assert (
        hashlib.sha256(bundle["extra_files"]["training/sft_runtime.py"].encode()).hexdigest()
        == binding["base_runtime_sha256"]
    )
    assert request["env"]["PYTHONPATH"] == plan["output_root"] + "/.runtime"


class _Logger:
    def __init__(self, *, fail: bool = False):
        self.events = []
        self.fail = fail

    def log(self, *, data, commit):
        if self.fail:
            raise RuntimeError("synthetic telemetry outage")
        self.events.append((dict(data), commit))


class _Tracker:
    def __init__(self, *, fail: bool = False):
        self.logger = _Logger(fail=fail)
        self.events = []

    def log(self, data, step, commit):
        self.events.append((dict(data), step, commit))


class _NativeTrainer:
    def __init__(self, output: Path, *, periodic: bool, tracker_fails: bool = False):
        self.output = output
        self.periodic = periodic
        self.global_step = 0
        self.tracker = _Tracker(fail=tracker_fails)
        self.audit_saw_pending = False

    def save_checkpoint(self):
        checkpoint = self.output / "checkpoints" / f"global_step_{self.global_step}"
        (checkpoint / "policy").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"abc")
        (checkpoint / "policy/model.bin").write_bytes(b"12345")
        return str(checkpoint)

    def _fire(self, event_name: str, **fields):
        if event_name == "on_log":
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "optimizer_step": self.global_step,
                            "time": 123.0,
                            **fields["logs"],
                        }
                    )
                    + "\n"
                )

    def train(self):
        self.global_step = 4
        self.save_checkpoint()
        if self.periodic:
            logs = {"train/loss": 1.0}
            self._fire("on_log", logs=logs)
            self.tracker.log(logs, step=self.global_step, commit=True)
        return "complete"


class _AuditedTrainer(_NativeTrainer):
    def save_checkpoint(self):
        path = super().save_checkpoint()
        self.audit_saw_pending = bool(telemetry._pending_metrics(self))
        return path


def _trainer(output: Path, *, periodic: bool, tracker_fails: bool = False):
    trainer_class = telemetry.telemetry_trainer_class(_AuditedTrainer)
    return trainer_class(output, periodic=periodic, tracker_fails=tracker_fails)


def _assert_metrics(metrics: dict, *, stat_errors: int = 0) -> None:
    assert set(metrics) == set(telemetry.METRIC_KEYS)
    assert metrics[telemetry.METRIC_KEYS[0]] == 2.5
    assert metrics[telemetry.METRIC_KEYS[1]] == (0 if stat_errors else 8)
    assert metrics[telemetry.METRIC_KEYS[2]] == (0 if stat_errors else 2)
    assert metrics[telemetry.METRIC_KEYS[3]] == stat_errors
    assert all(type(value) in {int, float} for value in metrics.values())


def test_periodic_save_joins_native_return_metrics_to_same_step_log(tmp_path, monkeypatch):
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(ticks))
    trainer = _trainer(tmp_path, periodic=True)

    assert trainer.train() == "complete"

    assert trainer.audit_saw_pending is True
    assert len(trainer.tracker.events) == 1
    metrics = {key: trainer.tracker.events[0][0][key] for key in telemetry.METRIC_KEYS}
    _assert_metrics(metrics)
    assert trainer.tracker.events[0][1:] == (4, True)
    assert trainer.tracker.logger.events == []
    local = json.loads((tmp_path / "metrics.jsonl").read_text())
    _assert_metrics({key: local[key] for key in telemetry.METRIC_KEYS})


def test_final_only_save_explicitly_flushes_without_a_later_log(tmp_path, monkeypatch):
    ticks = iter((20.0, 22.5))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(ticks))
    trainer = _trainer(tmp_path, periodic=False)

    assert trainer.train() == "complete"

    assert trainer.tracker.events == []
    assert len(trainer.tracker.logger.events) == 1
    metrics, commit = trainer.tracker.logger.events[0]
    _assert_metrics(metrics)
    assert commit is True
    local = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert local["optimizer_step"] == 4
    _assert_metrics({key: local[key] for key in telemetry.METRIC_KEYS})


def test_filesystem_observation_errors_are_counted_and_nonfatal(tmp_path, monkeypatch):
    ticks = iter((30.0, 32.5))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        telemetry.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(OSError("synthetic stat outage")),
    )
    trainer = _trainer(tmp_path, periodic=True)

    assert trainer.train() == "complete"
    metrics = {key: trainer.tracker.events[0][0][key] for key in telemetry.METRIC_KEYS}
    _assert_metrics(metrics, stat_errors=1)


def test_unexpected_measurement_failure_does_not_change_checkpoint_or_training(
    tmp_path, monkeypatch
):
    ticks = iter((35.0, 37.5))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        telemetry,
        "_observe_checkpoint_at_native_save_return",
        lambda _path: (_ for _ in ()).throw(RuntimeError("synthetic telemetry failure")),
    )
    trainer = _trainer(tmp_path, periodic=True)

    assert trainer.train() == "complete"
    assert (tmp_path / "checkpoints/global_step_4/data.pt").read_bytes() == b"abc"
    assert trainer.tracker.events == [({"train/loss": 1.0}, 4, True)]


def test_terminal_telemetry_output_failures_do_not_mask_training(tmp_path, monkeypatch):
    ticks = iter((40.0, 42.5))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        telemetry.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("synthetic local flush outage")),
    )
    trainer = _trainer(tmp_path, periodic=False, tracker_fails=True)

    assert trainer.train() == "complete"
