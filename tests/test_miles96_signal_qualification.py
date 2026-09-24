from __future__ import annotations

import asyncio
import base64
import gzip
import json
import subprocess
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cyber_post_train.jobs import validate_request
from training import miles96_mechanics_canary as mechanics
from training import miles96_mechanics_launch as launch
from training import miles96_signal_qualification as signal
from training import miles_signal_wave


def _valid_plan() -> dict:
    wave = miles_signal_wave.load()
    candidate = wave["candidates"][0]
    authority = miles_signal_wave.task_binding(wave, candidate)
    return signal.build_plan(
        name=candidate["identity"]["name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
        task_binding={
            **authority,
            "authority_receipt_sha256": candidate["authority_receipt_sha256"],
        },
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=candidate["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )


def _bundle(request: dict) -> dict:
    values = request["env"]
    if "CYBER_RUNTIME_BUNDLE" in values:
        encoded = values["CYBER_RUNTIME_BUNDLE"]
    else:
        encoded = "".join(
            value
            for _, value in sorted(
                (int(key.rsplit("_", 1)[1]), item)
                for key, item in values.items()
                if key.startswith("CYBER_RUNTIME_BUNDLE_")
                and key.removeprefix("CYBER_RUNTIME_BUNDLE_").isdigit()
                for value in (item,)
            )
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def _preflight(plan: dict) -> dict:
    body = {
        "schema": signal.RUNTIME_PREFLIGHT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "image": mechanics.IMAGE,
        "fti_v1_sha256": "sha256:" + signal.FTI_V1_SHA256,
        "miles_eval_sha256": "sha256:" + signal.MILES_INFERENCE_EVAL_SHA256,
        "sample_index_start": 0,
        "sample_index_end": 7,
        "sample_count": 8,
        "max_concurrent_envs": 2,
        "shielded_close": True,
        "release_absence_http_status": 404,
        "raw_tool_catalog_sha256": plan["tool_contract"]["raw_tool_catalog_sha256"],
        "openai_tool_catalog_sha256": plan["tool_contract"]["openai_tool_catalog_sha256"],
        "tool_transform_source_sha256": plan["tool_contract"]["transform_source_sha256"],
        "single_raw_tool_read_capture": True,
        "live_raw_tool_catalog_gate_at_session_open": True,
        "live_tool_schema_gate_at_session_open": True,
        "outer_episode_replacements": 0,
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


def test_signal_request_is_one_node_alert_off_eval_only() -> None:
    plan = _valid_plan()
    request = signal.job_request(plan)
    validate_request(request)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["fleet-api"]
    assert "wandb-api" not in request["secrets"]
    assert launch._expected_request(plan, request, None) == request

    argv = signal.native_arguments(plan)
    assert argv[argv.index("--mode") + 1] == "eval"
    assert argv[argv.index("--n-samples-per-prompt") + 1] == "8"
    assert argv[argv.index("--rollout-batch-size") + 1] == "1"
    extra = argv[argv.index("--extra-args") + 1]
    assert "generate_with_evidence" in extra
    assert "post_save_hook" not in extra
    assert "save-interval" not in extra
    row = json.loads(signal.task_rows(plan))
    assert row["metadata"]["fleet"]["max_concurrent_envs"] == 2
    assert len({item["slot_id"] for item in plan["qualification"]["sample_slots"]}) == 8
    assert [item["sample_index"] for item in plan["qualification"]["sample_slots"]] == list(
        range(8)
    )
    bundle = _bundle(request)
    assert set(bundle["files"]) >= {
        "model.json",
        "plan.json",
        "training/miles96_mechanics_canary.py",
        "training/miles96_signal_qualification.py",
        "training/miles_signal_wave.py",
        "configs/qualification/qwen38-miles-signal-wave-v1.json",
    }


def test_phase1_runtime_bundle_validates_closed_sources_in_isolated_interpreter(
    tmp_path: Path,
) -> None:
    bundle = _bundle(signal.job_request(_valid_plan()))
    runtime = tmp_path / "runtime"
    for name, content in bundle["files"].items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(runtime)!r});"
        "from training import miles96_signal_qualification as s;"
        f"s.validate_plan(json.load(open({str(runtime / 'plan.json')!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    wave_path = runtime / "configs/qualification/qwen38-miles-signal-wave-v1.json"
    wave_path.write_text(wave_path.read_text() + "\n")
    changed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode != 0
    assert "custom runtime source manifest drift" in changed.stderr


def _episode(plan: dict, index: int, reward: float, attempt_id: str) -> dict:
    slot = plan["qualification"]["sample_slots"][index]
    body = {
        "schema": mechanics.PRIVATE_EPISODE_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "task_key": plan["task_binding"]["task_key"],
        "task_version_id": plan["task_binding"]["task_version_id"],
        "verifier_version_id": plan["task_binding"]["verifier_version_id"],
        "sample_index": index,
        "slot_id": slot["slot_id"],
        "attempt_id": attempt_id,
        "instance_id": str(uuid.UUID(int=index + 1)),
        "verifier_execution_id": str(uuid.UUID(int=index + 101)),
        "reward": reward,
        "done_reason": "submitted",
        "tool_calls": 3,
        "cleanup_confirmed": True,
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


def _claim(plan: dict, index: int, attempt_id: str) -> dict:
    slot = plan["qualification"]["sample_slots"][index]
    body = {
        "schema": signal.PRIVATE_CLAIM_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "sample_index": index,
        "slot_id": slot["slot_id"],
        "attempt_number": 1,
        "attempt_id": attempt_id,
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


def _attempt(plan: dict, index: int, attempt_id: str, stage: str) -> dict:
    slot = plan["qualification"]["sample_slots"][index]
    body = {
        "schema": signal.PRIVATE_ATTEMPT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "slot_id": slot["slot_id"],
        "attempt_number": 1,
        "attempt_id": attempt_id,
        "stage": stage,
        "instance_id": str(uuid.UUID(int=index + 1)) if stage != "initialized" else None,
        "cleanup_confirmed": stage == "closed",
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


def _slot(
    plan: dict,
    index: int,
    attempt_id: str,
    episode: dict | None,
) -> dict:
    slot = plan["qualification"]["sample_slots"][index]
    body = {
        "schema": signal.PRIVATE_SLOT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "slot_id": slot["slot_id"],
        "attempt_number": 1,
        "attempt_id": attempt_id,
        "instance_id": str(uuid.UUID(int=index + 1)),
        "verifier_execution_id": (str(uuid.UUID(int=index + 101)) if episode is not None else None),
        "accepted_episode_receipt_sha256": episode["sha256"] if episode is not None else None,
        "cleanup_confirmed": True,
        "status": "accepted" if episode is not None else "infra_invalid_excluded",
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


def _localize(plan: dict, tmp_path: Path) -> dict:
    value = json.loads(json.dumps(plan))
    value["identity"]["run_dir"] = str(tmp_path / value["identity"]["name"])
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _populate(
    plan: dict,
    *,
    completed: int,
    vary: bool = True,
) -> Path:
    run_dir = Path(plan["identity"]["run_dir"])
    private = run_dir / mechanics.PRIVATE_EVIDENCE_DIR
    private.mkdir(parents=True, mode=0o700)
    preflight = _preflight(plan)
    _write(run_dir / signal.RUNTIME_PREFLIGHT_FILE, preflight)
    for index in range(signal.SAMPLES):
        attempt_id = str(uuid.UUID(int=index + 201))
        claim = _claim(plan, index, attempt_id)
        _write(private / f"claim-{claim['slot_id']}.json", claim)
        for stage in ("initialized", "opened", "closed"):
            event = _attempt(plan, index, attempt_id, stage)
            _write(private / f"signal-{attempt_id}-{stage}.json", event)
        episode = (
            _episode(plan, index, float(index % 2) if vary else 0.0, attempt_id)
            if index < completed
            else None
        )
        if episode is not None:
            _write(private / f"{episode['sha256'].removeprefix('sha256:')}.json", episode)
        terminal = _slot(plan, index, attempt_id, episode)
        _write(private / f"slot-{terminal['slot_id']}-1.json", terminal)
    body = {
        "schema": signal.NATIVE_TERMINAL_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "status": "succeeded",
        "returncode": 0,
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "runtime_source_manifest_sha256": "sha256:" + mechanics.digest(plan["runtime_sources"]),
        "runtime_bundle_sha256": "sha256:" + "a" * 64,
        "request_binding_sha256": "sha256:" + "b" * 64,
        "runtime_preflight_sha256": preflight["sha256"],
    }
    _write(
        run_dir / signal.NATIVE_TERMINAL_FILE,
        {**body, "sha256": "sha256:" + mechanics.digest(body)},
    )
    return private


@pytest.mark.parametrize("completed", [2, 8])
def test_signal_aggregator_accepts_exact_terminal_varied_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed: int
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    Path(plan["identity"]["run_dir"]).mkdir(parents=True)
    monkeypatch.setattr(signal, "validate_plan", lambda value: value)
    private = _populate(plan, completed=completed)

    receipt = signal.aggregate(plan)
    assert receipt["schema"] == mechanics.TASK_SIGNAL_EVIDENCE_SCHEMA
    assert receipt["completed_episode_count"] == completed
    assert receipt["planned_slot_count"] == receipt["terminal_slot_count"] == 8
    assert receipt["excluded_slot_count"] == 8 - completed
    assert receipt["optimizer_steps"] == 0
    assert receipt["checkpoint_artifacts_absent"] is True
    assert receipt["reward_variation"] is True
    assert receipt["all_instances_released"] is True
    public = json.dumps(receipt)
    assert '"rewards"' not in public
    assert "instance_id" not in public
    assert "verifier_execution_id" not in public
    group = json.loads((private / signal.PRIVATE_GROUP_FILE).read_text())
    assert len(group["admitted_episode_bindings"]) == completed


def test_signal_aggregator_rejects_no_variation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    monkeypatch.setattr(signal, "validate_plan", lambda value: value)
    _populate(plan, completed=8, vary=False)
    receipt = signal.aggregate(plan)
    assert receipt["schema"] == signal.REJECTION_SCHEMA
    assert receipt["reward_variation"] is False


def test_context_full_episode_cannot_qualify_as_normal_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    monkeypatch.setattr(signal, "validate_plan", lambda value: value)
    private = _populate(plan, completed=8)
    episode_path = next(path for path in private.glob("*.json") if len(path.stem) == 64)
    value = json.loads(episode_path.read_text())
    value["done_reason"] = "context_full"
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + mechanics.digest(body)
    _write(episode_path, value)
    with pytest.raises(ValueError, match="episode evidence is invalid"):
        signal.aggregate(plan)


def test_signal_aggregator_refuses_incomplete_or_cross_swapped_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    monkeypatch.setattr(signal, "validate_plan", lambda value: value)
    private = _populate(plan, completed=8)
    (private / f"slot-{plan['qualification']['sample_slots'][-1]['slot_id']}-1.json").unlink()
    with pytest.raises(ValueError, match="terminally accounted"):
        signal.aggregate(plan)
    assert not (Path(plan["identity"]["run_dir"]) / signal.REJECTION_FILE).exists()

    other = _localize(_valid_plan(), tmp_path / "swap")
    private = _populate(other, completed=8)
    first = private / f"slot-{other['qualification']['sample_slots'][0]['slot_id']}-1.json"
    value = json.loads(first.read_text())
    value["verifier_execution_id"] = str(uuid.UUID(int=102))
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + mechanics.digest(body)
    _write(first, value)
    receipt = signal.aggregate(other)
    assert receipt["schema"] == signal.REJECTION_SCHEMA


def test_signal_plan_rejects_optimizer_or_authority_drift() -> None:
    plan = _valid_plan()
    plan["qualification"]["optimizer_steps"] = 1
    with pytest.raises(ValueError, match="zero-update"):
        signal.validate_plan(plan)

    plan = _valid_plan()
    plan["selection_authority"]["current_binding_sha256"] = plan["task_binding"][
        "authority_receipt_sha256"
    ]
    with pytest.raises(ValueError, match="frozen wave authority"):
        signal.validate_plan(plan)

    for field in (
        "raw_tool_catalog_sha256",
        "openai_tool_catalog_sha256",
        "transform_source_sha256",
    ):
        plan = _valid_plan()
        plan["tool_contract"][field] = "sha256:" + "f" * 64
        with pytest.raises(ValueError, match="tool contract drift"):
            signal.validate_plan(plan)


def test_generate_exception_seals_unaccepted_slot_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    Path(plan["identity"]["run_dir"]).mkdir(parents=True)
    monkeypatch.setattr(mechanics, "_load_runtime_plan", lambda: plan)
    monkeypatch.setattr(signal, "_signal_session_class", lambda: object)

    client = types.ModuleType("fti.miles.v1.client_recording")

    async def fail(_input):
        raise asyncio.CancelledError

    client.generate = fail
    v1 = types.ModuleType("fti.miles.v1")
    v1.client_recording = client
    monkeypatch.setitem(sys.modules, "fti", types.ModuleType("fti"))
    monkeypatch.setitem(sys.modules, "fti.miles", types.ModuleType("fti.miles"))
    monkeypatch.setitem(sys.modules, "fti.miles.v1", v1)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(signal.generate_with_evidence(NS(sample=NS(index=0))))
    private = Path(plan["identity"]["run_dir"]) / mechanics.PRIVATE_EVIDENCE_DIR
    claims = list(private.glob("claim-*.json"))
    terminals = list(private.glob("slot-*.json"))
    assert len(claims) == len(terminals) == 1
    assert json.loads(terminals[0].read_text())["status"] == "unaccepted_stop"
    with pytest.raises(FileExistsError):
        asyncio.run(signal.generate_with_evidence(NS(sample=NS(index=0))))


@pytest.mark.parametrize(
    ("returncode", "write_checkpoint", "expected_status", "error"),
    [
        (3, False, "native_nonzero", subprocess.CalledProcessError),
        (0, True, "zero_update_assertion_failed", ValueError),
    ],
)
def test_run_seals_sanitized_native_rejection_before_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    write_checkpoint: bool,
    expected_status: str,
    error: type[BaseException],
) -> None:
    plan = _localize(_valid_plan(), tmp_path)
    run_dir = Path(plan["identity"]["run_dir"])
    run_dir.mkdir(parents=True)
    plan_sha = mechanics.digest(plan)
    env = {
        "RUN_DIR": str(run_dir),
        "MODEL_DIR": plan["prepared_model"]["root"],
        "FLEET_MODEL_DIR": str(run_dir / "model-output"),
        "CYBER_PLAN_SHA256": plan_sha,
        "CYBER_MODEL_BINDING_SHA256": plan["prepared_model"]["binding_sha256"],
        "CYBER_TASK_BINDING_SHA256": "sha256:" + mechanics.digest(plan["task_binding"]),
        "CYBER_RUNTIME_SOURCE_MANIFEST_SHA256": "sha256:"
        + mechanics.digest(plan["runtime_sources"]),
        "CYBER_RUNTIME_BUNDLE_SHA256": "sha256:" + "a" * 64,
        "CYBER_REQUEST_BINDING_SHA256": "sha256:" + "b" * 64,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(signal, "validate_plan", lambda value: value)
    monkeypatch.setattr(signal, "_prepared_model_exists", lambda _plan: None)
    preflight: list[bool] = []
    monkeypatch.setattr(
        signal,
        "_runtime_signal_binding",
        lambda _plan: (preflight.append(True), _preflight(plan))[1],
    )

    def native(_argv, **_kwargs):
        if write_checkpoint:
            checkpoint = run_dir / "model-output/checkpoints/unexpected.bin"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not-zero-update")
        return NS(returncode=returncode, args=["native"])

    monkeypatch.setattr(signal.subprocess, "run", native)
    with pytest.raises(error):
        signal.run(plan, plan_sha)
    assert preflight == [True]
    native_receipt = json.loads((run_dir / signal.NATIVE_TERMINAL_FILE).read_text())
    rejection = json.loads((run_dir / signal.REJECTION_FILE).read_text())
    assert native_receipt["status"] == rejection["status"] == expected_status
    assert rejection["native_terminal_sha256"] == native_receipt["sha256"]
    assert native_receipt["optimizer_steps"] is None
    assert native_receipt["checkpoint_artifacts_absent"] is (not write_checkpoint)
    assert "error" not in json.dumps(rejection).lower()
