from __future__ import annotations

import copy
import hashlib
import json
import shlex
import sys
import types
import uuid
from pathlib import Path

import pytest

from cyber_post_train.jobs import validate_request
from training import miles96_phase1a_qualifier as phase1a


def test_manifest_proves_exact_current_train_task_and_contract() -> None:
    manifest = phase1a.load_manifest()
    plan = phase1a.build_plan()

    assert manifest["sha256"] == phase1a.MANIFEST_SHA256
    assert plan["task"]["key"] == phase1a.EXPECTED_TASK_KEY
    assert plan["task"]["version_id"] == phase1a.EXPECTED_TASK_VERSION
    assert plan["train_authority"]["role"] == "train"
    assert plan["train_authority"]["component_id"] == phase1a.EXPECTED_COMPONENT
    assert plan["qualification"]["samples"] == 8
    assert plan["qualification"]["terminal_slots_required"] == 8
    assert plan["qualification"]["outer_episode_replacements"] == 0
    assert plan["qualification"]["minimum_completed_gradeable_episodes"] == 2
    assert plan["qualification"]["minimum_distinct_finite_rewards"] == 2
    assert plan["qualification"]["optimizer_steps"] == 0
    assert len(plan["qualification"]["sample_slots"]) == 8
    assert len(set(plan["qualification"]["sample_slots"])) == 8
    assert (
        phase1a._task_authority_receipt(manifest)
        == manifest["train_authority"]["task_authority_receipt_sha256"]
    )
    assert (
        "sha256:" + phase1a.digest(phase1a._live_binding(manifest))
        == manifest["train_authority"]["live_binding_sha256"]
    )


def test_task_and_live_authorities_are_recomputed_from_exact_tuple() -> None:
    manifest = phase1a.load_manifest()
    changed = copy.deepcopy(manifest)
    changed["verifier"]["function_name"] = "other"
    assert phase1a._live_binding(changed) != phase1a._live_binding(manifest)
    changed = copy.deepcopy(manifest)
    changed["train_authority"]["task_set_sha256"] = "sha256:" + "0" * 64
    assert phase1a._task_authority_receipt(changed) != phase1a._task_authority_receipt(manifest)


def _live_task_fixture(monkeypatch) -> tuple[dict, dict]:
    manifest = copy.deepcopy(phase1a.load_manifest())
    prompt, variables, output_schema = "test prompt", {"MODE": "test"}, {"type": "object"}
    manifest["task"].update(
        {
            "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
            "env_variables_sha256": "sha256:" + phase1a.digest(variables),
            "output_json_schema_sha256": "sha256:" + phase1a.digest(output_schema),
        }
    )
    response = {
        "id": manifest["task"]["id"],
        "key": manifest["task"]["key"],
        "eval_task_version_id": manifest["task"]["version_id"],
        "environment_version_id": manifest["environment"]["version_id"],
        "task_lifecycle_status": "production",
        "environment_id": manifest["environment"]["id"],
        "version": manifest["environment"]["version"],
        "seed_config": {
            "only": {
                "env_key": manifest["environment"]["id"],
                "data_key": manifest["environment"]["data_id"],
                "data_version": manifest["environment"]["data_version"],
            }
        },
        "prompt": prompt,
        "env_variables": variables,
        "output_json_schema": output_schema,
        "metadata": {
            "runtime_seed_manifest": {
                "content_sha256": manifest["environment"]["runtime_seed_content_sha256"]
            },
            "cyber_contract": phase1a.CYBER_CONTRACT,
        },
        "verifier_id": manifest["verifier"]["id"],
        "verifier": {
            "verifier_version_id": manifest["verifier"]["version_id"],
            "version": manifest["verifier"]["version"],
            "sha256": manifest["verifier"]["sha256"],
        },
    }
    manifest["train_authority"]["live_binding_sha256"] = "sha256:" + phase1a.digest(
        phase1a._live_binding(manifest)
    )
    monkeypatch.setattr(phase1a, "load_manifest", lambda _root=None: manifest)
    return manifest, response


def test_immediate_live_task_response_must_rebuild_full_binding(monkeypatch) -> None:
    manifest, response = _live_task_fixture(monkeypatch)
    receipt = phase1a.validate_live_task_response(response)
    assert receipt["live_binding_sha256"] == "sha256:" + phase1a.digest(
        phase1a._live_binding(manifest)
    )
    assert receipt["raw_tool_catalog_sha256"] == manifest["tools"]["raw_catalog_sha256"]

    response["verifier"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="live task/env/verifier"):
        phase1a.validate_live_task_response(response)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("task_lifecycle_status", "draft"),
        lambda value: value.__setitem__("eval_task_version_id", str(uuid.uuid4())),
        lambda value: value["metadata"]["runtime_seed_manifest"].__setitem__(
            "content_sha256", "0" * 64
        ),
        lambda value: value["seed_config"]["only"].__setitem__("data_version", "v9"),
        lambda value: value["verifier"].__setitem__("verifier_version_id", str(uuid.uuid4())),
        lambda value: value.__setitem__("prompt", "changed"),
    ],
)
def test_live_task_gate_rejects_each_authority_drift(monkeypatch, mutate) -> None:
    _manifest, response = _live_task_fixture(monkeypatch)
    mutate(response)
    with pytest.raises(ValueError, match="identity/lifecycle|runtime|task/env/verifier"):
        phase1a.validate_live_task_response(response)


def test_submit_collects_and_consumes_one_fresh_exact_live_task_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    _manifest, response = _live_task_fixture(monkeypatch)
    plan = phase1a.build_plan()
    request = phase1a.job_request(plan)
    calls = []

    class Reply:
        status_code = 200

        def __init__(self, value):
            self.value = value

        def json(self):
            return self.value

    class Fleet:
        def get(self, path, params=None):
            calls.append((path, params))
            if path == "/v1/account":
                return Reply(
                    {"team_id": phase1a.FLEET_TEAM_ID, "team_name": "fleet", "extra": True}
                )
            return Reply(response)

    class Jobs:
        def __init__(self):
            self.submission = None

        def submit_once(
            self,
            submitted,
            journal,
            *,
            journal_anchor,
            expected_preview_manifest_sha256,
            before_intent,
            not_after_epoch,
        ):
            self.submission = {
                "request": submitted,
                "journal": journal,
                "anchor": journal_anchor,
                "preview": expected_preview_manifest_sha256,
                "evidence": before_intent({"manifest_sha256": expected_preview_manifest_sha256}),
                "not_after": not_after_epoch,
            }
            return {"name": "exact-created-name", "job_id": str(uuid.uuid4())}

    launch, retained = tmp_path / "launch", tmp_path / "retained"
    launch.mkdir()
    retained.mkdir()
    live = retained / "LIVE_TASK.json"
    journal = launch / "SUBMISSION.jsonl"
    anchor = retained / f"{request['name']}-SUBMISSION.jsonl"
    jobs = Jobs()
    result = phase1a.submit_phase1a(
        plan,
        live,
        journal,
        anchor,
        "a" * 64,
        Fleet(),
        jobs,
        now=lambda: 1000,
    )
    assert result["name"] == "exact-created-name"
    assert calls == [
        ("/v1/account", None),
        (
            "/v1/tasks/" + plan["task"]["key"],
            {"version_id": plan["task"]["version_id"]},
        ),
    ]
    assert jobs.submission["request"] == request
    assert jobs.submission["anchor"] == anchor
    assert jobs.submission["not_after"] == 1900
    assert jobs.submission["evidence"].startswith("sha256:")
    assert live.stat().st_mode & 0o777 == 0o600
    assert b"test prompt" not in live.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        phase1a._read_live_task_receipt(live, plan, request, 1901)
    changed = json.loads(live.read_text())
    changed["raw_tool_catalog_sha256"] = "sha256:" + "0" * 64
    body = {key: value for key, value in changed.items() if key != "sha256"}
    changed["sha256"] = "sha256:" + phase1a.digest(body)
    live.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="reviewed plan"):
        phase1a._read_live_task_receipt(live, plan, request, 1000)


def test_submit_rejects_live_task_drift_before_jobs_submission(tmp_path: Path, monkeypatch) -> None:
    _manifest, response = _live_task_fixture(monkeypatch)
    plan = phase1a.build_plan()
    response["environment_version_id"] = str(uuid.uuid4())

    class Reply:
        status_code = 200

        def __init__(self, value):
            self.value = value

        def json(self):
            return self.value

    class Fleet:
        def get(self, path, params=None):
            del params
            return Reply(
                {"team_id": phase1a.FLEET_TEAM_ID, "team_name": "fleet"}
                if path == "/v1/account"
                else response
            )

    class Jobs:
        def submit_once(
            self,
            request,
            journal,
            *,
            journal_anchor,
            expected_preview_manifest_sha256,
            before_intent,
            not_after_epoch,
        ):
            del (
                request,
                journal,
                journal_anchor,
                expected_preview_manifest_sha256,
                not_after_epoch,
            )
            before_intent({})
            raise AssertionError("Jobs submission must not be reached")

    retained = tmp_path / "retained"
    retained.mkdir()
    live = retained / "LIVE_TASK.json"
    with pytest.raises(ValueError, match="identity/lifecycle"):
        phase1a.submit_phase1a(
            plan,
            live,
            tmp_path / "SUBMISSION.jsonl",
            retained / f"{plan['identity']['name']}-SUBMISSION.jsonl",
            "a" * 64,
            Fleet(),
            Jobs(),
            now=lambda: 1000,
        )
    assert not live.exists()


@pytest.mark.parametrize(
    ("document", "mutate"),
    [
        (
            "configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json",
            lambda value: value["tasks"][28].__setitem__("split", "dev"),
        ),
        (
            "configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json",
            lambda value: value["components"][76].__setitem__("inherited_role", "dev"),
        ),
        (
            "configs/data/qwen38-skyrl-production-task-set-v1.json",
            lambda value: value["tasks"][32].__setitem__("env_version", "v0.0.2"),
        ),
    ],
)
def test_train_authority_body_drift_is_rejected(tmp_path: Path, document: str, mutate) -> None:
    root = Path(phase1a.__file__).resolve().parents[1]
    for path in phase1a.runtime_source_files():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / path).read_bytes())
    value = json.loads((tmp_path / document).read_text())
    mutate(value)
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + phase1a.digest(body)
    (tmp_path / document).write_text(json.dumps(value))

    with pytest.raises(ValueError, match="authority source seal changed"):
        phase1a.load_manifest(tmp_path)


def test_plan_is_closed_world() -> None:
    plan = phase1a.build_plan()
    for path, replacement in (
        (("identity", "name"), "other"),
        (("qualification", "samples"), 7),
        (("qualification", "optimizer_steps"), 1),
        (("task", "version_id"), "00000000-0000-4000-8000-000000000000"),
    ):
        changed = copy.deepcopy(plan)
        changed[path[0]][path[1]] = replacement
        with pytest.raises(ValueError, match="sole reviewed task7317"):
            phase1a.validate_plan(changed)


def test_native_argv_is_exact_eval_only_sampling_contract(monkeypatch) -> None:
    plan = phase1a.build_plan()
    argv = phase1a.native_arguments(plan)
    assert argv[1:3] == ["-m", "fti.trainers.miles.run_fleet"]
    assert argv[argv.index("--mode") + 1] == "eval"
    assert argv[argv.index("--num-gpus-per-node") + 1] == "8"
    assert argv[argv.index("--n-samples-per-prompt") + 1] == "8"
    extra = shlex.split(argv[argv.index("--extra-args") + 1])
    assert extra[-6:] == ["--temperature", "1", "--top-p", "1", "--top-k", "-1"]
    assert not ({"--save", "--save-interval", "--optimizer-steps"} & set(argv + extra))

    original = phase1a._native_contract
    monkeypatch.setattr(
        phase1a,
        "_native_contract",
        lambda value: [*original(value), "--optimizer-steps", "1"],
    )
    with pytest.raises(ValueError, match="training/save flag"):
        phase1a.native_arguments(plan)


def test_request_is_one_node_alert_off_c1_zero_retry_runtime() -> None:
    plan = phase1a.build_plan()
    request = phase1a.job_request(plan)
    validate_request(request)
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["failureAlerts"] is False
    assert request["privileged"] is True
    assert request["image"] == plan["trainer"]["image"]
    assert request["run_dir"] == plan["identity"]["run_dir"]
    assert "CYBER_RUNTIME_BUNDLE" in " ".join(request["env"])


def _private_receipt(plan: dict, index: int, reward: float | None) -> dict:
    gradeable = reward is not None
    body = {
        "schema": phase1a.PRIVATE_SCHEMA,
        "plan_sha256": "sha256:" + phase1a.digest(plan),
        "sample_index": index,
        "slot_id": plan["qualification"]["sample_slots"][index],
        "terminally_accounted": True,
        "terminal_reason": "normally_completed" if gradeable else "returned_ungradeable",
        "normally_completed_gradeable": gradeable,
        "instance_created": True,
        "instance_id": f"00000000-0000-4000-8000-{index:012d}",
        "verifier_execution_id": (f"10000000-0000-4000-8000-{index:012d}" if gradeable else None),
        "reward": reward,
        "cleanup_confirmed": True,
    }
    return {**body, "sha256": "sha256:" + phase1a.digest(body)}


def _aggregate_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    plan = phase1a.build_plan()
    plan = copy.deepcopy(plan)
    plan["identity"] = {"name": "local", "run_dir": str(tmp_path)}
    private = tmp_path / phase1a.PRIVATE_DIR
    private.mkdir(mode=0o700)
    (tmp_path / "model-output").mkdir(mode=0o700)
    monkeypatch.setattr(phase1a, "validate_plan", lambda value: value)
    monkeypatch.setattr(phase1a, "_private_root", lambda _plan: private)
    phase1a._write_once(
        tmp_path / phase1a.PREFLIGHT_FILE,
        phase1a._preflight_receipt(plan),
    )
    for index in range(8):
        reward = 0.0 if index == 0 else 1.0 if index == 1 else None
        phase1a._write_once(private / f"sample-{index}.json", _private_receipt(plan, index, reward))
    return plan, private


def test_aggregate_accepts_all_eight_released_and_two_distinct_rewards(
    tmp_path: Path, monkeypatch
) -> None:
    plan, _private = _aggregate_fixture(tmp_path, monkeypatch)
    terminal = phase1a.aggregate(plan)
    assert terminal["terminal_slot_count"] == 8
    assert terminal["gradeable_episode_count"] == 2
    assert terminal["distinct_finite_reward_count"] == 2
    assert terminal["all_instances_released"] is True
    assert terminal["outer_episode_replacements"] == 0
    assert terminal["optimizer_steps"] == 0
    assert (tmp_path / phase1a.PUBLIC_FILE).is_file()


def test_no_instance_ungradeable_slot_is_accounted_without_cleanup_claim(
    tmp_path: Path, monkeypatch
) -> None:
    plan, private = _aggregate_fixture(tmp_path, monkeypatch)
    value = json.loads((private / "sample-7.json").read_text())
    value.update(
        {
            "terminal_reason": "runtime_exception",
            "instance_created": False,
            "instance_id": None,
            "cleanup_confirmed": None,
        }
    )
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + phase1a.digest(body)
    (private / "sample-7.json").unlink()
    phase1a._write_once(private / "sample-7.json", value)

    terminal = phase1a.aggregate(plan)
    assert terminal["terminal_slot_count"] == 8
    assert terminal["all_instances_released"] is True


def _returned_output(*, instance_id: str | None = None):
    fleet = {} if instance_id is None else {"instance_id": instance_id}
    sample = types.SimpleNamespace(
        metadata={"fleet_v1": fleet},
        status=types.SimpleNamespace(value="aborted"),
        reward=None,
    )
    return types.SimpleNamespace(samples=[sample])


def _completed_output(instance_id: str, verifier_id: str, reward: float, reason: str):
    sample = types.SimpleNamespace(
        metadata={
            "fleet_v1": {
                "instance_id": instance_id,
                "verifier_execution_id": verifier_id,
                "graded": True,
                "authority_verified": True,
                "cleanup_confirmed": True,
                "done_reason": reason,
                "tool_calls": 1,
            }
        },
        status=types.SimpleNamespace(value="completed"),
        reward=reward,
    )
    return types.SimpleNamespace(samples=[sample])


def test_max_turns_is_ungradeable_and_cannot_qualify_signal(tmp_path: Path, monkeypatch) -> None:
    plan, private = _aggregate_fixture(tmp_path, monkeypatch)
    for index, reward in ((0, 0.0), (1, 1.0)):
        instance_id = f"00000000-0000-4000-8000-{index:012d}"
        verifier_id = f"10000000-0000-4000-8000-{index:012d}"
        state = {
            "sample_index": index,
            "slot_id": plan["qualification"]["sample_slots"][index],
            "instance_id": instance_id,
            "verifier_execution_id": verifier_id,
            "cleanup_confirmed": True,
        }
        token = phase1a._SLOT.set(state)
        try:
            receipt = phase1a._slot_receipt(
                plan,
                index,
                _completed_output(instance_id, verifier_id, reward, "max_turns"),
            )
        finally:
            phase1a._SLOT.reset(token)
        assert receipt["terminal_reason"] == "returned_ungradeable"
        assert receipt["normally_completed_gradeable"] is False
        assert receipt["verifier_execution_id"] is None
        assert receipt["reward"] is None
        path = private / f"sample-{index}.json"
        path.unlink()
        phase1a._write_once(path, receipt)
    with pytest.raises(ValueError, match="reward/release qualification did not pass"):
        phase1a.aggregate(plan)


def test_returned_ungradeable_reconciles_created_instance_from_runtime_state() -> None:
    plan = phase1a.build_plan()
    instance = str(uuid.uuid4())
    state = {
        "sample_index": 7,
        "slot_id": plan["qualification"]["sample_slots"][7],
        "instance_id": instance,
        "cleanup_confirmed": True,
    }
    token = phase1a._SLOT.set(state)
    try:
        receipt = phase1a._slot_receipt(plan, 7, _returned_output())
    finally:
        phase1a._SLOT.reset(token)
    assert receipt["instance_created"] is True
    assert receipt["instance_id"] == instance
    assert receipt["cleanup_confirmed"] is True
    assert receipt["normally_completed_gradeable"] is False


def test_returned_ungradeable_rejects_unreleased_or_mismatched_instance() -> None:
    plan = phase1a.build_plan()
    instance = str(uuid.uuid4())
    state = {
        "sample_index": 7,
        "slot_id": plan["qualification"]["sample_slots"][7],
        "instance_id": instance,
        "cleanup_confirmed": False,
    }
    token = phase1a._SLOT.set(state)
    try:
        with pytest.raises(ValueError, match="not proven released"):
            phase1a._slot_receipt(plan, 7, _returned_output())
        state["cleanup_confirmed"] = True
        with pytest.raises(ValueError, match="instance disagrees"):
            phase1a._slot_receipt(plan, 7, _returned_output(instance_id=str(uuid.uuid4())))
    finally:
        phase1a._SLOT.reset(token)


def test_session_records_instance_before_open_or_base_close_failure(monkeypatch) -> None:
    class FakeTaskSession:
        def __init__(self, client):
            self.client = client
            self.instance = None
            self.verifier_execution_id = None
            self.deleted = False
            self.cleanup_error = None

        def open(self):
            self.instance = self.client.create_reward_instance()
            self.instance.list_tools()

        def close(self):
            raise RuntimeError("base close failed")

    fleet = types.ModuleType("fti.fleet")
    fleet.GradeResult = object
    fleet_v1 = types.ModuleType("fti.fleet.v1")
    fleet_v1.PlatformError = RuntimeError
    fleet_v1.openai_tools = lambda value: value
    common = types.ModuleType("fti.miles.v1.common")
    common.TaskSession = FakeTaskSession
    common.numeric_reward = float
    monkeypatch.setitem(sys.modules, "fti", types.ModuleType("fti"))
    monkeypatch.setitem(sys.modules, "fti.fleet", fleet)
    monkeypatch.setitem(sys.modules, "fti.fleet.v1", fleet_v1)
    monkeypatch.setitem(sys.modules, "fti.miles", types.ModuleType("fti.miles"))
    monkeypatch.setitem(sys.modules, "fti.miles.v1", types.ModuleType("fti.miles.v1"))
    monkeypatch.setitem(sys.modules, "fti.miles.v1.common", common)
    monkeypatch.setattr(phase1a, "_SESSION_CLASS", None)

    instance_id = str(uuid.uuid4())

    class Instance:
        def __init__(self):
            self.instance_id = instance_id

        def list_tools(self):
            raise RuntimeError("tool read failed")

    client = types.SimpleNamespace(create_reward_instance=lambda: Instance())
    state = {"sample_index": 0, "slot_id": "slot"}
    token = phase1a._SLOT.set(state)
    try:
        session = phase1a._session_class()(client)
        with pytest.raises(RuntimeError, match="tool read failed"):
            session.open()
        assert state["instance_id"] == instance_id
        assert state["cleanup_confirmed"] is False
        with pytest.raises(RuntimeError, match="base close failed"):
            session.close()
        assert state["instance_id"] == instance_id
        assert state["cleanup_confirmed"] is False
    finally:
        phase1a._SLOT.reset(token)
        phase1a._SESSION_CLASS = None


@pytest.mark.parametrize("attack", ["missing", "unreleased", "duplicate_verifier", "flat_rewards"])
def test_aggregate_rejects_incomplete_or_untrustworthy_signal(
    tmp_path: Path, monkeypatch, attack: str
) -> None:
    plan, private = _aggregate_fixture(tmp_path, monkeypatch)
    if attack == "missing":
        (private / "sample-7.json").unlink()
    else:
        first = json.loads((private / "sample-0.json").read_text())
        second = json.loads((private / "sample-1.json").read_text())
        target = first if attack == "unreleased" else second
        if attack == "unreleased":
            target["cleanup_confirmed"] = False
        elif attack == "duplicate_verifier":
            target["verifier_execution_id"] = first["verifier_execution_id"]
        else:
            target["reward"] = first["reward"]
        body = {key: value for key, value in target.items() if key != "sha256"}
        target["sha256"] = "sha256:" + phase1a.digest(body)
        path = private / ("sample-0.json" if attack == "unreleased" else "sample-1.json")
        path.unlink()
        phase1a._write_once(path, target)
    with pytest.raises(ValueError):
        phase1a.aggregate(plan)


@pytest.mark.parametrize("attack", ["missing", "tampered", "checkpoint", "symlink"])
def test_aggregate_requires_runtime_preflight_and_checkpoint_absence(
    tmp_path: Path, monkeypatch, attack: str
) -> None:
    plan, _private = _aggregate_fixture(tmp_path, monkeypatch)
    preflight = tmp_path / phase1a.PREFLIGHT_FILE
    if attack == "missing":
        preflight.unlink()
    elif attack == "tampered":
        value = json.loads(preflight.read_text())
        value["image"] = "attacker/image:latest"
        preflight.unlink()
        phase1a._write_once(preflight, value)
    elif attack == "checkpoint":
        checkpoints = tmp_path / "model-output" / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "forbidden.pt").write_text("not allowed")
    else:
        (tmp_path / "model-output" / "checkpoints").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="preflight|checkpoint"):
        phase1a.aggregate(plan)


def test_checkpoint_absence_rejects_files_and_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "model-output"
    output.mkdir()
    assert phase1a._checkpoint_absent(output)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    assert phase1a._checkpoint_absent(output)
    (checkpoints / "tensor.pt").write_text("not allowed")
    assert not phase1a._checkpoint_absent(output)
    (checkpoints / "tensor.pt").unlink()
    (checkpoints / "link").symlink_to(tmp_path)
    assert not phase1a._checkpoint_absent(output)
