"""CPU-only contract checks for the pinned Fleet FTI/Miles V1 image.

This file is mounted beside the bundled RL adapter in a short-lived dev Pod.
It deliberately uses fakes for Fleet calls and intercepts the Miles launcher;
no environment, model, GPU, or training process is started.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shlex
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace


def _last_value(tokens: list[str], option: str) -> str:
    values = []
    for index, token in enumerate(tokens):
        if token == option:
            values.append(tokens[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    if not values:
        raise AssertionError(f"missing required option: {option}")
    return values[-1]


def _check_session_contract(adapter) -> dict[str, bool]:
    from fti.miles.v1 import client_recording, common
    from fti.trainers.miles import agent

    signature = inspect.signature(common.TaskSession.grade)
    assert tuple(signature.parameters) == (
        "self",
        "answer",
        "reset_ack",
        "close_final_step",
    )
    assert not inspect.iscoroutinefunction(common.TaskSession.grade)

    recorder_module = adapter._client_recording_with_execution_ids()
    evidence_session_type = recorder_module.TaskSession
    assert evidence_session_type is not common.TaskSession
    assert not inspect.iscoroutinefunction(evidence_session_type.grade)
    evidence_signature = inspect.signature(evidence_session_type.grade)
    assert tuple(evidence_signature.parameters) == tuple(signature.parameters)
    assert {
        name: parameter.default for name, parameter in evidence_signature.parameters.items()
    } == {name: parameter.default for name, parameter in signature.parameters.items()}

    execution_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())

    class GradeClient:
        def execute_verifier(self, *args, **kwargs):
            del args, kwargs
            return {
                "success": True,
                "job_id": execution_id,
                "result": {"result": 0.5},
            }

    grade_session = evidence_session_type.__new__(evidence_session_type)
    grade_session.closed = threading.Event()
    grade_session.instance = SimpleNamespace(instance_id=instance_id)
    grade_session.verifier_version_id = str(uuid.uuid4())
    grade_session.client = GradeClient()
    grade_session.conversation = []
    grade_session.cfg = SimpleNamespace(
        pass_conversation_to_verifier=False,
        grade_timeout_s=30,
    )
    grade_session.task_key = "preflight-task"
    grade_session.task_version_id = str(uuid.uuid4())
    grade_session.cleanup_error = None
    grade_session.verifier_execution_id = None
    grade = grade_session.grade(None, None, False)
    details = grade_session.details()
    assert grade.reward == 0.5
    assert details["verifier_execution_id"] == execution_id

    run_agent_source = inspect.getsource(agent.run_agent)
    open_at = run_agent_source.index("session.open")
    messages_at = run_agent_source.index("build_messages")
    grade_at = run_agent_source.index("session.grade")
    return_at = run_agent_source.rindex("return EpisodeResult")
    assert open_at < messages_at < grade_at < return_at

    generate_source = inspect.getsource(client_recording.generate)
    episode_at = generate_source.index("await run_episode")
    metadata_at = generate_source.index("episode_metadata(session")
    finalize_at = generate_source.index("recorder.finalize")
    assert episode_at < metadata_at < finalize_at

    return {
        "grade_is_synchronous": True,
        "verifier_id_survives_details": True,
        "grade_precedes_metadata_serialization": True,
    }


def _check_prompt_binding(adapter, plan: dict) -> dict[str, bool]:
    from fti.miles.v1.common import Config, TaskSession
    from fti.trainers.miles.agent import build_messages

    marker = "Exact version-bound task instructions from the Fleet API."
    generic = json.loads(adapter.task_rows(plan).splitlines()[0])["messages"][0]["content"]
    instance_id = str(uuid.uuid4())

    class FakeInstance:
        def __init__(self):
            self.instance_id = instance_id
            self.closed = False

        def wait_ready(self, timeout_s, cancelled):
            assert timeout_s == 600
            assert not cancelled.is_set()

        def list_tools(self):
            return []

        def close(self):
            self.closed = True

    instance = FakeInstance()

    class FakeClient:
        def get_task(self, task_key, task_version_id):
            expected = plan["data"]["tasks"][0]
            assert (task_key, task_version_id) == (
                expected["task_key"],
                expected["task_version_id"],
            )
            return {
                "eval_task_version_id": task_version_id,
                "verifier": {"verifier_version_id": str(uuid.uuid4())},
                "prompt": marker,
                "environment_id": "preflight-env",
                "environment_version_id": str(uuid.uuid4()),
            }

        def create_instance(self, **kwargs):
            assert kwargs["env_key"] == "preflight-env"
            assert kwargs["async_provision"] is True
            return instance

    task = plan["data"]["tasks"][0]
    session = TaskSession.__new__(TaskSession)
    session.task_key = task["task_key"]
    session.task_version_id = task["task_version_id"]
    session.cfg = Config(**plan["episode"])
    session.client = FakeClient()
    session.conversation = []
    session.instance = None
    session.closed = threading.Event()
    session.lock = threading.Lock()
    session.deleted = False
    session.cleanup_error = None
    session.tools = []
    session.instructions = ""
    session.verifier_version_id = None
    session.open()
    messages = build_messages(session.instructions, 160, 1, 1)
    assert session.instructions == marker
    assert generic != marker
    assert all(generic not in str(message.get("content")) for message in messages)
    assert any(marker in str(message.get("content")) for message in messages)
    session.close()
    assert instance.closed
    return {
        "exact_task_version_requested": True,
        "api_prompt_replaces_dataset_placeholder": True,
    }


def _parse_run_fleet_cli(run_fleet, argv: list[str]):
    import typer
    from typer.testing import CliRunner

    captured = []
    original_prepare = run_fleet.prepare
    original_execute = run_fleet.execute
    run_fleet.prepare = lambda args: (_ for _ in ()).throw(
        AssertionError("preflight must not prepare a model")
    )
    run_fleet.execute = captured.append
    try:
        app = typer.Typer()
        app.command()(run_fleet.main)
        result = CliRunner().invoke(app, argv)
        if result.exit_code != 0:
            raise AssertionError(
                "native run_fleet CLI rejected the constructed arguments"
            ) from result.exception
        assert len(captured) == 1
        return captured[0]
    finally:
        run_fleet.prepare = original_prepare
        run_fleet.execute = original_execute


def _check_launcher_arguments(adapter, plan: dict) -> dict[str, bool]:
    from fti.trainers.miles import run_fleet

    assert adapter.file_sha256(Path("/opt/fleet/run.sh")) == adapter.RUN_SH_SHA256
    assert adapter.file_sha256(Path(run_fleet.__file__)) == adapter.RUN_FLEET_SHA256
    assert os.system("bash -n /opt/fleet/run.sh") == 0

    # This is the exact argv assembled by run.sh for its final run_fleet exec:
    # launcher-owned defaults first, then the payload arguments, whose repeated
    # values intentionally win.
    cli = [
        "--skip-prepare",
        "--run-id",
        plan["name"],
        "--dataset-dir",
        "/preflight/data",
        "--model-dir",
        "/mnt/sfs/miles-fleet/models",
        "--data-dir",
        "/preflight",
        "--output-dir",
        "/mnt/sfs/miles-fleet",
        *adapter.native_arguments(plan)[1:],
    ]
    script_args = _parse_run_fleet_cli(run_fleet, cli)
    assert script_args.skip_prepare is True
    assert script_args.model_name == "qwen3.8-27b-256k"
    assert script_args.platform == "v1"
    assert script_args.mode == "normal"
    assert script_args.num_nodes == 4
    assert script_args.num_gpus_per_node == 8
    assert script_args.rollout_batch_size == 1
    assert script_args.n_samples_per_prompt == 8
    assert script_args.output_dir == "/mnt/sfs/jobs"

    secret_markers = {
        "WANDB_API_KEY": "preflight-wandb-" + uuid.uuid4().hex,
        "FLEET_API_KEY": "preflight-fleet-" + uuid.uuid4().hex,
    }
    previous_secrets = {name: os.environ.get(name) for name in secret_markers}
    os.environ.update(secret_markers)
    captured = {}
    original_execute_train = run_fleet.U.execute_train
    run_fleet.U.execute_train = lambda **kwargs: captured.update(kwargs)
    try:
        run_fleet.execute(script_args)
    finally:
        run_fleet.U.execute_train = original_execute_train
    tokens = shlex.split(captured["train_args"])
    expected = {
        "--num-rollout": "1",
        "--save-interval": "1",
        "--lr": "1e-06",
        "--seed": "20260913",
        "--rollout-seed": "20260913",
        "--custom-generate-function-path": "training.fti_v1_training.generate",
        "--wandb-team": "thefleet",
        "--wandb-project": "cyber-post-train",
        "--wandb-group": plan["name"],
        "--wandb-run-id": plan["name"],
        "--input-key": "messages",
        "--rollout-batch-size": "1",
        "--n-samples-per-prompt": "8",
        "--global-batch-size": "8",
        "--rollout-max-context-len": "262144",
    }
    assert all(_last_value(tokens, key) == value for key, value in expected.items())
    assert "--use-wandb" in tokens
    assert "--wandb-key" not in tokens
    assert all(value not in captured["train_args"] for value in secret_markers.values())

    # Exercise the exact command renderer with execution intercepted. This
    # proves mounted secret values are absent not only from train_args, but
    # also from every shell command that the launcher would print or execute.
    rendered_commands = []
    original_exec_command_cpu = run_fleet.U.exec_command_cpu
    original_check_has_nvlink = run_fleet.U.check_has_nvlink
    previous_external_ray = os.environ.get("MILES_SCRIPT_EXTERNAL_RAY")
    os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = "1"
    run_fleet.U.exec_command_cpu = lambda command, *args, **kwargs: rendered_commands.append(
        command
    )
    run_fleet.U.check_has_nvlink = lambda: True
    try:
        original_execute_train(
            train_args=captured["train_args"],
            config=script_args,
            num_gpus_per_node=8,
            megatron_model_type="qwen3.8-27B",
            extra_env_vars={},
        )
    finally:
        run_fleet.U.exec_command_cpu = original_exec_command_cpu
        run_fleet.U.check_has_nvlink = original_check_has_nvlink
        if previous_external_ray is None:
            os.environ.pop("MILES_SCRIPT_EXTERNAL_RAY", None)
        else:
            os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = previous_external_ray
        for name, value in previous_secrets.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert rendered_commands
    assert all(
        marker not in command for command in rendered_commands for marker in secret_markers.values()
    )

    # Confirm every custom option is registered by the exact installed Miles
    # parser source. This is intentionally source-level: calling parse_args()
    # performs CUDA/model semantic validation, which is not a CPU preflight.
    import miles

    miles_root = Path(miles.__file__).resolve().parent
    parser_text = "\n".join(
        path.read_text(errors="ignore")
        for path in miles_root.rglob("*.py")
        if "tests" not in path.parts
    )
    for option in expected:
        assert option in parser_text, f"Miles parser does not register {option}"

    return {
        "run_sh_syntax_and_digest": True,
        "run_fleet_cli_parsed": True,
        "last_option_wins": True,
        "wandb_uses_environment_auth_without_cli_secret": True,
        "mounted_secret_values_absent_from_rendered_commands": True,
        "wandb_and_custom_generate_flags_registered": True,
    }


def _check_sanitized_evidence_spool(adapter, plan: dict) -> dict[str, bool]:
    from miles.rollout.base_types import GenerateFnOutput
    from miles.utils.types import Sample

    task = plan["data"]["tasks"][0]
    verifier_version_id = str(uuid.uuid4())
    verifier_execution_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())
    fleet = {
        "graded": True,
        "reward": 0.5,
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "verifier_version_id": verifier_version_id,
        "verifier_execution_id": verifier_execution_id,
        "instance_id": instance_id,
        "cleanup_error": None,
    }
    samples = [
        Sample(metadata={"fleet_v1": dict(fleet)}, reward=0.5),
        Sample(metadata={"fleet_v1": dict(fleet)}, reward=0.5),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        evidence_dir = root / "evidence"
        evidence_dir.mkdir()
        key_path = root / "reward.key"
        key_path.write_bytes(b"k" * 32)
        previous = {
            name: os.environ.get(name)
            for name in (
                "CYBER_PLAN_SHA256",
                "CYBER_REWARD_EVIDENCE_DIR",
                "CYBER_REWARD_HMAC_KEY",
            )
        }
        os.environ.update(
            {
                "CYBER_PLAN_SHA256": adapter.digest(plan),
                "CYBER_REWARD_EVIDENCE_DIR": str(evidence_dir),
                "CYBER_REWARD_HMAC_KEY": str(key_path),
            }
        )
        try:
            output = GenerateFnOutput(samples=samples)
            adapter._spool_reward_evidence(output, task)
            adapter._spool_reward_evidence(output, task)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        paths = list(evidence_dir.iterdir())
        assert [path.name for path in paths] == [verifier_execution_id + ".json"]
        serialized = paths[0].read_text()
        receipt = json.loads(serialized)
        receipt_sha256 = receipt.pop("receipt_sha256")
        assert receipt_sha256 == adapter.digest(receipt)
        assert receipt == {
            "schema": adapter.EVIDENCE_SCHEMA,
            "plan_sha256": adapter.digest(plan),
            "task_key": task["task_key"],
            "task_version_id": task["task_version_id"],
            "verifier_version_id": verifier_version_id,
            "verifier_execution_id": verifier_execution_id,
            "reward_class": "finite_numeric",
            "reward_fingerprint": receipt["reward_fingerprint"],
            "instance_cleanup_confirmed": True,
        }
        assert len(receipt["reward_fingerprint"]) == 64
        assert "0.5" not in serialized
        assert instance_id not in serialized
    return {
        "reward_evidence_create_once": True,
        "multi_segment_evidence_idempotent": True,
        "reward_value_and_instance_id_not_spooled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    from training import fti_v1_training as adapter

    plan = json.loads(Path(args.plan).read_text())
    assert adapter.digest(plan) == args.sha256
    checks = {
        **_check_session_contract(adapter),
        **_check_prompt_binding(adapter, plan),
        **_check_launcher_arguments(adapter, plan),
        **_check_sanitized_evidence_spool(adapter, plan),
    }
    receipt = {
        "schema": "cyber_fti_v1_exact_image_preflight_v1",
        "status": "passed",
        "gpu_count": 0,
        "image": adapter.IMAGE,
        "plan_sha256": args.sha256,
        "checks": checks,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
