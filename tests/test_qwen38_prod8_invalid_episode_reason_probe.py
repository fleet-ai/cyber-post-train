from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError, digest
from scripts import probe_qwen38_prod8_invalid_episode_reason as probe
from training import rl_episode
from training import skyrl_reward_rayjob as direct


def _rendered() -> dict:
    value = copy.deepcopy(probe.manifest())
    uid = "12345678-1234-4123-8123-123456789abc"
    labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": probe.NAME,
        "controller-uid": uid,
        "job-name": probe.NAME,
    }
    value["metadata"].update(
        {"creationTimestamp": "2026-09-21T12:00:00Z", "generation": 1, "uid": uid}
    )
    value["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
    )
    value["spec"]["template"]["metadata"]["labels"] = labels
    pod = value["spec"]["template"]["spec"]
    pod.update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    pod["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    value["status"] = {}
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _native_failure() -> dict:
    body = {
        "plan_sha256": probe.TRAINING_PLAN_SHA256,
        "causes": [
            {
                "error_class": "InvalidEpisode",
                "actor_init_failed": False,
                "remote_error_classes": ["ExceptionGroup"],
                "remote_frames": [
                    {"file": "skyrl_rollout.py", "line": 272, "function": "generate"}
                ],
                "local_frames": [{"file": "skyrl_training.py", "line": 692, "function": "main"}],
            }
        ],
    }
    return {**body, "sha256": digest(body)}


def _reseal_native(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": digest(body)}


def _episode_failure(*reasons: str) -> dict:
    causes = [
        {
            "error_type": "InvalidEpisode",
            "frames": [{"file": "rl_episode.py", "line": 100, "function": "_failure"}],
            "reason": reason,
        }
        for reason in reasons
    ]
    return {
        "error_type": "ExceptionGroup",
        "elapsed_seconds": 1.0,
        "run_id": "PRIVATE-RUN-ID-MUST-NOT-LEAK",
        "phase": "agent_interaction",
        "causes": causes,
    }


def _receipt(root: Path, receipt: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", probe._runtime()],
        env={
            **os.environ,
            "PROBE_TARGET": str(root),
            "PROBE_RECEIPT_SCHEMA": probe.RECEIPT_SCHEMA,
            "PROBE_TRAINING_PLAN_SHA256": probe.TRAINING_PLAN_SHA256,
            "PROBE_RECEIPT_PATH": str(receipt),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout)["status"] in {
        "classified",
        "ambiguous",
        "no_safe_reason",
        "unaccepted",
    }
    return json.loads(receipt.read_bytes())


def test_safe_reason_codes_are_the_runtime_failure_allowlist() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_episode._failure)))
    allowlists = [
        frozenset(ast.literal_eval(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Set)
        and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in node.elts
        )
    ]
    assert allowlists == [frozenset(probe.SAFE_REASON_CODES)]
    assert "skyrl_batch_failed_no_replacement" not in probe.SAFE_REASON_CODES


def test_manifest_is_read_only_zero_gpu_alert_safe_and_render_validation_gated() -> None:
    value = probe.manifest()
    pod = value["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert value["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert value["spec"]["activeDeadlineSeconds"] == 300
    assert value["spec"]["backoffLimit"] == 0
    assert pod["priorityClassName"] == "c1"
    assert pod["automountServiceAccountToken"] is False
    assert container["volumeMounts"] == [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}]
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert "nvidia.com/gpu" not in json.dumps(value, sort_keys=True)
    validation = probe.preview_evidence(_rendered(), direct.PROD_CONTEXT)
    assert validation["schema"] == probe.RENDER_VALIDATION_SCHEMA
    assert validation["status"] == "validated"
    assert validation["failure_alerts"] == "off" and validation["gpus"] == 0
    assert validation["target_binding"] == probe.TARGET_BINDING
    changed = _rendered()
    changed["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError):
        probe.preview_evidence(changed, direct.PROD_CONTEXT)


def test_server_preview_uses_non_mutating_kubernetes_server_dry_run() -> None:
    observed: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, json.dumps(_rendered()), "")

    proof = probe.server_preview(context=direct.PROD_CONTEXT, runner=runner)
    assert proof["schema"] == probe.SERVER_PREVIEW_SCHEMA
    assert proof["status"] == "passed"
    assert proof["proof_method"] == "kubectl_create_server_dry_run"
    assert proof["submitted"] is False
    assert (
        proof["render_validation_sha256"]
        == probe.preview_evidence(_rendered(), direct.PROD_CONTEXT)["sha256"]
    )
    assert observed["command"] == [
        "kubectl",
        "--context",
        direct.PROD_CONTEXT,
        "--namespace",
        direct.NAMESPACE,
        "create",
        "--dry-run=server",
        "-f",
        "-",
        "-o",
        "json",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert json.loads(str(kwargs["input"])) == probe.manifest()
    assert kwargs["capture_output"] is True and kwargs["text"] is True and kwargs["timeout"] == 60
    with pytest.raises(JobsError):
        probe.server_preview(context=direct.DEV_CONTEXT, runner=runner)


def test_runtime_emits_only_one_allowlisted_reason_and_no_private_data(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    _write(
        root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json",
        _episode_failure("generation_transport_failure"),
    )
    (root / "episodes/batches/0123456789abcdef01234567/episode-0/conversation.json").write_text(
        "PRIVATE PROMPT AND CREDENTIAL MUST NOT LEAK"
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "classified"
    assert receipt["target_binding"] == probe.TARGET_BINDING
    assert receipt["deterministic_reason_code"] == "generation_transport_failure"
    assert receipt["safe_reason_codes"] == ["generation_transport_failure"]
    serialized = json.dumps(receipt, sort_keys=True)
    assert str(root) not in serialized
    assert probe.TARGET not in serialized
    assert "PRIVATE" not in serialized
    assert "skyrl_rollout.py" not in serialized
    assert "012345" not in serialized


def test_runtime_refuses_unknown_reason_and_cannot_turn_it_into_evidence(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    _write(
        root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json",
        _episode_failure("private_reason_must_not_escape"),
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["deterministic_reason_code"] is None
    assert "private_reason_must_not_escape" not in json.dumps(receipt, sort_keys=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["causes"][0].__setitem__("remote_error_classes", None),
        lambda value: value["causes"][0].__setitem__(
            "remote_frames", [{"file": "safe.py", "line": "not-an-integer", "function": "f"}]
        ),
    ],
)
def test_runtime_seals_unaccepted_for_malformed_bound_native_failure(
    tmp_path: Path, mutate: object
) -> None:
    root = tmp_path / "run"
    native = _native_failure()
    assert callable(mutate)
    mutate(native)
    _write(root / "NATIVE_FAILURE.json", _reseal_native(native))
    _write(
        root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json",
        _episode_failure("generation_transport_failure"),
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["root_native_failure_bound"] is False
    assert receipt["safe_reason_codes"] == []


def test_runtime_seals_unaccepted_for_nonfinite_native_failure_json(tmp_path: Path) -> None:
    root = tmp_path / "run"
    native = root / "NATIVE_FAILURE.json"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text('{"plan_sha256":NaN,"causes":[],"sha256":"not-a-digest"}')
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["root_native_failure_bound"] is False


def test_runtime_seals_unaccepted_for_malformed_failure_receipt_frames(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    failure = _episode_failure("generation_transport_failure")
    failure["causes"][0]["frames"] = None
    _write(root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json", failure)
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["safe_reason_codes"] == []


def test_runtime_seals_unaccepted_for_huge_elapsed_integer(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    failure = _episode_failure("generation_transport_failure")
    # This is JSON-valid but would make math.isfinite(int) overflow if the
    # runtime tried to coerce it to float before applying its integer bound.
    failure["elapsed_seconds"] = 10**1000
    _write(root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json", failure)
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["safe_reason_codes"] == []


def test_runtime_rejects_malformed_existing_failure_before_later_safe_reason(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    malformed = root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{")
    _write(
        root / "episodes/batches/0123456789abcdef01234567/episode-1/failure.json",
        _episode_failure("generation_transport_failure"),
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "unaccepted"
    assert receipt["safe_reason_codes"] == []


def test_runtime_reports_multiple_allowlisted_reasons_as_ambiguous(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "NATIVE_FAILURE.json", _native_failure())
    _write(
        root / "episodes/batches/0123456789abcdef01234567/episode-0/failure.json",
        _episode_failure("generation_transport_failure", "tool_parser_contract_invalid"),
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt) == receipt
    assert receipt["status"] == "ambiguous"
    assert receipt["deterministic_reason_code"] is None
    assert receipt["safe_reason_codes"] == [
        "generation_transport_failure",
        "tool_parser_contract_invalid",
    ]


def test_runtime_does_not_read_logs_or_raw_trajectory_files() -> None:
    runtime = probe._runtime()
    for forbidden in (
        "private-skyrl.log",
        "conversation.json",
        "recording.json",
        "reward.json",
        "tool_calls",
        "traceback",
    ):
        assert forbidden not in runtime
