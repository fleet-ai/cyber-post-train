from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train.jobs import JobsError, digest
from scripts import finalize_qwen38_skyrl_prod11_fast3_launch as finalizer


def _at(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_fast3_finalizer_requires_outer_startup_and_preguard_margin() -> None:
    now = datetime(2026, 9, 23, 20, 0, tzinfo=UTC)
    runtime = {"checked_at": _at(now - timedelta(seconds=119))}
    capacity = {"observed_at": _at(now - timedelta(seconds=80))}
    result = finalizer.validate_create_margin(
        gpu_previews=[runtime, runtime],
        host_identity=runtime,
        operator_previews=[runtime, runtime],
        operator_duplicate=runtime,
        capacity=capacity,
        now=now,
    )
    assert result["runtime_proof_remaining_seconds"] == 181
    assert result["capacity_proof_remaining_seconds"] == 40

    with pytest.raises(ValueError, match="admission and preguard"):
        finalizer.validate_create_margin(
            gpu_previews=[runtime, runtime],
            host_identity={"checked_at": _at(now - timedelta(seconds=121))},
            operator_previews=[runtime, runtime],
            operator_duplicate=runtime,
            capacity=capacity,
            now=now,
        )
    with pytest.raises(ValueError, match="capacity proof lacks create margin"):
        finalizer.validate_create_margin(
            gpu_previews=[runtime, runtime],
            host_identity=runtime,
            operator_previews=[runtime, runtime],
            operator_duplicate=runtime,
            capacity={"observed_at": _at(now - timedelta(seconds=91))},
            now=now,
        )


def test_fast3_finalizer_orders_proofs_and_has_one_opt_in_create() -> None:
    source = inspect.getsource(finalizer.finalize)
    gpu = source.index("source_preview, expected, gpu_previews, provenance")
    capacity = source.index("capacity = _capacity")
    identity = source.index("host_identity = launch_direct.host_identity_proof")
    outer = source.index("operator_previews = operator_launch.server_previews")
    margin = source.index("margin = validate_create_margin")
    gate = source.index("if not args.create")
    assert gpu < capacity < identity < outer < margin < gate
    assert source.count("operator_launch.create_once(") == 1
    parsed = finalizer.parser().parse_args(
        [
            "--source-root",
            "/tmp/source",
            "--source-head",
            "a" * 40,
            "--launcher-head",
            "b" * 40,
            "--identity",
            "identity.json",
            "--plan-sha256",
            "sha256:" + "1" * 64,
            "--request-sha256",
            "sha256:" + "2" * 64,
            "--data-manifest-sha256",
            "sha256:" + "3" * 64,
            "--gpu-manifest-sha256",
            "sha256:" + "4" * 64,
            "--preflight-directory",
            "/tmp/preflight",
            "--operation-directory",
            "/tmp/output",
            "--failure-diagnostic",
            "diagnostic.json",
            "--fast2-retirement",
            "retirement.json",
            "--generation-retry-policy",
            "retry.json",
            "--predecessor-science",
            "science.json",
        ]
    )
    assert parsed.create is False
    assert finalizer.BASE_SOURCE_HEAD == "cc07933546abb82023cf0413b4538dcf9d992ed5"


def test_fast3_finalizer_rejects_symlink_output_before_any_provider_call(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    args = finalizer.parser().parse_args(
        [
            "--source-root",
            str(tmp_path / "missing-source"),
            "--source-head",
            "a" * 40,
            "--launcher-head",
            "b" * 40,
            "--identity",
            "identity.json",
            "--plan-sha256",
            "sha256:" + "1" * 64,
            "--request-sha256",
            "sha256:" + "2" * 64,
            "--data-manifest-sha256",
            "sha256:" + "3" * 64,
            "--gpu-manifest-sha256",
            "sha256:" + "4" * 64,
            "--preflight-directory",
            str(tmp_path / "missing-preflight"),
            "--operation-directory",
            str(alias),
            "--failure-diagnostic",
            "diagnostic.json",
            "--fast2-retirement",
            "retirement.json",
            "--generation-retry-policy",
            "retry.json",
            "--predecessor-science",
            "science.json",
        ]
    )
    with pytest.raises(JobsError, match="operation directory"):
        finalizer.finalize(args)


def test_fast3_source_evidence_rejects_in_tree_symlink(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="regular source file"):
        finalizer._source_file(tmp_path.resolve(), alias)


def test_fast3_preflight_binds_exact_packet_to_terminal_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan": "exact"}
    request = {"request": "exact"}
    packet = {
        "identity": finalizer.operator.FAST3_IDENTITY.sealed_mapping(),
        "plan": plan,
        "request": request,
        "sha256": "sha256:" + "1" * 64,
    }
    launch = {
        "status": "operator_succeeded_and_released",
        "package": {
            "name": finalizer.operator.FAST3_OPERATOR_NAMES["preflight"],
            "packet_sha256": "sha256:" + "2" * 64,
        },
        "observer": {
            "terminal_status": "Succeeded",
            "exit_codes": [0],
            "restarts": 0,
            "peak_gpus": 0,
            "receipt": {"status": "passed", "phase": "preflight", "gpus": 0},
        },
    }
    (tmp_path / "PREFLIGHT_PACKET.json").write_text("{}")
    (tmp_path / "PREFLIGHT_LAUNCH_RESULT.json").write_text(json.dumps(launch))
    (tmp_path / "OPERATOR_CREATE.jsonl").write_text("{}\n")
    monkeypatch.setattr(finalizer.operator, "_packet", lambda *_args: packet)
    monkeypatch.setattr(finalizer.prod10_finalizer, "_terminal_preflight", lambda *_args: launch)

    with pytest.raises(ValueError, match="preflight changed"):
        finalizer._preflight(tmp_path, plan, request)


def test_fast3_retry_policy_is_bounded_and_never_retries_an_episode(tmp_path: Path) -> None:
    body = {
        "schema": "cyber_skyrl_generation_http_retry_policy_v1",
        "max_http_attempts": 3,
        "retryable_http_statuses": {"exact": [429], "inclusive_ranges": [[500, 599]]},
        "backoff_seconds": [1, 2],
        "transport_retry": False,
        "follow_redirects": False,
        "whole_episode_retry": False,
        "failed_response_body_admitted": False,
        "failed_response_tokens_admitted": False,
        "admitted_response": "first_2xx_json_only",
    }
    value = {**body, "sha256": "sha256:" + digest(body)}
    path = tmp_path / "retry.json"
    path.write_text(json.dumps(value))
    checked, file_sha, self_sha = finalizer._retry_policy(path)
    assert checked == value
    assert file_sha.startswith("sha256:")
    assert self_sha == value["sha256"]

    changed_body = {**body, "whole_episode_retry": True}
    path.write_text(json.dumps({**changed_body, "sha256": "sha256:" + digest(changed_body)}))
    with pytest.raises(ValueError, match="bounded generation retry"):
        finalizer._retry_policy(path)


def test_fast3_finalizer_requires_eval_before_train_and_truthful_label() -> None:
    source = inspect.getsource(finalizer.finalize)
    assert 'get("trainer.eval_before_train") is not True' in source
    assert '"eval_before_train" in plan.get("arguments", {})' in source
    assert '"successor_class": "bounded_retry_policy_successor_not_exact_runtime_parity"' in source
    assert '"host_sfs_absence_claimed": False' in source
    assert '"runtime_jit_sfs_absence_required": True' in source
    assert '"outer_gpus": 0' in source
    assert '"inner_nodes": 1' in source
    assert '"inner_gpus": 8' in source


def test_fast3_topology_requires_root_alert_off_c1_q1_outer0_inner1x8() -> None:
    expected = {
        "metadata": {
            "annotations": {"fleet.ai/failure-alerts": "off"},
            "labels": {"kueue.x-k8s.io/priority-class": "q1"},
        },
        "spec": {
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "spec": {
                            "priorityClassName": "c1",
                            "containers": [
                                {
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": 8},
                                        "limits": {"nvidia.com/gpu": 8},
                                    }
                                }
                            ],
                        }
                    }
                },
                "workerGroupSpecs": [],
            }
        },
    }
    job = {
        "metadata": {
            "annotations": {"fleet.ai/failure-alerts": "off"},
            "labels": {"kueue.x-k8s.io/priority-class": "q1"},
        },
        "spec": {
            "template": {
                "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}},
                "spec": {"priorityClassName": "c1", "containers": [{"resources": {}}]},
            }
        },
    }
    package = operator_job.OperatorPackage({}, {}, job, {}, b"")
    finalizer._validate_topology(expected, package)
    changed = json.loads(json.dumps(expected))
    changed["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(ValueError, match="alert, priority, or one-node topology"):
        finalizer._validate_topology(changed, package)
