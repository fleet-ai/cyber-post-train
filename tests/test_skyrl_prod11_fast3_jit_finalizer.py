from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError
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


def test_fast3_finalizer_mints_identity_last_and_uses_one_create() -> None:
    source = inspect.getsource(finalizer.finalize)
    gpu_proofs = source.index("source_preview, expected, gpu_previews, provenance")
    capacity = source.index("capacity = _capacity")
    identity = source.index("duplicate = launch_direct.host_identity_proof")
    outer_previews = source.index("operator_previews = operator_launch.server_previews")
    margin = source.index("margin = validate_create_margin")
    create_gate = source.index("if not args.create")

    assert gpu_proofs < capacity < identity < outer_previews < margin < create_gate
    assert source.count("operator_launch.create_once(") == 1
    parsed = finalizer.parser().parse_args(
        [
            "--source-root",
            "/tmp/source",
            "--source-head",
            "a" * 40,
            "--preflight-directory",
            "/tmp/preflight",
            "--operation-directory",
            "/tmp/output",
        ]
    )
    assert parsed.create is False


def test_fast3_finalizer_rejects_symlink_output_before_network(tmp_path) -> None:
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
            "--preflight-directory",
            str(tmp_path / "missing-preflight"),
            "--operation-directory",
            str(alias),
        ]
    )
    with pytest.raises(JobsError, match="operation directory"):
        finalizer.finalize(args)


def test_fast3_finalizer_binds_identity_science_and_retry_policy() -> None:
    assert finalizer.BASE_SOURCE_HEAD == "58f81904a7478fd90dfe988d97b8fa831fbeadc1"
    assert finalizer.RUN_NAME == "chris-q38-rlreward-prod11-fast3"
    assert finalizer.OUTPUT_ROOT == "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast3"
    source = inspect.getsource(finalizer.finalize)
    for assertion in (
        'get("eval_before_train") is not False',
        'get("steps") != 1',
        'get("groups") != 1',
        'get("samples_per_prompt") != 8',
        'request.get("workers") != 1',
        'request.get("gpus_per_worker") != 8',
        'request.get("failureAlerts") is not False',
        'retry.get("max_http_attempts") != 3',
        'retry.get("backoff_seconds") != [1, 2]',
        'retry.get("whole_episode_retry") is not False',
        'retry.get("admitted_response") != "first_2xx_json_only"',
        '"host_sfs_absence_claimed": False',
        '"runtime_jit_sfs_absence_required": True',
    ):
        assert assertion in source


def test_fast3_finalizer_binds_exact_fast2_retirement(tmp_path) -> None:
    target = tmp_path / finalizer.FAST2_RETIREMENT_PATH
    target.parent.mkdir(parents=True)
    tracked = Path(__file__).resolve().parents[1] / finalizer.FAST2_RETIREMENT_PATH
    target.write_bytes(tracked.read_bytes())
    receipt = finalizer._fast2_retirement(tmp_path)
    assert receipt["gpu_launch"] == {
        "operator_name": "chris-q38-prod11-fast2-launch-operator-v1",
        "outer_created": False,
        "jobs_post_attempted": False,
        "inner_created": False,
        "rayjob_created": False,
        "gpus_requested": 0,
        "retry_same_identity": False,
    }
    assert receipt["output_absence"]["fresh_sfs_lstat_in_this_receipt"] is False

    target.write_text(tracked.read_text().replace('"inner_created":false', '"inner_created":true'))
    with pytest.raises(ValueError, match="retirement evidence"):
        finalizer._fast2_retirement(tmp_path)
