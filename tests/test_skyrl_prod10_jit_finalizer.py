from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest

from cyber_post_train.jobs import JobsError
from scripts import finalize_qwen38_skyrl_prod10_launch as finalizer


def _at(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_jit_finalizer_requires_full_runtime_and_capacity_margin() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    proof = {"checked_at": _at(now - timedelta(seconds=30))}
    capacity = {"observed_at": _at(now - timedelta(seconds=30))}
    result = finalizer.validate_remaining_ttl(
        gpu_previews=[proof, proof],
        gpu_duplicate=proof,
        operator_previews=[proof, proof],
        operator_duplicate=proof,
        capacity=capacity,
        now=now,
    )
    assert result["runtime_proof_remaining_seconds"] == 270
    assert result["capacity_proof_remaining_seconds"] == 90

    stale_runtime = {"checked_at": _at(now - timedelta(seconds=121))}
    with pytest.raises(ValueError, match="runtime proof margin"):
        finalizer.validate_remaining_ttl(
            gpu_previews=[stale_runtime, proof],
            gpu_duplicate=proof,
            operator_previews=[proof, proof],
            operator_duplicate=proof,
            capacity=capacity,
            now=now,
        )

    stale_capacity = {"observed_at": _at(now - timedelta(seconds=111))}
    with pytest.raises(ValueError, match="capacity proof margin"):
        finalizer.validate_remaining_ttl(
            gpu_previews=[proof, proof],
            gpu_duplicate=proof,
            operator_previews=[proof, proof],
            operator_duplicate=proof,
            capacity=stale_capacity,
            now=now,
        )


def test_jit_finalizer_orders_slow_work_before_late_proofs_and_one_create() -> None:
    source = inspect.getsource(finalizer.finalize)
    assert source.index("duplicate = launch_direct.duplicate_proof") < source.index(
        "gpu_previews = ["
    )
    assert source.index("gpu_previews = [") < source.index(
        "operator_previews = operator_launch.server_previews"
    )
    assert source.index("operator_previews = operator_launch.server_previews") < source.index(
        "margin = validate_remaining_ttl"
    )
    assert source.index("margin = validate_remaining_ttl") < source.index("if not args.create:")
    assert source.count("operator_launch.create_once(") == 1
    assert (
        finalizer.parser()
        .parse_args(
            [
                "--source-root",
                "/tmp/source",
                "--source-head",
                "a" * 40,
                "--operation-directory",
                "/tmp/output",
                "--capacity",
                "/tmp/capacity",
                "--manifest-result",
                "/tmp/manifest",
                "--preflight-journal",
                "/tmp/preflight",
            ]
        )
        .create
        is False
    )


def test_jit_finalizer_rejects_symlink_operation_directory_before_other_work(
    tmp_path,
) -> None:
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
            "--operation-directory",
            str(alias),
            "--capacity",
            str(tmp_path / "missing-capacity"),
            "--manifest-result",
            str(tmp_path / "missing-manifest"),
            "--preflight-journal",
            str(tmp_path / "missing-preflight"),
        ]
    )
    with pytest.raises(JobsError, match="operation directory"):
        finalizer.finalize(args)
