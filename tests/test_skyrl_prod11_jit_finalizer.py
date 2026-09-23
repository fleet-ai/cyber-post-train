from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest

from cyber_post_train.jobs import JobsError
from scripts import finalize_qwen38_skyrl_prod11_launch as finalizer


def _at(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_prod11_jit_finalizer_requires_comfortable_create_margin() -> None:
    now = datetime(2026, 9, 23, 17, 0, tzinfo=UTC)
    runtime = {"checked_at": _at(now - timedelta(seconds=30))}
    capacity = {"observed_at": _at(now - timedelta(seconds=80))}
    result = finalizer.validate_create_margin(
        gpu_previews=[runtime, runtime],
        gpu_duplicate=runtime,
        operator_previews=[runtime, runtime],
        operator_duplicate=runtime,
        capacity=capacity,
        now=now,
    )
    assert result == {
        "checked_at": _at(now),
        "runtime_proof_remaining_seconds": 270,
        "capacity_proof_remaining_seconds": 40,
    }

    with pytest.raises(ValueError, match="JIT create margin"):
        finalizer.validate_create_margin(
            gpu_previews=[runtime, runtime],
            gpu_duplicate=runtime,
            operator_previews=[runtime, runtime],
            operator_duplicate=runtime,
            capacity={"observed_at": _at(now - timedelta(seconds=91))},
            now=now,
        )


def test_prod11_jit_finalizer_orders_proofs_and_exactly_one_create() -> None:
    source = inspect.getsource(finalizer.finalize)
    assert source.index("_reviewed_immutables(") < source.index("with Jobs(token) as client:")
    assert source.index("duplicate = launch_direct.duplicate_proof") < source.index(
        "capacity = _capacity(request)"
    )
    assert source.index("capacity = _capacity(request)") < source.index("gpu_previews = [")
    assert source.index("gpu_previews = [") < source.index(
        "operator_previews = operator_launch.server_previews"
    )
    assert source.index("operator_previews = operator_launch.server_previews") < source.index(
        "margin = validate_create_margin"
    )
    assert source.index("margin = validate_create_margin") < source.index("if not args.create:")
    assert source.count("operator_launch.create_once(") == 1
    parsed = finalizer.parser().parse_args(
        [
            "--source-root",
            "/tmp/source",
            "--source-head",
            "a" * 40,
            "--preflight-directory",
            "/tmp/preflight",
            "--reviewed-packet-directory",
            "/tmp/reviewed",
            "--operation-directory",
            "/tmp/output",
        ]
    )
    assert parsed.create is False


def test_prod11_jit_finalizer_rejects_symlink_output_before_network(tmp_path) -> None:
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
            "--reviewed-packet-directory",
            str(tmp_path / "missing-reviewed"),
            "--operation-directory",
            str(alias),
        ]
    )
    with pytest.raises(JobsError, match="operation directory"):
        finalizer.finalize(args)


def test_prod11_jit_finalizer_binds_reviewed_immutable_digests() -> None:
    assert finalizer.BASE_SOURCE_HEAD == "f7452d7eafb33f2d84b724f07480b58c91826207"
    assert finalizer.PLAN_SHA256.startswith("sha256:f86ca0c9")
    assert finalizer.REQUEST_SHA256.startswith("sha256:954650c1")
    assert finalizer.MANIFEST_SHA256.startswith("sha256:b414749c")
    assert finalizer.OPERATOR_SOURCE_SHA256.startswith("sha256:e5ebe5f3")
    source = inspect.getsource(finalizer.finalize)
    assert "expected != reviewed_manifest" in source
    assert 'proof.get("source_sha256") != OPERATOR_SOURCE_SHA256' in source
