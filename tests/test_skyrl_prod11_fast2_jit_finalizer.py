from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError
from scripts import finalize_qwen38_skyrl_prod11_fast2_launch as finalizer


def _at(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_fast2_finalizer_requires_outer_startup_and_preguard_margin() -> None:
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

    stale = {"checked_at": _at(now - timedelta(seconds=121))}
    with pytest.raises(ValueError, match="admission and preguard"):
        finalizer.validate_create_margin(
            gpu_previews=[runtime, runtime],
            host_identity=stale,
            operator_previews=[runtime, runtime],
            operator_duplicate=runtime,
            capacity=capacity,
            now=now,
        )

    stale_capacity = {"observed_at": _at(now - timedelta(seconds=91))}
    with pytest.raises(ValueError, match="capacity proof lacks create margin"):
        finalizer.validate_create_margin(
            gpu_previews=[runtime, runtime],
            host_identity=runtime,
            operator_previews=[runtime, runtime],
            operator_duplicate=runtime,
            capacity=stale_capacity,
            now=now,
        )


def test_fast2_finalizer_mints_duplicate_last_and_uses_one_create() -> None:
    source = inspect.getsource(finalizer.finalize)
    gpu_proofs = source.index("source_preview, expected, gpu_previews, provenance")
    capacity = source.index("capacity = _capacity")
    duplicate = source.index("duplicate = launch_direct.host_identity_proof")
    operator_previews = source.index("operator_previews = operator_launch.server_previews")
    margin = source.index("margin = validate_create_margin")
    create_gate = source.index("if not args.create")

    assert gpu_proofs < capacity < duplicate < operator_previews < margin < create_gate
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


def test_fast2_finalizer_rejects_symlink_output_before_network(tmp_path) -> None:
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


def test_fast2_finalizer_binds_only_new_identity_and_exact_science() -> None:
    assert finalizer.BASE_SOURCE_HEAD == "3e7958eea4bc3dffe93af0d81fc8ce26f32d9545"
    assert finalizer.RUN_NAME == "chris-q38-rlreward-prod11-fast2"
    assert finalizer.OUTPUT_ROOT == "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2"
    source = inspect.getsource(finalizer.finalize)
    assert 'get("eval_before_train") is not False' in source
    assert 'get("steps") != 1' in source
    assert 'get("groups") != 1' in source
    assert 'get("samples_per_prompt") != 8' in source
    assert 'request.get("workers") != 1' in source
    assert 'request.get("gpus_per_worker") != 8' in source
    assert 'request.get("failureAlerts") is not False' in source
    assert '"host_sfs_absence_claimed": False' in source
    assert '"accepted_preflight_sfs_absence_receipt_sha256"' in source
    assert '"runtime_jit_sfs_absence_required": True' in source


def test_fast2_host_proof_is_truthful_and_runtime_keeps_sfs_gate(monkeypatch) -> None:
    identity = finalizer.historical.load_identity(
        Path(__file__).resolve().parents[1] / finalizer.IDENTITY_PATH
    )
    monkeypatch.setattr(
        finalizer.launch_direct.direct,
        "_direct_duplicate_checks",
        lambda *_args, **_kwargs: {
            "kubernetes_inventories_checked": 10,
            "jobs_api_rows_checked": 0,
        },
    )
    proof = finalizer.launch_direct.host_identity_proof(identity, token="token")
    assert set(proof) == {
        "schema",
        "status",
        "identity_sha256",
        "run_name",
        "kubernetes_inventories_checked",
        "jobs_api_rows_checked",
        "checked_at",
        "sha256",
    }
    assert proof["schema"] == finalizer.launch_direct.HOST_IDENTITY_SCHEMA
    assert proof["status"] == "kubernetes_and_jobs_identity_absent"
    assert "output_root" not in proof
    assert "output_absent" not in proof
    assert finalizer.launch_direct._duplicate(proof, identity, fresh=False) == proof
    assert "require_output_absent" in inspect.getsource(finalizer.launch_direct.jit_duplicate_proof)

    changed = {**proof, "output_absent": True}
    changed["sha256"] = "sha256:" + finalizer.digest(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    with pytest.raises(JobsError, match="duplicate proof changed"):
        finalizer.launch_direct._duplicate(changed, identity, fresh=False)

    fast1 = finalizer.historical.load_identity(
        Path(__file__).resolve().parents[1]
        / "configs/qualification/qwen38-rl-reward-canary-prod11-fast1-identity-v1.json"
    )
    with pytest.raises(JobsError, match="used for another run"):
        finalizer.launch_direct.host_identity_proof(fast1, token="token")


def test_fast2_finalizer_binds_the_retired_fast1_receipt(tmp_path) -> None:
    target = tmp_path / finalizer.FAST1_RETIREMENT_PATH
    target.parent.mkdir(parents=True)
    # The finalizer deliberately resolves the tracked receipt under the reviewed source root.
    tracked = Path(__file__).resolve().parents[1] / finalizer.FAST1_RETIREMENT_PATH
    target.write_bytes(tracked.read_bytes())
    receipt = finalizer._fast1_retirement(tmp_path)
    assert receipt["classification"]["retry_same_identity"] is False
    assert receipt["inner"]["created"] is False

    changed = tracked.read_text().replace('"prod_match_count": 0', '"prod_match_count": 1')
    target.write_text(changed)
    with pytest.raises(ValueError, match="retirement evidence"):
        finalizer._fast1_retirement(tmp_path)
