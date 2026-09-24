from __future__ import annotations

import copy
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train.jobs import JobsError, digest
from scripts import finalize_qwen38_skyrl_prod11_fast3_launch as finalizer

ROOT = Path(__file__).resolve().parents[1]


def _mock_reviewed_merge_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = json.loads((ROOT / finalizer.LAUNCH_GATE_PATH).read_bytes())
    by_commit = {item["commit"]: item for item in gate["merge_evidence"].values()}
    monkeypatch.setattr(
        finalizer, "_merge_identity", lambda commit: copy.deepcopy(by_commit[commit])
    )

    def reviewed_ancestry(command, *, cwd, check):
        assert command[:3] == ["git", "merge-base", "--is-ancestor"]
        assert command[3] in by_commit
        assert command[4] == "HEAD"
        assert cwd == finalizer.LAUNCHER_ROOT
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finalizer.subprocess, "run", reviewed_ancestry)


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
            "--launch-gate",
            "gate.json",
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
            "--launch-gate",
            "gate.json",
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


@pytest.mark.parametrize("payload", [None, b"{}"])
def test_fast3_finalizer_rejects_missing_or_invalid_gate_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes | None
) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    gate = tmp_path / "gate.json"
    if payload is not None:
        gate.write_bytes(payload)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("git subprocess reached before the Fast3 gate")

    monkeypatch.setattr(finalizer.subprocess, "check_output", forbidden)
    monkeypatch.setattr(finalizer.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unreadable|launch gate"):
        finalizer.finalize(
            SimpleNamespace(
                source_root=tmp_path,
                operation_directory=output,
                launch_gate=gate,
            )
        )


def test_fast3_preflight_binds_exact_packet_to_terminal_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    successor = {"manifest": "derived-from-stage"}
    plan = {"plan": "exact", "data": successor}
    request = {"request": "exact"}
    gate = {"gate": "exact"}
    packet = {
        "identity": finalizer.operator.FAST3_IDENTITY.sealed_mapping(),
        "launch_gate": gate,
        "stage": {"stage": "exact"},
        "stage_launch_result": {"launch": "exact"},
        "plan": plan,
        "request": request,
        "sha256": "sha256:" + "1" * 64,
    }
    launch = {
        "status": "operator_succeeded_and_released",
        "package": {
            "name": finalizer.operator.FAST3_OPERATOR_NAMES["preflight"],
            "packet_sha256": packet["sha256"],
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
    (tmp_path / "SUCCESSOR_MANIFEST.json").write_text('{"private":"ignored"}')
    (tmp_path / "OPERATOR_CREATE.jsonl").write_text("{}\n")
    monkeypatch.setattr(finalizer.operator, "_packet", lambda *_args: packet)
    monkeypatch.setattr(finalizer.prod10_finalizer, "_terminal_preflight", lambda *_args: launch)
    monkeypatch.setattr(
        finalizer.operator_job, "fast3_successor_manifest", lambda *_args, **_kwargs: successor
    )
    monkeypatch.setattr(
        finalizer.operator_job, "fast3_plan_from_successor", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(finalizer.launch_direct, "job_request", lambda *_args, **_kwargs: request)

    assert finalizer._preflight(tmp_path, plan, request, gate) == (launch, successor)

    launch["package"]["packet_sha256"] = "sha256:" + "2" * 64
    (tmp_path / "PREFLIGHT_LAUNCH_RESULT.json").write_text(json.dumps(launch))
    with pytest.raises(ValueError, match="preflight changed"):
        finalizer._preflight(tmp_path, plan, request, gate)


def test_fast3_preflight_rejects_recompiled_stage_plan_drift_before_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    successor = {"manifest": "stage-a"}
    plan = {"data": successor, "plan": "reviewed"}
    request = {"request": "reviewed"}
    gate = {"gate": "exact"}
    packet = {
        "identity": finalizer.operator.FAST3_IDENTITY.sealed_mapping(),
        "launch_gate": gate,
        "stage": {"stage": "exact"},
        "stage_launch_result": {"launch": "exact"},
        "plan": plan,
        "request": request,
    }
    (tmp_path / "PREFLIGHT_PACKET.json").write_text("{}")
    monkeypatch.setattr(finalizer.operator, "_packet", lambda *_args: packet)
    monkeypatch.setattr(
        finalizer.operator_job, "fast3_successor_manifest", lambda *_args, **_kwargs: successor
    )
    monkeypatch.setattr(
        finalizer.operator_job,
        "fast3_plan_from_successor",
        lambda *_args, **_kwargs: {"data": {"manifest": "plan-b"}, "plan": "reviewed"},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal/provider seam reached before stage-plan equality")

    monkeypatch.setattr(finalizer.prod10_finalizer, "_terminal_preflight", forbidden)
    with pytest.raises(ValueError, match="preflight changed"):
        finalizer._preflight(tmp_path, plan, request, gate)


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


def test_fast3_append_only_launch_gate_closes_only_merged_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_reviewed_merge_identities(monkeypatch)
    parent_path = ROOT / "configs/qualification/qwen38-rl-reward-canary-port-v9.json"
    gate_path = ROOT / finalizer.LAUNCH_GATE_PATH
    parent = json.loads(parent_path.read_bytes())
    plan = {
        "qualification": {
            "qualification_self_sha256": parent["sha256"],
            "submission_gate": parent["submission_gate"],
            "fast3": {"qualification": parent},
        }
    }
    gate = finalizer._launch_gate(gate_path, plan)

    assert parent["submission_gate"] == {
        "blockers": [
            "fast3_source_pr_not_merged",
            "fast3_launch_chain_not_separately_bound",
        ],
        "preview_authorized": False,
        "submission_authorized": False,
    }
    assert gate["submission_gate"] == {
        "blockers": [],
        "preview_authorized": True,
        "submission_authorized": True,
    }
    assert gate["merge_evidence"]["source"]["commit"] == finalizer.SOURCE_MERGE_HEAD
    assert gate["merge_evidence"]["launcher"]["commit"] == finalizer.LAUNCHER_MERGE_HEAD
    assert gate["preserved_identity"]["runtime_evidence"]["port_successor_sha256"] == (
        "sha256:38b2bfd0fa425c45a28cea50d488f4b9c57466cb7ffdeed952a349ca8d323f68"
    )
    assert gate["preserved_identity"]["reload_source"]["file_sha256"] == (
        "sha256:46c9ea1daeb6965b85f0eed8ab9dd2c50a13ef68916092b0cba306367031ee84"
    )
    assert gate["preserved_identity"]["failure_diagnostic"] == {
        "file_sha256": finalizer.DIAGNOSTIC_FILE_SHA256,
        "path": (
            "../../docs/evidence/qwen38-study/"
            "2026-09-23-skyrl-prod11-generation-failure-diagnostic-v1.json"
        ),
        "self_sha256": finalizer.DIAGNOSTIC_SELF_SHA256,
    }
    assert gate["preserved_identity"]["fast2_retirement"] == {
        "file_sha256": finalizer.FAST2_RETIREMENT_FILE_SHA256,
        "path": (
            "../../docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-fast2-retirement-v1.json"
        ),
        "self_sha256": finalizer.FAST2_RETIREMENT_SELF_SHA256,
    }


def test_fast3_source_head_is_exact_accepted_merge_before_git_access() -> None:
    with pytest.raises(ValueError, match="accepted Fast3 source merge"):
        finalizer._source_head(ROOT, "0" * 40)


def test_fast3_append_only_launch_gate_rejects_plan_or_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_reviewed_merge_identities(monkeypatch)
    parent = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-port-v9.json").read_bytes()
    )
    plan = {
        "qualification": {
            "qualification_self_sha256": parent["sha256"],
            "submission_gate": parent["submission_gate"],
            "fast3": {"qualification": parent},
        }
    }
    changed = copy.deepcopy(plan)
    changed["qualification"]["submission_gate"]["preview_authorized"] = True
    with pytest.raises(ValueError, match="append-only launch authorization"):
        finalizer._launch_gate(ROOT / finalizer.LAUNCH_GATE_PATH, changed)

    gate = json.loads((ROOT / finalizer.LAUNCH_GATE_PATH).read_bytes())
    gate["submission_gate"]["blockers"] = ["invented"]
    temporary = tmp_path / "changed-gate.json"
    temporary.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="append-only launch"):
        finalizer._launch_gate(temporary, plan)


def test_fast3_launch_gate_precedes_every_external_finalizer_action() -> None:
    source = inspect.getsource(finalizer.finalize)
    gate = source.index("launch_gate = _launch_gate")
    jobs = source.index("token = os.environ")
    gpu = source.index("source_preview, expected, gpu_previews, provenance")
    assert gate < jobs < gpu


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
