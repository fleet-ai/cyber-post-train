from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import exact_pass4_bulk_runtime_v3 as bulk_runtime
from evals.fleet import exact_pass4_bulk_v3 as bulk
from evals.fleet import exact_pass4_prebulk_reconciliation_package_v3 as package
from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconcile
from evals.fleet import exact_pass4_task_inventory_package as inventory_package

ROOT = Path.cwd()
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"
COLLECTOR_JOB_UID = "33333333-3333-4333-8333-333333333333"
COLLECTOR_POD_UID = "44444444-4444-4444-8444-444444444444"


def test_exact_remaining_cells_are_798_sanitized_unique_identities() -> None:
    rows = reconcile.remaining_cells(ROOT)
    assert len(rows) == 798
    assert len({row["cell_id"] for row in rows}) == 798
    assert len({row["execution_id"] for row in rows}) == 798
    assert len({row["run_id"] for row in rows}) == 798
    assert {row["model"] for row in rows} == {"qwen3.8-27b", "glm-5.3"}
    assert all(
        set(row)
        == {
            "model",
            "controller",
            "selection_rank",
            "attempt",
            "task_version_id",
            "cell_id",
            "execution_id",
            "run_id",
        }
        for row in rows
    )
    assert all(
        forbidden not in json.dumps(rows).lower()
        for forbidden in ("prompt", "trace", "flag", "score")
    )


def test_fixed_evidence_allowlist_maps_only_inventory_g7_terminals_and_claims(
    tmp_path: Path,
) -> None:
    logical = reconcile.fixed_evidence_paths(ROOT)
    assert set(logical) == {
        "inventory_terminal",
        "qwen3.8-27b_terminal",
        "qwen3.8-27b_claim",
        "glm-5.3_terminal",
        "glm-5.3_claim",
    }
    assert len(set(logical.values())) == 5
    for path in logical.values():
        target = tmp_path / path.relative_to(reconcile.SFS_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
        assert reconcile._evidence_path(path, tmp_path) == target.resolve()  # noqa: SLF001
    with pytest.raises(reconcile.ReconciliationError, match="logical_path_unsafe"):
        reconcile._evidence_path(Path("/tmp/not-sfs.json"), tmp_path)  # noqa: SLF001


def test_predecessor_failed_outputs_and_claims_are_rechecked_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.fleet import autocontinue_generation6_canary as generation6
    generation5 = generation6.generation5
    specs = {
        Path("g5-qwen.json"): {"execution": {"execution_id": "sha256:" + "1" * 64}},
        Path("g6-qwen.json"): {"execution": {"execution_id": "sha256:" + "2" * 64}},
    }
    monkeypatch.setattr(generation5, "G5_SPEC_PATHS", {"qwen3.8-27b": Path("g5-qwen.json")})
    monkeypatch.setattr(
        generation5,
        "EXPECTED",
        {"qwen3.8-27b": {"job_name": "chris-qwen-failed-g5"}},
    )
    monkeypatch.setattr(generation5, "validate_spec", lambda spec, root: None)
    monkeypatch.setattr(generation6, "G6_SPEC_PATHS", {"qwen3.8-27b": Path("g6-qwen.json")})
    monkeypatch.setattr(
        generation6,
        "EXPECTED",
        {"qwen3.8-27b": {"job_name": "chris-qwen-failed-g6"}},
    )
    monkeypatch.setattr(generation6, "validate_spec", lambda spec, root: None)
    monkeypatch.setattr(generation5, "CLAIM_ROOT", "/mnt/sfs/claims/test-predecessors")
    monkeypatch.setattr(reconcile, "_load", lambda path: specs[path.relative_to(ROOT)])

    reconcile.validate_predecessor_failure_absence(ROOT, evidence_root=tmp_path)

    failed_root = tmp_path / "jobs" / "chris-qwen-failed-g6"
    failed_root.mkdir(parents=True)
    with pytest.raises(reconcile.ReconciliationError, match="side_effect_appeared"):
        reconcile.validate_predecessor_failure_absence(ROOT, evidence_root=tmp_path)
    failed_root.rmdir()

    failed_claim = tmp_path / "claims" / "test-predecessors" / (("1" * 64) + ".json")
    failed_claim.parent.mkdir(parents=True)
    failed_claim.write_text("{}\n")
    with pytest.raises(reconcile.ReconciliationError, match="side_effect_appeared"):
        reconcile.validate_predecessor_failure_absence(ROOT, evidence_root=tmp_path)


def test_runtime_templates_cross_bind_model_endpoint_harness_and_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconcile.validate_template_bindings(ROOT)
    original = bulk.load
    target = ROOT / bulk.RUNTIME_TEMPLATE_PATHS["qwen3.8-27b"]

    def drifted(path: Path) -> dict:
        value = copy.deepcopy(original(path))
        if path == target:
            value["treatment_block"]["endpoint_origin"] = "https://wrong.invalid"
        return value

    monkeypatch.setattr(bulk, "load", drifted)
    with pytest.raises(reconcile.ReconciliationError, match="cross_field_drifted"):
        reconcile.validate_template_bindings(ROOT)


def test_route_context_contract_rejects_local_drift_before_any_live_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = bulk.load(ROOT / bulk.RUNTIME_TEMPLATE_PATHS["qwen3.8-27b"])
    template["harness"]["compaction_headroom_tokens"] = 20000
    template["treatment"] = copy.deepcopy(bulk.exact.EXPECTED_TREATMENT)
    template["harness"]["context_window_size"] = 131072
    opened = False

    def forbidden_client(key: str) -> object:
        nonlocal opened
        opened = True
        raise AssertionError(key)

    monkeypatch.setattr(bulk_runtime, "_client", forbidden_client)
    with pytest.raises(RuntimeError, match="context binding drifted"):
        bulk_runtime._fresh_route_check(template, "not-a-real-key")  # noqa: SLF001
    assert opened is False


def test_bulk_runtime_hydration_adds_frozen_compaction_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk, "validate_inventory_gate", lambda receipt, root: None)
    inventory = {
        "receipt_sha256": "sha256:" + "a" * 64,
        "tasks": [
            {
                "selection_rank": rank,
                "environment": {"version_id_authority": "immutable_selection"},
                "task": {"version": 1},
                "verifier": {},
            }
            for rank in range(1, 101)
        ],
    }
    plan = bulk.build_runtime_plan("qwen-a", inventory, ROOT)
    assert plan["harness"]["compaction_headroom_tokens"] == 20000
    settings = bulk.self_hosted.opencode_settings(plan)
    assert settings["compaction"] == {"auto": True, "reserved": 20000}
    assert settings["provider"]["fleet-cluster"]["models"]["qwen3.8-27b"]["limit"] == {
        "context": 262144,
        "output": 32768,
        "input": 229376,
    }


def _fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], dict]:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}")
    monkeypatch.setattr(reconcile, "INVENTORY_TERMINAL", inventory_path)
    monkeypatch.setattr(reconcile, "_evidence_path", lambda path, evidence_root: path)
    monkeypatch.setattr(bulk, "validate_inventory_gate", lambda receipt, root: None)
    monkeypatch.setattr(
        reconcile,
        "_load",
        lambda path: {
            "tasks": [{"task": {"key": "t"}}],
            "receipt_sha256": "sha256:" + "0" * 64,
        },
    )
    rows = [
        {
            "model": "qwen3.8-27b",
            "controller": "qwen-a",
            "selection_rank": 1,
            "attempt": 1,
            "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "cell_id": "sha256:" + "1" * 64,
            "execution_id": "sha256:" + "2" * 64,
            "run_id": "planned-run",
        }
    ]
    monkeypatch.setattr(reconcile, "remaining_cells", lambda root: rows)
    plans = {
        "qwen-a": {
            "job_name": "bulk-job",
            "sfs_root": str(tmp_path / "absent-output"),
        }
    }
    monkeypatch.setattr(bulk, "validate_all", lambda root: plans)
    monkeypatch.setattr(bulk, "CLAIM_ROOT", str(tmp_path / "claims"))
    monkeypatch.setattr(
        bulk,
        "_universe",
        lambda root: {"universe_sha256": "sha256:" + "3" * 64},
    )
    release_path = tmp_path / "g7-release.json"
    release_path.write_bytes(b'{\n  "receipt_sha256": "sha256:' + b"8" * 64 + b'"\n}\n')
    terminal_path = tmp_path / "g7-terminal.json"
    terminal_path.write_text("{}\n")
    canary = {
        "model": "qwen3.8-27b",
        "spec": {
            "generation7_spec_sha256": "sha256:" + "9" * 64,
            "statistical_cell": {
                "cell_id": "sha256:" + "4" * 64,
                "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            },
            "execution": {"execution_id": "sha256:" + "5" * 64},
        },
        "plan": {
            "plan_sha256": "sha256:" + "a" * 64,
            "scored_job_name": "g7",
            "attempts": [
                {
                    "run_id": "g7-run",
                    "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                }
            ],
        },
        "claim": {
            "job_uid": JOB_UID,
            "pod_uid": POD_UID,
            "receipt_sha256": "sha256:" + "6" * 64,
        },
        "terminal": {"receipt_sha256": "sha256:" + "7" * 64},
        "terminal_path": terminal_path,
        "logical_terminal_path": Path(
            "/mnt/sfs/jobs/chris-q38-ac-r004-a1-g7-v1/CANARY-TERMINAL.json"
        ),
        "release": {"receipt_sha256": "sha256:" + "8" * 64},
        "release_path": release_path,
        "package_commit": "1" * 40,
        "session_id": "55555555-5555-4555-8555-555555555555",
        "verifier_execution_id": "66666666-6666-4666-8666-666666666666",
    }
    monkeypatch.setattr(reconcile, "G7_OUTPUTS", {"qwen3.8-27b": tmp_path / "g7"})
    monkeypatch.setattr(
        reconcile, "_canary_chain", lambda model, root, evidence_root=None: canary
    )
    monkeypatch.setattr(reconcile, "validate_template_bindings", lambda root: None)
    monkeypatch.setattr(
        reconcile,
        "validate_predecessor_failure_absence",
        lambda root, evidence_root=None: None,
    )
    inventory_package = {
        "raw": b"{}\n",
        "binding": {
            "path": str(reconcile.OUTPUT_ROOT / reconcile.INVENTORY_PACKAGE_FILENAME),
            "file_sha256": bulk.self_hosted.sha256(b"{}\n"),
            "package_sha256": "sha256:" + "b" * 64,
            "package_commit": "1" * 40,
            "bootstrap_configmap": reconcile.INVENTORY_CONFIGMAP,
            "bootstrap_configmap_uid": "77777777-7777-4777-8777-777777777777",
            "intent_configmap": reconcile.INVENTORY_INTENT,
            "intent_configmap_uid": "88888888-8888-4888-8888-888888888888",
            "producer_runtime": {
                "job_uid": "99999999-9999-4999-8999-999999999999",
                "pod_uid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "pod_restarts": 0,
            },
        },
    }
    monkeypatch.setattr(
        reconcile,
        "_inventory_package_authority",
        lambda inventory, kubernetes: inventory_package,
    )
    return rows, canary


class _NotFound(RuntimeError):
    code = 404


def _kubernetes(kind: str, name: str | None) -> dict:
    if name == "g7" and kind == "job":
        return {
            "metadata": {"uid": JOB_UID},
            "status": {"conditions": [{"type": "Complete", "status": "True"}]},
        }
    if name == "g7" and kind == "pods":
        return {
            "items": [
                {
                    "metadata": {
                        "uid": POD_UID,
                        "ownerReferences": [
                            {
                                "apiVersion": "batch/v1",
                                "kind": "Job",
                                "name": "g7",
                                "uid": JOB_UID,
                                "controller": True,
                            }
                        ],
                    },
                    "status": {
                        "phase": "Succeeded",
                        "containerStatuses": [
                            {
                                "restartCount": 0,
                                "state": {"terminated": {"exitCode": 0}},
                            }
                        ],
                    },
                }
            ]
        }
    if kind == "job":
        raise _NotFound("not_found")
    return {"items": []}


def test_collect_accepts_only_canary_session_and_emits_no_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, canary = _fixtures(tmp_path, monkeypatch)
    predecessor5 = {"accepted_cells": 0, "sessions": 0, "model_calls": 0}
    predecessor6 = {"accepted_cells": 0, "sessions": 0, "model_calls": 0}
    monkeypatch.setattr(reconcile, "_generation5_failure_binding", lambda root: predecessor5)
    monkeypatch.setattr(reconcile, "_generation6_failure_binding", lambda root: predecessor6)

    def sessions(task_key: str, key: str) -> list[dict]:
        assert (task_key, key) == ("t", "secret")
        return [
            {
                "session_id": canary["session_id"],
                "status": "completed",
                "model": bulk.exact.EXPECTED_MODELS["qwen3.8-27b"]["session_model"],
                "verifier_execution": {"id": canary["verifier_execution_id"]},
                "metadata": {
                    "run_id": "g7-run",
                    "execution_id": canary["spec"]["execution"]["execution_id"],
                    "cell_id": canary["spec"]["statistical_cell"]["cell_id"],
                },
                "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            }
        ]

    receipt, _, _ = reconcile.collect(
        ROOT,
        job_uid=JOB_UID,
        pod_uid=POD_UID,
        sessions=sessions,
        kubernetes=_kubernetes,
        api_key="secret",
        observed_at="2026-09-05T08:00:00Z",
    )
    assert receipt["remaining_cells"] == rows
    assert receipt["planned_execution_count"] == 798
    assert receipt["collisions"] == {
        "fleet_api": 0,
        "kubernetes_job_or_pod": 0,
        "sfs_output": 0,
        "global_claim": 0,
        "active_or_accepted_cell": 0,
    }
    assert receipt["receipt_sha256"] == bulk.digest(receipt, "receipt_sha256")
    encoded = json.dumps(receipt).lower()
    assert all(token not in encoded for token in ('"trace":', '"flag":', '"score":'))


@pytest.mark.parametrize("missing", ["run_id", "execution_id", "cell_id", "task_version_id"])
def test_collect_rejects_canary_session_with_missing_exact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _, canary = _fixtures(tmp_path, monkeypatch)
    row = {
        "session_id": canary["session_id"],
        "status": "completed",
        "model": bulk.exact.EXPECTED_MODELS["qwen3.8-27b"]["session_model"],
        "verifier_execution": {"id": canary["verifier_execution_id"]},
        "metadata": {
            "run_id": "g7-run",
            "execution_id": canary["spec"]["execution"]["execution_id"],
            "cell_id": canary["spec"]["statistical_cell"]["cell_id"],
        },
        "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    }
    if missing in {"run_id", "execution_id", "cell_id"}:
        row["metadata"].pop(missing)
    else:
        row.pop("task_version_id")

    with pytest.raises(
        reconcile.ReconciliationError, match="generation7_authoritative_session_drifted"
    ):
        reconcile.collect(
            ROOT,
            job_uid=JOB_UID,
            pod_uid=POD_UID,
            sessions=lambda task_key, key: [row],
            kubernetes=_kubernetes,
            api_key="secret",
            observed_at="2026-09-05T08:00:00Z",
        )


def test_collect_rejects_any_planned_run_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, canary = _fixtures(tmp_path, monkeypatch)

    def sessions(task_key: str, key: str) -> list[dict]:
        return [
            {
                "session_id": canary["session_id"],
                "status": "completed",
                "model": bulk.exact.EXPECTED_MODELS["qwen3.8-27b"]["session_model"],
                "verifier_execution": {"id": canary["verifier_execution_id"]},
                "metadata": {
                    "run_id": "g7-run",
                    "execution_id": canary["spec"]["execution"]["execution_id"],
                    "cell_id": canary["spec"]["statistical_cell"]["cell_id"],
                },
                "task_version_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            },
            {
                "session_id": "77777777-7777-4777-8777-777777777777",
                "metadata": {"run_id": "planned-run"},
            },
        ]

    with pytest.raises(reconcile.ReconciliationError, match="session_collision"):
        reconcile.collect(
            ROOT,
            job_uid=JOB_UID,
            pod_uid=POD_UID,
            sessions=sessions,
            kubernetes=_kubernetes,
            api_key="secret",
            observed_at="2026-09-05T08:00:00Z",
        )


def test_succeeded_job_requires_zero_exit_and_zero_init_restarts() -> None:
    def dirty_kubernetes(kind: str, name: str | None) -> dict:
        value = _kubernetes(kind, name)
        if kind == "pods":
            value["items"][0]["status"]["initContainerStatuses"] = [
                {
                    "restartCount": 1,
                    "state": {"terminated": {"exitCode": 0}},
                }
            ]
        return value

    with pytest.raises(reconcile.ReconciliationError, match="clean_success"):
        reconcile._succeeded_job("g7", JOB_UID, POD_UID, dirty_kubernetes)


def test_succeeded_job_requires_exact_pod_owner_reference_uid() -> None:
    def wrong_owner_kubernetes(kind: str, name: str | None) -> dict:
        value = _kubernetes(kind, name)
        if kind == "pods":
            value["items"][0]["metadata"]["ownerReferences"][0]["uid"] = (
                "55555555-5555-4555-8555-555555555555"
            )
        return value

    with pytest.raises(reconcile.ReconciliationError, match="exclusively_complete"):
        reconcile._succeeded_job("g7", JOB_UID, POD_UID, wrong_owner_kubernetes)


def test_inventory_package_authority_binds_immutable_live_bytes_and_runtime() -> None:
    package_commit = "1" * 40
    files = {}
    data = {}
    for key, (source_path, install_path) in inventory_package.PACKAGE_FILES.items():
        raw = f"exact-{key}\n"
        data[key] = raw
        files[key] = {
            "source_path": source_path,
            "install_path": install_path,
            "sha256": inventory_package.sha256(raw.encode()),
        }
    package_receipt = {
        "schema_version": inventory_package.SCHEMA,
        "package_commit": package_commit,
        "files": files,
        "job_manifest": {
            "source_path": inventory_package.MANIFEST_PATH,
            "sha256": "sha256:" + "e" * 64,
        },
        "runtime_contract": {
            "fleet_methods": ["GET"],
            "task_version_gets": 100,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
            "mutation_calls": 0,
            "sfs_terminal_path": str(reconcile.INVENTORY_TERMINAL),
        },
    }
    package_receipt["package_sha256"] = inventory_package.digest_without(
        package_receipt, "package_sha256"
    )
    package_text = inventory_package.canonical_json(package_receipt).decode() + "\n"
    data.update({"package.json": package_text, "package_commit": package_commit})
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": reconcile.INVENTORY_CONFIGMAP,
            "namespace": reconcile.NAMESPACE,
            "uid": "77777777-7777-4777-8777-777777777777",
        },
        "immutable": True,
        "data": data,
    }
    intent = inventory_package.build_intent(configmap)
    intent["metadata"]["uid"] = "88888888-8888-4888-8888-888888888888"

    def kubernetes(kind: str, name: str | None) -> dict:
        if kind == "configmap":
            return configmap if name == reconcile.INVENTORY_CONFIGMAP else intent
        value = copy.deepcopy(_kubernetes(kind, "g7"))
        if kind == "pods":
            value["items"][0]["metadata"]["ownerReferences"][0]["name"] = (
                reconcile.INVENTORY_JOB
            )
        return value

    result = reconcile._inventory_package_authority(
        {"runtime": {"job_uid": JOB_UID, "pod_uid": POD_UID}}, kubernetes
    )
    assert result["raw"] == package_text.encode()
    assert result["binding"]["file_sha256"] == inventory_package.sha256(package_text.encode())
    assert result["binding"]["package_sha256"] == package_receipt["package_sha256"]
    assert result["binding"]["producer_runtime"] == {
        "job_uid": JOB_UID,
        "pod_uid": POD_UID,
        "pod_restarts": 0,
    }


def test_canary_gate_preserves_exact_raw_generation7_release_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, canary = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(reconcile, "OUTPUT_ROOT", tmp_path / "output")
    expected = canary["release_path"].read_bytes()
    gate = reconcile._canary_gate(
        canary, COLLECTOR_JOB_UID, COLLECTOR_POD_UID, "2026-09-05T08:00:00Z"
    )
    copied = reconcile.OUTPUT_ROOT / "qwen3.8-27b-generation7-release.json"
    assert copied.read_bytes() == expected
    assert gate["evidence"]["release_path"] == str(copied)
    assert bulk.file_sha256(copied) == bulk.file_sha256(canary["release_path"])


def test_accept_requires_source_job_success_and_rechecks_live_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconcile, "OUTPUT_ROOT", tmp_path)
    inventory_package_binding = {
        "path": str(tmp_path / reconcile.INVENTORY_PACKAGE_FILENAME),
        "file_sha256": bulk.self_hosted.sha256(b"{}\n"),
    }
    observation = {
        "runtime": {"job_uid": JOB_UID, "pod_uid": POD_UID},
        "receipt_sha256": "sha256:" + "7" * 64,
        "exact100_inventory": {"receipt_sha256": "sha256:" + "a" * 64},
        "exact100_inventory_package": inventory_package_binding,
        "superseded_generation5": {"accepted_cells": 0},
        "superseded_generation6": {"accepted_cells": 0},
        "predecessor_failure_output_roots_and_claims_absent": True,
        "generation7": {"q": {"receipt_sha256": "sha256:" + "b" * 64}},
    }
    monkeypatch.setattr(reconcile, "validate_observation", lambda value, root: None)
    calls: list[str] = []

    def succeeded(name: str, job: str, pod: str, kubernetes: object) -> dict:
        calls.append(name)
        assert (job, pod) == (JOB_UID, POD_UID)
        return {"job_uid": job, "pod_uid": pod, "pod_restarts": 0}

    monkeypatch.setattr(reconcile, "_succeeded_job", succeeded)
    fresh = {
        "observed_at_utc": "2026-09-05T08:01:00Z",
        "planned_execution_ids_sha256": "sha256:" + "8" * 64,
        "remaining_cells": [],
        "remaining_cells_sha256": "sha256:" + "9" * 64,
        "checked_job_names": ["bulk-job"],
        "checked_output_roots": ["/mnt/sfs/jobs/bulk-job"],
        "exact100_inventory": {"receipt_sha256": "sha256:" + "a" * 64},
        "exact100_inventory_package": inventory_package_binding,
        "superseded_generation5": {"accepted_cells": 0},
        "superseded_generation6": {"accepted_cells": 0},
        "predecessor_failure_output_roots_and_claims_absent": True,
        "generation7": {"q": {"receipt_sha256": "sha256:" + "b" * 64}},
    }
    monkeypatch.setattr(
        reconcile,
        "collect",
        lambda *args, **kwargs: (fresh, {"q": {}}, {"raw": b"{}\n"}),
    )
    monkeypatch.setattr(
        reconcile,
        "_canary_gate",
        lambda *args: {"receipt_sha256": "sha256:" + "b" * 64},
    )
    monkeypatch.setattr(reconcile, "_write_once", lambda path, value: None)
    monkeypatch.setattr(bulk, "validate_reconciliation_gate", lambda value, root: None)
    (tmp_path / "OBSERVATION.json").write_text("source")
    terminal = reconcile.accept(
        ROOT,
        observation,
        collector_job_uid=COLLECTOR_JOB_UID,
        collector_pod_uid=COLLECTOR_POD_UID,
        package_commit="1" * 40,
        sessions=lambda task, key: [],
        kubernetes=_kubernetes,
        api_key="secret",
    )
    assert calls == [reconcile.SOURCE_JOB]
    assert terminal["observer_job_succeeded"] is True
    assert terminal["runtime"] == observation["runtime"]
    assert terminal["collector_runtime"] == {
        "job_uid": COLLECTOR_JOB_UID,
        "pod_uid": COLLECTOR_POD_UID,
    }


def test_held_receipt_and_package_are_sanitized_and_nonlaunchable() -> None:
    held = reconcile.validate_held(ROOT)
    assert held["receipt_sha256"] == bulk.digest(held, "receipt_sha256")
    for key in ("superseded_generation5", "superseded_generation6"):
        predecessor = held[key]
        assert predecessor["accepted_cells"] == 0
        assert predecessor["sessions"] == 0
        assert predecessor["model_calls"] == 0
        assert predecessor["nonrepeatable"] is True
    built = package.build_package(ROOT)
    assert built["launch_authorized"] is False
    assert built["release_included"] is False
    assert max(built["object_json_bytes"].values()) < package.SAFETY_LIMIT
    assert set(built["configmaps"]) == {
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.OBSERVER_NAME,
    }


def test_manifest_is_held_create_once_cpu_only_and_read_only_rbac() -> None:
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    items = manifest["items"]
    role = next(item for item in items if item["kind"] == "Role")
    assert {verb for rule in role["rules"] for verb in rule["verbs"]} == {"get", "list"}
    configmap_rule = next(rule for rule in role["rules"] if rule["resources"] == ["configmaps"])
    assert configmap_rule["verbs"] == ["get"]
    assert configmap_rule["resourceNames"] == [
        reconcile.INVENTORY_CONFIGMAP,
        reconcile.INVENTORY_INTENT,
    ]
    jobs = [item for item in items if item["kind"] == "Job"]
    assert [item["metadata"]["name"] for item in jobs] == [
        reconcile.SOURCE_JOB,
        reconcile.ACCEPT_JOB,
    ]
    for job in jobs:
        assert (
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
        )
        assert job["spec"]["backoffLimit"] == 0
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert "nvidia.com/gpu" not in json.dumps(pod)


def test_submission_wrapper_is_create_once_and_never_deletes() -> None:
    script_path = ROOT / package.SUBMIT_PATH
    script = script_path.read_text()
    assert script_path.stat().st_mode & 0o111
    assert all(
        mode in script for mode in ("prepare-release", "submit-source", "submit-accept")
    )
    assert reconcile.SOURCE_JOB in script and reconcile.ACCEPT_JOB in script
    assert "SOURCE_INTENT=" in script and "ACCEPT_INTENT=" in script
    assert "create --dry-run=server" in script
    assert "status --porcelain=v1 --untracked-files=all" in script
    assert "merge-base --is-ancestor" in script
    assert "delete" not in script
    assert " apply " not in script
    assert 'create -f "$accept_job"' in script
    assert "source_job_uid" in script
    assert "RELEASE_SNAPSHOT=" in script
    assert 'os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400' in script
    assert "fixed_evidence_paths=(" in script
    assert "prebulk evidence mirror allowlist drifted" in script
    for path in reconcile.fixed_evidence_paths(ROOT).values():
        assert str(path) in script
    assert 'endswith("source-v3")' in script
    assert 'endswith("accept-v3")' in script
    assert "assert_observer" in script
    assert 'final_stability_check' in script
    assert 'reconcile.validate_release(release, root, evidence_root=Path(evidence_root))' in script
    assert "PREBULK_RELEASE_OUTPUT" in script
    assert "prebulk release rendering is nondeterministic" in script


def test_runtime_installs_dependencies_without_mutating_project_lock() -> None:
    script = (ROOT / package.RUN_PATH).read_text()
    assert script.count("uv run --no-project --with httpx==0.28.1") == 3
    assert "python3" not in script
    assert script.count("--evidence-root /mnt/sfs") == 2


def test_release_renderer_reads_only_an_exact_committed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.fleet import immutable_submission_snapshot as snapshot

    commit = "1" * 40
    calls: list[tuple[str, Path, str]] = []

    def stable(root: Path, expected: str) -> None:
        calls.append(("stable", root, expected))

    def materialize(root: Path, expected: str, destination: Path) -> None:
        calls.append(("materialize", root, expected))
        destination.mkdir()

    def build(root: Path, expected: str, *, evidence_root: Path | None = None) -> dict:
        assert root != tmp_path
        assert root.name == "package"
        assert expected == commit
        assert evidence_root == tmp_path / "evidence"
        return {"package_commit": expected}

    monkeypatch.setattr(snapshot, "assert_stable", stable)
    monkeypatch.setattr(snapshot, "materialize", materialize)
    monkeypatch.setattr(reconcile, "build_release", build)
    assert reconcile.render_release_from_commit(
        tmp_path, commit, evidence_root=tmp_path / "evidence"
    ) == {"package_commit": commit}
    assert calls == [
        ("stable", tmp_path, commit),
        ("materialize", tmp_path, commit),
    ]


def test_projected_package_reconstructs_from_empty_root(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    projected = tmp_path / "projected"
    projected.mkdir()
    for configmap in built["configmaps"].values():
        for key, value in configmap["data"].items():
            (projected / key).write_text(value)
    destination = tmp_path / "root"
    package.materialize(projected, destination, built["aggregate_sha256"])
    assert (destination / package.MODULE_PATH).is_file()
    assert (destination / package.RUN_PATH).is_file()


def test_submitter_python_heredocs_compile() -> None:
    submitter = ROOT / package.SUBMIT_PATH
    lines = submitter.read_text().splitlines()
    bodies: list[str] = []
    cursor = 0
    while cursor < len(lines):
        if "<<'PY'" not in lines[cursor]:
            cursor += 1
            continue
        try:
            end = lines.index("PY", cursor + 1)
        except ValueError as error:
            raise AssertionError(f"unterminated Python heredoc at line {cursor + 1}") from error
        bodies.append("\n".join(lines[cursor + 1 : end]) + "\n")
        cursor = end + 1
    assert bodies
    for index, body in enumerate(bodies, start=1):
        compile(body, f"{submitter}#heredoc-{index}", "exec")


def test_resealed_observation_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = reconcile.remaining_cells(ROOT)
    receipt = {
        "schema_version": reconcile.OBSERVATION_SCHEMA,
        "status": "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE",
        "campaign_id": bulk.exact.EXPECTED_CAMPAIGN_ID,
        "universe_sha256": bulk._universe(ROOT)["universe_sha256"],
        "remaining_cells": rows,
        "remaining_cells_sha256": bulk.self_hosted.sha256(bulk.self_hosted.canonical_json(rows)),
        "planned_execution_count": 798,
        "planned_execution_ids_sha256": bulk.self_hosted.sha256(
            bulk.self_hosted.canonical_json(sorted(row["execution_id"] for row in rows))
        ),
        "checked_job_names": sorted(plan["job_name"] for plan in bulk.validate_all(ROOT).values()),
        "checked_output_roots": sorted(
            plan["sfs_root"] for plan in bulk.validate_all(ROOT).values()
        ),
        "collisions": {
            "fleet_api": 0,
            "kubernetes_job_or_pod": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "active_or_accepted_cell": 0,
        },
        "observed_at_utc": "2026-09-05T08:00:00Z",
        "runtime": {"job_uid": JOB_UID, "pod_uid": POD_UID},
        "exact100_inventory": {
            "path": str(reconcile.INVENTORY_TERMINAL),
            "file_sha256": "sha256:" + "a" * 64,
            "receipt_sha256": "sha256:" + "b" * 64,
        },
        "exact100_inventory_package": {
            "path": str(reconcile.OUTPUT_ROOT / reconcile.INVENTORY_PACKAGE_FILENAME),
            "file_sha256": "sha256:" + "c" * 64,
            "package_sha256": "sha256:" + "d" * 64,
            "package_commit": "1" * 40,
            "bootstrap_configmap": reconcile.INVENTORY_CONFIGMAP,
            "bootstrap_configmap_uid": "77777777-7777-4777-8777-777777777777",
            "intent_configmap": reconcile.INVENTORY_INTENT,
            "intent_configmap_uid": "88888888-8888-4888-8888-888888888888",
            "producer_runtime": {
                "job_uid": "99999999-9999-4999-8999-999999999999",
                "pod_uid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "pod_restarts": 0,
            },
        },
        "superseded_generation5": reconcile._generation5_failure_binding(ROOT),
        "superseded_generation6": reconcile._generation6_failure_binding(ROOT),
        "predecessor_failure_output_roots_and_claims_absent": True,
        "generation7": {
            model: {
                "cell_id": "sha256:" + str(index) * 64,
                "execution_id": "sha256:" + str(index + 2) * 64,
                "session_id": f"{index}" * 8 + "-1111-4111-8111-" + f"{index}" * 12,
                "verifier_execution_id": f"{index + 2}" * 8
                + "-2222-4222-8222-"
                + f"{index + 2}" * 12,
                "claim_receipt_sha256": "sha256:" + str(index + 4) * 64,
                "terminal_receipt_sha256": "sha256:" + str(index + 6) * 64,
                "scoring_release_file_sha256": "sha256:" + "c" * 64,
                "scoring_release_receipt_sha256": "sha256:" + "d" * 64,
            }
            for index, model in enumerate(sorted(reconcile.G7_OUTPUTS), start=1)
        },
        "request_counts": {
            "task_session_queries": 100,
            "session_rows_examined": 2,
            "transcript_queries": 0,
            "mutation_calls": 0,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = bulk.digest(receipt, "receipt_sha256")
    reconcile.validate_observation(receipt, ROOT)
    changed = copy.deepcopy(receipt)
    changed["remaining_cells"][0]["attempt"] = 4
    changed["receipt_sha256"] = bulk.digest(changed, "receipt_sha256")
    with pytest.raises(reconcile.ReconciliationError, match="observation_invalid"):
        reconcile.validate_observation(changed, ROOT)
