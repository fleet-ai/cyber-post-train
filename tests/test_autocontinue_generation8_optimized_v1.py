from __future__ import annotations

import copy
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import autocontinue_generation8_optimized_v1 as generation8
from evals.fleet import autocontinue_generation8_package_v1 as package
from evals.fleet import hosted_sweep_controller as hosted

ROOT = Path(__file__).parents[1]


def _write(path: Path, value: dict) -> None:
    path.write_bytes(generation8.canonical(value) + b"\n")


def _static():
    ledger, held, static = generation8._all_static(ROOT)  # noqa: SLF001
    return ledger, held, static


def _terminal_kubernetes(kind: str, name: str) -> dict:
    model = next(
        model for model in generation8.MODELS if generation7.EXPECTED[model]["job_name"] == name
    )
    row = generation8.MODELS[model]
    if kind == "job":
        return {
            "metadata": {"name": name, "uid": row["g7_job_uid"]},
            "status": {
                "failed": 1,
                "conditions": [{"type": "Failed", "status": "True", "reason": "DeadlineExceeded"}],
            },
        }
    return {
        "items": [
            {
                "metadata": {"uid": row["g7_pod_uid"]},
                "status": {"phase": "Failed"},
            }
        ]
    }


def _active_kubernetes(kind: str, name: str) -> dict:
    model = next(
        model for model in generation8.MODELS if generation7.EXPECTED[model]["job_name"] == name
    )
    row = generation8.MODELS[model]
    if kind == "job":
        return {
            "metadata": {
                "name": name,
                "uid": row["g7_job_uid"],
                "resourceVersion": f"job-rv-{row['short']}",
            },
            "status": {"active": 1},
        }
    return {
        "items": [
            {
                "metadata": {
                    "name": f"{name}-pod",
                    "uid": row["g7_pod_uid"],
                    "resourceVersion": f"pod-rv-{row['short']}",
                },
                "status": {"phase": "Running"},
            }
        ]
    }


def _generation1_kubernetes(kind: str, name: str) -> dict:
    model = next(
        model
        for model, row in generation8.PRESERVED_GENERATION1_ROOTS.items()
        if row["job_name"] == name
    )
    row = generation8.PRESERVED_GENERATION1_ROOTS[model]
    if kind == "job":
        return {
            "metadata": {"name": name, "uid": row["job_uid"]},
            "status": {
                "failed": 1,
                "conditions": [{"type": "Failed", "status": "True"}],
            },
        }
    return {
        "items": [
            {
                "metadata": {
                    "uid": row["pod_uid"],
                    "ownerReferences": [{"kind": "Job", "uid": row["job_uid"]}],
                },
                "status": {"phase": "Failed"},
            }
        ]
    }


def _tombstone(static):
    return generation8.build_tombstone(
        static,
        kubernetes=_terminal_kubernetes,
        path_exists=lambda _path: False,
        sessions=lambda _task: [],
        observed_at_utc="2026-09-05T15:00:00Z",
    )


def _generation1_root(tmp_path: Path, model: str) -> Path:
    binding = generation8.PRESERVED_GENERATION1_ROOTS[model]
    root = tmp_path / f"preserved-{generation8.MODELS[model]['short']}"
    root.mkdir()
    for name in ("attempts", "claims", "quarantine", "ramps", "task-claims", "task-results"):
        (root / name).mkdir()
    (root / ".attempt-claim-gate.lock").touch()
    plan = json.loads((ROOT / binding["plan"]).read_text())
    release = json.loads((ROOT / binding["release"]).read_text())
    _write(root / "PLAN.json", plan)
    _write(root / "SCORING-RELEASE.json", release)
    task = plan["tasks"][0]
    item = plan["attempts"][0]
    task_claim = {
        "schema_version": "fleet-hosted-opencode-task-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": int(task["rank"]),
        "source_rank": int(task["source_rank"]),
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "run_ids": [item["run_id"]],
    }
    task_claim["claim_sha256"] = generation8.legacy.digest_without(task_claim, "claim_sha256")
    config = hosted._attempt_config(plan, task, item)  # noqa: SLF001
    attempt_claim = {
        "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "task_claim_sha256": task_claim["claim_sha256"],
        "run_id": item["run_id"],
        "rank": int(task["rank"]),
        "source_rank": int(item["source_rank"]),
        "attempt": 1,
        "network": item["network"],
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "config_sha256": config["config_sha256"],
    }
    attempt_claim["claim_sha256"] = generation8.legacy.digest_without(attempt_claim, "claim_sha256")
    _write(root / "task-claims/rank-001.json", task_claim)
    _write(root / "claims" / f"{item['run_id']}.json", attempt_claim)
    return root


@pytest.mark.parametrize("model", sorted(generation8.MODELS))
def test_exact_preserved_generation1_root_is_safe_nonexecution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    root = _generation1_root(tmp_path, model)
    monkeypatch.setitem(generation8.PRESERVED_GENERATION1_ROOTS[model], "root", str(root))
    receipt = generation8.validate_preserved_generation1_root(
        model, root, repository_root=ROOT, kubernetes=_generation1_kubernetes
    )
    assert receipt["status"] == "VALIDATED_GENERATION1_PRE_MODEL_ROOT"
    assert receipt["attempt_directories"] == receipt["model_calls"] == 0
    assert receipt["sessions"] == receipt["verifier_executions"] == 0
    assert receipt["accepted_or_terminal_receipts"] == 0


@pytest.mark.parametrize(
    "mutation",
    ("attempt", "terminal", "extra-claim", "changed-plan", "symlink-claim"),
)
def test_preserved_generation1_root_fails_closed_on_execution_or_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    model = "qwen3.8-27b"
    root = _generation1_root(tmp_path, model)
    monkeypatch.setitem(generation8.PRESERVED_GENERATION1_ROOTS[model], "root", str(root))
    if mutation == "attempt":
        (root / "attempts/unexpected").mkdir()
    elif mutation == "terminal":
        _write(root / "CANARY-TERMINAL.json", {})
    elif mutation == "extra-claim":
        _write(root / "claims/extra.json", {})
    elif mutation == "changed-plan":
        _write(root / "PLAN.json.changed", {})
    else:
        claim = next((root / "claims").iterdir())
        claim.unlink()
        claim.symlink_to(root / "PLAN.json")
    with pytest.raises(RuntimeError):
        generation8.validate_preserved_generation1_root(
            model, root, repository_root=ROOT, kubernetes=_generation1_kubernetes
        )


def test_preserved_generation1_root_requires_exact_failed_job_and_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = "qwen3.8-27b"
    root = _generation1_root(tmp_path, model)
    monkeypatch.setitem(generation8.PRESERVED_GENERATION1_ROOTS[model], "root", str(root))

    def wrong_job(kind: str, name: str) -> dict:
        value = copy.deepcopy(_generation1_kubernetes(kind, name))
        if kind == "job":
            value["metadata"]["uid"] = "wrong-job-uid"
        return value

    with pytest.raises(RuntimeError, match="terminal identity drifted"):
        generation8.validate_preserved_generation1_root(
            model, root, repository_root=ROOT, kubernetes=wrong_job
        )


def test_absence_row_allows_only_the_validated_preserved_generation1_root() -> None:
    _, _, static = _static()
    model = "qwen3.8-27b"
    spec, plan = static[model]
    preserved = generation8.PRESERVED_GENERATION1_ROOTS[model]["root"]
    calls: list[tuple[str, Path]] = []

    def validate(selected_model: str, selected_path: Path) -> dict:
        calls.append((selected_model, selected_path))
        expected = generation8.PRESERVED_GENERATION1_ROOTS[selected_model]
        return {
            "path": str(selected_path),
            "status": "VALIDATED_GENERATION1_PRE_MODEL_ROOT",
            "plan_sha256": expected["plan_sha256"],
            "scoring_release_receipt_sha256": expected["release_receipt_sha256"],
            "task_claim_sha256": expected["task_claim_sha256"],
            "attempt_claim_sha256": expected["attempt_claim_sha256"],
            "incident_receipt_sha256": generation8.generation1_evidence.INCIDENT_SHA,
            "generation1_tombstone_receipt_sha256": (
                "sha256:256398d56acef1caacb9de7f2d8ca7d336cdb154c6191f8986797cb75afcbab8"
            ),
            "generation1_job_uid": expected["job_uid"],
            "generation1_pod_uid": expected["pod_uid"],
            "generation1_terminal": "Failed",
            "attempt_directories": 0,
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "accepted_or_terminal_receipts": 0,
        }

    row = generation8._absence_row(  # noqa: SLF001
        model,
        spec,
        plan,
        path_exists=lambda path: path == preserved,
        sessions=lambda _task: [],
        preserved_root_validator=validate,
    )
    generation8._validate_absence_row(row, model, spec, plan)  # noqa: SLF001
    assert calls == [(model, Path(preserved))]
    changed = copy.deepcopy(row)
    changed["preserved_generation1_root"]["model_calls"] = 1
    with pytest.raises(ValueError, match="absence binding drifted"):
        generation8._validate_absence_row(changed, model, spec, plan)  # noqa: SLF001

    other = generation8.generation_output_paths(model, spec)[0]
    with pytest.raises(RuntimeError, match="claim/output identity exists"):
        generation8._absence_row(  # noqa: SLF001
            model,
            spec,
            plan,
            path_exists=lambda path: path == other,
            sessions=lambda _task: [],
            preserved_root_validator=validate,
        )


def test_static_validation_is_exactly_once_and_fast_without_recursive_g7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generation7,
        "validate_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recursive G7 validator used")
        ),
    )
    started = time.monotonic()
    ledger, held, static = _static()
    elapsed = time.monotonic() - started
    assert held["status"] == "HELD"
    assert set(static) == set(generation8.MODELS)
    assert ledger.read_counts == {label: 1 for label in ledger.read_counts}
    assert (
        generation8.MEASURED_G7_ONE_MODEL_VALIDATE_SECONDS / elapsed
        >= generation8.REQUIRED_VALIDATION_SPEEDUP
    )


def test_frozen_generation8_plans_are_exact_g7_treatments_with_only_successor_fields() -> None:
    """Authoring-time parity check; runtime intentionally does not repeat it."""
    for model, row in generation8.MODELS.items():
        spec8 = json.loads((ROOT / row["spec"]).read_text())
        plan8 = json.loads((ROOT / row["plan"]).read_text())
        spec7 = generation7.load(ROOT / generation7.G7_SPEC_PATHS[model])
        plan7 = generation7.validate_spec(spec7, ROOT)
        for field in ("model", "tasks", "harness", "authority", "pass_k"):
            assert plan8[field] == plan7[field]
        assert spec8["statistical_cell"] == spec7["statistical_cell"]
        assert spec8["execution"]["execution_generation"] == 8
        assert spec8["execution"]["execution_id"] != spec7["execution"]["execution_id"]
        assert plan8["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan8["harness"]["context_window_size"] == 262144
        assert plan8["harness"]["compaction_headroom_tokens"] == 20000


def test_tombstone_requires_both_exact_g7_jobs_terminal_and_all_absence() -> None:
    _, _, static = _static()
    value = _tombstone(static)
    generation8.validate_tombstone(value, static)
    assert value["claims_outputs_sessions_absent"] is True
    assert all(len(row["checked_claim_paths"]) == 8 for row in value["models"])
    assert all(len(row["checked_output_paths"]) == 9 for row in value["models"])
    assert all(len(row["run_ids_checked"]) == 9 for row in value["models"])

    def active(kind: str, name: str) -> dict:
        value = _terminal_kubernetes(kind, name)
        if kind == "job":
            value["status"]["active"] = 1
        return value

    with pytest.raises(RuntimeError, match="not exclusively terminally stopped"):
        generation8.build_tombstone(
            static,
            kubernetes=active,
            path_exists=lambda _path: False,
            sessions=lambda _task: [],
            observed_at_utc="2026-09-05T15:00:00Z",
        )
    first_claim = generation8.generation_claim_paths(next(iter(static.values()))[0])[0]
    with pytest.raises(RuntimeError, match="claim/output identity exists"):
        generation8.build_tombstone(
            static,
            kubernetes=_terminal_kubernetes,
            path_exists=lambda path: path == first_claim,
            sessions=lambda _task: [],
            observed_at_utc="2026-09-05T15:00:00Z",
        )


def test_tombstone_refuses_any_matching_authoritative_session() -> None:
    _, _, static = _static()
    cell = static["qwen3.8-27b"][0]["statistical_cell"]["cell_id"]
    with pytest.raises(RuntimeError, match="session identity exists"):
        generation8.build_tombstone(
            static,
            kubernetes=_terminal_kubernetes,
            path_exists=lambda _path: False,
            sessions=lambda task: [{"cell_id": cell}] if "" in task else [],
            observed_at_utc="2026-09-05T15:00:00Z",
        )


def test_controlled_stop_is_uid_preconditioned_race_closed_and_releaseable(
    tmp_path: Path,
) -> None:
    _, _, static = _static()
    prestop = generation8.build_prestop(
        static,
        kubernetes=_active_kubernetes,
        path_exists=lambda _path: False,
        sessions=lambda _task: [],
        observed_at_utc="2026-09-05T15:00:00Z",
    )
    authorization = generation8.build_delete_authorization(prestop, static)
    assert all(row["uid_precondition"] for row in authorization["requests"])
    assert all(row["propagation_policy"] == "Foreground" for row in authorization["requests"])
    deleted: set[str] = set()
    calls: list[tuple[str, str]] = []

    def kubernetes(kind: str, name: str) -> dict:
        if name in deleted:
            return {} if kind == "job" else {"items": []}
        return _active_kubernetes(kind, name)

    def delete(name: str, uid: str) -> None:
        model = next(
            model for model in generation8.MODELS if generation7.EXPECTED[model]["job_name"] == name
        )
        assert uid == generation8.MODELS[model]["g7_job_uid"]
        calls.append((name, uid))
        deleted.add(name)

    claims = tmp_path / "claims"
    claims.mkdir()
    protocol = tmp_path / "protocol"

    def now() -> datetime:
        return datetime(2026, 9, 5, 15, 0, tzinfo=UTC)

    bundle = generation8.execute_controlled_stop(
        static,
        prestop,
        authorization,
        protocol,
        kubernetes=kubernetes,
        delete_job=delete,
        path_exists=lambda _path: False,
        sessions=lambda _task: [],
        claim_root=claims,
        now=now,
        wait=lambda _seconds: None,
    )
    assert len(calls) == 2
    assert sorted(path.name for path in protocol.iterdir()) == [
        "DELETE-AUTHORIZATION.json",
        "DELETION.json",
        "FRESH-PRESTOP.json",
        "POSTSTOP.json",
        "PRESTOP.json",
    ]
    tombstone = generation8.build_tombstone(
        static,
        kubernetes=kubernetes,
        path_exists=lambda _path: False,
        sessions=lambda _task: [],
        observed_at_utc="2026-09-05T15:01:00Z",
        controlled_stop=bundle,
    )
    generation8.validate_tombstone(tombstone, static)
    assert tombstone["stop_mode"] == "uid_preconditioned_foreground_delete"


@pytest.mark.parametrize("collision", ["claim", "session", "output", "uid"])
def test_controlled_prestop_refuses_every_identity_collision(collision: str) -> None:
    _, _, static = _static()
    cell = static["qwen3.8-27b"][0]["statistical_cell"]["cell_id"]

    def kubernetes(kind: str, name: str) -> dict:
        value = _active_kubernetes(kind, name)
        if collision == "uid" and kind == "job" and name.startswith("chris-q38"):
            value["metadata"]["uid"] = "11111111-1111-4111-8111-111111111111"
        return value

    with pytest.raises(RuntimeError):
        generation8.build_prestop(
            static,
            kubernetes=kubernetes,
            path_exists=lambda path: (
                collision in {"claim", "output"}
                and (("claims" in path) if collision == "claim" else ("jobs" in path))
            ),
            sessions=lambda _task: [{"cell_id": cell}] if collision == "session" else [],
            observed_at_utc="2026-09-05T15:00:00Z",
        )


@pytest.mark.parametrize("race", ["claim", "session", "output", "uid"])
def test_controlled_stop_refuses_every_race_before_delete(tmp_path: Path, race: str) -> None:
    _, _, static = _static()
    prestop = generation8.build_prestop(
        static,
        kubernetes=_active_kubernetes,
        path_exists=lambda _path: False,
        sessions=lambda _task: [],
        observed_at_utc="2026-09-05T15:00:00Z",
    )
    authorization = generation8.build_delete_authorization(prestop, static)
    calls = 0

    def changed(kind: str, name: str) -> dict:
        value = _active_kubernetes(kind, name)
        if race == "uid" and kind == "job" and name.startswith("chris-q38"):
            value["metadata"]["uid"] = "11111111-1111-4111-8111-111111111111"
        return value

    def delete(_name: str, _uid: str) -> None:
        nonlocal calls
        calls += 1

    claims = tmp_path / "claims"
    claims.mkdir()
    cell = static["qwen3.8-27b"][0]["statistical_cell"]["cell_id"]
    with pytest.raises(RuntimeError):
        generation8.execute_controlled_stop(
            static,
            prestop,
            authorization,
            tmp_path / "protocol",
            kubernetes=changed,
            delete_job=delete,
            path_exists=lambda path: (
                race in {"claim", "output"}
                and (("claims" in path) if race == "claim" else ("jobs" in path))
            ),
            sessions=lambda _task: [{"cell_id": cell}] if race == "session" else [],
            claim_root=claims,
            now=lambda: datetime(2026, 9, 5, 15, 0, tzinfo=UTC),
            wait=lambda _seconds: None,
        )
    assert calls == 0


def test_release_cannot_bypass_two_model_tombstone() -> None:
    _, held, static = _static()
    built = package.build_package(ROOT)
    tombstone = _tombstone(static)
    model = "qwen3.8-27b"
    value = generation8.build_release(
        model,
        static,
        held,
        tombstone,
        built["model_manifests"][model],
        "1" * 40,
    )
    assert value["launch_authorized"] is True
    changed = copy.deepcopy(tombstone)
    changed["claims_outputs_sessions_absent"] = False
    changed["receipt_sha256"] = generation8.digest(changed)
    with pytest.raises(ValueError, match="tombstone"):
        generation8.build_release(
            model,
            static,
            held,
            changed,
            built["model_manifests"][model],
            "1" * 40,
        )


def test_context_reuses_typed_values_and_claim_is_global_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, held, static = _static()
    tombstone = _tombstone(static)
    model = "qwen3.8-27b"
    built = package.build_package(ROOT)
    manifest = built["model_manifests"][model]
    release = generation8.build_release(model, static, held, tombstone, manifest, "2" * 40)
    tombstone_path, manifest_path, release_path = (
        tmp_path / "tombstone.json",
        tmp_path / "manifest.json",
        tmp_path / "release.json",
    )
    _write(tombstone_path, tombstone)
    _write(manifest_path, manifest)
    _write(release_path, release)
    tombstone_validations = 0
    original_validate_tombstone = generation8.validate_tombstone

    def counted_validate_tombstone(*args, **kwargs):
        nonlocal tombstone_validations
        tombstone_validations += 1
        return original_validate_tombstone(*args, **kwargs)

    monkeypatch.setattr(generation8, "validate_tombstone", counted_validate_tombstone)
    context = generation8.validate_context(
        ROOT,
        model,
        "2" * 40,
        release_path=release_path,
        tombstone_path=tombstone_path,
        package_manifest_path=manifest_path,
    )
    assert tombstone_validations == 1
    assert all(count == 1 for count in context.ledger.read_counts.values())
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    claims = tmp_path / "claims"
    claims.mkdir()
    claim = generation8.claim_execution(context, claim_root=claims)
    assert claim["execution_generation"] == 8
    with pytest.raises(RuntimeError, match="already has a generation claim"):
        generation8.claim_execution(context, claim_root=claims)
    context.plan["harness"]["context_window_size"] = 1
    with pytest.raises(RuntimeError, match="changed on reuse"):
        context.assert_unchanged()


def test_held_manifest_and_shells_cannot_launch() -> None:
    manifest = yaml.safe_load((ROOT / generation8.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for job in manifest["items"]:
        pod = job["spec"]["template"]["spec"]
        assert (
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
        )
        assert job["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert "gpu" not in json.dumps(job).lower()
    for path in (ROOT / generation8.RUN_PATH, ROOT / generation8.SUBMIT_PATH):
        subprocess.run(["bash", "-n", str(path)], check=True)
    submit = (ROOT / generation8.SUBMIT_PATH).read_text()
    runner = (ROOT / generation8.RUN_PATH).read_text()
    assert "exit 78" in submit
    assert "kubectl create" not in submit
    # The bootstrap performs the sole mounted-package byte/digest validation.
    assert " verify-mounted " not in runner


def test_preview_performs_no_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden_delete(_name: str, _uid: str) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("preview attempted a delete")

    monkeypatch.setattr(generation8, "_delete_kubernetes_job", forbidden_delete)
    preview = generation8.held_preview(ROOT)
    assert preview["status"] == "HELD"
    assert preview["objects_created"] is False
    assert calls == 0


def test_package_is_held_bounded_and_contains_optimized_runtime() -> None:
    built = package.build_package(ROOT)
    assert built["launch_authorized"] is False
    assert built["release_included"] is False
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.base.PACKAGE_OBJECT_LIMIT
    paths = {
        entry["source_path"]
        for obj in built["object_manifests"].values()
        for entry in obj["entries"]
    }
    assert generation8.MODULE_PATH in paths
    assert generation8.RUN_PATH in paths
