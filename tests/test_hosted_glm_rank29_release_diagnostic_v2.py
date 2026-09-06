import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_package_v2 as package
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v2 as diagnostic
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _uids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")


def test_runtime_plan_uses_distinct_inventory_bound_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = {"receipt_sha256": diagnostic.INVENTORY_RECEIPT_SHA256}
    plan = {"plan_sha256": diagnostic.RUNTIME_PLAN_SHA256}
    monkeypatch.setattr(
        diagnostic.successor,
        "validate_all",
        lambda _root: {
            diagnostic.successor.CONTROLLER: {
                "plan_sha256": diagnostic.STATIC_PLAN_SHA256
            }
        },
    )
    monkeypatch.setattr(diagnostic.successor, "load", lambda _path: inventory)
    monkeypatch.setattr(
        diagnostic.successor,
        "build_runtime_plan",
        lambda controller, value, root: plan,
    )
    state = {}
    diagnostic._build_runtime_plan(ROOT, state)  # noqa: SLF001
    assert state["runtime_plan_sha256"] == diagnostic.RUNTIME_PLAN_SHA256
    assert state["static_plan_sha256"] == diagnostic.STATIC_PLAN_SHA256
    assert state["inventory_receipt_sha256"] == diagnostic.INVENTORY_RECEIPT_SHA256


@pytest.mark.parametrize(
    ("inventory_sha", "plan_sha"),
    [
        ("sha256:" + "0" * 64, diagnostic.RUNTIME_PLAN_SHA256),
        (diagnostic.INVENTORY_RECEIPT_SHA256, "sha256:" + "0" * 64),
    ],
)
def test_runtime_plan_binding_rejects_drift(
    monkeypatch: pytest.MonkeyPatch, inventory_sha: str, plan_sha: str
) -> None:
    monkeypatch.setattr(
        diagnostic.successor,
        "validate_all",
        lambda _root: {
            diagnostic.successor.CONTROLLER: {
                "plan_sha256": diagnostic.STATIC_PLAN_SHA256
            }
        },
    )
    monkeypatch.setattr(
        diagnostic.successor, "load", lambda _path: {"receipt_sha256": inventory_sha}
    )
    monkeypatch.setattr(
        diagnostic.successor,
        "build_runtime_plan",
        lambda controller, value, root: {"plan_sha256": plan_sha},
    )
    with pytest.raises(RuntimeError):
        diagnostic._build_runtime_plan(ROOT, {})  # noqa: SLF001


def test_static_plan_binding_rejects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostic.successor,
        "validate_all",
        lambda _root: {
            diagnostic.successor.CONTROLLER: {"plan_sha256": "sha256:" + "0" * 64}
        },
    )
    with pytest.raises(RuntimeError, match="static plan binding drifted"):
        diagnostic._build_runtime_plan(ROOT, {})  # noqa: SLF001


def test_v1_phase07_failure_is_reproduced_without_live_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {"receipt_sha256": diagnostic.INVENTORY_RECEIPT_SHA256}
    monkeypatch.setattr(diagnostic.prior.successor, "load", lambda _path: inventory)
    monkeypatch.setattr(
        diagnostic.prior.successor,
        "build_runtime_plan",
        lambda controller, value, root: {
            "plan_sha256": diagnostic.RUNTIME_PLAN_SHA256
        },
    )
    assert diagnostic.RUNTIME_PLAN_SHA256 != diagnostic.prior.PLAN_SHA256
    with pytest.raises(RuntimeError, match="successor plan binding drifted"):
        diagnostic.prior._build_plan(ROOT, {})  # noqa: SLF001


def test_fresh_v2_identity_and_held_credential_free_package() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert "diagnostic_v1.py" in configmap["data"]
    assert "diagnostic.py" in configmap["data"]
    env_names = {
        row["name"]
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "FLEET_API_KEY" not in env_names


def test_v2_receipt_is_sanitized_and_self_digesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _uids(monkeypatch)
    phases = tuple((name, lambda _root, _state: None) for name, _function in diagnostic.PHASES)
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 0
    value = json.loads(output.read_text())
    assert value["status"] == "PASSED_TO_NETWORK_BOUNDARY"
    assert value["diagnostic_job"] == diagnostic.JOB_NAME
    assert value["prior_diagnostic_receipt_sha256"].startswith("sha256:")
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["model_calls"] == value["scoring_calls"] == 0
    assert value["api_mutation_calls"] == 0


@pytest.mark.parametrize("failed_index", range(len(diagnostic.PHASES)))
def test_every_v2_phase_failure_emits_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_index: int
) -> None:
    _uids(monkeypatch)

    def callback(index: int):
        def run(_root: Path, _state: dict) -> None:
            if index == failed_index:
                raise RuntimeError("must-not-be-persisted")

        return run

    phases = tuple(
        (name, callback(index))
        for index, (name, _function) in enumerate(diagnostic.PHASES)
    )
    output = tmp_path / str(failed_index) / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 1
    value = json.loads(output.read_text())
    assert value["failed_phase"] == diagnostic.PHASES[failed_index][0]
    assert "must-not-be-persisted" not in output.read_text()
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_materialized_v2_package_imports_in_isolation(tmp_path: Path) -> None:
    data = package.render(ROOT)["objects"]["items"][0]["data"]
    install = {
        "self_hosted.py": "self_hosted.py",
        "runner.py": "opencode_train_sweep_runner.py",
        "endpoint_lease.py": "endpoint_lease.py",
        "predecessor.py": "exact_pass4_bulk_v3.py",
        "engine.py": "exact_pass4_bulk_runtime_v3.py",
        "universe.py": "exact_pass4_universe.py",
        "crypto.py": "exact_pass4_crypto.py",
        "inventory.py": "exact_pass4_task_inventory.py",
        "bulk.py": "hosted_glm_exact_bulk_v1.py",
        "bulk_runtime.py": "hosted_glm_exact_bulk_runtime_v1.py",
        "original_release.py": "hosted_glm_exact_bulk_release_v1.py",
        "successor.py": "hosted_glm_rank29_a3a4_c2_successor_v1.py",
        "successor_runtime.py": "hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "source_package.py": "hosted_glm_rank29_a3a4_c2_package_v1.py",
        "release.py": "hosted_glm_rank29_a3a4_c2_release_v1.py",
        "diagnostic_v1.py": "hosted_glm_rank29_a3a4_c2_release_diagnostic_v1.py",
        "diagnostic.py": "hosted_glm_rank29_a3a4_c2_release_diagnostic_v2.py",
        "fixed_proxy.py": "fixed_proxy.py",
    }
    fleet = tmp_path / "evals/fleet"
    (fleet / "configs").mkdir(parents=True)
    (tmp_path / "evals/__init__.py").touch()
    (fleet / "__init__.py").touch()
    for key, name in install.items():
        (fleet / name).write_text(data[key])
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; "
            f"sys.path.insert(0, {str(tmp_path)!r}); "
            "from evals.fleet import "
            "hosted_glm_rank29_a3a4_c2_release_diagnostic_v2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_v2_runner_and_package_have_no_scored_or_create_path() -> None:
    runner = (
        ROOT / "evals/fleet/scripts/run_hosted_glm_rank29_release_diagnostic_v2.sh"
    ).read_text()
    assert "hosted_glm_rank29_a3a4_c2_release_diagnostic_v2" in runner
    assert "00-bootstrap-runtime" in runner
    assert "kubectl create" not in runner
    source = (
        ROOT / "evals/fleet/hosted_glm_rank29_a3a4_c2_release_diagnostic_v2.py"
    ).read_text()
    assert "_fresh_route_check" not in source
    assert "_task_sessions" not in source
    assert "engine._client" not in source


def test_v1_terminal_and_v2_held_receipts_are_self_digesting() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-diagnostic-v1-terminal.json"
        ).read_text()
    )
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-diagnostic-v2-held.json"
        ).read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["retry_same_identity"] is False
    assert terminal["model_calls"] == terminal["scoring_calls"] == 0
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["launch_authorized"] is False
