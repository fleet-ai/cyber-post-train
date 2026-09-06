import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_package_v1 as package
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v1 as diagnostic
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def _set_uids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_UID", JOB_UID)
    monkeypatch.setenv("POD_UID", POD_UID)


@pytest.mark.parametrize("failed_index", range(len(diagnostic.PHASES)))
def test_every_phase_failure_emits_one_sanitized_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_index: int
) -> None:
    _set_uids(monkeypatch)
    calls = []

    def callback(index: int):
        def run(_root: Path, _state: dict) -> None:
            calls.append(index)
            if index == failed_index:
                raise RuntimeError("sanitized injected failure")
            if index > failed_index:
                raise AssertionError("phase ran after terminal failure")

        return run

    phases = tuple(
        (name, callback(index)) for index, (name, _function) in enumerate(diagnostic.PHASES)
    )
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 1
    value = json.loads(output.read_text())
    assert calls == list(range(failed_index + 1))
    assert value["status"] == "FAILED"
    assert value["failed_phase"] == diagnostic.PHASES[failed_index][0]
    assert value["completed_phases"] == [
        name for name, _function in diagnostic.PHASES[:failed_index]
    ]
    assert value["error_classification"] == "gate-runtime-error"
    assert value["error_sha256"].startswith("sha256:")
    assert "sanitized injected failure" not in output.read_text()
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["model_calls"] == 0
    assert value["task_calls"] == 0
    assert value["session_calls"] == 0
    assert value["verifier_calls"] == 0
    assert value["scoring_calls"] == 0
    assert value["api_mutation_calls"] == 0


def test_all_phases_pass_to_forbidden_network_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_uids(monkeypatch)
    phases = tuple((name, lambda _root, _state: None) for name, _function in diagnostic.PHASES)
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 0
    value = json.loads(output.read_text())
    assert value["status"] == "PASSED_TO_NETWORK_BOUNDARY"
    assert value["completed_phases"] == [name for name, _function in diagnostic.PHASES]
    assert value["failed_phase"] is None
    assert value["error_classification"] is None
    assert value["network_bound_phases_executed"] is False


def test_unallowlisted_exception_is_reduced_to_sanitized_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_uids(monkeypatch)

    def fail(_root: Path, _state: dict) -> None:
        raise KeyError("must-not-be-serialized")

    phases = ((diagnostic.PHASES[0][0], fail), *diagnostic.PHASES[1:])
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 1
    value = json.loads(output.read_text())
    assert value["error_classification"] == "unclassified-exception"
    assert "must-not-be-serialized" not in output.read_text()


def test_diagnostic_source_is_exact_merged_release_source() -> None:
    state = {}
    diagnostic._validate_source_bindings(ROOT, state)  # noqa: SLF001
    assert diagnostic.SOURCE_COMMIT == "193f660ec80420cd26f49757e67169a13ed63d12"
    assert state["source_sha256s"] == diagnostic.SOURCE_SHA256S


def test_diagnostic_package_is_immutable_held_and_credential_free() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    assert rendered["model_calls_authorized"] is False
    assert rendered["task_session_verifier_calls_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert (
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
        == "false"
    )
    env_names = {
        row["name"]
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "FLEET_API_KEY" not in env_names
    assert "SCORED_SOURCE_SHA256" not in env_names
    assert "SCORED_PACKAGE_TEMPLATE_SHA256" not in env_names
    assert "kubectl create" not in json.dumps(configmap["data"])


def test_materialized_diagnostic_package_imports_in_isolation(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["objects"]["items"][0]
    install = {
        "self_hosted.py": "evals/fleet/self_hosted.py",
        "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
        "endpoint_lease.py": "evals/fleet/endpoint_lease.py",
        "predecessor.py": "evals/fleet/exact_pass4_bulk_v3.py",
        "engine.py": "evals/fleet/exact_pass4_bulk_runtime_v3.py",
        "universe.py": "evals/fleet/exact_pass4_universe.py",
        "crypto.py": "evals/fleet/exact_pass4_crypto.py",
        "inventory.py": "evals/fleet/exact_pass4_task_inventory.py",
        "bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
        "bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
        "original_release.py": "evals/fleet/hosted_glm_exact_bulk_release_v1.py",
        "successor.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "source_package.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_package_v1.py",
        "release.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v1.py",
        "diagnostic.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_diagnostic_v1.py",
        "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    }
    bound_sources = {
        "release.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v1.py",
        "successor_runtime.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py",
        "successor.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py",
        "source_package.py": "evals/fleet/hosted_glm_rank29_a3a4_c2_package_v1.py",
    }
    assert {
        relative: self_hosted.sha256(configmap["data"][key].encode())
        for key, relative in bound_sources.items()
    } == diagnostic.SOURCE_SHA256S
    for key, relative in install.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(configmap["data"][key])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; "
            f"sys.path.insert(0, {str(tmp_path)!r}); "
            "from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_diagnostic_runtime_contains_no_network_or_create_boundary() -> None:
    source = (
        ROOT / "evals/fleet/hosted_glm_rank29_a3a4_c2_release_diagnostic_v1.py"
    ).read_text()
    assert "_fresh_route_check" not in source
    assert "_task_sessions" not in source
    assert "engine._client" not in source
    assert "kubectl" not in source
    assert "subprocess" not in source
    assert "render_scored" not in source
    runner = (
        ROOT / "evals/fleet/scripts/run_hosted_glm_rank29_release_diagnostic_v1.sh"
    ).read_text()
    assert "00-bootstrap-runtime" in runner
    assert "bootstrap-runtime-error" in runner
    assert 'if (( status != 0 )) && [[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in runner
    assert "api_mutation_calls\": 0" in runner


def test_tracked_terminal_and_held_receipts_are_self_digesting() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-v1-preoutput-terminal.json"
        ).read_text()
    )
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank29-release-diagnostic-v1-held.json"
        ).read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["release_receipt_present"] is False
    assert terminal["claims_created"] == 0
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["failed_release_terminal_receipt_sha256"] == terminal["receipt_sha256"]
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["source_sha256s"] == diagnostic.SOURCE_SHA256S
    assert held["launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
