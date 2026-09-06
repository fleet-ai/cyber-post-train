import copy
import json
import os
import subprocess
import sys
from datetime import UTC
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_preclaim_phase_bootstrap_v1 as bootstrap
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_package_v1 as package
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_v1 as diagnostic
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_COMMIT = "f30a1c0b24785ece6e6c68b839b02fea196a728d"


def test_rank30_preclaim_package_is_exact_create_once_and_score_free() -> None:
    rendered = package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert rendered["launch_authorized"] is False
    assert rendered["scored_successor_launch_authorized"] is False
    assert rendered["source_package_sha256"] == diagnostic.SOURCE_PACKAGE_SHA256
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 600
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert {row["name"] for row in container["env"]} == {"JOB_UID", "POD_UID"}
    assert "FLEET_API_KEY" not in json.dumps(job)
    assert all(
        not container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu")
        for container in pod["containers"]
    )
    assert len(json.dumps(configmap).encode()) < 900_000


def test_materialized_exact_scored_package_imports_in_isolation(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    configmap = rendered["objects"]["items"][0]
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    for name, text in configmap["data"].items():
        (projected / name).write_text(text)
    repo = tmp_path / "repo"
    result = subprocess.run(
        [sys.executable, str(projected / "bootstrap.py")],
        env={
            **os.environ,
            "BOOTSTRAP_ROOT": str(projected),
            "REPO_ROOT": str(repo),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((projected / "frozen-source-manifest.json").read_text())
    source = {
        key: (repo / binding["relative_path"]).read_text()
        for key, binding in manifest["files"].items()
    }
    assert self_hosted.sha256(self_hosted.canonical_json(source)) == (
        diagnostic.SOURCE_PACKAGE_SHA256
    )
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import evals.fleet.hosted_glm_rank30_preclaim_phase_observer_v1",
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    with pytest.raises(RuntimeError, match="materialization is unsafe"):
        bootstrap.materialize(projected, repo)


def _phases(failure: int | None = None, secret: str = "") -> tuple[diagnostic.Phase, ...]:
    result = []
    for index, (name, _function) in enumerate(diagnostic.PHASES, start=1):
        if index == failure:
            result.append((name, lambda _root, _state: (_ for _ in ()).throw(RuntimeError(secret))))
        else:
            result.append((name, lambda _root, _state: None))
    return tuple(result)


def test_observer_receipt_exposes_only_sanitized_phase_and_zero_call_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    output = tmp_path / "failure" / "DIAGNOSTIC.json"
    secret = "do-not-serialize-this-runtime-detail"
    assert diagnostic.run(ROOT, output_path=output, phases=_phases(6, secret)) == 1
    payload = output.read_text()
    assert secret not in payload
    receipt = json.loads(payload)
    assert receipt["failed_phase"] == "06-frozen-release-replay"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error_sha256"].startswith("sha256:")
    assert receipt["completed_phases"] == [name for name, _fn in diagnostic.PHASES[:5]]
    assert receipt["provider_constructed"] is False
    assert receipt["session_model_boundary_crossed"] is False
    assert set(receipt["zero_call_counters"].values()) == {0}
    assert receipt["privacy"] == {
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "protected_content_included": False,
    }
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    with pytest.raises(FileExistsError):
        diagnostic.run(ROOT, output_path=output, phases=_phases())


def test_frozen_release_replay_uses_failed_execution_clock_and_exact_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {"schema_version": "test"}
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    raw = self_hosted.canonical_json(receipt) + b"\n"
    path = tmp_path / "RELEASE.json"
    path.write_bytes(raw)
    monkeypatch.setattr(diagnostic, "RELEASE_PATH", path)
    monkeypatch.setattr(diagnostic, "RELEASE_FILE_SHA256", diagnostic._file_sha256(raw))  # noqa: SLF001
    monkeypatch.setattr(diagnostic, "RELEASE_RECEIPT_SHA256", receipt["receipt_sha256"])
    seen: dict[str, object] = {}

    def validate(value: dict[str, object], plan: dict[str, object], source: str) -> None:
        rank30_v3 = successor.prior.prior
        rank30_v1 = rank30_v3.prior
        seen.update(
            value=value,
            plan=plan,
            source=source,
            v3_now=rank30_v3.datetime.now(UTC),
            v1_now=rank30_v1.datetime.now(UTC),
        )

    monkeypatch.setattr(successor, "validate_release", validate)
    diagnostic._validate_frozen_release(ROOT, {"plan": {"plan_sha256": "x"}})  # noqa: SLF001
    assert seen["source"] == diagnostic.SOURCE_PACKAGE_SHA256
    assert seen["v3_now"] == diagnostic.FAILED_STARTED_AT
    assert seen["v1_now"] == diagnostic.FAILED_STARTED_AT


def test_held_receipt_is_digest_valid_and_forbids_launch() -> None:
    held = package.build_held(ROOT, PACKAGE_COMMIT)
    tracked = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-hosted-rank30-preclaim-phase-observer-held-v1.json"
        ).read_text()
    )
    assert tracked == held
    assert held["status"] == "PASSED_HELD_NO_LAUNCH"
    assert held["observer_launch_authorized"] is False
    assert held["scored_successor_launch_authorized"] is False
    assert held["stops_before_provider_session_model"] is True
    assert held["phase_order"][-1] == "07-strict-current-peer"
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )


def test_manifest_binding_mutation_is_rejected(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    data = copy.deepcopy(rendered["objects"]["items"][0]["data"])
    manifest = json.loads(data["frozen-source-manifest.json"])
    first = next(iter(manifest["files"]))
    manifest["files"][first]["relative_path"] = "../escape"
    data["frozen-source-manifest.json"] = json.dumps(manifest)
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    for name, text in data.items():
        (projected / name).write_text(text)
    with pytest.raises(RuntimeError, match="source path is invalid"):
        bootstrap.materialize(projected, tmp_path / "repo")
