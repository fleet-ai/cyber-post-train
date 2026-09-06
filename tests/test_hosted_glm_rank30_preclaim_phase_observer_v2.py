import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_preclaim_phase_bootstrap_v2 as bootstrap
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_package_v2 as package
from evals.fleet import hosted_glm_rank30_preclaim_phase_observer_v2 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _project(data: dict[str, str], root: Path) -> None:
    version = root / "..2026_09_06_16_31_43"
    version.mkdir(parents=True)
    for name, text in data.items():
        (version / name).write_text(text)
        (root / name).symlink_to(Path(version.name) / name)


def test_v2_package_is_fresh_create_once_and_score_free() -> None:
    rendered = package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    pod = job["spec"]["template"]["spec"]
    assert rendered["launch_authorized"] is False
    assert rendered["scored_successor_launch_authorized"] is False
    assert configmap["metadata"]["name"] == observer.CONFIGMAP_NAME
    assert job["metadata"]["name"] == observer.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert pod["preemptionPolicy"] == "Never"
    assert all(
        not container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu")
        for container in pod["containers"]
    )
    assert "FLEET_API_KEY" not in json.dumps(job)


def test_configmap_symlink_projection_materializes_without_escape(tmp_path: Path) -> None:
    data = package.render(ROOT)["objects"]["items"][0]["data"]
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    _project(data, projected)
    repo = tmp_path / "repo"

    bootstrap.materialize(projected, repo)

    assert (repo / "evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v1.py").is_file()
    assert (repo / "evals/fleet/hosted_glm_rank30_preclaim_phase_observer_v2.py").is_file()
    manifest = json.loads((projected / "frozen-source-manifest.json").read_text())
    assert all((repo / row["relative_path"]).is_file() for row in manifest["files"].values())


def test_configmap_symlink_projection_rejects_escape(tmp_path: Path) -> None:
    data = package.render(ROOT)["objects"]["items"][0]["data"]
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text(data["frozen-source-manifest.json"])
    (projected / "frozen-source-manifest.json").symlink_to(outside)

    with pytest.raises(RuntimeError, match="projected source is unsafe"):
        bootstrap.materialize(projected, tmp_path / "repo")


def test_materialization_rejects_symlinked_destination_parent(tmp_path: Path) -> None:
    data = package.render(ROOT)["objects"]["items"][0]["data"]
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    _project(data, projected)
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "evals").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="destination is unsafe"):
        bootstrap.materialize(projected, repo)
    assert list(outside.iterdir()) == []


def test_v2_source_phase_accepts_projected_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = package.render(ROOT)["objects"]["items"][0]["data"]
    projected = tmp_path / "bootstrap"
    projected.mkdir()
    _project(data, projected)
    repo = tmp_path / "repo"
    bootstrap.materialize(projected, repo)
    monkeypatch.setattr(observer, "MANIFEST_PATH", projected / "frozen-source-manifest.json")
    state: dict[str, object] = {}

    observer._validate_sources(repo, state)  # noqa: SLF001

    assert state["source_manifest_sha256"] == prior_file_sha256(
        (projected / "frozen-source-manifest.json").read_bytes()
    )


@pytest.mark.parametrize("replace_all", [False, True])
def test_noop_or_replaced_v2_phase_callables_cannot_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_all: bool,
) -> None:
    phases = list(observer.PHASES)

    def noop(_root: Path, _state: dict[str, object]) -> None:
        return None

    if replace_all:
        phases = [(name, noop) for name, _function in phases]
    else:
        phases[0] = (phases[0][0], noop)
    monkeypatch.setattr(observer, "PHASES", tuple(phases))
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"

    with pytest.raises(RuntimeError, match="phase binding drifted"):
        observer.run(ROOT, output_path=output)
    assert not output.exists()


def prior_file_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_held_receipt_is_digest_valid_and_forbids_launch() -> None:
    held = package.build_held(ROOT, "a03c5c80fddce915bc05ce7171189653f1aad962")
    tracked = json.loads(
        (
            ROOT
            / "docs/evidence/glm53-study/"
            "2026-09-06-glm53-hosted-rank30-preclaim-phase-observer-held-v2.json"
        ).read_text()
    )
    assert tracked == held
    assert held["status"] == "PASSED_HELD_NO_LAUNCH"
    assert held["observer_launch_authorized"] is False
    assert held["scored_successor_launch_authorized"] is False
    assert held["receipt_sha256"] == self_hosted.digest_without(held, "receipt_sha256")
