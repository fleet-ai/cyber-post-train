from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_canary as generation2
from evals.fleet import self_hosted

ROOT = Path.cwd()
Q_PLAN = Path("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json")
G_PLAN = Path("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json")
HELD = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json"
)
TOMBSTONES = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-generation1-tombstones-v1.json"
)
MANIFEST = Path("evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v1.yaml")
RUN = Path("evals/fleet/scripts/run_opencode_autocontinue_generation2_canary.sh")
SUBMIT = Path("evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v1.sh")


def _load(path: Path) -> dict:
    return generation2.load(path)


@pytest.mark.parametrize(
    ("path", "model", "rank", "execution_id"),
    [
        (
            Q_PLAN,
            "qwen3.8-27b",
            4,
            "sha256:71236ff990db210eb4989ee9e2b4c7afc4b7153b008178c2ae31d5149c3dc8ce",
        ),
        (
            G_PLAN,
            "glm-5.3",
            13,
            "sha256:db5c16dfac428436d4b70dd68ac65b5d1f835937cd4a620839ec9c9a0f4bac5f",
        ),
    ],
)
def test_generation2_plan_preserves_cell_and_renders_exact_treatment(
    path: Path, model: str, rank: int, execution_id: str
) -> None:
    spec = _load(path)
    plan = generation2.validate_plan(spec, ROOT)
    assert spec["model"] == model
    assert spec["statistical_cell"]["selection_rank"] == rank
    assert spec["execution"] == {
        "schema_version": "fleet-statistical-cell-execution-v1",
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_generation": 2,
        "execution_id": execution_id,
    }
    assert plan["harness"]["compaction_headroom_tokens"] == 20000
    assert plan["treatment_block"]["harness"]["compaction_headroom_tokens"] == 20000
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    settings = self_hosted.opencode_settings(plan)
    assert settings["compaction"] == {"auto": True, "reserved": 20000}
    assert "plugin" not in settings
    assert plan["attempts"][0]["execution_generation"] == 2
    assert plan["attempts"][0]["run_id"] == spec["identities"]["run_id"]


@pytest.mark.parametrize("path", [Q_PLAN, G_PLAN])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launch_authorized", True),
        ("execution", {}),
        ("supersession", {}),
        ("identities", {}),
    ],
)
def test_generation2_plan_tampering_fails_closed(
    path: Path, field: str, value: object
) -> None:
    spec = copy.deepcopy(_load(path))
    spec[field] = value
    spec["plan_sha256"] = generation2.digest(spec, "plan_sha256")
    with pytest.raises(ValueError, match="generation-2 canary plan drifted"):
        generation2.validate_plan(spec, ROOT)


def test_generation1_tombstones_bind_zero_side_effect_incident_and_old_claims() -> None:
    bundle = _load(TOMBSTONES)
    generation2.validate_tombstones(bundle, ROOT)
    assert bundle["receipt_sha256"] == generation2.digest(bundle, "receipt_sha256")
    for row in bundle["models"]:
        predecessor = row["predecessor"]
        assert predecessor["attempt_directory_count"] == 0
        assert predecessor["fleet_session_count"] == 0
        assert predecessor["verifier_execution_count"] == 0
        assert row["claims"]["preserved_not_deleted_or_reused"] is True
        assert row["tombstone"]["execution_generation"] == 1
        assert row["tombstone"]["retry_allowed"] is True


def test_tombstone_side_effect_or_claim_drift_fails_closed() -> None:
    bundle = copy.deepcopy(_load(TOMBSTONES))
    bundle["models"][0]["tombstone"]["model_called"] = True
    bundle["models"][0]["tombstone"]["receipt_sha256"] = generation2.digest(
        bundle["models"][0]["tombstone"], "receipt_sha256"
    )
    bundle["receipt_sha256"] = generation2.digest(bundle, "receipt_sha256")
    with pytest.raises(ValueError):
        generation2.validate_tombstones(bundle, ROOT)

    bundle = copy.deepcopy(_load(TOMBSTONES))
    bundle["models"][0]["claims"]["preserved_not_deleted_or_reused"] = False
    bundle["receipt_sha256"] = generation2.digest(bundle, "receipt_sha256")
    with pytest.raises(ValueError, match="claim binding drifted"):
        generation2.validate_tombstones(bundle, ROOT)


def test_generation2_claim_is_fresh_and_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load(Q_PLAN)
    plan = generation2.validate_plan(spec, ROOT)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    receipt = generation2.claim_execution_generation(spec, plan, tmp_path / "claims")
    assert receipt["execution_generation"] == 2
    assert receipt["generation_1_claims_preserved"] is True
    assert receipt["receipt_sha256"] == generation2.digest(receipt, "receipt_sha256")
    claim = tmp_path / "claims" / f"{spec['execution']['execution_id'][7:]}.json"
    assert json.loads(claim.read_text()) == receipt
    with pytest.raises(RuntimeError, match="already claimed"):
        generation2.claim_execution_generation(spec, plan, tmp_path / "claims")


def test_held_release_binds_all_package_files_and_remains_nonlaunching() -> None:
    release = _load(HELD)
    generation2.validate_held(release, [_load(Q_PLAN), _load(G_PLAN)], ROOT)
    assert release["launch_authorized"] is False
    assert release["bulk_release_authorized"] is False
    assert release["dedicated_serving_authorized"] is False
    for prefix in ("module", "manifest", "run", "submit"):
        path = ROOT / release["implementation"][f"{prefix}_path"]
        assert release["implementation"][f"{prefix}_sha256"] == generation2.file_sha256(path)


def test_manifest_is_two_fresh_held_high_priority_never_preempted_jobs() -> None:
    docs = list(yaml.safe_load_all(MANIFEST.read_text()))
    assert {doc["metadata"]["name"] for doc in docs} == {
        "chris-q38-ac-r004-a1-g2-v1",
        "chris-glm53-ac-r013-a1-g2-v1",
    }
    assert len(docs) == 2
    for doc in docs:
        assert doc["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        assert doc["spec"]["backoffLimit"] == 0
        pod = doc["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-train-high"
        assert pod["preemptionPolicy"] == "Never"
        assert pod["restartPolicy"] == "Never"
        assert {mount["mountPath"] for mount in pod["initContainers"][0]["volumeMounts"]} >= {
            "/workspace",
            "/mnt/sfs",
        }


@pytest.mark.parametrize(
    ("plan_source", "predecessor_source"),
    [
        (Q_PLAN, Path("evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json")),
        (G_PLAN, Path("evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json")),
    ],
)
def test_manifest_shipped_workspace_passes_renderer_gate(
    tmp_path: Path, plan_source: Path, predecessor_source: Path
) -> None:
    root = tmp_path / "cyber-post-train"
    mappings = {
        Path("evals/fleet/self_hosted.py"): Path("evals/fleet/self_hosted.py"),
        Path("evals/fleet/opencode_train_sweep_runner.py"): Path(
            "evals/fleet/opencode_train_sweep_runner.py"
        ),
        Path("evals/fleet/hosted_sweep_controller.py"): Path(
            "evals/fleet/hosted_sweep_controller.py"
        ),
        Path("evals/fleet/autocontinue_canary_controller.py"): Path(
            "evals/fleet/autocontinue_canary_controller.py"
        ),
        Path("evals/fleet/autocontinue_canary_hosted_release.py"): Path(
            "evals/fleet/autocontinue_canary_hosted_release.py"
        ),
        Path("evals/fleet/autocontinue_canary_hosted_runtime.py"): Path(
            "evals/fleet/autocontinue_canary_hosted_runtime.py"
        ),
        Path("evals/fleet/autocontinue_hosted_health.py"): Path(
            "evals/fleet/autocontinue_hosted_health.py"
        ),
        Path("evals/fleet/endpoint_lease.py"): Path("evals/fleet/endpoint_lease.py"),
        Path("evals/fleet/fixed_proxy.py"): Path("evals/fleet/fixed_proxy.py"),
        Path("evals/fleet/exact_pass4_universe.py"): Path(
            "evals/fleet/exact_pass4_universe.py"
        ),
        Path("evals/fleet/autocontinue_generation2_canary.py"): Path(
            "evals/fleet/autocontinue_generation2_canary.py"
        ),
        Path("evals/fleet/configs/generation2-plan.json"): plan_source,
        Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"): Path(
            "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
        ),
        Path("evals/fleet/configs/opencode-easiest-train100-selection-v2.json"): Path(
            "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
        ),
        Path(generation2.INCIDENT_PATH): Path(generation2.INCIDENT_PATH),
        Path(generation2.TOMBSTONE_PATH): TOMBSTONES,
    }
    predecessor_name = generation2.EXPECTED[_load(plan_source)["model"]][
        "predecessor_plan_path"
    ]
    mappings[Path(predecessor_name)] = predecessor_source
    for target, source in mappings.items():
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (root / "evals/__init__.py").touch()
    (root / "evals/fleet/__init__.py").touch()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.fleet.autocontinue_generation2_canary",
            "validate-plan",
            "--plan",
            str(root / "evals/fleet/configs/generation2-plan.json"),
            "--repo",
            str(root),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_wrapper_renders_before_images_route_or_claims() -> None:
    source = RUN.read_text()
    renderer = source.index("validate-plan")
    docker = source.index("docker info")
    runtime = source.index("autocontinue_generation2_canary run")
    assert renderer < docker < runtime
    assert "GENERATION2_RENDERER_GATE_ONLY" in source


def test_submitter_is_create_once_held_and_has_no_local_sfs_authority() -> None:
    source = SUBMIT.read_text()
    validation = source.index("autocontinue_generation2_canary preview")
    server_preview = source.index("create --dry-run=server")
    absence = source.index('get configmap "$name"')
    held_exit = source.index("package is HELD")
    assert validation < server_preview < absence < held_exit
    assert "test ! -e /mnt/sfs" not in source
    assert "kubectl delete" not in source
    assert "kubectl apply" not in source
