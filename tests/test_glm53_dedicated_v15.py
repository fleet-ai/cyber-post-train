import json
import subprocess
from pathlib import Path

import httpx
import pytest

from evals.fleet import glm53_dedicated_v14 as v14
from evals.fleet import glm53_dedicated_v15 as v15
from evals.fleet import glm53_dedicated_v15_canary_heartbeat_v1 as heartbeat
from evals.fleet import glm53_dedicated_v15_canary_release_package_v1 as release_package
from evals.fleet import glm53_dedicated_v15_live as live
from evals.fleet import glm53_dedicated_v16 as v16
from evals.fleet import glm53_dedicated_v16_bootstrap_package_v1 as bootstrap
from evals.fleet import glm53_dedicated_v16_canary_launch_v1 as launch
from evals.fleet import glm53_dedicated_v17 as v17
from evals.fleet import glm53_dedicated_v17_live as v17_live
from evals.fleet import glm53_dedicated_runtime_gate_package_v1 as runtime_gate_package
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_v15_preserves_exact_v14_runtime_with_fresh_identity() -> None:
    old = v14.payload(v14.spec(ROOT), ROOT)
    new = v15.payload(v15.spec(ROOT), ROOT)
    assert new["title"] == v15.TITLE
    assert new["run_dir"] == v15.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v15.RUN_DIR}
    for field in set(old) - {"title", "run_dir", "env"}:
        assert new[field] == old[field]
    assert new["workers"] == 1
    assert new["gpus_per_worker"] == 8
    assert new["priority_class"] == "fleet-infra-quiet"


def test_v15_rejects_pre_admission_drift() -> None:
    value = v15.spec(ROOT)
    value["pre_admission"]["selection_rank"] = 52
    with pytest.raises(ValueError, match="pre-admission"):
        v15.validate(value, ROOT)


def _prepared() -> dict:
    value = {
        "schema_version": "fleet-glm53-dedicated-v14-scored-canary-preflight-v1",
        "status": "CLEAR_HELD",
        "launch_authorized": False,
        "controller_package_sha256": v15.PRE_ADMISSION["controller_package_sha256"],
        "controller_objects_sha256": "sha256:" + "1" * 64,
        "held_plan_sha256": v15.PRE_ADMISSION["held_plan_sha256"],
        "selection_rank": 51,
        "reserved_cell_ids": [f"sha256:{number:064x}" for number in range(1, 5)],
        "reserved_execution_ids": [f"sha256:{number:064x}" for number in range(5, 9)],
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "observer_job_uid": "4cf6c1ae-6533-4a1d-bcf9-a206297bd240",
        "observer_pod_uid": "dc79b655-90a2-4828-8d74-dd363704583b",
        "checked_at_utc": "2026-09-05T00:00:00Z",
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "fresh_server_binding_required_after_admission": True,
        "fresh_non_scored_parity_required_after_admission": True,
        "fresh_release_required_immediately_before_canary_create": True,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_prepared_validator_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _prepared()
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    monkeypatch.setitem(v15.PRE_ADMISSION, "file_sha256", self_hosted.sha256(raw))
    monkeypatch.setitem(v15.PRE_ADMISSION, "receipt_sha256", value["receipt_sha256"])
    assert live._validate_prepared(raw)["status"] == "CLEAR_HELD"
    changed = {**value, "fleet_session_collisions": 1}
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    changed_raw = json.dumps(changed, indent=2, sort_keys=True).encode() + b"\n"
    monkeypatch.setitem(v15.PRE_ADMISSION, "file_sha256", self_hosted.sha256(changed_raw))
    monkeypatch.setitem(v15.PRE_ADMISSION, "receipt_sha256", changed["receipt_sha256"])
    with pytest.raises(RuntimeError, match="fields drifted"):
        live._validate_prepared(changed_raw)


def test_project_shape_allows_only_exact_active_qwen_peer() -> None:
    row = {
        "name": "ft-run-qwen",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
    }
    shape = live._project_shape([row])
    assert shape["planned_nodes"] == 2
    assert shape["planned_gpus"] == 9
    with pytest.raises(RuntimeError, match="unknown active"):
        live._project_shape(
            [{"name": "ft-run-other", "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-other"}]
        )


def test_live_rows_reconciles_stale_list_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("stale"):
            return httpx.Response(404)
        return httpx.Response(200, json={"status": "RUNNING"})

    rows = [
        {"name": "ft-run-stale", "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-old"},
        {
            "name": "ft-run-qwen",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
        },
    ]
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://test") as client:
        assert [row["name"] for row in live._live_rows(client, rows)] == ["ft-run-qwen"]


def test_v18_release_package_binds_bootstrap_qualified_controller() -> None:
    parity = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json"
    binding = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-server-binding.json"
    built = release_package.render(
        ROOT,
        parity,
        binding,
        "http://glm-v17-head-svc.fleet-train-jobs.svc.cluster.local:8000",
    )
    assert built["controller_package_sha256"] == release_package.CONTROLLER_PACKAGE_SHA256
    configmap, job = built["objects"]["items"]
    assert configmap["metadata"]["name"].endswith("release-v5-run")
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_v15_heartbeat_is_uid_bound_and_nonpreempting() -> None:
    value = heartbeat.render(
        canary_job_uid="11111111-1111-4111-8111-111111111111",
        server_api_run_id="ft-run-v16",
        server_rayjob_uid="22222222-2222-4222-8222-222222222222",
        server_run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v16",
    )
    job = value["object"]
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    script = pod["containers"][0]["command"][2]
    compile(script, "heartbeat.py", "exec")
    assert "ft-run-v16" in script
    assert "11111111-1111-4111-8111-111111111111" in script
    assert value["heartbeat_path"].endswith("v16/lifecycle/traffic-stream-1")


def test_projected_configmap_evidence_is_copied_to_regular_files(tmp_path: Path) -> None:
    projected = tmp_path / "projected"
    version = projected / "..data" / "version"
    version.mkdir(parents=True)
    destination = tmp_path / "private"
    destination.mkdir(mode=0o700)
    for name in ("parity.json", "binding.json", "release.json"):
        source = version / name
        source.write_text('{"safe":true}\n')
        (projected / name).symlink_to(Path("..data/version") / name)
        assert (projected / name).is_symlink()
        subprocess.run(
            ["install", "-m", "0600", str(projected / name), str(destination / name)],
            check=True,
        )
        assert (destination / name).is_file()
        assert not (destination / name).is_symlink()
        assert (destination / name).read_bytes() == source.read_bytes()
    canary_script = (
        ROOT / "evals/fleet/scripts/run_glm53_dedicated_v14_scored_canary_v1.sh"
    ).read_text()
    release_script = (
        ROOT / "evals/fleet/scripts/run_glm53_dedicated_v15_canary_release_v1.sh"
    ).read_text()
    assert "install -m 0600 /evidence/parity.json" in canary_script
    assert "install -m 0600 /evidence/release.json" in canary_script
    assert "install -m 0600 /bootstrap/parity.json" in release_script
    assert '--controller-package "$EVIDENCE_DIR/controller-package.json"' in release_script


def test_v16_preserves_v15_runtime_and_binds_corrected_preflight() -> None:
    old = v15.payload(v15.spec(ROOT), ROOT)
    new = v16.payload(v16.spec(ROOT), ROOT)
    for field in set(old) - {"title", "run_dir", "env"}:
        assert new[field] == old[field]
    assert new["title"] == v16.TITLE
    assert new["run_dir"] == v16.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v16.RUN_DIR}
    assert v16.PRE_ADMISSION["receipt_sha256"].endswith("5f9339d")
    assert v16.PRE_ADMISSION["controller_package_sha256"] != release_package.CONTROLLER_PACKAGE_SHA256


def test_v16_canary_launch_binds_current_controller(tmp_path: Path) -> None:
    parity = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json"
    binding = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-server-binding.json"
    origin = "http://glm-v16-head-svc.fleet-train-jobs.svc.cluster.local:8000"
    plan = launch.canary.build_plan(
        ROOT, service_origin=origin, parity_path=parity, binding_path=binding
    )
    item = plan["attempts"][0]
    release = {
        "schema_version": launch.runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "plan_sha256": "sha256:" + "4" * 64,
        "selection_rank": 51,
        "attempt": 1,
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "sfs_output_collisions": 0,
        "kubernetes_object_collisions": 0,
        "checked_immediately_before_create": True,
    }
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    built = launch.render(
        ROOT,
        parity_path=parity,
        binding_path=binding,
        release_path=release_path,
        service_origin=origin,
    )
    assert built["controller_package_sha256"] == release_package.CONTROLLER_PACKAGE_SHA256


def test_v16_bootstrap_runs_exact_controller_source_and_stops_pre_model() -> None:
    built = bootstrap.render(ROOT)
    source, evidence, job = built["objects"]["items"]
    final = bootstrap.controller.render(ROOT)
    assert source["data"] == final["objects"]["items"][0]["data"]
    assert built["uses_exact_controller_source_data"] is True
    assert built["controller_package_sha256"] == final["package_sha256"]
    assert evidence["immutable"] is True
    env = {
        item["name"]: item.get("value")
        for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    names = [
        item["name"]
        for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
    ]
    assert len(names) == len(set(names))
    assert env["DEDICATED_BOOTSTRAP_ONLY"] == "1"
    assert env["DEDICATED_CONTROLLER_PACKAGE_SHA256"] == final["package_sha256"]
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert evidence["data"]["parity.json"] != "{}\n"
    assert evidence["data"]["binding.json"] != "{}\n"


def test_v17_preserves_server_runtime_and_binds_both_admission_receipts() -> None:
    old = v15.payload(v15.spec(ROOT), ROOT)
    new = v17.payload(v17.spec(ROOT), ROOT)
    for field in set(old) - {"title", "run_dir", "env"}:
        assert new[field] == old[field]
    assert new["title"] == v17.TITLE
    assert new["run_dir"] == v17.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v17.RUN_DIR}
    assert v17.PRE_ADMISSION["controller_package_sha256"] == v17.CONTROLLER_BOOTSTRAP["controller_package_sha256"]
    assert v17.CONTROLLER_BOOTSTRAP["controller_package_sha256"] == (
        "sha256:86468d5893a7f8be20315a82b44532fcfa9d53fc5fff937f431197aca9a0b249"
    )
    assert release_package.CONTROLLER_PACKAGE_SHA256 != v17.CONTROLLER_BOOTSTRAP[
        "controller_package_sha256"
    ]
    assert v17.CONTROLLER_BOOTSTRAP["receipt_sha256"].endswith("e3661333")
    assert v17.RUNTIME_GATE["receipt_sha256"].endswith("9c7930")


def test_v17_bootstrap_validator_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    value = {
        "schema_version": "fleet-glm53-dedicated-controller-bootstrap-v1",
        "status": "PASSED_PRE_MODEL",
        "job_uid": v17.CONTROLLER_BOOTSTRAP["job_uid"],
        "pod_uid": v17.CONTROLLER_BOOTSTRAP["pod_uid"],
        "controller_package_sha256": v17.CONTROLLER_BOOTSTRAP["controller_package_sha256"],
        "harness_image": "chris/opencode:1.18.27-cyber-v1",
        "harness_version": "1.18.27",
        "projected_evidence_copied_to_private_regular_files": True,
        "runtime_import_closure_valid": True,
        "docker_build_and_version_check_completed_by_exact_run_sh": True,
        "claim_calls": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    raw = self_hosted.canonical_json(value) + b"\n"
    monkeypatch.setattr(v17_live.shared, "_observer_sfs_path", lambda _: "/receipt")
    monkeypatch.setattr(v17_live.v17, "CONTROLLER_BOOTSTRAP", {**v17.CONTROLLER_BOOTSTRAP, "file_sha256": self_hosted.sha256(raw), "receipt_sha256": value["receipt_sha256"]})
    class Done:
        stdout = raw
    monkeypatch.setattr(v17_live.subprocess, "run", lambda *a, **k: Done())
    assert v17_live._bootstrap()[1]["status"] == "PASSED_PRE_MODEL"


def test_runtime_gate_observer_is_non_scored_and_uses_actual_evidence() -> None:
    parity = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json"
    binding = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-server-binding.json"
    release = ROOT / "docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json"
    objects = runtime_gate_package.render(
        ROOT, parity, binding, release,
        "http://glm-v17-head-svc.fleet-train-jobs.svc.cluster.local:8000",
    )
    source, job = objects["items"]
    assert source["data"]["parity.json"] == parity.read_text()
    assert "_runtime_gate(plan)" in source["data"]["gate.py"]
    assert "run_controller" not in source["data"]["gate.py"]
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
