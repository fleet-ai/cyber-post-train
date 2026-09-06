from pathlib import Path

from evals.fleet import hosted_glm_s1_r2_c2_package_v2 as package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v2 as release_package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v3 as release_package_v3
from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as successor
from evals.fleet import hosted_glm_s1_r2_c2_package_v3 as package_v3
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v4 as release_package_v4
from evals.fleet import hosted_glm_s1_r2_c2_bootstrap_package_v1 as bootstrap_package
from evals.fleet import hosted_glm_s1_r2_c2_bootstrap_package_v2 as bootstrap_package_v2
from evals.fleet import hosted_glm_s1_r2_c2_package_v4 as package_v4
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v5 as release_package_v5
from evals.fleet import hosted_glm_s1_r2_c2_successor_v4 as successor_v4
from evals.fleet import exact_pass4_bulk_runtime_v3 as engine

ROOT = Path(__file__).resolve().parents[1]


def test_v2_is_fresh_complete_rank2_and_retains_max_two_lease():
    plan = successor.validate_all(ROOT)
    assert plan["job_name"].endswith("successor-v2")
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (2, 1), (2, 2), (2, 3), (2, 4)
    ]
    assert plan["partition"]["v1_preclaim_failure_excluded"] is True
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2


def test_v2_packages_are_create_once_nonpreempting_and_held_until_release():
    held = package.render(ROOT)
    assert held["launch_authorized"] is False
    job = held["objects"]["items"][1]
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    observer = release_package.render(ROOT)["objects"]["items"][1]
    assert observer["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_v2_runtime_accepts_zero_or_one_preexisting_slot_but_never_two():
    source = (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py").read_text()
    runtime = (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_runtime_v2.py").read_text()
    assert "occupied not in (0, 1)" in source
    assert "_active_lease_slots() > 1" in runtime
    assert '"maximum_streams": 2' in (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_successor_v1.py").read_text()


def test_fresh_release_installs_the_full_immutable_config_set():
    data = release_package_v3.render(ROOT)["objects"]["items"][0]["data"]
    script = data["run.sh"]
    assert "mkdir -p \"$ROOT/evals/fleet/configs\"" in script
    assert "campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json" in script
    assert "successor_release_v2.py:hosted_glm_s1_r2_c2_release_v2.py" in script


def test_v3_controller_and_release_have_complete_rendered_install_closure():
    controller_data = package_v3.render(ROOT)["objects"]["items"][0]["data"]
    package_v3.validate_install_closure(controller_data)
    assert "original_release.py" in controller_data
    release_data = release_package_v4.render(ROOT)["objects"]["items"][0]["data"]
    package_v3.validate_install_closure(release_data)


def test_v3_scored_package_requires_exact_bootstrap_receipt():
    release = ROOT / "docs/evidence/glm53-study/2026-09-06-glm53-hosted-rank2-c2-release-v4.json"
    bootstrap = ROOT / "docs/evidence/glm53-study/2026-09-06-glm53-hosted-rank2-c2-bootstrap-v1.json"
    assert package_v3.render(ROOT, release_path=release, bootstrap_path=bootstrap)["launch_authorized"] is True
    try:
        package_v3.render(ROOT, release_path=release)
    except ValueError as exc:
        assert "release and exact bootstrap" in str(exc)
    else:
        raise AssertionError("scored package was authorized without CPU bootstrap")


def test_install_closure_rejects_the_exact_v2_omission():
    data = package_v3.render(ROOT)["objects"]["items"][0]["data"].copy()
    del data["original_release.py"]
    try:
        package_v3.validate_install_closure(data)
    except ValueError as exc:
        assert "original_release.py" in str(exc)
    else:
        raise AssertionError("missing install source was accepted")


def test_bootstrap_executes_exact_controller_package_with_claims_disabled(tmp_path):
    from evals.fleet import self_hosted
    from evals.fleet import hosted_glm_s1_r2_c2_release_v4 as release
    plan = __import__("evals.fleet.hosted_glm_s1_r2_c2_successor_v3", fromlist=["x"]).validate_all(ROOT)
    receipt = {
        "schema_version": release.SCHEMA,
        "status": "CLEAR",
        "successor_job": plan["job_name"],
        "successor_configmap": plan["configmap_name"],
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    path = tmp_path / "release.json"
    path.write_bytes(self_hosted.canonical_json(receipt))
    rendered = bootstrap_package.render(ROOT, release_path=path)
    assert rendered["claims_authorized"] is False
    env = {row["name"]: row.get("value") for row in rendered["objects"]["items"][1]["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["HOSTED_BOOTSTRAP_ONLY"] == "1"
    assert env["HOSTED_BOOTSTRAP_RECEIPT"].endswith("/BOOTSTRAP.json")


def test_v4_is_a_complete_engine_adapter_with_real_signatures(monkeypatch):
    engine.validate_bulk_adapter(successor_v4)
    plans = successor_v4.validate_all(ROOT)
    assert list(plans) == [successor_v4.CONTROLLER]
    held = plans[successor_v4.CONTROLLER]
    assert held["job_name"].endswith("successor-v4")
    assert held["execution"]["endpoint_lease"]["maximum_streams"] == 2


def test_v4_packages_have_complete_projected_install_closure():
    data = package_v4.render(ROOT)["objects"]["items"][0]["data"]
    package_v3.validate_install_closure(data)
    release_data = release_package_v5.render(ROOT)["objects"]["items"][0]["data"]
    package_v3.validate_install_closure(release_data)
