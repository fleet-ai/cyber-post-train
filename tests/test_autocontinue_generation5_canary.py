from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from evals.fleet import autocontinue_generation2_canary_v2 as generation2
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import autocontinue_generation4_canary as generation4
from evals.fleet import autocontinue_generation5_authority_package_v1 as package
from evals.fleet import autocontinue_generation5_canary as generation5
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

ROOT = Path.cwd()


@pytest.fixture(scope="module")
def built() -> dict:
    return package.build_package(ROOT)


def test_incident_and_tombstones_bind_zero_side_effects_and_both_causes() -> None:
    incident = generation5.load(ROOT / generation5.INCIDENT_PATH)
    tombstones = generation5.load(ROOT / generation5.TOMBSTONE_PATH)
    generation5.validate_incident(incident, ROOT)
    generation5.validate_tombstones(tombstones, ROOT)
    assert incident["side_effects"] == {
        "model_calls": 0,
        "sessions": 0,
        "verifier_executions": 0,
        "generation4_claims": 0,
        "output_roots": 0,
    }
    assert incident["bootstrap_failure_proof"]["failures"] == [
        "release_file_sha256_bound_pretty_repository_bytes_but_mounted_canonical_bytes",
        "model_projection_omitted_opposite_generation4_spec_and_transitive_predecessor_specs",
    ]
    assert incident["bootstrap_failure_proof"]["runner_reached_route_claim_or_model"] is False


@pytest.mark.parametrize("field", ["sessions", "model_calls"])
def test_incident_resealed_side_effect_tamper_rejects(field: str) -> None:
    receipt = generation5.load(ROOT / generation5.INCIDENT_PATH)
    changed = copy.deepcopy(receipt)
    changed["side_effects"][field] = 1
    changed["receipt_sha256"] = generation5.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="incident"):
        generation5.validate_incident(changed, ROOT)


def test_generation5_specs_preserve_cells_and_use_fresh_execution_identities() -> None:
    for model, expected in generation5.EXPECTED.items():
        spec = generation5.load(ROOT / generation5.G5_SPEC_PATHS[model])
        plan = generation5.validate_spec(spec, ROOT)
        old = generation5.load(ROOT / generation4.G4_SPEC_PATHS[model])
        assert spec["statistical_cell"] == old["statistical_cell"]
        assert spec["execution"]["execution_generation"] == 5
        assert spec["execution"]["execution_id"] != old["execution"]["execution_id"]
        assert plan["scored_job_name"] == expected["job_name"]
        assert plan["attempts"][0]["run_id"] == expected["run_id"]
        assert plan["execution"]["required_priority_class"] == "fleet-serve-low"
        assert plan["execution"]["launch_authorized"] is False


def test_package_includes_missing_dependencies_and_fits_configmaps(built: dict) -> None:
    paths = {
        entry["source_path"]
        for obj in built["object_manifests"].values()
        for entry in obj["entries"]
    }
    required_specs = {
        *generation2.V1_SPEC_PATHS.values(),
        *generation3.G2_SPEC_PATHS.values(),
        *generation3.G3_SPEC_PATHS.values(),
        *generation4.G4_SPEC_PATHS.values(),
        *generation5.G5_SPEC_PATHS.values(),
    }
    assert required_specs <= paths
    assert "evals/fleet/exact_pass4_crypto.py" in paths
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.PACKAGE_OBJECT_LIMIT


def test_projected_generation5_attempt_uses_live_task_uuid() -> None:
    spec = generation5.load(ROOT / generation5.G5_SPEC_PATHS["qwen3.8-27b"])
    plan = generation5.validate_spec(spec, ROOT)
    config = hosted._attempt_config(plan, plan["tasks"][0], plan["attempts"][0])
    assert "id" not in config["task"]
    live_id = "11111111-1111-4111-8111-111111111111"
    task = {
        "id": live_id,
        "metadata": {
            "runtime_seed_manifest": {
                "files": [
                    {"target_path": "/task/a", "s3_key": "key/a", "bucket": "bucket"}
                ]
            }
        },
    }
    assert self_hosted.build_instance_payload(config, task)["task_id"] == live_id


def test_scored_generation5_bootstrap_reaches_authoritative_provisioning_without_task_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = generation5.load(ROOT / generation5.G5_SPEC_PATHS["qwen3.8-27b"])
    plan = generation5.validate_spec(spec, ROOT)
    config = hosted._attempt_config(plan, plan["tasks"][0], plan["attempts"][0])
    assert "id" not in config["task"]

    class ProvisioningReached(RuntimeError):
        pass

    class Client:
        def post(self, url: str, **kwargs: object) -> None:
            assert url.endswith(self_hosted.authoritative_route(config, "provisioning"))
            assert kwargs["json"] == {}
            raise ProvisioningReached

        def close(self) -> None:
            return None

    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("AGENT_HARNESS_IMAGE", "agent@sha256:test")
    monkeypatch.setenv("FIXED_PROXY_IMAGE", "proxy@sha256:test")
    monkeypatch.setattr(self_hosted.httpx, "Client", lambda **kwargs: Client())
    monkeypatch.setattr(
        self_hosted,
        "_request",
        lambda client, method, route, **kwargs: {
            "team_name": "fleet",
            "team_id": self_hosted.FLEET_TEAM_ID,
        },
    )
    monkeypatch.setattr(
        self_hosted,
        "load_and_verify_task",
        lambda client, current: {
            "id": "11111111-1111-4111-8111-111111111111",
            "prompt": "sealed-test-prompt",
        },
    )
    monkeypatch.setattr(
        self_hosted, "assert_authoritative_routes_deployed", lambda client, current: {}
    )
    monkeypatch.setattr(
        self_hosted,
        "build_instance_payload",
        lambda *args, **kwargs: pytest.fail(
            "scored provisioning must not use the legacy instance payload"
        ),
    )
    monkeypatch.setattr(
        self_hosted,
        "_docker",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    with pytest.raises(ProvisioningReached):
        self_hosted.run(config, tmp_path / "attempt", tmp_path / "proxy.py")
