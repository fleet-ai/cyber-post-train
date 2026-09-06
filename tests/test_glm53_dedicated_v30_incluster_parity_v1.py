import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v30_incluster_parity_v1 as rail
from evals.fleet import opencode_actual_harness_parity_v1 as parity

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "fec6d9b852ba4522646903a6ee4c76159305d6fd"


def binding() -> dict[str, object]:
    return {
        "server_title": rail.SERVER_TITLE,
        "server_run_dir": rail.SERVER_RUN_DIR,
        "api_run_id": "ft-run-1234abcd",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": (
            "http://ft-run-1234abcd-abcde-head-svc.fleet-train-jobs.svc:8000"
        ),
        "served_id": rail.SERVED_ID,
        "model_revision": rail.MODEL_REVISION,
        "context_length": rail.CONTEXT_LENGTH,
    }


def authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": rail.SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_INCLUSTER_PARITY",
        "observed_at_epoch": time.time(),
        "server_binding": binding(),
        "server_ready_receipt_sha256": "sha256:" + "1" * 64,
        "watchdog_active_receipt_sha256": "sha256:" + "2" * 64,
        "job_identity_absent": True,
        "result_root_absent": True,
        "cpu_priority_class": "fleet-serve-low",
        "cpu_preemption_policy": "Never",
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def parity_receipt() -> dict[str, object]:
    model_binding = rail.canonical_binding(binding())
    value: dict[str, object] = {
        "schema_version": parity.SCHEMA,
        "status": "PASSED_NON_SCORED",
        "classification": "ACTUAL_HARNESS_PARITY",
        "model": {},
        "endpoint": {
            "origin": binding()["service_origin"],
            "kind": "dedicated_uid_bound_inference",
            "server_binding": model_binding,
            "server_binding_sha256": crypto.sha256(
                crypto.canonical_json(model_binding)
            ),
        },
        "execution": {
            "harness_exit_code": 0,
            "model_requests": 4,
            "final_marker_observed": True,
            "docker_host_gateway_added": True,
            "task_instance_session_verifier_scoring_calls": 0,
            "scored_launch_authorized": False,
        },
        "harness": {},
        "tool_contract": {"calls_observed_in_order": ["bash", "submit_report"]},
        "privacy": {
            "credentials_included": False,
            "prompt_included": False,
            "responses_or_model_outputs_included": False,
            "tool_arguments_included": False,
            "stderr_or_stdout_included": False,
            "benchmark_content_included": False,
        },
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_linux_dind_transport_uses_host_network_and_loopback_urls() -> None:
    host, args = parity.docker_transport_args(
        docker_add_host_gateway=True, docker_network_host=True
    )
    assert host == "127.0.0.1"
    assert args == [
        "--network",
        "host",
        "--add-host",
        "host.docker.internal:host-gateway",
    ]
    settings = parity.render_settings("glm-5.3", 18080, 18081, transport_host=host)
    assert settings["provider"]["fleet-cluster"]["options"]["baseURL"] == (
        "http://127.0.0.1:18080/v1"
    )
    assert settings["mcp"]["fleet"]["url"] == "http://127.0.0.1:18081/mcp"
    with pytest.raises(parity.ActualHarnessParityError):
        parity.docker_transport_args(
            docker_add_host_gateway=False, docker_network_host=True
        )


def test_authorization_is_exact_fresh_score_free_and_generation_bound() -> None:
    value = authorization()
    rail.validate_authorization(value)
    for mutation in (
        {"extra": True},
        {"server_binding": {**binding(), "server_run_dir": "stale-v29"}},
        {"scored_launch_authorized": True},
        {"protected_content_included": True},
    ):
        changed = copy.deepcopy(value)
        changed.update(mutation)
        changed["receipt_sha256"] = crypto.digest_without(changed, "receipt_sha256")
        with pytest.raises(rail.ParityPackageError):
            rail.validate_authorization(changed)


def test_runtime_invokes_exact_linux_dind_transport_and_writes_uid_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return parity_receipt()

    monkeypatch.setattr(rail, "RESULT_ROOT", tmp_path / "result")
    shared_temp_root = tmp_path / "workspace" / "parity-tmp"
    shared_temp_root.mkdir(parents=True)
    monkeypatch.setattr(rail, "PARITY_TEMP_ROOT", str(shared_temp_root))
    monkeypatch.setattr(rail.parity, "run", run)
    result = rail.execute(
        authorization(),
        job_uid="55555555-5555-4555-8555-555555555555",
        pod_uid="66666666-6666-4666-8666-666666666666",
    )
    assert observed["args"] == ("glm-5.3", "")
    assert observed["kwargs"] == {
        "upstream_origin": binding()["service_origin"],
        "server_binding": rail.canonical_binding(binding()),
        "docker_add_host_gateway": True,
        "docker_network_host": True,
        "temp_root": shared_temp_root,
    }
    assert result["nested_container_network"] == "host"
    assert result["local_proxy_bind_address"] == "127.0.0.1"
    assert result["fleet_task_instance_calls"] == 0
    assert result["fleet_session_calls"] == 0
    assert result["verifier_calls"] == 0
    assert result["scoring_calls"] == 0
    assert result["receipt_sha256"] == crypto.digest_without(
        result, "receipt_sha256"
    )
    assert json.loads((rail.RESULT_ROOT / "PARITY.json").read_text())[
        "receipt_sha256"
    ] == parity_receipt()["receipt_sha256"]


def test_rendered_package_is_create_once_dind_and_held_from_scoring() -> None:
    rendered = rail.render(ROOT, COMMIT, authorization())
    configmap, auth, job = rendered["objects"]["items"]
    package = json.loads(configmap["data"]["package.json"])
    assert package["package_commit"] == COMMIT
    assert package["nested_container_network"] == "host"
    assert package["local_proxy_bind_address"] == "127.0.0.1"
    assert package["shared_temp_root"] == "/workspace/parity-tmp"
    assert package["fleet_api_secret_mounted"] is False
    assert auth["immutable"] is True
    assert job["metadata"]["name"] == rail.JOB_NAME
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["priorityClassName"] == (
        "fleet-serve-low"
    )
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert any(
        row["name"] == "dind" and row["securityContext"]["privileged"] is True
        for row in job["spec"]["template"]["spec"]["initContainers"]
    )
    pod_spec = job["spec"]["template"]["spec"]
    dind = next(
        row for row in pod_spec["initContainers"] if row["name"] == "dind"
    )
    runner = next(row for row in pod_spec["containers"] if row["name"] == "parity")
    dind_workspace = next(
        mount for mount in dind["volumeMounts"] if mount["name"] == "workspace"
    )
    runner_workspace = next(
        mount for mount in runner["volumeMounts"] if mount["name"] == "workspace"
    )
    assert dind_workspace == runner_workspace == {
        "name": "workspace",
        "mountPath": "/workspace",
    }
    env = {row["name"]: row for row in runner["env"]}
    assert env["TMPDIR"] == {"name": "TMPDIR", "value": "/workspace/parity-tmp"}
    assert "FLEET_API_KEY" not in env
    assert rendered["scored_launch_authorized"] is False


def test_parity_temp_root_requires_an_existing_absolute_nonsymlink_dir(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "workspace" / "parity-tmp"
    shared.mkdir(parents=True)
    assert parity.validate_temp_root(shared) == shared
    with pytest.raises(parity.ActualHarnessParityError, match="parity_temp_root_invalid"):
        parity.validate_temp_root(Path("relative"))
    link = tmp_path / "linked"
    link.symlink_to(shared, target_is_directory=True)
    with pytest.raises(parity.ActualHarnessParityError, match="parity_temp_root_invalid"):
        parity.validate_temp_root(link)


def test_materialized_run_reaches_and_enforces_its_digest_gate(tmp_path: Path) -> None:
    configmap = rail.build_configmap(ROOT, COMMIT)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for key in ("package.json", "run.sh"):
        (bootstrap / key).write_text(configmap["data"][key])
    run_path = bootstrap / "run.sh"
    env = {
        **os.environ,
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "PARITY_RESULT_ROOT": str(tmp_path / "result"),
        "BOOTSTRAP_ROOT": str(bootstrap),
        "PACKAGE_COMMIT": COMMIT,
        "PARITY_BOOTSTRAP_VERIFY_ONLY": "true",
        "TMPDIR": str(tmp_path / "workspace" / "parity-tmp"),
    }
    syntax = subprocess.run(
        ["bash", "-n", str(run_path)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    completed = subprocess.run(
        ["bash", str(run_path)], env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr

    run_path.write_text(run_path.read_text() + "# tampered\n")
    rejected = subprocess.run(
        ["bash", str(run_path)], env=env, capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0


def test_held_contract_never_authorizes_launch() -> None:
    held = rail.build_held()
    assert held["nested_container_network"] == "host"
    assert held["local_proxy_bind_address"] == "127.0.0.1"
    assert held["shared_temp_root"] == "/workspace/parity-tmp"
    assert held["fleet_api_secret_mounted"] is False
    assert held["endpoint_origin_must_equal_internal_service_origin"] is True
    assert held["qualification_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")
    path = (
        ROOT
        / "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v30-incluster-parity-held-v1.json"
    )
    assert json.loads(path.read_text()) == held
