import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation10_qwen_package_v1 as package
from evals.fleet import autocontinue_generation10_qwen_v1 as runtime
from evals.fleet import kubernetes_create_relay as relay

ROOT = Path(__file__).parents[1]


def _duplicate(spec: dict) -> dict:
    value = {
        "schema_version": runtime.DUPLICATE_SCHEMA,
        "status": "FRESH_ABSENT",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": runtime.MODEL,
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "identities": {
            "job_name": runtime.JOB_NAME,
            "configmap_name": runtime.CONFIGMAP_NAME,
            "output_root": runtime.OUTPUT_ROOT,
        },
        "kubernetes": {
            "job_absent": True,
            "pods_absent": True,
            "configmap_absent": True,
            "checked_namespace": runtime.NAMESPACE,
        },
        "sfs": {
            "output_root_absent": True,
            "generation7_claim_present": True,
            "generation7_claim_receipt_sha256": "sha256:" + "1" * 64,
            "generation8_claim_absent": True,
            "generation9_claim_absent": True,
            "generation10_claim_absent": True,
        },
        "fleet": {
            f"generation{generation}_matching_session_rows": 0
            for generation in (8, 9, 10)
        },
        "secret": {
            "name": runtime.SECRET_NAME,
            "uid": runtime.SECRET_UID,
            "key_names": [runtime.SECRET_KEY],
            "value_read": False,
            "value_persisted": False,
        },
        "account": {"team_name": "fleet", "team_id": runtime.FLEET_TEAM_ID},
        "response_bodies_persisted": False,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    value["fleet"]["session_rows_examined"] = 0
    value["receipt_sha256"] = runtime.digest(value)
    return value


def _route() -> dict:
    return {
        "schema_version": "test-route",
        "receipt_sha256": "sha256:" + "2" * 64,
    }


def test_qwen_static_is_independent_and_exact() -> None:
    spec, plan, held = runtime.static(ROOT)
    assert spec["execution"] == {
        "schema_version": "fleet-statistical-cell-execution-v1",
        "cell_id": "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a",
        "execution_generation": 10,
        "execution_id": "sha256:84df092e1904306b308326188609a520f9ffd870a45fad26f65ad1eb60f3aa0b",
    }
    assert plan["campaign_id"] == runtime.JOB_NAME
    assert spec["identities"]["configmap_name"] == runtime.CONFIGMAP_NAME
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["harness"]["context_window_size"] == 262144
    assert held["launch_authorized"] is False
    assert runtime.preparer.MODELS["glm-5.3"]["tombstone"] not in package.QWEN_PATHS


def test_held_package_is_bounded_and_contains_corrected_runtime() -> None:
    built = package.build_package(ROOT)
    assert built["launch_authorized"] is False
    assert built["release_included"] is False
    assert set(built["configmaps"]) == {
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.MODEL_NAME,
    }
    assert all(
        size < package.base.PACKAGE_OBJECT_LIMIT for size in built["object_json_bytes"].values()
    )
    manifest = built["model_manifest"]
    assert manifest["model"] == runtime.MODEL
    assert (
        manifest["generation10_spec_sha256"] == runtime.static(ROOT)[0]["generation10_spec_sha256"]
    )
    paths = {row["source_path"] for obj in manifest["objects"] for row in obj["entries"]}
    assert runtime.MODULE_PATH in paths
    assert runtime.RUN_PATH in paths
    assert runtime.preparer.MODULE_PATH in paths
    assert "evals/fleet/self_hosted.py" in paths


def test_qwen_package_imports_from_an_isolated_materialization(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    for obj in built["model_manifest"]["objects"]:
        data = built["configmaps"][obj["name"]]["data"]
        for entry in obj["entries"]:
            target = tmp_path / entry["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(data[entry["data_key"]])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from evals.fleet import autocontinue_generation10_qwen_v1 as runtime; "
                "runtime.static(Path.cwd())"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_duplicate_preflight_is_fail_closed() -> None:
    spec = runtime.static(ROOT)[0]
    value = _duplicate(spec)
    runtime.validate_duplicate(value, spec)
    changed = copy.deepcopy(value)
    changed["fleet"]["generation10_matching_session_rows"] = 1
    changed["receipt_sha256"] = runtime.digest(changed)
    with pytest.raises(ValueError, match="duplicate preflight"):
        runtime.validate_duplicate(changed, spec)
    stale = copy.deepcopy(value)
    stale["observed_at_utc"] = (
        (datetime.now(UTC) - timedelta(seconds=301)).isoformat().replace("+00:00", "Z")
    )
    stale["receipt_sha256"] = runtime.digest(stale)
    with pytest.raises(ValueError, match="duplicate preflight"):
        runtime.validate_duplicate(stale, spec)


def test_release_binds_package_duplicate_route_secret_and_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = runtime.static(ROOT)[0]
    built = package.build_package(ROOT)
    duplicate = _duplicate(spec)
    route = _route()
    monkeypatch.setattr(runtime.hosted_runtime, "validate_live_route", lambda *a, **k: None)
    release = runtime.build_release(ROOT, built["model_manifest"], "1" * 40, duplicate, route)
    assert release["launch_authorized"] is True
    assert release["fresh_duplicate_preflight_receipt_sha256"] == duplicate["receipt_sha256"]
    assert release["secret"] == {
        "name": runtime.SECRET_NAME,
        "uid": runtime.SECRET_UID,
        "key": runtime.SECRET_KEY,
    }
    assert release["fleet_team_id"] == runtime.FLEET_TEAM_ID
    assert release["authorization"] == {
        "execution_generation": 10,
        "same_statistical_cell": True,
        "hosted_only": True,
        "priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
        "cpu_only": True,
        "create_once": True,
        "bulk_release_authorized": False,
    }


def test_read_only_observer_produces_exact_duplicate_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tombstone = runtime.preparer.read_canonical(ROOT / runtime.TOMBSTONE_PATH)
    g7_receipt = tombstone["generation7_global_claim"]["receipt_sha256"]
    monkeypatch.setenv("SFS_OBSERVER_UID", "11111111-1111-4111-8111-111111111111")

    def command(argv: list[str]) -> str:
        joined = " ".join(argv)
        if "jsonpath={.metadata.uid}" in joined and "allie-dev" in joined:
            return "11111111-1111-4111-8111-111111111111"
        if "jsonpath={.status.phase}" in joined:
            return "Running"
        if "get secret" in joined and "jsonpath={.metadata.uid}" in joined:
            return runtime.SECRET_UID
        if "go-template=" in joined:
            return runtime.SECRET_KEY
        if "get job" in joined or "get pod -l" in joined or "get configmap" in joined:
            return ""
        raise AssertionError(argv)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime, "_allie_exists", lambda _path: False)
    monkeypatch.setattr(runtime, "_allie_claim", lambda _path: {"receipt_sha256": g7_receipt})
    monkeypatch.setattr(runtime, "_validate_g7_claim", lambda _value, _tombstone: None)
    monkeypatch.setattr(runtime.hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(
        runtime.self_hosted,
        "_request",
        lambda _client, method, path: (
            {"team_name": "fleet", "team_id": runtime.FLEET_TEAM_ID}
            if (method, path) == ("GET", "/v1/account")
            else None
        ),
    )
    from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

    monkeypatch.setattr(reconciliation, "_default_sessions", lambda _task, _key: [])
    value = runtime.observe_duplicate(ROOT, "not-persisted")
    runtime.validate_duplicate(value, runtime.static(ROOT)[0])
    assert value["status"] == "FRESH_ABSENT"
    assert value["fleet"]["session_rows_examined"] == 0
    assert value["secret"]["value_read"] is False
    assert value["response_bodies_persisted"] is False
    assert "not-persisted" not in json.dumps(value)


def test_g7_global_claim_requires_its_own_digest() -> None:
    body = {"schema_version": "test-global-claim", "immutable": True}
    claim = {**body, "receipt_sha256": runtime.digest(body)}
    tombstone = {"generation7_global_claim": {"receipt_sha256": claim["receipt_sha256"]}}
    runtime._validate_g7_claim(claim, tombstone)  # noqa: SLF001
    changed = {**claim, "immutable": False}
    with pytest.raises(RuntimeError, match="global claim drifted"):
        runtime._validate_g7_claim(changed, tombstone)  # noqa: SLF001


def test_package_manifest_cli_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "package.json"
    assert package.main is not None
    # Exercise the same write primitive used by the CLI and prove canonical,
    # exclusive output bytes without invoking a subprocess.
    from evals.fleet import self_hosted

    value = package.build_package(ROOT)["model_manifest"]
    self_hosted.write_json_once(output, value)
    assert output.read_bytes() == package.canonical(value) + b"\n"
    with pytest.raises(FileExistsError):
        self_hosted.write_json_once(output, value)


def test_held_manifest_is_one_cpu_never_preempt_job() -> None:
    job = yaml.safe_load((ROOT / runtime.MANIFEST_PATH).read_text())
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == runtime.JOB_NAME
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert (
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/secret-uid"] == runtime.SECRET_UID
    )
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert "gpu" not in json.dumps(job).lower()
    env = {row["name"]: row for row in pod["containers"][0]["env"]}
    assert env["FLEET_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": runtime.SECRET_NAME,
        "key": runtime.SECRET_KEY,
    }
    released = copy.deepcopy(job)
    released["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "true"
    for configmap in package.build_package(ROOT)["configmaps"].values():
        relay.validate_object(configmap)
    relay.validate_object(released)


def test_scripts_preserve_optimized_path_and_create_once_guards() -> None:
    run = (ROOT / runtime.RUN_PATH).read_text()
    submit = (ROOT / runtime.SUBMIT_PATH).read_text()
    assert "autocontinue_generation10_qwen_v1 run" in run
    assert "autocontinue_generation7_authority_v1 run" not in run
    assert "docker build --pull --platform linux/amd64" in run
    assert 'launch-authorized: "false"' in (ROOT / runtime.MANIFEST_PATH).read_text()
    assert "get secret \"$SECRET\" -o jsonpath='{.metadata.uid}'" in submit
    assert "base64 --decode" not in submit
    assert "kubernetes_create_relay validate" in submit
    assert "kubernetes_create_relay create" in submit
    assert "render-package-manifest" in submit
    assert "tool-rendered package manifest drifted" in submit
    assert "observe-duplicate" in (ROOT / runtime.MODULE_PATH).read_text()
    assert submit.count('get job "$name" --ignore-not-found') == 2
    assert "SFS_OBSERVER_UID" in submit
    assert "b2991e6062643b6a7875d3dd8d677a63e0f848323c2faedf3022b16ea1616d98" in submit
