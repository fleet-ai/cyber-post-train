from __future__ import annotations

import json
import subprocess
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from evals.fleet import exact_pass4_final_duplicate_observer_v5 as observer
from evals.fleet import exact_pass4_final_duplicate_package_v5 as package
from evals.fleet import exact_pass4_final_duplicate_renderer_v5 as renderer
from evals.fleet import kubernetes_create_relay as relay


def _uid() -> str:
    return str(uuid.uuid4())


def _fake_bulk(tmp_path: Path) -> types.SimpleNamespace:
    attempts = [
        {
            "cell_id": "cell-a",
            "execution_id": "sha256:" + "a" * 64,
            "run_id": "run-a",
            "task_key": "task-a",
            "task_version_id": "task-version-a",
            "environment_version_id": "environment-version-a",
        },
        {
            "cell_id": "cell-b",
            "execution_id": "sha256:" + "b" * 64,
            "run_id": "run-b",
            "task_key": "task-b",
            "task_version_id": "task-version-b",
            "environment_version_id": "environment-version-b",
        },
    ]
    plans = {
        "controller-a": {
            "job_name": "chris-cyber-controller-a",
            "configmap_name": "chris-cyber-controller-a-run",
            "sfs_root": str(tmp_path / "controller-a"),
            "attempts": attempts,
            "controller": "controller-a",
            "plan_sha256": "sha256:" + "c" * 64,
            "model": "qwen3.8-27b",
            "serving_kind": "hosted",
            "serving_block": "qwen-hosted-autocontinue-v1",
        }
    }

    def validate_fresh(value, group, root, package_commit):
        assert group == "hosted-qwen"
        assert value["schema_version"] == "fleet-exact-pass4-final-bulk-fresh-duplicate-v5"
        assert value["observer_package_commit"] == package_commit
        observed = datetime.strptime(value["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
        assert (datetime.now(UTC) - observed).total_seconds() <= 900
        assert value["checked_cell_ids"] == ["cell-a", "cell-b"]

    return types.SimpleNamespace(
        GROUPS={"hosted-qwen": ["controller-a"]},
        FRESH_SCHEMA="fleet-exact-pass4-final-bulk-fresh-duplicate-v5",
        hosted=types.SimpleNamespace(CLAIM_ROOT=str(tmp_path / "claims")),
        validate_all=lambda _root: plans,
        _group_identity=lambda group, _root: {"group": group, "attempts": attempts},
        canonical=observer.canonical,
        sha256=observer.sha256,
        validate_fresh_duplicate=validate_fresh,
    )


def _metadata(kind: str, name: str):
    if kind == "secret":
        return {
            "metadata": {
                "name": observer.SECRET_NAME,
                "namespace": observer.NAMESPACE,
                "uid": observer.SECRET_UID,
            }
        }
    return [] if kind == "pods" else None


def _account(_key: str) -> dict[str, str]:
    return {"team_name": "fleet", "team_id": observer.FLEET_TEAM_ID}


def test_collect_binds_exact_cells_and_is_score_blind(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, "_bulk", lambda: _fake_bulk(tmp_path))
    monkeypatch.setattr(
        observer,
        "_evidence",
        lambda: types.SimpleNamespace(_default_sessions=lambda _task, _key: []),
    )
    receipt = observer.collect(
        tmp_path,
        "hosted-qwen",
        job_uid=_uid(),
        pod_uid=_uid(),
        package_commit="1" * 40,
        metadata=_metadata,
        account=_account,
        api_key="redacted-test-key",
        claim_root=tmp_path / "claims",
    )
    assert receipt["checked_cell_ids"] == ["cell-a", "cell-b"]
    assert receipt["collisions"] == {
        "fleet_api": 0,
        "kubernetes_job_pod_or_configmap": 0,
        "sfs_output": 0,
        "global_claim": 0,
        "accepted_active_or_model_started_cell": 0,
    }
    serialized = json.dumps(receipt)
    assert "redacted-test-key" not in serialized
    assert receipt["prompts_traces_flags_or_scores_included"] is False


@pytest.mark.parametrize("identity", ["cell-a", "run-a", "sha256:" + "a" * 64])
def test_claim_collision_fails_closed(tmp_path, monkeypatch, identity):
    bulk = _fake_bulk(tmp_path)
    monkeypatch.setattr(observer, "_bulk", lambda: bulk)
    binding = observer.group_binding(tmp_path, "hosted-qwen")
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "unrelated.json").write_text(json.dumps({"cell_id": identity}))
    with pytest.raises(observer.DuplicateError, match="global_generation_claim_collision"):
        observer._claim_files_clear(claims, binding)


def test_claim_protected_content_fails_closed(tmp_path, monkeypatch):
    bulk = _fake_bulk(tmp_path)
    monkeypatch.setattr(observer, "_bulk", lambda: bulk)
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "unrelated.json").write_text('{"scores":[0]}')
    with pytest.raises(observer.DuplicateError, match="protected_claim_key_forbidden"):
        observer._claim_files_clear(claims, observer.group_binding(tmp_path, "hosted-qwen"))


def test_session_and_kubernetes_collisions_fail_closed(tmp_path, monkeypatch):
    bulk = _fake_bulk(tmp_path)
    monkeypatch.setattr(observer, "_bulk", lambda: bulk)
    binding = observer.group_binding(tmp_path, "hosted-qwen")
    with pytest.raises(observer.DuplicateError, match="fleet_session_collision"):
        observer._session_absence(
            binding, lambda _task, _key: [{"metadata": {"run_id": "run-a"}}], "key"
        )
    with pytest.raises(observer.DuplicateError, match="kubernetes_job_or_pod_collision"):
        observer._metadata_absence(binding, lambda kind, _name: {} if kind == "job" else [])


def test_accept_requires_source_success_and_release_checks_both_jobs(tmp_path, monkeypatch):
    bulk = _fake_bulk(tmp_path)
    monkeypatch.setattr(observer, "_bulk", lambda: bulk)
    source_uid, source_pod = _uid(), _uid()
    accept_uid, accept_pod = _uid(), _uid()
    observed = observer.collect(
        tmp_path,
        "hosted-qwen",
        job_uid=source_uid,
        pod_uid=source_pod,
        package_commit="2" * 40,
        sessions=lambda _task, _key: [],
        metadata=_metadata,
        account=_account,
        api_key="redacted-test-key",
        claim_root=tmp_path / "claims",
    )
    succeeded: list[tuple[str, str, str]] = []

    def success(name, job_uid, pod_uid, _cluster):
        succeeded.append((name, job_uid, pod_uid))
        return {"pod_restarts": 0}

    prior = types.SimpleNamespace(
        _succeeded_job=success,
        _default_kubernetes=lambda _kind, _name: {},
        _default_sessions=lambda _task, _key: [],
    )
    monkeypatch.setattr(observer, "_evidence", lambda: prior)
    accepted = observer.accept(
        tmp_path,
        "hosted-qwen",
        observed,
        collector_job_uid=accept_uid,
        collector_pod_uid=accept_pod,
        package_commit="2" * 40,
        metadata=_metadata,
        account=_account,
        api_key="redacted-test-key",
        claim_root=tmp_path / "claims",
    )
    observer.validate_accepted_for_release(
        accepted,
        "hosted-qwen",
        tmp_path,
        "2" * 40,
        kubernetes=prior._default_kubernetes,
    )
    changed = json.loads(json.dumps(accepted))
    changed["execution_contract"]["context_window_size"] = 1
    changed["receipt_sha256"] = observer.digest(changed)
    with pytest.raises(observer.DuplicateError, match="accepted_duplicate_receipt_invalid"):
        observer.validate_accepted_structure(changed, "hosted-qwen", tmp_path, "2" * 40)
    assert succeeded == [
        (observer.source_job("hosted-qwen"), source_uid, source_pod),
        (observer.source_job("hosted-qwen"), source_uid, source_pod),
        (observer.accept_job("hosted-qwen"), accept_uid, accept_pod),
    ]


def test_package_bootstrap_names_and_mounted_verification(tmp_path):
    source_package = tmp_path / package.PACKAGE_PATH
    source_runner = tmp_path / package.RUN_PATH
    source_package.parent.mkdir(parents=True)
    source_runner.parent.mkdir(parents=True)
    source_package.write_text("package")
    source_runner.write_text("runner")
    payload, entries = package._payload(tmp_path, (package.PACKAGE_PATH, package.RUN_PATH))
    assert set(payload) == {"package.py", "run.sh"}
    obj = package._object("chris-test", entries)
    manifest = {
        "schema_version": package.SCHEMA,
        "objects": [obj],
        "aggregate_sha256": package.sha256(package.canonical([obj])),
        "release_included": False,
        "launch_authorized": False,
    }
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for key, value in payload.items():
        (bootstrap / key).write_text(value)
    manifest_path = bootstrap / "package-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "destination"
    package.verify_mounted(manifest_path, bootstrap, destination, manifest["aggregate_sha256"])
    assert (destination / package.PACKAGE_PATH).read_text() == "package"


def test_held_manifest_has_twelve_safe_cpu_jobs():
    root = Path(__file__).parents[1]
    held = yaml.safe_load((root / observer.MANIFEST_PATH).read_text())
    jobs = held["items"]
    assert len(jobs) == 12
    assert len({row["metadata"]["name"] for row in jobs}) == 12
    for row in jobs:
        pod = row["spec"]["template"]["spec"]
        env = {
            item["name"]: item.get("value")
            for item in pod["containers"][0]["env"]
            if "value" in item
        }
        group = env["DUPLICATE_GROUP"]
        mode = env["DUPLICATE_MODE"]
        assert row["metadata"]["name"] == (
            observer.source_job(group) if mode == "source" else observer.accept_job(group)
        )
        projected = pod["volumes"][0]["projected"]["sources"]
        projected_names = [item["configMap"]["name"] for item in projected]
        assert projected_names[:5] == [
            observer.package_configmap(group, component)
            for component in ("core-a", "core-b", "core-c", "core-d", "runtime")
        ]
        assert projected_names[5] == f"chris-final-v5-dup-release-{group}"
        assert row["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert pod["restartPolicy"] == "Never"
        assert "gpu" not in json.dumps(row).lower()
        relay.validate_object(row)
        secret_refs = json.dumps(row)
        assert observer.SECRET_NAME in secret_refs
        assert "chris-cyber-opencode-evals-v3" not in secret_refs


def test_shell_wrappers_are_parseable_and_submit_via_relay_only():
    root = Path(__file__).parents[1]
    scripts = [root / observer.RUN_PATH, root / observer.SUBMIT_PATH]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    submit = (root / observer.SUBMIT_PATH).read_text()
    assert "kubernetes_create_relay create" in submit
    assert "kubectl create" not in submit
    assert "DUPLICATE_CREATE_RECEIPT" in submit


def test_renderer_frozen_allowlist_matches_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, "_groups", lambda: ("hosted-qwen",))
    monkeypatch.setattr(observer, "validate_release", lambda _value, _root: None)
    monkeypatch.setattr(
        observer,
        "group_binding",
        lambda _root, _group: {
            "group": "hosted-qwen",
            "cell_ids": ["cell-a"],
            "planned_execution_count": 1,
        },
    )
    generic = (*package.CORE_NAMES, package.OBSERVER_NAME)
    configmaps = {name: package._configmap(name, {}) for name in generic}
    monkeypatch.setattr(
        package,
        "build_package",
        lambda _root: {
            "configmaps": configmaps,
            "aggregate_sha256": "sha256:" + "a" * 64,
        },
    )
    monkeypatch.setattr(
        renderer,
        "_load_job",
        lambda _root, name: {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": observer.NAMESPACE},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "preemptionPolicy": "Never",
                        "priorityClassName": "fleet-serve-low",
                        "containers": [{"name": "observer", "resources": {}}],
                    }
                },
            },
        },
    )
    release = {
        "package_commit": "3" * 40,
        "package_aggregate_sha256": "sha256:" + "a" * 64,
    }
    result = renderer.render(tmp_path, release, "hosted-qwen", "source")
    envelope = relay.build_envelope(result["manifest"]["items"], result["allowlist"])
    assert envelope["envelope_sha256"] == result["envelope_sha256"]
    assert result["object_count"] == 8
