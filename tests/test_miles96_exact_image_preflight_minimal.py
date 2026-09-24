import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from tests.test_sfs_output_job import completed_objects, service_account
from training import miles96_exact_image_preflight as preflight
from training import skyrl_prod9_direct as direct

SOURCE_COMMIT = "a" * 40


@pytest.fixture(scope="module")
def package():
    return preflight.build_package(SOURCE_COMMIT)


def _receipt(package):
    proof = preflight.validate_package(package)
    tools = package.plan["tool_contract"]
    body = {
        "schema": preflight.SCHEMA,
        "status": "passed",
        "gpus": 0,
        "job_name": preflight.NAME,
        "runtime_image": package.request["image"],
        "image_digest": package.request["image"].rsplit("@", 1)[-1],
        "source_closure_sha256": preflight.SOURCE_CLOSURE_SHA256,
        "driver_sha256": proof["driver_sha256"],
        "runtime_bundle_sha256": proof["runtime_bundle_sha256"],
        "plan_sha256": proof["plan_sha256"],
        "request_sha256": proof["request_sha256"],
        "runtime_binding_sha256": "sha256:" + "b" * 64,
        "raw_tool_catalog_sha256": tools["raw_tool_catalog_sha256"],
        "openai_tool_catalog_sha256": tools["openai_tool_catalog_sha256"],
        "tool_transform_source_sha256": tools["transform_source_sha256"],
        "checks": {
            "pinned_runtime_binding": True,
            "zero_update_entrypoint": True,
            "session_open_exact_catalog": True,
            "session_open_raw_drift_rejected_and_closed": True,
        },
        "observed_at_unix": 1.0,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _authorization(package, root):
    proof = preflight.validate_package(package)
    checked_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    preview = direct._seal(
        {
            "schema": direct.CPU_PREVIEW_SCHEMA,
            "status": "passed",
            "purpose": "preflight",
            "context": direct.PROD_CONTEXT,
            "name": preflight.NAME,
            "manifest_sha256": proof["manifest_sha256"],
            "server_render_sha256": "sha256:" + "1" * 64,
            "gpus": 0,
            "priority": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "submitted": False,
            "checked_at": checked_at,
        }
    )
    observer = direct._seal(
        {
            "schema": "cyber_direct_cleanup_observer_armed_v1",
            "status": "armed",
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "kind": "job",
            "name": preflight.NAME,
            "maximum_seconds": direct.CPU_MAXIMUM_SECONDS,
            "expected_gpus": 0,
            "plan_sha256": proof["plan_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "observer_pid": os.getpid(),
            "creator_binding_path": str(root / "creator-binding.json"),
            "armed_at": checked_at,
        }
    )
    parent = "sha256:" + "2" * 64
    authorization = preflight.authorize_create_once(
        package,
        prod_previews=[preview, deepcopy(preview)],
        observer=observer,
        operation_root=root,
        parent_review_sha256=parent,
    )
    return authorization, parent


def test_package_reuses_alert_off_c1_q1_zero_gpu_job_without_sfs_or_config_map(package):
    proof = preflight.validate_package(package)
    job = package.job
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert job["spec"]["suspend"] is True and job["spec"]["backoffLimit"] == 0
    assert pod["priorityClassName"] == "c1" and pod["priority"] == 10_000
    assert pod["automountServiceAccountToken"] is False
    assert container["terminationMessagePath"] == "/tmp/preflight.json"
    assert container["terminationMessagePolicy"] == "File"
    assert container["volumeMounts"] == [{"name": "tmp", "mountPath": "/tmp"}]
    assert pod["volumes"] == [{"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}]
    assert "persistentVolumeClaim" not in json.dumps(job)
    assert "configMap" not in json.dumps(job)
    assert "nvidia.com/gpu" not in json.dumps(job)
    assert proof["runtime_bundle_sha256"] == package.runtime_bundle_sha256


@pytest.mark.parametrize("fault", ["env-from", "secret-volume", "gpu", "root-alert"])
def test_server_response_rejects_unreviewed_security_or_resource_drift(package, fault):
    actual = deepcopy(package.job)
    pod = actual["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if fault == "env-from":
        container["envFrom"] = [{"secretRef": {"name": "unreviewed"}}]
    elif fault == "secret-volume":
        pod["volumes"].append({"name": "secret", "secret": {"secretName": "unreviewed"}})
    elif fault == "gpu":
        container["resources"]["limits"]["nvidia.com/gpu"] = "1"
    else:
        actual["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(ValueError):
        preflight.validate_response(actual, package, require_uid=False)


def test_package_rebuild_rejects_driver_or_manifest_drift(package):
    changed = deepcopy(package.job)
    changed["spec"]["template"]["spec"]["containers"][0]["command"][-1] += ";pass"
    with pytest.raises(ValueError, match="differs from rebuilt"):
        preflight.validate_package(replace(package, job=changed))


def test_terminal_job_and_receipt_are_bound_to_exact_package(package):
    receipt = _receipt(package)
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    expected = package.job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assignment["resourceUsage"] = deepcopy(expected)
    assignment["flavors"] = {name: "cpu-head" for name in expected}
    pods["items"][0]["status"]["containerStatuses"][0]["imageID"] = (
        "containerd://" + package.request["image"].rsplit("@", 1)[-1]
    )
    logs = preflight.LOG_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    assert (
        preflight.validate_completed_preflight(
            package, job, workloads, pods, service_account(), logs
        )
        == receipt
    )


def test_receipt_binding_drift_fails_closed(package):
    receipt = _receipt(package)
    receipt["plan_sha256"] = "sha256:" + "0" * 64
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    receipt["sha256"] = "sha256:" + digest(body)
    with pytest.raises(ValueError, match="receipt is incomplete"):
        preflight.validate_receipt(receipt, package)


def test_shared_create_rail_issues_exactly_one_mutation(package, tmp_path, monkeypatch):
    authorization, parent = _authorization(package, tmp_path)
    calls = []
    monkeypatch.setattr(direct, "_cpu_duplicate_checks", lambda *_args, **_kwargs: {"checked": 10})
    monkeypatch.setattr(direct, "server_dry_run", lambda *_args, **_kwargs: package.job)
    monkeypatch.setattr(direct, "validate_cpu_preview", lambda *_args, **_kwargs: {})

    def runner(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "metadata": {
                        "name": preflight.NAME,
                        "uid": "00000000-0000-4000-8000-000000000001",
                        "creationTimestamp": "2026-09-24T09:00:00Z",
                    }
                }
            ),
            stderr="",
        )

    created = preflight.create_once(
        package,
        tmp_path,
        authorization,
        expected_parent_review_sha256=parent,
        runner=runner,
    )
    assert created["status"] == "created_once"
    assert len(calls) == 1 and "create" in calls[0]
    assert len((tmp_path / "PROD9_STAGE_CREATE.jsonl").read_text().splitlines()) == 2
    with pytest.raises(Exception, match="observer binding|create intent"):
        preflight.create_once(
            package,
            tmp_path,
            authorization,
            expected_parent_review_sha256=parent,
            runner=runner,
        )
    assert len(calls) == 1


def test_ambiguous_create_reconciliation_is_read_only_and_uid_bound(package, tmp_path):
    authorization, parent = _authorization(package, tmp_path)
    proof = preflight.validate_package(package)
    intent = {
        "state": "CREATE_INTENT_DO_NOT_RETRY",
        "purpose": "stage",
        "plan_sha256": proof["plan_sha256"],
        "manifest_sha256": proof["manifest_sha256"],
        "authorization_sha256": authorization["core_authorization"]["sha256"],
        "duplicate_checks": {"checked": 10},
    }
    (tmp_path / "PROD9_STAGE_CREATE.jsonl").write_text(
        json.dumps(intent, sort_keys=True, separators=(",", ":")) + "\n"
    )
    actual = deepcopy(package.job)
    actual["metadata"].update(
        {
            "uid": "00000000-0000-4000-8000-000000000001",
            "creationTimestamp": "2026-09-24T09:00:00Z",
        }
    )
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(actual), stderr="")

    observed = preflight.reconcile_create_intent(
        package,
        tmp_path,
        authorization,
        expected_parent_review_sha256=parent,
        runner=runner,
    )
    assert observed["status"] == "observed_exact_job_requires_recovery_observer"
    assert observed["job_uid"] == actual["metadata"]["uid"]
    assert observed["recovery_observer_required"] is True
    assert len(calls) == 1 and "get" in calls[0] and "create" not in calls[0]
