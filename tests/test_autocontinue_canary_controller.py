from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_canary_controller as canary

ROOT = Path(__file__).parents[1]
Q_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_HELD = (
    ROOT / "docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-held-release-v1.json"
)
G_HELD = (
    ROOT / "docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-held-release-v1.json"
)
PRE_MANIFEST = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml"
SCORED_MANIFEST = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
BASE_PACKAGE_COMMIT = "25fe5bfe45cd91bc9e877af2982bbacd4e94c2d2"


def _cell_claim_worker(plan_path: str, claim_root: str, queue) -> None:
    import os

    os.environ["JOB_UID"] = "11111111-1111-4111-8111-111111111111"
    os.environ["POD_UID"] = "22222222-2222-4222-8222-222222222222"
    try:
        receipt = canary.claim_global_cell(canary.load_object(Path(plan_path)), Path(claim_root))
        queue.put(("claimed", receipt["receipt_sha256"]))
    except RuntimeError as exc:
        queue.put(("rejected", str(exc)))


@pytest.mark.parametrize("path", [Q_PLAN, G_PLAN])
def test_one_cell_canary_plan_is_exact_and_legacy_credit_free(path: Path) -> None:
    plan = canary.load_object(path)
    canary.validate_plan(plan)
    assert plan["source_job_id"] == plan["campaign_id"]
    assert plan["legacy_credited_sessions"] == 0
    assert plan["credited_sessions"] == []
    assert plan["execution"]["launch_authorized"] is False


@pytest.mark.parametrize("path", [Q_PLAN, G_PLAN])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("source_job_id", "wrong"),
        lambda value: value["model"].__setitem__("session_model", value["model"]["served_id"]),
        lambda value: value["model"].__setitem__("repository", "wrong/repository"),
        lambda value: value["harness"].__setitem__("provider_adapter", "wrong-provider"),
        lambda value: value["execution"]["required_task_tools"].__setitem__(0, "wrong-tool"),
        lambda value: value["treatment_block"].__setitem__("model_revision", "wrong"),
        lambda value: value["execution"].__setitem__("retry_policy", "retry"),
        lambda value: value["source"].__setitem__(
            "scientific_mapping_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value.__setitem__("canary_acceptance_required_before_bulk", False),
        lambda value: value["privacy"].__setitem__("scores_included", True),
        lambda value: value.__setitem__("unknown_authority", {"accepted": True}),
        lambda value: value["execution"]["endpoint_lease"].__setitem__("maximum_streams", 1),
        lambda value: value["execution"].__setitem__(
            "required_priority_class", "fleet-infra-quiet"
        ),
        lambda value: value["attempts"][0].__setitem__("attempt", 2),
    ],
)
def test_resealed_canary_plan_tampering_is_rejected(path: Path, mutate) -> None:
    plan = copy.deepcopy(canary.load_object(path))
    mutate(plan)
    plan["plan_sha256"] = canary.digest_without(plan, "plan_sha256")
    with pytest.raises(ValueError, match="canary"):
        canary.validate_plan(plan)


@pytest.mark.parametrize(("plan_path", "release_path"), [(Q_PLAN, Q_HELD), (G_PLAN, G_HELD)])
def test_held_release_cannot_preflight_or_run_scored_work(
    plan_path: Path, release_path: Path, tmp_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    release = canary.load_object(release_path)
    canary.validate_held_release(release, plan)
    with pytest.raises(ValueError, match="preflight authorization"):
        canary.preflight(plan, release, tmp_path / "absent", "unused", ROOT, BASE_PACKAGE_COMMIT)
    with pytest.raises(ValueError, match="not authoritative"):
        canary.run(plan, release, tmp_path / "absent", tmp_path / "proxy.py", ROOT)
    assert not (tmp_path / "absent").exists()


def test_frozen_campaign_controller_is_unchanged_and_compatibility_receipt_is_exact() -> None:
    frozen = ROOT / "evals/fleet/hosted_sweep_controller.py"
    compatibility_path = ROOT / "docs/evidence/qwen38-study"
    compatibility = canary.load_object(
        compatibility_path
        / "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
    )
    assert canary._sha(frozen) == (
        "sha256:e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a"
    )
    assert compatibility["receipt_sha256"] == canary.digest_without(compatibility, "receipt_sha256")
    assert compatibility["canary_controller"]["sha256"] == canary._sha(
        ROOT / "evals/fleet/autocontinue_canary_controller.py"
    )
    canary.validate_compatibility(compatibility, ROOT)
    changed = copy.deepcopy(compatibility)
    changed["allowed_append_only_overlay"]["campaign_mutation_allowed"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="compatibility"):
        canary.validate_compatibility(changed, ROOT)


@pytest.mark.parametrize("manifest", [PRE_MANIFEST, SCORED_MANIFEST])
def test_held_manifests_have_two_create_once_high_priority_jobs(manifest: Path) -> None:
    docs = list(yaml.safe_load_all(manifest.read_text()))
    assert len(docs) == 2
    names = [doc["metadata"]["name"] for doc in docs]
    assert len(names) == len(set(names))
    for doc in docs:
        assert doc["kind"] == "Job"
        assert doc["spec"]["backoffLimit"] == 0
        assert doc["spec"]["template"]["spec"]["priorityClassName"] == "fleet-train-high"
        assert doc["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
        by_name = {row["name"]: row for row in env}
        assert "controller-uid" in by_name["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"]
        assert by_name["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    configmaps = {
        doc["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] for doc in docs
    }
    assert len(configmaps) == 2
    if manifest == PRE_MANIFEST:
        assert all(name.endswith("-preflight") for name in names)
        assert all("-pre-v1" in name for name in configmaps)
    else:
        assert all(not name.endswith("-preflight") for name in names)
        assert all("-run-v2" in name for name in configmaps)


def test_submitter_is_preview_only_until_append_only_final_package() -> None:
    text = (ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh").read_text()
    assert "final scored submission requires a separately reviewed v2 manifest" in text
    assert "validate-release" in text
    assert "--dry-run=server" in text
    assert 'test ! -e "/mnt/sfs/' not in text


def _synthetic_preflight_authorization(plan: dict) -> dict:
    expected = canary.EXPECTED[plan["shard_key"]]
    compatibility_path = (
        ROOT / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
    )
    compatibility = canary.load_object(compatibility_path)
    authorization = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-authorization-v1",
        "append_only": True,
        "status": "PREFLIGHT_AUTHORIZED",
        "authorized_at_utc": "2026-09-04T20:47:35Z",
        "package_commit": BASE_PACKAGE_COMMIT,
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "cell": {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
        },
        "preflight_identity": {
            "configmap_name": expected["preflight_configmap"],
            "job_name": plan["preflight_job_name"],
            "sfs_root": f"/mnt/sfs/jobs/{plan['preflight_job_name']}",
            "scored_configmap_name": expected["scored_configmap"],
            "scored_job_name": plan["scored_job_name"],
            "scored_job_created": False,
        },
        "evidence": {
            "task_inventory_receipt_sha256": canary.TASK_INVENTORY_SHA,
            "task_inventory_execution_sha256": canary.TASK_INVENTORY_EXECUTION_SHA,
            "dedicated_parity_receipt_sha256": canary.DEDICATED_PARITY_SHA,
            "hosted_health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": canary.FLOCK_GATE_SHA,
            "controller_compatibility_receipt_sha256": compatibility["receipt_sha256"],
            "fresh_duplicate_inventory_receipt_sha256": None,
        },
        "implementation": {
            "package_commit": BASE_PACKAGE_COMMIT,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": expected["plan_file_sha256"],
            "controller_sha256": canary._sha(
                ROOT / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "frozen_controller_sha256": canary._sha(
                ROOT / "evals/fleet/hosted_sweep_controller.py"
            ),
            "self_hosted_sha256": canary.SELF_HOSTED_SHA,
            "runner_sha256": canary.RUNNER_SHA,
            "endpoint_lease_sha256": canary.ENDPOINT_LEASE_SHA,
            "compatibility_file_sha256": canary._sha(compatibility_path),
            "preflight_manifest_sha256": canary.PRE_MANIFEST_SHA,
            "scored_manifest_sha256": canary.SCORED_MANIFEST_SHA,
        },
        "authorization": {
            "preflight_authorized": True,
            "launch_authorized": False,
            "author": "/root",
            "statement": expected["preflight_statement"],
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    authorization["receipt_sha256"] = canary.digest_without(authorization, "receipt_sha256")
    return authorization


@pytest.mark.parametrize("plan_path", [Q_PLAN, G_PLAN])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("package_commit", "0" * 40),
        lambda value: value.__setitem__("authorized_at_utc", "not-a-time"),
        lambda value: value["authorization"].__setitem__("statement", "arbitrary"),
        lambda value: value["preflight_identity"].__setitem__("job_name", "wrong"),
        lambda value: value["implementation"].__setitem__("package_commit", "0" * 40),
        lambda value: value["implementation"].__setitem__("plan_sha256", "sha256:" + "0" * 64),
        lambda value: value["implementation"].__setitem__("plan_file_sha256", "sha256:" + "0" * 64),
        lambda value: value["implementation"].__setitem__(
            "controller_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "frozen_controller_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "self_hosted_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__("runner_sha256", "sha256:" + "0" * 64),
        lambda value: value["implementation"].__setitem__(
            "endpoint_lease_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "compatibility_file_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "preflight_manifest_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "scored_manifest_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value.__setitem__("unknown_authority", True),
    ],
)
def test_resealed_stage_a_preflight_authorization_tampering_is_rejected(
    plan_path: Path, mutate
) -> None:
    plan = canary.load_object(plan_path)
    authorization = _synthetic_preflight_authorization(plan)
    canary.validate_preflight_authorization(authorization, plan, ROOT, BASE_PACKAGE_COMMIT)
    mutate(authorization)
    authorization["receipt_sha256"] = canary.digest_without(authorization, "receipt_sha256")
    with pytest.raises(ValueError, match="preflight authorization"):
        canary.validate_preflight_authorization(authorization, plan, ROOT, BASE_PACKAGE_COMMIT)


def test_preflight_stage_has_no_scored_execution_payload() -> None:
    manifest_text = PRE_MANIFEST.read_text()
    submitter_text = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh"
    ).read_text()
    for forbidden in (
        "/bootstrap/scored-manifest.yaml",
        "/bootstrap/submit.sh",
        "/bootstrap/run.sh",
        "/bootstrap/Dockerfile.opencode",
        "/bootstrap/fixed_proxy.py",
    ):
        assert forbidden not in manifest_text
    stage_a = submitter_text.split('if [[ "$MODE" == submit ]]', 1)[1].split(
        'kubectl -n "$NS" create --dry-run=server -f "$PRE_MANIFEST"', 1
    )[0]
    for forbidden in (
        "--from-file=scored-manifest.yaml",
        "--from-file=submit.sh",
        "--from-file=run.sh",
        "--from-file=Dockerfile.opencode",
        "--from-file=fixed_proxy.py",
    ):
        assert forbidden not in stage_a


def test_scored_dind_and_evaluator_share_workspace_and_sfs() -> None:
    for doc in yaml.safe_load_all(SCORED_MANIFEST.read_text()):
        pod_spec = doc["spec"]["template"]["spec"]
        dind = pod_spec["initContainers"][0]
        evaluator = pod_spec["containers"][0]
        assert dind["name"] == "dind"
        assert dind["restartPolicy"] == "Always"
        assert dind["readinessProbe"]["tcpSocket"] == {"port": 2375}
        dind_mounts = {row["name"]: row for row in dind["volumeMounts"]}
        evaluator_mounts = {row["name"]: row for row in evaluator["volumeMounts"]}
        for name, path in (("workspace", "/workspace"), ("sfs", "/mnt/sfs")):
            assert dind_mounts[name]["mountPath"] == path
            assert evaluator_mounts[name]["mountPath"] == path
            assert dind_mounts[name].get("readOnly", False) is False
            assert evaluator_mounts[name].get("readOnly", False) is False
        assert dind["resources"] == {
            "requests": {"cpu": "500m", "memory": "2Gi", "ephemeral-storage": "20Gi"},
            "limits": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "40Gi"},
        }


def test_global_cell_claim_allows_exactly_one_concurrent_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_cell_claim_worker,
            args=(str(Q_PLAN), str(tmp_path / "claims"), queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    outcomes = sorted(queue.get(timeout=1)[0] for _ in processes)
    assert outcomes == ["claimed", "rejected"]
    assert len(list((tmp_path / "claims").glob("*.json"))) == 1


def test_global_cell_claim_rejects_symlink_root_and_lock(tmp_path: Path, monkeypatch) -> None:
    plan = canary.load_object(Q_PLAN)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="root is unsafe"):
        canary.claim_global_cell(plan, linked)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="root is unsafe"):
        canary.claim_global_cell(plan, parent_link / "claims")
    assert not (target / "claims").exists()
    root = tmp_path / "claims"
    root.mkdir()
    (root / ".claim.lock").symlink_to(tmp_path / "missing")
    with pytest.raises(OSError):
        canary.claim_global_cell(plan, root)


def _synthetic_final_release(plan: dict) -> dict:
    release = {
        "schema_version": canary.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": "2026-09-04T20:30:00Z",
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "cell": {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
        },
        "evidence": {
            "fresh_inventory_receipt_sha256": canary.TASK_INVENTORY_SHA,
            "fresh_inventory_execution_sha256": canary.TASK_INVENTORY_EXECUTION_SHA,
            "fresh_duplicate_inventory_receipt_sha256": "sha256:" + "1" * 64,
            "fresh_duplicate_inventory_receipt_path": "unused/duplicate.json",
            "preflight_receipt_sha256": "sha256:" + "2" * 64,
            "preflight_receipt_path": "unused/preflight.json",
            "preflight_post_exit_receipt_sha256": "sha256:" + "3" * 64,
            "preflight_post_exit_receipt_path": "unused/preflight-post-exit.json",
            "preflight_authorization_receipt_sha256": "sha256:" + "4" * 64,
            "preflight_authorization_receipt_path": "unused/preflight-authorization.json",
            "dedicated_parity_receipt_sha256": canary.DEDICATED_PARITY_SHA,
            "hosted_health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": canary.FLOCK_GATE_SHA,
            "controller_compatibility_receipt_sha256": canary.load_object(
                ROOT / "docs/evidence/qwen38-study/"
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
            )["receipt_sha256"],
        },
        "implementation": {
            "package_commit": BASE_PACKAGE_COMMIT,
            "plan_sha256": plan["plan_sha256"],
            "controller_sha256": canary._sha(
                ROOT / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "frozen_controller_sha256": canary._sha(
                ROOT / "evals/fleet/hosted_sweep_controller.py"
            ),
            "preflight_manifest_sha256": canary._sha(PRE_MANIFEST),
            "scored_manifest_sha256": canary._sha(SCORED_MANIFEST),
            "launcher_sha256": canary._sha(
                ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh"
            ),
        },
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "required_priority_class": "fleet-train-high",
            "author": "/root",
            "statement": "exact future authorization",
        },
        "terminal_contract": {
            "downward_job_uid_required": True,
            "downward_pod_uid_required": True,
            "post_exit_k8s_observer_required": True,
            "terminal_schema_version": canary.TERMINAL_SCHEMA,
            "post_exit_schema_version": canary.POST_EXIT_SCHEMA,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    release["receipt_sha256"] = canary.digest_without(release, "receipt_sha256")
    return release


def _write_receipt(path: Path, value: dict) -> dict:
    value["receipt_sha256"] = canary.digest_without(value, "receipt_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _synthetic_preflight_evidence(tmp_path: Path, plan: dict) -> dict:
    cell = {
        "source_rank": plan["tasks"][0]["source_rank"],
        "attempt": 1,
        "task_version_id": plan["tasks"][0]["task"]["version_id"],
    }
    preauth = _write_receipt(tmp_path / "preauth.json", {"kind": "preauth"})
    duplicate = _write_receipt(
        tmp_path / "duplicate.json",
        {
            "schema_version": "fleet-opencode-autocontinue-canary-duplicate-inventory-v1",
            "status": "PASSED",
            "fleet_team_id": canary.self_hosted.FLEET_TEAM_ID,
            "pagination_exhausted": True,
            "plan_sha256": plan["plan_sha256"],
            "cell": cell,
            "exact_treatment_session_rows": 0,
            "current_plan_run_rows": 0,
            "active_attempts": 0,
            "global_cell_claim_absent": True,
            "job_pod_and_sfs_identities_absent": True,
            "observed_at_utc": "2026-09-04T20:00:00Z",
            "privacy": {
                "credentials_included": False,
                "prompts_or_traces_included": False,
                "scores_included": False,
            },
        },
    )
    preflight = _write_receipt(
        tmp_path / "preflight.json",
        {
            "schema_version": "fleet-opencode-autocontinue-canary-preflight-v1",
            "status": "PASSED",
            "plan_sha256": plan["plan_sha256"],
            "release_receipt_sha256": preauth["receipt_sha256"],
            "fleet_team_id": canary.self_hosted.FLEET_TEAM_ID,
            "current_plan_run_and_claim_identities_absent": True,
            "output_root_absent": True,
            "sfs_job_roots_reconciled": 61,
            "exact_treatment_sessions_reconciled": 0,
            "job_uid": "11111111-1111-4111-8111-111111111111",
            "pod_uid": "22222222-2222-4222-8222-222222222222",
            "scores_read": False,
            "prompts_or_traces_read": False,
        },
    )
    observed = _write_receipt(
        tmp_path / "observed.json",
        {
            "schema_version": "fleet-opencode-autocontinue-canary-preflight-post-exit-v1",
            "status": "PASSED",
            "plan_sha256": plan["plan_sha256"],
            "preflight_receipt_sha256": preflight["receipt_sha256"],
            "job": {
                "name": plan["preflight_job_name"],
                "uid": preflight["job_uid"],
                "succeeded": 1,
                "failed": 0,
                "priority_class": "fleet-train-high",
                "preemption_policy": "PreemptLowerPriority",
                "completion_time": "2026-09-04T20:01:00Z",
            },
            "pod": {
                "name": "preflight-pod",
                "uid": preflight["pod_uid"],
                "phase": "Succeeded",
                "exit_code": 0,
                "restart_count": 0,
                "finished_at": "2026-09-04T20:01:00Z",
            },
            "configmap": {
                "name": canary.EXPECTED[plan["shard_key"]]["preflight_configmap"],
                "uid": "33333333-3333-4333-8333-333333333333",
                "immutable": True,
            },
            "scored_job_created": False,
            "scored_sfs_root_absent": True,
            "privacy": {
                "credentials_included": False,
                "prompts_or_traces_included": False,
                "scores_included": False,
            },
        },
    )
    return {
        "fresh_duplicate_inventory_receipt_path": "duplicate.json",
        "fresh_duplicate_inventory_receipt_sha256": duplicate["receipt_sha256"],
        "preflight_authorization_receipt_path": "preauth.json",
        "preflight_authorization_receipt_sha256": preauth["receipt_sha256"],
        "preflight_receipt_path": "preflight.json",
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "preflight_post_exit_receipt_path": "observed.json",
        "preflight_post_exit_receipt_sha256": observed["receipt_sha256"],
    }


def test_final_preflight_evidence_loads_exact_receipts_and_rejects_uid_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    plan = canary.load_object(Q_PLAN)
    evidence = _synthetic_preflight_evidence(tmp_path, plan)
    monkeypatch.setattr(canary, "validate_preflight_authorization", lambda *_: None)
    canary._validate_final_preflight_evidence(evidence, plan, tmp_path, BASE_PACKAGE_COMMIT)
    observed_path = tmp_path / "observed.json"
    observed = canary.load_object(observed_path)
    observed["job"]["uid"] = "99999999-9999-4999-8999-999999999999"
    _write_receipt(observed_path, observed)
    evidence["preflight_post_exit_receipt_sha256"] = observed["receipt_sha256"]
    with pytest.raises(ValueError, match="not authoritative"):
        canary._validate_final_preflight_evidence(evidence, plan, tmp_path, BASE_PACKAGE_COMMIT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("status", "HELD"),
        lambda value: value["evidence"].__setitem__(
            "fresh_inventory_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["implementation"].__setitem__(
            "controller_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["authorization"].__setitem__("launch_authorized", False),
        lambda value: value["terminal_contract"].__setitem__(
            "post_exit_k8s_observer_required", False
        ),
    ],
)
def test_final_release_rejects_resealed_gate_tampering(mutate, monkeypatch) -> None:
    plan = canary.load_object(Q_PLAN)
    release = _synthetic_final_release(plan)
    monkeypatch.setattr(canary, "_validate_final_preflight_evidence", lambda *_: None)
    canary.validate_release(release, plan, ROOT)
    mutate(release)
    release["receipt_sha256"] = canary.digest_without(release, "receipt_sha256")
    with pytest.raises(ValueError, match="not authoritative"):
        canary.validate_release(release, plan, ROOT)


def _synthetic_terminal(plan: dict, release: dict) -> dict:
    terminal = {
        "schema_version": canary.TERMINAL_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "controller_compatibility_receipt_sha256": release["evidence"][
            "controller_compatibility_receipt_sha256"
        ],
        "release_receipt_sha256": release["receipt_sha256"],
        "job_uid": "11111111-1111-4111-8111-111111111111",
        "pod_uid": "22222222-2222-4222-8222-222222222222",
        "source_rank": plan["tasks"][0]["source_rank"],
        "attempt": 1,
        "task_version_id": plan["tasks"][0]["task"]["version_id"],
        "run_id": plan["attempts"][0]["run_id"],
        "global_cell_claim_receipt_sha256": "sha256:" + "5" * 64,
        "model_revision": plan["model"]["revision"],
        "served_id": plan["model"]["served_id"],
        "session_model": plan["model"]["session_model"],
        "model_sha256": canary.self_hosted.sha256(canary.self_hosted.canonical_json(plan["model"])),
        "harness_sha256": canary.self_hosted.sha256(
            canary.self_hosted.canonical_json(plan["harness"])
        ),
        "treatment_block_sha256": canary.self_hosted.sha256(
            canary.self_hosted.canonical_json(plan["treatment_block"])
        ),
        "context_management": canary.CONTEXT,
        "settings_file_sha256": plan["harness"]["settings_file_sha256"],
        "required_task_tools": ["bash", "submit_report"],
        "required_task_tool_catalog_sha256": plan["execution"]["required_task_tool_catalog_sha256"],
        "endpoint_lease": plan["execution"]["endpoint_lease"],
        "attempt_config_sha256": "sha256:" + "6" * 64,
        "claim_sha256": "sha256:" + "7" * 64,
        "terminal_at_utc": "2026-09-04T21:00:00Z",
        "accepted": True,
        "credited": True,
        "quarantined": False,
        "exact_cell_count": 1,
        "legacy_credited_sessions": 0,
        "retry_allowed": False,
        "bulk_release_granted": False,
        "post_exit_k8s_observer_required": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "acceptance_receipt_sha256": "sha256:" + "8" * 64,
        "session_id": "33333333-3333-4333-8333-333333333333",
        "verifier_execution_id": "44444444-4444-4444-8444-444444444444",
        "session_ingest_completed": True,
        "cleanup_completed": True,
    }
    terminal["receipt_sha256"] = canary.digest_without(terminal, "receipt_sha256")
    return terminal


def test_terminal_and_post_exit_require_exact_uid_lease_and_success() -> None:
    plan = canary.load_object(Q_PLAN)
    release = _synthetic_final_release(plan)
    terminal = _synthetic_terminal(plan, release)
    canary.validate_terminal(terminal, plan, release)
    observer = {
        "schema_version": canary.POST_EXIT_SCHEMA,
        "status": "PASSED",
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "final_release_receipt_sha256": release["receipt_sha256"],
        "canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "release_receipt_sha256": release["receipt_sha256"],
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "acceptance_receipt_sha256": terminal["acceptance_receipt_sha256"],
        "cell": {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
            "run_id": plan["attempts"][0]["run_id"],
            "claim_sha256": terminal["claim_sha256"],
            "acceptance_receipt_sha256": terminal["acceptance_receipt_sha256"],
            "session_id": terminal["session_id"],
            "verifier_execution_id": terminal["verifier_execution_id"],
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
        },
        "job": {
            "name": plan["scored_job_name"],
            "uid": terminal["job_uid"],
            "succeeded": 1,
            "failed": 0,
            "priority_class": "fleet-train-high",
            "preemption_policy": "PreemptLowerPriority",
            "completion_time": "2026-09-04T21:00:01Z",
        },
        "pod": {
            "name": "pod-name",
            "uid": terminal["pod_uid"],
            "phase": "Succeeded",
            "exit_code": 0,
            "restart_count": 0,
            "finished_at": "2026-09-04T21:00:00Z",
        },
        "endpoint_lease": plan["execution"]["endpoint_lease"],
        "endpoint_lease_reacquired_after_job_exit": True,
        "endpoint_lease_released": True,
        "exact_claim_count": 1,
        "exact_accepted_cell_count": 1,
        "quarantine_count": 0,
        "bulk_release_eligible": True,
        "evidence": {
            "exact_claim_count": 1,
            "exact_accepted_count": 1,
            "exact_session_count": 1,
            "exact_verifier_execution_count": 1,
            "global_cell_claim_receipt_sha256": terminal["global_cell_claim_receipt_sha256"],
            "claim_sha256": terminal["claim_sha256"],
            "acceptance_receipt_sha256": terminal["acceptance_receipt_sha256"],
            "session_id": terminal["session_id"],
            "verifier_execution_id": terminal["verifier_execution_id"],
            "final_release_receipt_sha256": release["receipt_sha256"],
            "canary_terminal_receipt_sha256": terminal["receipt_sha256"],
            "quarantine_count": 0,
            "session_ingest_completed": True,
            "cleanup_completed": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    observer["receipt_sha256"] = canary.digest_without(observer, "receipt_sha256")
    canary.validate_post_exit(observer, terminal, plan, release)
    observer["endpoint_lease_released"] = False
    observer["receipt_sha256"] = canary.digest_without(observer, "receipt_sha256")
    with pytest.raises(ValueError, match="post-exit"):
        canary.validate_post_exit(observer, terminal, plan, release)


@pytest.mark.parametrize(
    ("field", "value"),
    [("job_uid", "not-a-uuid"), ("pod_uid", ""), ("terminal_at_utc", "")],
)
def test_terminal_rejects_invalid_uid_or_timestamp(field: str, value: str) -> None:
    plan = canary.load_object(Q_PLAN)
    release = _synthetic_final_release(plan)
    terminal = _synthetic_terminal(plan, release)
    terminal[field] = value
    terminal["receipt_sha256"] = canary.digest_without(terminal, "receipt_sha256")
    with pytest.raises(ValueError, match="terminal"):
        canary.validate_terminal(terminal, plan, release)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sfs_job_roots_reconciled", None), ("exact_treatment_sessions_reconciled", 1)],
)
def test_final_preflight_evidence_requires_reconciliation_outputs(
    field: str, value: object, tmp_path: Path, monkeypatch
) -> None:
    plan = canary.load_object(Q_PLAN)
    evidence = _synthetic_preflight_evidence(tmp_path, plan)
    preflight_path = tmp_path / "preflight.json"
    preflight = canary.load_object(preflight_path)
    if value is None:
        preflight.pop(field)
    else:
        preflight[field] = value
    _write_receipt(preflight_path, preflight)
    evidence["preflight_receipt_sha256"] = preflight["receipt_sha256"]
    observed_path = tmp_path / "observed.json"
    observed = canary.load_object(observed_path)
    observed["preflight_receipt_sha256"] = preflight["receipt_sha256"]
    _write_receipt(observed_path, observed)
    evidence["preflight_post_exit_receipt_sha256"] = observed["receipt_sha256"]
    monkeypatch.setattr(canary, "validate_preflight_authorization", lambda *_: None)
    with pytest.raises(ValueError, match="not authoritative"):
        canary._validate_final_preflight_evidence(evidence, plan, tmp_path, BASE_PACKAGE_COMMIT)
