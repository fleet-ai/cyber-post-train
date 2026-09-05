from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_canary_controller as canary
from evals.fleet import (
    autocontinue_successor_manifest_authorization as successor_manifest_authorization,
)
from evals.fleet import scored_manifest_authorization as manifest_authorization

ROOT = Path(__file__).parents[1]
Q_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_PLAN_V2 = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_PLAN_V2 = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"
Q_HELD = (
    ROOT / "docs/evidence/qwen38-study/2026-09-04-qwen38-autocontinue-canary-held-release-v1.json"
)
G_HELD = (
    ROOT / "docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-held-release-v1.json"
)
Q_PREAUTH = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v1.json"
)
G_PREAUTH = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-glm53-autocontinue-canary-preflight-authorization-v1.json"
)
Q_PREAUTH_V2 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v2.json"
)
G_PREAUTH_V2 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-glm53-autocontinue-canary-preflight-authorization-v2.json"
)
Q_PREAUTH_V3 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v3.json"
)
G_PREAUTH_V3 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-glm53-autocontinue-canary-preflight-authorization-v3.json"
)
PRE_MANIFEST = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml"
PRE_MANIFEST_V2 = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml"
SCORED_MANIFEST = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
PRE_MANIFEST_V3 = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
SCORED_MANIFEST_V3 = ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v3.yaml"
SUCCESSOR_SCORED_MANIFEST = (
    ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
)
SCORED_V1_FAILURE = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"
)
SUCCESSOR_COMPATIBILITY = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
)
Q_HELD_V2 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-qwen38-autocontinue-canary-successor-held-release-v2.json"
)
G_HELD_V2 = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-glm53-autocontinue-canary-successor-held-release-v2.json"
)
BASE_PACKAGE_COMMIT = "25fe5bfe45cd91bc9e877af2982bbacd4e94c2d2"
PREFLIGHT_PACKAGE_COMMIT = "fdad6c80bcbfa999cd98e6f3194040bd1d85d679"
PREFLIGHT_V2_PACKAGE_COMMIT = "9c93095f60d627c239fe2d01f35efd918f328fe6"
PREFLIGHT_V3_PACKAGE_COMMIT = "f2c32e4571fddfa43770249f323e1843fc205f95"


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
    with pytest.raises(ValueError, match="compatibility|preflight authorization"):
        canary.preflight(
            plan,
            release,
            tmp_path / "absent",
            "unused",
            ROOT,
            BASE_PACKAGE_COMMIT,
            canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
            "2026-09-04T20:47:35Z",
            "held receipt cannot authorize",
        )
    with pytest.raises(ValueError, match="not authoritative"):
        canary.run(plan, release, tmp_path / "absent", tmp_path / "proxy.py", ROOT)
    assert not (tmp_path / "absent").exists()


def test_frozen_campaign_controller_is_unchanged_and_successor_compatibility_is_exact() -> None:
    frozen = ROOT / "evals/fleet/hosted_sweep_controller.py"
    compatibility_path = ROOT / "docs/evidence/qwen38-study"
    compatibility = canary.load_object(
        compatibility_path
        / "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
    )
    assert canary._sha(frozen) == (
        "sha256:e14670e40d2b1fbe4896e4b6dfb2902f121b81c8103efea3a74a11fed496809a"
    )
    assert compatibility["receipt_sha256"] == canary.digest_without(compatibility, "receipt_sha256")
    assert compatibility["canary_controller"]["sha256"] == (
        "sha256:412bf8c0dd33d23e50a56c4597e5e0dfc90b122a8b06af0987059afdca53f7ef"
    )
    successor = canary.load_object(SUCCESSOR_COMPATIBILITY)
    assert successor["implementation"]["canary_controller_sha256"] == canary._sha(
        ROOT / "evals/fleet/autocontinue_canary_controller.py"
    )
    canary.validate_successor_compatibility(successor, ROOT)
    changed = copy.deepcopy(compatibility)
    changed["allowed_append_only_overlay"]["campaign_mutation_allowed"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="compatibility"):
        canary.validate_compatibility(changed, ROOT)


def test_v1_preflight_failure_incident_is_sanitized_and_terminal() -> None:
    incident = canary.load_object(
        ROOT / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
    )
    assert incident["receipt_sha256"] == canary.V1_PREFLIGHT_FAILURE_SHA
    assert incident["receipt_sha256"] == canary.digest_without(incident, "receipt_sha256")
    assert incident["status"] == "TERMINAL_INFRASTRUCTURE_FAILURE"
    assert len(incident["terminal_preflights"]) == 2
    assert all(row["pod"]["restart_count"] == 0 for row in incident["terminal_preflights"])
    diagnosis = incident["sanitized_diagnosis"]
    assert diagnosis["classification"] == "deterministic_bootstrap_validation_failure_before_api"
    assert diagnosis["fleet_api_called"] is False
    assert diagnosis["model_called"] is False
    assert diagnosis["verifier_called"] is False
    assert diagnosis["session_created"] is False
    assert diagnosis["cell_claim_created"] is False
    assert diagnosis["scored_job_created"] is False
    assert incident["privacy"] == {
        "logs_read": False,
        "credentials_included": False,
        "prompts_or_traces_included": False,
        "scores_included": False,
    }
    canary.validate_v1_preflight_failure(incident)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["intent_configmap"].__setitem__("uid", "wrong"),
        lambda value: value["scheduling"].__setitem__("priority_class", "wrong"),
        lambda value: value["scheduling"].__setitem__("preemption_policy", "Never"),
        lambda value: value["image"].__setitem__("resolved", "wrong"),
        lambda value: value["terminal_preflights"][0]["configmap"].__setitem__(
            "created_at_utc", "wrong"
        ),
        lambda value: value["terminal_preflights"][0]["job"].__setitem__("reason", "Complete"),
        lambda value: value["terminal_preflights"][0]["pod"].__setitem__("owner_job_uid", "wrong"),
        lambda value: value["terminal_preflights"][1]["pod"].__setitem__("restart_count", 1),
        lambda value: value["sanitized_diagnosis"].__setitem__("fleet_api_called", True),
        lambda value: value["absence_observation"].__setitem__("global_cell_claims_present", 1),
        lambda value: value["absence_observation"]["checked_sfs_roots"].pop(),
        lambda value: value["privacy"].__setitem__("logs_read", True),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_v1_preflight_failure_resealed_tampering_is_rejected(mutate) -> None:
    incident = copy.deepcopy(
        canary.load_object(
            ROOT / "docs/evidence/qwen38-study/"
            "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
        )
    )
    mutate(incident)
    incident["receipt_sha256"] = canary.digest_without(incident, "receipt_sha256")
    with pytest.raises(ValueError, match="failure receipt"):
        canary.validate_v1_preflight_failure(incident)


@pytest.mark.parametrize("plan_path", [Q_PLAN, G_PLAN])
def test_v2_preflight_bootstrap_validates_without_unshipped_scored_files(
    plan_path: Path, tmp_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    shipped = (
        "evals/fleet/autocontinue_canary_controller.py",
        "evals/fleet/hosted_sweep_controller.py",
        "evals/fleet/self_hosted.py",
        "evals/fleet/opencode_train_sweep_runner.py",
        "evals/fleet/endpoint_lease.py",
        "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml",
        "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json",
        "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json",
    )
    for relative in shipped:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assert not (
        tmp_path / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
    ).exists()
    assert not (tmp_path / canary.EXPECTED[plan["shard_key"]]["plan_path"]).exists()
    assert not (
        tmp_path / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml"
    ).exists()
    authorization = _synthetic_preflight_authorization(plan)
    canary.validate_preflight_authorization(
        authorization,
        plan,
        tmp_path,
        BASE_PACKAGE_COMMIT,
        canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
        "2026-09-04T20:47:35Z",
        "synthetic exact preflight authorization",
    )


def test_v2_compatibility_rejects_wrong_present_campaign(tmp_path: Path) -> None:
    compatibility = canary.load_object(
        ROOT / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
    )
    campaign = (
        tmp_path / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
    )
    campaign.parent.mkdir(parents=True)
    campaign.write_text("{}\n")
    incident = (
        tmp_path / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
    )
    incident.parent.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json",
        incident,
    )
    with pytest.raises(ValueError, match="compatibility"):
        canary.validate_compatibility(compatibility, tmp_path, allow_missing_campaign=True)


@pytest.mark.parametrize("manifest", [PRE_MANIFEST, PRE_MANIFEST_V2, SCORED_MANIFEST])
def test_held_manifests_have_two_create_once_high_priority_jobs(manifest: Path) -> None:
    docs = list(yaml.safe_load_all(manifest.read_text()))
    assert len(docs) == 2
    names = [doc["metadata"]["name"] for doc in docs]
    assert len(names) == len(set(names))
    for doc in docs:
        assert doc["kind"] == "Job"
        assert doc["spec"]["backoffLimit"] == 0
        assert doc["spec"]["template"]["spec"]["priorityClassName"] == "fleet-train-high"
        expected_annotations = (
            {
                "cyber-post-train.fleet.ai/preview-only": "false",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
            }
            if manifest == SCORED_MANIFEST
            else {
                "cyber-post-train.fleet.ai/preview-only": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "false",
            }
        )
        assert doc["metadata"]["annotations"] == expected_annotations
        env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
        by_name = {row["name"]: row for row in env}
        assert "controller-uid" in by_name["JOB_UID"]["valueFrom"]["fieldRef"]["fieldPath"]
        assert by_name["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    configmaps = {
        doc["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] for doc in docs
    }
    assert len(configmaps) == 2
    if manifest in {PRE_MANIFEST, PRE_MANIFEST_V2}:
        assert all(name.endswith("-preflight") for name in names)
        expected_version = "-pre-v2" if manifest == PRE_MANIFEST_V2 else "-pre-v1"
        assert all(expected_version in name for name in configmaps)
        for doc in docs:
            container = doc["spec"]["template"]["spec"]["containers"][0]
            by_name = {row["name"]: row for row in container["env"]}
            package_source = by_name["PACKAGE_COMMIT"]["valueFrom"]["configMapKeyRef"]
            assert package_source["name"] in configmaps
            assert package_source["key"] == "package_commit"
            assert '--package-commit "$PACKAGE_COMMIT"' in container["args"][0]
            if manifest == PRE_MANIFEST_V2:
                for name, key in (
                    ("PLAN_FILE_SHA256", "plan_file_sha256"),
                    ("AUTHORIZED_AT_UTC", "authorized_at_utc"),
                    ("AUTHORIZATION_STATEMENT", "authorization_statement"),
                ):
                    assert by_name[name]["valueFrom"]["configMapKeyRef"]["key"] == key
                assert "scored-manifest" not in container["args"][0]
    else:
        assert all(not name.endswith("-preflight") for name in names)
        assert all("-run-v2" in name for name in configmaps)


def test_submitter_is_fail_closed_until_hosted_final_package_and_live_route() -> None:
    text = (ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh").read_text()
    assert "validate-release" in text
    assert "observe-route" in text
    assert "validate-route" in text
    assert "--maximum-age-seconds 120" in text
    assert "chris-ac-canary1-hosted-scored-submit-v1" in text
    assert 'git show "$PACKAGE_COMMIT:$path"' in text
    assert "autocontinue-canary-hosted-scoring-release-v3.json" in text
    assert "evals.fleet.scored_manifest_authorization" in text
    assert "evals/fleet/scored_manifest_authorization.py" in text
    assert "yq " not in text
    assert "--dry-run=server" in text
    assert 'test ! -e "/mnt/sfs/' not in text
    assert "kubectl apply" not in text


def test_scored_manifest_authorization_is_exact_and_fail_closed(tmp_path: Path) -> None:
    expected = ("chris-q38-ac-canary1-v1", "chris-glm53-ac-canary1-v1")
    manifest_authorization.validate_scored_manifest(SCORED_MANIFEST, expected)

    documents = list(yaml.safe_load_all(SCORED_MANIFEST.read_text()))
    documents[0]["metadata"]["annotations"][manifest_authorization.LAUNCH_AUTHORIZED] = "false"
    unauthorized = tmp_path / "unauthorized.yaml"
    unauthorized.write_text(yaml.safe_dump_all(documents))
    with pytest.raises(ValueError, match="unauthorized"):
        manifest_authorization.validate_scored_manifest(unauthorized, expected)

def test_successor_submitter_is_release_and_inventory_gated() -> None:
    path = ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh"
    text = path.read_text()
    assert path.stat().st_mode & 0o111
    assert 'if [[ "$MODE" != preview && "$MODE" != submit ]]' in text
    assert "test -f \"$Q_RELEASE\" && test -f \"$G_RELEASE\"" in text
    assert "autocontinue_successor_manifest_authorization" in text
    assert "validate-release" in text
    assert "observe-route" in text
    assert "validate-route" in text
    assert "_validate_inventory_for_task" in text
    assert "SFS_OBSERVER_UID" in text
    assert "git show \"$PACKAGE_COMMIT:$path\"" in text
    assert "--dry-run=server" in text
    assert "kubectl apply" not in text
    assert 'test ! -e "/mnt/sfs/' not in text
    assert '"scored_launch_authorized":false' in text
    assert text.index("validate-release") < text.index("observe-route")
    assert text.index("observe-route") < text.index("_validate_inventory_for_task")
    assert text.index("_validate_inventory_for_task") < text.index("create configmap \"$INTENT\"")
    assert text.index("SFS_OBSERVER_UID") < text.index("create configmap \"$INTENT\"")
    assert text.rindex("validate-route") < text.rindex('create -f "$MANIFEST"')


def test_successor_submitter_missing_releases_cannot_create_objects(
    tmp_path: Path,
) -> None:
    test_root = tmp_path / "repo"
    test_root.mkdir()
    release_paths = [
        test_root
        / "docs/evidence/qwen38-study/"
        "2026-09-04-qwen38-autocontinue-canary-successor-hosted-scoring-release-v4.json",
        test_root
        / "docs/evidence/qwen38-study/"
        "2026-09-04-glm53-autocontinue-canary-successor-hosted-scoring-release-v4.json",
    ]
    assert not any(path.exists() for path in release_paths)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "kubectl-calls"
    (fake_bin / "git").write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "rev-parse --show-toplevel" ]; then '\
        'printf "%s\\n" "$TEST_ROOT"; exit 0; fi\n'
        "exit 97\n"
    )
    (fake_bin / "uv").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "kubectl").write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$TEST_CALLS"\n'
        "exit 0\n"
    )
    for path in fake_bin.iterdir():
        path.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TEST_ROOT": str(test_root),
            "TEST_CALLS": str(calls),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh"),
            "submit",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    observed = calls.read_text().splitlines()
    assert all("create configmap" not in row for row in observed)
    assert all(not ("create -f" in row and "--dry-run=server" not in row) for row in observed)


def test_successor_scored_manifest_is_exactly_executable() -> None:
    expected = ("chris-q38-ac-canary1-v2", "chris-glm53-ac-canary1-v2")
    successor_manifest_authorization.validate_scored_manifest(SUCCESSOR_SCORED_MANIFEST, expected)
    with pytest.raises(ValueError, match="unauthorized"):
        successor_manifest_authorization.validate_scored_manifest(
            SUCCESSOR_SCORED_MANIFEST, expected, launch_authorized=False
        )

    documents = list(yaml.safe_load_all(SUCCESSOR_SCORED_MANIFEST.read_text()))
    assert len(documents) == 2
    for document in documents:
        spec = document["spec"]["template"]["spec"]
        assert spec["priorityClassName"] == "fleet-train-high"
        evaluator = spec["containers"][0]
        script = evaluator["args"][0]
        assert (
            'install -m 0755 /bootstrap/run.sh '
            '"$root/evals/fleet/scripts/run_opencode_autocontinue_canary.sh"'
        ) in script
        assert (
            'exec "$root/evals/fleet/scripts/run_opencode_autocontinue_canary.sh"'
            in script
        )
        for required in (
            "preflight-authorization-v3.json",
            "preflight-v3-pass.json",
            "preflight-v3-post-exit.json",
            "duplicate-inventory-v3.json",
            "controller-compatibility-v3.json",
            "scored-v1-bootstrap-failure.json",
        ):
            assert required in script
        assert "hosted-only-route-v1.json" not in script
        mounts = {row["name"]: row["mountPath"] for row in evaluator["volumeMounts"]}
        dind_mounts = {
            row["name"]: row["mountPath"]
            for row in spec["initContainers"][0]["volumeMounts"]
        }
        assert mounts["workspace"] == dind_mounts["workspace"] == "/workspace"
        assert mounts["sfs"] == dind_mounts["sfs"] == "/mnt/sfs"


def test_manifest_validator_command_failure_prevents_all_cluster_calls(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_calls = tmp_path / "uv-calls.txt"
    kubectl_calls = tmp_path / "kubectl-calls.txt"

    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "rev-parse --show-toplevel" ]; then\n'
        '  printf "%s\\n" "$TEST_REPO_ROOT"\n'
        "  exit 0\n"
        "fi\n"
        "exit 97\n"
    )
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$TEST_UV_CALLS"\n'
        'case "$*" in\n'
        '  *"evals.fleet.scored_manifest_authorization"*) exit 86 ;;\n'
        "esac\n"
        "exit 0\n"
    )
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_KUBECTL_CALLS"\nexit 0\n')
    for executable in (fake_git, fake_uv, fake_kubectl):
        executable.chmod(0o755)

    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TEST_REPO_ROOT": str(ROOT),
            "TEST_UV_CALLS": str(uv_calls),
            "TEST_KUBECTL_CALLS": str(kubectl_calls),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh"),
            "submit",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 86
    assert "evals.fleet.scored_manifest_authorization" in uv_calls.read_text()
    assert not kubectl_calls.exists()


def test_scored_manifest_packages_hosted_route_gate_before_claim() -> None:
    text = SCORED_MANIFEST.read_text()
    for required in (
        "hosted_release.py",
        "hosted_runtime.py",
        "hosted_health.py",
        "launch-route.json",
        "autocontinue_canary_hosted_runtime.py",
        "opencode-autocontinue-canary-preflights-v2.yaml",
        "controller-compatibility-v2.json",
        "preflight-v1-bootstrap-failure.json",
    ):
        assert required in text
    for doc in yaml.safe_load_all(text):
        env = {
            item["name"]: item for item in doc["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["LAUNCH_ROUTE_FILE"]["value"] == "launch-route.json"
        assert env["PACKAGE_COMMIT"]["valueFrom"]["configMapKeyRef"]["key"] == ("package_commit")


def _synthetic_preflight_authorization(plan: dict) -> dict:
    expected = canary.EXPECTED[plan["shard_key"]]
    compatibility_path = (
        ROOT / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
    )
    compatibility = canary.load_object(compatibility_path)
    authorization = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-authorization-v2",
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
            "configmap_name": expected["preflight_configmap_v2"],
            "job_name": expected["preflight_job_v2"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['preflight_job_v2']}",
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
            "v1_bootstrap_failure_receipt_sha256": canary.V1_PREFLIGHT_FAILURE_SHA,
            "fresh_duplicate_inventory_receipt_sha256": None,
        },
        "implementation": {
            "package_commit": BASE_PACKAGE_COMMIT,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": expected["plan_file_sha256"],
            "controller_sha256": (
                "sha256:412bf8c0dd33d23e50a56c4597e5e0dfc90b122a8b06af0987059afdca53f7ef"
            ),
            "frozen_controller_sha256": canary._sha(
                ROOT / "evals/fleet/hosted_sweep_controller.py"
            ),
            "self_hosted_sha256": canary.SELF_HOSTED_SHA,
            "runner_sha256": canary.RUNNER_SHA,
            "endpoint_lease_sha256": canary.ENDPOINT_LEASE_SHA,
            "compatibility_file_sha256": canary._sha(compatibility_path),
            "preflight_manifest_sha256": canary.PRE_MANIFEST_SHA,
        },
        "authorization": {
            "preflight_authorized": True,
            "launch_authorized": False,
            "author": "/root",
            "statement": "synthetic exact preflight authorization",
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
        lambda value: value.__setitem__("unknown_authority", True),
    ],
)
def test_resealed_stage_a_preflight_authorization_tampering_is_rejected(
    plan_path: Path, mutate
) -> None:
    plan = canary.load_object(plan_path)
    authorization = _synthetic_preflight_authorization(plan)
    canary.validate_preflight_authorization(
        authorization,
        plan,
        ROOT,
        BASE_PACKAGE_COMMIT,
        canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
        "2026-09-04T20:47:35Z",
        "synthetic exact preflight authorization",
    )
    mutate(authorization)
    authorization["receipt_sha256"] = canary.digest_without(authorization, "receipt_sha256")
    with pytest.raises(ValueError, match="preflight authorization"):
        canary.validate_preflight_authorization(
            authorization,
            plan,
            ROOT,
            BASE_PACKAGE_COMMIT,
            canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
            "2026-09-04T20:47:35Z",
            "synthetic exact preflight authorization",
        )


@pytest.mark.parametrize(
    ("plan_path", "authorization_path"), [(Q_PLAN, Q_PREAUTH), (G_PLAN, G_PREAUTH)]
)
def test_v1_preflight_authorization_is_retired_after_terminal_bootstrap_failure(
    plan_path: Path, authorization_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    authorization = canary.load_object(authorization_path)
    with pytest.raises(ValueError, match="preflight authorization"):
        canary.validate_preflight_authorization(
            authorization,
            plan,
            ROOT,
            PREFLIGHT_PACKAGE_COMMIT,
            canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
            authorization["authorized_at_utc"],
            authorization["authorization"]["statement"],
        )


def test_phase_b_submitter_pins_intent_and_packages_preflight_only() -> None:
    text = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canary_preflights_v1.sh"
    ).read_text()
    for required in (
        f"PACKAGE_COMMIT={PREFLIGHT_PACKAGE_COMMIT}",
        "Q_AUTH_RAW=45acad27d6389d0da90491090d9adcae5455952473aac27d8fc302f7096e0f3d",
        "G_AUTH_RAW=8ecbfeb079a3ed0241e81835e8e1180674ba132a14645a279f602722427cb4bd",
        "Q_AUTH_SELF=sha256:9f8eae90f2d59a1c1a39f94d38adbd345eaa5006cd73d2eec850701a32e59b59",
        "G_AUTH_SELF=sha256:8870a2ff27c6b22a70a28e2c58add13e459754dbe12480ba9786feb7f4654ab4",
        'git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"',
        'authorization.get("receipt_sha256") != sys.argv[5]',
        '--from-literal=package_commit="$PACKAGE_COMMIT"',
        "CREATE_ONCE_INTENT",
        "manual_reconciliation_required",
        '"scored_objects_created":0',
    ):
        assert required in text
    intent_position = text.index("stage=create-intent")
    assert intent_position < text.index("stage=create-qwen-configmap")
    assert text.index("stage=create-qwen-configmap") < text.index("stage=create-glm-configmap")
    assert text.index("stage=create-glm-configmap") < text.index("stage=create-preflight-jobs")
    for forbidden in (
        "--from-file=scored-manifest.yaml",
        "--from-file=submit.sh",
        "--from-file=run.sh",
        "--from-file=Dockerfile.opencode",
        "--from-file=fixed_proxy.py",
        'test ! -e "/mnt/sfs/',
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    ("plan_path", "authorization_path"),
    [(Q_PLAN, Q_PREAUTH_V2), (G_PLAN, G_PREAUTH_V2)],
)
def test_v2_preflight_authorization_is_exact_and_read_only(
    plan_path: Path, authorization_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    authorization = canary.load_object(authorization_path)
    assert authorization["receipt_sha256"] == canary.digest_without(authorization, "receipt_sha256")
    canary.validate_preflight_authorization(
        authorization,
        plan,
        ROOT,
        PREFLIGHT_V2_PACKAGE_COMMIT,
        canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
        "2026-09-04T21:31:11Z",
        authorization["authorization"]["statement"],
    )
    assert authorization["authorization"]["launch_authorized"] is False
    assert authorization["preflight_identity"]["scored_job_created"] is False


def test_v2_submitter_pins_phase_a3_and_is_create_once_preflight_only() -> None:
    text = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canary_preflights_v2.sh"
    ).read_text()
    required = (
        f"PACKAGE_COMMIT={PREFLIGHT_V2_PACKAGE_COMMIT}",
        "Q_AUTH_RAW=ad4171eec459d4925f5f12b97a3040ac3fa30e65b434bf2592a8ee43c9070249",
        "G_AUTH_RAW=7e037f0d8436ad04e66056718d7dc94d1bb5f00f57b477aebedec3c193ae06cf",
        "Q_AUTH_SELF=sha256:d1b764cd8f96bbe5d36a62956ea32931066740c8c77b8ac60c5474c7209f3720",
        "G_AUTH_SELF=sha256:a7cef96859a38ace334d5edbef22ffa1d87ed74d5c1d216857945035bf07f530",
        'git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"',
        'authorization.get("receipt_sha256") != sys.argv[8]',
        '--from-literal=plan_file_sha256="sha256:$plan_raw"',
        '--from-literal=authorized_at_utc="$AUTHORIZED_AT_UTC"',
        '--from-literal=authorization_statement="$statement"',
        '--from-file=v1-incident.json="$INCIDENT"',
        "CREATE_ONCE_INTENT",
        "manual_reconciliation_required",
        '"scored_objects_created":0',
    )
    for value in required:
        assert value in text
    validation = text.index("validate_preflight_authorization(")
    intent = text.index("stage=create-intent")
    assert validation < intent
    assert text.index('test "$(sha256sum "$Q_AUTH"') < intent
    assert text.index('test "$(sha256sum "$G_AUTH"') < intent
    assert intent < text.index("stage=create-qwen-configmap")
    assert text.index("stage=create-qwen-configmap") < text.index("stage=create-glm-configmap")
    assert text.index("stage=create-glm-configmap") < text.index("stage=create-preflight-jobs")
    for forbidden in (
        "--from-file=scored-manifest.yaml",
        "--from-file=proxy.py",
        "--from-file=Dockerfile",
        "kubectl apply",
        'test ! -e "/mnt/sfs/',
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    ("plan_path", "authorization_path"),
    [(Q_PLAN_V2, Q_PREAUTH_V3), (G_PLAN_V2, G_PREAUTH_V3)],
)
def test_v3_preflight_authorization_binds_phase_a_successor_package(
    plan_path: Path, authorization_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    authorization = canary.load_object(authorization_path)
    assert authorization["receipt_sha256"] == canary.digest_without(
        authorization, "receipt_sha256"
    )
    canary.validate_preflight_authorization(
        authorization,
        plan,
        ROOT,
        PREFLIGHT_V3_PACKAGE_COMMIT,
        canary.EXPECTED[plan["shard_key"]]["plan_file_sha256"],
        "2026-09-04T22:26:36Z",
        authorization["authorization"]["statement"],
    )
    assert authorization["authorization"]["launch_authorized"] is False
    assert authorization["preflight_identity"]["scored_job_created"] is False


def test_v3_submitter_is_create_once_and_packages_only_preflight_dependencies() -> None:
    path = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canary_preflights_v3.sh"
    )
    text = path.read_text()
    required = (
        f"PACKAGE_COMMIT={PREFLIGHT_V3_PACKAGE_COMMIT}",
        "Q_AUTH_RAW=1059c755a352b5f2320b540094ba7f4e5cc575b7b6ff5af4ca68a601b0f4f80c",
        "G_AUTH_RAW=de0d7aec69f8471c5e3b35797de5f7dda723a70b90c97407c8aeb9c21c31f69e",
        "Q_AUTH_SELF=sha256:2ccba714a6885ff8de4cd13f7a7857df2ccdf75bd8a258be04969e502d1ccf10",
        "G_AUTH_SELF=sha256:f5766496049bfe8e649e95b2ef450263b461c03b803a7b39b0da8bca63b3ea5c",
        '--from-file=campaign.json="$CAMPAIGN"',
        '--from-file=compatibility.json="$COMPATIBILITY"',
        '--from-file=scored-v1-incident.json="$INCIDENT"',
        'git show "$PACKAGE_COMMIT:$relative" | cmp - "$file"',
        "CREATE_ONCE_INTENT",
        "manual_reconciliation_required",
        '"scored_objects_created":0',
    )
    for value in required:
        assert value in text
    validation = text.index("validate_preflight_authorization(")
    intent = text.index("stage=create-intent")
    assert validation < intent
    assert intent < text.index("stage=create-qwen-configmap")
    assert text.index("stage=create-qwen-configmap") < text.index(
        "stage=create-glm-configmap"
    )
    assert text.index("stage=create-glm-configmap") < text.index(
        "stage=create-preflight-jobs"
    )
    for forbidden in (
        "--from-file=scored-manifest.yaml",
        "--from-file=manifest_authorization.py",
        "--from-file=submit.sh",
        "--from-file=run.sh",
        "--from-file=Dockerfile.opencode",
        "kubectl apply",
        'test ! -e "/mnt/sfs/',
    ):
        assert forbidden not in text
    assert path.stat().st_mode & 0o111


def test_preflight_stage_has_no_scored_execution_payload() -> None:
    manifest_text = PRE_MANIFEST.read_text()
    submitter_text = (
        ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canary_preflights_v2.sh"
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


@pytest.mark.parametrize(
    ("plan_path", "held_path"), [(Q_PLAN_V2, Q_HELD_V2), (G_PLAN_V2, G_HELD_V2)]
)
def test_successor_plans_and_held_releases_are_exact_and_unlaunched(
    plan_path: Path, held_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    held = canary.load_object(held_path)
    canary.validate_plan(plan)
    canary.validate_held_release(held, plan, ROOT)
    assert plan["execution"]["launch_authorized"] is False
    assert held["authorization"]["preflight_authorized"] is False
    assert held["authorization"]["launch_authorized"] is False
    assert plan["source"]["predecessor_scored_bootstrap_failure_receipt_sha256"] == (
        canary.SCORED_V1_FAILURE_SHA
    )
    assert plan["attempts"][0]["run_id"].startswith(plan["campaign_id"])


@pytest.mark.parametrize(
    ("plan_path", "held_path"), [(Q_PLAN_V2, Q_HELD_V2), (G_PLAN_V2, G_HELD_V2)]
)
def test_successor_held_release_rejects_resealed_evidence_and_identity_drift(
    plan_path: Path, held_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    held = canary.load_object(held_path)

    def set_nested(value: dict, path: tuple[str, ...], replacement) -> None:
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement

    mutations = (
        (("remaining_gates",), []),
        (("evidence", "controller_compatibility_receipt_sha256"), "sha256:" + "0" * 64),
        (("implementation", "controller_sha256"), "sha256:" + "0" * 64),
        (("predecessor", "intent_configmap", "uid"), "00000000-0000-4000-8000-000000000000"),
        (("predecessor", "scored_configmap", "uid"), "00000000-0000-4000-8000-000000000000"),
        (("predecessor", "scored_job", "uid"), "00000000-0000-4000-8000-000000000000"),
        (("predecessor", "scored_pod", "uid"), "00000000-0000-4000-8000-000000000000"),
        (("predecessor", "global_claim_absent"), False),
    )
    for path, replacement in mutations:
        changed = copy.deepcopy(held)
        set_nested(changed, path, replacement)
        changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="held release"):
            canary.validate_held_release(changed, plan, ROOT)


def test_scored_v1_failure_and_successor_compatibility_are_semantic() -> None:
    incident = canary.load_object(SCORED_V1_FAILURE)
    canary.validate_scored_v1_failure(incident)
    compatibility = canary.load_object(SUCCESSOR_COMPATIBILITY)
    canary.validate_successor_compatibility(compatibility, ROOT)

    for mutate in (
        lambda value: value["terminal_canaries"][0]["job"].__setitem__("failed", 0),
        lambda value: value["terminal_canaries"][1].__setitem__("exact_treatment_sessions", 1),
        lambda value: value["sfs_observer"].__setitem__("global_claim_json_count", 1),
        lambda value: value["api_observer"].__setitem__("fleet_team_name", "other"),
        lambda value: value["sanitized_diagnosis"].__setitem__("model_called", True),
    ):
        changed = copy.deepcopy(incident)
        mutate(changed)
        changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="scored v1"):
            canary.validate_scored_v1_failure(changed)


def test_successor_manifests_use_fresh_held_identities_and_canonical_scripts() -> None:
    pre_docs = list(yaml.safe_load_all(PRE_MANIFEST_V3.read_text()))
    scored_docs = list(yaml.safe_load_all(SCORED_MANIFEST_V3.read_text()))
    assert {row["metadata"]["name"] for row in pre_docs} == {
        "chris-q38-ac-canary1-v3-preflight",
        "chris-glm53-ac-canary1-v3-preflight",
    }
    assert {row["metadata"]["name"] for row in scored_docs} == {
        "chris-q38-ac-canary1-v2",
        "chris-glm53-ac-canary1-v2",
    }
    for row in pre_docs + scored_docs:
        assert row["spec"]["backoffLimit"] == 0
        assert row["spec"]["template"]["spec"]["priorityClassName"] == "fleet-train-high"
        assert row["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
    for row in pre_docs:
        script = row["spec"]["template"]["spec"]["containers"][0]["args"][0]
        assert "scored-manifest" not in script
        assert "/bootstrap/run.sh" not in script
        assert "/bootstrap/submit.sh" not in script
        assert (
            "/bootstrap/campaign.json "
            '"$root/evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"'
        ) in script
        assert (
            "/bootstrap/compatibility.json "
            '"$root/docs/evidence/qwen38-study/'
            '2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"'
        ) in script
        assert (
            "/bootstrap/scored-v1-incident.json "
            '"$root/docs/evidence/qwen38-study/'
            '2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"'
        ) in script
        assert "controller-compatibility-v2.json" not in script
        assert "preflight-v1-bootstrap-failure.json" not in script
    for row in scored_docs:
        script = row["spec"]["template"]["spec"]["containers"][0]["args"][0]
        canonical_run = '"$root/evals/fleet/scripts/run_opencode_autocontinue_canary.sh"'
        assert f"/bootstrap/run.sh {canonical_run}" in script
        assert f"exec {canonical_run}" in script
        assert '"$root/evals/fleet/scripts/run.sh"' not in script
        assert (
            "/bootstrap/manifest_authorization.py "
            '"$root/evals/fleet/scored_manifest_authorization.py"'
        ) in script
        assert "opencode-autocontinue-canary-preflights-v3.yaml" in script
        assert "controller-compatibility-v3.json" in script
        assert "scored-v1-bootstrap-failure.json" in script
        assert "controller-compatibility-v2.json" not in script
        assert "preflight-v1-bootstrap-failure.json" not in script


def _successor_authorization(plan: dict, root: Path) -> tuple[dict, str, str]:
    expected = canary.EXPECTED[plan["shard_key"]]
    compatibility_path = (
        root
        / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
    )
    compatibility = canary.load_object(compatibility_path)
    package_commit = "1" * 40
    statement = "exact successor preflight authorization"
    authorization = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-authorization-v3",
        "append_only": True,
        "status": "PREFLIGHT_AUTHORIZED",
        "authorized_at_utc": "2026-09-04T22:26:36Z",
        "package_commit": package_commit,
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "cell": {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        },
        "preflight_identity": {
            "configmap_name": expected["preflight_configmap_v2"],
            "job_name": expected["preflight_job_v2"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['preflight_job_v2']}",
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
            "v1_bootstrap_failure_receipt_sha256": canary.V1_PREFLIGHT_FAILURE_SHA,
            "fresh_duplicate_inventory_receipt_sha256": None,
            "scored_v1_failure_receipt_sha256": canary.SCORED_V1_FAILURE_SHA,
        },
        "implementation": {
            "package_commit": package_commit,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": expected["plan_file_sha256"],
            "controller_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "frozen_controller_sha256": canary._sha(
                root / "evals/fleet/hosted_sweep_controller.py"
            ),
            "self_hosted_sha256": canary.SELF_HOSTED_SHA,
            "runner_sha256": canary.RUNNER_SHA,
            "endpoint_lease_sha256": canary.ENDPOINT_LEASE_SHA,
            "compatibility_file_sha256": canary._sha(compatibility_path),
            "preflight_manifest_sha256": canary.PRE_MANIFEST_V3_SHA,
        },
        "authorization": {
            "preflight_authorized": True,
            "launch_authorized": False,
            "author": "/root",
            "statement": statement,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    authorization["receipt_sha256"] = canary.digest_without(
        authorization, "receipt_sha256"
    )
    return authorization, package_commit, statement


@pytest.mark.parametrize("plan_path", [Q_PLAN_V2, G_PLAN_V2])
def test_successor_v3_preflight_authorization_contract_is_exact(plan_path: Path) -> None:
    plan = canary.load_object(plan_path)
    expected = canary.EXPECTED[plan["shard_key"]]
    authorization, package_commit, statement = _successor_authorization(plan, ROOT)
    canary.validate_preflight_authorization(
        authorization,
        plan,
        ROOT,
        package_commit,
        expected["plan_file_sha256"],
        "2026-09-04T22:26:36Z",
        statement,
    )
    changed = copy.deepcopy(authorization)
    changed["evidence"]["scored_v1_failure_receipt_sha256"] = "sha256:" + "0" * 64
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="preflight authorization"):
        canary.validate_preflight_authorization(
            changed,
            plan,
            ROOT,
            package_commit,
            expected["plan_file_sha256"],
            "2026-09-04T22:26:36Z",
            statement,
        )


@pytest.mark.parametrize("plan_path", [Q_PLAN_V2, G_PLAN_V2])
def test_reconstructed_successor_workspace_executes_preflight_validation(
    plan_path: Path, tmp_path: Path, monkeypatch
) -> None:
    reconstructed = tmp_path / "workspace/cyber-post-train"
    sources = {
        "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json": (
            ROOT
            / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
        ),
        "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml": (
            PRE_MANIFEST_V3
        ),
        (
            "docs/evidence/qwen38-study/"
            "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
        ): SUCCESSOR_COMPATIBILITY,
        (
            "docs/evidence/qwen38-study/"
            "2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"
        ): SCORED_V1_FAILURE,
        "evals/fleet/autocontinue_canary_controller.py": (
            ROOT / "evals/fleet/autocontinue_canary_controller.py"
        ),
        "evals/fleet/hosted_sweep_controller.py": (
            ROOT / "evals/fleet/hosted_sweep_controller.py"
        ),
        "evals/fleet/self_hosted.py": ROOT / "evals/fleet/self_hosted.py",
        "evals/fleet/opencode_train_sweep_runner.py": (
            ROOT / "evals/fleet/opencode_train_sweep_runner.py"
        ),
        "evals/fleet/endpoint_lease.py": ROOT / "evals/fleet/endpoint_lease.py",
    }
    for relative, source in sources.items():
        destination = reconstructed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assert not (
        reconstructed / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v3.yaml"
    ).exists()
    assert not (reconstructed / "evals/fleet/scored_manifest_authorization.py").exists()

    plan = canary.load_object(plan_path)
    expected = canary.EXPECTED[plan["shard_key"]]
    authorization, package_commit, statement = _successor_authorization(plan, reconstructed)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(canary.hosted, "_validate_plan_identity_absence", lambda *_: [])
    monkeypatch.setattr(
        canary.hosted, "_client", lambda *_: contextlib.nullcontext(object())
    )
    monkeypatch.setattr(
        canary.self_hosted,
        "_request",
        lambda *_: {"team_name": "fleet", "team_id": canary.self_hosted.FLEET_TEAM_ID},
    )
    monkeypatch.setattr(canary.hosted, "_validate_inventory_for_task", lambda *_: [])
    receipt = canary.preflight(
        plan,
        authorization,
        tmp_path / "absent-sfs-root",
        "redacted-test-key",
        reconstructed,
        package_commit,
        expected["plan_file_sha256"],
        "2026-09-04T22:26:36Z",
        statement,
    )
    assert receipt["status"] == "PASSED"
    assert receipt["current_plan_run_and_claim_identities_absent"] is True


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


def test_global_cell_claim_first_use_is_stable_under_concurrent_stress(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    for round_number in range(12):
        claim_root = tmp_path / f"claims-{round_number}"
        queue = context.Queue()
        processes = [
            context.Process(
                target=_cell_claim_worker,
                args=(str(Q_PLAN), str(claim_root), queue),
            )
            for _ in range(6)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        outcomes = sorted(queue.get(timeout=1)[0] for _ in processes)
        assert outcomes == ["claimed", "rejected", "rejected", "rejected", "rejected", "rejected"]
        assert len(list(claim_root.glob("*.json"))) == 1


def test_claim_lock_retries_transient_first_create_visibility(tmp_path: Path, monkeypatch) -> None:
    root_fd = canary._open_directory_nofollow(tmp_path / "claims")
    original_open = canary.os.open
    attempts = 0

    def transient_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attempts
        if path == ".claim.lock" and attempts == 0:
            attempts += 1
            raise FileNotFoundError(path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(canary.os, "open", transient_open)
    try:
        lock_fd = canary._open_claim_lock(root_fd)
        assert attempts == 1
        assert canary.stat.S_ISREG(canary.os.fstat(lock_fd).st_mode)
        canary.os.close(lock_fd)
    finally:
        canary.os.close(root_fd)


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
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
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
            "preflight_manifest_sha256": canary._sha(PRE_MANIFEST_V2),
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
                "name": canary.EXPECTED[plan["shard_key"]]["preflight_job_v2"],
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
                "name": canary.EXPECTED[plan["shard_key"]]["preflight_configmap_v2"],
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
