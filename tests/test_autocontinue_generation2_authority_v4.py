from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_authority_v4 as authority
from evals.fleet import autocontinue_generation2_canary_v3 as contract
from evals.fleet import autocontinue_generation2_runtime_v3 as runtime_v3

ROOT = Path.cwd()
AUTH = ROOT / authority.AUTH_PATH
HELD = ROOT / authority.HELD_PATH
Q_SPEC = ROOT / authority.SPEC_PATHS[0]
G_SPEC = ROOT / authority.SPEC_PATHS[1]
PACKAGE = "1" * 40


def _load(path: Path) -> dict:
    return authority.load(path)


def _release(spec: dict) -> dict:
    auth = _load(AUTH)
    plan = contract.validate_spec(spec, ROOT)
    entry = next(row for row in auth["models"] if row["model"] == spec["model"])
    release = {
        "schema_version": authority.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": entry["authorized_at_utc"],
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "incident_receipt_sha256": authority.held_v1.INCIDENT_SHA,
        "tombstone_bundle_receipt_sha256": authority.held_v1._tombstones(ROOT)[
            "receipt_sha256"
        ],
        "package_commit": PACKAGE,
        "root_authorization": authority._authority_binding(auth, ROOT),
        "implementation": authority._implementation(ROOT, spec),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": entry["authorized_at_utc"],
            "statement": entry["statement"],
        },
        "route_and_inventory": {
            "fresh_authenticated_hosted_route_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "old_claims_must_exist_and_match": True,
            "generation_2_claim_must_be_absent": True,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    release["receipt_sha256"] = authority.digest(release, "receipt_sha256")
    return release


def test_root_authorization_binds_both_exact_cells() -> None:
    receipt = _load(AUTH)
    specs = [_load(Q_SPEC), _load(G_SPEC)]
    authority.validate_root_authorization(receipt, specs, ROOT)
    assert [row["generation2_spec_sha256"] for row in receipt["models"]] == [
        row["generation2_spec_sha256"] for row in specs
    ]
    assert all(row["authorized_at_utc"] == authority.AUTH_TIME for row in receipt["models"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("models", 0, "statement"), "arbitrary"),
        (("models", 1, "authorized_at_utc"), "not-a-time"),
        (("models", 0, "generation2_spec_sha256"), "sha256:" + "0" * 64),
        (("models", 1, "rendered_plan_sha256"), "sha256:" + "0" * 64),
        (("models", 0, "cell_id"), "sha256:" + "0" * 64),
        (("scope", "cluster_mutation_authorized_by_this_package"), True),
    ],
)
def test_root_authorization_resealed_tampering_fails(path: tuple, value: object) -> None:
    changed = copy.deepcopy(_load(AUTH))
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="root authorization"):
        authority.validate_root_authorization(
            changed, [_load(Q_SPEC), _load(G_SPEC)], ROOT
        )


@pytest.mark.parametrize("spec_path", [Q_SPEC, G_SPEC])
def test_release_is_bound_to_independent_authorization(spec_path: Path) -> None:
    spec = _load(spec_path)
    auth = _load(AUTH)
    release = _release(spec)
    authority.validate_release(release, spec, auth, ROOT, PACKAGE)
    for path, value in [
        (("released_at_utc",), "2026-09-05T04:34:28Z"),
        (("authorization", "statement"), "arbitrary"),
        (("root_authorization", "file_sha256"), "sha256:" + "0" * 64),
        (("root_authorization", "receipt_sha256"), "sha256:" + "0" * 64),
        (("implementation", "module_sha256"), "sha256:" + "0" * 64),
    ]:
        changed = copy.deepcopy(release)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="scoring release"):
            authority.validate_release(changed, spec, auth, ROOT, PACKAGE)


def test_v4_terminal_binds_authorization_raw_and_self_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load(Q_SPEC)
    plan = contract.validate_spec(spec, ROOT)
    job_uid = "11111111-1111-4111-8111-111111111111"
    pod_uid = "22222222-2222-4222-8222-222222222222"
    monkeypatch.setenv("JOB_UID", job_uid)
    monkeypatch.setenv("POD_UID", pod_uid)
    claim = {
        "schema_version": "fleet-statistical-cell-execution-claim-v2",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "claimed_at_utc": "2026-09-05T04:40:00Z",
        "generation_1_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    claim["receipt_sha256"] = contract.digest(claim, "receipt_sha256")
    attempt, config = contract._expected_cell_bindings(plan)
    inner = contract.terminal_receipt(
        spec,
        plan,
        claim,
        {
            "accepted": False,
            "quarantined": True,
            "claim_sha256": attempt,
            "attempt_config_sha256": config,
        },
        root=ROOT,
        terminal_at_utc="2026-09-05T04:41:00Z",
    )
    auth = _load(AUTH)
    terminal = authority.terminal_receipt(inner, spec, plan, claim, auth, root=ROOT)
    authority.validate_terminal(terminal, spec, plan, claim, auth, root=ROOT)
    changed = copy.deepcopy(terminal)
    changed["root_authorization"]["receipt_sha256"] = "sha256:" + "0" * 64
    changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="terminal"):
        authority.validate_terminal(changed, spec, plan, claim, auth, root=ROOT)


def test_held_package_and_manifest_are_non_launching() -> None:
    authority.validate_held(_load(HELD), ROOT)
    items = yaml.safe_load((ROOT / authority.MANIFEST_PATH).read_text())["items"]
    assert len(items) == 2
    assert all(
        item["metadata"]["annotations"]
        == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        for item in items
    )


def test_bootstrap_and_launcher_bind_authority_before_side_effects() -> None:
    manifest = yaml.safe_load((ROOT / authority.MANIFEST_PATH).read_text())
    bootstrap_files = set(
        re.findall(
            r"/bootstrap/([A-Za-z0-9_.-]+)",
            (ROOT / authority.RUN_PATH).read_text(),
        )
    )
    launcher = (ROOT / authority.SUBMIT_PATH).read_text()
    supplied = set(re.findall(r"--from-file=([A-Za-z0-9_.-]+)=", launcher))
    assert bootstrap_files <= supplied
    assert {
        "generation2_v1.py",
        "generation2_v2.py",
        "generation2_v3.py",
        "runtime_v3.py",
        "authority_v4.py",
        "v1-manifest.yaml",
        "v1-run.sh",
        "v1-submit.sh",
        "v2-manifest.yaml",
        "v2-run.sh",
        "v2-submit.sh",
        "v3-manifest.yaml",
        "v3-run.sh",
        "v3-submit.sh",
        "v4-manifest.yaml",
        "v4-run.sh",
        "v4-submit.sh",
    } <= supplied
    assert manifest["items"][0]["spec"]["template"]["spec"]["containers"][0][
        "command"
    ] == ["/bin/bash", "/bootstrap/run-v4.sh"]
    authority_gate = launcher.index("validate-authority")
    release_gate = launcher.index("validate-release")
    first_real_create = launcher.index("| kubectl create -f -")
    assert authority_gate < release_gate < first_real_create
    runtime = (ROOT / authority.MODULE_PATH).read_text()
    assert runtime.index("validate_release(release") < runtime.index(
        "acquire_endpoint_lease"
    ) < runtime.index("claim_execution_generation")
    assert "kubectl delete" not in launcher and "kubectl apply" not in launcher
    assert not Path(
        "docs/evidence/qwen38-study/"
        "2026-09-04-qwen38-autocontinue-generation2-scoring-release-v4.json"
    ).exists()
    assert not Path(
        "docs/evidence/qwen38-study/"
        "2026-09-04-glm53-autocontinue-generation2-scoring-release-v4.json"
    ).exists()


def test_predecessor_executable_receipt_still_valid() -> None:
    specs = [_load(Q_SPEC), _load(G_SPEC)]
    runtime_v3.validate_executable_held(
        _load(ROOT / runtime_v3.EXECUTABLE_HELD_PATH), specs, ROOT
    )
    assert os.environ.get("FLEET_API_KEY") is None
