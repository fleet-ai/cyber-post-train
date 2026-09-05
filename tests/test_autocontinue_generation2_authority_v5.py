from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_authority_v5 as authority
from evals.fleet import autocontinue_generation2_canary_v3 as contract

ROOT = Path.cwd()
AUTH = ROOT / authority.AUTH_PATH
HELD = ROOT / authority.HELD_PATH
Q_SPEC = ROOT / authority.SPEC_PATHS[0]
G_SPEC = ROOT / authority.SPEC_PATHS[1]
PACKAGE = "1" * 40
RELEASE_FILE_SHA = "sha256:" + "2" * 64


def _load(path: Path) -> dict:
    return authority.load(path)


def _release(spec: dict) -> dict:
    auth = _load(AUTH)
    held = _load(HELD)
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
        "root_authorization": authority.authority_v4._authority_binding(auth, ROOT),
        "executable_held_authority": authority._held_binding(held, ROOT),
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


def _claim_and_terminal(spec: dict, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    plan = contract.validate_spec(spec, ROOT)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    claim = {
        "schema_version": "fleet-statistical-cell-execution-claim-v2",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "claimed_at_utc": "2026-09-05T05:40:00Z",
        "generation_1_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    claim["receipt_sha256"] = contract.digest(claim, "receipt_sha256")
    attempt, config = contract._expected_cell_bindings(plan)
    terminal = contract.terminal_receipt(
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
        terminal_at_utc="2026-09-05T05:41:00Z",
    )
    return claim, terminal


@pytest.mark.parametrize("spec_path", [Q_SPEC, G_SPEC])
def test_v5_release_binds_held_self_raw_and_exact_package(spec_path: Path) -> None:
    spec = _load(spec_path)
    release = _release(spec)
    authority.validate_release(
        release, spec, _load(AUTH), _load(HELD), ROOT, PACKAGE
    )
    for field in ("file_sha256", "receipt_sha256"):
        changed = copy.deepcopy(release)
        changed["executable_held_authority"][field] = "sha256:" + "0" * 64
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="scoring release"):
            authority.validate_release(
                changed, spec, _load(AUTH), _load(HELD), ROOT, PACKAGE
            )
    with pytest.raises(ValueError, match="scoring release"):
        authority.validate_release(
            release, spec, _load(AUTH), _load(HELD), ROOT, "3" * 40
        )


def test_v5_terminal_binds_release_raw_self_package_and_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load(Q_SPEC)
    plan = contract.validate_spec(spec, ROOT)
    claim, inner = _claim_and_terminal(spec, monkeypatch)
    release = _release(spec)
    auth = _load(AUTH)
    held = _load(HELD)
    terminal = authority.terminal_receipt(
        inner,
        spec,
        plan,
        claim,
        release,
        RELEASE_FILE_SHA,
        auth,
        held,
        PACKAGE,
        root=ROOT,
    )
    authority.validate_terminal(
        terminal,
        spec,
        plan,
        claim,
        release,
        RELEASE_FILE_SHA,
        auth,
        held,
        PACKAGE,
        root=ROOT,
    )
    for field in ("receipt_sha256", "file_sha256", "package_commit"):
        changed = copy.deepcopy(terminal)
        changed["scoring_release"][field] = (
            "3" * 40 if field == "package_commit" else "sha256:" + "0" * 64
        )
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError):
            authority.validate_terminal(
                changed,
                spec,
                plan,
                claim,
                release,
                RELEASE_FILE_SHA,
                auth,
                held,
                PACKAGE,
                root=ROOT,
            )
    for field in ("receipt_sha256", "file_sha256"):
        changed = copy.deepcopy(terminal)
        changed["executable_held_authority"][field] = "sha256:" + "0" * 64
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="terminal"):
            authority.validate_terminal(
                changed,
                spec,
                plan,
                claim,
                release,
                RELEASE_FILE_SHA,
                auth,
                held,
                PACKAGE,
                root=ROOT,
            )
    with pytest.raises(ValueError, match="terminal"):
        authority.validate_terminal(
            terminal,
            spec,
            plan,
            claim,
            release,
            "sha256:" + "0" * 64,
            auth,
            held,
            PACKAGE,
            root=ROOT,
        )
    with pytest.raises(ValueError, match="scoring release"):
        authority.validate_terminal(
            terminal,
            spec,
            plan,
            claim,
            release,
            RELEASE_FILE_SHA,
            auth,
            held,
            "3" * 40,
            root=ROOT,
        )


def test_v5_held_package_manifest_and_wrappers_are_safe() -> None:
    authority.validate_held(_load(HELD), ROOT)
    manifest = yaml.safe_load((ROOT / authority.MANIFEST_PATH).read_text())
    assert all(
        item["metadata"]["annotations"]
        == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        for item in manifest["items"]
    )
    assert manifest["items"][0]["spec"]["template"]["spec"]["containers"][0][
        "command"
    ] == ["/bin/bash", "/bootstrap/run-v5.sh"]
    for path in (ROOT / authority.RUN_PATH, ROOT / authority.SUBMIT_PATH):
        assert path.stat().st_mode & stat.S_IXUSR
    run = (ROOT / authority.RUN_PATH).read_text()
    submit = (ROOT / authority.SUBMIT_PATH).read_text()
    supplied = set(re.findall(r"--from-file=([A-Za-z0-9_.-]+)=", submit))
    referenced = set(re.findall(r"/bootstrap/([A-Za-z0-9_.-]+)", run))
    referenced.discard("v")  # The versioned wrapper names are expanded by a loop.
    assert referenced <= supplied
    assert submit.index("validate-release") < submit.index("api_key=$(")
    assert "qwen_release_file_sha256" in submit
    assert "glm_release_file_sha256" in submit
    assert "kubectl delete" not in submit and "kubectl apply" not in submit
    assert not Path(
        "docs/evidence/qwen38-study/"
        "2026-09-04-qwen38-autocontinue-generation2-scoring-release-v5.json"
    ).exists()
    assert not Path(
        "docs/evidence/qwen38-study/"
        "2026-09-04-glm53-autocontinue-generation2-scoring-release-v5.json"
    ).exists()


def test_release_raw_digest_is_distinct_from_self_digest() -> None:
    release = _release(_load(Q_SPEC))
    raw = hashlib.sha256(
        authority.self_hosted.canonical_json(release) + b"\n"
    ).hexdigest()
    assert "sha256:" + raw != release["receipt_sha256"]
