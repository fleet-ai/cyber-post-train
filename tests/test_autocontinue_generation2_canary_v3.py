from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import autocontinue_generation2_canary_v2 as held_v2
from evals.fleet import autocontinue_generation2_canary_v3 as generation2

ROOT = Path.cwd()
Q_SPEC = Path("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json")
G_SPEC = Path("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json")
HELD = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json"
)
AUTH_TIME = "2026-09-05T05:00:00Z"
AUTH_STATEMENT = "Authorize one exact create-once generation-2 canary cell."


def _load(path: Path) -> dict:
    return generation2.load(path)


def _release(spec: dict, package_commit: str = "1" * 40) -> dict:
    plan = generation2.validate_spec(spec, ROOT)
    release = {
        "schema_version": generation2.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": AUTH_TIME,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "incident_receipt_sha256": held_v1.INCIDENT_SHA,
        "tombstone_bundle_receipt_sha256": held_v1._tombstones(ROOT)[
            "receipt_sha256"
        ],
        "package_commit": package_commit,
        "implementation": generation2._expected_implementation(ROOT, spec["model"]),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": AUTH_TIME,
            "statement": AUTH_STATEMENT,
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
    release["receipt_sha256"] = generation2.digest(release, "receipt_sha256")
    return release


def _claim(spec: dict, plan: dict) -> dict:
    claim = {
        "schema_version": "fleet-statistical-cell-execution-claim-v2",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": "11111111-1111-4111-8111-111111111111",
        "pod_uid": "22222222-2222-4222-8222-222222222222",
        "claimed_at_utc": "2026-09-05T05:01:00Z",
        "generation_1_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    claim["receipt_sha256"] = generation2.digest(claim, "receipt_sha256")
    return claim


@pytest.mark.parametrize("path", [Q_SPEC, G_SPEC])
def test_v3_release_is_exact(path: Path) -> None:
    spec = _load(path)
    release = _release(spec)
    generation2.validate_release(
        release,
        spec,
        ROOT,
        "1" * 40,
        authorized_at_utc=AUTH_TIME,
        authorization_statement=AUTH_STATEMENT,
    )
    mutations = [
        ("released_at_utc", "not-a-time"),
        ("package_commit", "2" * 40),
    ]
    for field, value in mutations:
        changed = copy.deepcopy(release)
        changed[field] = value
        changed["receipt_sha256"] = generation2.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError):
            generation2.validate_release(
                changed,
                spec,
                ROOT,
                "1" * 40,
                authorized_at_utc=AUTH_TIME,
                authorization_statement=AUTH_STATEMENT,
            )
    changed = copy.deepcopy(release)
    changed["authorization"]["statement"] = "arbitrary"
    changed["receipt_sha256"] = generation2.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        generation2.validate_release(
            changed,
            spec,
            ROOT,
            "1" * 40,
            authorized_at_utc=AUTH_TIME,
            authorization_statement=AUTH_STATEMENT,
        )
    changed = copy.deepcopy(release)
    changed["implementation"]["unexpected"] = True
    changed["receipt_sha256"] = generation2.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        generation2.validate_release(
            changed,
            spec,
            ROOT,
            "1" * 40,
            authorized_at_utc=AUTH_TIME,
            authorization_statement=AUTH_STATEMENT,
        )
    for expected_time, expected_statement in [
        ("2026-09-05T05:00:01Z", AUTH_STATEMENT),
        (AUTH_TIME, "different exact authority"),
        ("not-a-time", AUTH_STATEMENT),
    ]:
        with pytest.raises(ValueError):
            generation2.validate_release(
                release,
                spec,
                ROOT,
                "1" * 40,
                authorized_at_utc=expected_time,
                authorization_statement=expected_statement,
            )


def test_claim_rejects_cross_model_spec_plan_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q_spec = _load(Q_SPEC)
    g_plan = generation2.validate_spec(_load(G_SPEC), ROOT)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    claim_root = tmp_path / "claims"
    with pytest.raises(ValueError, match="spec-plan chain drifted"):
        generation2.claim_execution_generation(q_spec, g_plan, claim_root, repo=ROOT)
    assert not claim_root.exists()


def test_terminal_validates_claim_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _load(Q_SPEC)
    plan = generation2.validate_spec(spec, ROOT)
    claim = _claim(spec, plan)
    monkeypatch.setenv("JOB_UID", claim["job_uid"])
    monkeypatch.setenv("POD_UID", claim["pod_uid"])
    attempt_claim_sha256, config_sha256 = generation2._expected_cell_bindings(plan)
    result = {
        "accepted": True,
        "quarantined": False,
        "claim_sha256": attempt_claim_sha256,
        "attempt_config_sha256": config_sha256,
        "acceptance_receipt_sha256": "sha256:" + "5" * 64,
        "session_id": "33333333-3333-4333-8333-333333333333",
        "verifier_execution_id": "44444444-4444-4444-8444-444444444444",
        "session_ingest_completed": True,
        "cleanup_completed": True,
    }
    terminal = generation2.terminal_receipt(
        spec,
        plan,
        claim,
        result,
        root=ROOT,
        terminal_at_utc="2026-09-05T05:02:00Z",
    )
    generation2.validate_terminal(terminal, spec, plan, claim, root=ROOT)

    bad_claim = copy.deepcopy(claim)
    bad_claim["plan_sha256"] = "sha256:" + "0" * 64
    bad_claim["receipt_sha256"] = generation2.digest(bad_claim, "receipt_sha256")
    with pytest.raises(ValueError):
        generation2.validate_terminal(terminal, spec, plan, bad_claim, root=ROOT)

    for path, value in [
        (("terminal_at_utc",), ""),
        (("result", "accepted"), 99),
        (("result", "quarantined"), True),
        (("result", "claim_sha256"), "sha256:" + "0" * 64),
        (("result", "attempt_config_sha256"), "sha256:" + "0" * 64),
        (("result", "session_ingest_completed"), False),
    ]:
        changed = copy.deepcopy(terminal)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        changed["receipt_sha256"] = generation2.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError):
            generation2.validate_terminal(changed, spec, plan, claim, root=ROOT)


def test_v3_held_receipt_is_append_only_and_valid() -> None:
    specs = [_load(Q_SPEC), _load(G_SPEC)]
    release = _load(HELD)
    generation2.validate_held(release, specs, ROOT)
    assert release["supersedes_held_receipt"] == {
        "path": generation2.V2_HELD_PATH,
        "receipt_sha256": _load(ROOT / generation2.V2_HELD_PATH)["receipt_sha256"],
    }


def test_v1_and_v2_paths_remain_the_authoritative_predecessors() -> None:
    assert held_v2.V1_HELD_PATH.endswith("held-v1.json")
    assert generation2.V2_HELD_PATH.endswith("held-v2.json")
    assert json.loads((ROOT / Q_SPEC).read_text())["schema_version"].endswith("spec-v2")
