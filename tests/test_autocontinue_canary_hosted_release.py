from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import autocontinue_canary_controller as canary
from evals.fleet import autocontinue_canary_hosted_release as hosted_release

ROOT = Path(__file__).parents[1]
Q_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
G_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json"
Q_SUCCESSOR_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_SUCCESSOR_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"


def _leaf_paths(value, prefix=()):
    for key, child in value.items():
        path = (*prefix, key)
        if isinstance(child, dict):
            yield from _leaf_paths(child, path)
        else:
            yield path


def _change_leaf(value, path):
    parent = value
    for key in path[:-1]:
        parent = parent[key]
    old = parent[path[-1]]
    if isinstance(old, bool):
        parent[path[-1]] = not old
    elif isinstance(old, int):
        parent[path[-1]] = old + 1
    elif isinstance(old, str):
        parent[path[-1]] = f"{old}-tampered"
    elif old is None:
        parent[path[-1]] = "unexpected"
    elif isinstance(old, list):
        parent[path[-1]] = [*old, "unexpected"]
    else:  # pragma: no cover - evidence is intentionally scalar-only at leaves
        raise AssertionError(f"unsupported evidence leaf type: {type(old)}")


@pytest.mark.parametrize("plan_path", [Q_PLAN, G_PLAN])
def test_hosted_only_preflight_bundle_is_exact(plan_path: Path) -> None:
    plan = canary.load_object(plan_path)
    bundle = hosted_release.validate_preflight_bundle(plan, ROOT)
    assert bundle["route"]["authorization"] == {
        "route": "HOSTED_ONLY",
        "scored_launch_authorized": False,
        "dedicated_recreation_authorized": False,
    }
    assert bundle["route"]["dedicated_serving"]["state"] == "USER_STOPPED_UNAVAILABLE"
    assert bundle["duplicate"]["exact_treatment_session_rows"] == 0
    assert bundle["duplicate"]["global_cell_claim_absent"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["fleet_team"].__setitem__(
            "both_preflights_authenticated_as_fleet", False
        ),
        lambda value: value["hosted_routes"].__setitem__(
            "qwen_route_available_in_bound_health_receipt", False
        ),
        lambda value: value["dedicated_serving"].__setitem__("state", "RUNNING"),
        lambda value: value["dedicated_serving"].__setitem__("rayjobs_present", 1),
        lambda value: value["dedicated_serving"].__setitem__(
            "historical_parity_receipt_is_not_current_live_route_authority", False
        ),
        lambda value: value["candidate_absence"].__setitem__("exact_scored_jobs_present", 1),
        lambda value: value["candidate_absence"].__setitem__(
            "qwen_global_cell_claim_absent", False
        ),
        lambda value: value["authorization"].__setitem__("scored_launch_authorized", True),
        lambda value: value["privacy"].__setitem__("scores_included", True),
    ],
)
def test_hosted_only_route_rejects_resealed_tampering(mutate) -> None:
    route = copy.deepcopy(canary.load_object(ROOT / hosted_release.ROUTE_PATH))
    mutate(route)
    route["receipt_sha256"] = canary.digest_without(route, "receipt_sha256")
    with pytest.raises(ValueError, match="route evidence"):
        hosted_release.validate_hosted_route(route)


def test_hosted_only_route_semantics_reject_every_resealed_leaf_and_unknown_key(
    monkeypatch,
) -> None:
    original = canary.load_object(ROOT / hosted_release.ROUTE_PATH)
    for path in _leaf_paths({k: v for k, v in original.items() if k != "receipt_sha256"}):
        changed = copy.deepcopy(original)
        _change_leaf(changed, path)
        changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
        monkeypatch.setattr(hosted_release, "ROUTE_SHA", changed["receipt_sha256"])
        with pytest.raises(ValueError, match="route evidence"):
            hosted_release.validate_hosted_route(changed)

    changed = copy.deepcopy(original)
    changed["unknown_authority"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    monkeypatch.setattr(hosted_release, "ROUTE_SHA", changed["receipt_sha256"])
    with pytest.raises(ValueError, match="route evidence"):
        hosted_release.validate_hosted_route(changed)


@pytest.mark.parametrize(
    ("target", "mutate"),
    [
        ("preflight", lambda value: value.__setitem__("exact_treatment_sessions_reconciled", 1)),
        ("preflight", lambda value: value.__setitem__("fleet_team_id", "wrong")),
        ("post_exit", lambda value: value["job"].__setitem__("succeeded", 0)),
        ("post_exit", lambda value: value["pod"].__setitem__("restart_count", 1)),
        (
            "post_exit",
            lambda value: value["reconciliation"].__setitem__("global_cell_claim_absent", False),
        ),
        ("duplicate", lambda value: value.__setitem__("pagination_exhausted", False)),
        ("duplicate", lambda value: value.__setitem__("exact_treatment_session_rows", 1)),
        ("duplicate", lambda value: value.__setitem__("active_attempts", 1)),
    ],
)
def test_hosted_only_bundle_rejects_semantic_tampering(monkeypatch, target, mutate) -> None:
    plan = canary.load_object(Q_PLAN)
    expected = hosted_release.BUNDLES[plan["shard_key"]]
    originals = {
        "preauth": canary.load_object(ROOT / expected["preauth_path"]),
        "preflight": canary.load_object(ROOT / expected["preflight_path"]),
        "post_exit": canary.load_object(ROOT / expected["post_exit_path"]),
        "duplicate": canary.load_object(ROOT / expected["duplicate_path"]),
        "route": canary.load_object(ROOT / hosted_release.ROUTE_PATH),
    }
    changed = copy.deepcopy(originals[target])
    mutate(changed)
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")

    path_to_name = {
        expected["preauth_path"]: "preauth",
        expected["preflight_path"]: "preflight",
        expected["post_exit_path"]: "post_exit",
        expected["duplicate_path"]: "duplicate",
        hosted_release.ROUTE_PATH: "route",
    }

    def fake_load(_root: Path, path: str, _receipt_sha: str):
        name = path_to_name[path]
        return changed if name == target else originals[name]

    monkeypatch.setattr(hosted_release, "_load_exact", fake_load)
    with pytest.raises(ValueError, match="preflight bundle"):
        hosted_release.validate_preflight_bundle(plan, ROOT)


@pytest.mark.parametrize("target", ["preflight", "post_exit", "duplicate"])
def test_hosted_only_bundle_rejects_every_resealed_leaf_and_unknown_key(
    monkeypatch, target
) -> None:
    plan = canary.load_object(Q_PLAN)
    bundle = hosted_release.BUNDLES[plan["shard_key"]]
    paths = {
        "preauth": bundle["preauth_path"],
        "preflight": bundle["preflight_path"],
        "post_exit": bundle["post_exit_path"],
        "duplicate": bundle["duplicate_path"],
        "route": hosted_release.ROUTE_PATH,
    }
    originals = {name: canary.load_object(ROOT / path) for name, path in paths.items()}
    path_to_name = {path: name for name, path in paths.items()}

    candidates = []
    for leaf_path in _leaf_paths(
        {key: value for key, value in originals[target].items() if key != "receipt_sha256"}
    ):
        changed = copy.deepcopy(originals[target])
        _change_leaf(changed, leaf_path)
        changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
        candidates.append(changed)
    changed = copy.deepcopy(originals[target])
    changed["unknown_authority"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    candidates.append(changed)

    for changed in candidates:

        def fake_load(_root: Path, path: str, _receipt_sha: str, changed=changed):
            name = path_to_name[path]
            return changed if name == target else originals[name]

        monkeypatch.setattr(hosted_release, "_load_exact", fake_load)
        with pytest.raises(ValueError, match="preflight bundle"):
            hosted_release.validate_preflight_bundle(plan, ROOT)


@pytest.mark.parametrize("plan_path", [Q_SUCCESSOR_PLAN, G_SUCCESSOR_PLAN])
def test_successor_preflight_bundle_is_exact_and_does_not_reinterpret_phase_a(
    plan_path: Path,
) -> None:
    plan = canary.load_object(plan_path)
    evidence = hosted_release.validate_preflight_bundle(plan, ROOT)
    assert evidence["route"] is None
    assert evidence["preflight"]["sfs_job_roots_reconciled"] == 101
    assert evidence["post_exit"]["reconciliation"]["sfs_mount_path"] == "/shared"
    assert evidence["duplicate"]["exact_treatment_session_rows"] == 0


def test_successor_preflight_authorization_rejects_semantic_drift(monkeypatch) -> None:
    plan = canary.load_object(Q_SUCCESSOR_PLAN)
    bundle = hosted_release.BUNDLES[plan["shard_key"]]
    original = canary.load_object(ROOT / bundle["preauth_path"])
    changed = copy.deepcopy(original)
    changed["authorization"]["launch_authorized"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    real_load = hosted_release._load_exact

    def load(root: Path, path: str, receipt_sha: str):
        if path == bundle["preauth_path"]:
            return changed
        return real_load(root, path, receipt_sha)

    monkeypatch.setattr(hosted_release, "_load_exact", load)
    with pytest.raises(ValueError, match="preflight authorization"):
        hosted_release.validate_preflight_bundle(plan, ROOT)


@pytest.mark.parametrize(
    ("plan_path", "release_path"),
    [
        (
            Q_SUCCESSOR_PLAN,
            ROOT
            / "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-successor-held-release-v3.json",
        ),
        (
            G_SUCCESSOR_PLAN,
            ROOT
            / "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-successor-held-release-v3.json",
        ),
    ],
)
def test_successor_held_release_is_exact_and_rejects_all_resealed_drift(
    plan_path: Path, release_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    original = canary.load_object(release_path)
    hosted_release.validate_successor_held_release(original, plan, ROOT)
    assert original["authorization"]["launch_authorized"] is False
    assert original["route"]["dedicated_serving_state"] == "USER_STOPPED_UNAVAILABLE"

    candidates = []
    for path in _leaf_paths(
        {key: value for key, value in original.items() if key != "receipt_sha256"}
    ):
        changed = copy.deepcopy(original)
        _change_leaf(changed, path)
        changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
        candidates.append(changed)
    changed = copy.deepcopy(original)
    changed["unknown_authority"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    candidates.append(changed)

    for changed in candidates:
        with pytest.raises(ValueError, match="held release"):
            hosted_release.validate_successor_held_release(changed, plan, ROOT)
