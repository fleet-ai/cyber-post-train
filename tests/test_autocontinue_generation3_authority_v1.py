from __future__ import annotations

import copy
import hashlib
import stat
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation3_authority_package_v1 as package
from evals.fleet import autocontinue_generation3_authority_v1 as authority
from evals.fleet import autocontinue_generation3_canary as generation3

ROOT = Path.cwd()
PACKAGE_COMMIT = "1" * 40
RELEASE_FILE_SHA = "sha256:" + "2" * 64


@pytest.fixture(scope="module")
def chain() -> tuple[dict, dict[str, dict], dict[str, dict]]:
    built = package.build_package(ROOT)
    specs = {spec["model"]: spec for spec in authority.specs(ROOT)}
    plans = {model: generation3._render_plan(spec, ROOT) for model, spec in specs.items()}
    return built, specs, plans


def _binding(path: str) -> dict:
    receipt = authority.load(ROOT / path)
    return {
        "path": path,
        "file_sha256": authority.file_sha256(ROOT / path),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _release(spec: dict, plan: dict, built: dict) -> dict:
    auth = authority.load(ROOT / authority.AUTH_PATH)
    entry = next(row for row in auth["models"] if row["model"] == spec["model"])
    receipt = {
        "schema_version": authority.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": entry["authorized_at_utc"],
        "model": spec["model"],
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "package_commit": PACKAGE_COMMIT,
        "root_authorization": _binding(authority.AUTH_PATH),
        "semantic_held_authority": _binding(authority.SEMANTIC_HELD_PATH),
        "split_configmap_package": {
            "schema_version": package.SCHEMA,
            "aggregate_sha256": built["aggregate_sha256"],
            "model_aggregate_sha256": built["model_manifests"][spec["model"]][
                "aggregate_sha256"
            ],
            "objects": sorted(built["configmaps"]),
            "object_json_bytes": built["object_json_bytes"],
        },
        "implementation": authority.implementation(ROOT, spec, built),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 3,
            "hosted_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": entry["authorized_at_utc"],
            "statement": entry["statement"],
        },
        "route_and_inventory": {
            "fresh_fleet_team_and_hosted_models_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "generation2_tombstones_must_match": True,
            "generation3_claim_must_be_absent": True,
            "generation3_job_configmap_and_sfs_root_must_be_absent": True,
            "dedicated_serving_must_remain_user_stopped": True,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = authority.digest(receipt, "receipt_sha256")
    return receipt


def _fast_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    built: dict,
    specs: dict[str, dict],
    plans: dict[str, dict],
) -> None:
    monkeypatch.setattr(package, "build_package", lambda root: built)
    monkeypatch.setattr(
        authority, "validate_root_authorization", lambda receipt, root: list(plans.values())
    )
    monkeypatch.setattr(
        authority, "semantic_held_binding", lambda root: _binding(authority.SEMANTIC_HELD_PATH)
    )
    monkeypatch.setattr(
        generation3, "validate_spec", lambda spec, root: plans[spec["model"]]
    )


def test_root_authorization_is_exact_and_self_digested(chain: tuple) -> None:
    receipt = authority.load(ROOT / authority.AUTH_PATH)
    assert receipt["receipt_sha256"] == authority.digest(receipt, "receipt_sha256")
    assert authority.file_sha256(ROOT / authority.AUTH_PATH) != receipt["receipt_sha256"]
    assert [row["authorized_at_utc"] for row in receipt["models"]] == [
        authority.AUTHORIZED_AT,
        authority.AUTHORIZED_AT,
    ]


@pytest.mark.parametrize("field", ["statement", "execution_id", "cell_id"])
def test_root_authorization_resealed_tamper_rejects(
    chain: tuple, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    _, specs, plans = chain
    monkeypatch.setattr(
        generation3, "validate_spec", lambda spec, root: plans[spec["model"]]
    )
    receipt = authority.load(ROOT / authority.AUTH_PATH)
    changed = copy.deepcopy(receipt)
    changed["models"][0][field] = "tampered"
    changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="root authorization"):
        authority.validate_root_authorization(changed, ROOT)


@pytest.mark.parametrize("model", ["qwen3.8-27b", "glm-5.3"])
def test_release_binds_authority_held_package_and_exact_model(
    chain: tuple, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    built, specs, plans = chain
    _fast_dependencies(monkeypatch, built, specs, plans)
    release = _release(specs[model], plans[model], built)
    authority.validate_release(
        release,
        specs[model],
        authority.load(ROOT / authority.AUTH_PATH),
        ROOT,
        PACKAGE_COMMIT,
    )
    mutations = [
        ("root_authorization", "file_sha256"),
        ("root_authorization", "receipt_sha256"),
        ("semantic_held_authority", "file_sha256"),
        ("semantic_held_authority", "receipt_sha256"),
        ("split_configmap_package", "aggregate_sha256"),
        ("implementation", "package_module_sha256"),
    ]
    for section, field in mutations:
        changed = copy.deepcopy(release)
        changed[section][field] = "sha256:" + "0" * 64
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="scoring release"):
            authority.validate_release(
                changed,
                specs[model],
                authority.load(ROOT / authority.AUTH_PATH),
                ROOT,
                PACKAGE_COMMIT,
            )


def test_claim_and_terminal_bind_release_raw_self_and_package(
    chain: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, specs, plans = chain
    _fast_dependencies(monkeypatch, built, specs, plans)
    spec = specs["qwen3.8-27b"]
    plan = plans["qwen3.8-27b"]
    auth = authority.load(ROOT / authority.AUTH_PATH)
    release = _release(spec, plan, built)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    claim = authority._claim_receipt(
        spec,
        plan,
        release,
        RELEASE_FILE_SHA,
        auth,
        ROOT,
        PACKAGE_COMMIT,
        claimed_at_utc="2026-09-05T05:40:00Z",
    )
    authority.validate_claim(
        claim, spec, plan, release, RELEASE_FILE_SHA, auth, ROOT, PACKAGE_COMMIT
    )
    attempt, config = generation3.generation2._expected_cell_bindings(plan)
    terminal = authority.terminal_receipt(
        spec,
        plan,
        claim,
        {
            "accepted": False,
            "quarantined": True,
            "claim_sha256": attempt,
            "attempt_config_sha256": config,
        },
        release,
        RELEASE_FILE_SHA,
        auth,
        ROOT,
        PACKAGE_COMMIT,
        terminal_at_utc="2026-09-05T05:41:00Z",
    )
    authority.validate_terminal(
        terminal,
        spec,
        plan,
        claim,
        release,
        RELEASE_FILE_SHA,
        auth,
        ROOT,
        PACKAGE_COMMIT,
    )
    for section, field in (
        ("scoring_release", "file_sha256"),
        ("scoring_release", "receipt_sha256"),
        ("scoring_release", "package_commit"),
        ("semantic_held_authority", "receipt_sha256"),
    ):
        changed = copy.deepcopy(terminal)
        changed[section][field] = (
            "3" * 40 if field == "package_commit" else "sha256:" + "0" * 64
        )
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
                ROOT,
                PACKAGE_COMMIT,
            )


def test_split_package_size_sources_and_projected_manifest(chain: tuple) -> None:
    built, _, _ = chain
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.PACKAGE_OBJECT_LIMIT
    for obj in built["object_manifests"].values():
        for entry in obj["entries"]:
            raw = (ROOT / entry["source_path"]).read_bytes()
            assert len(raw) == entry["bytes"]
            assert package.sha256(raw) == entry["sha256"]
    manifest = yaml.safe_load((ROOT / authority.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for item in manifest["items"]:
        assert item["metadata"]["annotations"][
            "cyber-post-train.fleet.ai/launch-authorized"
        ] == "false"
        pod = item["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert item["spec"]["backoffLimit"] == 0
        assert len(pod["volumes"][0]["projected"]["sources"]) == 3


def test_launcher_is_paired_create_once_and_releases_are_absent() -> None:
    submit = (ROOT / authority.SUBMIT_PATH).read_text()
    assert submit.index('test -f "$Q_RELEASE"') < submit.index("api_key=$(")
    assert submit.count("validate-release") == 1  # loop validates both pairs
    assert submit.index("validate-release") < submit.index("kubectl -n \"$NS\" create cm")
    assert "kubectl delete" not in submit and "kubectl apply" not in submit
    assert "fresh_exact_treatment_session_inventory_required" not in submit
    assert (ROOT / authority.SUBMIT_PATH).stat().st_mode & stat.S_IXUSR
    assert (ROOT / authority.RUN_PATH).stat().st_mode & stat.S_IXUSR
    assert not (ROOT / authority.RELEASE_PATHS["qwen3.8-27b"]).exists()
    assert not (ROOT / authority.RELEASE_PATHS["glm-5.3"]).exists()


def test_raw_digests_are_distinct_from_self_digests(chain: tuple) -> None:
    built, specs, plans = chain
    release = _release(specs["glm-5.3"], plans["glm-5.3"], built)
    raw = "sha256:" + hashlib.sha256(package.canonical(release) + b"\n").hexdigest()
    assert raw != release["receipt_sha256"]
    auth = authority.load(ROOT / authority.AUTH_PATH)
    assert authority.file_sha256(ROOT / authority.AUTH_PATH) != auth["receipt_sha256"]
