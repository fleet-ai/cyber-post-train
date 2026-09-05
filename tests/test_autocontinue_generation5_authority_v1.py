from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation5_authority_package_v1 as package
from evals.fleet import autocontinue_generation5_authority_v1 as authority
from evals.fleet import autocontinue_generation5_canary as generation5

ROOT = Path.cwd()
PACKAGE_COMMIT = "1" * 40
RELEASE_FILE_SHA = "sha256:" + "2" * 64


@pytest.fixture(scope="module")
def chain() -> tuple[dict, dict[str, dict], dict[str, dict]]:
    built = package.build_package(ROOT)
    specs = {spec["model"]: spec for spec in authority.specs(ROOT)}
    plans = {
        model: generation5.validate_spec(spec, ROOT) for model, spec in specs.items()
    }
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
        "generation5_spec_sha256": spec["generation5_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "package_commit": PACKAGE_COMMIT,
        "root_authorization": _binding(authority.AUTH_PATH),
        "semantic_held_authority": _binding(generation5.HELD_PATH),
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
            "execution_generation": 5,
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
            "generation1_through_generation5_claims_must_be_absent": True,
            "generation5_job_configmap_and_sfs_root_must_be_absent": True,
            "generation4_incident_and_tombstones_must_match": True,
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
    plans: dict[str, dict],
) -> None:
    monkeypatch.setattr(package, "build_package", lambda root: built)
    monkeypatch.setattr(
        authority, "validate_root_authorization", lambda receipt, root: list(plans.values())
    )
    monkeypatch.setattr(
        authority, "held_binding", lambda root: _binding(generation5.HELD_PATH)
    )
    monkeypatch.setattr(
        generation5, "validate_spec", lambda spec, root: plans[spec["model"]]
    )


def test_root_authorization_is_exact_and_self_digested(chain: tuple) -> None:
    receipt = authority.load(ROOT / authority.AUTH_PATH)
    authority.validate_root_authorization(receipt, ROOT)
    assert receipt["receipt_sha256"] == authority.digest(receipt, "receipt_sha256")
    assert authority.file_sha256(ROOT / authority.AUTH_PATH) != receipt["receipt_sha256"]
    assert [row["authorized_at_utc"] for row in receipt["models"]] == [
        authority.AUTHORIZED_AT,
        authority.AUTHORIZED_AT,
    ]


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("models", "statement"),
        ("models", "execution_id"),
        ("scope", "hosted_only"),
        ("scope", "bulk_release_authorized"),
        ("evidence", "generation4_incident_receipt_sha256"),
    ],
)
def test_root_authorization_resealed_tamper_rejects(
    chain: tuple,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
) -> None:
    built, _, plans = chain
    selected = authority.specs(ROOT)
    monkeypatch.setattr(authority, "specs", lambda root: selected)
    monkeypatch.setattr(
        generation5, "validate_spec", lambda spec, root: plans[spec["model"]]
    )
    receipt = authority.load(ROOT / authority.AUTH_PATH)
    changed = copy.deepcopy(receipt)
    target = changed[section][0] if section == "models" else changed[section]
    target[field] = not target[field] if isinstance(target[field], bool) else "tampered"
    changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="root authorization"):
        authority.validate_root_authorization(changed, ROOT)


@pytest.mark.parametrize("model", ["qwen3.8-27b", "glm-5.3"])
def test_release_binds_authority_held_package_and_exact_model(
    chain: tuple, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    built, specs, plans = chain
    _fast_dependencies(monkeypatch, built, plans)
    release = _release(specs[model], plans[model], built)
    auth = authority.load(ROOT / authority.AUTH_PATH)
    assert release == authority._release_receipt(
        specs[model], auth, ROOT, PACKAGE_COMMIT, built=built
    )
    authority.validate_release(release, specs[model], auth, ROOT, PACKAGE_COMMIT)
    mutations = [
        ("root_authorization", "file_sha256"),
        ("root_authorization", "receipt_sha256"),
        ("semantic_held_authority", "file_sha256"),
        ("semantic_held_authority", "receipt_sha256"),
        ("split_configmap_package", "aggregate_sha256"),
        ("implementation", "package_module_sha256"),
        ("authorization", "dedicated_serving_authorized"),
    ]
    for section, field in mutations:
        changed = copy.deepcopy(release)
        changed[section][field] = (
            True if isinstance(changed[section][field], bool) else "sha256:" + "0" * 64
        )
        changed["receipt_sha256"] = authority.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="scoring release"):
            authority.validate_release(changed, specs[model], auth, ROOT, PACKAGE_COMMIT)


def test_claim_and_terminal_bind_release_raw_self_and_package(
    chain: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, specs, plans = chain
    _fast_dependencies(monkeypatch, built, plans)
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
        claimed_at_utc="2026-09-05T06:40:00Z",
    )
    authority.validate_claim(
        claim, spec, plan, release, RELEASE_FILE_SHA, auth, ROOT, PACKAGE_COMMIT
    )
    attempt, config = generation5.generation2._expected_cell_bindings(plan)
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
        terminal_at_utc="2026-09-05T06:41:00Z",
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
        ("root_authorization", "receipt_sha256"),
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


def test_released_configmaps_preserve_exact_raw_release_bytes(
    chain: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built, specs, plans = chain
    releases = {
        model: _release(specs[model], plans[model], built) for model in specs
    }
    paths: dict[str, str] = {}
    raw_by_model: dict[str, bytes] = {}
    for model, release in releases.items():
        raw = (json.dumps(release, indent=2) + "\n").encode()
        path = tmp_path / f"{model}.json"
        path.write_bytes(raw)
        paths[model] = path.name
        raw_by_model[model] = raw
    monkeypatch.setattr(package, "build_package", lambda root: built)
    monkeypatch.setattr(authority, "RELEASE_PATHS", paths)
    monkeypatch.setattr(authority, "load", lambda path: {})
    monkeypatch.setattr(authority, "validate_release", lambda *args: None)
    bound = {
        model: (release, "sha256:" + hashlib.sha256(raw_by_model[model]).hexdigest())
        for model, release in releases.items()
    }
    configmaps = package.released_configmaps(
        tmp_path, bound, PACKAGE_COMMIT, {"schema_version": "test-route"}
    )
    for model, name in package.MODEL_NAMES.items():
        assert configmaps[name]["data"]["release.json"].encode() == raw_by_model[model]
        assert configmaps[name]["data"]["release_file_sha256"] == bound[model][1]


def test_split_package_includes_g5_and_semantic_dependencies(chain: tuple) -> None:
    built, _, _ = chain
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.PACKAGE_OBJECT_LIMIT
    packaged = {
        entry["source_path"]
        for obj in built["object_manifests"].values()
        for entry in obj["entries"]
    }
    required = {
        generation5.INCIDENT_PATH,
        generation5.TOMBSTONE_PATH,
        generation5.HELD_PATH,
        generation5.MODULE_PATH,
        authority.MODULE_PATH,
        authority.PACKAGE_MODULE_PATH,
        authority.AUTH_PATH,
        authority.MANIFEST_PATH,
        authority.RUN_PATH,
        authority.SUBMIT_PATH,
        authority.MANIFEST_AUTH_PATH,
    }
    assert required <= packaged
    assert all((ROOT / entry).is_file() for entry in packaged)


def test_manifest_and_runner_preserve_exact_execution_policy() -> None:
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
    run = (ROOT / authority.RUN_PATH).read_text()
    assert run.index('cd "$ROOT"') < run.index(
        "-m evals.fleet.autocontinue_generation5_authority_package_v1"
    )
    assert (ROOT / authority.RUN_PATH).stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("model", ["qwen3.8-27b", "glm-5.3"])
def test_authority_runner_reconstructs_empty_root_and_validates_raw_release(
    chain: tuple, tmp_path: Path, model: str
) -> None:
    built, specs, plans = chain
    bootstrap = tmp_path / "bootstrap"
    destination = tmp_path / "reconstructed"
    outside = tmp_path / "outside"
    for directory in (bootstrap, outside):
        directory.mkdir()
    for name in (
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.MODEL_NAMES[model],
    ):
        for key, value in built["configmaps"][name]["data"].items():
            (bootstrap / key).write_text(value)
    release = _release(specs[model], plans[model], built)
    raw_release = (json.dumps(release, indent=2) + "\n").encode()
    release_digest = "sha256:" + hashlib.sha256(raw_release).hexdigest()
    (bootstrap / "release.json").write_bytes(raw_release)
    (bootstrap / "launch-route.json").write_text("{}\n")
    env = dict(os.environ)
    env.update(
        {
            "CYBER_ROOT": str(destination),
            "GENERATION5_BOOTSTRAP": str(bootstrap),
            "GENERATION5_PACKAGE_AGGREGATE_SHA256": built["model_manifests"][model][
                "aggregate_sha256"
            ],
            "GENERATION5_RELEASE_FILE_SHA256": release_digest,
            "GENERATION5_PACKAGE_COMMIT": PACKAGE_COMMIT,
            "GENERATION5_MODEL": model,
            "GENERATION5_JOB_NAME": generation5.EXPECTED[model]["job_name"],
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / authority.RUN_PATH)],
        cwd=outside,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert "JOB_UID" in result.stderr
    assert (destination / authority.AUTH_PATH).is_file()
    assert (destination / generation5.INCIDENT_PATH).is_file()
    assert (destination / generation5.TOMBSTONE_PATH).is_file()
    installed = destination / "evals/fleet/configs/generation5-release.json"
    assert installed.read_bytes() == raw_release
    assert authority.file_sha256(installed) == release_digest


def test_launcher_is_paired_create_once_and_releases_are_absent() -> None:
    submit = (ROOT / authority.SUBMIT_PATH).read_text()
    assert submit.index('test -f "$Q_RELEASE"') < submit.index("api_key=$(")
    assert submit.index("validate-release") < submit.index(
        'kubectl -n "$NS" create cm "$INTENT"'
    )
    assert submit.index('kubectl -n "$NS" create cm "$INTENT"') < submit.index(
        'kubectl -n "$NS" create -f "$bundle"'
    )
    assert submit.count("/shared/cell-execution-claims/") == 10
    assert "observe-route" in submit and "_validate_inventory_for_task" in submit
    assert submit.count("immutable_submission_snapshot assert-stable") == 2
    assert "immutable_submission_snapshot materialize" in submit
    assert authority.SNAPSHOT_MODULE_PATH in package.CORE_B_PATHS
    assert 'bundle_sha=$(sha256sum "$bundle"' in submit
    assert 'cat-file -e "$PACKAGE_COMMIT:$release_path"' in submit
    assert submit.index('create --dry-run=server -f "$bundle"') < submit.index(
        'kubectl -n "$NS" create -f "$bundle"'
    )
    assert "kubectl delete" not in submit and "kubectl apply" not in submit
    for kind, name in (
        ("raycluster", "ft-run-98c32208-5gpb2"),
        ("raycluster", "ft-run-9e92209d-pppzg"),
        ("pod", "ft-run-98c32208-5gpb2-head-7qxwc"),
        ("pod", "ft-run-9e92209d-pppzg-head-rc9nd"),
        ("workload", "rayjob-ft-run-98c32208-f6a62"),
        ("workload", "rayjob-ft-run-9e92209d-0580b"),
    ):
        assert f'get {kind} "$name" --ignore-not-found' in submit
        assert name in submit
    assert (ROOT / authority.SUBMIT_PATH).stat().st_mode & stat.S_IXUSR
    releases = [ROOT / authority.RELEASE_PATHS[model] for model in generation5.EXPECTED]
    assert len({path.exists() for path in releases}) == 1
    if releases[0].exists():
        receipts = {
            model: authority.load(ROOT / authority.RELEASE_PATHS[model])
            for model in generation5.EXPECTED
        }
        assert len({receipt["package_commit"] for receipt in receipts.values()}) == 1
        commit = next(iter(receipts.values()))["package_commit"]
        auth = authority.load(ROOT / authority.AUTH_PATH)
        for model, receipt in receipts.items():
            spec = authority.load(ROOT / generation5.G5_SPEC_PATHS[model])
            authority.validate_release(receipt, spec, auth, ROOT, commit)
            result = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}:{authority.RELEASE_PATHS[model]}"],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            assert result.returncode != 0


def test_raw_digests_are_distinct_from_self_digests(chain: tuple) -> None:
    built, specs, plans = chain
    release = _release(specs["glm-5.3"], plans["glm-5.3"], built)
    raw = "sha256:" + hashlib.sha256(package.canonical(release) + b"\n").hexdigest()
    assert raw != release["receipt_sha256"]
    auth = authority.load(ROOT / authority.AUTH_PATH)
    assert authority.file_sha256(ROOT / authority.AUTH_PATH) != auth["receipt_sha256"]
