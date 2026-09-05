from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import autocontinue_generation2_canary_v2 as generation2

ROOT = Path.cwd()
Q_SPEC = Path("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json")
G_SPEC = Path("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json")
HELD = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json"
)
MANIFEST = Path("evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml")
RUN = Path("evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh")
SUBMIT = Path("evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh")


def _load(path: Path) -> dict:
    return generation2.load(path)


@pytest.mark.parametrize("path", [Q_SPEC, G_SPEC])
def test_spec_and_rendered_plan_have_separate_self_digests(path: Path) -> None:
    spec = _load(path)
    plan = generation2.validate_spec(spec, ROOT)
    old = _load(ROOT / spec["predecessor_generation2_spec"]["path"])

    assert spec["generation2_spec_sha256"] == generation2.digest(
        spec, "generation2_spec_sha256"
    )
    assert plan["plan_sha256"] == generation2.digest(plan, "plan_sha256")
    assert plan["plan_sha256"] == spec["rendered_plan_sha256"]
    assert plan["plan_sha256"] != old["plan_sha256"]
    assert plan["harness"]["compaction_headroom_tokens"] == 20000
    assert plan["treatment_block"]["harness"]["compaction_headroom_tokens"] == 20000


@pytest.mark.parametrize("path", [Q_SPEC, G_SPEC])
def test_spec_or_rendered_plan_digest_tampering_fails_closed(path: Path) -> None:
    spec = copy.deepcopy(_load(path))
    spec["generation2_spec_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="specification drifted"):
        generation2.validate_spec(spec, ROOT)

    spec = copy.deepcopy(_load(path))
    spec["rendered_plan_sha256"] = "sha256:" + "0" * 64
    spec["generation2_spec_sha256"] = generation2.digest(
        spec, "generation2_spec_sha256"
    )
    with pytest.raises(ValueError, match="rendered plan digest drifted"):
        generation2.validate_spec(spec, ROOT)


def _release(spec: dict, package_commit: str) -> dict:
    old = _load(ROOT / generation2.V1_SPEC_PATHS[spec["model"]])
    release = {
        "schema_version": generation2.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": "2026-09-05T04:00:00Z",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": spec["rendered_plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "incident_receipt_sha256": held_v1.INCIDENT_SHA,
        "tombstone_bundle_receipt_sha256": held_v1._tombstones(ROOT)[
            "receipt_sha256"
        ],
        "package_commit": package_commit,
        "implementation": {
            "job_name": held_v1.EXPECTED[spec["model"]]["job_name"],
            "configmap_name": held_v1.EXPECTED[spec["model"]]["configmap_name"],
            "sfs_root": old["identities"]["sfs_root"],
        },
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "statement": "release one exact canary cell",
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


def test_release_binds_both_spec_and_rendered_plan_digests() -> None:
    spec = _load(Q_SPEC)
    package_commit = "1" * 40
    release = _release(spec, package_commit)
    generation2.validate_release(release, spec, ROOT, package_commit)

    for field in ("generation2_spec_sha256", "plan_sha256"):
        changed = copy.deepcopy(release)
        changed[field] = "sha256:" + "0" * 64
        changed["receipt_sha256"] = generation2.digest(changed, "receipt_sha256")
        with pytest.raises(ValueError, match="not authoritative"):
            generation2.validate_release(changed, spec, ROOT, package_commit)


def test_claim_and_terminal_bind_both_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load(Q_SPEC)
    plan = generation2.validate_spec(spec, ROOT)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    receipt = generation2.claim_execution_generation(
        spec, plan, tmp_path / "claims", repo=ROOT
    )
    assert receipt["generation2_spec_sha256"] == spec["generation2_spec_sha256"]
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert json.loads(
        (tmp_path / "claims" / f"{spec['execution']['execution_id'][7:]}.json").read_text()
    ) == receipt
    with pytest.raises(RuntimeError, match="already claimed"):
        generation2.claim_execution_generation(
            spec, plan, tmp_path / "claims", repo=ROOT
        )

    terminal = generation2.terminal_receipt(
        spec, plan, receipt, {"accepted": 1, "quarantined": 0}
    )
    assert terminal["generation2_spec_sha256"] == spec["generation2_spec_sha256"]
    assert terminal["plan_sha256"] == plan["plan_sha256"]
    assert terminal["receipt_sha256"] == generation2.digest(terminal, "receipt_sha256")


def test_held_successor_binds_new_files_without_changing_v1() -> None:
    release = _load(HELD)
    generation2.validate_held(release, [_load(Q_SPEC), _load(G_SPEC)], ROOT)
    assert release["launch_authorized"] is False
    assert release["supersedes_held_receipt"] == {
        "path": generation2.V1_HELD_PATH,
        "receipt_sha256": _load(ROOT / generation2.V1_HELD_PATH)["receipt_sha256"],
    }
    for prefix in ("module", "manifest", "run", "submit"):
        path = ROOT / release["implementation"][f"{prefix}_path"]
        assert release["implementation"][f"{prefix}_sha256"] == generation2.file_sha256(
            path
        )


def test_successor_manifest_is_held_high_priority_and_never_preempted() -> None:
    items = yaml.safe_load(MANIFEST.read_text())["items"]
    assert {item["metadata"]["name"] for item in items} == {
        "chris-q38-ac-r004-a1-g2-v1",
        "chris-glm53-ac-r013-a1-g2-v1",
    }
    for item in items:
        assert item["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        pod = item["spec"]["template"]["spec"]
        assert item["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-train-high"
        assert pod["preemptionPolicy"] == "Never"
        assert pod["restartPolicy"] == "Never"


def test_successor_wrapper_gates_before_images_and_submitter_remains_held() -> None:
    run = RUN.read_text()
    assert run.index("validate-spec") < run.index("docker info") < run.index(
        "autocontinue_generation2_canary_v2 run"
    )
    submit = SUBMIT.read_text()
    assert submit.index("autocontinue_generation2_canary_v2 preview") < submit.index(
        "create --dry-run=server"
    ) < submit.index("successor package is HELD")
    assert "kubectl delete" not in submit
    assert "kubectl apply" not in submit
