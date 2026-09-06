import copy
import json
import re
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_single_slot_package_v4 as scored_package
from evals.fleet import hosted_glm_rank30_single_slot_release_package_v4 as release_package
from evals.fleet import hosted_glm_rank30_single_slot_release_v4 as release
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
V6 = (
    ROOT
    / "docs/evidence/glm53-study"
    / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v6-passed.json"
)


def test_v6_pass_receipt_is_exact_and_score_free() -> None:
    binding = successor.validate_runtime_gate_diagnostic(V6)
    assert binding == {
        "path": str(V6),
        "receipt_sha256": successor.DIAGNOSTIC_SELF_SHA256,
        "file_sha256": successor.DIAGNOSTIC_FILE_SHA256,
        "job_uid": successor.DIAGNOSTIC_JOB_UID,
        "pod_uid": successor.DIAGNOSTIC_POD_UID,
        "status": "PASSED_TO_SESSION_BOUNDARY",
        "last_completed_phase": "09-release-projection",
        "plan_sha256": successor.DIAGNOSTIC_PLAN_SHA256,
        "projected_attempts": [1, 2, 3, 4],
        "model_task_session_verifier_scoring_calls": 0,
        "api_mutation_calls": 0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "FAILED"),
        ("projected_attempts", [1, 2, 3]),
        ("model_calls", 1),
        ("session_boundary_executed", True),
    ],
)
def test_v6_pass_receipt_mutations_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    receipt = json.loads(V6.read_text())
    receipt[field] = value
    receipt["receipt_sha256"] = self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    path = tmp_path / "DIAGNOSTIC.json"
    path.write_bytes(self_hosted.canonical_json(receipt))
    with pytest.raises(RuntimeError, match="diagnostic drifted"):
        successor.validate_runtime_gate_diagnostic(path)


def test_scored_package_uses_corrected_v6_gated_runtime_and_stays_held() -> None:
    rendered = scored_package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    assert rendered["launch_authorized"] is False
    assert rendered["scoring_authorized"] is False
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == (
        "chris-glm53-exact100-hosted-r030-whole-task-g1-package-v1"
    )
    assert job["metadata"]["name"] == (
        "chris-glm53-exact100-hosted-r030-whole-task-g1-v1"
    )
    assert "hosted_glm_rank30_single_slot_v5" in configmap["data"]["run.sh"]
    assert "hosted_glm_rank30_single_slot_runtime_v4" in configmap["data"]["run.sh"]
    assert "validate_runtime_gate_diagnostic" in configmap["data"]["bulk.py"]
    assert "build_runtime_plan" in configmap["data"]["single_slot_v4.py"]
    assert "release.json" not in configmap["data"]


def test_scored_package_closure_materializes_every_imported_successor() -> None:
    configmap = scored_package.render(ROOT)["objects"]["items"][0]
    mappings = dict(re.findall(r"([\w.-]+):([\w.-]+)", configmap["data"]["run.sh"]))
    assert mappings["single_slot_v1.py"] == "hosted_glm_rank30_single_slot_v1.py"
    assert mappings["single_slot_v3.py"] == "hosted_glm_rank30_single_slot_v3.py"
    assert mappings["single_slot_v4.py"] == "hosted_glm_rank30_single_slot_v4.py"
    assert mappings["bulk.py"] == "hosted_glm_rank30_single_slot_v5.py"
    assert mappings["bulk_runtime.py"] == "hosted_glm_rank30_single_slot_runtime_v4.py"
    assert {
        "single_slot_v1.py",
        "single_slot_v3.py",
        "single_slot_v4.py",
        "bulk.py",
        "bulk_runtime.py",
    }.issubset(configmap["data"])


def test_release_package_is_fresh_held_and_binds_scored_and_v6_bytes() -> None:
    rendered = release_package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    assert rendered["launch_authorized"] is False
    assert rendered["scoring_authorized"] is False
    assert rendered["model_calls_authorized"] is False
    assert rendered["source_package_sha256"] == scored_package.source_package_sha256(
        ROOT
    )
    assert (
        rendered["v6_diagnostic_receipt_sha256"]
        == successor.DIAGNOSTIC_SELF_SHA256
    )
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == release.CONFIGMAP_NAME
    assert job["metadata"]["name"] == release.JOB_NAME
    assert len(release.JOB_NAME) <= 63
    assert len(release.CONFIGMAP_NAME) <= 63
    assert "hosted_glm_rank30_single_slot_v5" in configmap["data"]["run.sh"]
    assert "hosted_glm_rank30_single_slot_release_v4" in configmap["data"]["run.sh"]
    assert "runtime_gate_diagnostic" in configmap["data"]["release.py"]


def test_release_validator_requires_exact_v6_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = successor.build_plan(successor.CONTROLLER, ROOT)
    binding = successor.validate_runtime_gate_diagnostic(V6)
    binding["path"] = str(successor.DIAGNOSTIC_PATH)
    monkeypatch.setattr(
        successor, "validate_runtime_gate_diagnostic", lambda: copy.deepcopy(binding)
    )
    called = []
    monkeypatch.setattr(successor.prior, "validate_release", lambda *_args: called.append(1))
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": "2026-09-06T15:01:44Z",
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(plan),
        "source_package_sha256": "sha256:" + "1" * 64,
        "ledger_authority": release.prior.whole.LEDGER_AUTHORITY,
        "selection_authority": release.prior.whole.SELECTION_AUTHORITY,
        "live_rank29_peer": {},
        "endpoint_lease_observer": {},
        "fresh_collision_reconciliation": {},
        "runtime_gate_diagnostic": binding,
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    receipt = {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }
    successor.validate_release(receipt, plan, receipt["source_package_sha256"])
    assert called == [1]
    drifted = copy.deepcopy(receipt)
    drifted["runtime_gate_diagnostic"]["job_uid"] = (
        "11111111-1111-4111-8111-111111111111"
    )
    drifted["receipt_sha256"] = self_hosted.digest_without(
        drifted, "receipt_sha256"
    )
    with pytest.raises(RuntimeError, match="v6-gated release drifted"):
        successor.validate_release(
            drifted, plan, drifted["source_package_sha256"]
        )


def test_held_release_evidence_binds_packages_v6_and_exact_partition() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study"
        / "2026-09-06-glm53-hosted-rank30-scored-release-v4-held.json"
    )
    held = json.loads(path.read_text())
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["held_scored_package_sha256"] == scored_package.render(ROOT)[
        "package_sha256"
    ]
    assert held["scored_source_package_sha256"] == (
        scored_package.source_package_sha256(ROOT)
    )
    assert held["held_release_package_sha256"] == release_package.render(ROOT)[
        "package_sha256"
    ]
    assert held["v6_runtime_gate_diagnostic"]["receipt_sha256"] == (
        successor.DIAGNOSTIC_SELF_SHA256
    )
    assert held["campaign_reconciliation"]["glm_counts"] == {
        "accepted": 23,
        "active": 1,
        "blocked": 4,
        "total": 400,
        "unstarted": 372,
    }
    assert held["campaign_reconciliation"]["rank30_cells_unstarted"] == 4
    safe = held["campaign_reconciliation"]["safe_counts"]
    assert safe["claim_collisions"] == 0
    assert safe["accepted_collisions"] == 0
    assert safe["output_collisions"] == 0
    assert safe["target_object_collisions"] == 0
    assert held["launch_authorized"] is False
    assert held["scoring_authorized"] is False
