from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from evals.fleet import campaign_supervisor as supervisor

CAMPAIGN = Path("evals/fleet/configs/q38-glm53-primary-campaign-v1.json")
SCIENTIFIC_MAPPING = Path(
    "evals/fleet/configs/q38-glm53-primary-scientific-mapping-v2.json"
)
SCIENTIFIC_MAPPING_RELEASE_PREVIEW = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-04-primary-600-cell-executable-universe-import-release-preview-v1.json"
)
SUPERVISOR_MANIFEST = Path(
    "evals/fleet/cluster/q38-glm53-campaign-supervisor-v1.yaml"
)
WORKER_UID = "11111111-1111-4111-8111-111111111111"


def _released_fixture_campaign() -> dict:
    """Release only an isolated test fixture; the checked-in campaign stays blocked."""
    campaign = supervisor.read_object(CAMPAIGN)
    campaign["release_status"] = "released"
    campaign["launch_authorized"] = True
    campaign["campaign_sha256"] = supervisor.digest_without(campaign, "campaign_sha256")
    return campaign


def _ledger(tmp_path: Path) -> tuple[Path, dict, dict]:
    campaign = _released_fixture_campaign()
    root = tmp_path / "ledger"
    universe = supervisor.initialize_ledger(campaign, root, cluster=False)
    import_manifest = {
        "schema_version": supervisor.LEGACY_IMPORT_SCHEMA,
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "complete_inventory": True,
        "observed_cell_count": 0,
        "entries": [],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    import_manifest["import_sha256"] = supervisor.digest_without(
        import_manifest, "import_sha256"
    )
    supervisor.import_legacy_evidence(root, import_manifest)
    qwen = next(
        cell for cell in universe["cells"] if cell["component_id"] == "qwen-hosted-v8-primary49"
    )
    registration = {
        "schema_version": supervisor.WORKER_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "worker_name": "qwen-tail-a-v9",
        "worker_uid": WORKER_UID,
        "namespace": "fleet-train-jobs",
        "plan_sha256": "sha256:" + "a" * 64,
        "run_id_prefix": "chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-",
        "owned_partitions": [
            {
                "component_id": "qwen-hosted-v8-primary49",
                "source_ranks": list(range(11, 33)),
            }
        ],
        "priority_class": "fleet-train-high",
        "owned": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    registration["registration_sha256"] = supervisor.digest_without(
        registration, "registration_sha256"
    )
    supervisor.register_worker(root, registration)
    return root, universe, qwen


def test_frozen_campaign_is_exact_600_cell_disjoint_universe(tmp_path: Path) -> None:
    campaign = _released_fixture_campaign()
    universe = supervisor.initialize_ledger(campaign, tmp_path / "ledger", cluster=False)
    assert universe["cell_count"] == 600
    assert universe["model_counts"] == {"glm-5.3": 400, "qwen3.8-27b": 200}
    assert universe["component_counts"] == {
        "glm-dedicated-a-v5-primary27": 108,
        "glm-dedicated-b-v5-primary27": 108,
        "glm-hosted-v12-primary46": 184,
        "qwen-hosted-retained-source4": 4,
        "qwen-hosted-v8-primary49": 196,
    }
    identities = [
        (cell["model"], cell["task_version_id"], cell["attempt"]) for cell in universe["cells"]
    ]
    assert len(identities) == len(set(identities)) == 600


def test_checked_in_campaign_cannot_initialize_before_replacements_are_bound(
    tmp_path: Path,
) -> None:
    campaign = supervisor.read_object(CAMPAIGN)
    with pytest.raises(ValueError, match="blocked pending"):
        supervisor.initialize_ledger(campaign, tmp_path / "ledger", cluster=False)


def test_held_v2_scientific_mapping_is_exact_and_executable_but_not_released() -> None:
    mapping = supervisor.read_object(SCIENTIFIC_MAPPING)
    summary = supervisor.validate_scientific_mapping(mapping)
    assert summary == {
        "tasks": 150,
        "cells": 600,
        "model_cells": {"glm-5.3": 400, "qwen3.8-27b": 200},
        "unresolved_cells": 0,
        "unresolved_components": [],
        "executable_cells": 600,
        "executable_universe_sha256": (
            "sha256:17d1fa8734e5b3e16c902ea35ba7bcd0eacdb7f7f834f311b1f5c0777a7bb559"
        ),
    }
    assert mapping["launch_authorized"] is False
    assert mapping["ledger_initialization_authorized"] is False
    replacements = {
        (row["model"], row["excluded_source_rank"]): row["replacement_source_rank"]
        for row in mapping["replacement_mappings"]
    }
    assert replacements == {
        ("qwen3.8-27b", 1): 55,
        ("qwen3.8-27b", 2): 52,
        ("qwen3.8-27b", 3): 53,
        ("qwen3.8-27b", 5): 54,
        ("qwen3.8-27b", 10): 56,
        ("glm-5.3", 1): 101,
        ("glm-5.3", 2): 102,
        ("glm-5.3", 3): 103,
        ("glm-5.3", 4): 110,
        ("glm-5.3", 5): 104,
        ("glm-5.3", 6): 112,
        ("glm-5.3", 7): 105,
        ("glm-5.3", 9): 109,
        ("glm-5.3", 11): 108,
        ("glm-5.3", 14): 113,
        ("glm-5.3", 54): 107,
        ("glm-5.3", 56): 111,
    }


def test_blocked_v2_mapping_status_command_is_read_only(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "campaign_supervisor",
            "mapping-status",
            "--mapping",
            str(SCIENTIFIC_MAPPING),
        ],
    )
    assert supervisor.main() == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["cells"] == 600
    assert snapshot["unresolved_cells"] == 0
    assert snapshot["executable_cells"] == 600
    assert snapshot["launch_authorized"] is False
    assert snapshot["scores_read"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda mapping: mapping.__setitem__("launch_authorized", True),
            "must remain blocked",
        ),
        (
            lambda mapping: mapping["components"][3]["scientific_task_sources"][1][
                "tasks"
            ][0]["task"].__setitem__(
                "version_id", "21baee36-dc26-4dcd-983e-ae58a665dfa9"
            ),
            "attempt task identity",
        ),
        (
            lambda mapping: mapping["components"][3]["attempt_ownership"][-1][
                "source_ranks"
            ].pop(),
            "attempt ownership has gaps",
        ),
        (
            lambda mapping: mapping["components"][3]["attempt_ownership"][-1].__setitem__(
                "repo_plan_path", "evals/fleet/configs/does-not-exist.json"
            ),
            "attempt ownership plan missing",
        ),
        (
            lambda mapping: mapping["replacement_mappings"][0].__setitem__(
                "replacement_source_rank", 999
            ),
            "replacement mapping is invalid",
        ),
        (
            lambda mapping: mapping["legacy_import_semantics"].__setitem__(
                "active_claim", "reclaim"
            ),
            "import semantics drifted",
        ),
    ],
)
def test_blocked_v2_scientific_mapping_tampering_fails_closed(
    mutation, match: str
) -> None:
    mapping = supervisor.read_object(SCIENTIFIC_MAPPING)
    mutation(mapping)
    mapping["mapping_sha256"] = supervisor.digest_without(mapping, "mapping_sha256")
    with pytest.raises(ValueError, match=match):
        supervisor.validate_scientific_mapping(mapping)


def test_600_cell_executable_universe_import_release_is_held() -> None:
    mapping = supervisor.read_object(SCIENTIFIC_MAPPING)
    preview = supervisor.read_object(SCIENTIFIC_MAPPING_RELEASE_PREVIEW)
    summary = supervisor.validate_scientific_mapping_release_preview(preview, mapping)
    assert summary["executable_cells"] == 600
    assert summary["unresolved_cells"] == 0
    assert preview["status"] == "HELD"
    assert preview["authorization"] == {
        "launch_authorized": False,
        "ledger_initialization_authorized": False,
        "supervisor_deployment_authorized": False,
    }
    assert preview["legacy_import"]["complete"] is False


def test_600_cell_release_preview_rejects_fragment_or_authorization_drift() -> None:
    mapping = supervisor.read_object(SCIENTIFIC_MAPPING)
    preview = supervisor.read_object(SCIENTIFIC_MAPPING_RELEASE_PREVIEW)
    for mutation in (
        lambda row: row["execution_fragments"][0].__setitem__("cell_count", 3),
        lambda row: row["authorization"].__setitem__("launch_authorized", True),
    ):
        drifted = copy.deepcopy(preview)
        mutation(drifted)
        drifted["receipt_sha256"] = supervisor.digest_without(
            drifted, "receipt_sha256"
        )
        with pytest.raises(ValueError, match="release"):
            supervisor.validate_scientific_mapping_release_preview(drifted, mapping)


def test_campaign_rejects_capacity_or_plan_drift() -> None:
    campaign = _released_fixture_campaign()
    drifted = copy.deepcopy(campaign)
    drifted["expected"]["dedicated_gpus"] = 24
    drifted["campaign_sha256"] = supervisor.digest_without(drifted, "campaign_sha256")
    with pytest.raises(ValueError, match="denominator or dedicated capacity"):
        supervisor.validate_campaign(drifted)

    drifted = copy.deepcopy(campaign)
    drifted["components"][0]["plan_sha256"] = "sha256:" + "0" * 64
    drifted["campaign_sha256"] = supervisor.digest_without(drifted, "campaign_sha256")
    with pytest.raises(ValueError, match="plan digest binding"):
        supervisor.build_universe(drifted, cluster=False)


def test_execution_fragments_must_exactly_partition_component() -> None:
    campaign = _released_fixture_campaign()
    component = campaign["components"][1]
    ranks = component["source_ranks"]
    shared = {
        "repo_plan_path": component["repo_plan_path"],
        "cluster_plan_path": component["cluster_plan_path"],
        "plan_sha256": component["plan_sha256"],
    }
    component["execution_fragments"] = [
        {**shared, "source_ranks": ranks[:24]},
        {**shared, "source_ranks": ranks[24:]},
    ]
    campaign["campaign_sha256"] = supervisor.digest_without(
        campaign, "campaign_sha256"
    )
    assert supervisor.build_universe(campaign, cluster=False)["cell_count"] == 600

    component["execution_fragments"][1]["source_ranks"].append(ranks[0])
    campaign["campaign_sha256"] = supervisor.digest_without(
        campaign, "campaign_sha256"
    )
    with pytest.raises(ValueError, match="exactly partition"):
        supervisor.validate_campaign(campaign)


def test_execution_fragments_can_split_one_task_at_attempt_boundaries() -> None:
    campaign = _released_fixture_campaign()
    component = campaign["components"][1]
    ranks = component["source_ranks"]
    shared = {
        "repo_plan_path": component["repo_plan_path"],
        "cluster_plan_path": component["cluster_plan_path"],
        "plan_sha256": component["plan_sha256"],
    }
    component["execution_fragments"] = [
        {**shared, "cells": [{"source_rank": ranks[0], "attempts": [1, 2]}]},
        {**shared, "cells": [{"source_rank": ranks[0], "attempts": [3, 4]}]},
        {**shared, "source_ranks": ranks[1:]},
    ]
    campaign["campaign_sha256"] = supervisor.digest_without(
        campaign, "campaign_sha256"
    )
    universe = supervisor.build_universe(campaign, cluster=False)
    assert universe["cell_count"] == 600
    cells = [
        row
        for row in universe["cells"]
        if row["component_id"] == component["id"]
        and row["source_rank"] == ranks[0]
    ]
    assert [row["attempt"] for row in cells] == [1, 2, 3, 4]


def test_atomic_claim_allows_exactly_one_controller(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49"
        and row["source_rank"] == 11
        and row["attempt"] == 1
    )
    kwargs = {
        "cell_id": cell["cell_id"],
        "worker_name": "qwen-tail-a-v9",
        "worker_uid": WORKER_UID,
        "run_id": "chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(supervisor.claim_cell, root, **kwargs) for _ in range(2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)


def test_claim_is_blocked_until_authoritative_legacy_inventory_is_sealed(
    tmp_path: Path,
) -> None:
    campaign = _released_fixture_campaign()
    root = tmp_path / "ledger"
    universe = supervisor.initialize_ledger(campaign, root, cluster=False)
    cell = universe["cells"][0]
    with pytest.raises(RuntimeError, match="legacy inventory is not sealed"):
        supervisor.claim_cell(
            root,
            cell_id=cell["cell_id"],
            worker_name="absent",
            worker_uid=WORKER_UID,
            run_id=cell["source_run_id"],
        )


def test_legacy_import_is_exact_atomic_and_prevents_reclaim(tmp_path: Path) -> None:
    campaign = _released_fixture_campaign()
    root = tmp_path / "ledger"
    universe = supervisor.initialize_ledger(campaign, root, cluster=False)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49"
        and row["source_rank"] == 11
        and row["attempt"] == 1
    )
    evidence = {
        "schema_version": "fleet-hosted-opencode-attempt-accepted-v1",
        "accepted": True,
        "credited": True,
        "plan_sha256": cell["source_plan_sha256"],
        "run_id": cell["source_run_id"],
        "task_version_id": cell["task_version_id"],
        "source_rank": cell["source_rank"],
        "attempt": cell["attempt"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    evidence["receipt_sha256"] = supervisor.digest_without(
        evidence, "receipt_sha256"
    )
    evidence_path = tmp_path / "accepted.json"
    evidence_path.write_text(json.dumps(evidence))
    entry = {
        "cell_id": cell["cell_id"],
        "state": "accepted",
        "worker_name": "legacy-qwen-worker",
        "worker_uid": WORKER_UID,
        "component_id": cell["component_id"],
        "source_plan_sha256": cell["source_plan_sha256"],
        "run_id": cell["source_run_id"],
        "task_version_id": cell["task_version_id"],
        "source_rank": cell["source_rank"],
        "attempt": cell["attempt"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence["receipt_sha256"],
    }
    manifest = {
        "schema_version": supervisor.LEGACY_IMPORT_SCHEMA,
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "complete_inventory": True,
        "observed_cell_count": 1,
        "entries": [entry],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    manifest["import_sha256"] = supervisor.digest_without(
        manifest, "import_sha256"
    )
    supervisor.import_legacy_evidence(root, manifest)
    block = next(
        row
        for row in supervisor.status(root)["blocks"]
        if row["model"] == "qwen3.8-27b"
        and row["serving_block"] == "qwen-hosted-no-autocontinue"
    )
    assert block["accepted"] == 1
    with pytest.raises(RuntimeError, match="legacy inventory"):
        supervisor.claim_cell(
            root,
            cell_id=cell["cell_id"],
            worker_name="legacy-qwen-worker",
            worker_uid=WORKER_UID,
            run_id=cell["source_run_id"],
        )
    with pytest.raises(FileExistsError):
        supervisor.import_legacy_evidence(root, manifest)

    tampered = copy.deepcopy(manifest)
    tampered["entries"][0]["task_version_id"] = "wrong"
    tampered["import_sha256"] = supervisor.digest_without(tampered, "import_sha256")
    other_root = tmp_path / "other-ledger"
    supervisor.initialize_ledger(campaign, other_root, cluster=False)
    with pytest.raises(ValueError, match="cell binding drifted"):
        supervisor.import_legacy_evidence(other_root, tampered)


def test_imported_active_claim_can_terminalize_without_reclaim(tmp_path: Path) -> None:
    campaign = _released_fixture_campaign()
    root = tmp_path / "ledger"
    universe = supervisor.initialize_ledger(campaign, root, cluster=False)
    cell = universe["cells"][0]
    claim = {
        "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
        "plan_sha256": cell["source_plan_sha256"],
        "run_id": cell["source_run_id"],
        "source_rank": cell["source_rank"],
        "attempt": cell["attempt"],
    }
    claim["claim_sha256"] = supervisor.digest_without(claim, "claim_sha256")
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(claim))
    entry = {
        "cell_id": cell["cell_id"],
        "state": "claimed",
        "worker_name": "legacy-active-worker",
        "worker_uid": WORKER_UID,
        "component_id": cell["component_id"],
        "source_plan_sha256": cell["source_plan_sha256"],
        "run_id": cell["source_run_id"],
        "task_version_id": cell["task_version_id"],
        "source_rank": cell["source_rank"],
        "attempt": cell["attempt"],
        "evidence_path": str(claim_path),
        "evidence_sha256": claim["claim_sha256"],
    }
    manifest = {
        "schema_version": supervisor.LEGACY_IMPORT_SCHEMA,
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "complete_inventory": True,
        "observed_cell_count": 1,
        "entries": [entry],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    manifest["import_sha256"] = supervisor.digest_without(
        manifest, "import_sha256"
    )
    supervisor.import_legacy_evidence(root, manifest)

    accepted = {
        "accepted": True,
        "credited": True,
        "task_version_id": cell["task_version_id"],
        "attempt": cell["attempt"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    accepted["receipt_sha256"] = supervisor.digest_without(
        accepted, "receipt_sha256"
    )
    accepted_path = tmp_path / "accepted-after-import.json"
    accepted_path.write_text(json.dumps(accepted))
    supervisor.record_outcome(
        root,
        cell_id=cell["cell_id"],
        kind="accepted",
        evidence_path=accepted_path,
    )
    block = next(
        row
        for row in supervisor.status(root)["blocks"]
        if row["model"] == cell["model"]
        and row["serving_block"] == cell["serving_block"]
    )
    assert block["accepted"] == 1
    assert block["claimed"] == 0


def test_claim_rejects_foreign_partition_and_run_namespace(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    foreign = next(
        row for row in universe["cells"] if row["component_id"] == "glm-hosted-v12-primary46"
    )
    with pytest.raises(ValueError, match="partition does not own"):
        supervisor.claim_cell(
            root,
            cell_id=foreign["cell_id"],
            worker_name="qwen-tail-a-v9",
            worker_uid=WORKER_UID,
            run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-x",
        )
    owned = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49" and row["source_rank"] == 11
    )
    with pytest.raises(ValueError, match="run ID"):
        supervisor.claim_cell(
            root,
            cell_id=owned["cell_id"],
            worker_name="qwen-tail-a-v9",
            worker_uid=WORKER_UID,
            run_id="foreign-run",
        )


def test_outcomes_are_append_only_and_mutually_exclusive(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49"
        and row["source_rank"] == 11
        and row["attempt"] == 1
    )
    supervisor.claim_cell(
        root,
        cell_id=cell["cell_id"],
        worker_name="qwen-tail-a-v9",
        worker_uid=WORKER_UID,
        run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    )
    accepted = {
        "schema_version": "fleet-hosted-opencode-attempt-accepted-v1",
        "accepted": True,
        "credited": True,
        "task_version_id": cell["task_version_id"],
        "attempt": 1,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    accepted["receipt_sha256"] = supervisor.digest_without(accepted, "receipt_sha256")
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(accepted))
    supervisor.record_outcome(
        root, cell_id=cell["cell_id"], kind="accepted", evidence_path=accepted_path
    )
    quarantine = {
        "schema_version": "fleet-score-blind-quarantine-v1",
        "retry_allowed": False,
        "task_version_id": cell["task_version_id"],
        "attempt": 1,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    quarantine["receipt_sha256"] = supervisor.digest_without(quarantine, "receipt_sha256")
    quarantine_path = tmp_path / "quarantine.json"
    quarantine_path.write_text(json.dumps(quarantine))
    with pytest.raises(RuntimeError, match="contradictory terminal"):
        supervisor.record_outcome(
            root,
            cell_id=cell["cell_id"],
            kind="quarantine",
            evidence_path=quarantine_path,
        )
    snapshot = supervisor.status(root)
    qwen = next(row for row in snapshot["blocks"] if row["model"] == "qwen3.8-27b")
    assert qwen == {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-no-autocontinue",
        "accepted": 1,
        "quarantined": 0,
        "claimed": 0,
        "unclaimed": 199,
        "total": 200,
    }


def test_outcome_rejects_score_bearing_evidence(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49" and row["source_rank"] == 11
    )
    supervisor.claim_cell(
        root,
        cell_id=cell["cell_id"],
        worker_name="qwen-tail-a-v9",
        worker_uid=WORKER_UID,
        run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    )
    evidence = {
        "accepted": True,
        "score": 0.5,
        "scores_included": True,
        "prompts_or_traces_included": False,
    }
    evidence["receipt_sha256"] = supervisor.digest_without(evidence, "receipt_sha256")
    path = tmp_path / "score-bearing.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="score/content blind"):
        supervisor.record_outcome(
            root, cell_id=cell["cell_id"], kind="accepted", evidence_path=path
        )


def test_heartbeat_uses_exact_name_uid_and_high_priority(tmp_path: Path) -> None:
    root, _, _ = _ledger(tmp_path)

    class Kube:
        routes: list[str] = []

        def get(self, route: str):
            self.routes.append(route)
            return {
                "metadata": {"uid": WORKER_UID},
                "spec": {"template": {"spec": {"priorityClassName": "fleet-train-high"}}},
                "status": {"active": 1},
            }

    kube = Kube()
    receipt = supervisor.heartbeat_once(root, kube)  # type: ignore[arg-type]
    assert kube.routes == ["/apis/batch/v1/namespaces/fleet-train-jobs/jobs/qwen-tail-a-v9"]
    assert receipt["workers"][0]["state"] == "active"
    assert receipt["exact_named_gets_only"] is True
    assert receipt["peer_objects_mutated"] is False
    assert len(list((root / "heartbeats").glob("*.json"))) == 1


def test_heartbeat_fails_closed_on_uid_or_priority_drift(tmp_path: Path) -> None:
    root, _, _ = _ledger(tmp_path)

    class Kube:
        def __init__(self, uid: str, priority: str) -> None:
            self.uid = uid
            self.priority = priority

        def get(self, _route: str):
            return {
                "metadata": {"uid": self.uid},
                "spec": {"template": {"spec": {"priorityClassName": self.priority}}},
                "status": {"active": 1},
            }

    with pytest.raises(RuntimeError, match="UID drifted"):
        supervisor.heartbeat_once(
            root,
            Kube("22222222-2222-4222-8222-222222222222", "fleet-train-high"),  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeError, match="priority drifted"):
        supervisor.heartbeat_once(root, Kube(WORKER_UID, "fleet-infra-quiet"))  # type: ignore[arg-type]


def test_cluster_supervisor_is_read_only_high_priority_and_held() -> None:
    documents = list(yaml.safe_load_all(SUPERVISOR_MANIFEST.read_text()))
    assert [document["kind"] for document in documents] == [
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "Deployment",
    ]
    role = documents[1]
    assert role["rules"] == [
        {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["get"]}
    ]
    deployment = documents[3]
    assert deployment["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/launch-authorized"
    ] == "false"
    pod = deployment["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-train-high"
    command = pod["containers"][0]["args"][0]
    assert "campaign_supervisor status" in command
    assert "campaign_supervisor watch" in command
    assert "FLEET_API_KEY" not in SUPERVISOR_MANIFEST.read_text()
