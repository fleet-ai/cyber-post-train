from pathlib import Path

from evals.fleet import glm53_dedicated_v22_rank100_held_v1 as held

ROOT = Path(__file__).resolve().parents[1]


def test_rank100_successor_is_whole_task_and_fail_closed() -> None:
    value = held.render(ROOT)
    assert value["status"] == "READY_HELD"
    assert value["launch_authorized"] is False
    assert value["scoring_create_permitted"] is False
    assert value["attempts"] == [1, 2, 3, 4]
    assert value["cell_ids"] == held.EXPECTED_CELL_IDS
    assert len(set(value["execution_ids"])) == 4
    assert value["launch_time_gates"] == {
        "fresh_global_ledger_required": True,
        "all_four_cells_unstarted_required": True,
        "authoritative_session_collisions_required": 0,
        "kubernetes_object_collisions_required": 0,
        "sfs_output_collisions_required": 0,
        "canonical_claim_collisions_required": 0,
        "server_uid_and_workload_history_revalidation_required": True,
        "atomic_four_claim_reservation_before_first_model_call_required": True,
        "rollback_all_four_claims_on_partial_pre_model_reservation_failure": True,
        "authoritative_acceptance_before_each_next_attempt": True,
    }
