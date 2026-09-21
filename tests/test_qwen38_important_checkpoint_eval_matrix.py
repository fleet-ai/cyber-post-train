from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/evidence/qwen38-important-checkpoint-eval-matrix-20260921.json"
CAMPAIGN_PATH = (
    ROOT / "configs/evaluation/qwen38-important-checkpoints-opencode-wbe-campaign-v1.json"
)
LORA_PLAN_PATH = ROOT / "configs/qualification/qwen38-lora-step60-zero-update-export-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> dict[str, dict]:
    return {row["artifact_id"]: row for row in _load(MATRIX_PATH)["rows"]}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def test_matrix_is_score_blind_point_in_time_and_self_digesting() -> None:
    matrix = _load(MATRIX_PATH)
    unsigned = dict(matrix)
    claimed = unsigned.pop("receipt_sha256")

    assert matrix["schema"] == "cyber_qwen38_important_checkpoint_eval_matrix_v1"
    assert matrix["source_main_commit"] == ("91ab058ebbf744c0931dbb9692831bdbd7ecefaf")
    assert claimed == _digest(unsigned)
    assert matrix["scope"]["external_mutations_by_this_audit"] == 0
    assert matrix["scope"]["accepted_complete_fleet_dev17_capability_results"] == 0
    assert matrix["scope"]["accepted_matched_web_capability_results"] == 0
    assert matrix["scope"]["fleet_final_test_status"] == "sealed_not_accessed"
    assert matrix["interpretation"]["complete_fleet_result_exists"] is False
    assert matrix["interpretation"]["matched_web_result_exists"] is False

    assert matrix["privacy"] == {
        "scores_included": False,
        "session_identifiers_included": False,
        "task_keys_or_versions_included": False,
        "prompts_traces_flags_answers_or_benchmark_content_included": False,
        "credentials_included": False,
        "private_logs_included": False,
    }
    forbidden_exact_keys = {
        "score",
        "session_id",
        "task_key",
        "task_version_id",
        "prompt",
        "trace",
        "flag",
        "answer",
    }
    assert _all_keys(matrix).isdisjoint(forbidden_exact_keys)


def test_matrix_binds_exact_protocols_and_evidence_files() -> None:
    matrix = _load(MATRIX_PATH)
    bindings = matrix["protocol_bindings"]

    for name in ("fleet_split", "fleet_seed43_protocol", "web_campaign"):
        binding = bindings[name]
        assert _file_sha256(ROOT / binding["path"]) == binding["file_sha256"]

    campaign = _load(CAMPAIGN_PATH)
    assert bindings["web_campaign"]["plan_sha256"] == campaign["sha256"]
    assert campaign["execution"]["launchable_now"] is False
    assert campaign["status"]["state"] == "prepared_not_launchable"

    for source in matrix["evidence_sources"]:
        assert _file_sha256(ROOT / source["path"]) == source["file_sha256"]


def test_exact_six_checkpoint_roster_and_durable_artifact_bindings() -> None:
    matrix = _load(MATRIX_PATH)
    rows = matrix["rows"]
    assert [row["artifact_id"] for row in rows] == [
        "qwen38-27b-base-matched-control",
        "q38-fresh75-step230",
        "q38-teacher-dense-v5-step186",
        "q38-self-sft-step44",
        "q38-available-a-lr30-step76",
        "q38-lora-step60",
    ]
    assert [row["priority"] for row in rows] == list(range(6))
    assert matrix["scope"]["artifact_ids"] == [row["artifact_id"] for row in rows]

    by_id = _rows()
    expected = {
        "q38-fresh75-step230": (
            230,
            "sha256:bf92e29d9f19dd589201214f25fe8d0e3b0f65de14ec08fc573217f50580a486",
            "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029",
        ),
        "q38-teacher-dense-v5-step186": (
            186,
            "sha256:1b671c6ec4b39d1141e0d28c530243624f6a97a7922f9f222ffcbe20b8434d3e",
            "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
        ),
        "q38-self-sft-step44": (
            44,
            "sha256:0094cfeb62f120b55d9e2f7fdfd4217c4c0aab66f369e3c52c89fc5e7ccfff9b",
            "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
        ),
        "q38-available-a-lr30-step76": (
            76,
            "sha256:cc4e2997e120a80bb1e209dc027f18e12ce1ebe12639594d28e23b08c459cf1a",
            "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
        ),
    }
    for artifact_id, (step, manifest, revision) in expected.items():
        artifact = by_id[artifact_id]["artifact"]
        assert artifact["optimizer_step"] == step
        assert artifact["checkpoint_manifest_file_sha256"] == manifest
        assert artifact["export_payload_manifest_sha256"] == revision
        assert by_id[artifact_id]["serving"]["model_revision"] == revision

    base = by_id["qwen38-27b-base-matched-control"]["artifact"]
    assert base["repository"] == "Qwen/Qwen3.8-27B"
    assert base["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert base["weights_manifest_sha256"] == (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    )

    lora = by_id["q38-lora-step60"]
    assert lora["artifact"]["optimizer_step"] == 60
    assert lora["artifact"]["gpu_reload_verified"] is False
    assert lora["artifact"]["export_path"] is None
    assert lora["serving"]["phase"] == "absent"
    assert _file_sha256(LORA_PLAN_PATH) == lora["promotion"]["plan_file_sha256"]


def test_web_campaign_identities_match_and_all_launches_remain_closed() -> None:
    matrix = _load(MATRIX_PATH)
    rows = _rows()
    campaign = _load(CAMPAIGN_PATH)

    assert matrix["web_global_gate"]["status"] == "closed"
    assert matrix["web_global_gate"]["retired_v23"] == {
        "accepted_rollouts": 0,
        "failure_preserved_and_released": 60,
        "collection_export_failures": 45,
        "collection_run_failures": 15,
        "capability_result": False,
    }
    assert matrix["web_global_gate"]["causal_repair"]["pull_request"].endswith("/pull/278")
    assert len(matrix["web_global_gate"]["required_before_any_paid_launch"]) == 5

    shared_base = campaign["shared_base"]
    base = rows[shared_base["artifact_id"]]
    assert base["serving"]["served_id"] == shared_base["served_model"]
    assert base["web"]["canary_campaign_id"] == shared_base["campaigns"]["canary"]
    assert base["web"]["full_campaign_ids"] == shared_base["campaigns"]["full_replicas"]

    for candidate in campaign["candidates"]:
        row = rows[candidate["artifact_id"]]
        assert row["web"]["canary_campaign_id"] == candidate["campaigns"]["canary"]
        assert row["web"]["full_campaign_ids"] == candidate["campaigns"]["full_replicas"]
        assert row["web"]["launched"] is False
        if candidate["served_model"] is None:
            assert row["serving"]["served_id"] is None
        else:
            assert row["serving"]["served_id"] == candidate["served_model"]
            assert row["serving"]["model_revision"] == candidate["model_revision"]


def test_fleet_statuses_are_uid_bound_and_make_no_capability_claim() -> None:
    rows = _rows()
    expected = {
        "qwen38-27b-base-matched-control": (
            "6dc5320f-02f5-470d-ae83-a5dec0fe2a62",
            "43f91553-77ac-439d-b402-e9dafc805ff2",
            "ed4aff2f-d1e5-4273-9ba7-1e67ca88bf1e",
        ),
        "q38-fresh75-step230": (
            "d8fd2e3f-db6a-4ec5-89f3-fec695668486",
            "b19b47bd-81a7-4ced-aac1-fdcbed2d6053",
            "9f3af19f-0896-47dc-a032-5372e42df8f4",
        ),
        "q38-self-sft-step44": (
            "3d514935-1bbf-4ee7-97c3-22a34656c659",
            "ad6c0173-aa2a-4820-ba21-9a0d6e5214de",
            "daccf181-2f11-4c2a-a3bc-cb58d0b0655d",
        ),
        "q38-available-a-lr30-step76": (
            "78b2d32d-8324-4b51-98a4-0b14fbde8fbe",
            "4c65cd50-c2ae-47d2-a0a6-91c52b456527",
            "5dcdd828-7b62-42f2-81a7-3b01e996939a",
        ),
    }
    for artifact_id, (job_uid, workload_uid, pod_uid) in expected.items():
        fleet = rows[artifact_id]["fleet_dev17"]
        assert (fleet["job_uid"], fleet["workload_uid"], fleet["pod_uid"]) == (
            job_uid,
            workload_uid,
            pod_uid,
        )
        assert fleet["complete_capability_result"] is False

    teacher = rows["q38-teacher-dense-v5-step186"]["fleet_dev17"]
    assert teacher["seed43"]["job_uid"] == ("6303b84c-63f5-472b-b7aa-f785e3d56932")
    assert teacher["seed42_descriptive"]["job_uid"] == ("a929fb5d-03f8-404f-9620-f36d3c25926f")
    assert teacher["seed42_descriptive"]["workload_uid"] == ("2da0cdc7-d742-4f30-b2f4-ffb0f2f2eba5")
    assert teacher["seed43"]["complete_capability_result"] is False
    assert teacher["seed42_descriptive"]["complete_capability_result"] is False
    assert "Never splice" in teacher["cross_seed_rule"]

    lora = rows["q38-lora-step60"]["fleet_dev17"]
    assert lora == {
        "status": "not_started_checkpoint_not_yet_promoted_for_serving",
        "complete_capability_result": False,
    }
