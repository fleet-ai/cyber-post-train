"""Exact provenance successors, bounded dev authority, and no production opening."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training.io import digest_json, file_sha256
from training.sft import compile_sft, job_request
from training.sft_runtime import optimizer_schedule

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v5.json"
V6 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v6.json"


def read(path):
    return json.loads(path.read_bytes())


def sealed(path):
    value = read(path)
    assert value["sha256"] == digest_json({k: v for k, v in value.items() if k != "sha256"})
    return value


def test_v6_records_conversation_scope_not_a_signed_receipt_or_production_authority():
    old, value = sealed(V5), sealed(V6)
    assert value["supersedes"]["file_sha256"] == file_sha256(V5)
    assert value["supersedes"]["embedded_sha256"] == old["sha256"]
    assert value["authorization"]["evidence_type"] == (
        "conversation_and_active_goal_not_a_cryptographic_user_receipt"
    )
    assert value["authorization"]["thread_id"] == "01a08909-b4fa-7fb1-b207-68f0b494e78d"
    assert value["authorization"]["production_authorized"] is False
    assert value["authorization"]["peer_mutations_authorized"] is False
    assert value["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "conditional_dev_scope_recorded": True,
        "all_live_gates_verified": False,
        "cluster_or_api_mutations_performed": False,
        "production_submission_authorized": False,
    }
    assert value["production"]["state"] == "blocked"
    assert value["production"]["authorized_by_this_file"] is False
    assert value["production"]["first_lr_1e_5_cell_must_not_be_replayed"] is True
    assert (
        value["acceptance"]["per_lr_receipts"]
        == old["remaining_lr_qualification"]["accepted_receipts"]
    )
    assert value["inherited_cadence_gate"]["source_zero_restarts_verified"] is False
    assert value["inherited_cadence_gate"]["source_pod_provenance_gap_preserved"] is True
    assert [wave["max_nodes"] for wave in value["dev_waves"]] == [2, 1]


def test_current_available_a_provenance_is_retained_in_the_new_protocol_manifest():
    old, value = sealed(V5), sealed(V6)
    binding = value["corpus_successor"]
    proof = sealed(ROOT / binding["receipt_path"])
    assert binding["receipt_sha256"] == proof["sha256"]
    assert (
        binding["source_manifest_sha256"] == old["first_production_cell"]["data"]["manifest_sha256"]
    )
    assert binding["source_manifest_sha256"] == proof["source_manifest_sha256"]
    assert binding["manifest_sha256"] == proof["successor_manifest_sha256"]
    assert binding["manifest_file_sha256"] == proof["successor_manifest_file_sha256"]
    assert binding["staged_receipt_sha256"] is None
    assert proof["changed_manifest_fields"] == ["fleet_dev_protocol_sha256", "sha256"]
    assert proof["launch_authorized"] is False
    assert proof["parquet_or_source_records_read"] is False
    assert proof["remote_operations_performed"] is False
    for key in ("source_selection_sha256", "target_policy_sha256", "train_file_sha256"):
        assert binding[key] == proof["preserved"][key]
    assert (
        binding["source_selection_sha256"]
        == old["first_production_cell"]["data"]["source_selection_sha256"]
    )
    assert binding["rows"] == proof["preserved"]["train_rows"] == 602
    assert binding["supervised_tokens"] == proof["preserved"]["supervised_tokens"] == 700359
    assert (
        binding["protocol_sha256"]
        == old["opening_gates"]["split_a_base_control"]["outcome_protocol_sha256"]
    )


def test_new_dev_configs_bind_current_manifest_and_preserve_the_scientific_recipe(tmp_path):
    study = sealed(V6)
    expected = {"lr1": (1e-6, 6, 6), "lr30": (3e-5, 6, 6), "lr100": (1e-4, 21, 20)}
    names = set()
    for cell in (cell for wave in study["dev_waves"] for cell in wave["cells"]):
        path = ROOT / cell["config_path"]
        config = read(path)
        assert cell["config_file_sha256"] == file_sha256(path)
        old = read(path.with_name(path.name.replace("-v2.json", "-v1.json")))
        assert config["recipe"] == old["recipe"]
        assert config["cluster"] == old["cluster"]
        assert config["model"] == old["model"]
        assert config["data"]["root"] == old["data"]["root"]
        assert config["data"]["manifest"] != old["data"]["manifest"]
        assert config["data"]["manifest"] == study["corpus_successor"]["staged_manifest_path"]
        assert config["data"]["manifest_sha256"] == study["corpus_successor"]["manifest_sha256"]
        assert config["name"] == cell["name"] == config["wandb"]["run_id"]
        assert config["name"] == config["wandb"]["name"] != old["name"]
        assert config["output_root"] == cell["output_root"] != old["output_root"]
        assert config["name"] not in names
        names.add(config["name"])
        label = next(key for key in expected if f"-{key}-" in config["name"])
        lr, pause, checkpoint = expected[label]
        assert (
            config["recipe"]["lr"],
            config["pause_after_step"],
            config["recipe"]["checkpoint_interval"],
        ) == (lr, pause, checkpoint)

        # Synthetic metadata only: substitute its digest as well as its path.
        # The exact production corpus binding is checked above; no private data
        # is needed to exercise the compiler/rendered request boundary in CI.
        manifest = {
            "tokenizer": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            },
            "split_sha256": "sha256:" + "a" * 64,
            "validation_mode": "task_outcomes_only",
            "fleet_dev_protocol_sha256": config["fleet_dev_protocol_sha256"],
            "files": {
                "train": {
                    "path": "train.parquet",
                    "rows": 602,
                    "sha256": "b" * 64,
                    "task_keys": ["synthetic"],
                }
            },
        }
        manifest["sha256"] = "sha256:" + digest(manifest)
        source = tmp_path / f"{label}.json"
        source.write_text(json.dumps(manifest))
        config["data"]["manifest"] = str(source)
        config["data"]["manifest_sha256"] = manifest["sha256"]
        plan = compile_sft(config, relative_to=path.parent)
        assert plan["recipe"]["max_steps"] == 76
        assert optimizer_schedule(plan["recipe"])["num_warmup_steps"] == 4
        assert plan["pause_after_step"] == pause
        assert plan["recipe"]["eval_interval"] == 0
        request = job_request(plan)
        assert request["workers"] == 1 and request["gpus_per_worker"] == 8
        assert request["priority_class"] == "c1"
        assert request["requeueIfPreempted"] is False
        assert request["secrets"] == ["wandb-api"]
    assert len(names) == 3
