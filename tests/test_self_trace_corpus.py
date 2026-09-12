import copy
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evals.fleet import opencode_self_hosted as fleet
from training import rl_data, sft
from training import self_trace_corpus as self_trace
from training.io import digest_json, file_sha256

TASK_VERSION = "11111111-1111-4111-8111-111111111111"
VERIFIER_EXECUTION = "22222222-2222-4222-8222-222222222222"
INSTANCE_ID = "synthetic-instance-001"
EVIDENCE_RUN_ID = "44444444-4444-4444-8444-444444444444"
VERIFIER_ID = "55555555-5555-4555-8555-555555555555"
VERIFIER_VERSION = "66666666-6666-4666-8666-666666666666"
ENVIRONMENT_VERSION = "77777777-7777-4777-8777-777777777777"
REVISION = "1" * 40
TEMPLATE = "fixture-chat-template"
TASK_PROMPT = "private task"
TOOL_CATALOG_SHA256 = "sha256:" + "8" * 64


class Tokenizer:
    chat_template = TEMPLATE

    def __len__(self):
        return 1024


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def seal(value: dict) -> dict:
    return {**value, "sha256": digest_json(value)}


def fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "sources"
    episode_id = "base-episode-001"
    episode = source / "episodes" / episode_id
    episode.mkdir(parents=True)
    binding = {
        "run_id": episode_id,
        "task": {
            "key": "train-task",
            "version_id": TASK_VERSION,
            "prompt_sha256": fleet.sha256(TASK_PROMPT.encode()),
            "env_variables_sha256": "sha256:" + "9" * 64,
            "output_json_schema_sha256": "sha256:" + "a" * 64,
            "cyber_contract": copy.deepcopy(rl_data.AUTHORITY["required_cyber_contract"]),
        },
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": REVISION,
            "root": str(tmp_path / "tokenizer"),
            "runtime_chat_template_sha256": fleet.sha256(TEMPLATE.encode()),
        },
        "environment": {
            "id": "synthetic-env",
            "version": "v1",
            "version_id": ENVIRONMENT_VERSION,
            "data_id": "synthetic-data",
            "data_version": "v1",
            "runtime_seed_content_sha256": "sha256:" + "b" * 64,
            "ttl_seconds": 3600,
        },
        "verifier": {
            "id": VERIFIER_ID,
            "version_id": VERIFIER_VERSION,
            "version": "v1",
            "sha256": "sha256:" + "c" * 64,
            "function_name": "verify",
        },
        "authority": copy.deepcopy(rl_data.AUTHORITY),
        "execution": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
        },
        "rl": {
            "max_turns": 4,
            "episode_seconds": 300,
            "tool_seconds": 60,
            "tool_result_chars": 10000,
            "max_tokens_per_turn": 8,
            "context_tokens": 32,
        },
        "initial_prompt_sha256": "sha256:" + "d" * 64,
        "initial_prompt_tokens_sha256": fleet.sha256(fleet.canonical_json([1, 2, 3])),
    }
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    payloads = {
        "binding.json": binding,
        "instance.json": {
            "instance_id": INSTANCE_ID,
            "evidence_run_id": EVIDENCE_RUN_ID,
        },
        "conversation.json": {
            "messages": [
                {"role": "user", "content": TASK_PROMPT},
                {"role": "assistant", "content": "private first action"},
                {"role": "tool", "name": "bash", "content": "private observation"},
                {"role": "assistant", "content": "private report"},
                {"role": "tool", "name": "submit_report", "content": "private result"},
            ]
        },
        "reward.json": {
            "task_key": "train-task",
            "task_version_id": TASK_VERSION,
            "instance_id": INSTANCE_ID,
            "reward": 1.0,
            "verifier_execution_id": VERIFIER_EXECUTION,
            "direct_authority_attestation": {
                "schema_version": fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA,
                "context": {
                    "task_key": "train-task",
                    "task_version_id": TASK_VERSION,
                    "instance_id": INSTANCE_ID,
                    "evidence_run_id": EVIDENCE_RUN_ID,
                    "verifier_version_id": VERIFIER_VERSION,
                    "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
                },
                "activity": {
                    "result_schema_version": "cyber_verification_result_v3",
                    "reward": 1.0,
                    "task_version_id": TASK_VERSION,
                    "verifier_execution_id": VERIFIER_EXECUTION,
                },
                "shadow": {
                    "mode": "authoritative",
                    "status": "authoritative",
                    "match": True,
                    "production_execution_id": VERIFIER_EXECUTION,
                    "direct_verifier": {
                        "status": "authoritative",
                        "match": True,
                        "execution_id": VERIFIER_EXECUTION,
                        "verifier_contract_version": rl_data.AUTHORITY["required_cyber_contract"][
                            "verifier_contract"
                        ],
                        "context_schema_version": "cyber_verification_context_v1",
                    },
                },
                "data_minimization": {
                    "components_included": False,
                    "diagnostics_included": False,
                    "evidence_payloads_included": False,
                    "prompts_included": False,
                    "traces_included": False,
                    "flags_included": False,
                },
            },
        },
        "cleanup.json": {
            "create_attempted": True,
            "instance_created": True,
            "instance_id": INSTANCE_ID,
            "instance_closed": True,
            "possible_instance_leak": False,
        },
        "recording.json": {
            "samples": [
                {
                    "tokens": [1, 2, 3, 10, 11, 20, 21, 30, 40],
                    "response_length": 6,
                    "loss_mask": [1, 1, 0, 0, 1, 0],
                    "rollout_log_probs": [-0.1, -0.2, 0.0, 0.0, -0.3, 0.0],
                }
            ]
        },
    }
    for name, value in payloads.items():
        write(episode / name, value)
    receipt = {
        "task_version_id": TASK_VERSION,
        "instance_id": INSTANCE_ID,
        "verifier_execution_id": VERIFIER_EXECUTION,
        "done_reason": "report_submitted",
        "config_sha256": binding["config_sha256"],
        "sample_count": 1,
        "files": {name: fleet.sha256((episode / name).read_bytes()) for name in payloads},
    }
    receipt["sha256"] = fleet.sha256(fleet.canonical_json(receipt))
    write(episode / "ACCEPTED.json", receipt)

    lock = {
        "schema": "huggingface_model_lock_v1",
        "repo": "Qwen/Qwen3.8-27B",
        "revision": REVISION,
        "weights": {"manifest_sha256": digest_json([]), "shards": 0},
        "tokenizer": {
            "manifest_sha256": digest_json([]),
            "files": [
                {
                    "path": "chat_template.jinja",
                    "sha256": fleet.sha256(TEMPLATE.encode()),
                }
            ],
        },
        "configuration": {},
    }
    weights = {
        "schema": "huggingface_weight_manifest_v1",
        "repo": "Qwen/Qwen3.8-27B",
        "revision": REVISION,
        "files": [],
    }
    split = seal(
        {
            "schema": "cyber_task_split_v2",
            "tasks": [
                {
                    "task_key": "train-task",
                    "task_version_id": TASK_VERSION,
                    "split": "train",
                }
            ],
        }
    )
    lock_path, weights_path, split_path = (
        tmp_path / "model.lock.json",
        tmp_path / "weights.json",
        tmp_path / "split.json",
    )
    write(lock_path, lock)
    write(weights_path, weights)
    write(split_path, split)
    index = seal(
        {
            "schema": self_trace.INDEX_SCHEMA,
            "review_receipt_sha256": "sha256:" + "a" * 64,
            "producer_plan_sha256": "sha256:" + "b" * 64,
            "route_certificate_sha256": "sha256:" + "e" * 64,
            "runtime_image": "registry.invalid/skyrl@sha256:" + "c" * 64,
            "model": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": REVISION,
                "root": str(tmp_path / "tokenizer"),
                "model_lock_file_sha256": file_sha256(lock_path),
                "model_weights_file_sha256": file_sha256(weights_path),
                "initialization": "exact_fresh_base",
            },
            "interface": {
                "backend": "skyrl_direct",
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
                "compaction": "disabled",
                "runtime_chat_template_sha256": fleet.sha256(TEMPLATE.encode()),
                "recorder_source_sha256": file_sha256(
                    Path(self_trace.__file__).with_name("skyrl_episode.py")
                ),
            },
            "episodes": [
                {
                    "episode_id": episode_id,
                    "directory": f"episodes/{episode_id}",
                    "task_key": "train-task",
                    "task_version_id": TASK_VERSION,
                    "accepted_sha256": file_sha256(episode / "ACCEPTED.json"),
                }
            ],
        }
    )
    index_path = tmp_path / "index.json"
    write(index_path, index)
    config = {
        "schema": self_trace.CONFIG_SCHEMA,
        "source_root": str(source),
        "source_index": {"path": str(index_path), "sha256": file_sha256(index_path)},
        "split": {"path": str(split_path), "sha256": file_sha256(split_path)},
        "model_lock": {"path": str(lock_path), "sha256": file_sha256(lock_path)},
        "model_weights": {"path": str(weights_path), "sha256": file_sha256(weights_path)},
        "tokenizer_root": str(tmp_path / "tokenizer"),
        "max_length": 32,
        "context_tokens": 8,
        "max_episodes_per_task_version": 3,
        "max_supervised_tokens_per_task_version": 49152,
        "max_submit_token_share": 0.5,
        "max_submit_response_share": 0.5,
        "source_seed": "self-trace-fixture-v1",
        "fleet_dev_protocol_sha256": "sha256:" + "d" * 64,
        "output": str(tmp_path / "corpus"),
    }
    monkeypatch.setattr(
        self_trace,
        "local_tokenizer",
        lambda model_lock, root: (
            Tokenizer(),
            {
                "repo": model_lock["repo"],
                "revision": model_lock["revision"],
                "chat_template_sha256": fleet.sha256(TEMPLATE.encode()),
            },
        ),
    )
    return config, index_path, episode


def test_direct_recording_compiles_every_assistant_once_and_masks_observations(
    tmp_path, monkeypatch
):
    config, _index_path, _episode = fixture(tmp_path, monkeypatch)
    result = self_trace.build(config, relative_to=tmp_path)
    assert result == {
        "manifest_sha256": json.loads((tmp_path / "corpus/manifest.json").read_text())["sha256"],
        "source_sessions": 1,
        "rows": 1,
        "assistant_responses": 2,
        "supervised_tokens": 3,
        "task_versions": 1,
        "submitted_jobs": 0,
    }
    rows = pq.read_table(tmp_path / "corpus/train.parquet").to_pylist()
    assert [s["assistant_index"] for row in rows for s in row["target_spans"]] == [0, 1]
    assert sum(rows[0]["loss_mask"]) == 3
    assert rows[0]["loss_mask"][5:7] == [0, 0]
    manifest = json.loads((tmp_path / "corpus/manifest.json").read_text())
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["source_selection"]["terminal_submission_response_share"] == 0.5
    assert manifest["source_selection"]["terminal_submission_token_share"] == pytest.approx(1 / 3)
    assert manifest["source_selection"]["required_task_tool_catalog_sha256"] == TOOL_CATALOG_SHA256
    assert os.stat(tmp_path / "corpus/train.parquet").st_mode & 0o777 == 0o600
    with pytest.raises(self_trace.SelfTraceError, match="new exact absolute"):
        self_trace.build(config, relative_to=tmp_path)


def test_resulting_manifest_compiles_as_outcome_only_dense_sft(tmp_path, monkeypatch):
    config, _index_path, _episode = fixture(tmp_path, monkeypatch)
    self_trace.build(config, relative_to=tmp_path)
    monkeypatch.setattr(
        sft,
        "bound_model",
        lambda lock, weights, root: {
            "repo": lock["repo"],
            "revision": lock["revision"],
            "root": root,
            "files": [
                {"path": "config.json", "sha256": "sha256:" + "1" * 64},
                {"path": "tokenizer_config.json", "sha256": "sha256:" + "2" * 64},
            ],
        },
    )
    training = {
        "name": "synthetic-self-trace-dev",
        "output_root": "/mnt/sfs/jobs/synthetic-self-trace-dev",
        "model": {
            "lock": str(tmp_path / "model.lock.json"),
            "weights": str(tmp_path / "weights.json"),
            "root": "/mnt/sfs/models/synthetic-qwen",
        },
        "data": {
            "manifest": str(tmp_path / "corpus/manifest.json"),
            "root": "/mnt/sfs/jobs/synthetic-self-corpus",
        },
        "recipe": {
            "epochs": 1,
            "batch_size": 1,
            "microbatch_per_gpu": 1,
            "nodes": 1,
            "gpus_per_node": 1,
            "lr": 1e-5,
            "max_length": 32,
            "eval_interval": 0,
            "checkpoint_interval": 1,
            "keep_checkpoints": 2,
            "seed": 42,
        },
        "wandb": {
            "entity": "synthetic",
            "project": "synthetic",
            "group": "synthetic",
            "run_id": "synthetic-self-trace-dev",
            "name": "synthetic-self-trace-dev",
        },
        "cluster": {"priority": "c1"},
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": config["fleet_dev_protocol_sha256"],
    }
    plan = sft.compile_sft(training, relative_to=tmp_path)
    assert plan["schema"] == "cyber_sft_runtime_dense_v1"
    assert plan["validation_mode"] == "task_outcomes_only"
    assert plan["recipe"]["max_steps"] == 1
    assert set(plan["datasets"]) == {"train"}
    assert plan["execution"]["priority"] == "c1"


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("heldout", "outside the training split"),
        ("zero_reward", "verified positive"),
        ("bad_attestation", "verified positive"),
        ("leak", "verified positive"),
        ("opencode", "exact direct fresh-base"),
        ("bad_log_probs", "token inventory"),
        ("tamper", "payload digest"),
        ("final_only", "non-submission"),
        ("symlink", "symlink"),
    ],
)
def test_unqualified_or_terminal_only_sources_fail_closed(tmp_path, monkeypatch, defect, reason):
    config, index_path, episode = fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    if defect == "heldout":
        index["episodes"][0]["task_key"] = "heldout-task"
        index["sha256"] = digest_json({k: v for k, v in index.items() if k != "sha256"})
        write(index_path, index)
        config["source_index"]["sha256"] = file_sha256(index_path)
    elif defect == "zero_reward":
        reward = json.loads((episode / "reward.json").read_text())
        reward["reward"] = 0.0
        write(episode / "reward.json", reward)
        _rebind_acceptance(episode, index_path, config)
    elif defect == "bad_attestation":
        reward = json.loads((episode / "reward.json").read_text())
        reward["direct_authority_attestation"]["context"]["task_version_id"] = (
            "44444444-4444-4444-8444-444444444444"
        )
        write(episode / "reward.json", reward)
        _rebind_acceptance(episode, index_path, config)
    elif defect == "leak":
        cleanup = json.loads((episode / "cleanup.json").read_text())
        cleanup["instance_closed"] = False
        cleanup["possible_instance_leak"] = True
        write(episode / "cleanup.json", cleanup)
        _rebind_acceptance(episode, index_path, config)
    elif defect == "opencode":
        binding = json.loads((episode / "binding.json").read_text())
        binding.pop("initial_prompt_tokens_sha256")
        binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
        write(episode / "binding.json", binding)
        _rebind_acceptance(episode, index_path, config, config_sha=binding["config_sha256"])
    elif defect == "bad_log_probs":
        recording = json.loads((episode / "recording.json").read_text())
        recording["samples"][0]["rollout_log_probs"][2] = -0.1
        write(episode / "recording.json", recording)
        _rebind_acceptance(episode, index_path, config)
    elif defect == "tamper":
        reward = json.loads((episode / "reward.json").read_text())
        reward["private"] = "changed"
        write(episode / "reward.json", reward)
    elif defect == "final_only":
        recording = json.loads((episode / "recording.json").read_text())
        recording["samples"][0].update(
            tokens=[1, 2, 3, 30, 40],
            response_length=2,
            loss_mask=[1, 0],
            rollout_log_probs=[-0.1, 0.0],
        )
        conversation = json.loads((episode / "conversation.json").read_text())
        conversation["messages"] = [conversation["messages"][0], *conversation["messages"][-2:]]
        write(episode / "recording.json", recording)
        write(episode / "conversation.json", conversation)
        _rebind_acceptance(episode, index_path, config)
    else:
        original = episode.rename(episode.with_name("real-episode"))
        episode.symlink_to(original, target_is_directory=True)
    with pytest.raises(self_trace.SelfTraceError, match=reason):
        self_trace.build(config, relative_to=tmp_path)


def _rebind_acceptance(
    episode: Path, index_path: Path, config: dict, *, config_sha: str | None = None
) -> None:
    receipt = json.loads((episode / "ACCEPTED.json").read_text())
    for name in receipt["files"]:
        receipt["files"][name] = fleet.sha256((episode / name).read_bytes())
    if config_sha is not None:
        receipt["config_sha256"] = config_sha
    receipt["sha256"] = fleet.sha256(
        fleet.canonical_json({k: v for k, v in receipt.items() if k != "sha256"})
    )
    write(episode / "ACCEPTED.json", receipt)
    index = json.loads(index_path.read_text())
    index["episodes"][0]["accepted_sha256"] = file_sha256(episode / "ACCEPTED.json")
    index["sha256"] = digest_json({k: v for k, v in index.items() if k != "sha256"})
    write(index_path, index)
    config["source_index"]["sha256"] = file_sha256(index_path)


@pytest.mark.parametrize(
    "identity",
    ["instance_id", "evidence_run_id", "verifier_id", "verifier_version_id", "execution_id"],
)
def test_missing_or_noncanonical_execution_identities_fail_closed(tmp_path, monkeypatch, identity):
    config, index_path, episode = fixture(tmp_path, monkeypatch)
    binding = json.loads((episode / "binding.json").read_text())
    instance = json.loads((episode / "instance.json").read_text())
    reward = json.loads((episode / "reward.json").read_text())
    cleanup = json.loads((episode / "cleanup.json").read_text())
    accepted = json.loads((episode / "ACCEPTED.json").read_text())
    attestation = reward["direct_authority_attestation"]
    if identity == "instance_id":
        instance["instance_id"] = None
        reward["instance_id"] = None
        attestation["context"]["instance_id"] = None
        cleanup["instance_id"] = None
        accepted["instance_id"] = None
    elif identity == "evidence_run_id":
        instance["evidence_run_id"] = None
        attestation["context"]["evidence_run_id"] = None
    elif identity == "verifier_id":
        binding["verifier"]["id"] = None
    elif identity == "verifier_version_id":
        binding["verifier"]["version_id"] = None
        attestation["context"]["verifier_version_id"] = None
    else:
        accepted["verifier_execution_id"] = None
        reward["verifier_execution_id"] = None
        attestation["activity"]["verifier_execution_id"] = None
        attestation["shadow"]["production_execution_id"] = None
        attestation["shadow"]["direct_verifier"]["execution_id"] = None
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    for name, value in (
        ("binding.json", binding),
        ("instance.json", instance),
        ("reward.json", reward),
        ("cleanup.json", cleanup),
        ("ACCEPTED.json", accepted),
    ):
        write(episode / name, value)
    _rebind_acceptance(
        episode,
        index_path,
        config,
        config_sha=binding["config_sha256"],
    )
    with pytest.raises(self_trace.SelfTraceError):
        self_trace.build(config, relative_to=tmp_path)


def test_identity_validator_requires_canonical_nonzero_uuid():
    assert self_trace._canonical_uuid(EVIDENCE_RUN_ID)
    assert not self_trace._canonical_uuid(None)
    assert not self_trace._canonical_uuid("00000000-0000-0000-0000-000000000000")
    assert not self_trace._canonical_uuid("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA")


def test_instance_identity_validator_matches_fleet_canonical_contract():
    assert self_trace._canonical_instance_id(INSTANCE_ID)
    assert self_trace._canonical_instance_id("a")
    assert not self_trace._canonical_instance_id(None)
    assert not self_trace._canonical_instance_id("")
    assert not self_trace._canonical_instance_id("Synthetic-Instance")
    assert not self_trace._canonical_instance_id("synthetic_instance")


@pytest.mark.parametrize(
    "defect",
    ["tool_catalog", "prompt_digest", "rl_limit", "provisioning_route", "catalog_mismatch"],
)
def test_incomplete_or_mismatched_direct_rl_binding_fails_closed(tmp_path, monkeypatch, defect):
    config, index_path, episode = fixture(tmp_path, monkeypatch)
    binding = json.loads((episode / "binding.json").read_text())
    if defect == "tool_catalog":
        binding["execution"].pop("required_task_tool_catalog_sha256")
    elif defect == "prompt_digest":
        binding.pop("initial_prompt_sha256")
    elif defect == "rl_limit":
        binding["rl"].pop("max_turns")
    elif defect == "provisioning_route":
        binding["authority"].pop("provisioning_route_template")
    else:
        binding["execution"]["required_task_tool_catalog_sha256"] = "sha256:" + "f" * 64
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    write(episode / "binding.json", binding)
    _rebind_acceptance(
        episode,
        index_path,
        config,
        config_sha=binding["config_sha256"],
    )
    with pytest.raises(self_trace.SelfTraceError):
        self_trace.build(config, relative_to=tmp_path)


def test_recorded_task_prompt_must_match_exact_task_binding(tmp_path, monkeypatch):
    config, index_path, episode = fixture(tmp_path, monkeypatch)
    conversation = json.loads((episode / "conversation.json").read_text())
    conversation["messages"][0]["content"] = "different private synthetic task"
    write(episode / "conversation.json", conversation)
    _rebind_acceptance(episode, index_path, config)
    with pytest.raises(self_trace.SelfTraceError, match="task prompt differs"):
        self_trace.build(config, relative_to=tmp_path)


def test_recorded_prompt_digest_is_recomputed(tmp_path, monkeypatch):
    config, index_path, episode = fixture(tmp_path, monkeypatch)
    binding = json.loads((episode / "binding.json").read_text())
    binding["initial_prompt_tokens_sha256"] = "sha256:" + "f" * 64
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    write(episode / "binding.json", binding)
    _rebind_acceptance(episode, index_path, config, config_sha=binding["config_sha256"])
    with pytest.raises(self_trace.SelfTraceError, match="prompt token digest"):
        self_trace.build(config, relative_to=tmp_path)


def test_template_is_inert_but_binds_wandb_loss_and_q1():
    root = Path(__file__).parents[1]
    index = json.loads(
        (root / "configs/qualification/qwen38-self-trace-index-a-v1.template.json").read_text()
    )
    corpus = json.loads(
        (root / "configs/qualification/qwen38-self-trace-corpus-a-v1.template.json").read_text()
    )
    sft = json.loads(
        (root / "configs/qualification/qwen38-self-trace-sft-a-dev-v1.template.json").read_text()
    )
    assert corpus["schema"] == self_trace.CONFIG_SCHEMA
    assert index["schema"] == self_trace.INDEX_SCHEMA
    assert index["episodes"] == []
    assert "PENDING" in index["runtime_image"] and "PENDING" in index["sha256"]
    assert "PENDING" in index["route_certificate_sha256"]
    assert "PENDING" in index["interface"]["required_task_tool_catalog_sha256"]
    assert index["interface"]["recorder_source_sha256"] == file_sha256(
        root / "training/skyrl_episode.py"
    )
    assert corpus["split"]["sha256"] == file_sha256(
        root / "configs/data/qwen-blackbox-study-train-a-v1.json"
    )
    assert "PENDING" in corpus["source_index"]["sha256"]
    assert sft["cluster"]["priority"] == "c1"
    assert sft["validation_mode"] == "task_outcomes_only"
    assert sft["recipe"]["eval_interval"] == 0
    assert sft["pause_after_step"] == 6
    assert sft["wandb"]["project"] == "cyber-post-train"
    assert "self-sft" in sft["wandb"]["tags"]
