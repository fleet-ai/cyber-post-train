"""Static safety and provenance gates for the first real Qwen3.8 RL dev canary."""
# ruff: noqa: F811

import hashlib
import json
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import JobsError, digest
from training import rl_data, rl_reward_canary, skyrl, skyrl_training

ROOT = Path(__file__).resolve().parents[1]
TASK_SET = ROOT / "configs/data/qwen38-rl-reward-canary-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-rl-reward-canary-split-v1.json"
DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-dev-v1.json"
RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-dev-v1.json"
RELOAD = ROOT / ("configs/qualification/qwen38-rl-reward-canary-reload-dev-v1.template.json")
FILTERED_TASK_SET = ROOT / "configs/data/qwen38-rl-filtered-canary-task-set-v1.json"
DEV8_DATA = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v8.json"
)
DEV8_RUN = ROOT / ("configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v8.json")
PROD2_RUN = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-prod-v2.json"
)
PROD2_QUALIFICATION = ROOT / (
    "docs/evidence/qwen38-study/2026-09-14-skyrl-prod2-engine-qualification-v1.json"
)
SUCCESSOR_EVIDENCE = ROOT / (
    "configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json"
)
VERSION_EVIDENCE = ROOT / ("configs/data/qwen38-rl-reward-canary-exact-version-evidence-v1.json")
HORIZON_CONTRACT = ROOT / (
    "configs/runs/qwen38-27b-native-rl-reward-acquisition-canary.template.json"
)
HORIZON_AGGREGATE = ROOT / "docs/FLEET_NATIVE_HARNESS_PARITY.md"
DEFAULT_USER_EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-worker-rpc-relay-image-default-user-qualification-v1.json"
)
_OMITTED = object()


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def assert_self_digest(value: dict) -> None:
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def exact_reward_plan(skyrl_prepared) -> dict:
    plan = deepcopy(skyrl_prepared.plan)
    plan["run_name"] = skyrl_training.REWARD_CANARY_ARGUMENTS["name"]
    plan["output_root"] = skyrl_training.REWARD_CANARY_ARGUMENTS["output_root"]
    plan["arguments"] = deepcopy(skyrl_training.REWARD_CANARY_ARGUMENTS)
    plan["native_overrides"] = skyrl.overrides(skyrl.SkyRLConfig(**plan["arguments"]))
    plan["model"].update(skyrl_training.REWARD_CANARY_MODEL_IDENTITY)
    plan["data"].update(
        {
            "name": skyrl_training.REWARD_CANARY_ARGUMENTS["name"],
            **deepcopy(skyrl_training.REWARD_CANARY_DATA_CONTRACT),
        }
    )
    plan["data"].pop("rows")
    for split, rows in skyrl_training.REWARD_CANARY_DATA_CONTRACT["rows"].items():
        plan["data"]["files"][split]["path"] = f"{split}.jsonl"
        plan["data"]["files"][split]["rows"] = rows
    plan["data"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in plan["data"].items() if key != "sha256"}
    )
    plan["execution"] = {
        "image": skyrl_training.IMAGE,
        "image_cpu_qualification": deepcopy(skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION),
        "reward_canary_source": {"bound": "synthetic-test-only"},
        "runtime_user": deepcopy(skyrl_training.REWARD_CANARY_RUNTIME_USER),
        "engine_diagnostic_prerequisite": {"bound": "synthetic-test-only"},
        "cluster_target": "dev",
        "priority": "c1",
        "resources": deepcopy(skyrl_training.REWARD_CANARY_RESOURCES),
    }
    return plan


def allow_synthetic_reward_bindings(monkeypatch) -> None:
    monkeypatch.setattr(
        skyrl_training,
        "_validate_embedded_engine_prerequisite",
        lambda value, *, required: None,
    )
    monkeypatch.setattr(rl_reward_canary, "validate_source_proof", lambda proof, metadata: None)


def reward_preview(*, uid: int = 1000, gid: int = 100, nonroot: bool = True) -> dict:
    pod = {
        "spec": {
            "securityContext": {
                "runAsUser": uid,
                "runAsGroup": gid,
                "runAsNonRoot": nonroot,
            },
            "containers": [{}],
        }
    }
    return {
        "manifest_yaml": yaml.safe_dump(
            {
                "metadata": {"namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE},
                "spec": {
                    "rayClusterSpec": {
                        "headGroupSpec": {"template": pod},
                        "workerGroupSpecs": [],
                    }
                },
            }
        )
    }


def reward_preview_with_contexts(
    *, pod_context: object = _OMITTED, container_context: object = _OMITTED
) -> dict:
    pod_spec = {"containers": [{}]}
    if pod_context is not _OMITTED:
        pod_spec["securityContext"] = pod_context
    if container_context is not _OMITTED:
        pod_spec["containers"][0]["securityContext"] = container_context
    return {
        "manifest_yaml": yaml.safe_dump(
            {
                "metadata": {"namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE},
                "spec": {
                    "rayClusterSpec": {
                        "headGroupSpec": {"template": {"spec": pod_spec}},
                        "workerGroupSpecs": [],
                    }
                },
            }
        )
    }


def diagnostic_preview(*, uid: int = 1000, gid: int = 100, nonroot: bool = True) -> dict:
    pod = {
        "spec": {
            "securityContext": {
                "runAsUser": uid,
                "runAsGroup": gid,
                "runAsNonRoot": nonroot,
            },
            "containers": [{}],
        }
    }
    return {
        "manifest_yaml": yaml.safe_dump(
            {
                "metadata": {"namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE},
                "spec": {
                    "rayClusterSpec": {
                        "headGroupSpec": {"template": deepcopy(pod)},
                        "workerGroupSpecs": [{"replicas": 1, "template": deepcopy(pod)}],
                    }
                },
            }
        )
    }


def test_task_selection_uses_the_authoritative_source_directly() -> None:
    task_set, split = load(TASK_SET), load(SPLIT)
    old_task_set, evidence = load(FILTERED_TASK_SET), load(SUCCESSOR_EVIDENCE)
    version_evidence = load(VERSION_EVIDENCE)
    assert_self_digest(task_set)
    assert_self_digest(split)
    assert_self_digest(version_evidence)
    provenance = task_set["reward_signal_provenance"]
    assert provenance["kind"] == "eligible_authoritative_source_direct"
    assert provenance["evidence_path"] == str(SUCCESSOR_EVIDENCE.relative_to(ROOT))
    assert (
        provenance["evidence_file_sha256"]
        == "sha256:" + hashlib.sha256(SUCCESSOR_EVIDENCE.read_bytes()).hexdigest()
    )
    assert provenance["exact_version_evidence"] == {
        "path": "../data/qwen38-rl-reward-canary-exact-version-evidence-v1.json",
        "file_sha256": "sha256:" + hashlib.sha256(VERSION_EVIDENCE.read_bytes()).hexdigest(),
        "self_sha256": version_evidence["sha256"],
    }
    assert task_set["source_manifest_sha256"] == old_task_set["source_manifest_sha256"]
    assert task_set["tool_catalog_sha256"] == old_task_set["tool_catalog_sha256"]

    old_by_key = {row["task_key"]: row for row in old_task_set["tasks"]}
    new_by_key = {row["task_key"]: row for row in task_set["tasks"]}
    assignments = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]
    }
    train = [
        row
        for row in task_set["tasks"]
        if assignments[(row["task_key"], row["task_version_id"])] == "train"
    ]
    dev = [
        row
        for row in task_set["tasks"]
        if assignments[(row["task_key"], row["task_version_id"])] == "dev"
    ]
    assert len(train) == 1 and len(dev) == 1
    source = old_by_key[train[0]["task_key"]]
    assert train[0] == source
    assert dev[0] == old_by_key[dev[0]["task_key"]]
    assert set(new_by_key) < set(old_by_key)
    assert rl_data.selection(task_set, split) == [
        {**dev[0], "split": "dev"},
        {**train[0], "split": "train"},
    ]
    assert "webexploit" not in json.dumps(task_set).lower()
    assert all(row["reference_session_id"] is None for row in split["tasks"])

    selected = version_evidence["selected_version"]
    assert selected["task_version_id"] == train[0]["task_version_id"]
    assert selected["eligibility_status"] == "eligible_authoritative_source_direct"
    assert task_set["training_data_eligible"] is True
    assert version_evidence["eligible_source"]["task_version_id"] == source["task_version_id"]
    rejected = version_evidence["rejected_metadata_only_successor"]
    assert (
        rejected["successor_task_version_id"]
        == (evidence["task_successors"][0]["metadata_only_successor_task_version_id"])
    )
    assert rejected["eligibility_status"] == "rejected_missing_versioned_starting_data"
    assert rejected["authoritative_differing_fields"] == ["data_id", "data_version"]
    tools = version_evidence["tool_surface_authority"]
    assert tools["task_metadata_tools_required"] is False
    assert tools["source_metadata_tools"] is None
    assert tools["ordered_tools"] == ["bash", "submit_report"]
    assert tools["local_code"]["episode_runtime"]["file_sha256"] == (
        "sha256:" + hashlib.sha256((ROOT / "training/rl_episode.py").read_bytes()).hexdigest()
    )
    rl_reward_canary.validate_exact_version_evidence(
        task_set,
        split,
        load(DATA)["limits"],
        task_set_dir=TASK_SET.parent,
    )


def test_exact_version_and_horizon_evidence_are_fail_closed() -> None:
    task_set, split, limits = load(TASK_SET), load(SPLIT), load(DATA)["limits"]
    evidence = load(VERSION_EVIDENCE)
    assert evidence["episode_horizon"] == {
        "max_turns": 600,
        "task_specific_contract_path": (
            "../runs/qwen38-27b-native-rl-reward-acquisition-canary.template.json"
        ),
        "task_specific_contract_file_sha256": (
            "sha256:" + hashlib.sha256(HORIZON_CONTRACT.read_bytes()).hexdigest()
        ),
        "selected_task_row_sha256": (
            "sha256:1cda928ba27c57235eafed117db3187c64e984b237ebd8ee56743711c68f07b4"
        ),
        "rollout_contract_sha256": (
            "sha256:5303bc66daa31bff54c6a0d396fd8ada98cf8655078255ffb0cec53588635d9c"
        ),
        "aggregate_evidence_path": "../../docs/FLEET_NATIVE_HARNESS_PARITY.md",
        "aggregate_evidence_file_sha256": (
            "sha256:" + hashlib.sha256(HORIZON_AGGREGATE.read_bytes()).hexdigest()
        ),
        "rationale": evidence["episode_horizon"]["rationale"],
        "fixed_limits": {key: value for key, value in limits.items() if key != "max_turns"},
    }
    assert limits["max_turns"] == 600

    changed = deepcopy(task_set)
    changed["reward_signal_provenance"]["exact_version_evidence"]["self_sha256"] = (
        "sha256:" + "0" * 64
    )
    changed["sha256"] = "sha256:" + digest(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="exact immutable source identity"):
        rl_reward_canary.validate_exact_version_evidence(
            changed,
            split,
            limits,
            task_set_dir=TASK_SET.parent,
        )
    with pytest.raises(ValueError, match="exact immutable source identity"):
        rl_reward_canary.validate_exact_version_evidence(
            task_set,
            split,
            {**limits, "max_turns": 80},
            task_set_dir=TASK_SET.parent,
        )

    assert rl_reward_canary.validate_config(load(DATA), relative_to=DATA.parent) == (
        rl_reward_canary.source_proof()
    )
    assert hashlib.sha256((ROOT / "training/rl_data.py").read_bytes()).hexdigest() == (
        "f0c327d2ecc464610a5fd86e11b112870d28feb43d9155d7507d8fedbd87632b"
    )


def test_real_canary_is_one_dev_only_update_with_durable_evidence() -> None:
    data, run = load(DATA), load(RUN)
    dev8_data, dev8_run = load(DEV8_DATA), load(DEV8_RUN)
    assert data["name"] == run["name"] == run["wandb"]["run_id"]
    assert data["name"] == "chris-q38-rlreward-dev1"
    assert data["backend"] == run["backend"] == "skyrl"
    assert run["output_root"] == "/mnt/sfs/jobs/chris-q38-rlreward-dev1"
    assert data["output"] == run["data"]["root"]
    assert run["data"]["manifest"] == data["output"] + "/manifest.json"
    assert data["task_set"].endswith("qwen38-rl-reward-canary-task-set-v1.json")
    assert data["split"].endswith("qwen38-rl-reward-canary-split-v1.json")

    for key in ("backend", "model_lock", "model_root"):
        assert data[key] == dev8_data[key]
    assert {key: value for key, value in data["limits"].items() if key != "max_turns"} == {
        key: value for key, value in dev8_data["limits"].items() if key != "max_turns"
    }
    assert data["limits"]["max_turns"] == 600
    assert dev8_data["limits"]["max_turns"] == 80
    assert data["tool_catalog"] == dev8_data["tool_catalog"]
    assert run["model"] == dev8_run["model"]
    for key in (
        "nodes",
        "steps",
        "lr",
        "eval_interval",
        "checkpoint_interval",
        "keep_checkpoints",
        "seed",
    ):
        assert run["recipe"][key] == dev8_run["recipe"][key]
    assert (
        run["recipe"]["groups"] * run["recipe"]["samples_per_prompt"]
        == dev8_run["recipe"]["groups"] * dev8_run["recipe"]["samples_per_prompt"]
    )
    assert run["recipe"] == {
        "nodes": 1,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    assert run["cluster"] == {
        "target": "dev",
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    assert run["recipe"]["groups"] * run["recipe"]["samples_per_prompt"] == 8
    assert run["recipe"]["steps"] == run["recipe"]["checkpoint_interval"] == 1
    assert run["wandb"] == {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": run["name"],
    }
    assert data["output"] != dev8_data["output"]
    assert run["output_root"] != dev8_run["output_root"]
    gate = run["prerequisites"]["engine_diagnostic"]
    assert gate["config_path"] == PROD2_RUN.name
    assert gate["config_file_sha256"] == (
        "sha256:" + hashlib.sha256(PROD2_RUN.read_bytes()).hexdigest()
    )
    assert gate["terminal_receipt_path"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD2_TERMINAL_PATH
    assert gate["terminal_receipt_file_sha256"] == (
        skyrl_training.ENGINE_DIAGNOSTIC_PROD2_TERMINAL_FILE_SHA256
    )
    assert gate["qualification_file_sha256"] == (
        "sha256:" + hashlib.sha256(PROD2_QUALIFICATION.read_bytes()).hexdigest()
    )
    qualification = load(PROD2_QUALIFICATION)
    assert_self_digest(qualification)
    assert gate["qualification_self_sha256"] == qualification["sha256"]


def _synthetic_prod2_receipt() -> dict:
    receipt = {
        "schema": skyrl_training.ENGINE_DIAGNOSTIC_SCHEMA,
        "plan_sha256": "9f332dd6cdde4b8ec8d3f7f456fcd71f5954068857a354459232f61a85f0f1d9",
        "status": "passed",
        "startup_phase": "complete",
        "engine_start_state": "all",
        "engine_started": True,
        "diagnostic_completed": True,
        "engine_start_qualified": True,
        "training_qualified": False,
        "production_training_shape_qualified": False,
        "runtime_user": {"uid": 1000, "gid": 100},
        "diagnostic_workers": 2,
        "diagnostic_gpus_per_worker": 4,
        "diagnostic_total_gpus": 8,
        "num_engines": 2,
        "tensor_parallel_size": 4,
        "ray_gpu_nodes_expected": 2,
        "ray_gpu_nodes_discovered": 2,
        "ray_gpu_nodes_probed": 2,
        "ray_actor_environment_probes_passed": 2,
        "ray_actor_environment_probe_failures": 0,
        "ray_actor_nonempty_scrubbed_credentials": 0,
        "router_start_attempts": 1,
        "router_environment_probes_passed": 1,
        "router_environment_probe_failures": 0,
        "task_rows_read": 0,
        "rollouts": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "checkpoints_created": 0,
        "registry_actors_created": 0,
        "checkpoint_created": False,
        "wandb_initialized": False,
        "credential_environment_isolation_proven": True,
        "ray_actor_environment_isolation_proven": True,
        "router_child_credential_environment_isolation": "proven",
        "cleanup": {
            "cleanup_proven": True,
            "active_owned_actors": 0,
            "active_owned_placement_groups": 0,
        },
        "output_postconditions": {
            "checkpoint_artifacts": 0,
            "episode_artifacts": 0,
            "runtime_files_unchanged": True,
            "task_artifacts": 0,
            "unexpected_output_artifacts": 0,
        },
    }
    receipt["sha256"] = digest(receipt)
    return receipt


def test_prod2_prerequisite_reopens_terminal_and_release_proofs(monkeypatch) -> None:
    receipt = _synthetic_prod2_receipt()
    qualification = load(PROD2_QUALIFICATION)
    qualification["receipt"]["self_sha256"] = receipt["sha256"]
    qualification["sha256"] = "sha256:" + digest(
        {key: value for key, value in qualification.items() if key != "sha256"}
    )
    monkeypatch.setattr(
        skyrl_training,
        "ENGINE_DIAGNOSTIC_PROD2_QUALIFICATION_SELF_SHA256",
        qualification["sha256"],
    )

    skyrl_training._validate_prod2_terminal_receipt(receipt)
    skyrl_training._validate_prod2_qualification(qualification, receipt)
    proof = {
        "schema": "cyber_skyrl_engine_prerequisite_v2",
        "status": "accepted",
        "config_path": skyrl_training.ENGINE_DIAGNOSTIC_PROD2_CONFIG_PATH,
        "config_file_sha256": skyrl_training.ENGINE_DIAGNOSTIC_PROD2_CONFIG_FILE_SHA256,
        "terminal_receipt_path": skyrl_training.ENGINE_DIAGNOSTIC_PROD2_TERMINAL_PATH,
        "terminal_receipt_file_sha256": (
            skyrl_training.ENGINE_DIAGNOSTIC_PROD2_TERMINAL_FILE_SHA256
        ),
        "qualification_path": skyrl_training.ENGINE_DIAGNOSTIC_PROD2_QUALIFICATION_PATH,
        "qualification_file_sha256": (
            skyrl_training.ENGINE_DIAGNOSTIC_PROD2_QUALIFICATION_FILE_SHA256
        ),
        "qualification_self_sha256": qualification["sha256"],
        "terminal_receipt_self_sha256": receipt["sha256"],
        "terminal_receipt": receipt,
        "qualification": qualification,
    }
    proof["sha256"] = "sha256:" + digest(proof)
    skyrl_training._validate_embedded_engine_prerequisite(proof, required=True)

    changed = deepcopy(proof)
    changed["terminal_receipt"]["optimizer_steps"] = 1
    changed["terminal_receipt"]["sha256"] = digest(
        {
            key: value
            for key, value in changed["terminal_receipt"].items()
            if key != "sha256"
        }
    )
    changed["terminal_receipt_self_sha256"] = changed["terminal_receipt"]["sha256"]
    changed["sha256"] = "sha256:" + digest(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="accepted zero-work engine start"):
        skyrl_training._validate_embedded_engine_prerequisite(changed, required=True)


def test_canary_prerequisite_is_bound_to_data_identity_not_run_name(skyrl_prepared) -> None:
    plan = deepcopy(skyrl_prepared.plan)
    metadata = plan["data"]
    metadata.update(
        {
            "name": "renamed-reward-canary",
            **{
                key: deepcopy(value)
                for key, value in skyrl_training.REWARD_CANARY_DATA_CONTRACT.items()
                if key != "rows"
            },
        }
    )
    for split, rows in skyrl_training.REWARD_CANARY_DATA_CONTRACT["rows"].items():
        metadata["files"][split]["path"] = f"{split}.jsonl"
        metadata["files"][split]["rows"] = rows
    metadata["sha256"] = "sha256:" + digest(
        {key: value for key, value in metadata.items() if key != "sha256"}
    )
    plan["arguments"]["name"] = plan["run_name"] = metadata["name"]
    assert skyrl_training._validate_reward_canary_data(metadata) is True
    with pytest.raises(ValueError, match="exact accepted engine diagnostic prerequisite"):
        skyrl_training.job_request(plan)

    changed = deepcopy(metadata)
    changed["limits"]["max_turns"] = 80
    with pytest.raises(ValueError, match="exact reviewed contract"):
        skyrl_training._validate_reward_canary_data(changed)


def test_historical_dev8_receipt_validation_is_read_only_and_cannot_promote(
    tmp_path, skyrl_prepared, monkeypatch
) -> None:
    def seal(value: dict) -> dict:
        body = {key: item for key, item in value.items() if key != "sha256"}
        return {**body, "sha256": digest(body)}

    def write_json(path: Path, value: dict) -> str:
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    # Reopen the historical closure under its own image pins without allowing
    # any executable path to reuse it. The active globals remain dev9-only
    # outside this synthetic read-only validator test.
    monkeypatch.setattr(skyrl_training, "IMAGE", skyrl_training.DEV8_ENGINE_IMAGE)
    monkeypatch.setattr(
        skyrl_training,
        "ENGINE_IMAGE_CPU_QUALIFICATION",
        skyrl_training.DEV8_ENGINE_IMAGE_CPU_QUALIFICATION,
    )
    monkeypatch.setattr(skyrl_training, "_require_fresh_engine_diagnostic_identity", lambda _: None)

    run = load(RUN)
    diagnostic_path = tmp_path / DEV8_RUN.name
    diagnostic_path.write_bytes(DEV8_RUN.read_bytes())
    config_sha = "sha256:" + hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()

    plan = deepcopy(skyrl_prepared.plan)
    plan["model"]["root"] = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
    plan["run_name"] = skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME
    plan["output_root"] = skyrl_training.ENGINE_DIAGNOSTIC_OUTPUT_ROOT
    data_root = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev8/data"
    lock = load(ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json")
    plan["data"].update(
        {
            "name": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME,
            "gpus": 0,
            "environment_creates": 0,
            "selection_sha256": skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY["selection_sha256"],
            "split_sha256": skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY["split_sha256"],
            "tool_catalog_sha256": skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY[
                "tool_catalog_sha256"
            ],
            "template_sha256": skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY["template_sha256"],
            "limits": deepcopy(skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY["limits"]),
            "tokenizer": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "files": lock["tokenizer"]["files"],
                "backend_sha256": (
                    "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
                ),
                "chat_template_sha256": skyrl_training.ENGINE_DIAGNOSTIC_DATA_IDENTITY[
                    "template_sha256"
                ].removeprefix("sha256:"),
            },
            "files": {
                "train": {
                    "path": "train.jsonl",
                    "rows": 2,
                    "max_prompt_tokens": 1241,
                    "sha256": "sha256:" + "1" * 64,
                },
                "dev": {
                    "path": "dev.jsonl",
                    "rows": 1,
                    "max_prompt_tokens": 1256,
                    "sha256": "sha256:" + "2" * 64,
                },
            },
        }
    )
    plan["data"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in plan["data"].items() if key != "sha256"}
    )
    plan["arguments"] = {
        "name": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "output_root": skyrl_training.ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
        "model": "Qwen/Qwen3.8-27B",
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "train_data": data_root + "/train.jsonl",
        "dev_data": data_root + "/dev.jsonl",
        "data_manifest": data_root + "/manifest.json",
        "train_rows": 2,
        "dev_rows": 1,
        "wandb_entity": "thefleet",
        "wandb_project": "cyber-post-train",
        "wandb_run_id": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "context_tokens": 98304,
        "response_tokens": 81920,
        "tokens_per_turn": 4096,
        "max_turns": 80,
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    plan["native_overrides"] = skyrl.overrides(skyrl.SkyRLConfig(**plan["arguments"]))
    plan["execution"] = {
        "image": skyrl_training.IMAGE,
        "image_cpu_qualification": deepcopy(skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION),
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    request = skyrl_training.engine_diagnostic_request(plan)
    plan_sha256, request_sha256 = digest(plan), digest(request)

    prepared_root = tmp_path / "durable" / "prepared-v1"
    prepared_root.mkdir(parents=True)
    plan_file_sha256 = write_json(prepared_root / "plan.json", plan)
    request_file_sha256 = write_json(prepared_root / "request.json", request)
    prepared_file_sha256 = write_json(
        prepared_root / "PREPARED.json",
        {"plan_sha256": plan_sha256, "request_sha256": request_sha256},
    )
    preflight = {
        "schema": "cyber_skyrl_engine_diagnostic_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "native_parser_checked": True,
        "engine_cli_args_checked": True,
        "startup_error_transport_checked": True,
        "model_files": len(plan["model"]["files"]),
        "task_rows_read": 0,
        "rollouts": False,
        "verifier_calls": False,
        "optimizer_updates": False,
        "checkpoints": False,
        "wandb": False,
        "engine_start_qualified": False,
        "diagnostic_workers": 2,
        "diagnostic_gpus_per_worker": 4,
        "diagnostic_total_gpus": 8,
        "num_engines": 2,
        "tensor_parallel_size": 4,
    }
    preflight["sha256"] = digest(preflight)
    preflight_file_sha256 = write_json(prepared_root / "PREFLIGHT.json", preflight)

    api_run_name = skyrl_training.ENGINE_DIAGNOSTIC_API_RUN_NAME
    api_job_id = skyrl_training.ENGINE_DIAGNOSTIC_API_JOB_ID
    rayjob_name = api_run_name
    journal = [
        {
            "state": "POST_INTENT_DO_NOT_RETRY",
            "api_base_url": "https://api.ft.dev.flt.build",
            "request_sha256": request_sha256,
            "manifest_sha256": skyrl_training.ENGINE_DIAGNOSTIC_MANIFEST_SHA256,
            "nodes": 2,
            "gpus": 8,
            "image": skyrl_training.IMAGE,
        },
        {
            "state": "POST_RESPONSE",
            "name": rayjob_name,
            "job_id": None,
            "run_dir": skyrl_training.ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
            "status": None,
            "created_at": None,
            "finished_at": None,
        },
    ]
    journal_path = prepared_root / "SUBMISSION.jsonl"
    journal_path.write_text("\n".join(json.dumps(row) for row in journal) + "\n")
    journal_file_sha256 = "sha256:" + hashlib.sha256(journal_path.read_bytes()).hexdigest()
    for name, value in (
        ("ENGINE_DIAGNOSTIC_PLAN_FILE_SHA256", plan_file_sha256),
        ("ENGINE_DIAGNOSTIC_REQUEST_FILE_SHA256", request_file_sha256),
        ("ENGINE_DIAGNOSTIC_PREPARED_FILE_SHA256", prepared_file_sha256),
        ("ENGINE_DIAGNOSTIC_PREFLIGHT_FILE_SHA256", preflight_file_sha256),
        ("ENGINE_DIAGNOSTIC_SUBMISSION_JOURNAL_FILE_SHA256", journal_file_sha256),
        ("ENGINE_DIAGNOSTIC_PLAN_SHA256", plan_sha256),
        ("ENGINE_DIAGNOSTIC_REQUEST_SHA256", request_sha256),
    ):
        monkeypatch.setattr(skyrl_training, name, value)

    output_root = tmp_path / "diagnostic-output"
    output_root.mkdir()
    native = seal(
        {
            "schema": skyrl_training.ENGINE_DIAGNOSTIC_SCHEMA,
            "status": "passed",
            "engine_start_state": "all",
            "engine_started": True,
            "plan_sha256": plan_sha256,
            "runtime_user": {"uid": 1000, "gid": 100},
            "task_rows_read": 0,
            "rollouts": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "checkpoint_created": False,
            "wandb_initialized": False,
            "diagnostic_completed": True,
            "engine_start_qualified": True,
            "training_qualified": False,
            "production_training_shape_qualified": False,
            "registry_actors_created": 0,
            "diagnostic_workers": 2,
            "diagnostic_gpus_per_worker": 4,
            "diagnostic_total_gpus": 8,
            "num_engines": 2,
            "tensor_parallel_size": 4,
            "ray_gpu_nodes_expected": 2,
            "ray_gpu_nodes_discovered": 2,
            "ray_gpu_nodes_probed": 2,
            "ray_actor_environment_probes_passed": 2,
            "ray_actor_environment_probe_failures": 0,
            "ray_actor_nonempty_scrubbed_credentials": 0,
            "credential_environment_isolation_proven": True,
            "ray_actor_environment_isolation_proven": True,
            "ray_initialization_attempted": True,
            "router_child_credential_environment_isolation": "proven",
            "router_start_attempts": 1,
            "router_environment_probes_passed": 1,
            "router_environment_probe_failures": 0,
            "cleanup": {
                "tracked_engine_actors": 2,
                "active_owned_actors": 0,
                "active_owned_placement_groups": 0,
                "cleanup_proven": True,
            },
            "output_postconditions": {
                "checkpoint_artifacts": 0,
                "episode_artifacts": 0,
                "task_artifacts": 0,
                "unexpected_output_artifacts": 0,
                "runtime_files_unchanged": True,
            },
            "completed_at": 1.0,
        }
    )
    native_path = output_root / "ENGINE_DIAGNOSTIC.json"
    native_file_sha256 = write_json(native_path, native)

    rayjob_uid = skyrl_training.ENGINE_DIAGNOSTIC_RAYJOB_UID
    workload_uid = skyrl_training.ENGINE_DIAGNOSTIC_WORKLOAD_UID
    raycluster_uid = skyrl_training.ENGINE_DIAGNOSTIC_RAYCLUSTER_UID
    pod_uids = [uid for _, uid in skyrl_training.ENGINE_DIAGNOSTIC_PODS]
    runtime_image_id = skyrl_training.IMAGE
    audit = seal(
        {
            "schema": "cyber_skyrl_engine_diagnostic_controller_audit_v1",
            "cluster": "dev",
            "api_base_url": "https://api.ft.dev.flt.build",
            "kubernetes_context": skyrl_training.DEV_KUBERNETES_CONTEXT,
            "namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE,
            "namespace_uid": skyrl_training.DEV_KUBERNETES_NAMESPACE_UID,
            "config_name": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME,
            "plan_sha256": plan_sha256,
            "request_sha256": request_sha256,
            "api_run_name": api_run_name,
            "api_job_id": api_job_id,
            "rayjob_name": rayjob_name,
            "rayjob_uid": rayjob_uid,
            "workload_name": skyrl_training.ENGINE_DIAGNOSTIC_WORKLOAD_NAME,
            "workload_uid": workload_uid,
            "workload_owner_rayjob_uid": rayjob_uid,
            "raycluster_name": skyrl_training.ENGINE_DIAGNOSTIC_RAYCLUSTER_NAME,
            "raycluster_uid": raycluster_uid,
            "raycluster_owner_rayjob_uid": rayjob_uid,
            "pods": [
                {
                    "name": name,
                    "uid": uid,
                    "owner_raycluster_uid": raycluster_uid,
                    "runtime_image_id": runtime_image_id,
                    "runtime_uid": 1000,
                    "runtime_gid": 100,
                    "effective_security_context": {
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "runAsNonRoot": True,
                    },
                    "phase": "Succeeded",
                    "container_exit_code": 0,
                    "termination_reason": "Completed",
                    "terminated_at": "2026-09-11T23:59:00Z",
                    "container_restarts": 0,
                    "gpus": 4,
                }
                for name, uid in skyrl_training.ENGINE_DIAGNOSTIC_PODS
            ],
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 2,
            "gpus_per_worker": 4,
            "total_gpus": 8,
            "observed_at": "2026-09-12T00:00:00Z",
        }
    )
    audit_path = output_root / "CONTROLLER_AUDIT.json"
    audit_file_sha256 = write_json(audit_path, audit)
    release = seal(
        {
            "schema": "cyber_skyrl_engine_diagnostic_release_v1",
            "status": "released",
            "cluster": "dev",
            "api_base_url": "https://api.ft.dev.flt.build",
            "kubernetes_context": skyrl_training.DEV_KUBERNETES_CONTEXT,
            "namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE,
            "namespace_uid": skyrl_training.DEV_KUBERNETES_NAMESPACE_UID,
            "config_name": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_NAME,
            "plan_sha256": plan_sha256,
            "request_sha256": request_sha256,
            "prepared_directory": str(prepared_root),
            "preflight_file_sha256": preflight_file_sha256,
            "submission_journal_file_sha256": journal_file_sha256,
            "diagnostic_receipt_file_sha256": native_file_sha256,
            "controller_audit_file_sha256": audit_file_sha256,
            "controller_audit_self_sha256": audit["sha256"],
            "api_run_name": api_run_name,
            "api_job_id": api_job_id,
            "rayjob_name": rayjob_name,
            "rayjob_uid": rayjob_uid,
            "workload_name": audit["workload_name"],
            "workload_uid": workload_uid,
            "raycluster_name": audit["raycluster_name"],
            "raycluster_uid": raycluster_uid,
            "pod_uids": pod_uids,
            "runtime_image_ids": [runtime_image_id, runtime_image_id],
            "runtime_uids": [1000, 1000],
            "runtime_gids": [100, 100],
            "effective_security_contexts": [
                {
                    "runAsUser": 1000,
                    "runAsGroup": 100,
                    "runAsNonRoot": True,
                },
                {
                    "runAsUser": 1000,
                    "runAsGroup": 100,
                    "runAsNonRoot": True,
                },
            ],
            "pod_terminal_observations": [
                {
                    "uid": uid,
                    "phase": "Succeeded",
                    "container_exit_code": 0,
                    "termination_reason": "Completed",
                    "terminated_at": "2026-09-11T23:59:00Z",
                }
                for uid in pod_uids
            ],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 2,
            "gpus_per_worker": 4,
            "total_gpus": 8,
            "container_restarts": 0,
            "raycluster_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
            "gpu_release_proven": True,
            "observed_at": "2026-09-12T00:01:00Z",
        }
    )
    release_path = output_root / "RELEASE.json"
    release_file_sha256 = write_json(release_path, release)
    prepared_inputs = {
        "directory": str(prepared_root),
        "source_commit": skyrl_training.ENGINE_DIAGNOSTIC_SOURCE_COMMIT,
        "config_file_sha256": config_sha,
        "plan_file_sha256": plan_file_sha256,
        "request_file_sha256": request_file_sha256,
        "prepared_receipt_file_sha256": prepared_file_sha256,
        "preflight_file_sha256": preflight_file_sha256,
        "submission_journal_file_sha256": journal_file_sha256,
    }

    def binding(path: Path, value: dict) -> dict:
        return {
            "path": str(path),
            "file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "value": value,
        }

    receipt = {
        "schema": skyrl_training.ENGINE_DIAGNOSTIC_ACCEPTANCE_SCHEMA,
        "classification": "accepted_engine_start_zero_work",
        "prepared_inputs": prepared_inputs,
        "diagnostic_receipt": binding(native_path, native),
        "controller_audit": binding(audit_path, audit),
        "release_evidence": binding(release_path, release),
    }
    receipt["sha256"] = digest(receipt)
    receipt_path = tmp_path / "dev8-terminal.json"
    receipt_path.write_bytes(skyrl_training._canonical_json(receipt))
    gate = {
        "engine_diagnostic": {
            "config_path": diagnostic_path.name,
            "config_file_sha256": config_sha,
            "terminal_receipt_path": receipt_path.name,
            "terminal_receipt_file_sha256": (
                "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            ),
        }
    }
    monkeypatch.setattr(skyrl_training, "_engine_durable_root", lambda: tmp_path / "durable")
    monkeypatch.setattr(skyrl_training, "_engine_evidence_root", lambda: output_root)
    monkeypatch.setattr(
        skyrl_training,
        "_compile_exact_engine_diagnostic",
        lambda config, *, relative_to: (plan, request),
    )

    immutable = skyrl_training._validate_engine_diagnostic_receipt(
        receipt,
        config_file_sha256=config_sha,
        expected_plan_sha256=plan_sha256,
        expected_request_sha256=request_sha256,
    )
    assert immutable["pod_uids"] == pod_uids
    assert immutable["runtime_user"] == {"uid": 1000, "gid": 100}
    assert immutable["release_file_sha256"] == release_file_sha256
    with pytest.raises(ValueError, match="historical and privacy-disqualified"):
        skyrl_training._accepted_engine_diagnostic(gate, run, relative_to=tmp_path)

    changed_native = deepcopy(native)
    changed_native["runtime_user"] = {"uid": 0, "gid": 0}
    changed_native = seal(changed_native)
    write_json(native_path, changed_native)
    changed = deepcopy(receipt)
    changed["diagnostic_receipt"] = binding(native_path, changed_native)
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="native receipt"):
        skyrl_training._validate_engine_diagnostic_receipt(
            changed,
            config_file_sha256=config_sha,
            expected_plan_sha256=plan_sha256,
            expected_request_sha256=request_sha256,
        )
    write_json(native_path, native)

    arbitrary = deepcopy(receipt)
    arbitrary["prepared_inputs"]["plan_file_sha256"] = "sha256:" + "9" * 64
    arbitrary["sha256"] = digest(
        {key: value for key, value in arbitrary.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="immutable active closure"):
        skyrl_training._validate_engine_diagnostic_receipt(
            arbitrary,
            config_file_sha256=config_sha,
            expected_plan_sha256=plan_sha256,
            expected_request_sha256=request_sha256,
        )

    changed_plan = deepcopy(plan)
    changed_plan["model"]["root"] = "/mnt/sfs/models/other-qwen-root"
    changed_plan["arguments"]["model_root"] = "/mnt/sfs/models/other-qwen-root"
    changed_plan["native_overrides"] = skyrl.overrides(
        skyrl.SkyRLConfig(**changed_plan["arguments"])
    )
    changed_request = skyrl_training.engine_diagnostic_request(changed_plan)
    changed_plan_sha256, changed_request_sha256 = digest(changed_plan), digest(changed_request)
    changed = deepcopy(receipt)
    changed["prepared_inputs"]["plan_file_sha256"] = write_json(
        prepared_root / "plan.json", changed_plan
    )
    changed["prepared_inputs"]["request_file_sha256"] = write_json(
        prepared_root / "request.json", changed_request
    )
    changed["prepared_inputs"]["prepared_receipt_file_sha256"] = write_json(
        prepared_root / "PREPARED.json",
        {
            "plan_sha256": changed_plan_sha256,
            "request_sha256": changed_request_sha256,
        },
    )
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with monkeypatch.context() as changed_constants:
        changed_constants.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_PLAN_FILE_SHA256",
            changed["prepared_inputs"]["plan_file_sha256"],
        )
        changed_constants.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_REQUEST_FILE_SHA256",
            changed["prepared_inputs"]["request_file_sha256"],
        )
        changed_constants.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_PREPARED_FILE_SHA256",
            changed["prepared_inputs"]["prepared_receipt_file_sha256"],
        )
        changed_constants.setattr(
            skyrl_training, "ENGINE_DIAGNOSTIC_PLAN_SHA256", changed_plan_sha256
        )
        changed_constants.setattr(
            skyrl_training, "ENGINE_DIAGNOSTIC_REQUEST_SHA256", changed_request_sha256
        )
        with pytest.raises(ValueError, match="exact source config closure"):
            skyrl_training._validate_engine_diagnostic_receipt(
                changed,
                config_file_sha256=config_sha,
                expected_plan_sha256=changed_plan_sha256,
                expected_request_sha256=changed_request_sha256,
            )
    write_json(prepared_root / "plan.json", plan)
    write_json(prepared_root / "request.json", request)
    write_json(
        prepared_root / "PREPARED.json",
        {"plan_sha256": plan_sha256, "request_sha256": request_sha256},
    )

    for mutate in (
        lambda value: value.update(kubernetes_context="other-context"),
        lambda value: value.update(workload_owner_rayjob_uid=workload_uid),
        lambda value: value["pods"][0].update(
            runtime_image_id=skyrl_training.IMAGE.rsplit("sha256:", 1)[0] + "sha256:" + "0" * 64
        ),
        lambda value: value["pods"][0].update(runtime_uid=0),
        lambda value: value["pods"][0].update(
            effective_security_context={
                "runAsUser": 0,
                "runAsGroup": 100,
                "runAsNonRoot": False,
            }
        ),
        lambda value: value["pods"][0].update(container_exit_code=1),
        lambda value: value["pods"][0].update(phase="Failed"),
    ):
        changed_audit = deepcopy(audit)
        mutate(changed_audit)
        changed_audit = seal(changed_audit)
        write_json(audit_path, changed_audit)
        changed = deepcopy(receipt)
        changed["controller_audit"] = binding(audit_path, changed_audit)
        changed["sha256"] = digest(
            {key: value for key, value in changed.items() if key != "sha256"}
        )
        with pytest.raises(ValueError, match="controller audit"):
            skyrl_training._validate_engine_diagnostic_receipt(
                changed,
                config_file_sha256=config_sha,
                expected_plan_sha256=plan_sha256,
                expected_request_sha256=request_sha256,
            )
    write_json(audit_path, audit)

    changed_release = seal({**release, "gpu_release_proven": False})
    write_json(release_path, changed_release)
    changed = deepcopy(receipt)
    changed["release_evidence"] = binding(release_path, changed_release)
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="release evidence"):
        skyrl_training._validate_engine_diagnostic_receipt(
            changed,
            config_file_sha256=config_sha,
            expected_plan_sha256=plan_sha256,
            expected_request_sha256=request_sha256,
        )

    changed_preflight = {**preflight, "verifier_calls": True}
    changed_preflight["sha256"] = digest(
        {key: value for key, value in changed_preflight.items() if key != "sha256"}
    )
    changed_preflight_sha256 = write_json(prepared_root / "PREFLIGHT.json", changed_preflight)
    changed = deepcopy(receipt)
    changed["prepared_inputs"]["preflight_file_sha256"] = changed_preflight_sha256
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with monkeypatch.context() as changed_constant:
        changed_constant.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_PREFLIGHT_FILE_SHA256",
            changed_preflight_sha256,
        )
        with pytest.raises(ValueError, match="CPU preflight"):
            skyrl_training._validate_engine_diagnostic_receipt(
                changed,
                config_file_sha256=config_sha,
                expected_plan_sha256=plan_sha256,
                expected_request_sha256=request_sha256,
            )
    write_json(prepared_root / "PREFLIGHT.json", preflight)

    changed_journal = deepcopy(journal)
    changed_journal[0]["request_sha256"] = "0" * 64
    journal_path.write_text("\n".join(json.dumps(row) for row in changed_journal) + "\n")
    changed = deepcopy(receipt)
    changed_journal_sha256 = "sha256:" + hashlib.sha256(journal_path.read_bytes()).hexdigest()
    changed["prepared_inputs"]["submission_journal_file_sha256"] = changed_journal_sha256
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with monkeypatch.context() as changed_constant:
        changed_constant.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_SUBMISSION_JOURNAL_FILE_SHA256",
            changed_journal_sha256,
        )
        with pytest.raises(ValueError, match="submission journal"):
            skyrl_training._validate_engine_diagnostic_receipt(
                changed,
                config_file_sha256=config_sha,
                expected_plan_sha256=plan_sha256,
                expected_request_sha256=request_sha256,
            )

    changed_journal = deepcopy(journal)
    changed_journal[0]["manifest_sha256"] = "0" * 64
    journal_path.write_text("\n".join(json.dumps(row) for row in changed_journal) + "\n")
    changed = deepcopy(receipt)
    changed_journal_sha256 = "sha256:" + hashlib.sha256(journal_path.read_bytes()).hexdigest()
    changed["prepared_inputs"]["submission_journal_file_sha256"] = changed_journal_sha256
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with monkeypatch.context() as changed_constant:
        changed_constant.setattr(
            skyrl_training,
            "ENGINE_DIAGNOSTIC_SUBMISSION_JOURNAL_FILE_SHA256",
            changed_journal_sha256,
        )
        with pytest.raises(ValueError, match="submission journal"):
            skyrl_training._validate_engine_diagnostic_receipt(
                changed,
                config_file_sha256=config_sha,
                expected_plan_sha256=plan_sha256,
                expected_request_sha256=request_sha256,
            )


def test_cluster_target_is_immutable_and_request_never_requeues(skyrl_prepared) -> None:
    plan = deepcopy(skyrl_prepared.plan)
    plan["execution"]["cluster_target"] = "dev"
    request = skyrl_training.job_request(plan)
    cli._require_prepared_cluster(plan, cli.Cluster.dev)
    with pytest.raises(JobsError, match="dev-cluster-only"):
        cli._require_prepared_cluster(plan, cli.Cluster.prod)
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["fleet-api", "wandb-api"]
    assert request["env"]["WANDB_MODE"] == "online"


@pytest.mark.parametrize(
    ("surface", "key", "value"),
    [
        ("arguments", "lr", 2e-6),
        ("arguments", "seed", 43),
        ("arguments", "context_tokens", 65536),
        ("arguments", "response_tokens", 65535),
        ("arguments", "tokens_per_turn", 2048),
        ("arguments", "engine_start_timeout_seconds", 1799),
        ("arguments", "engine_cleanup_timeout_seconds", 299),
        ("execution", "priority", "c2"),
        ("execution", "resources", {**skyrl_training.REWARD_CANARY_RESOURCES, "cpu_limit": "63"}),
        ("execution", "runtime_user", {"uid": 0, "gid": 0, "run_as_non_root": False}),
        ("model", "revision", "0" * 40),
    ],
)
def test_reward_canary_classification_covers_complete_reviewed_recipe(
    skyrl_prepared, surface: str, key: str, value: object
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    assert skyrl_training.is_reward_canary(plan)
    plan[surface][key] = value
    if surface == "arguments":
        with suppress(ValueError):
            plan["native_overrides"] = skyrl.overrides(skyrl.SkyRLConfig(**plan["arguments"]))
    with pytest.raises(ValueError, match="complete exact reviewed recipe"):
        skyrl_training.is_reward_canary(plan)


def test_reward_canary_classification_rejects_native_override_drift(skyrl_prepared) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    plan["native_overrides"]["trainer.max_training_steps"] = 2
    with pytest.raises(ValueError, match="complete exact reviewed recipe"):
        skyrl_training.is_reward_canary(plan)


def test_reward_canary_classification_binds_every_model_file(skyrl_prepared) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    plan["model"]["files"][0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="complete exact reviewed recipe"):
        skyrl_training.is_reward_canary(plan)


def test_reward_canary_uses_cpu_qualified_worker_rpc_relay_image(skyrl_prepared) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    skyrl_training._require_engine_start_qualified_image(plan)
    assert plan["execution"]["image"] == skyrl_training.IMAGE
    assert plan["execution"]["image_cpu_qualification"] == (
        skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION
    )
    assert skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION["source_commit"] == (
        "8d62868d6dc00eee793d83efe5738dc21e42758d"
    )


def test_reward_canary_default_user_fallback_binds_exact_image_and_receipt() -> None:
    payload = DEFAULT_USER_EVIDENCE.read_bytes()
    evidence = json.loads(payload)
    unsigned = {key: value for key, value in evidence.items() if key != "receipt_sha256"}
    qualification = cli._ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION

    assert hashlib.sha256(payload).hexdigest() == qualification["default_user_evidence_file_sha256"]
    assert (
        hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        ).hexdigest()
        == evidence["receipt_sha256"]
        == qualification["default_user_evidence_receipt_sha256"]
    )
    assert qualification["image"] == skyrl_training.IMAGE
    assert evidence["requested_image"] == skyrl_training.IMAGE
    assert evidence["runtime"]["runtime_image_id"] == skyrl_training.IMAGE
    assert evidence["runtime"]["pod_security_context"] == {}
    assert evidence["runtime"]["container_security_context"] is None
    assert (evidence["runtime"]["effective_uid"], evidence["runtime"]["effective_gid"]) == (
        1000,
        100,
    )


@pytest.mark.parametrize("missing", [None, "runAsUser", "runAsGroup", "runAsNonRoot"])
def test_reward_canary_preview_accepts_only_absent_identity_fields(
    skyrl_prepared, monkeypatch, missing: str | None
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    expected = {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
    pod_context = (
        _OMITTED
        if missing is None
        else {key: value for key, value in expected.items() if key != missing}
    )

    result = cli._validate_reward_canary_preview(
        plan,
        request,
        reward_preview_with_contexts(pod_context=pod_context),
    )

    assert result["runtime_user"] == {"uid": 1000, "gid": 100}
    assert result["pods"] == 1
    assert result["runtime_user_evidence"] == {
        "mode": "qualified_image_default_with_gpu_entrypoint_recheck",
        "qualification_receipt_sha256": (
            "d1080784334becc76edb698ce363ecc513e5c98f9fce6c9510de59136192129f"
        ),
        "omitted_preview_fields": 3 if missing is None else 1,
        "explicit_conflicts_rejected": True,
    }


@pytest.mark.parametrize(
    ("pod_context", "container_context"),
    [
        ({"runAsUser": 0}, _OMITTED),
        (_OMITTED, {"runAsGroup": 0}),
        ({"runAsNonRoot": False}, _OMITTED),
        (_OMITTED, {"runAsUser": None}),
        ({"runAsUser": True}, _OMITTED),
        (_OMITTED, {"runAsNonRoot": 1}),
        ({"runAsUser": 0}, {"runAsUser": 1000}),
    ],
)
def test_reward_canary_default_user_fallback_rejects_every_explicit_conflict(
    skyrl_prepared, monkeypatch, pod_context: object, container_context: object
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)

    with pytest.raises(JobsError, match="explicit runtime user conflict"):
        cli._validate_reward_canary_preview(
            plan,
            request,
            reward_preview_with_contexts(
                pod_context=pod_context,
                container_context=container_context,
            ),
        )


def test_reward_canary_absent_context_requires_its_exact_qualification(
    skyrl_prepared, monkeypatch, tmp_path: Path
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    changed = json.loads(DEFAULT_USER_EVIDENCE.read_bytes())
    changed["requested_image"] = changed["requested_image"].replace("89758df2", "09758df2")
    changed_path = tmp_path / "changed-default-user-evidence.json"
    changed_path.write_text(json.dumps(changed))
    monkeypatch.setitem(
        cli._ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION,
        "default_user_evidence_path",
        str(changed_path),
    )

    with pytest.raises(JobsError, match="exact qualified image"):
        cli._validate_reward_canary_preview(
            plan,
            request,
            reward_preview_with_contexts(),
        )


def test_reward_canary_absent_context_does_not_waive_exact_request_image(
    skyrl_prepared, monkeypatch
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    request["image"] = request["image"].replace("89758df2", "09758df2")

    with pytest.raises(JobsError, match="request differs from its exact plan"):
        cli._validate_reward_canary_preview(
            plan,
            request,
            reward_preview_with_contexts(),
        )


def test_reward_canary_explicit_context_does_not_use_default_user_fallback(
    skyrl_prepared, monkeypatch, tmp_path: Path
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    monkeypatch.setitem(
        cli._ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION,
        "default_user_evidence_path",
        str(tmp_path / "absent.json"),
    )

    assert cli._validate_reward_canary_preview(
        plan,
        request,
        reward_preview(),
    ) == {"runtime_user": {"uid": 1000, "gid": 100}, "pods": 1}


def test_reward_canary_default_user_fallback_is_not_global(skyrl_prepared) -> None:
    assert (
        cli._validate_reward_canary_preview(
            skyrl_prepared.plan,
            {},
            reward_preview_with_contexts(),
        )
        == {}
    )


@pytest.mark.parametrize(
    ("uid", "gid", "nonroot"),
    [(0, 100, True), (1000, 0, True), (1000, 100, False)],
)
def test_reward_canary_preview_rejects_effective_root_or_wrong_group(
    skyrl_prepared, monkeypatch, uid: int, gid: int, nonroot: bool
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    with pytest.raises(JobsError, match="1000:100"):
        skyrl_training.validate_reward_canary_preview(
            plan,
            request,
            reward_preview(uid=uid, gid=gid, nonroot=nonroot),
        )


@pytest.mark.parametrize(("uid", "gid"), [(0, 100), (1000, 0)])
def test_reward_canary_gpu_process_checks_actual_uid_gid(
    skyrl_prepared, monkeypatch, uid: int, gid: int
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_UID", "1000")
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_GID", "100")
    monkeypatch.setattr(skyrl_training.os, "geteuid", lambda: uid)
    monkeypatch.setattr(skyrl_training.os, "getegid", lambda: gid)
    with pytest.raises(ValueError, match="GPU runtime.*1000:100"):
        skyrl_training._require_reward_canary_runtime_identity(plan)


@pytest.mark.parametrize(("uid", "gid"), [(0, 100), (1000, 0)])
def test_engine_diagnostic_gpu_process_checks_actual_uid_gid(
    skyrl_prepared, monkeypatch, uid: int, gid: int
) -> None:
    plan = skyrl_prepared.plan
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_UID", "1000")
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_GID", "100")
    monkeypatch.setattr(skyrl_training.os, "geteuid", lambda: uid)
    monkeypatch.setattr(skyrl_training.os, "getegid", lambda: gid)
    with pytest.raises(ValueError, match="diagnostic GPU runtime.*1000:100"):
        skyrl_training._require_engine_diagnostic_runtime_identity(plan)


@pytest.mark.parametrize(("uid", "gid"), [(0, 100), (1000, 0)])
def test_engine_diagnostic_preview_rejects_root_or_wrong_group(
    skyrl_prepared, uid: int, gid: int
) -> None:
    plan = skyrl_prepared.plan
    request = skyrl_training.engine_diagnostic_request(plan)
    with pytest.raises(JobsError, match="1000:100"):
        skyrl_training.validate_engine_diagnostic_preview(
            plan, request, diagnostic_preview(uid=uid, gid=gid)
        )


@pytest.mark.parametrize("command", ["preview", "submit"])
@pytest.mark.parametrize(("uid", "gid"), [(0, 100), (1000, 0)])
def test_engine_diagnostic_cli_fails_before_post_on_runtime_user_drift(
    tmp_path: Path,
    skyrl_prepared,
    monkeypatch,
    command: str,
    uid: int,
    gid: int,
) -> None:
    plan = skyrl_prepared.plan
    request = skyrl_training.engine_diagnostic_request(plan)
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "validate_preview", lambda *_: {})
    proof = {
        "schema": "cyber_skyrl_engine_diagnostic_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    proof["sha256"] = digest(proof)
    monkeypatch.setattr(cli, "_read", lambda _: proof)
    submitted = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def preview(self, value):
            assert value == request
            return diagnostic_preview(uid=uid, gid=gid)

        def submit_once(self, *args):
            submitted.append(args)
            return {"status": "unexpected"}

    monkeypatch.setattr(cli, "_client", lambda _: Client())
    result = CliRunner().invoke(cli.app, [command, str(tmp_path)])
    assert result.exit_code == 2
    assert "1000:100" in result.stderr
    assert submitted == []


@pytest.mark.parametrize("command", ["preview", "submit"])
@pytest.mark.parametrize(("uid", "gid"), [(0, 100), (1000, 0)])
def test_reward_canary_cli_preview_and_submit_fail_before_post_on_runtime_user_drift(
    tmp_path: Path,
    skyrl_prepared,
    monkeypatch,
    command: str,
    uid: int,
    gid: int,
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    allow_synthetic_reward_bindings(monkeypatch)
    request = skyrl_training.job_request(plan)
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_skyrl_mode", lambda *_: "training")
    monkeypatch.setattr(cli, "validate_preview", lambda *_: {})
    proof = {
        "schema": "cyber_skyrl_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    proof["sha256"] = digest(proof)
    monkeypatch.setattr(cli, "_read", lambda _: proof)
    submitted = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def preview(self, value):
            assert value == request
            return reward_preview(uid=uid, gid=gid)

        def submit_once(self, *args):
            submitted.append(args)
            return {"status": "unexpected"}

    monkeypatch.setattr(cli, "_client", lambda _: Client())
    result = CliRunner().invoke(cli.app, [command, str(tmp_path)])
    assert result.exit_code == 2
    assert "1000:100" in result.stderr
    assert submitted == []


def test_fabricated_self_digested_version_receipt_cannot_replace_get_observation(
    tmp_path: Path, monkeypatch
) -> None:
    source = load(ROOT / "configs/data/qwen38-rl-reward-canary-source-version-receipt-v1.json")
    source["schema"] = rl_reward_canary.VERSION_RECEIPT_SCHEMA
    source["sha256"] = "sha256:" + digest(
        {key: value for key, value in source.items() if key != "sha256"}
    )
    path = tmp_path / "fabricated.json"
    path.write_text(json.dumps(source, sort_keys=True))
    binding = {
        "path": path.name,
        "file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "self_sha256": source["sha256"],
        "authority_payload_sha256": source["authority_payload_sha256"],
    }
    monkeypatch.setattr(rl_reward_canary, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="retained sanitized authoritative GET observation"):
        rl_reward_canary._version_receipt(
            binding,
            evidence_path=tmp_path / "evidence.json",
            role="eligible_source",
            task_key=source["task_key"],
            task_version_id=source["task_version_id"],
        )


def test_complete_fabricated_receipt_observation_and_files_cannot_enable_source(
    tmp_path: Path, monkeypatch
) -> None:
    source = load(ROOT / "configs/data/qwen38-rl-reward-canary-source-version-receipt-v1.json")
    request = {
        "method": "GET",
        "url": f"https://orchestrator.fleetai.com/v1/tasks/{source['task_key']}",
        "query": {"version_id": source["task_version_id"]},
    }
    fabricated_response = {"schema": "fabricated_response_v1", "fields": {"matching": True}}
    response_path = tmp_path / "fabricated-response.json"
    response_path.write_bytes(
        json.dumps(fabricated_response, sort_keys=True, separators=(",", ":")).encode()
    )
    fabricated_journal = {
        "schema": "fabricated_journal_v1",
        "request": request,
        "status_code": 200,
    }
    journal_path = tmp_path / "fabricated-journal.json"
    journal_path.write_bytes(
        json.dumps(fabricated_journal, sort_keys=True, separators=(",", ":")).encode()
    )
    observation = {
        "schema": rl_reward_canary.VERSION_OBSERVATION_SCHEMA,
        "authority": "https://orchestrator.fleetai.com",
        "request": request,
        "status_code": 200,
        "response_body_sha256": source["authority_payload_sha256"],
        "observed_at": "2026-09-12T00:00:00Z",
        "task_key": source["task_key"],
        "task_version_id": source["task_version_id"],
        "sanitized_response": {
            "path": response_path.name,
            "file_sha256": "sha256:" + hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "self_sha256": "sha256:" + "2" * 64,
        },
        "acquisition_journal": {
            "path": journal_path.name,
            "file_sha256": "sha256:" + hashlib.sha256(journal_path.read_bytes()).hexdigest(),
            "self_sha256": "sha256:" + "4" * 64,
        },
    }
    observation["sha256"] = "sha256:" + digest(observation)
    observation_path = tmp_path / "fabricated-observation.json"
    observation_path.write_text(json.dumps(observation, sort_keys=True))
    observation_binding = {
        "path": observation_path.name,
        "file_sha256": "sha256:" + hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        "self_sha256": observation["sha256"],
    }
    source.update(
        schema=rl_reward_canary.VERSION_RECEIPT_SCHEMA,
        authority_observation=observation_binding,
    )
    source["sha256"] = "sha256:" + digest(
        {key: value for key, value in source.items() if key != "sha256"}
    )
    receipt_path = tmp_path / "fabricated-receipt.json"
    receipt_path.write_text(json.dumps(source, sort_keys=True))
    binding = {
        "path": receipt_path.name,
        "file_sha256": "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "self_sha256": source["sha256"],
        "authority_payload_sha256": source["authority_payload_sha256"],
        "authority_observation": observation_binding,
    }
    monkeypatch.setattr(rl_reward_canary, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        rl_reward_canary._version_receipt(
            binding,
            evidence_path=tmp_path / "evidence.json",
            role="eligible_source",
            task_key=source["task_key"],
            task_version_id=source["task_version_id"],
        )


def test_reward_acceptance_does_not_trust_operator_booleans(tmp_path: Path, monkeypatch) -> None:
    config = {
        "task": {"key": "task", "version_id": "version"},
        "verifier": {"version_id": "verifier"},
        "authority": {"required_cyber_contract": {"verifier_contract": "v3"}},
    }
    with pytest.raises(ValueError, match="direct-authority receipt"):
        skyrl_training._validate_authoritative_reward(
            config,
            {
                "reward": True,
                "authoritative": True,
                "verifier_ran": True,
                "nonzero": True,
            },
        )

    plan = {
        "output_root": str(tmp_path),
        "data": {},
        "native_overrides": {
            "trainer.resume_mode": "none",
            "trainer.max_training_steps": 1,
        },
    }
    monkeypatch.setattr(skyrl_training, "is_reward_canary", lambda _: True)
    monkeypatch.setattr(skyrl_training, "_reward_canary_source_rows", lambda _: {})
    with pytest.raises(ValueError, match="batch directory is missing"):
        skyrl_training.reward_canary_episode_audit(plan)

    ranks = [
        {
            "rank": rank,
            "policy_state_sha256": "sha256:" + str(rank) * 64,
            "optimizer_state_sha256": "sha256:" + str(rank) * 64,
            "optimizer_states": 1,
            "optimizer_step_states": 1,
            "optimizer_step": 1,
            "scheduler_last_epoch": 1,
        }
        for rank in range(8)
    ]
    ranks[0]["optimizer_states"] = 0
    manifest = {
        "source_plan": plan,
        "optimizer_update_independently_verified": True,
        "checkpoint": {"step": 1, "ranks": ranks},
    }
    monkeypatch.setattr(
        "training.skyrl_rl_checkpoint.verify_manifest", lambda *_args, **_kwargs: None
    )
    with pytest.raises(ValueError, match="exactly one optimizer update"):
        skyrl_training.reward_canary_update_proof(plan, tmp_path / "manifest.json", manifest)


def _write_base_policy_rank(plan: dict, rank: int, policy_sha256: str) -> Path:
    value = {
        "schema": skyrl_training.REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
        "source_plan_sha256": digest(plan),
        "model_root": plan["model"]["root"],
        "model_revision": plan["model"]["revision"],
        "weight_manifest_sha256": plan["model"]["weight_manifest_sha256"],
        "rank": rank,
        "world_size": 8,
        "initial_optimizer_states": 0,
        "policy_state_sha256": policy_sha256,
    }
    value["sha256"] = digest(value)
    path = Path(plan["output_root"]) / "reward_canary_base_policy" / f"rank-{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_optimizer_counters_cannot_replace_semantic_base_to_checkpoint_delta(
    tmp_path: Path, skyrl_prepared, monkeypatch
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    plan["output_root"] = str(tmp_path)
    ranks = [
        {
            "rank": rank,
            "policy_state_sha256": "sha256:" + f"{rank:x}" * 64,
            "optimizer_state_sha256": "sha256:" + "a" * 64,
            "optimizer_states": 1,
            "optimizer_step_states": 1,
            "optimizer_step": 1,
            "scheduler_last_epoch": 1,
        }
        for rank in range(8)
    ]
    manifest = {
        "source_plan": plan,
        "checkpoint": {"step": 1, "ranks": ranks},
    }
    manifest["sha256"] = digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    monkeypatch.setattr(skyrl_training, "is_reward_canary", lambda _: True)
    monkeypatch.setattr(
        "training.skyrl_rl_checkpoint.verify_manifest", lambda *_args, **_kwargs: None
    )

    with pytest.raises(ValueError, match="missing or indirect"):
        skyrl_training.reward_canary_update_proof(plan, manifest_path, manifest)

    for rank, checkpoint_rank in enumerate(ranks):
        _write_base_policy_rank(plan, rank, checkpoint_rank["policy_state_sha256"])
    with pytest.raises(ValueError, match="no semantic policy delta"):
        skyrl_training.reward_canary_update_proof(plan, manifest_path, manifest)

    _write_base_policy_rank(plan, 0, "sha256:" + "f" * 64).write_text(
        json.dumps(
            _sealed_receipt(
                {
                    "schema": skyrl_training.REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
                    "source_plan_sha256": digest(plan),
                    "model_root": plan["model"]["root"],
                    "model_revision": plan["model"]["revision"],
                    "weight_manifest_sha256": plan["model"]["weight_manifest_sha256"],
                    "rank": 0,
                    "world_size": 8,
                    "initial_optimizer_states": 0,
                    "policy_state_sha256": "sha256:" + "f" * 64,
                }
            ),
            sort_keys=True,
        )
    )
    proof = skyrl_training.reward_canary_update_proof(plan, manifest_path, manifest)
    assert proof["optimizer_updates"] == 1
    assert proof["changed_policy_ranks"] == 1
    assert [row["rank"] for row in proof["base_policy_ranks"]] == list(range(8))
    assert proof["ranks"][0]["policy_changed"] is True
    assert all(not row["policy_changed"] for row in proof["ranks"][1:])


def test_policy_semantic_digest_matches_checkpoint_validator() -> None:
    import torch

    from training import skyrl_rl_checkpoint

    state = {
        "b": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "a": {3: torch.tensor([4], dtype=torch.int64)},
    }
    assert skyrl_training._policy_state_digest(state) == skyrl_rl_checkpoint._policy_state_digest(
        state
    )


def _sealed_receipt(value: dict) -> dict:
    return {**value, "sha256": digest(value)}


def _wandb_evidence(plan: dict) -> tuple[dict, dict, list[dict]]:
    episode = _sealed_receipt(
        {
            "schema": skyrl_training.REWARD_CANARY_EPISODE_AUDIT_SCHEMA,
            "source_plan_sha256": digest(plan),
            "train_reward_mean": 0.375,
            "train_nonzero_count": 3,
            "train_reward_population_variance": 0.109375,
        }
    )
    base_policy_ranks = [
        {
            "rank": rank,
            "path": f"/evidence/base/rank-{rank}.json",
            "file_sha256": "sha256:" + "a" * 64,
            "receipt_self_sha256": "sha256:" + "b" * 64,
            "policy_state_sha256": "sha256:" + "c" * 64,
        }
        for rank in range(8)
    ]
    update_ranks = [
        {
            "rank": rank,
            "policy_state_sha256": "sha256:" + ("d" if rank == 0 else "c") * 64,
            "base_policy_state_sha256": "sha256:" + "c" * 64,
            "checkpoint_policy_state_sha256": "sha256:" + ("d" if rank == 0 else "c") * 64,
            "policy_changed": rank == 0,
        }
        for rank in range(8)
    ]
    update = _sealed_receipt(
        {
            "schema": skyrl_training.REWARD_CANARY_UPDATE_PROOF_SCHEMA,
            "source_plan_sha256": digest(plan),
            "initial_optimizer_step": 0,
            "final_optimizer_step": 1,
            "optimizer_updates": 1,
            "base_policy_ranks": base_policy_ranks,
            "changed_policy_ranks": 1,
            "policy_delta_basis": (
                "native_rank_state_digest_before_first_optim_step_vs_sealed_step1_checkpoint"
            ),
            "ranks": update_ranks,
        }
    )
    history = [
        {
            "_step": 0,
            "eval/all/avg_score": 0.0,
            "cyber/optimizer_step": 0.0,
        },
        {
            "_step": 1,
            "trainer/global_step": 1.0,
            "reward/avg_raw_reward": 0.375,
            "cyber/train/episodes": 8.0,
            "cyber/train/reward_mean": 0.375,
            "cyber/train/reward_nonzero_count": 3.0,
            "cyber/train/reward_population_variance": 0.109375,
            "cyber/optimizer_step": 1.0,
            "policy/policy_loss": 0.01,
            "policy/policy_lr": 1e-6,
            "policy/grad_norm": 0.2,
        },
    ]
    return episode, update, history


def _wandb_run(plan: dict, history: list[dict], *, artifacts=(), files=()):
    args = plan["arguments"]
    return SimpleNamespace(
        entity=args["wandb_entity"],
        project=args["wandb_project"],
        id=args["wandb_run_id"],
        name=args["name"],
        state="finished",
        config={"plan_sha256": digest(plan), "arguments": args},
        summary={key: value for row in history for key, value in row.items()},
        scan_history=lambda: iter(history),
        logged_artifacts=lambda: iter(artifacts),
        files=lambda: [SimpleNamespace(name=name) for name in files],
    )


def _local_history(history: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                {
                    "optimizer_step": row["_step"],
                    "time": 1.0 + row["_step"],
                    **{key: value for key, value in row.items() if key != "_step"},
                }
            )
            for row in history
        )
        + "\n"
    ).encode()


def test_wandb_terminal_history_is_scalar_only_exact_and_cross_bound(
    skyrl_prepared, monkeypatch
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    episode, update, history = _wandb_evidence(plan)
    payload = _local_history(history)
    monkeypatch.setattr(
        skyrl_training,
        "_snapshot",
        lambda path: (payload, "sha256:" + hashlib.sha256(payload).hexdigest()),
    )
    result = skyrl_training.reward_canary_wandb_history(
        plan,
        episode,
        update,
        api=SimpleNamespace(run=lambda _: _wandb_run(plan, history)),
    )
    assert result["logged_artifact_count"] == result["rich_payload_count"] == 0
    assert result["required_step_1_scalars"]["cyber/train/episodes"] == 8.0


@pytest.mark.parametrize(
    "fault",
    [
        "junk_only",
        "reward_mismatch",
        "missing_optimizer",
        "missing_policy",
        "rich",
        "artifact",
        "episode_file",
        "trace_file",
    ],
)
def test_wandb_terminal_rejects_incomplete_or_non_scalar_evidence(
    skyrl_prepared, monkeypatch, fault: str
) -> None:
    plan = exact_reward_plan(skyrl_prepared)
    episode, update, history = _wandb_evidence(plan)
    artifacts = ()
    files = ()
    if fault == "junk_only":
        history[1] = {"_step": 1, "junk": 0.0}
    elif fault == "reward_mismatch":
        history[1]["reward/avg_raw_reward"] = 0.5
        history[1]["cyber/train/reward_mean"] = 0.5
    elif fault == "missing_optimizer":
        history[1].pop("cyber/optimizer_step")
    elif fault == "missing_policy":
        history[1].pop("policy/grad_norm")
    elif fault == "rich":
        history[1]["private_table"] = {"columns": ["secret"]}
    elif fault == "artifact":
        artifacts = (SimpleNamespace(name="private"),)
    elif fault == "episode_file":
        files = ("episodes.jsonl",)
    elif fault == "trace_file":
        files = ("trace.txt",)
    payload = _local_history(history)
    monkeypatch.setattr(
        skyrl_training,
        "_snapshot",
        lambda path: (payload, "sha256:" + hashlib.sha256(payload).hexdigest()),
    )
    with pytest.raises(ValueError):
        skyrl_training.reward_canary_wandb_history(
            plan,
            episode,
            update,
            api=SimpleNamespace(
                run=lambda _: _wandb_run(plan, history, artifacts=artifacts, files=files)
            ),
        )


def test_reload_gate_is_digest_bound_and_deliberately_not_launchable() -> None:
    reload = load(RELOAD)
    assert_self_digest(reload)
    assert reload["schema"] == "cyber_skyrl_rl_reload_config_template_v2"
    assert reload["status"] == "waiting_for_runtime_evidence"
    assert reload["launchable"] is False
    resolved = reload["resolved_config"]
    assert resolved == {
        "schema": "cyber_skyrl_rl_reload_config_v1",
        "name": "chris-q38-rlreward-reload-dev1",
        "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-reload-dev1",
        "source": {
            "run_name": load(RUN)["name"],
            "plan_sha256": None,
            "request_sha256": None,
            "checkpoint_global_step": 1,
            "checkpoint_manifest_path": None,
            "checkpoint_manifest_file_sha256": None,
            "checkpoint_manifest_self_sha256": None,
            "reward_terminal_receipt_path": (
                "/mnt/sfs/jobs/chris-q38-rlreward-dev1/REWARD_CANARY_TERMINAL.json"
            ),
            "reward_terminal_receipt_file_sha256": None,
            "reward_terminal_receipt_self_sha256": None,
            "source_external_release_path": (
                "/mnt/sfs/jobs/chris-q38-rlreward-dev1/SOURCE_RELEASE.json"
            ),
            "source_external_release_file_sha256": None,
            "source_external_release_self_sha256": None,
        },
        "cluster": {
            "target": "dev",
            "priority": "c1",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "64",
                "memory_request": "512Gi",
                "memory_limit": "768Gi",
            },
        },
        "lifecycle": {"startup_timeout_seconds": 1800, "cleanup_timeout_seconds": 300},
    }
    assert "source" not in reload
    assert reload["implementation"]["reward_terminal_schema"] == (
        "cyber_skyrl_rl_reward_canary_terminal_v1"
    )
    fixed = reload["fixed_contract"]
    assert fixed["cluster"] == "dev" and fixed["priority_value"] == 10000
    assert fixed["workers"] == 1 and fixed["gpus_per_worker"] == 8
    assert fixed["requeue_if_preempted"] is False and fixed["secrets"] == []
    assert fixed["wandb"] is False
    assert all(fixed[key] == 0 for key in fixed if key.endswith("_authorized"))
    assert len(reload["unresolved_before_materialization"]) == 6
