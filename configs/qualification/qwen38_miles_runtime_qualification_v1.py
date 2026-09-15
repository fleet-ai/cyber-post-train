"""In-image source gate and GPU parser gate for the exact Qwen3.8 Miles runtime."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from cyber_post_train.jobs import digest
from training import miles, miles_opencode, miles_training

CHECKPOINT_ROOT = Path("/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist")
CHECKPOINT_RECEIPT = Path(
    "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/NATIVE_CHECKPOINT.json"
)
CHECKPOINT_RECEIPT_FILE_SHA256 = (
    "b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c"
)
CHECKPOINT_RECEIPT_SHA256 = "19c8e93482530170e0f648815ab74233719e6f2b3bb7879a6564b42c3abec371"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def option(argv: list[str], name: str) -> str:
    return argv[argv.index("--" + name) + 1]


def forest(lengths: list[int]):
    nodes, leaves, samples = [], [], []
    offset = 0
    for length in lengths:
        path = []
        for index in range(length):
            node_id = offset + index
            nodes.append(
                {
                    "id": node_id,
                    "seq": node_id,
                    "parent": None if index == 0 else node_id - 1,
                    "truncated": False,
                }
            )
            path.append(node_id)
        leaves.append({"node_id": path[-1], "path_node_ids": path})
        sample = SimpleNamespace(
            metadata={"leaf": {"node_id": path[-1]}},
            reward=0.5,
            response_length=2,
            loss_mask=[1, 0],
        )
        samples.append(sample)
        offset += length
    metadata = {
        "tree": {"nodes": nodes, "leaves": leaves},
        "agent": {
            "reward": 0.5,
            "cyber_compaction": {
                "schema": "cyber_miles_opencode_compaction_v1",
                "primary_model_calls": len(nodes),
                "summary_calls": len(lengths) - 1,
                "expected_segments": len(lengths),
                "summary_token_treatment": miles_opencode.SUMMARY_TOKEN_TREATMENT,
                "session_node_cap": miles_opencode.SESSION_NODE_CAP,
            },
        },
    }
    return samples, metadata


def qualify() -> dict:
    source_commit = os.environ["QUALIFICATION_SOURCE_COMMIT"]
    image = os.environ["QUALIFICATION_IMAGE"]
    build_source_sha256 = os.environ["QUALIFICATION_BUILD_SOURCE_SHA256"]
    parse_native = os.environ.get("QUALIFICATION_NATIVE_PARSE") == "1"
    model_root = Path(
        os.environ.get(
            "QUALIFICATION_MODEL_ROOT", "/mnt/sfs/models/q38-long-runtime-qualification/hf"
        )
    )
    assert re.fullmatch(r"[a-f0-9]{40}", source_commit)
    assert re.fullmatch(r"[^@\s]+@sha256:[a-f0-9]{64}", image)
    assert re.fullmatch(r"sha256:[a-f0-9]{64}", build_source_sha256)
    assert model_root.is_absolute()
    model_config_sha256 = None
    if parse_native:
        model_config_sha256 = sha256(model_root / "config.json")
        assert (
            model_config_sha256
            == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
        )
        assert CHECKPOINT_ROOT.is_dir()
        assert sha256(CHECKPOINT_RECEIPT) == CHECKPOINT_RECEIPT_FILE_SHA256
        checkpoint_receipt = json.loads(CHECKPOINT_RECEIPT.read_text())
        assert checkpoint_receipt["sha256"] == CHECKPOINT_RECEIPT_SHA256
        assert checkpoint_receipt["root"] == str(CHECKPOINT_ROOT)
        assert checkpoint_receipt["optimizer_steps"] == 0
    from miles.utils.external_utils.command_utils import repo_base_dir

    miles_root = Path(repo_base_dir)
    assert sha256(miles_root / "train.py") == miles.LONG_NATIVE_DRIVER_SHA256
    assert (
        sha256(miles_root / "tools/convert_hf_to_torch_dist.py")
        == miles.LONG_NATIVE_CONVERTER_SHA256
    )
    import miles.rollout.session.v2.tree_trajectory as tree_module

    tree_path = Path(inspect.getsourcefile(tree_module) or "")
    assert sha256(tree_path) == miles.LONG_INSTALLED_SESSION_TREE_SHA256
    tito_path = miles_root / "miles/utils/chat_template_utils/tito_tokenizer.py"
    assert sha256(tito_path) == "72650e3b337d69d237088c03cafa12b066a2c31fe1ffd96fab2d49d832f4a33c"

    binary = Path(miles_opencode.OPENCODE_BINARY)
    assert sha256(binary) == miles_opencode.OPENCODE_BINARY_SHA256
    version = subprocess.run(
        [str(binary), "--version"], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    assert version == miles_opencode.OPENCODE_VERSION

    config = miles.MilesConfig(
        name="q38-long-runtime-qualification",
        output_root="/mnt/sfs/jobs/q38-long-runtime-qualification",
        model_root=str(model_root),
        torch_dist_root=str(CHECKPOINT_ROOT),
        train_data="/mnt/sfs/data/q38-long-runtime-qualification/train.jsonl",
        dev_data="/mnt/sfs/data/q38-long-runtime-qualification/dev.jsonl",
        data_manifest="/mnt/sfs/data/q38-long-runtime-qualification/manifest.json",
        wandb_entity="thefleet",
        wandb_project="cyber-post",
        wandb_run_id="q38-long-runtime-qualification",
        nodes=4,
        gpus_per_node=8,
        steps=1,
        groups=1,
        samples_per_prompt=2,
        lr=1e-6,
        temperature=0.7,
        kl_loss_coef=0.001,
        max_tokens_per_gpu=65536,
        eval_interval=1,
        checkpoint_interval=1,
        seed=42,
        context_tokens=262144,
        response_tokens=245760,
        tokens_per_turn=32768,
        native_profile="qwen3.8-27b-256k",
        harness="opencode",
        runtime_image=image,
        session_node_cap=4096,
    )
    argv = miles.arguments(config)
    expected = {
        "tensor-model-parallel-size": "8",
        "pipeline-model-parallel-size": "1",
        "context-parallel-size": "4",
        "actor-num-nodes": "4",
        "actor-num-gpus-per-node": "8",
        "rollout-max-context-len": "262144",
        "rollout-max-response-len": "245760",
        "max-tokens-per-gpu": "65536",
        "fleet-max-tokens-per-turn": "32768",
        "tito-model": "qwen38small",
        "fleet-tito-model": "qwen38small",
        "use-session-server": "v2",
        "custom-agent-function-path": "training.miles_opencode.run",
        "session-sample-picker-path": "training.miles_opencode.pick_compaction_segments",
        "session-sample-postprocessor-path": (
            "training.miles_opencode.postprocess_compaction_segments"
        ),
        "sglang-router-policy": "consistent_hashing",
        "fleet-session-node-cap": "4096",
        "load": str(CHECKPOINT_ROOT),
        "ref-load": str(CHECKPOINT_ROOT),
    }
    assert all(option(argv, key) == value for key, value in expected.items())
    assert "--chat-template-path" not in argv

    from miles.utils.chat_template_utils.tito_tokenizer import (
        resolve_fixed_chat_template,
        resolve_reasoning_and_tool_call_parser,
    )

    template, kwargs = resolve_fixed_chat_template("qwen38small")
    assert template and sha256(Path(template)) == miles.LONG_TITO_TEMPLATE_SHA256
    assert kwargs == {"preserve_thinking": True, "reasoning_effort": "xhigh"}
    assert resolve_reasoning_and_tool_call_parser("qwen38small") == (
        "qwen3",
        "qwen3_coder",
    )

    if parse_native:
        os.environ["MILES_USE_LEGACY_ROLLOUT_V1"] = "0"
        from miles.utils.arguments import parse_args

        previous = sys.argv
        try:
            sys.argv = [str(miles_root / "train.py"), *argv]
            parsed = parse_args()
        finally:
            sys.argv = previous
        assert parsed.tensor_model_parallel_size == 8
        assert parsed.context_parallel_size == 4
        assert parsed.use_session_server == "v2"
        assert parsed.max_seq_len == 262144
        assert parsed.tito_model == "qwen38small"
        assert parsed.load == parsed.ref_load == str(CHECKPOINT_ROOT)
        assert sha256(Path(parsed.chat_template_path)) == miles.LONG_TITO_TEMPLATE_SHA256
        assert parsed.sglang_router_policy == "consistent_hashing"

    harness_config = {
        "harness": miles_opencode.harness_contract(),
        "rl": {
            "max_turns": 2048,
            "episode_seconds": 28800,
            "tool_seconds": 300,
            "max_tokens_per_turn": 32768,
            "context_tokens": 262144,
            "response_tokens": 245760,
        },
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "served_id": "model",
            "tito_family": miles_opencode.TITO_FAMILY,
            "runtime_chat_template_sha256": "sha256:" + miles_opencode.TEMPLATE_SHA256,
        },
        "environment": {"ttl_seconds": 32400},
    }
    miles_opencode.validate_episode(harness_config)
    settings = miles_opencode.settings(
        harness_config,
        primary_base_url="http://primary/sessions/" + "a" * 32,
        summary_base_url="http://summary/sessions/" + "b" * 32,
        mcp_url="http://challenge/mcp",
        runner_header="X-Runner-Auth",
    )
    assert settings["compaction"] == {
        "auto": True,
        "reserved": 52768,
        "preserve_recent_tokens": 15000,
    }
    assert settings["agent"]["compaction"]["model"].startswith("fleet-summary/")
    assert "tool_result_chars" not in json.dumps(settings, sort_keys=True)

    tree = tree_module.SessionTree()
    parent = None
    for index in range(1105):
        parent = tree.create_node(
            parent,
            delta_messages=[{"role": "assistant", "content": str(index)}],
            token_ids=[index],
            completion_span=(0, 1),
            committed_at=float(index),
            response_id=str(index),
            record=object(),
            finish_reason="stop",
        )
    assert tree_module.MAX_NODES == 4096 and len(tree.nodes) == 1105

    samples, metadata = forest([370, 370, 365])
    assert miles_opencode.pick_compaction_segments(samples, metadata) == samples
    fake = ModuleType("miles.rollout.session.v2.postprocessor_hub.default_postprocess")
    fake.default_postprocess = lambda values, _: values
    module_name = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
    original = sys.modules.get(module_name)
    sys.modules[module_name] = fake
    try:
        processed = miles_opencode.postprocess_compaction_segments(samples, metadata)
    finally:
        if original is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = original
    assert len(processed) == 3
    assert all(sample.reward == 0.5 and sum(sample.loss_mask) == 1 for sample in processed)
    assert all(
        sample.metadata["cyber_segment"]["summary_token_treatment"]
        == "excluded_separate_session_v1"
        for sample in processed
    )

    checks = {key: True for key in miles_training.LONG_RUNTIME_CHECKS}
    checks["native_256k_parser_checked"] = parse_native
    if not parse_native:
        value = {
            "schema": "cyber_miles_opencode_source_qualification_v1",
            "status": "source_qualified_zero_gpu",
            "source_commit": source_commit,
            "qualification_script_sha256": "sha256:" + sha256(Path(__file__)),
            "image": image,
            "base_image": miles_training.LONG_RUNTIME_BASE_IMAGE,
            "build_source_sha256": build_source_sha256,
            "miles_source_commit": "9e178ca16839b0600155f3927f57ce0670b8f453",
            "miles_tito_backport_commit": "257992eb52bfa1f5248b5a5ae8f5a959be500788",
            "installed_tito_source_sha256": "sha256:" + sha256(tito_path),
            "checks": checks,
        }
        value["sha256"] = digest(value)
        return value
    value = {
        "schema": "cyber_miles_opencode_runtime_qualification_v1",
        "status": "image_qualified_for_dev",
        "source_commit": source_commit,
        "qualification_script_sha256": "sha256:" + sha256(Path(__file__)),
        "image": image,
        "base_image": miles_training.LONG_RUNTIME_BASE_IMAGE,
        "fti_version": "0.8.4",
        "native_profile": "qwen3.8-27b-256k",
        "opencode_version": "1.18.27",
        "opencode_source_commit": "4b7e19e315cca414121ba1d61523fef74bb3ae8b",
        "opencode_binary_sha256": "sha256:" + miles_opencode.OPENCODE_BINARY_SHA256,
        "miles_source_commit": "9e178ca16839b0600155f3927f57ce0670b8f453",
        "miles_tito_backport_commit": "257992eb52bfa1f5248b5a5ae8f5a959be500788",
        "installed_tito_source_sha256": "sha256:" + sha256(tito_path),
        "miles_tree_source_sha256": (
            "sha256:fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732"
        ),
        "native_driver_sha256": "sha256:" + miles.LONG_NATIVE_DRIVER_SHA256,
        "native_converter_sha256": "sha256:" + miles.LONG_NATIVE_CONVERTER_SHA256,
        "installed_session_tree_sha256": ("sha256:" + miles.LONG_INSTALLED_SESSION_TREE_SHA256),
        "build_source_sha256": build_source_sha256,
        "model_config_sha256": "sha256:" + str(model_config_sha256),
        "parser_checkpoint_root": str(CHECKPOINT_ROOT),
        "parser_checkpoint_receipt_file_sha256": (
            "sha256:" + CHECKPOINT_RECEIPT_FILE_SHA256
        ),
        "parser_checkpoint_receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
        "checks": checks,
    }
    value["sha256"] = digest(value)
    miles_training._validate_long_runtime_receipt(
        value, image=image, build_source_sha256=build_source_sha256
    )
    return value


def main() -> None:
    output = Path(sys.argv[1])
    receipt = qualify()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
