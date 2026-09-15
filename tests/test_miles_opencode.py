import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from training import miles_opencode as opencode
from training.rl_episode import InvalidEpisode


def _config():
    value = {
        "harness": opencode.harness_contract(),
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
            "tito_family": opencode.TITO_FAMILY,
            "runtime_chat_template_sha256": "sha256:" + opencode.TEMPLATE_SHA256,
        },
        "environment": {"ttl_seconds": 32400},
    }
    return value


def _forest(lengths):
    nodes = []
    leaves = []
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
        offset += length
    samples = [SimpleNamespace(metadata={"leaf": {"node_id": row["node_id"]}}) for row in leaves]
    metadata = {
        "tree": {"nodes": nodes, "leaves": leaves},
        "agent": {
            "reward": 0.5,
            "cyber_compaction": {
                "schema": "cyber_miles_opencode_compaction_v1",
                "primary_model_calls": len(nodes),
                "summary_calls": len(lengths) - 1,
                "expected_segments": len(lengths),
                "summary_token_treatment": opencode.SUMMARY_TOKEN_TREATMENT,
                "session_node_cap": 4096,
            },
        },
    }
    return samples, metadata


def test_settings_bind_exact_native_compaction_without_tool_prefix_truncation():
    config = _config()
    rendered = opencode.settings(
        config,
        primary_base_url="http://primary/sessions/" + "a" * 32,
        summary_base_url="http://summary/sessions/" + "b" * 32,
        mcp_url="http://challenge/mcp",
        runner_header="X-Runner-Auth",
    )
    assert opencode.COMPACTION_THRESHOLD_TOKENS == 176608
    assert rendered["provider"]["fleet-primary"]["models"]["model"]["limit"] == {
        "context": 262144,
        "input": 229376,
        "output": 32768,
    }
    assert rendered["provider"]["fleet-summary"]["models"]["model"]["limit"]["output"] == 4096
    assert rendered["compaction"] == {
        "auto": True,
        "reserved": 52768,
        "preserve_recent_tokens": 15000,
    }
    assert rendered["agent"]["build"]["steps"] == 2048
    assert "tool_result_chars" not in str(rendered)


def test_opencode_child_environment_excludes_parent_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("FLEET_API_KEY", "must-not-cross")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-cross")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross")
    monkeypatch.setenv("PATH", "/qualified/bin")
    result = opencode._child_env(
        home=tmp_path / "home",
        prompt_path=tmp_path / "prompt",
        workspace=tmp_path / "workspace",
        model="fleet-primary/model",
        runner_token="runner-only",
    )
    assert result["PATH"] == "/qualified/bin"
    assert result["OPENAI_API_KEY"] == "session-only"
    assert result["CYBER_OPENCODE_WORKSPACE"] == str(tmp_path / "workspace")
    assert result["CYBER_RUNNER_AUTH_TOKEN"] == "runner-only"
    assert not {"FLEET_API_KEY", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY"}.intersection(result)


def test_summary_evidence_rejects_truncation_or_impossible_count():
    opencode._validate_summary_nodes([{"id": 0, "seq": 0, "parent": None, "truncated": False}], 2)
    with pytest.raises(InvalidEpisode, match="summary_session_tree_invalid"):
        opencode._validate_summary_nodes(
            [{"id": 0, "seq": 0, "parent": None, "truncated": True}], 2
        )
    with pytest.raises(InvalidEpisode, match="summary_model_request_count_invalid"):
        opencode._validate_summary_nodes(
            [{"id": 0, "seq": 0, "parent": None, "truncated": False}], 1
        )


def test_picker_keeps_repeated_compaction_segments_past_old_1024_cap():
    samples, metadata = _forest([370, 370, 365])
    assert opencode.pick_compaction_segments(samples, metadata) == samples


@pytest.mark.parametrize("mutation", ["shared", "truncated", "missing"])
def test_picker_rejects_branch_retry_truncation_or_missing_node(mutation):
    samples, metadata = _forest([3, 2])
    if mutation == "shared":
        metadata["tree"]["leaves"][1]["path_node_ids"] = [0, 3, 4]
    elif mutation == "truncated":
        metadata["tree"]["nodes"][1]["truncated"] = True
    else:
        metadata["tree"]["leaves"][1]["path_node_ids"] = [3]
    with pytest.raises(ValueError):
        opencode.pick_compaction_segments(samples, metadata)


def test_postprocessor_broadcasts_one_reward_and_preserves_one_objective(monkeypatch):
    samples, metadata = _forest([2, 2, 2])
    for sample in samples:
        sample.reward = 0.5
        sample.response_length = 2
        sample.loss_mask = [1, 0]

    package = ModuleType("miles.rollout.session.v2.postprocessor_hub.default_postprocess")
    package.default_postprocess = lambda values, _: values
    monkeypatch.setitem(
        sys.modules,
        "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        package,
    )
    result = opencode.postprocess_compaction_segments(samples, metadata)
    assert [sample.metadata["cyber_segment"]["index"] for sample in result] == [0, 1, 2]
    assert all(sample.reward == 0.5 and sum(sample.loss_mask) == 1 for sample in result)


def test_long_contract_rejects_any_context_or_summary_policy_drift():
    config = _config()
    opencode.validate_episode(config)
    for path, value in (
        (("rl", "context_tokens"), 98304),
        (("harness", "summary_token_treatment"), "trained"),
        (("harness", "session_node_cap"), 1024),
    ):
        changed = copy.deepcopy(config)
        changed[path[0]][path[1]] = value
        with pytest.raises(InvalidEpisode):
            opencode.validate_episode(changed)
