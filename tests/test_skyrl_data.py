"""SkyRL data uses the common reviewed split/GET boundary; no real Fleet calls."""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_rl_data import build, setup  # noqa: F401

from evals.fleet import opencode_self_hosted as fleet
from training import rl_data as data

NATIVE = data._native_skyrl


@pytest.fixture
def skyrl(setup, monkeypatch):  # noqa: F811
    setup.config["backend"] = "skyrl"
    tokenizer, _, _, identity = data._native(setup.lock, setup.config["model_root"])
    tokenizer.chat_template = "synthetic-skyrl-template"

    class Dataset:
        def __init__(self, path, tokenizer, budget, *, num_workers):
            assert num_workers == 1 and budget == 24576
            self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            if setup.drift == "drop":
                self.rows.pop()

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index].copy()
            prompt, env = row.pop("prompt"), row.pop("env_class")
            if setup.drift == "prompt":
                prompt = [{"role": "user", "content": "changed"}]
            elif setup.drift == "binding":
                row["cyber_config_json"] += " "
            elif setup.drift == "env":
                env = "changed"
            return prompt, env, row, str(index)

    monkeypatch.setattr(
        data, "_native_skyrl", lambda *args: (tokenizer, tokenizer, Dataset, identity)
    )
    return setup


def test_skyrl_get_only_data_and_token_identity(skyrl):
    result = build(skyrl)
    assert len(skyrl.calls) == 3 and result["submitted"] is False
    assert "Private" not in json.dumps(result)
    root = skyrl.tmp / "out"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "cyber_skyrl_data_v1"
    assert manifest["template_sha256"] == fleet.sha256(b"synthetic-skyrl-template")
    for split in ("train", "dev"):
        row = json.loads((root / (split + ".jsonl")).read_text())
        binding = json.loads(row["cyber_config_json"])
        assert row["split"] == split and row["env_class"] == binding["environment"]["id"]
        assert "tito_family" not in binding["model"]
        assert binding["config_sha256"] == fleet.digest_without(binding, "config_sha256")
        tokenizer = data._native_skyrl(None, None)[0]
        # The actual catalog, not a newly invented schema, owns initial tokens.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in skyrl.catalog
        ]
        tokens = tokenizer.apply_chat_template(
            row["prompt"], tools=tools, tokenize=True, return_dict=False, add_generation_prompt=True
        )
        assert binding["initial_prompt_tokens_sha256"] == fleet.sha256(fleet.canonical_json(tokens))
        assert binding["model"]["runtime_chat_template_sha256"] == manifest["template_sha256"]


@pytest.mark.parametrize("fault", ["drop", "prompt", "binding", "env"])
def test_skyrl_native_filter_or_edit_is_fatal(skyrl, fault):
    skyrl.drift = fault
    with pytest.raises(ValueError, match="native SkyRL dataset"):
        build(skyrl)
    assert not (skyrl.tmp / "out/manifest.json").exists()


def test_unknown_backend_never_fetches_tasks(setup):  # noqa: F811
    setup.config["backend"] = "invented"
    with pytest.raises(ValueError, match="backend"):
        build(setup)
    assert not setup.calls


@pytest.mark.parametrize("gpu", [True, False])
def test_skyrl_native_loader_boundary(monkeypatch, gpu):
    import sys

    from training import skyrl_episode

    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: gpu)))
    calls = []
    monkeypatch.setattr(
        skyrl_episode,
        "_module",
        lambda name, sha: calls.append((name, sha)) or NS(PromptDataset="dataset"),
    )
    monkeypatch.setattr(data, "local_tokenizer", lambda lock, root: ("tokenizer", "identity"))
    if gpu:
        with pytest.raises(ValueError, match="must not hold a GPU"):
            NATIVE({}, "/model")
        assert not calls
    else:
        assert NATIVE({}, "/model") == ("tokenizer", "tokenizer", "dataset", "identity")
        assert calls[0][0] == "skyrl.train.dataset.dataset" and len(calls[0][1]) == 64


def test_real_skyrl_prompt_dataset_and_tokenizer(setup, monkeypatch):  # noqa: F811
    pytest.importorskip("skyrl.train.dataset.dataset")
    root = os.environ.get("CYBER_TEST_QWEN_ROOT")
    if not root:
        pytest.skip("pinned SkyRL image and exact Qwen tokenizer required")
    setup.config["backend"] = "skyrl"
    setup.lock = json.loads(
        (Path(__file__).parents[1] / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()
    )
    monkeypatch.setattr(data, "_native_skyrl", lambda lock, _: NATIVE(lock, root))
    result = build(setup)
    assert len(setup.calls) == 3
    assert [v["rows"] for v in result["files"].values()] == [1, 1]
