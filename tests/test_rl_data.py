"""Data/split/HTTP boundaries. Synthetic fixtures contain no benchmark content."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import rl_data as data

NATIVE = data._native


def seal(value):
    value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
    return value


@pytest.fixture
def setup(tmp_path, monkeypatch):
    catalog = [
        {"name": n, "description": "synthetic tool", "inputSchema": {"type": "object"}}
        for n in ("bash", "submit_report")
    ]
    tasks = [
        {
            "task_key": "synthetic-" + str(i),
            "task_version_id": f"11111111-1111-4111-8111-{i:012}",
            "env_key": "synthetic-env",
            "env_version": "v1",
            "environment_version_id": "22222222-2222-4222-8222-222222222222",
            "data_key": "synthetic-data",
            "data_version": "v1",
            "lineage": {"application": "same-app", "task_family": f"family-{i}"},
        }
        for i in range(1, 4)
    ]
    selection = seal(
        {
            "schema": "cyber_rl_task_set_v1",
            "training_data_eligible": True,
            "tool_catalog_sha256": fleet.sha256(fleet.canonical_json(catalog)),
            "tasks": tasks,
        }
    )
    split = seal(
        {
            "schema": "cyber_task_split_v1",
            "tasks": [
                {"task_key": t["task_key"], "task_version_id": t["task_version_id"], "split": s}
                for t, s in zip(tasks, ("train", "dev", "test"), strict=True)
            ],
        }
    )
    config = {
        "name": "synthetic-rl",
        "output": "out",
        "task_set": "tasks.json",
        "split": "split.json",
        "model_lock": "model.json",
        "tool_catalog": "tools.json",
        "model_root": "/mnt/sfs/models/synthetic-model",
        "limits": {
            "context_tokens": 32768,
            "response_tokens": 8192,
            "max_tokens_per_turn": 4096,
            "max_turns": 40,
            "episode_seconds": 1800,
            "tool_seconds": 60,
            "tool_result_chars": 10000,
        },
    }
    lock = {"repo": "Qwen/Qwen3.8-27B", "revision": "a" * 40}
    state = NS(
        config=config,
        selection=selection,
        split=split,
        lock=lock,
        catalog=catalog,
        responses={},
        calls=[],
        tmp=tmp_path,
        native_calls=[],
        drift=None,
        bad_tokens=False,
    )
    for selected in tasks:
        state.responses[selected["task_key"]] = {
            "key": selected["task_key"],
            "eval_task_version_id": selected["task_version_id"],
            "environment_id": selected["env_key"],
            "version": selected["env_version"],
            "data_id": selected["data_key"],
            "data_version": selected["data_version"],
            "environment_version_id": selected["environment_version_id"],
            "prompt": "Private synthetic prompt " + selected["task_key"],
            "verifier_id": "33333333-3333-4333-8333-333333333333",
            "verifier": {
                "verifier_version_id": "44444444-4444-4444-8444-444444444444",
                "version": "v1",
                "sha256": "sha256:" + "b" * 64,
            },
            "metadata": {
                "cyber_contract": copy.deepcopy(data.AUTHORITY["required_cyber_contract"]),
                "runtime_seed_manifest": {"content_sha256": "sha256:" + "c" * 64},
            },
        }

    class Tokenizer:
        def encode(self, text, **kw):
            return list(text.encode())

        def apply_chat_template(
            self, messages, *, tools, tokenize, add_generation_prompt, return_dict=None
        ):
            assert add_generation_prompt and len(messages) == 1
            text = "synthetic-header " + json.dumps(tools) + messages[0]["content"]
            if not tokenize:
                return text
            tokens = [0] if state.bad_tokens else self.encode(text)
            # Transformers 5.8 defaults to BatchEncoding, unlike native Miles.
            if config.get("backend") == "skyrl" and return_dict is not False:
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
            return tokens

    class Dataset(list):
        def __init__(self, path, tokenizer, processor, budget, **kwargs):
            state.native_calls.append((path, budget, kwargs))
            values = [json.loads(line) for line in Path(path).read_text().splitlines()]
            if state.drift == "drop":
                values.pop()
            if state.drift == "prompt":
                values[0]["input"] = "changed"
            if state.drift == "metadata":
                values[0]["metadata"]["split"] = "test"
            super().__init__(NS(prompt=v["input"], metadata=v["metadata"]) for v in values)

    monkeypatch.setattr(
        data, "_native", lambda lock, root: (Tokenizer(), Tokenizer(), Dataset, lock)
    )

    def handler(request):
        state.calls.append(request)
        assert request.method == "GET" and request.url.host == "orchestrator.fleetai.com"
        if request.url.path == "/v1/account":
            return httpx.Response(200, json={"team_name": "fleet", "team_id": fleet.FLEET_TEAM_ID})
        key = request.url.path.removeprefix("/v1/tasks/")
        task = state.responses[key]
        assert request.url.params["version_id"] == task["eval_task_version_id"]
        return httpx.Response(200, json=task)

    state.handler = handler
    return state


def build(state):
    for name, value in (
        ("tasks.json", state.selection),
        ("split.json", state.split),
        ("model.json", state.lock),
        ("tools.json", state.catalog),
    ):
        (state.tmp / name).write_text(json.dumps(value))
    with httpx.Client(transport=httpx.MockTransport(state.handler)) as client:
        return data.build(state.config, relative_to=state.tmp, client=client)


def test_exact_task_set_get_only_native_retention_and_private_files(setup):
    result = build(setup)
    assert len(setup.calls) == 3
    assert all("synthetic-3" not in r.url.path for r in setup.calls)  # Test never fetched.
    assert {k: v["rows"] for k, v in result["files"].items()} == {"train": 1, "dev": 1}
    assert result["submitted"] is False and "Private" not in json.dumps(result)
    directory = setup.tmp / "out"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["sha256"] == seal(copy.deepcopy(manifest))["sha256"]
    assert manifest["split_sha256"] == setup.split["sha256"]
    assert (directory.stat().st_mode & 0o777) == 0o700
    for path in directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    for split, spec in result["files"].items():
        raw = (directory / spec["path"]).read_bytes()
        assert fleet.sha256(raw) == spec["sha256"]
        row = json.loads(raw)
        config = row["metadata"]["cyber_config"]
        assert row["metadata"]["split"] == split
        assert config["initial_prompt_sha256"] == fleet.sha256(row["input"].encode())
        assert config["config_sha256"] == fleet.digest_without(config, "config_sha256")
        assert config["authority"]["scoring_payload_mode"] == fleet.RUNTIME_EVIDENCE_ONLY_V3
    assert all(n[1] == 24576 and not n[2]["apply_chat_template"] for n in setup.native_calls)
    with pytest.raises(FileExistsError):
        build(setup)
    assert len(setup.calls) == 3


@pytest.mark.parametrize(
    "fault",
    [
        "set_digest",
        "split_digest",
        "ineligible",
        "duplicate_split",
        "bad_split",
        "missing_tuple",
        "duplicate_task",
        "missing_split",
        "uuid",
        "uuid_noncanonical",
        "key",
        "lineage",
        "family_leak",
        "family_rename",
        "unresolved_split",
        "no_dev",
    ],
)
def test_invalid_selection_stops_before_network(setup, fault):
    tasks, split = setup.selection["tasks"], setup.split["tasks"]
    if fault == "set_digest":
        setup.selection["schema"] = "wrong"
    elif fault == "split_digest":
        setup.split["schema"] = "wrong"
    elif fault == "ineligible":
        setup.selection["training_data_eligible"] = False
    elif fault == "duplicate_split":
        split.append(copy.deepcopy(split[0]))
    elif fault == "bad_split":
        split[0]["split"] = "unknown"
    elif fault == "missing_tuple":
        tasks[0].pop("data_version")
    elif fault == "duplicate_task":
        tasks.append(copy.deepcopy(tasks[0]))
    elif fault == "missing_split":
        split.pop()
    elif fault == "uuid":
        tasks[0]["environment_version_id"] = "current"
    elif fault == "uuid_noncanonical":
        tasks[0]["environment_version_id"] = tasks[0]["environment_version_id"].replace("-", "")
    elif fault == "key":
        tasks[0]["task_key"] = split[0]["task_key"] = "../bad"
    elif fault == "lineage":
        tasks[0]["lineage"]["application"] = "unknown"
    elif fault == "family_leak":
        tasks[1]["lineage"] = tasks[0]["lineage"]
    elif fault == "family_rename":
        tasks[1]["task_key"] = split[1]["task_key"] = tasks[0]["task_key"]
    elif fault == "unresolved_split":
        tasks.pop()
    elif fault == "no_dev":
        split[1]["split"] = "reserved_dev"
    seal(setup.selection)
    seal(setup.split)
    with pytest.raises(ValueError):
        build(setup)
    assert not setup.calls and not (setup.tmp / "out").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "name",
        "model",
        "catalog",
        "catalog_order",
        "response_bool",
        "response_large",
        "limits",
        "prompt_empty",
        "contract",
        "tokens",
        "length",
        "team",
    ],
)
def test_data_contracts_fail_without_creating_environments(setup, fault):
    if fault == "unknown":
        setup.config["extra_args"] = "--override"
    elif fault == "name":
        setup.config["name"] = "unsafe/name"
    elif fault == "model":
        setup.lock["repo"] = "zai-org/GLM-5.3"
    elif fault == "catalog":
        setup.catalog[0]["description"] = "changed"
    elif fault == "catalog_order":
        setup.catalog.reverse()
        setup.selection["tool_catalog_sha256"] = fleet.sha256(fleet.canonical_json(setup.catalog))
        seal(setup.selection)
    elif fault == "response_bool":
        setup.config["limits"]["response_tokens"] = True
    elif fault == "response_large":
        setup.config["limits"]["response_tokens"] = 40000
    elif fault == "limits":
        setup.config["limits"]["tool_seconds"] = -1
    elif fault == "prompt_empty":
        setup.responses["synthetic-1"]["prompt"] = ""
    elif fault == "contract":
        setup.responses["synthetic-1"]["metadata"]["cyber_contract"] = {}
    elif fault == "tokens":
        setup.bad_tokens = True
    elif fault == "length":
        setup.config["limits"]["response_tokens"] = 32767
    elif fault == "team":
        setup.handler = lambda request: httpx.Response(200, json={"team_name": "other"})
    with pytest.raises((ValueError, data.rl_episode.InvalidEpisode)):
        build(setup)
    assert not (setup.tmp / "out").exists()


@pytest.mark.parametrize("drift", ["drop", "prompt", "metadata"])
def test_no_receipt_when_native_loader_drops_or_changes_a_row(setup, drift):
    setup.drift = drift
    with pytest.raises(ValueError, match="native Miles dataset"):
        build(setup)
    assert (setup.tmp / "out" / "train.jsonl").exists()
    assert not (setup.tmp / "out" / "manifest.json").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_shared_binding_preserves_exact_runtime_and_legacy_data_gate(setup, legacy):
    task, selected = setup.responses["synthetic-1"], setup.selection["tasks"][0]
    if legacy:
        task.pop("data_id")
        task.pop("data_version")
    binding, env, verifier = fleet.bind_task(task, selected)
    assert ("data_binding_validation" in binding) == legacy
    config = {"task": binding, "environment": env, "verifier": verifier}
    assert fleet.verify_task(config, task) == task


@pytest.mark.parametrize("fault", ["key", "runtime", "legacy_version", "seed", "verifier"])
def test_shared_binding_rejects_drift(setup, fault):
    task, selected = setup.responses["synthetic-1"], setup.selection["tasks"][0]
    if fault == "key":
        task["key"] = "other"
    elif fault == "runtime":
        task["data_id"] = "other"
    elif fault == "legacy_version":
        task.pop("data_id")
        task.pop("data_version")
        task["environment_version_id"] = None
    elif fault == "seed":
        task["metadata"]["runtime_seed_manifest"] = {}
    else:
        task["verifier"]["sha256"] = ""
    with pytest.raises(RuntimeError):
        fleet.bind_task(task, selected)


@pytest.mark.parametrize("fault", [None, "gpu", "template"])
def test_native_tokenizer_tito_and_dataset_keep_every_row(setup, monkeypatch, fault, tmp_path):
    native = pytest.importorskip("fti.trainers.miles.run_fleet")
    import torch

    root = os.environ.get("CYBER_TEST_QWEN_ROOT")
    if not root:
        pytest.skip("pinned Miles image plus exact staged Qwen tokenizer required")
    setup.lock = json.loads(
        (Path(__file__).parents[1] / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()
    )
    monkeypatch.setattr(data, "_native", lambda lock, _: NATIVE(lock, root))
    if fault == "gpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    elif fault == "template":
        (tmp_path / "qwen3.8_fixed.jinja").write_text("changed")
        monkeypatch.setattr(native, "TEMPLATES", tmp_path)
    if fault:
        with pytest.raises(ValueError):
            build(setup)
        assert not setup.calls
    else:
        result = build(setup)
        assert [v["rows"] for v in result["files"].values()] == [1, 1]
        assert len(setup.calls) == 3  # Real native code; only Fleet replies are synthetic.
