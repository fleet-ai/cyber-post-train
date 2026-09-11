"""Native-shaped batch contracts with synthetic episodes; no paid requests."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from test_rl_episode import fixture as fleet_fixture  # noqa: F401
from test_rl_episode import seal

from evals.fleet import opencode_self_hosted as fleet
from training import skyrl_rollout as batch


@pytest.fixture
def setup(tmp_path, monkeypatch, request):
    fixture_state = request.getfixturevalue("fleet_fixture")
    config = copy.deepcopy(fixture_state.config)
    config["model"]["repo"] = "Qwen/Qwen3.8-27B"
    seal(config)
    files, rows = {}, {}
    for split in ("train", "dev"):
        row = {
            "prompt": [{"role": "user", "content": "synthetic task"}],
            "env_class": "env",
            "split": split,
            "cyber_config_json": fleet.canonical_json(config).decode(),
        }
        raw = fleet.canonical_json(row) + b"\n"
        (tmp_path / f"{split}.jsonl").write_bytes(raw)
        files[split] = {"path": f"{split}.jsonl", "sha256": fleet.sha256(raw), "rows": 1}
        rows[split] = row
    manifest = {
        "schema": "cyber_skyrl_data_v1",
        "files": files,
        "name": config["run_id"],
        "template_sha256": fleet.sha256(b"synthetic-template"),
    }
    manifest["sha256"] = fleet.digest_without(manifest, "sha256")
    path = tmp_path / "manifest.json"
    path.write_bytes(fleet.canonical_json(manifest))
    state = NS(
        path=path,
        manifest=manifest,
        rows=rows,
        calls=[],
        cleaned=[],
        tokenizer=NS(chat_template="synthetic-template"),
        engine=NS(),
        root=tmp_path / "out",
        response_tokens=16000,
        repetitions={"train": 2, "eval": 1},
        concurrency=2,
    )

    @asynccontextmanager
    async def engine(*args):
        yield state.engine

    def recorder(*args):
        return NS(config=args[0])

    async def collect(config, directory, *args, **kwargs):
        state.calls.append((config, directory))
        return [
            NS(
                tokens=[1, 2, 3, 4],
                response_length=2,
                loss_mask=[1, 0],
                rollout_log_probs=[-0.25, 0.0],
                reward=0.0,
                status=NS(name="COMPLETED"),
            )
        ]

    monkeypatch.setenv("FLEET_API_KEY", "synthetic-not-a-real-key")
    monkeypatch.setattr(batch.skyrl_episode, "single_attempt_engine", engine)
    monkeypatch.setattr(batch.skyrl_episode, "_module", lambda *args: NS(__file__="/helper.py"))
    monkeypatch.setattr(batch.skyrl_episode, "Recorder", recorder)
    monkeypatch.setattr(batch.rl_episode, "collect", collect)
    return state


def generator(state):
    return batch.Generator(
        state.path,
        state.manifest["sha256"],
        state.tokenizer,
        state.engine,
        state.root,
        response_tokens=state.response_tokens,
        repetitions=state.repetitions,
        concurrency=state.concurrency,
    )


def input_batch(state, phase="train", step=1):
    row = state.rows["train" if phase == "train" else "dev"]
    n = state.repetitions[phase]
    return {
        "batch_metadata": NS(training_phase=phase, global_step=step),
        "prompts": [copy.deepcopy(row["prompt"]) for _ in range(n)],
        "env_classes": [row["env_class"]] * n,
        "env_extras": [
            {k: v for k, v in row.items() if k not in {"prompt", "env_class"}} for _ in range(n)
        ],
        "trajectory_ids": [NS(instance_id="0", repetition_id=i) for i in range(n)],
        "sampling_params": {"temperature": 0.7},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_native_shapes_keep_tokens_zero_rewards_ids_and_source(setup, phase):
    value = generator(setup)
    source = input_batch(setup, phase)
    before = copy.deepcopy(source)
    output = await value.generate(source)
    assert source == before
    n = len(source["prompts"])
    assert output["prompt_token_ids"] == [[1, 2]] * n
    assert output["response_ids"] == [[3, 4]] * n
    assert output["loss_masks"] == [[1, 0]] * n
    assert output["rollout_logprobs"] == [[-0.25, 0.0]] * n
    assert output["rewards"] == [0.0] * n
    assert output["trajectory_ids"] is source["trajectory_ids"]
    assert output["is_last_step"] == [True] * n
    assert output["stop_reasons"] == ["stop"] * n
    assert len(set(c["run_id"] for c, _ in setup.calls)) == n
    assert all(
        c["config_sha256"] == fleet.digest_without(c, "config_sha256") for c, _ in setup.calls
    )
    receipt = json.loads(next(setup.root.glob("batches/*/COLLECTED.json")).read_bytes())
    assert receipt["sha256"] == fleet.digest_without(receipt, "sha256")
    assert not receipt["optimizer_step_verified"]
    assert not value.busy and not value.failed
    with pytest.raises(batch.rl_episode.InvalidEpisode, match="no_replacement"):
        await value.generate(source)
    assert len(setup.calls) == n and value.failed


@pytest.mark.parametrize(
    "field,value",
    [
        ("concurrency", 0),
        ("concurrency", True),
        ("repetitions", {"train": 1, "eval": 1}),
        ("repetitions", {"train": 2}),
        ("repetitions", {"train": 2, "eval": False}),
        ("root", "relative"),
    ],
)
def test_bad_constructor_policy(setup, field, value):
    setattr(setup, field, value)
    with pytest.raises(batch.rl_episode.InvalidEpisode):
        generator(setup)


@pytest.mark.parametrize(
    "mutation", ["schema", "digest", "template", "path", "bytes", "rows", "split", "run"]
)
def test_frozen_data_and_identity(setup, mutation):
    manifest = setup.manifest
    if mutation == "schema":
        manifest["schema"] = "cyber_miles_data_v1"
    elif mutation == "template":
        setup.tokenizer.chat_template = "changed"
    elif mutation == "path":
        manifest["files"]["train"]["path"] = "../elsewhere"
    elif mutation == "rows":
        manifest["files"]["train"]["rows"] = 2
    elif mutation in {"split", "run", "bytes"}:
        row = setup.rows["train"]
        if mutation == "split":
            row["split"] = "dev"
        if mutation == "run":
            config = json.loads(row["cyber_config_json"])
            config["run_id"] = "other"
            row["cyber_config_json"] = fleet.canonical_json(seal(config)).decode()
        raw = fleet.canonical_json(row) + b"\n\n"
        if mutation != "bytes":
            raw = fleet.canonical_json(row) + b"\n"
            manifest["files"]["train"]["sha256"] = fleet.sha256(raw)
        (setup.path.parent / "train.jsonl").write_bytes(raw)
    manifest["sha256"] = fleet.digest_without(manifest, "sha256")
    setup.path.write_bytes(fleet.canonical_json(manifest))
    if mutation == "digest":
        manifest["sha256"] = "wrong"
    with pytest.raises(batch.rl_episode.InvalidEpisode):
        generator(setup)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "phase",
        "step",
        "columns",
        "empty",
        "uid",
        "leadingzero",
        "rowmissing",
        "repetition",
        "duplicate",
        "partial",
        "prompt",
        "env",
        "extras",
        "auth",
    ],
)
async def test_invalid_native_input_creates_no_episodes(setup, monkeypatch, mutation):
    value = generator(setup)
    source = input_batch(setup)
    if mutation == "phase":
        source["batch_metadata"].training_phase = "test"
    elif mutation == "step":
        source["batch_metadata"].global_step = True
    elif mutation == "columns":
        source["prompts"].pop()
    elif mutation == "empty":
        source["trajectory_ids"] = []
    elif mutation == "uid":
        source["trajectory_ids"][0].instance_id = "../x"
    elif mutation == "leadingzero":
        source["trajectory_ids"][0].instance_id = "00"
    elif mutation == "rowmissing":
        source["trajectory_ids"][0].instance_id = "1"
    elif mutation == "repetition":
        source["trajectory_ids"][0].repetition_id = True
    elif mutation == "duplicate":
        source["trajectory_ids"][1].repetition_id = 0
    elif mutation == "partial":
        for key in ("prompts", "env_classes", "env_extras", "trajectory_ids"):
            source[key].pop()
    elif mutation == "prompt":
        source["prompts"][0] = []
    elif mutation == "env":
        source["env_classes"][0] = "other"
    elif mutation == "extras":
        source["env_extras"][0] = {}
    else:
        monkeypatch.delenv("FLEET_API_KEY")
    with pytest.raises(batch.rl_episode.InvalidEpisode, match="no_replacement"):
        await value.generate(source)
    assert setup.calls == [] and value.failed and not value.busy
    with pytest.raises(batch.rl_episode.InvalidEpisode, match="previous_failure"):
        await value.generate(input_batch(setup))


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,cleanup_fault", [(False, False), (True, False), (True, True)])
async def test_failed_episode_cancels_and_awaits_sibling_cleanup_without_refill(
    setup, monkeypatch, budget, cleanup_fault
):
    ready = asyncio.Event()

    async def collect(config, directory, *args, **kwargs):
        setup.calls.append(directory.name)
        if directory.name == "episode-0":
            await ready.wait()
            if budget:
                raise batch.rl_episode.EpisodeBudgetExceeded("generation_incomplete_length")
            raise RuntimeError("private server content")
        try:
            ready.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            setup.cleaned.append(directory.name)
            if cleanup_fault:
                raise batch.rl_episode.InvalidEpisode("instance_release_unconfirmed")

    monkeypatch.setattr(batch.rl_episode, "collect", collect)
    value = generator(setup)
    with pytest.raises(batch.rl_episode.InvalidEpisode) as error:
        await value.generate(input_batch(setup))
    assert "private" not in str(error.value)
    rejected = budget and not cleanup_fault
    assert isinstance(error.value, batch.rl_episode.EpisodeBudgetExceeded) == rejected
    assert sorted(setup.calls) == ["episode-0", "episode-1"]
    assert setup.cleaned == ["episode-1"]
    assert not list(setup.root.glob("batches/*/COLLECTED.json"))
    assert bool(list(setup.root.glob("batches/*/REJECTED.json"))) == rejected
    assert bool(list(setup.root.glob("batches/*/FAILED.json"))) == (not rejected)


@pytest.mark.asyncio
async def test_caller_cancellation_and_reentry(setup, monkeypatch):
    ready = asyncio.Event()

    async def collect(*args, **kwargs):
        try:
            ready.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            setup.cleaned.append(True)

    monkeypatch.setattr(batch.rl_episode, "collect", collect)
    value = generator(setup)
    task = asyncio.create_task(value.generate(input_batch(setup)))
    await ready.wait()
    with pytest.raises(batch.rl_episode.InvalidEpisode, match="reentry"):
        await value.generate(input_batch(setup))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert value.failed and len(setup.cleaned) == 2 and not value.busy


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty", "multiple", "reward", "tokens"])
async def test_invalid_results_never_become_training_data(setup, monkeypatch, mode):
    original = batch.rl_episode.collect

    async def collect(*args, **kwargs):
        samples = await original(*args, **kwargs)
        if mode == "empty":
            return []
        if mode == "multiple":
            return samples * 2
        if mode == "reward":
            samples[0].reward = float("nan")
        if mode == "tokens":
            samples[0].loss_mask = []
        return samples

    monkeypatch.setattr(batch.rl_episode, "collect", collect)
    with pytest.raises(batch.rl_episode.InvalidEpisode):
        await generator(setup).generate(input_batch(setup))


@pytest.mark.asyncio
async def test_cancellation_before_claim_writes_nothing(setup, monkeypatch):
    value = generator(setup)

    def cancel(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr(value, "_inputs", cancel)
    with pytest.raises(asyncio.CancelledError):
        await value.generate(input_batch(setup))
    assert value.failed and not value.busy and not setup.root.exists()


@pytest.mark.asyncio
async def test_real_native_output_validator(setup):
    native = pytest.importorskip("skyrl.train.utils.trainer_utils")
    types = pytest.importorskip("skyrl.train.generators.base")
    source = input_batch(setup)
    source["trajectory_ids"] = [types.TrajectoryID("0", i) for i in range(2)]
    source["batch_metadata"] = types.BatchMetadata(global_step=1, training_phase="train")
    result = await generator(setup).generate(source)
    native.validate_generator_output(2, result, step_wise=False)
    native.validate_generator_output(2, result, step_wise=True)
