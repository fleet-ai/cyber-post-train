"""Run the strict collector with both local doubles and the pinned native Miles.

All samples and rewards are synthetic. No engine, Fleet environment or optimizer
is started; the native variant covers real Dataset/cursor/group cancellation/GRPO.
"""

import asyncio
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from evals.fleet import opencode_self_hosted as fleet
from training.miles_rollout import Rollout
from training.rl_episode import InvalidEpisode


class Tokenizer:
    chat_template = "synthetic template"

    def __call__(self, prompts, **kwargs):
        return {"input_ids": [[1] * len(p) for p in prompts]}


@pytest.fixture(params=["local", "native"])
def runtime(request, monkeypatch):
    if request.param == "native":
        common = pytest.importorskip("miles.rollout.inference_rollout.inference_rollout_common")
        from miles.rollout import data_source
        from miles.rollout.base_types import (
            GenerateFnOutput,
            RolloutFnEvalInput,
            RolloutFnTrainInput,
        )
        from miles.rollout.inference_rollout.compatibility import load_rollout_function
        from miles.utils.eval_config import EvalDatasetConfig
        from miles.utils.types import Sample

        from training import miles_text

        for module in (common, data_source, miles_text):
            monkeypatch.setattr(module, "load_tokenizer", lambda *a, **k: Tokenizer())
        for module in (common, data_source):
            # Real Qwen AutoProcessor is truthy despite the text-only recipe.
            monkeypatch.setattr(module, "load_processor", lambda *a, **k: object())
        return NS(
            native=True,
            Source=miles_text.TextDataSource,
            Sample=Sample,
            Eval=RolloutFnEvalInput,
            Train=RolloutFnTrainInput,
            Output=GenerateFnOutput,
            Config=EvalDatasetConfig,
            load=load_rollout_function,
        )

    class Sample(NS):
        def __init__(self, **kwargs):
            super().__init__(
                index=None,
                group_index=None,
                rollout_id=None,
                response="",
                reward=None,
                generate_function_path=None,
                status=NS(name="PENDING"),
                **kwargs,
            )

    class Dataset:
        def __init__(self, path, *a, **kw):
            if len(a) > 1:
                assert a[1] is None
            self.origin_samples = [
                Sample(prompt=row["input"], metadata=row["metadata"])
                for row in map(json.loads, Path(path).read_text().splitlines())
            ]
            self.samples = self.origin_samples

        def __len__(self):
            return len(self.samples)

    class Source:
        def __init__(self, args):
            self.dataset, self.args = Dataset(args.prompt_data), args
            self.sample_offset = self.sample_index = 0

        def get_samples(self, n):
            groups = []
            for _ in range(n):
                sample = self.dataset.samples[self.sample_offset % len(self.dataset)]
                group = [copy.deepcopy(sample) for _ in range(self.args.n_samples_per_prompt)]
                for s in group:
                    s.index, s.group_index = self.sample_index, self.sample_offset
                    self.sample_index += 1
                self.sample_offset += 1
                groups.append(group)
            return groups

    class State:
        def __init__(self, args):
            self.args, self.tokenizer, self.processor = args, Tokenizer(), object()
            self.sampling_params, self.aborted = {"temperature": 1}, False

    @dataclass(frozen=True)
    class Output:
        samples: Sample | list[Sample]

    async def group(state, samples, sampling, evaluation=False):
        tasks = [
            asyncio.create_task(
                state.generate_function(
                    NS(
                        sample=s,
                        evaluation=evaluation,
                        sampling_params=sampling,
                    )
                )
            )
            for s in samples
        ]
        try:
            return [(await task).samples for task in tasks]
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    for name, attrs in {
        "miles.rollout.inference_rollout.inference_rollout_common": {
            "GenerateState": State,
            "generate_and_rm_group": group,
            "compute_sampling_params": lambda args, **kw: kw,
        },
        "miles.utils.data": {"Dataset": Dataset},
        "miles.rollout.base_types": {"RolloutFnEvalOutput": NS, "RolloutFnTrainOutput": NS},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return NS(
        native=False,
        Source=Source,
        Sample=Sample,
        Output=Output,
        Config=NS,
        Eval=lambda rollout_id: NS(
            rollout_id=rollout_id, evaluation=True, generate_state=None, hf_dir=None
        ),
        Train=lambda rollout_id: NS(rollout_id=rollout_id, evaluation=False),
        load=lambda input, path: Rollout(input),
    )


@pytest.fixture
def fixture(tmp_path, runtime):
    paths = {group: tmp_path / (group + ".jsonl") for group in ("train", "dev")}
    for group, path in paths.items():
        rows = [
            {"input": "synthetic prompt", "metadata": {"split": group, "task": i}} for i in range(2)
        ]
        path.write_bytes(b"".join(fleet.canonical_json(row) + b"\n" for row in rows))
    manifest = {
        "schema": "cyber_miles_data_v1",
        "name": "synthetic-batch",
        "template_sha256": fleet.sha256(Tokenizer.chat_template.encode()),
        "files": {
            group: {"path": path.name, "sha256": fleet.sha256(path.read_bytes()), "rows": 2}
            for group, path in paths.items()
        },
    }
    manifest["sha256"] = fleet.digest_without(manifest, "sha256")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = NS(
        prompt_data=str(paths["train"]),
        input_key="input",
        metadata_key="metadata",
        hf_checkpoint="/synthetic-model",
        chat_template_path=None,
        rollout_global_dataset=True,
        rollout_shuffle=False,
        rollout_seed=42,
        rollout_max_prompt_len=1000,
        label_key=None,
        tool_key=None,
        multimodal_keys=None,
        dump_details=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_temperature=1,
        rollout_top_p=1,
        rollout_top_k=-1,
        rollout_max_response_len=100,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_server_concurrency=4,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        sglang_router_policy="round_robin",
        partial_rollout=False,
        group_rm=False,
        custom_rm_path=None,
        custom_generate_function_path="training.rl_episode.generate",
        rollout_function_path="training.miles_rollout.Rollout",
        eval_function_path="training.miles_rollout.Rollout",
        cyber_data_manifest=str(manifest_path),
        cyber_run_id="synthetic-batch",
        cyber_output_root=str(tmp_path / "episodes"),
        eval_max_prompt_len=1000,
        num_rollout=3,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        advantage_estimator="grpo",
        grpo_std_normalization=True,
        eval_datasets=[
            runtime.Config(
                name="dev",
                path=str(paths["dev"]),
                input_key="input",
                metadata_key="metadata",
                metadata_overrides={},
                custom_generate_function_path=None,
                rm_type=None,
                n_samples_per_eval_prompt=1,
                temperature=1,
                top_p=1,
                top_k=-1,
                max_response_len=100,
            )
        ],
    )
    source = runtime.Source(args)
    state = NS(args=args, data_source=source, calls=[], paths=paths, manifest=manifest)

    async def generate(input):
        state.calls.append(
            (input.evaluation, input.sample.index, input.sample.metadata["cyber_batch"])
        )
        s = copy.deepcopy(input.sample)
        s.response, s.tokens, s.response_length = "synthetic response", [1, 2], 1
        s.loss_mask, s.rollout_log_probs = [1], [-0.1]
        s.status = runtime.Sample.Status.COMPLETED if runtime.native else NS(name="COMPLETED")
        s.reward = float(s.index % 2)
        return runtime.Output(samples=s)

    state.generate = generate
    return state


def build(fixture, runtime):
    rollout = runtime.load(fixture, "training.miles_rollout.Rollout")
    rollout.state.generate_function = fixture.generate
    return rollout


def test_native_batch_contract_files_match_reviewed_commit(runtime):
    if not runtime.native:
        return
    import miles.rollout.base_types as types

    root = Path(types.__file__).parents[1]
    for path, digest in {
        "rollout/inference_rollout/inference_rollout_common.py": (
            "96e3cba12ae033527e823ed3dd8cb43c31d244775756ecf0bab4ad81eb4f06a4"
        ),
        "rollout/data_source.py": (
            "e97d7df23644d231480079d03823c2d72b8f0cc4383f33c528ae17871c09e7e5"
        ),
        "ray/rollout/train_data_conversion.py": (
            "4fa430c5e5a0395e042dc7f20864baf3bc85e71c9d3f05b063be44ddab8bd60a"
        ),
        "rollout/base_types.py": "3f3bd4c5bb9b7adb5aafc1b806153f41c642fa232d4baa9bfe17d91b74a1cce8",
        "rollout/inference_rollout/compatibility.py": (
            "d55d682bf15673cd90aefee59ad174b82d77497c880fd2537b577c2faf3feca0"
        ),
        "ray/rollout/rollout_manager.py": (
            "81881154c84ed0c8fcb0eb2ec52c0f215b1776d8f7b0e0aedf6ba6bf8868101c"
        ),
    }.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


@pytest.mark.asyncio
async def test_baseline_training_final_and_native_competing_reward_groups(fixture, runtime):
    rollout = build(fixture, runtime)
    baseline = await rollout(runtime.Eval(0))
    trained = await rollout(runtime.Train(0))
    final = await rollout(runtime.Eval(0))
    assert baseline.data["dev"]["rewards"] == final.data["dev"]["rewards"] == [0, 1]
    assert baseline.metrics["cyber/dev_is_baseline"] == 1
    assert final.metrics["cyber/dev_is_baseline"] == 0
    flat = [s for group in trained.samples for s in group]
    assert [s.rollout_id for s in flat] == [0, 1, 2, 3]
    assert [s.group_index for s in flat] == [0, 0, 1, 1]
    assert fixture.data_source.sample_index == 4  # Exactly one native cursor advance.
    assert sorted(p.name for p in rollout.root.iterdir()) == [
        "dev-after-r0",
        "dev-baseline-r0",
        "train-r0",
    ]
    assert all(
        not json.loads((p / "COLLECTED.json").read_text())["optimizer_step_verified"]
        for p in rollout.root.iterdir()
    )
    if runtime.native:
        from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout

        normalized = _normalize_rewards_by_rollout(fixture.args, flat, [0, 1, 0, 1], None)
        assert normalized[0] < 0 < normalized[1]
        assert normalized[:2] == normalized[2:]
    following = await rollout(runtime.Train(1))
    assert [s.index for group in following.samples for s in group] == [4, 5, 6, 7]


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [True, False])
async def test_failure_cancels_and_awaits_all_groups_without_refilling(
    fixture, runtime, evaluation
):
    rollout = build(fixture, runtime)
    if not evaluation:
        await rollout(runtime.Eval(0))
    started, released = [], []
    all_started = asyncio.Event()
    count = 2 if evaluation else 4

    async def fail(input):
        started.append(input.sample.index)
        if len(started) == count:
            all_started.set()
        try:
            await all_started.wait()
            if input.sample.index == 0:
                raise ValueError("PRIVATE SDK RESPONSE")
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            released.append(input.sample.index)

    rollout.state.generate_function = fail
    async with asyncio.timeout(5):
        with pytest.raises(InvalidEpisode, match="batch_failed_no_replacement") as caught:
            await rollout(runtime.Eval(0) if evaluation else runtime.Train(0))
    assert "PRIVATE" not in str(caught.value)
    assert sorted(started) == sorted(released) == list(range(count))
    assert fixture.data_source.sample_index == (0 if evaluation else 4)
    assert rollout.failed and not rollout.busy
    path = rollout.root / ("dev-baseline-r0" if evaluation else "train-r0")
    assert (path / "FAILED.json").exists() and not (path / "COLLECTED.json").exists()
    with pytest.raises(InvalidEpisode):
        await rollout(runtime.Train(1))
    assert len(started) == count


@pytest.mark.asyncio
async def test_external_cancellation_awaits_cleanup_and_poisoned_batch_stays_stopped(
    fixture, runtime
):
    rollout = build(fixture, runtime)
    started, released = asyncio.Event(), []

    async def wait(input):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            released.append(input.sample.index)

    rollout.state.generate_function = wait
    task = asyncio.create_task(rollout(runtime.Eval(0)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released and rollout.failed and not rollout.busy
    assert json.loads((rollout.root / "dev-baseline-r0/FAILED.json").read_text()) == {
        "reason": "cancelled"
    }


@pytest.mark.asyncio
async def test_duplicate_invocation_and_new_instance_cannot_replay(fixture, runtime):
    rollout = build(fixture, runtime)
    await rollout(runtime.Eval(0))
    count = len(fixture.calls)
    with pytest.raises(InvalidEpisode):
        await rollout(runtime.Eval(0))
    replacement = build(fixture, runtime)
    with pytest.raises(InvalidEpisode):
        await replacement(runtime.Eval(0))
    assert len(fixture.calls) == count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["groups", "indices", "started", "output-size", "second-baseline"]
)
async def test_native_shape_drift_and_reentry_stop_without_replacement(
    fixture, runtime, fault, monkeypatch
):
    rollout = build(fixture, runtime)
    await rollout(runtime.Eval(0))
    if fault == "second-baseline":
        with pytest.raises(InvalidEpisode):
            await rollout(runtime.Eval(1))
        return
    original = fixture.data_source.get_samples

    def drift(n):
        groups = original(n)
        if fault == "groups":
            groups.pop()
        elif fault == "indices":
            groups[0][0].index = groups[0][1].index
        elif fault == "started":
            groups[0][0].reward = 0.0
        return groups

    monkeypatch.setattr(fixture.data_source, "get_samples", drift)
    if fault == "output-size":
        common = sys.modules["miles.rollout.inference_rollout.inference_rollout_common"]

        async def invalid(*args, **kwargs):
            return []

        monkeypatch.setattr(common, "generate_and_rm_group", invalid)
    with pytest.raises(InvalidEpisode, match="batch_failed_no_replacement"):
        await rollout(runtime.Train(0))
    assert fixture.data_source.sample_index == 4


@pytest.mark.asyncio
async def test_compacted_segments_share_only_their_own_episode(fixture, runtime):
    rollout = build(fixture, runtime)
    await rollout(runtime.Eval(0))

    async def compact(input):
        output = await fixture.generate(input)
        return runtime.Output(samples=[output.samples, copy.deepcopy(output.samples)])

    rollout.state.generate_function = compact
    data = (await rollout(runtime.Train(0))).samples
    assert [[s.rollout_id for s in episode] for group in data for episode in group] == [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    dev = await rollout(runtime.Eval(0))
    assert len(dev.data["dev"]["samples"]) == 2
    assert dev.metrics["cyber/dev_segments"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["no-baseline", "negative", "boolean", "busy", "skipped", "wrong-eval", "snapshot"]
)
async def test_sequence_errors_do_not_generate(fixture, runtime, fault):
    rollout = build(fixture, runtime)
    call = runtime.Train(0)
    if fault != "no-baseline":
        await rollout(runtime.Eval(0))
        await rollout(runtime.Train(0))
        call = runtime.Train(1)
    if fault in {"negative", "boolean", "skipped"}:
        call = runtime.Train({"negative": -1, "boolean": True, "skipped": 2}[fault])
    elif fault == "busy":
        rollout.busy = True
    elif fault == "wrong-eval":
        call = runtime.Eval(1)
    elif fault == "snapshot":
        call = NS(evaluation=True, rollout_id=0, generate_state=object(), hf_dir=None)
    count = len(fixture.calls)
    with pytest.raises(InvalidEpisode):
        await rollout(call)
    assert len(fixture.calls) == count


@pytest.mark.parametrize(
    "fault",
    [
        "policy",
        "path",
        "digest",
        "template",
        "filtered",
        "count",
        "override",
        "empty-dev",
        "groups",
        "oversample",
        "batch",
        "relative",
    ],
)
def test_preflight_rejects_drift_before_generation(fixture, runtime, fault):
    if fault == "policy":
        fixture.args.dynamic_sampling_filter_path = "unsafe"
    elif fault == "path":
        fixture.args.prompt_data = str(fixture.paths["dev"])
    elif fault in {"digest", "filtered", "count", "template"}:
        if fault == "filtered":
            fixture.data_source.dataset.samples.pop()
        elif fault == "digest":
            fixture.paths["train"].write_bytes(fixture.paths["train"].read_bytes() + b"\n")
        else:
            manifest = copy.deepcopy(fixture.manifest)
            if fault == "count":
                manifest["files"]["train"]["rows"] = 3
            else:
                manifest["template_sha256"] = fleet.sha256(b"other")
            manifest["sha256"] = fleet.digest_without(manifest, "sha256")
            Path(fixture.args.cyber_data_manifest).write_text(json.dumps(manifest))
    elif fault == "override":
        fixture.args.eval_datasets[0].metadata_overrides = {"split": "train"}
    elif fault == "empty-dev":
        fixture.args.eval_datasets = []
    elif fault == "groups":
        fixture.args.rollout_batch_size = 3
    elif fault == "oversample":
        fixture.args.over_sampling_batch_size = 3
    elif fault == "batch":
        fixture.args.global_batch_size = 8
    elif fault == "relative":
        fixture.args.cyber_output_root = "relative"
    with pytest.raises((InvalidEpisode, ValueError)):
        build(fixture, runtime)
    assert not fixture.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["count", "identity", "reward", "index", "status", "multi"])
async def test_invalid_generated_groups_never_become_training_data(fixture, runtime, fault):
    rollout = build(fixture, runtime)
    await rollout(runtime.Eval(0))

    async def corrupt(input):
        output = await fixture.generate(input)
        s = output.samples
        if fault == "identity":
            s.rollout_id = 999
        elif fault == "reward":
            s.reward = float("nan")
        elif fault == "index":
            s.index = None
        elif fault == "status":
            s.status = NS(name="FAILED")
        elif fault == "multi":
            other = copy.deepcopy(s)
            other.reward = 1 - s.reward
            return runtime.Output(samples=[s, other])
        else:
            return runtime.Output(samples=[])
        return output

    rollout.state.generate_function = corrupt
    with pytest.raises(InvalidEpisode, match="batch_failed_no_replacement"):
        await rollout(runtime.Train(0))
    assert rollout.last_train is None
    assert not (rollout.root / "train-r0/COLLECTED.json").exists()
