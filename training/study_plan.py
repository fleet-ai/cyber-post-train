"""Compile an explicit, bounded Qwen SFT study; never read data or submit work."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from training.io import digest_json

SCHEMA = "cyber_qwen_sft_study_v2"
HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


def _fields(value: object, required: set[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() != required:
        raise ValueError(f"{label}: provide exactly the documented fields")


def _hash(value: object, label: str) -> None:
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{label}: require a full lowercase sha256: digest")


def _name(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) > 31 or not NAME.fullmatch(value):
        raise ValueError(f"{label}: require a lowercase hyphenated name of at most 31 characters")


def _int(value: object, minimum: int, label: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: require an integer >= {minimum}")


def compile_study(config: dict) -> dict:
    """Validate metadata and freeze treatment rows, not executable SFT requests.

    Digests bind declarations; this function cannot prove corpus contents,
    family disjointness, current capacity, or successful dev-cluster testing.
    """
    _fields(
        config,
        {"name", "model", "splits", "sources", "wandb", "capacity", "evaluation", "stages"},
        "study",
    )
    _name(config["name"], "study name")
    model = config["model"]
    _fields(model, {"repo", "revision", "lock_sha256"}, "model")
    if model["repo"] != "Qwen/Qwen3.8-27B":
        raise ValueError("this study planner currently supports Qwen/Qwen3.8-27B only")
    if not isinstance(model["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", model["revision"]
    ):
        raise ValueError("model revision: require an exact lowercase Git commit")
    _hash(model["lock_sha256"], "model lock")
    splits = config["splits"]
    if not isinstance(splits, dict) or not splits or splits.keys() - {"a", "b"}:
        raise ValueError("splits: declare exact A and/or B variants")
    for split in splits.values():
        _fields(
            split,
            {
                "variant_seed",
                "study_split_sha256",
                "training_split_sha256",
                "dev_task_set_sha256",
                "fleet_dev_protocol_sha256",
                "final_test_lock_sha256",
            },
            "split",
        )
        if not isinstance(split["variant_seed"], str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", split["variant_seed"]
        ):
            raise ValueError("split variant_seed: require the exact named split seed")
        for key, value in split.items():
            if key != "variant_seed":
                _hash(value, key)
    for key in ("study_split_sha256", "training_split_sha256", "dev_task_set_sha256"):
        if len({s[key] for s in splits.values()}) != len(splits):
            raise ValueError("A/B must bind distinct outer/train/dev split identities")
    sources = config["sources"]
    _fields(sources, set(splits), "source split variants")
    for variant, treatments in sources.items():
        if (
            not isinstance(treatments, dict)
            or not treatments
            or treatments.keys() - {"teacher", "self"}
        ):
            raise ValueError("sources: each split declares teacher and/or self only")
        for source in treatments.values():
            _fields(
                source,
                {
                    "corpus_sha256",
                    "study_split_sha256",
                    "training_split_sha256",
                    "source_selection_sha256",
                    "fleet_dev_protocol_sha256",
                    "status",
                    "source_sessions",
                    "qualification_sha256",
                },
                "source",
            )
            for key, value in source.items():
                if key.endswith("sha256"):
                    _hash(value, key)
            if source["status"] != "qualified":
                raise ValueError("blocked or unqualified source cannot enter a frozen study")
            _int(source["source_sessions"], 1, "qualified source sessions")
            if any(
                source[key] != splits[variant][key]
                for key in (
                    "study_split_sha256",
                    "training_split_sha256",
                    "fleet_dev_protocol_sha256",
                )
            ):
                raise ValueError("source must bind its own exact split and Fleet dev protocol")
    wandb = config["wandb"]
    _fields(wandb, {"entity", "project"}, "wandb")
    for value in wandb.values():
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value
        ):
            raise ValueError("wandb entity/project: require an explicit safe identifier")
    capacity = config["capacity"]
    _fields(
        capacity,
        {"max_active_nodes", "other_active_study_nodes", "excluded_inference_endpoints"},
        "capacity",
    )
    _int(capacity["max_active_nodes"], 1, "max_active_nodes")
    _int(capacity["other_active_study_nodes"], 0, "other_active_study_nodes")
    if (
        type(capacity["excluded_inference_endpoints"]) is not int
        or capacity["excluded_inference_endpoints"] != 4
    ):
        raise ValueError(
            "only the four pre-existing inference endpoints are outside this study cap"
        )
    if capacity["max_active_nodes"] > 8:
        raise ValueError("max_active_nodes cannot exceed the eight-node study limit")
    evaluation = config["evaluation"]
    _fields(evaluation, {"final_fleet", "final_webexploitbench"}, "evaluation")
    for name, protocol in evaluation.items():
        fields = {"task_set_sha256", "protocol_sha256"}
        if name == "final_fleet":
            fields.add("lock_sha256")
        _fields(protocol, fields, "evaluation protocol")
        for key, value in protocol.items():
            _hash(value, key)
    task_sets = [s["dev_task_set_sha256"] for s in splits.values()] + [
        p["task_set_sha256"] for p in evaluation.values()
    ]
    if len(set(task_sets)) != len(task_sets):
        raise ValueError("dev and final task-set identities must be distinct")
    if any(
        s["final_test_lock_sha256"] != evaluation["final_fleet"]["lock_sha256"]
        for s in splits.values()
    ):
        raise ValueError("all split variants require one common sealed Fleet final-test lock")
    if not isinstance(config["stages"], list) or not config["stages"]:
        raise ValueError("stages: require a nonempty explicit list, never a parameter grid")

    stages, stage_ids, run_ids, treatments, used_sources = [], set(), set(), set(), set()
    for stage in config["stages"]:
        _fields(stage, {"id", "decision_sha256", "arms"}, "stage")
        _name(stage["id"], "stage id")
        if stage["id"] in stage_ids:
            raise ValueError("duplicate stage identity")
        stage_ids.add(stage["id"])
        if stages or stage["decision_sha256"] is not None:
            _hash(stage["decision_sha256"], "preceding fresh-dev selection decision")
        if not isinstance(stage["arms"], list) or not stage["arms"]:
            raise ValueError("arms: require a nonempty explicit list")
        if len(stage["arms"]) + capacity["other_active_study_nodes"] > capacity["max_active_nodes"]:
            raise ValueError("stage plus other study nodes exceeds declared study capacity")
        arms = []
        for arm in stage["arms"]:
            _fields(arm, {"id", "split", "source", "lr", "batch_size", "epochs", "seed"}, "arm")
            _name(arm["id"], "arm id")
            run_id = f"{config['name']}-{arm['id']}"
            _name(run_id, "derived run id")
            if run_id in run_ids:
                raise ValueError("duplicate run/output/W&B identity across stages")
            run_ids.add(run_id)
            if not isinstance(arm["split"], str) or arm["split"] not in splits:
                raise ValueError("arm split must select a declared exact variant")
            if not isinstance(arm["source"], str) or arm["source"] not in sources[arm["split"]]:
                raise ValueError("arm source must select a declared teacher/self source")
            if type(arm["lr"]) not in (int, float) or not 0 < arm["lr"] <= 1e-4:
                raise ValueError("lr: require one number in (0, 1e-4], not a grid or placeholder")
            for key in ("batch_size", "epochs", "seed"):
                _int(arm[key], 0 if key == "seed" else 1, key)
            if arm["batch_size"] % 8:
                raise ValueError("batch_size must be divisible by eight for the one-node treatment")
            if arm["seed"] >= 2**32:
                raise ValueError("training seed must fit an unsigned 32-bit integer")
            treatment = tuple(
                arm[k] for k in ("split", "source", "lr", "batch_size", "epochs", "seed")
            )
            if treatment in treatments:
                raise ValueError(
                    "duplicate scientific treatment; confirmation needs a distinct seed"
                )
            treatments.add(treatment)
            used_sources.add((arm["split"], arm["source"]))
            arms.append(
                {
                    **arm,
                    "run_name": run_id,
                    "output_root": f"/mnt/sfs/jobs/{run_id}",
                    "wandb": {**wandb, "group": config["name"], "run_id": run_id, "name": run_id},
                    "data": dict(sources[arm["split"]][arm["source"]]),
                    "validation_mode": "task_outcomes_only",
                    "fleet_dev_protocol_sha256": splits[arm["split"]]["fleet_dev_protocol_sha256"],
                    "eval_interval": 0,
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "microbatch_per_gpu": 1,
                }
            )
        stages.append(
            {
                "id": stage["id"],
                "decision_sha256": stage["decision_sha256"],
                "requires_completed_stages": [previous["id"] for previous in stages],
                "arms": arms,
            }
        )
    if used_sources != {
        (variant, source) for variant, group in sources.items() for source in group
    }:
        raise ValueError("unused source declaration; include only the treatments being studied")
    plan = copy.deepcopy(
        {
            "schema": SCHEMA,
            "input_sha256": digest_json(config),
            "name": config["name"],
            "model": {**model, "initialization": "fresh_base"},
            "splits": splits,
            "sources": sources,
            "evaluation": evaluation,
            "selection": {
                "source": "fresh_fleet_dev_task_outcomes_only",
                "primary_metric": "fleet_dev_pass_at_1",
                "tie_rule": "retain_all_ties_for_fresh_dev_confirmation",
                "training_loss": "diagnostic_only",
                "teacher_reference_ce": "not_computed_or_used_for_selection",
                "final_fleet_and_webexploitbench": "sealed_until_selection_decision_is_frozen",
            },
            "capacity": {
                **capacity,
                "peak_planned_study_nodes": max(len(s["arms"]) for s in stages)
                + capacity["other_active_study_nodes"],
                "stage_barriers_required": True,
                "stage_barrier": "all_previous_stages_terminal_and_allocations_released",
                "live_aggregate_capacity_recheck_required": True,
            },
            "execution": {
                "kind": "metadata_only_not_a_job_request",
                "dev_cluster_qualification_required": True,
                "new_jobs_authorized": False,
            },
            "stages": stages,
        }
    )
    return {**plan, "sha256": digest_json(plan)}


def _unique_pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("study input contains a non-string or duplicate field")
        result[key] = value
    return result


def read_study(path: Path) -> dict:
    """Read only the explicit metadata input, rejecting silent duplicate keys."""
    text = path.read_text()
    try:
        return json.loads(text, object_pairs_hook=_unique_pairs)
    except json.JSONDecodeError:
        pass

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        lambda loader, node: _unique_pairs(loader.construct_pairs(node)),
    )
    return yaml.load(text, Loader=Loader)


def write_plan(path: Path, plan: dict) -> None:
    """Atomically publish once. Existing files/symlinks are never replaced."""
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != digest_json(payload):
        raise ValueError("study plan digest mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="explicit JSON/YAML metadata manifest")
    parser.add_argument("--output", required=True, type=Path, help="create-once local plan JSON")
    args = parser.parse_args()
    try:
        plan = compile_study(read_study(args.input))
        write_plan(args.output, plan)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        # Parser and OS messages may echo input values or paths; keep errors private.
        parser.exit(
            2, f"study plan not written ({type(exc).__name__}); validate metadata locally\n"
        )
    print(json.dumps({"schema": SCHEMA, "sha256": plan["sha256"], "submitted_jobs": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
