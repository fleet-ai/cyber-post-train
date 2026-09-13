"""Miles reload qualification is offline here; no Ray, GPU or API calls."""

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import miles
from training import miles_reload as reload


def _args(output: str = "/mnt/sfs/jobs/source-miles") -> miles.MilesConfig:
    return miles.MilesConfig(
        name="source-miles",
        output_root=output,
        model_root="/mnt/sfs/models/synthetic/qwen",
        torch_dist_root="/mnt/sfs/jobs/base-miles/checkpoint",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="source-miles",
        nodes=1,
        gpus_per_node=8,
        steps=1,
        groups=1,
        samples_per_prompt=8,
        checkpoint_interval=1,
    )


def _source_plan(source: Path) -> dict:
    arguments = dataclasses.asdict(_args(str(source)))
    return {
        "schema": reload.SOURCE_SCHEMA,
        "run_name": "source-miles",
        "output_root": str(source),
        "model": {"repo": "Qwen/Qwen3.8-27B", "revision": "synthetic"},
        "arguments": arguments,
        "native_driver_sha256": reload.NATIVE_DRIVER_SHA256,
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "128",
                "memory_request": "1536Gi",
                "memory_limit": "2048Gi",
            },
        },
    }


def _terminal_source(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "source"
    checkpoint = source / "checkpoints"
    generation = checkpoint / "iter_0000000"
    generation.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n")
    (generation / ".metadata").write_bytes(b"metadata")
    for rank in range(8):
        (generation / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}".encode())
    plan = _source_plan(source)
    reload._write(
        source / "NATIVE_TRAINING_COMPLETE.json",
        {
            "status": "native_loop_returned",
            "plan_sha256": digest(plan),
            "checkpoint_rollout_index": 0,
            "completed_batches": 3,
            "optimizer_update_independently_verified": False,
            "checkpoint_reload_verified": False,
        },
    )
    return plan, source


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    arguments = dataclasses.asdict(_args())
    resources = {
        "cpu_request": "64",
        "cpu_limit": "128",
        "memory_request": "1536Gi",
        "memory_limit": "2048Gi",
    }
    value = {
        "schema": reload.CHECKPOINT_SCHEMA,
        "image": miles.IMAGE,
        "root": "/mnt/sfs/jobs/source-miles/checkpoints",
        "rollout_index": 0,
        "next_rollout_id": 1,
        "world_size": 8,
        "topology": {"nodes": 1, "gpus_per_node": 8},
        "model": {"repo": "Qwen/Qwen3.8-27B", "revision": "synthetic"},
        "source": {
            "run_name": "source-miles",
            "output_root": arguments["output_root"],
            "plan_sha256": "a" * 64,
            "completion_sha256": "b" * 64,
            "arguments": arguments,
            "execution": {"image": miles.IMAGE, "priority": "c1", "resources": resources},
            "native_driver_sha256": reload.NATIVE_DRIVER_SHA256,
        },
        "files": [
            {
                "path": "latest_checkpointed_iteration.txt",
                "size": 2,
                "sha256": "c" * 64,
            },
            {"path": "iter_0000000/.metadata", "size": 8, "sha256": "d" * 64},
            *(
                {
                    "path": f"iter_0000000/__{rank}_0.distcp",
                    "size": 6,
                    "sha256": f"{rank + 1:x}" * 64,
                }
                for rank in range(8)
            ),
        ],
        "source_optimizer_update_claimed": False,
        "gpu_reload_verified": False,
        "optimizer_update_during_reload": False,
    }
    value["sha256"] = digest(value)
    path = tmp_path / "MILES_TRAINING_CHECKPOINT.json"
    path.write_text(json.dumps(value))
    return value, path


def _config(manifest_path: Path) -> dict:
    return {
        "schema": reload.CONFIG_SCHEMA,
        "name": "q38-miles-reload-dev",
        "output_root": "/mnt/sfs/jobs/q38-miles-reload-dev",
        "checkpoint": {
            "manifest": str(manifest_path),
            "sha256": "sha256:" + reload._hash(manifest_path),
        },
        "cluster": {
            "target": "dev",
            "priority": "c1",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "128",
                "memory_request": "1536Gi",
                "memory_limit": "2048Gi",
            },
        },
    }


def test_cpu_seal_binds_complete_all_rank_numeric_checkpoint(tmp_path, monkeypatch):
    from training import miles_training

    monkeypatch.setattr(miles_training, "job_request", lambda _: {})
    plan, source = _terminal_source(tmp_path)
    output = tmp_path / "seal" / "checkpoint.json"
    output.parent.mkdir()
    result = reload.seal_training_checkpoint(plan, output)

    assert result["schema"] == reload.CHECKPOINT_SCHEMA
    assert result["world_size"] == 8
    assert result["topology"] == {"nodes": 1, "gpus_per_node": 8}
    assert result["rollout_index"] == 0 and result["next_rollout_id"] == 1
    assert len(result["files"]) == 10
    assert result["gpu_reload_verified"] is False
    assert result["sha256"] == digest({k: v for k, v in result.items() if k != "sha256"})
    assert not (source / "ACCEPTED.json").exists()


@pytest.mark.parametrize("fault", ["missing_rank", "symlink", "tracker", "conflict", "receipt"])
def test_cpu_seal_rejects_partial_or_conflicting_source(tmp_path, monkeypatch, fault):
    from training import miles_training

    monkeypatch.setattr(miles_training, "job_request", lambda _: {})
    plan, source = _terminal_source(tmp_path)
    checkpoint = source / "checkpoints"
    if fault == "missing_rank":
        (checkpoint / "iter_0000000/__7_0.distcp").unlink()
    elif fault == "symlink":
        target = checkpoint / "iter_0000000/__7_0.distcp"
        target.unlink()
        target.symlink_to(checkpoint / "iter_0000000/__6_0.distcp")
    elif fault == "tracker":
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n")
    elif fault == "conflict":
        (source / "FAILED.json").write_text("{}")
    else:
        receipt = json.loads((source / "NATIVE_TRAINING_COMPLETE.json").read_text())
        receipt["checkpoint_rollout_index"] = 9
        (source / "NATIVE_TRAINING_COMPLETE.json").write_text(json.dumps(receipt))

    with pytest.raises(ValueError):
        reload.seal_training_checkpoint(plan, tmp_path / "seal.json")


def test_reload_plan_is_dev_only_exact_topology_and_secret_free(tmp_path):
    manifest, path = _manifest(tmp_path)
    plan = reload.compile_reload(_config(path), relative_to=tmp_path)
    request = reload.job_request(plan)

    assert plan["source_manifest"] == manifest
    assert plan["optimizer_updates"] == plan["rollouts"] == 0
    assert plan["execution"]["cluster_target"] == "dev"
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["env"]["WANDB_MODE"] == "disabled"
    assert request["env"]["MILES_EXPERIMENTAL_FT_TRAINER"] == "0"
    assert "FLEET_API_KEY" not in request["env"]
    assert "WANDB_API_KEY" not in request["env"]


@pytest.mark.parametrize(
    "fault", ["prod", "c2", "digest", "world", "fragmented", "underreserve", "overlap"]
)
def test_reload_plan_rejects_unqualified_bindings(tmp_path, fault):
    manifest, path = _manifest(tmp_path)
    config = _config(path)
    if fault == "prod":
        config["cluster"]["target"] = "prod"
    elif fault == "c2":
        config["cluster"]["priority"] = "c2"
    elif fault == "digest":
        config["checkpoint"]["sha256"] = "0" * 64
    elif fault == "world":
        manifest["world_size"] = 7
        manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
        path.write_text(json.dumps(manifest))
        config["checkpoint"]["sha256"] = reload._hash(path)
    elif fault == "fragmented":
        manifest["topology"] = {"nodes": 2, "gpus_per_node": 4}
        manifest["source"]["arguments"].update({"nodes": 2, "gpus_per_node": 4})
        manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
        path.write_text(json.dumps(manifest))
        config["checkpoint"]["sha256"] = reload._hash(path)
    elif fault == "underreserve":
        config["cluster"]["resources"]["memory_request"] = "64Gi"
    else:
        config["output_root"] = "/mnt/sfs/jobs/source-miles/reload"

    with pytest.raises(ValueError):
        reload.compile_reload(config, relative_to=tmp_path)


def test_argument_transform_keeps_recovery_and_removes_side_effects(monkeypatch):
    source = _args()
    original = [
        "--load",
        "/mnt/sfs/jobs/base/checkpoint",
        "--ref-load",
        "/mnt/sfs/jobs/base/checkpoint",
        "--save",
        source.output_root + "/checkpoints",
        "--save-interval",
        "1",
        "--num-rollout",
        "1",
        "--eval-interval",
        "1",
        "--eval-prompt-data",
        "fleet-dev",
        source.dev_data,
        "--wandb-team",
        "synthetic",
        "--wandb-mode",
        "online",
        "--colocate",
        "--offload-train",
        "--use-kl-loss",
        "--use-wandb",
    ]
    monkeypatch.setattr(miles, "arguments", lambda _: original)
    argv = reload.reload_arguments(source, "/mnt/sfs/jobs/source-miles/checkpoints")

    assert argv[argv.index("--load") + 1] == "/mnt/sfs/jobs/source-miles/checkpoints"
    assert "--num-rollout" in argv and argv[argv.index("--num-rollout") + 1] == "1"
    assert "--debug-train-only" in argv
    assert "--use-checkpoint-opt-param-scheduler" in argv
    assert not ({*reload._REMOVE_ONE_VALUE, *reload._REMOVE_FLAGS} & set(argv))
    assert not (reload._FORBIDDEN_FLAGS & set(argv))


@pytest.mark.parametrize("flag", sorted(reload._FORBIDDEN_FLAGS))
def test_argument_transform_rejects_recovery_bypass(monkeypatch, flag):
    monkeypatch.setattr(
        miles,
        "arguments",
        lambda _: ["--load", "/mnt/sfs/jobs/base/checkpoint", flag],
    )
    with pytest.raises(ValueError, match="disable exact optimizer/RNG recovery"):
        reload.reload_arguments(_args(), "/mnt/sfs/jobs/source-miles/checkpoints")


def _rank(rank: int) -> dict:
    return {
        "rank": rank,
        "world_size": 8,
        "load": "/mnt/sfs/jobs/source-miles/checkpoints",
        "save_is_none": True,
        "debug_train_only": True,
        "no_load_optim": False,
        "no_load_rng": False,
        "finetune": False,
        "use_checkpoint_opt_param_scheduler": True,
        "model": {"tensors": 10, "local_numel": 100, "structure_sha256": str(rank)},
        "optimizer": {
            "objects": 2,
            "state_entries": 3,
            "parameter_groups": 1,
            "structure_sha256": str(rank),
        },
        "scheduler": {"positive_progress_counters": 1, "state_sha256": str(rank)},
        "rng_sha256": str(rank),
    }


def test_all_rank_probe_requires_stable_model_optimizer_scheduler_and_rng():
    manifest = {
        "world_size": 8,
        "rollout_index": 0,
        "next_rollout_id": 1,
        "root": "/mnt/sfs/jobs/source-miles/checkpoints",
    }
    rows = [_rank(rank) for rank in range(8)]
    result = reload.validate_rank_probes(manifest, [1] * 8, rows, rows)
    assert result["ranks"] == list(range(8))
    assert result["all_rank_optimizer_loaded"] is True
    assert result["all_rank_rng_loaded"] is True
    assert result["state_stable_across_zero_updates"] is True

    with pytest.raises(ValueError, match="incomplete"):
        reload.validate_rank_probes(manifest, [1] * 8, rows[:-1], rows[:-1])
    changed = [dict(row) for row in rows]
    changed[7] = {**changed[7], "rng_sha256": "changed"}
    with pytest.raises(ValueError, match="state changed"):
        reload.validate_rank_probes(manifest, [1] * 8, rows, changed)
    empty = [dict(row) for row in rows]
    empty[0] = {**empty[0], "optimizer": {**empty[0]["optimizer"], "state_entries": 0}}
    with pytest.raises(ValueError, match="recoverable training state"):
        reload.validate_rank_probes(manifest, [1] * 8, empty, empty)


def test_cli_prepares_and_dispatches_miles_reload_without_network(tmp_path, monkeypatch):
    manifest, path = _manifest(tmp_path)
    config = tmp_path / "reload.json"
    config.write_text(json.dumps(_config(path)))
    prepared = tmp_path / "prepared"
    runner = CliRunner()

    result = runner.invoke(cli.app, ["miles-rl-reload", str(config), "--output", str(prepared)])
    assert result.exit_code == 0
    plan, request = cli._prepared(prepared)
    assert plan["source_manifest"] == manifest
    assert request["workers"] * request["gpus_per_worker"] == 8

    proof = {
        "schema": reload.PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    monkeypatch.setattr(reload, "preflight", lambda _: proof)
    assert runner.invoke(cli.app, ["preflight", str(prepared)]).exit_code == 0


@pytest.mark.parametrize("command", ["preview", "submit"])
def test_miles_reload_cannot_be_routed_to_production(tmp_path, monkeypatch, command):
    manifest, path = _manifest(tmp_path)
    plan = reload.compile_reload(_config(path), relative_to=tmp_path)
    request = reload.job_request(plan)
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_client", lambda _: pytest.fail("reached production API"))

    result = CliRunner().invoke(cli.app, [command, str(tmp_path), "--cluster", "prod"])
    assert result.exit_code == 2
    assert "dev-cluster-only" in result.stderr


def test_template_is_intentionally_unmaterialized_dev_c1():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/qualification/qwen38-miles-rl-reward-canary-reload-dev-v1.template.json"
    value = json.loads(path.read_text())
    assert value["schema"] == reload.CONFIG_SCHEMA
    assert value["name"] == "chris-q38-miles-reload-dev3"
    assert value["output_root"] == "/mnt/sfs/jobs/chris-q38-miles-reload-dev3"
    assert value["checkpoint"]["manifest"] == (
        "/mnt/sfs/jobs/chris-q38-miles-rlreward-dev3-seal/MILES_TRAINING_CHECKPOINT.json"
    )
    assert value["cluster"]["target"] == "dev"
    assert value["cluster"]["priority"] == "c1"
    assert "REPLACE_WITH" in value["checkpoint"]["sha256"]
