"""The study compiler validates metadata only and cannot launch work."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from training.io import digest_json
from training.study_plan import compile_study, read_study, write_plan


def sha(character):
    return "sha256:" + character * 64


@pytest.fixture
def config():
    splits = {
        variant: {
            "variant_seed": f"dev-search-{variant}",
            "study_split_sha256": sha(outer),
            "training_split_sha256": sha(train),
            "dev_task_set_sha256": sha(dev),
            "fleet_dev_protocol_sha256": sha(protocol),
            "final_test_lock_sha256": sha("8"),
        }
        for variant, outer, train, dev, protocol in (
            ("a", "a", "b", "0", "3"),
            ("b", "c", "d", "4", "5"),
        )
    }
    return {
        "name": "q38-dev-study",
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "lock_sha256": sha("a"),
        },
        "splits": splits,
        "sources": {
            variant: {
                source: {
                    "corpus_sha256": sha(corpus),
                    "source_selection_sha256": sha(selection),
                    "status": "qualified",
                    "source_sessions": 10,
                    "qualification_sha256": sha(selection),
                    **{
                        k: splits[variant][k]
                        for k in (
                            "study_split_sha256",
                            "training_split_sha256",
                            "fleet_dev_protocol_sha256",
                        )
                    },
                }
                for source, corpus, selection in (
                    ("teacher", "1" if variant == "a" else "6", "a" if variant == "a" else "b"),
                    ("self", "2" if variant == "a" else "7", "c" if variant == "a" else "d"),
                )
            }
            for variant in splits
        },
        "wandb": {"entity": "test-team", "project": "test-project"},
        "capacity": {
            "max_active_nodes": 8,
            "other_active_study_nodes": 2,
            "excluded_inference_endpoints": 4,
        },
        "evaluation": {
            "final_fleet": {
                "task_set_sha256": sha("1"),
                "protocol_sha256": sha("3"),
                "lock_sha256": sha("8"),
            },
            "final_webexploitbench": {"task_set_sha256": sha("2"), "protocol_sha256": sha("3")},
        },
        "stages": [
            {
                "id": "source-screen",
                "decision_sha256": None,
                "arms": [
                    {
                        "id": f"{source}-lr3",
                        "split": "a",
                        "source": source,
                        "lr": 3e-6,
                        "batch_size": 8,
                        "epochs": 1,
                        "seed": 42,
                    }
                    for source in ("teacher", "self")
                ],
            },
            {
                "id": "split-check",
                "decision_sha256": sha("9"),
                "arms": [
                    {
                        "id": f"{source}-b",
                        "split": "b",
                        "source": source,
                        "lr": 3e-6,
                        "batch_size": 8,
                        "epochs": 1,
                        "seed": 42,
                    }
                    for source in ("teacher", "self")
                ],
            },
        ],
    }


def test_deterministic_metadata_only_plan(config, monkeypatch):
    original = copy.deepcopy(config)

    def forbid(*args, **kwargs):
        raise AssertionError("compiler must not read task/source files")

    monkeypatch.setattr(Path, "read_text", forbid)
    monkeypatch.setattr(Path, "read_bytes", forbid)
    plan = compile_study(config)
    assert config == original
    assert plan == compile_study(config)
    assert plan["input_sha256"] == digest_json(config)
    assert plan["sha256"] == digest_json({k: v for k, v in plan.items() if k != "sha256"})
    assert plan["model"]["initialization"] == "fresh_base"
    assert plan["execution"]["new_jobs_authorized"] is False
    assert plan["execution"]["dev_cluster_qualification_required"] is True
    assert plan["capacity"]["peak_planned_study_nodes"] == 4
    assert plan["capacity"]["live_aggregate_capacity_recheck_required"] is True
    assert plan["stages"][0]["requires_completed_stages"] == []
    assert plan["stages"][1]["requires_completed_stages"] == ["source-screen"]
    for arm in (arm for stage in plan["stages"] for arm in stage["arms"]):
        assert arm["run_name"] == arm["wandb"]["run_id"]
        assert arm["output_root"] == "/mnt/sfs/jobs/" + arm["run_name"]
        assert arm["wandb"]["group"] == config["name"]
        assert (arm["nodes"], arm["gpus_per_node"], arm["microbatch_per_gpu"]) == (1, 8, 1)
        assert arm["data"] == config["sources"][arm["split"]][arm["source"]]
        assert arm["validation_mode"] == "task_outcomes_only" and arm["eval_interval"] == 0
        assert (
            arm["fleet_dev_protocol_sha256"]
            == config["splits"][arm["split"]]["fleet_dev_protocol_sha256"]
        )
    config["sources"]["a"]["teacher"]["corpus_sha256"] = sha("9")
    assert plan["sources"]["a"]["teacher"]["corpus_sha256"] == sha("1")


def test_selection_never_uses_ce_or_training_loss(config):
    selection = compile_study(config)["selection"]
    assert selection == {
        "source": "fresh_fleet_dev_task_outcomes_only",
        "primary_metric": "fleet_dev_pass_at_1",
        "tie_rule": "retain_all_ties_for_fresh_dev_confirmation",
        "training_loss": "diagnostic_only",
        "teacher_reference_ce": "not_computed_or_used_for_selection",
        "final_fleet_and_webexploitbench": "sealed_until_selection_decision_is_frozen",
    }


def test_single_source_is_supported_without_unused_declarations(config):
    del config["sources"]["a"]["self"]
    del config["sources"]["b"]["self"]
    config["stages"][1]["arms"].pop()
    config["stages"][0]["arms"].pop()
    assert all(set(group) == {"teacher"} for group in compile_study(config)["sources"].values())


@pytest.mark.parametrize(
    "section", [None, "model", "splits", "sources", "wandb", "capacity", "evaluation"]
)
def test_unknown_or_missing_fields_fail(config, section):
    target = config if section is None else config[section]
    target["grid"] = {"lr": [1e-6, 3e-6]}
    with pytest.raises(ValueError):
        compile_study(config)
    target.pop("grid")
    target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        compile_study(config)


@pytest.mark.parametrize(
    "field",
    [
        "corpus_sha256",
        "study_split_sha256",
        "training_split_sha256",
        "source_selection_sha256",
        "fleet_dev_protocol_sha256",
    ],
)
@pytest.mark.parametrize("value", ["a" * 64, "sha256:short", "sha256:" + "A" * 64, 3, None])
def test_every_source_binding_must_be_a_digest(config, field, value):
    config["sources"]["a"]["teacher"][field] = value
    with pytest.raises(ValueError):
        compile_study(config)


def test_split_mismatch_and_evaluation_identity_reuse_fail(config):
    config["sources"]["a"]["self"]["training_split_sha256"] = sha("a")
    with pytest.raises(ValueError, match="own exact split"):
        compile_study(config)
    config["sources"]["a"]["self"]["training_split_sha256"] = sha("b")
    config["evaluation"]["final_fleet"]["task_set_sha256"] = sha("0")
    with pytest.raises(ValueError, match="identities must be distinct"):
        compile_study(config)


@pytest.mark.parametrize(
    "other,count,limit,valid",
    [
        (0, 8, 8, True),
        (0, 9, 8, False),
        (6, 2, 8, True),
        (7, 2, 8, False),
        (0, 2, 9, False),
        (0, 2, 1, False),
    ],
)
def test_capacity_counts_one_node_per_arm_and_other_allocations(config, other, count, limit, valid):
    config["capacity"] = {
        "other_active_study_nodes": other,
        "max_active_nodes": limit,
        "excluded_inference_endpoints": 4,
    }
    config["splits"].pop("b")
    config["sources"].pop("b")
    config["stages"] = config["stages"][:1]
    base = config["stages"][0]["arms"][0]
    config["stages"][0]["arms"] = [
        {**base, "id": f"arm{i}", "source": "teacher" if i % 2 == 0 else "self", "seed": i}
        for i in range(count)
    ]
    if valid:
        plan = compile_study(config)
        assert plan["capacity"]["peak_planned_study_nodes"] == other + count
    else:
        with pytest.raises(ValueError):
            compile_study(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", [1e-6, 3e-6]),
        ("lr", "3e-6"),
        ("lr", True),
        ("lr", 0),
        ("lr", -1e-6),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("lr", 10**1000),
        ("batch_size", 7),
        ("batch_size", 8.0),
        ("batch_size", 0),
        ("epochs", [1, 2]),
        ("epochs", True),
        ("epochs", 0),
        ("seed", -1),
        ("seed", 2**32),
        ("source", ["teacher", "self"]),
        ("source", "other"),
        ("id", "../escape"),
        ("id", "a" * 31),
    ],
)
def test_treatments_are_valid_scalar_rows(config, key, value):
    config["stages"][0]["arms"][0][key] = value
    with pytest.raises(ValueError):
        compile_study(config)


@pytest.mark.parametrize("case", ["stage", "arm", "treatment"])
def test_no_duplicate_identities_or_hidden_repeats(config, case):
    if case == "stage":
        config["stages"][1]["id"] = config["stages"][0]["id"]
    elif case == "arm":
        config["stages"][1]["arms"][0]["id"] = config["stages"][0]["arms"][0]["id"]
    else:
        config["stages"][1]["arms"][0] = {**config["stages"][0]["arms"][0], "id": "repeat"}
    with pytest.raises(ValueError, match="duplicate"):
        compile_study(config)


def test_same_treatment_distinct_training_seed_is_explicit_replication(config):
    config["stages"][1]["arms"].append(
        {**config["stages"][0]["arms"][0], "id": "repeat", "seed": 43}
    )
    assert len(compile_study(config)["stages"]) == 2


@pytest.mark.parametrize(
    "defect",
    ["blocked", "empty", "qualification", "protocol", "final_lock", "decision", "excluded"],
)
def test_source_and_followup_gates_fail_closed(config, defect):
    source = config["sources"]["a"]["self"]
    if defect == "blocked":
        source["status"] = "blocked_no_compatible_sources"
    elif defect == "empty":
        source["source_sessions"] = 0
    elif defect == "qualification":
        source["qualification_sha256"] = "pending"
    elif defect == "protocol":
        source["fleet_dev_protocol_sha256"] = sha("f")
    elif defect == "final_lock":
        config["splits"]["b"]["final_test_lock_sha256"] = sha("f")
    elif defect == "decision":
        config["stages"][1]["decision_sha256"] = None
    else:
        config["capacity"]["excluded_inference_endpoints"] = 5
    with pytest.raises(ValueError):
        compile_study(config)


@pytest.mark.parametrize("value", [[], {}, "lr-grid", None])
def test_no_implicit_stage_or_arm_expansion(config, value):
    config["stages"] = value
    with pytest.raises(ValueError):
        compile_study(config)


@pytest.mark.parametrize("kind", ["json", "yaml"])
def test_input_roundtrip_and_reject_duplicate_fields(config, tmp_path, kind):
    path = tmp_path / "study"
    path.write_text(json.dumps(config) if kind == "json" else yaml.safe_dump(config))
    assert read_study(path) == config
    duplicate = (
        '{"name":"first","name":"second"}' if kind == "json" else "name: first\nname: second\n"
    )
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        read_study(path)


def test_create_once_publication_and_tamper_detection(config, tmp_path):
    plan = compile_study(config)
    path = tmp_path / "plans" / "plan.json"
    write_plan(path, plan)
    original = path.read_bytes()
    assert json.loads(original) == plan
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_plan(path, plan)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    plan["stages"][0]["arms"][0]["lr"] = 1e-3
    with pytest.raises(ValueError, match="digest mismatch"):
        write_plan(tmp_path / "tampered.json", plan)
    assert not (tmp_path / "tampered.json").exists()


def test_symlink_is_not_replaced(config, tmp_path):
    victim = tmp_path / "original"
    victim.write_text("preserve")
    path = tmp_path / "plan.json"
    path.symlink_to(victim)
    with pytest.raises(FileExistsError):
        write_plan(path, compile_study(config))
    assert victim.read_text() == "preserve"
    assert path.is_symlink()


def test_cli_writes_only_plan_and_reports_zero_submissions(config, tmp_path):
    path, output = tmp_path / "study.json", tmp_path / "plan.json"
    path.write_text(json.dumps(config))
    command = [sys.executable, "-m", "training.study_plan", str(path), "--output", str(output)]
    env = {**os.environ, "WANDB_MODE": "disabled"}
    result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
    assert json.loads(result.stdout)["submitted_jobs"] == 0
    assert json.loads(output.read_text())["sha256"] == compile_study(config)["sha256"]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "FileExistsError" in result.stderr


def test_tracked_template_is_explicit_and_gated_against_current_split_metadata():
    root = Path(__file__).resolve().parents[1]
    template = read_study(root / "configs/studies/qwen-blackbox-sft-v1.template.json")
    with pytest.raises(ValueError):
        compile_study(template)
    for variant in ("a", "b"):
        split = json.loads(
            (root / f"configs/data/qwen-blackbox-study-split-{variant}-v1.json").read_text()
        )
        binding = template["splits"][variant]
        assert binding["study_split_sha256"] == split["sha256"]
        assert binding["training_split_sha256"] == split["training_split"]["sha256"]
        assert binding["dev_task_set_sha256"] == split["evaluation"]["dev"]["sha256"]
        assert binding["variant_seed"] == split["variant_seed"]
        assert template["sources"][variant]["self"]["status"] == "blocked_no_compatible_sources"
        assert template["sources"][variant]["self"]["source_sessions"] == 0
    first, *followups = template["stages"]
    assert first["decision_sha256"] is None
    assert len(first["arms"]) == 8
    for source in ("teacher", "self"):
        rows = [r for r in first["arms"] if r["source"] == source]
        assert [r["lr"] for r in rows] == [1e-6, 3e-6, 1e-5, 3e-5]
        assert all(r["epochs"] == 1 and r["batch_size"] == 8 and r["split"] == "a" for r in rows)
    assert all(
        len(s["arms"]) == 4 and s["decision_sha256"].startswith("PENDING_") for s in followups
    )
    assert all(isinstance(r["lr"], str) for s in followups for r in s["arms"])


def test_frozen_teacher_lr_screen_recompiles_exactly():
    root = Path(__file__).resolve().parents[1]
    source = read_study(
        root / "configs/studies/qwen-blackbox-teacher-a-lr-screen-v1.json"
    )
    frozen = json.loads(
        (root / "configs/studies/qwen-blackbox-teacher-a-lr-screen-v1.plan.json").read_text()
    )
    assert compile_study(source) == frozen
    assert len(frozen["stages"]) == 1
    assert [arm["lr"] for arm in frozen["stages"][0]["arms"]] == [
        1e-6,
        3e-6,
        1e-5,
        3e-5,
    ]
    assert frozen["capacity"]["peak_planned_study_nodes"] == 4
