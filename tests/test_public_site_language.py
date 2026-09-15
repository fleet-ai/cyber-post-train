import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ROOT / "site" / "index.html",
    ROOT / "site" / "report-data.js",
    ROOT / "site" / "app.js",
    ROOT / "site" / "training-decision-space.json",
]


def test_public_report_does_not_reintroduce_unexplained_internal_terms() -> None:
    text = "\n".join(path.read_text().lower() for path in PUBLIC_FILES)
    unexplained_internal_phrases = {
        "study roster",
        "current binding",
        "runtime seed",
        "run receipt",
        "deterministic verifier",
        "objective hits",
        "adjudication",
        "ots cyber",
        "census binds",
        "task keys",
    }

    found = sorted(phrase for phrase in unexplained_internal_phrases if phrase in text)
    assert not found, f"replace internal language with a plain explanation: {found}"


def test_training_terms_have_visible_definitions() -> None:
    html = (ROOT / "site" / "index.html").read_text()

    assert "Supervised fine-tuning (SFT)" in html
    assert "Reinforcement learning (RL)" in html
    assert "Select any column title to read its plain-language meaning" in html


def test_public_task_report_points_to_the_latest_exact_inventory() -> None:
    html = (ROOT / "site" / "index.html").read_text()
    data = (ROOT / "site" / "report-data.js").read_text()

    assert "1,055 current blackbox tasks" in html
    assert "fleet-blackbox-current-production-20260915-v1.json" in html
    assert "docs/FLEET_BLACKBOX_TASK_REFRESH_2026-09-15.md" in html
    assert 'funnel: [["Current production blackbox tasks",1055]' in data


def test_experiment_plan_has_two_large_ranked_run_tables() -> None:
    html = (ROOT / "site" / "index.html").read_text()
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())

    assert 'id="sft-plan-table"' in html
    assert 'id="rl-plan-table"' in html
    assert set(data["tables"]) == {"sft", "rl"}
    assert "decisions" not in data

    minimum_columns = {"sft": 75, "rl": 90}
    minimum_runs = {"sft": 20, "rl": 20}
    required_run_fields = {
        "rank",
        "run_id",
        "research_question",
        "status",
        "why_priority",
        "compare_with",
        "launch_after",
    }

    for kind, table in data["tables"].items():
        columns = [column for group in table["groups"] for column in group["columns"]]
        keys = [column[0] for column in columns]
        assert len(columns) >= minimum_columns[kind]
        assert len(table["runs"]) >= minimum_runs[kind]
        assert len(keys) == len(set(keys))
        assert all(len(column) == 3 and all(column) for column in columns)
        assert [run["rank"] for run in table["runs"]] == list(range(1, len(table["runs"]) + 1))
        assert len({run["run_id"] for run in table["runs"]}) == len(table["runs"])

        for run in table["runs"]:
            assert required_run_fields <= run.keys()
            assert set(run.get("changes", {})) <= set(keys)
            for key in keys:
                value = run.get(key, run.get("changes", {}).get(key, table["defaults"].get(key)))
                assert value not in (None, ""), f"{kind} {run['run_id']} does not specify {key}"


def test_experiment_plan_uses_current_split_and_current_execution_policy() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    guide = (ROOT / "docs" / "TRAINING_DECISION_SPACE.md").read_text()
    sft = data["tables"]["sft"]
    rl = data["tables"]["rl"]

    for defaults in (sft["defaults"], rl["defaults"]):
        assert defaults["train_pool"].startswith("50 ")
        assert defaults["dev_pool"].startswith("17 ")
        assert defaults["final_pool"].startswith("8 ")
        assert defaults["external_harness"] == "Frozen OpenCode 1.18.27 only"
        assert "never a selection signal" in defaults["webexploitbench"]

    serialized = json.dumps(data)
    assert "Qwen Code" not in serialized
    assert rl["runs"][0]["run_id"] == "RL-01-miles-one-update-gate"
    assert rl["defaults"]["context_tokens"].startswith("262,144 ")
    assert rl["defaults"]["max_turns"] == "1,200"
    assert "75-task execution-proven set" in guide
    assert "exact 50/17/8 split" in guide
    assert "262,144-token live context" in guide
    assert "Use SkyRL only as a matched fallback" in guide


def test_completed_fresh75_row_matches_the_accepted_manifest_and_run_config() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    manifest = json.loads(
        (ROOT / "configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json").read_text()
    )
    config = json.loads(
        (ROOT / "configs/runs/qwen38-fresh75-teacher-sft-largest-full-v4.json").read_text()
    )
    table = data["tables"]["sft"]
    run = table["runs"][0]
    values = {**table["defaults"], **run["changes"]}
    sources = manifest["catalog_provenance"]["teacher_model_sessions"]

    assert run["run_id"] == "SFT-01-teacher-broad / Fresh75 step 230"
    assert run["status"] == "Training completed; evaluation pending"
    assert values["teacher_models"] == (
        f"{sources['gpt-5.6-sol']} GPT sessions, {sources['grok-4.5']} Grok session, "
        f"and {sources['kimi-k3']} Kimi session"
    )
    assert values["global_batch"] == f"{config['recipe']['batch_size']} examples"
    assert values["epochs"] == str(config["recipe"]["epochs"])
    assert values["learning_rate"] == "1e-5"
    assert values["checkpoint_every"] == (
        f"{config['recipe']['checkpoint_interval']} optimizer updates"
    )
    assert values["shuffle_seed"] == str(config["recipe"]["seed"])
    assert str(manifest["files"]["train"]["source_sessions"]) in values["trace_cap"]
    assert str(manifest["catalog_provenance"]["selected_task_versions"]) in values["task_balance"]
    assert "evaluation is still pending" in values["checkpoint_choice"]


def test_primary_sft_source_comparisons_hold_the_accepted_dose_constant() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    manifest = json.loads(
        (ROOT / "configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json").read_text()
    )
    table = data["tables"]["sft"]
    accepted = {**table["defaults"], **table["runs"][0]["changes"]}
    controlled = {
        "learning_rate",
        "global_batch",
        "microbatch",
        "gradient_accumulation",
        "epochs",
        "checkpoint_every",
        "shuffle_seed",
    }
    tokens_per_pass = manifest["files"]["train"]["supervised_tokens"]

    for run in table["runs"][1:4]:
        values = {**table["defaults"], **run["changes"]}
        assert {key: values[key] for key in controlled} == {
            key: accepted[key] for key in controlled
        }
        assert f"{tokens_per_pass:,}" in values["data_fraction"]
        assert f"{tokens_per_pass * int(accepted['epochs']):,}" in values["relative_cost"]
        assert "230 optimizer updates" in values["relative_cost"]


def test_sft_source_comparisons_hold_family_coverage_and_weight_constant() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["sft"]

    for run_id in ("SFT-02-self-broad", "SFT-03-mix-50-50"):
        run = next(run for run in table["runs"] if run["run_id"] == run_id)
        values = {**table["defaults"], **run["changes"]}
        assert values["train_pool"] == "The exact same 37 training task families covered by SFT-01"
        assert "same 37 task families" in values["task_balance"]
        assert "exact supervised-token total" in values["task_balance"]
        assert "same 37 task families" in run["launch_after"]
        assert "family" in run["launch_after"] and "target-token weight" in run["launch_after"]


def test_larger_sft_batches_use_native_kept_tail_update_counts() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    manifest = json.loads(
        (ROOT / "configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json").read_text()
    )
    table = data["tables"]["sft"]
    rows = manifest["files"]["train"]["rows"]
    epochs = int(table["defaults"]["epochs"])

    for run_id, batch in (("SFT-08-batch-32", 32), ("SFT-09-batch-64", 64)):
        run = next(run for run in table["runs"] if run["run_id"] == run_id)
        expected = epochs * math.ceil(rows / batch)
        explanation = run["changes"]["relative_cost"]
        assert expected in (58, 30)
        assert f"{expected} optimizer updates" in explanation
        assert f"ceil({rows} examples ÷ {batch})" in explanation
        assert "keeps the last partial batch" in explanation


def test_sft_seed_repeat_changes_both_recorded_random_seeds() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["sft"]
    run = next(run for run in table["runs"] if run["run_id"] == "SFT-22-seed-repeat")

    assert "20260915" in run["changes"]["data_seed"]
    assert run["changes"]["shuffle_seed"] == "20260915"
    assert "data and training random seeds 20260914" in run["compare_with"]


def test_rl_primary_contract_and_skyrl_control_are_not_mixed() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    defaults = table["defaults"]
    miles = next(run for run in table["runs"] if run["run_id"] == "RL-03-base-miles-10u")
    skyrl = next(run for run in table["runs"] if run["run_id"] == "RL-07-skyrl-matched-10u")

    assert defaults["nodes"] == "4"
    assert defaults["gpus_per_node"] == "8"
    assert "8 GPUs" in defaults["tensor_parallel"]
    assert "context parallel size 4" in defaults["tensor_parallel"]
    assert "OpenCode 1.18.27" in defaults["harness"]
    assert "SkyRL" in skyrl["changes"]["backend"]
    assert "SkyRL" in skyrl["changes"]["algorithm"]
    assert "kl_coefficient" not in skyrl["changes"]
    assert "clip_upper" not in skyrl["changes"]
    for key in (
        "train_tasks_per_update",
        "task_groups_per_update",
        "total_rollouts_update",
        "updates",
        "global_episode_batch",
        "minibatch",
    ):
        assert skyrl["changes"][key] == miles["changes"][key]
    for key in ("eval_interval", "checkpoint_interval"):
        assert {**defaults, **skyrl["changes"]}[key] == {**defaults, **miles["changes"]}[key]


def test_rl_invalid_attempts_never_become_zero_reward() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    defaults = data["tables"]["rl"]["defaults"]

    assert "exclude" in defaults["invalid_retry"].lower()
    assert "at most one" in defaults["invalid_retry"].lower()
    assert "abort the update" in defaults["invalid_retry"].lower()
    assert "exclude from rewards and optimizer input" in defaults["invalid_handling"].lower()
    assert "never convert" in defaults["invalid_handling"].lower()


def test_rl_matched_ten_update_runs_share_checkpoint_and_eval_cadence() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    defaults = table["defaults"]
    ten_update_runs = []

    for run in table["runs"]:
        values = {**defaults, **run["changes"]}
        if values["updates"] == "10":
            ten_update_runs.append(run["run_id"])
            assert values["eval_interval"] == "Every 5 updates"
            assert values["checkpoint_interval"] == "Every 5 updates"

    assert "RL-03-base-miles-10u" in ten_update_runs
    assert "RL-04-teacher-sft-start" in ten_update_runs
    assert "RL-07-skyrl-matched-10u" in ten_update_runs


def test_rl_one_update_gate_has_operational_not_lift_acceptance() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    run = next(run for run in table["runs"] if run["run_id"] == "RL-01-miles-one-update-gate")
    rule = run["changes"]["advance_rule"].lower()

    assert "reward variation" in rule
    assert "finite nonzero optimizer update" in rule
    assert "checkpoint reloads" in rule
    assert "cleanly released" in rule
    assert "fleet development lift is not required" in rule


def test_every_current_training_run_uses_c1_priority() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())

    for name, table in data["tables"].items():
        column_keys = [key for group in table["groups"] for key, _, _ in group["columns"]]
        assert "priority" in column_keys
        assert table["defaults"]["priority"] == "c1"
        for run in table["runs"]:
            assert {**table["defaults"], **run["changes"]}["priority"] == "c1", (
                name,
                run["run_id"],
            )


def test_webexploitbench_runs_after_acceptance_but_never_selects() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    guide = (ROOT / "docs" / "TRAINING_DECISION_SPACE.md").read_text()

    for table in data["tables"].values():
        rule = table["defaults"]["webexploitbench"].lower()
        assert "immediately after every accepted checkpoint" in rule
        assert "never a selection signal" in rule
    assert "Run the matched report-only WebExploitBench transfer check immediately" in guide
    assert "never use those benchmark results to choose" in guide


def test_alternate_splits_lock_the_exact_final_holdout() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    for table_name, run_id in (("sft", "SFT-05-split-b"), ("rl", "RL-23-split-b")):
        run = next(
            run for run in data["tables"][table_name]["runs"] if run["run_id"] == run_id
        )
        assert run["status"].startswith("Blocked until the splitter")
        assert "preserving the exact final 8" in run["launch_after"]
        assert run["changes"]["final_pool"].startswith("The same exact 8 untouched")


def test_current_cluster_and_public_benchmark_policy_is_unambiguous() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    cluster = (ROOT / "docs" / "CLUSTER_ALERTS_AND_INFERENCE_SERVING.md").read_text()
    html = (ROOT / "site" / "index.html").read_text()

    assert "priority `c1` for current project work" in agents
    assert "Request pod priority `c1` for all current project work" in cluster
    assert "`c2`" not in agents + cluster
    assert "never submit at `c0`" in cluster
    assert "development cluster is for bounded qualification and debugging only" in agents
    assert "fixed deadline" in agents
    assert "verify the exact UIDs are absent" in agents
    assert "fixed wall-clock deadline" in cluster
    assert "teardown path before it is created" in cluster
    assert "WebExploitBench is a report-only transfer check" in html
    assert "development evidence" not in html
