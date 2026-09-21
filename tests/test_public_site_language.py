import json
import math
import re
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
        "dose matcher",
        "request prefix",
        "target-token exposures",
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


def test_fleet_evaluation_page_does_not_turn_incomplete_attempts_into_scores() -> None:
    html = (ROOT / "site" / "index.html").read_text()
    data = (ROOT / "site" / "report-data.js").read_text()

    assert 'data-tab-page="fleet-evaluations"' in html
    assert "We have not established either an improvement or a decline." in html
    assert '"Fair trained-versus-original comparisons", "0"' in data
    assert '"17 of 17 completed, but not a fair comparison"' in data


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
    assert "Use SkyRL only as a separately reported native" in guide


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
        assert "family" in run["launch_after"] and "supervised-token weight" in run["launch_after"]


def test_every_self_trace_row_renders_the_full_source_quality_contract() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["sft"]
    required_overrides = {
        "source_manifest",
        "source_harness",
        "conversation_format",
        "tools",
        "compaction_rule",
        "source_context",
        "prompt_version",
        "malformed_trace_rule",
        "target_actions",
        "final_report_weight",
        "terminal_only",
    }
    self_rows = [
        run
        for run in table["runs"]
        if "Qwen successes" in run.get("changes", {}).get("trace_source", "")
    ]
    assert len(self_rows) >= 8
    for run in self_rows:
        assert required_overrides <= run["changes"].keys(), run["run_id"]
        values = {**table["defaults"], **run["changes"]}
        assert "SHA-256" in values["source_manifest"]
        assert "runtime-file" in values["source_manifest"]
        assert "source-record" in values["source_manifest"]
        assert values["train_pool"] == "The exact same 37 training task families covered by SFT-01"
        assert "same 37" in values["task_balance"]
        assert "OpenCode 1.18.27" in values["source_harness"]
        assert "4af5494f9433f59d" in values["source_harness"]
        assert "262,144" in values["source_context"]
        assert "20,000" in values["source_context"]
        assert "byte for byte" in values["compaction_rule"]
        assert "Exclude" in values["malformed_trace_rule"]
        assert "non-submit" in values["target_actions"]
        assert (
            "final-report" in values["final_report_weight"]
            or "final report" in values["final_report_weight"]
        )
        assert values["terminal_only"].startswith("No")


def test_a_nonbase_rl_start_never_inherits_the_base_revision() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    base_checkpoint = table["defaults"]["start_checkpoint"]
    base_revision = table["defaults"]["model_revision"]
    changed = 0
    for run in table["runs"]:
        values = {**table["defaults"], **run["changes"]}
        if values["start_checkpoint"] != base_checkpoint:
            changed += 1
            assert "model_revision" in run["changes"]
            assert values["model_revision"] != base_revision
            assert "SHA-256" in values["model_revision"] or re.fullmatch(
                r"[a-f0-9]{64}", values["model_revision"]
            )
    assert changed == 9


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


def test_every_sft_dose_change_has_an_exact_kept_tail_update_count() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["sft"]
    expected = {
        "SFT-10-one-epoch": (916, 8, 1, 115),
        "SFT-11-four-epochs": (916, 8, 4, 460),
        "SFT-14-window-8k": (916, 8, 2, 230),
        "SFT-15-window-32k": (916, 8, 2, 230),
        "SFT-26-final-only-control": (115, 8, 2, 30),
    }
    by_id = {run["run_id"]: run for run in table["runs"]}

    for run_id, (rows, batch, epochs, updates) in expected.items():
        values = {**table["defaults"], **by_id[run_id]["changes"]}
        assert updates == epochs * math.ceil(rows / batch)
        assert f"{updates} optimizer updates" in values["relative_cost"]


def test_sft_predeclares_three_exact_seeds_for_each_primary_source_arm() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["sft"]
    expected = {
        "teacher": {
            "SFT-01-teacher-broad / Fresh75 step 230": "20260914",
            "SFT-19-teacher-seed-20260915": "20260915",
            "SFT-20-teacher-seed-20260916": "20260916",
        },
        "self": {
            "SFT-02-self-broad": "20260914",
            "SFT-21-self-seed-20260915": "20260915",
            "SFT-22-self-seed-20260916": "20260916",
        },
        "mixed": {
            "SFT-03-mix-50-50": "20260914",
            "SFT-23-mix-seed-20260915": "20260915",
            "SFT-24-mix-seed-20260916": "20260916",
        },
    }

    by_id = {run["run_id"]: run for run in table["runs"]}
    for rows in expected.values():
        assert len(rows) == 3
        for run_id, seed in rows.items():
            values = {**table["defaults"], **by_id[run_id]["changes"]}
            assert values["data_seed"] == seed
            assert values["shuffle_seed"] == seed
            assert values["eval_seed"] == "20260914"


def test_rendered_skyrl_rows_override_every_miles_only_operational_field() -> None:
    from training import skyrl as native_skyrl

    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    defaults = table["defaults"]

    assert defaults["nodes"] == "4"
    assert defaults["gpus_per_node"] == "8"
    assert "8 GPUs" in defaults["tensor_parallel"]
    assert "context parallel size 4" in defaults["tensor_parallel"]
    assert "OpenCode 1.18.27" in defaults["harness"]

    native_rows = {
        "RL-02-skyrl-one-update-control": dict(nodes=1, steps=1, groups=2, samples=4),
        "RL-07-skyrl-native-10u": dict(nodes=2, steps=10, groups=4, samples=8),
    }
    assert native_skyrl.NATIVE_SOURCES["skyrl.train.config.config"] == (
        "a3b36099c5308fc9bc658394f10cfff0da3f0cadcc067aa2b44e95870a609c90"
    )
    by_id = {run["run_id"]: run for run in table["runs"]}
    for run_id, spec in native_rows.items():
        run = by_id[run_id]
        # A native SkyRL row may not inherit even a coincidentally equal Miles
        # setting. This checks what the reader actually sees after rendering.
        assert set(defaults) <= run["changes"].keys()
        values = {**defaults, **run["changes"]}
        cfg = native_skyrl.SkyRLConfig(
            name="rendered-row-test",
            output_root="/mnt/sfs/jobs/rendered-row-test/output",
            model_root="/mnt/sfs/models/rendered-row-test/model",
            train_data="/mnt/sfs/data/rendered-row-test/train.jsonl",
            dev_data="/mnt/sfs/data/rendered-row-test/dev.jsonl",
            data_manifest="/mnt/sfs/data/rendered-row-test/manifest.json",
            train_rows=8,
            dev_rows=17,
            wandb_entity="test",
            wandb_project="test",
            wandb_run_id="test",
            nodes=spec["nodes"],
            steps=spec["steps"],
            groups=spec["groups"],
            samples_per_prompt=spec["samples"],
            eval_interval=1 if spec["steps"] == 1 else 5,
            checkpoint_interval=1 if spec["steps"] == 1 else 5,
        )
        overrides = native_skyrl.overrides(cfg)
        assert values["nodes"] == str(overrides["trainer.placement.policy_num_nodes"])
        assert values["gpus_per_node"] == str(
            overrides["trainer.placement.policy_num_gpus_per_node"]
        )
        assert values["inference_engines"].startswith(
            str(overrides["generator.inference_engine.num_engines"])
        )
        assert values["tensor_parallel"].startswith(
            str(overrides["generator.inference_engine.tensor_parallel_size"])
        )
        assert values["context_tokens"].startswith(f"{cfg.context_tokens:,}")
        assert values["response_tokens"].startswith(f"{cfg.response_tokens:,}")
        assert values["tokens_per_turn"] == f"{cfg.tokens_per_turn:,}"
        assert values["max_turns"] == str(cfg.max_turns)
        assert values["samples_per_task"] == str(cfg.samples_per_prompt)
        assert values["task_groups_per_update"] == str(cfg.groups)
        assert values["total_rollouts_update"] == str(cfg.groups * cfg.samples_per_prompt)
        assert float(values["actor_lr"]) == cfg.lr
        assert values["generation_seed"] == str(cfg.seed)
        assert values["training_seed"] == str(cfg.seed)
        assert values["updates_per_batch"].startswith("1 native optimizer pass")
        assert values["weight_decay"] == "0.01"
        assert values["grad_clip"] == "1.0"
        assert values["clip_lower"] == values["clip_upper"] == "0.20"
        assert values["loss_averaging"] == "Mean over supervised response tokens"
        assert values["keep_checkpoints"] == str(cfg.keep_checkpoints)
        assert "No compaction" in values["compaction"]
        assert "abort the whole batch" in values["invalid_retry"]
        assert "same" in values["placement"]
        assert "Miles-matched" not in json.dumps(run)


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
    assert "RL-07-skyrl-native-10u" in ten_update_runs


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


def test_every_rendered_row_uses_the_frozen_split_and_has_no_dynamic_winner() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    dynamic = re.compile(r"\b(best|winner|winning|leading)\b", re.IGNORECASE)

    for table in data["tables"].values():
        for run in table["runs"]:
            values = {**table["defaults"], **run["changes"]}
            assert values["task_split"] == "Fixed representative split from 14 September 2026"
            assert values["final_pool"] == "8 untouched Fleet tasks"
            assert not dynamic.search(json.dumps(run)), run["run_id"]


def test_every_rendered_row_has_exact_numeric_seeds() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    for kind, seed_fields in {
        "sft": ("data_seed", "shuffle_seed", "eval_seed"),
        "rl": ("data_seed", "training_seed", "generation_seed", "eval_seed"),
    }.items():
        table = data["tables"][kind]
        for run in table["runs"]:
            values = {**table["defaults"], **run["changes"]}
            for field in seed_fields:
                assert re.fullmatch(r"[0-9]+", values[field]), (run["run_id"], field)


def test_rl_predeclares_three_native_seeds_for_each_start_checkpoint() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())
    table = data["tables"]["rl"]
    groups = [
        ("RL-03-base-miles-10u", "RL-23-base-seed-43", "RL-24-base-seed-44"),
        ("RL-04-teacher-sft-start", "RL-27-teacher-seed-43", "RL-28-teacher-seed-44"),
        ("RL-05-self-sft-start", "RL-29-self-seed-43", "RL-30-self-seed-44"),
        ("RL-06-mixed-sft-start", "RL-31-mix-seed-43", "RL-32-mix-seed-44"),
    ]
    by_id = {run["run_id"]: run for run in table["runs"]}
    for ids in groups:
        rendered = [{**table["defaults"], **by_id[run_id]["changes"]} for run_id in ids]
        assert [row["data_seed"] for row in rendered] == ["42", "43", "44"]
        assert [row["training_seed"] for row in rendered] == ["42", "43", "44"]
        assert [row["generation_seed"] for row in rendered] == ["42", "43", "44"]
        assert [row["eval_seed"] for row in rendered] == ["20260914"] * 3
        assert len({row["start_checkpoint"] for row in rendered}) == 1


def test_current_cluster_and_public_benchmark_policy_is_unambiguous() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    cluster = (ROOT / "docs" / "CLUSTER_ALERTS_AND_INFERENCE_SERVING.md").read_text()
    html = (ROOT / "site" / "index.html").read_text()

    assert "priority `c1` for current project work" in agents
    assert "Request pod priority `c1` for all current project work" in cluster
    assert "`c2`" not in agents + cluster
    assert "never submit at `c0`" in cluster
    assert "development cluster is for bounded qualification and debugging only" in agents
    assert "fixed deadline no later than 30 minutes" in agents
    assert "Verify every recorded UID and named resource is" in agents
    assert "GPU allocation is exactly zero" in agents
    assert "fixed wall-clock deadline no later" in cluster
    assert "30 minutes after object creation" in cluster
    assert "teardown path" in cluster and "before\nit is created" in cluster
    assert "immediately" in cluster
    assert "New dev tests may not retain a" in cluster
    assert "exact GPU allocation is zero" in cluster
    assert "WebExploitBench is a report-only transfer check" in html
    assert "development evidence" not in html
