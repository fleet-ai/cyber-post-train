"""Research-plan contracts, not assertions about live cluster readiness."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "site/training-decision-space.json").read_text())


def cli(*args, ok=True):
    result = subprocess.run(
        ["node", "scripts/plan-experiments.cjs", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stderr
    return json.loads(result.stdout) if ok else result.stderr


def test_compact_tables_are_proposals_with_sources_and_controls():
    assert set(PLAN["tables"]) == {"sft", "rl"}
    sources = {source["id"] for source in PLAN["sources"]}
    for table in PLAN["tables"].values():
        assert 6 <= len(table["columns"]) <= 10
        ids = [row["id"] for row in table["runs"]]
        assert len(ids) == len(set(ids))
        for row in table["runs"]:
            assert {"question", "why", "gate", "compare", "sources"} <= row.keys()
            assert row["sources"] and set(row["sources"]) <= sources
    for source in PLAN["sources"]:
        assert source["url"].startswith("https://")
        assert all(source[key] for key in ("section", "observed", "use", "limit"))


def test_initial_bracket_is_concrete_and_differs_only_in_rate():
    result = cli("--stage", "first")
    assert result["submits_jobs"] is False
    rows = result["candidates"]
    assert [row["settings"]["lr"] for row in rows] == [5e-6, 2e-5, 1e-6]
    for row in rows:
        assert row["resolved"] and row["optimizer_steps"] is None
        assert "not launch-qualified" in row["status"]
        assert row["settings"]["context"] == 262144
        assert row["settings"]["priority"] == "c1"
        assert row["settings"]["max_active_nodes"] == 8
    controls = [{k: v for k, v in row["settings"].items() if k != "lr"} for row in rows]
    assert controls[0] == controls[1] == controls[2]
    assert result == cli("--stage", "first")


def test_later_rows_do_not_silently_choose_a_winner():
    result = cli("--stage", "efficient")
    for row in result["candidates"]:
        if row["inherit"]:
            assert not row["resolved"] and row["inherited_from"] is None
            assert row["spec_sha256"] is None
    result = cli("--stage", "efficient", "--winner", "S02")
    rows = {row["id"]: row for row in result["candidates"]}
    assert rows["S07"]["settings"]["lr"] == 2e-5
    assert rows["S10"]["settings"]["lr"] == 4e-5
    assert rows["S10"]["settings"]["batch"] == 64


def test_confirmation_inherits_method_data_and_adapter_not_just_lr():
    for finalist, data, method in [
        ("S05", "Matched self", "Full"),
        ("S11", "Broad teacher", "LoRA r64"),
    ]:
        rows = cli("--stage", "confirm", "--finalist", finalist)["candidates"]
        assert [row["settings"]["seed"] for row in rows] == [43, 44]
        for row in rows:
            assert row["settings"]["data"] == data
            assert row["settings"]["method"] == method
            if finalist == "S11":
                assert row["settings"]["lora_rank"] == 64
                assert row["settings"]["lora_alpha"] == 128
    assert not cli("--stage", "confirm", "--finalist", "S08")["candidates"][0]["resolved"]
    row = cli("--stage", "confirm", "--finalist", "S08", "--winner", "S02")["candidates"][0]
    assert row["resolved"] and row["settings"]["epochs"] == 4
    assert row["settings"]["lr"] == 2e-5


def test_rejects_bad_or_ambiguous_options():
    for args in [
        ("--stage", "wrong"),
        ("--limit", "0"),
        ("--limit", "1.5"),
        ("--winner", "S99"),
        ("--winner", "S05"),
        ("--stage", "first", "--stage", "all"),
        ("--format", "yaml"),
        ("--budget-gpu-hours", "20"),
        ("--submit", "true"),
    ]:
        assert "Plan not generated" in cli(*args, ok=False)


def inventory(tmp_path, data=None):
    value = data or {
        "Broad teacher": {
            "examples": 1234,
            "supervised_tokens": 9876543,
            "manifest_sha256": "a" * 64,
        }
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(value))
    return path


def test_counts_use_actual_corpus_and_keep_final_partial_batch(tmp_path):
    source = inventory(tmp_path)
    row = cli("--inventory", source)["candidates"][0]
    assert row["optimizer_steps"] == 2 * 78
    assert row["supervised_tokens_total"] == 2 * 9876543
    later = cli("--inventory", source, "--stage", "efficient", "--winner", "S01")["candidates"]
    shorter = next(row for row in later if row["id"] == "S13")
    assert shorter["corpus"] is None and shorter["optimizer_steps"] is None
    assert shorter["inventory_key"] == "Broad teacher / context 65536"
    large_batch = next(row for row in later if row["id"] == "S09")
    assert large_batch["optimizer_steps"] == 40


def test_matched_source_budgets_are_checked(tmp_path):
    pools = {
        name: {
            "examples": 200,
            "supervised_tokens": 10000,
            "manifest_sha256": "a" * 64,
            "family_budget_sha256": "b" * 64,
        }
        for name in ("Matched teacher", "Matched self", "Matched 50:50")
    }
    path = inventory(tmp_path, pools)
    assert len(cli("--stage", "data", "--inventory", path)["candidates"]) == 3
    pools["Matched self"]["supervised_tokens"] += 1
    assert "different family budgets" in cli(
        "--stage", "data", "--inventory", inventory(tmp_path, pools), ok=False
    )


def test_bad_inventory_fails_closed(tmp_path):
    for value in [0, -1, 2.5, True, "100"]:
        path = inventory(
            tmp_path,
            {
                "Broad teacher": {
                    "examples": value,
                    "supervised_tokens": 10,
                    "manifest_sha256": "a" * 64,
                }
            },
        )
        cli("--inventory", path, ok=False)


def test_budget_uses_exact_recipe_and_lists_deferred_rows(tmp_path):
    source = inventory(tmp_path)
    rows = cli("--inventory", source)["candidates"]
    costs = {
        "includes": "training_export_and_fleet_dev",
        "runs": {
            row["id"]: {
                "spec_sha256": row["spec_sha256"],
                "gpu_hours": 80,
                "evidence": "synthetic test cost",
            }
            for row in rows
        },
    }
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(costs))
    result = cli("--inventory", source, "--costs", path, "--budget-gpu-hours", "160")
    assert [row["id"] for row in result["candidates"]] == ["S01", "S02"]
    assert result["budget"]["estimated_used"] == 160
    assert result["deferred"] == [{"id": "S03", "reason": "Outside remaining estimated budget"}]
    costs["runs"]["S01"]["spec_sha256"] = "0" * 64
    del costs["runs"]["S02"]
    path.write_text(json.dumps(costs))
    result = cli("--inventory", source, "--costs", path, "--budget-gpu-hours", "79")
    assert not result["candidates"] and len(result["deferred"]) == 3
    assert "different recipe" in result["deferred"][0]["reason"]


def test_external_results_cannot_enter_selection_or_generator():
    for filename in ("site/experiment-plan.js", "scripts/plan-experiments.cjs"):
        code = (ROOT / filename).read_text()
        for forbidden in (
            "REPORT_DATA",
            "web-evals",
            "httpx",
            "fetch(",
            "https.request",
            "execSync",
            "child_process",
        ):
            assert forbidden not in code
    assert "never WebExploitBench results" in PLAN["scope"]
    assert "Fleet dev task success" in PLAN["defaults"]["selection"]
    assert "not supported" in PLAN["tables"]["sft"]["runs"][10]["gate"]


def test_csv_conditional_settings_are_not_presented_as_concrete():
    result = subprocess.run(
        ["node", "scripts/plan-experiments.cjs", "--stage", "confirm", "--format", "csv"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert all(row["lr"] == "" and row["batch"] == "" for row in rows)


def test_context_policy_and_script_order():
    html = (ROOT / "site/index.html").read_text()
    assert (
        html.index('src="experiment-plan.js"')
        < html.index('src="experiment-view.js"')
        < html.index('src="app.js"')
    )
    assert "No compaction" not in json.dumps(PLAN)
    assert all(row["context"] == 262144 for row in PLAN["tables"]["rl"]["runs"])
    assert "teacher-imitation loss on held-out traces" in html
