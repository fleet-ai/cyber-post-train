import json
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
    }

    found = sorted(phrase for phrase in unexplained_internal_phrases if phrase in text)
    assert not found, f"replace internal language with a plain explanation: {found}"


def test_training_terms_have_visible_definitions() -> None:
    html = (ROOT / "site" / "index.html").read_text()

    assert "Supervised fine-tuning (SFT)" in html
    assert "Reinforcement learning (RL)" in html
    assert "Select any column title to read its plain-language meaning" in html


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
