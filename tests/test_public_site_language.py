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
    data = (ROOT / "site" / "training-decision-space.json").read_text()

    assert 'id="experiment-definitions"' in html
    for required_term in ("Supervised fine-tuning", "Reinforcement learning"):
        assert required_term in data


def test_experiment_map_is_complete_and_well_formed() -> None:
    data = json.loads((ROOT / "site" / "training-decision-space.json").read_text())

    assert len(data["definitions"]) >= 50
    assert len(data["sequence"]) == 7
    assert len(data["decisions"]) >= 120
    assert {row["priority"] for row in data["decisions"]} == {
        "Hold fixed",
        "Test now",
        "Test later",
    }
    identities = [(row["area"], row["choice"]) for row in data["decisions"]]
    assert len(identities) == len(set(identities))
    assert all(row["values"] for row in data["decisions"])
