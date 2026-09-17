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
