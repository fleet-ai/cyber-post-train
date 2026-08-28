from __future__ import annotations

import re
from pathlib import Path

from evals.secondary.config import SecondaryEvalConfig
from evals.secondary.plan import build_plan, prompt_digest, render_xben_prompt


def test_xben_plan_is_secret_free(tmp_path: Path) -> None:
    config = SecondaryEvalConfig(
        benchmark="xben",
        target="XBEN-001-24",
        variant="blackbox",
        target_url="http://target:80",
    )
    plan = build_plan(config, checkout=tmp_path / "xben", project_root=tmp_path / "repo")
    serialized = plan.canonical_json()
    assert "FLEET_API_KEY" in serialized
    assert "Bearer" not in serialized
    assert re.search(r"sk_[A-Za-z0-9]{20,}", serialized) is None
    assert plan.upstream.commit == "a16cb1ba0701b20d63913846536fc7b0672082af"
    assert plan.command.paid is True
    assert "--max-connections" in plan.command.argv
    assert "--log-dir" in plan.command.argv
    assert "glm52-secondary-smoke-v1" in " ".join(plan.command.argv)
    assert plan.command.required_secret_env == ("FLEET_API_KEY", "XBEN_FLAG")


def test_cvebench_plan_is_zero_day_and_pinned(tmp_path: Path) -> None:
    config = SecondaryEvalConfig(
        benchmark="cvebench",
        target="CVE-2024-2624",
        variant="zero_day",
    )
    plan = build_plan(config, checkout=tmp_path / "cve", project_root=tmp_path / "repo")
    assert plan.upstream.release == "v2.1.0"
    assert "challenges=CVE-2024-2624" in plan.command.argv
    assert "variants=zero_day" in plan.command.argv
    assert "max_messages=60" in plan.command.argv
    assert plan.command.argv.count("1") >= 2


def test_xben_prompt_is_stable_and_target_scoped() -> None:
    rendered = render_xben_prompt(
        description="Find the access-control flaw", target_url="http://app:80"
    )
    assert "authorized security evaluation" in rendered
    assert "Do not access any other host" in rendered
    assert "http://app:80" in rendered
    digest = prompt_digest(
        description="Find the access-control flaw",
        target_url="http://app:80",
    )
    assert digest == ("3e7d7cf33d05e6a64b50755a638d4455c156dde3c4d079dd7be9746fde712496")
