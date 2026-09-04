from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = (
    "cyber-train-operator",
    "cyber-eval-parity",
    "cyber-run-evidence",
    "cyber-experiment-maintainer",
    "cyber-gpu-steward",
)
UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
SECRET_LIKE = re.compile(r"(?<![A-Za-z0-9])(?:sk|ghp)_[A-Za-z0-9_-]{16,}")


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text()
    assert text.startswith("---\n")
    _, raw, _ = text.split("---\n", 2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def test_agent_skill_packages_are_complete_and_routed() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    for name in SKILLS:
        skill_dir = ROOT / "skills" / name
        skill_path = skill_dir / "SKILL.md"
        interface_path = skill_dir / "agents" / "openai.yaml"
        assert skill_path.is_file()
        assert interface_path.is_file()
        assert f"skills/{name}/SKILL.md" in agents

        skill_text = skill_path.read_text()
        metadata = _frontmatter(skill_path)
        assert metadata["name"] == name
        assert isinstance(metadata.get("description"), str)
        assert 40 <= len(str(metadata["description"])) <= 400
        assert "TODO" not in skill_text
        assert len(skill_text.splitlines()) <= 160

        interface = yaml.safe_load(interface_path.read_text())
        assert isinstance(interface, dict)
        ui = interface.get("interface")
        assert isinstance(ui, dict)
        assert 25 <= len(str(ui.get("short_description") or "")) <= 64
        assert f"${name}" in str(ui.get("default_prompt") or "")

        for target in MARKDOWN_LINK.findall(skill_text):
            if "://" in target or target.startswith("#"):
                continue
            assert (skill_dir / target).is_file(), f"broken skill reference: {name}/{target}"


def test_skills_contain_no_ephemeral_or_secret_material() -> None:
    forbidden_literals = ("FLEET_API_KEY=", "FLEET_TRAINING_API_TOKEN=")
    for path in sorted((ROOT / "skills").rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text()
        assert not UUID.search(text), f"ephemeral UUID in {path}"
        assert not SECRET_LIKE.search(text), f"secret-like token in {path}"
        assert "ft-run-" not in text, f"ephemeral run name in {path}"
        for literal in forbidden_literals:
            assert literal not in text, f"secret-like literal {literal!r} in {path}"
