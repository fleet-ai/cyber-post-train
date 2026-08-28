from __future__ import annotations

from pathlib import Path

from evals.secondary.workspace import prepare_xben_compose


def test_xben_wrapper_is_secret_free_and_internal(tmp_path: Path) -> None:
    checkout = tmp_path / "xben"
    challenge = checkout / "benchmarks" / "XBEN-001-24"
    challenge.mkdir(parents=True)
    (challenge / "docker-compose.yml").write_text("services: {target: {image: nginx}}\n")
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "Dockerfile.agent").write_text("FROM scratch\n")

    wrapper = prepare_xben_compose(
        checkout=checkout,
        target="XBEN-001-24",
        agent_context=agent,
    )
    rendered = wrapper.read_text()
    assert "internal: true" in rendered
    assert "FLEET_API_KEY" not in rendered
    assert "XBEN_FLAG" not in rendered
    assert wrapper.stat().st_mode & 0o777 == 0o600
