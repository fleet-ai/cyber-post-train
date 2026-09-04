"""Machine-readable inventory of supported experiment components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Component:
    """One public adapter or experiment component family."""

    kind: str
    name: str
    implementation: str
    lifecycle: tuple[str, ...]
    status: str = "available"
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


COMMON_EVAL_LIFECYCLE = ("validate", "preview", "launch", "status", "accept")
COMMON_TRAIN_LIFECYCLE = (
    "validate",
    "preview",
    "launch",
    "status",
    "export",
    "accept",
)

CATALOG = (
    Component(
        kind="evaluation",
        name="fleet",
        implementation="evals.fleet",
        lifecycle=COMMON_EVAL_LIFECYCLE,
        notes="Exact Fleet task versions, controlled agent runtime, authoritative verifier.",
    ),
    Component(
        kind="evaluation",
        name="webexploitbench",
        implementation="evals.webexploitbench",
        lifecycle=COMMON_EVAL_LIFECYCLE,
        notes="Pinned CAGE and harness integration; evaluation-only and sealed.",
    ),
    Component(
        kind="evaluation",
        name="exploitgym",
        implementation="evals.exploitgym",
        lifecycle=COMMON_EVAL_LIFECYCLE,
        notes="Pinned linux/amd64 control image, dynamic graders, paired arm support.",
    ),
    Component(
        kind="training",
        name="fleet-sft",
        implementation="training",
        lifecycle=COMMON_TRAIN_LIFECYCLE,
        notes=(
            "Training Jobs API with exact corpus, model, trainer, optimizer and "
            "checkpoint gates."
        ),
    ),
    Component(
        kind="training",
        name="fleet-rl",
        implementation="training",
        lifecycle=COMMON_TRAIN_LIFECYCLE,
        notes="SFT controls plus authoritative tool allowlists and reward-acquisition gate.",
    ),
)


def catalog_dict() -> dict[str, object]:
    """Return a deterministic public catalog envelope."""

    return {
        "schema": "cyber_post_train_catalog_v1",
        "components": [component.to_dict() for component in CATALOG],
    }


def doctor() -> dict[str, object]:
    """Check the local interface without credentials or network access."""

    required = {
        "repository_instructions": ROOT / "AGENTS.md",
        "scientific_protocol": ROOT / "docs" / "SCIENTIFIC_PROTOCOL.md",
        "repository_design": ROOT / "docs" / "REPOSITORY_DESIGN.md",
        "fleet_adapter": ROOT / "evals" / "fleet" / "__init__.py",
        "webexploitbench_adapter": ROOT / "evals" / "webexploitbench" / "__init__.py",
        "exploitgym_adapter": ROOT / "evals" / "exploitgym" / "__init__.py",
        "training_adapter": ROOT / "training" / "__init__.py",
        "experiment_operator_skill": ROOT
        / "skills"
        / "cyber-experiment-operator"
        / "SKILL.md",
    }
    checks = {name: path.is_file() for name, path in required.items()}
    component_names = {component.name for component in CATALOG}
    expected_names = {
        "fleet",
        "webexploitbench",
        "exploitgym",
        "fleet-sft",
        "fleet-rl",
    }
    checks["component_catalog"] = component_names == expected_names
    return {
        "schema": "cyber_post_train_doctor_v1",
        "ok": all(checks.values()),
        "checks": checks,
    }
