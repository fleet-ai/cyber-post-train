"""Content-free bootstrap diagnostic invariants for Qwen bulk controllers."""

STAGE_EVENTS = (
    ("00-start", "BEFORE"),
    ("01-hydration", "BEFORE"),
    ("01-hydration", "AFTER"),
    ("02-docker-cli", "BEFORE"),
    ("02-docker-cli", "AFTER"),
    ("03-dind-ready", "BEFORE"),
    ("03-dind-ready", "AFTER"),
    ("04-image-build", "BEFORE"),
    ("04-image-build", "AFTER"),
    ("05-version", "BEFORE"),
    ("05-version", "AFTER"),
    ("06-clear", "CLEAR"),
)


def stage_receipt_name(stage: str, phase: str) -> str:
    """Return the create-once filename for one diagnostic boundary."""
    if not stage or not phase or "/" in stage or "/" in phase:
        raise ValueError("stage and phase must be non-empty path components")
    return f"{stage}-{phase}.json"


def stage_receipt_names() -> tuple[str, ...]:
    """Enumerate and validate the complete diagnostic receipt namespace."""
    names = tuple(stage_receipt_name(stage, phase) for stage, phase in STAGE_EVENTS)
    if len(names) != len(set(names)):
        raise ValueError("bootstrap diagnostic receipt paths collide")
    return names
