import pytest

from evals.fleet import qwen_bootstrap_diagnostic as diagnostic


def test_complete_stage_receipt_namespace_is_unique() -> None:
    assert diagnostic.stage_receipt_names() == (
        "00-start-BEFORE.json",
        "01-hydration-BEFORE.json",
        "01-hydration-AFTER.json",
        "02-docker-cli-BEFORE.json",
        "02-docker-cli-AFTER.json",
        "03-dind-ready-BEFORE.json",
        "03-dind-ready-AFTER.json",
        "04-image-build-BEFORE.json",
        "04-image-build-AFTER.json",
        "05-version-BEFORE.json",
        "05-version-AFTER.json",
        "06-clear-CLEAR.json",
    )


def test_duplicate_stage_phase_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostic,
        "STAGE_EVENTS",
        (("01-hydration", "BEFORE"), ("01-hydration", "BEFORE")),
    )
    with pytest.raises(ValueError, match="receipt paths collide"):
        diagnostic.stage_receipt_names()

