from __future__ import annotations

from typer.testing import CliRunner

from cyber_post_train.catalog import CATALOG, catalog_dict, doctor
from cyber_post_train.cli import app


def test_catalog_has_every_primary_adapter() -> None:
    assert {component.name for component in CATALOG} == {
        "fleet",
        "webexploitbench",
        "exploitgym",
        "fleet-sft",
        "fleet-rl",
    }
    assert catalog_dict()["schema"] == "cyber_post_train_catalog_v1"


def test_doctor_is_offline_and_green() -> None:
    receipt = doctor()
    assert receipt["schema"] == "cyber_post_train_doctor_v1"
    assert receipt["ok"] is True
    assert all(receipt["checks"].values())


def test_cli_catalog_json_is_stable() -> None:
    result = CliRunner().invoke(app, ["catalog", "--json"])
    assert result.exit_code == 0
    assert '"schema": "cyber_post_train_catalog_v1"' in result.stdout


def test_cli_doctor_json_is_green() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert '"ok": true' in result.stdout
