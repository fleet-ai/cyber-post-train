from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import production_blackbox_tool_catalog_v1 as catalog
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_imported_catalog_matches_runtime_authority_exactly() -> None:
    tools = catalog.load(ROOT)
    assert [row["name"] for row in tools] == ["bash", "submit_report"]
    assert self_hosted.sha256(self_hosted.canonical_json(tools)) == catalog.CATALOG_SHA256
    assert tools[0]["inputSchema"]["required"] == ["script"]
    assert set(tools[0]["inputSchema"]["properties"]) == {"script", "timeoutMs"}
    assert tools[1]["inputSchema"]["required"] == ["explanation"]
    assert set(tools[1]["inputSchema"]["properties"]) == {
        "flag",
        "flags",
        "explanation",
        "verdict",
    }


def test_loader_returns_a_copy() -> None:
    changed = catalog.load(ROOT)
    changed[0]["name"] = "different"
    assert catalog.load(ROOT)[0]["name"] == "bash"


def test_any_descriptor_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = copy.deepcopy(catalog.load(ROOT))
    tools[0]["inputSchema"]["properties"]["script"]["minLength"] = 2
    path = tmp_path / catalog.CATALOG_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tools))
    with pytest.raises(catalog.ToolCatalogError, match="digest_drift"):
        catalog.load(tmp_path)


def test_provenance_binds_exact_production_builder_sources() -> None:
    receipt = catalog.provenance(ROOT)
    assert receipt["catalog_sha256"] == catalog.CATALOG_SHA256
    assert receipt["source"]["commit"] == "0fff262678a80901fee6aa398e0142039e8a40bc"
    assert set(receipt["source"]["files"]) == {
        "shared/sandbox-mcp/src/index.ts",
        "shared/sandbox-mcp/src/bash-tool.ts",
        "shared/sandbox-mcp/src/cyber/submit_report.ts",
    }
