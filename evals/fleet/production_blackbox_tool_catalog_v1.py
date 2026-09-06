"""Imported production blackbox CTF tool catalog.

The JSON snapshot is the canonical ``tools/list`` result emitted by the
Theseus sandbox MCP in cyber transition mode with the patch phase disabled.
Parity probes load these bytes; they must not reconstruct a similar-looking
catalog from a hand-written OpenAI request.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto

CATALOG_PATH = Path("evals/fleet/configs/blackbox-ctf-tool-catalog-v1.json")
CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
SOURCE = {
    "repository": "fleet-ai/theseus",
    "commit": "0fff262678a80901fee6aa398e0142039e8a40bc",
    "mode": "CYBER_MODE=1,CYBER_PATCH_PHASE=off,CYBER_SUBMIT_REPORT_MODE=transition",
    "files": {
        "shared/sandbox-mcp/src/index.ts": (
            "sha256:faee62251b84b8a38693967cfb762b77af8e2c57863ed3a7b2e30bc02352a07d"
        ),
        "shared/sandbox-mcp/src/bash-tool.ts": (
            "sha256:bd6b3e8837397020652ece964b0321da67148951c96bd2f1294908389889e06d"
        ),
        "shared/sandbox-mcp/src/cyber/submit_report.ts": (
            "sha256:ecd5af054fdf97a3597adfdfa5de5d1b88befcdbba55fc29718471cf19441172"
        ),
    },
}


class ToolCatalogError(RuntimeError):
    """Stable failure for catalog provenance or schema drift."""


def load(repo_root: Path) -> list[dict[str, Any]]:
    """Load and validate the imported production catalog snapshot."""
    path = repo_root / CATALOG_PATH
    value = json.loads(path.read_text())
    if not isinstance(value, list) or [row.get("name") for row in value] != [
        "bash",
        "submit_report",
    ]:
        raise ToolCatalogError("production_tool_surface_drift")
    if not all(isinstance(row, dict) for row in value):
        raise ToolCatalogError("production_tool_catalog_invalid")
    digest = crypto.sha256(crypto.canonical_json(value))
    if digest != CATALOG_SHA256:
        raise ToolCatalogError("production_tool_catalog_digest_drift")
    return copy.deepcopy(value)


def provenance(repo_root: Path) -> dict[str, Any]:
    """Return content-free immutable provenance for a parity receipt."""
    load(repo_root)
    return {
        "catalog_path": str(CATALOG_PATH),
        "catalog_sha256": CATALOG_SHA256,
        "source": copy.deepcopy(SOURCE),
    }
