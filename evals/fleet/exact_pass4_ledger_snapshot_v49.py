"""Build the fully enumerated, score-blind exact-pass@4 v49 audit snapshot.

The snapshot deliberately separates cell-level evidence from later aggregate
claims.  A successful controller Job is not sufficient to call every planned
cell accepted: the controller may also exit zero after quarantine or preserved
claim outcomes.  Likewise, a task-level blocked tally that does not name an
attempt cannot be projected onto one of the four statistical cells.

No workload or Fleet API is touched.  Inputs are the immutable v48 evidence
manifest, its successful live-validation receipt, and the fixed public
score-blind metadata recorded below for the later Qwen lineage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

V48_COMMIT = "c99d24dc0451e5710217b0f44e8e72872f98a25f"
V48_MERGE_COMMIT = "045e8870395e3e4f1e03e5981c9c921af13dcaa1"
V48_MANIFEST_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-exact-pass4-ledger-evidence-snapshot-v48.json"
)
V48_MANIFEST_FILE_SHA256 = "sha256:332c5ceae480e9c73cf1e391aeb59994049b69b240a8bbc106aefc35a9a01f25"
V48_MANIFEST_RECEIPT_SHA256 = (
    "sha256:1275d84f1b6aad8d02bc916be55f3a1f3ea99c6817f465905ec0791b0f66cd7a"
)
V48_LIVE_VALIDATION_PATH = Path(
    "docs/evidence/glm53-study/2026-09-06-glm53-ledger-v48-live-validation.json"
)
V48_LIVE_VALIDATION_FILE_SHA256 = (
    "sha256:c3c527a68217190555af394b94b7996901dd46e1e52505c9843d02afe66c7738"
)
V48_LIVE_VALIDATION_RECEIPT_SHA256 = (
    "sha256:7b77ed00959f6bb219c5c20724ed6958ee85b56d01771f9f158a575cd7301c96"
)
G21_RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-release-v6.json"
)
G21_RELEASE_FILE_SHA256 = "sha256:ec4820996726da53fdddd152ca52cab2153be835e7c489fea5038a8249b9ef38"
G21_RELEASE_RECEIPT_SHA256 = (
    "sha256:e056cc61667434df920d66782d2145bc886d4c255a05ac83daa4a98ff242df01"
)
CAMPAIGN_PATH = Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json")
OUTPUT_PATH = Path("docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v49.json")

STRICT_V48_TALLY = {
    "qwen3.8-27b": {
        "accepted": 42,
        "active": 1,
        "blocked_nonrepeatable": 8,
        "unstarted": 349,
    },
    "glm-5.3": {
        "accepted": 24,
        "active": 0,
        "blocked_nonrepeatable": 4,
        "unstarted": 372,
    },
}
POST_V48_QWEN_TALLY_CLAIM = {
    "accepted": 51,
    "active": 0,
    "blocked_nonrepeatable": 9,
    "unstarted": 340,
}

# These files were inspected in the exact merged Qwen lineage.  They are not
# copied here because none supplies the missing cell-level acceptance receipts.
# Both file and self-receipt digests are retained to make the inspection exact.
LATER_QWEN_EVIDENCE = [
    {
        "kind": "rank17_held_plan",
        "merge_commit": "e7f1772ee7727e348523735387f2a7861308553e",
        "path": "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-held-v1.json",
        "file_sha256": "sha256:71a99f7a90a16c6e69eff13ad9d66d6287020e4776658a8250120c7c38a2364a",
        "receipt_sha256": "sha256:05e054ad8dee94d64b3b7be8ee3b66c66909129e4d5d000e8fcd7bb280c01cbf",
        "cell_authority": False,
    },
    {
        "kind": "rank17_release_failure",
        "merge_commit": "04d4a2732fa9917a7534e02fbe9eb50cc9f4da36",
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-rank17-g22-release-observer-failed-v1.json"
        ),
        "file_sha256": "sha256:1eca3eb03764d2575e0a029320214aa0d49db7e5f89259193bc30a0ab1d38290",
        "receipt_sha256": "sha256:eeb08c375f014fba20cc00c371a1f235f81c086f22cd885ca54c330e443b6472",
        "cell_authority": False,
    },
    {
        "kind": "rank17_release_held",
        "merge_commit": "04d4a2732fa9917a7534e02fbe9eb50cc9f4da36",
        "path": (
            "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-release-held-v4.json"
        ),
        "file_sha256": "sha256:f920650251240456e4f178b8121c56f39815b6a21b58de4aad32583045f2a60d",
        "receipt_sha256": "sha256:575b532f42e454ad41ea8cc75a23393d1df63afe40e401009af782f2016e06a8",
        "cell_authority": False,
    },
    {
        "kind": "rank18_held_plan",
        "merge_commit": "04d4a2732fa9917a7534e02fbe9eb50cc9f4da36",
        "path": "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank18-g23-held-v1.json",
        "file_sha256": "sha256:bb5205cf05ec42cd0777e429cef69f21b6efb1f5f3d67cd7b42d1e68c476ad3b",
        "receipt_sha256": "sha256:7bdd013e9156355c4de991f5f62fbffa98eaeecaa101582a09de6fe1674653bf",
        "cell_authority": False,
    },
    {
        "kind": "rank18_release_failure",
        "merge_commit": "a6873e2c2ef7e64074a7919425921e9bfac3eaa3",
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-rank18-g23-release-observer-failed-v1.json"
        ),
        "file_sha256": "sha256:dde19b57d48be29ac802c1eacc7321285c9c561d0756d58d3968adbbe93eb176",
        "receipt_sha256": "sha256:115c51cce93f37103d5457576c68b7593d85c92642c5af38ea74c6d776916fbd",
        "cell_authority": False,
    },
    {
        "kind": "rank18_terminal_tally_claim",
        "merge_commit": "a6873e2c2ef7e64074a7919425921e9bfac3eaa3",
        "path": "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank18-g23-terminal-v1.json",
        "file_sha256": "sha256:e7fe740407bded286b95a650a1f4cf6efbb4e5f36b63c3528ca866a6246e7b67",
        "receipt_sha256": "sha256:af8dd95d432981b3fc28c2c83b85f6e28f4740d3fc95de7f66114bcbf32c7b8a",
        "cell_authority": False,
    },
    {
        "kind": "selector_v2_observation",
        "merge_commit": "6cb00254212f48c8becfaafaf9ab8897955bdc66",
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-identity-selector-v2-observation.json"
        ),
        "file_sha256": "sha256:7100a8f062d3f02a665e5d98e64b348968f98f6246b3b37c4215c91db5ae6965",
        "receipt_sha256": "sha256:e3ba8cc7556075efb857daa648ea56b2b918e81b9823ffee277f49376bb93847",
        "cell_authority": False,
    },
    {
        "kind": "selector_v2_terminal",
        "merge_commit": "6cb00254212f48c8becfaafaf9ab8897955bdc66",
        "path": (
            "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-identity-selector-v2-terminal.json"
        ),
        "file_sha256": "sha256:faa19b2cdb1becb27ee4a14e164c4eef7c3a96c7de97b391665502cc6ab78eb9",
        "receipt_sha256": "sha256:4c106088355f0508572c4efed1d36cec30780a2054431cbc87800e26ef8614f8",
        "cell_authority": False,
    },
]

G21_JOB_OBSERVATIONS = [
    {
        "selection_rank": 15,
        "job_name": "chris-q38-hosted-r015-whole-task-g21-v1",
        "job_uid": "2d2ae798-4721-43ce-b809-04f0a3c5746a",
        "pod_uid": "fe524d28-033b-47b6-aacc-0c5d088ca6a4",
        "pod_exit_code": 0,
        "pod_restarts": 0,
        "completion_time": "2026-09-06T15:25:40Z",
    },
    {
        "selection_rank": 16,
        "job_name": "chris-q38-hosted-r016-whole-task-g21-v1",
        "job_uid": "6c44bb16-daf4-4443-adfc-34761536e072",
        "pod_uid": "d93da52a-3a3c-4abd-b02d-693bebc71a9e",
        "pod_exit_code": 0,
        "pod_restarts": 0,
        "completion_time": "2026-09-06T12:49:20Z",
    },
]

_SPECIAL_ACCEPTED = {
    "docs/evidence/qwen38-study/2026-09-05-glm53-hosted-c2-accepted-validated-v1.json": (
        "glm-5.3",
        26,
        1,
    ),
    "docs/evidence/qwen38-study/2026-09-05-qwen38-generation15-accepted-gate-v1.json": (
        "qwen3.8-27b",
        4,
        1,
    ),
}
_COORDINATE_PATTERNS = (
    re.compile(r"(?:sr|r)(\d{3})-a(\d)"),
    re.compile(r"rank(\d+)[-_](?:attempt|a)(\d)"),
    re.compile(r"rank(\d+).*?a(\d)"),
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _validate_receipt(path: Path, *, file_sha256: str, receipt_sha256: str) -> dict[str, Any]:
    if _sha256(path) != file_sha256:
        raise ValueError(f"file digest drifted: {path}")
    value = _load(path)
    if value.get("receipt_sha256") != receipt_sha256 or receipt_sha256 != (
        self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise ValueError(f"receipt digest drifted: {path}")
    return value


def _accepted_coordinates(path: str) -> tuple[str, int, int]:
    if path in _SPECIAL_ACCEPTED:
        return _SPECIAL_ACCEPTED[path]
    lowered = path.lower()
    model = "glm-5.3" if "glm" in lowered else "qwen3.8-27b"
    for pattern in _COORDINATE_PATTERNS:
        matches = list(pattern.finditer(path))
        if matches:
            match = matches[-1]
            return model, int(match.group(1)), int(match.group(2))
    raise ValueError(f"accepted v48 path has no exact cell coordinates: {path}")


def _execution_lookup(cells: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], int]]:
    lookup: dict[str, tuple[dict[str, Any], int]] = {}
    for cell in cells:
        for generation in range(1, 24):
            execution_id = exact.execution_for(cell["cell_id"], generation)["execution_id"]
            if execution_id in lookup:
                raise ValueError("execution identity collision")
            lookup[execution_id] = (cell, generation)
    return lookup


def _v48_cells(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _validate_receipt(
        repo_root / V48_MANIFEST_PATH,
        file_sha256=V48_MANIFEST_FILE_SHA256,
        receipt_sha256=V48_MANIFEST_RECEIPT_SHA256,
    )
    live = _validate_receipt(
        repo_root / V48_LIVE_VALIDATION_PATH,
        file_sha256=V48_LIVE_VALIDATION_FILE_SHA256,
        receipt_sha256=V48_LIVE_VALIDATION_RECEIPT_SHA256,
    )
    if (
        live.get("status") != "PASSED_SCORE_BLIND_800_CELL_RECONCILIATION"
        or live.get("campaign_total_cells") != 800
        or live.get("glm53_tally")
        != {
            "target": 400,
            "accepted": 24,
            "active": 0,
            "retryable_infra_failed": 0,
            "blocked_nonrepeatable": 4,
            "unstarted": 372,
        }
    ):
        raise ValueError("v48 live-validation authority drifted")
    g21_release = _validate_receipt(
        repo_root / G21_RELEASE_PATH,
        file_sha256=G21_RELEASE_FILE_SHA256,
        receipt_sha256=G21_RELEASE_RECEIPT_SHA256,
    )
    if (
        g21_release.get("status") != "CLEAR"
        or g21_release.get("launch_authorized") is not True
        or g21_release.get("scoring_authorized") is not True
    ):
        raise ValueError("g21 release authority drifted")

    campaign = exact.read_object(repo_root / CAMPAIGN_PATH)
    universe = exact.build_universe(campaign, repo_root)
    cells = universe["cells"]
    by_coordinates = {(row["model"], row["selection_rank"], row["attempt"]): row for row in cells}
    g21_by_cell: dict[str, dict[str, Any]] = {}
    for controller in g21_release.get("controllers", []):
        for released in controller.get("cells", []):
            key = ("qwen3.8-27b", controller.get("selection_rank"), released.get("attempt"))
            cell = by_coordinates.get(key)
            if (
                cell is None
                or released.get("cell_id") != cell["cell_id"]
                or released.get("execution_generation") != 21
                or released.get("execution_id")
                != exact.execution_for(cell["cell_id"], 21)["execution_id"]
            ):
                raise ValueError("g21 released cell identity drifted")
            g21_by_cell[cell["cell_id"]] = {
                **released,
                "selection_rank": controller["selection_rank"],
                "job_name": controller["job_name"],
            }
    if len(g21_by_cell) != 8:
        raise ValueError("g21 release must bind exactly eight cells")
    executions = _execution_lookup(cells)
    evidence: dict[str, list[dict[str, Any]]] = {row["cell_id"]: [] for row in cells}

    for index, entry in enumerate(manifest.get("entries", [])):
        kind = entry.get("kind")
        path = entry.get("path")
        digest = entry.get("expected_receipt_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("v48 manifest entry is malformed")
        if kind == "accepted":
            key = _accepted_coordinates(path)
            cell = by_coordinates.get(key)
            generation = None
        elif kind in {"active_claim", "nonrepeatable_claim"}:
            name = Path(path).stem
            execution = executions.get("sha256:" + name)
            if execution is None:
                raise ValueError(f"v48 claim does not map to the exact universe: {path}")
            cell, generation = execution
        else:
            continue
        if cell is None:
            raise ValueError(f"v48 evidence coordinates are outside the universe: {path}")
        evidence[cell["cell_id"]].append(
            {
                "manifest_entry": index,
                "kind": kind,
                "path": path,
                "receipt_sha256": digest,
                **({"execution_generation": generation} if generation is not None else {}),
            }
        )

    result: list[dict[str, Any]] = []
    for cell in cells:
        refs = evidence[cell["cell_id"]]
        kinds = {row["kind"] for row in refs}
        if "accepted" in kinds:
            state = "accepted"
        elif "active_claim" in kinds:
            state = "active"
        elif "nonrepeatable_claim" in kinds:
            state = "blocked_nonrepeatable"
        else:
            state = "unstarted"
        observations: list[dict[str, Any]] = []
        if cell["model"] == "qwen3.8-27b" and cell["selection_rank"] in {15, 16}:
            released = g21_by_cell[cell["cell_id"]]
            job = next(
                row
                for row in G21_JOB_OBSERVATIONS
                if row["selection_rank"] == cell["selection_rank"]
            )
            observations.append(
                {
                    "kind": "g21_controller_terminal_success",
                    "job_uid": job["job_uid"],
                    "pod_uid": job["pod_uid"],
                    "execution_generation": released["execution_generation"],
                    "execution_id": released["execution_id"],
                    "run_id": released["run_id"],
                    "release_receipt_sha256": G21_RELEASE_RECEIPT_SHA256,
                    "scientific_effect": "UNRESOLVED_WITHOUT_CELL_ACCEPTED_RECEIPT",
                }
            )
        if cell["model"] == "qwen3.8-27b" and cell["selection_rank"] == 99 and cell["attempt"] == 4:
            observations.append(
                {
                    "kind": "later_aggregate_active_to_accepted_claim",
                    "scientific_effect": "NOT_ADOPTED_WITHOUT_CELL_ACCEPTED_RECEIPT",
                }
            )
        if cell["model"] == "qwen3.8-27b" and cell["selection_rank"] == 17:
            observations.append(
                {
                    "kind": "rank17_release_identity_ambiguous",
                    "scientific_effect": "NO_CELL_STATE_CHANGE",
                }
            )
        if cell["model"] == "qwen3.8-27b" and cell["selection_rank"] == 18:
            observations.append(
                {
                    "kind": "rank18_task_level_block_claim_without_attempt_identity",
                    "scientific_effect": "NOT_PROJECTED_TO_ANY_CELL",
                }
            )
        result.append(
            {
                "model": cell["model"],
                "selection_rank": cell["selection_rank"],
                "attempt": cell["attempt"],
                "cell_id": cell["cell_id"],
                "task_version_id": cell["task_version_id"],
                "state": state,
                "v48_evidence": refs,
                **({"post_v48_observations": observations} if observations else {}),
            }
        )

    counts: dict[str, dict[str, int]] = {}
    for model in exact.EXPECTED_MODELS:
        counter = Counter(row["state"] for row in result if row["model"] == model)
        counts[model] = {
            "accepted": counter["accepted"],
            "active": counter["active"],
            "blocked_nonrepeatable": counter["blocked_nonrepeatable"],
            "unstarted": counter["unstarted"],
        }
    if counts != STRICT_V48_TALLY or len(result) != 800:
        raise ValueError(
            f"offline v48 projection does not reproduce the authoritative tally: {counts!r}"
        )
    return sorted(
        result, key=lambda row: (row["model"], row["selection_rank"], row["attempt"])
    ), universe


def build(repo_root: Path) -> dict[str, Any]:
    cells, universe = _v48_cells(repo_root)
    total = Counter()
    for model_tally in STRICT_V48_TALLY.values():
        total.update(model_tally)
    body: dict[str, Any] = {
        "schema_version": "fleet-exact-pass4-ledger-audit-v49",
        "status": "VERIFIED_BASELINE_WITH_POST_V48_CELL_AUTHORITY_GAPS",
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "universe_sha256": universe["universe_sha256"],
        "universe_cell_count": 800,
        "source_lineage": {
            "authoritative_v48_commit": V48_COMMIT,
            "authoritative_v48_merge_commit": V48_MERGE_COMMIT,
            "required_predecessor_commit": "c476104e6ec9ed587cc4983f6addb7eb711abad8",
            "rank15_rank16_release_merge_commit": "df94bba327e31d419e8e1835933918c32a62f394",
            "later_qwen_inspection_head": "30cad56619e80e1d2fdbfd2417dbd18f36f8f73b",
            "later_qwen_terminal_merge_commit": "6cb00254212f48c8becfaafaf9ab8897955bdc66",
        },
        "v48_authority": {
            "manifest": {
                "path": str(V48_MANIFEST_PATH),
                "file_sha256": V48_MANIFEST_FILE_SHA256,
                "receipt_sha256": V48_MANIFEST_RECEIPT_SHA256,
            },
            "live_validation": {
                "path": str(V48_LIVE_VALIDATION_PATH),
                "file_sha256": V48_LIVE_VALIDATION_FILE_SHA256,
                "receipt_sha256": V48_LIVE_VALIDATION_RECEIPT_SHA256,
            },
        },
        "authoritative_tally": {
            "models": STRICT_V48_TALLY,
            "total": dict(total),
        },
        "post_v48_qwen_tally_claim": {
            "claimed": POST_V48_QWEN_TALLY_CLAIM,
            "adopted": False,
            "reason": (
                "the later receipts carry an aggregate constant but do not bind the nine "
                "additional accepted cell receipts, and the rank18 block does not name an attempt"
            ),
        },
        "post_v48_controller_observations": G21_JOB_OBSERVATIONS,
        "post_v48_g21_release_authority": {
            "path": str(G21_RELEASE_PATH),
            "file_sha256": G21_RELEASE_FILE_SHA256,
            "receipt_sha256": G21_RELEASE_RECEIPT_SHA256,
        },
        "later_qwen_evidence": LATER_QWEN_EVIDENCE,
        "ambiguities": [
            {
                "kind": "missing_cell_acceptance_receipts",
                "cells": [
                    row["cell_id"]
                    for row in cells
                    if row["model"] == "qwen3.8-27b" and row["selection_rank"] in {15, 16}
                ],
                "required_resolution": (
                    "digest-valid accepted or nonrepeatable receipt for each exact g21 cell"
                ),
            },
            {
                "kind": "missing_rank99_attempt4_terminal_receipt",
                "cells": [
                    row["cell_id"]
                    for row in cells
                    if row["model"] == "qwen3.8-27b"
                    and row["selection_rank"] == 99
                    and row["attempt"] == 4
                ],
                "required_resolution": "digest-valid exact accepted or still-active receipt",
            },
            {
                "kind": "rank18_block_has_no_attempt_identity",
                "cells": [
                    row["cell_id"]
                    for row in cells
                    if row["model"] == "qwen3.8-27b" and row["selection_rank"] == 18
                ],
                "required_resolution": (
                    "an exact cell_id plus a nonrepeatable claim; otherwise all four "
                    "remain unstarted"
                ),
            },
        ],
        "cells": cells,
        "privacy": {
            "prompts_read": False,
            "traces_read": False,
            "flags_read": False,
            "answers_read": False,
            "scores_read": False,
            "credentials_read": False,
        },
        "effects": {
            "workload_mutations": 0,
            "api_mutations": 0,
            "model_calls": 0,
            "scoring_calls": 0,
        },
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def write(repo_root: Path, output: Path = OUTPUT_PATH) -> dict[str, Any]:
    value = build(repo_root)
    target = output if output.is_absolute() else repo_root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    receipt = write(root)
    print(receipt["receipt_sha256"])
