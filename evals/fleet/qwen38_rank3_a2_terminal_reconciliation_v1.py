"""Validate the score-blind terminal reconciliation for Qwen rank3 attempt2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-rank3-a2-terminal-reconciliation-v1"
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-dedicated-rank3-a2-g22-release-v1.json"
)
RELEASE_SELF = "sha256:d99e169afe08aed6c78cf6c80c2e3f91fca563022c229516c3258b689a41933c"
RELEASE_FILE = "sha256:e5202618dc374db7135d6d57c717698acbf7de9448235c48e8acd91bbba163fb"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError("rank3/a2 terminal reconciliation digest drifted")
    return value


def validate(value: dict[str, Any], root: Path) -> None:
    release_path = root / RELEASE_PATH
    release = load(release_path)
    if (
        release["receipt_sha256"] != RELEASE_SELF
        or self_hosted.sha256(release_path.read_bytes()) != RELEASE_FILE
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "NO_AUTHORITATIVE_SESSION_OR_VERIFIER"
        or value.get("campaign_id")
        != "chris-cyber-q38-glm53-exact-easiest100-pass4-v1"
        or value.get("task_key")
        != "cysec1-2-current-gen_blackbox-607e28ace466c1f71ece3d32__blackbox_ctf_v1"
        or value.get("task_version_id") != "33d37078-0669-478e-af39-43cd245f0da8"
        or value.get("cell_id") != release["cell_id"]
        or value.get("execution_id") != release["execution_id"]
        or value.get("execution_generation") != 22
        or value.get("selection_rank") != 3
        or value.get("attempt") != 2
        or value.get("run_id") != release["run_id"]
        or value.get("source_job")
        != {
            "name": release["run_id"],
            "uid": "6d65c2e3-7162-4a02-86b9-03bbf216239b",
            "pod_uid": "30b2682f-e873-4e77-8e39-d6c41ceecc64",
            "terminal_reason": "BackoffLimitExceeded",
            "pod_exit_code": 1,
            "pod_restarts": 0,
        }
        or value.get("release_authority")
        != {
            "path": str(RELEASE_PATH),
            "receipt_sha256": RELEASE_SELF,
            "file_sha256": RELEASE_FILE,
            "prelaunch_session_rows": 29,
        }
        or value.get("fresh_authoritative_reconciliation")
        != {
            "current_session_rows": 29,
            "new_session_rows": 0,
            "matching_session_count": 0,
            "verifier_execution_presence": False,
        }
        or value.get("local_authoritative_outputs_present")
        != {
            "reward_result": False,
            "result": False,
            "session_ingest": False,
            "accepted": False,
            "terminal": False,
        }
        or value.get("classification") != "POST_MODEL_NONREPEATABLE_UNCREDITED"
        or value.get("retry_allowed") is not False
        or value.get("request_counts")
        != {"account_get": 1, "session_list_get": 1, "api_mutations": 0}
        or value.get("privacy")
        != {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        }
        or value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise ValueError("rank3/a2 terminal reconciliation fields drifted")
