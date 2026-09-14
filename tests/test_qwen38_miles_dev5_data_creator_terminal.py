from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-13-miles-rlreward-dev5-data-creator-v4-terminal-v1.json"
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _load() -> tuple[dict[str, object], str]:
    serialized = EVIDENCE.read_text(encoding="utf-8")
    return json.loads(serialized), serialized


def test_dev5_v4_creator_evidence_is_self_digested_and_sanitized() -> None:
    evidence, serialized = _load()
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    assert evidence["sha256"] == expected
    assert "ASIA" not in serialized
    assert re.search(r"(?<![A-Za-z0-9_])sk_[A-Za-z0-9_-]{16,}", serialized) is None
    assert "tl_apiKey_" not in serialized
    assert re.search(r"Bearer\s+[A-Za-z0-9._-]{16,}", serialized) is None
    assert evidence["privacy"] == {
        "evidence_record_contains_credentials": False,
        "evidence_record_contains_prompts": False,
        "evidence_record_contains_task_payloads": False,
        "evidence_record_contains_traces": False,
        "evidence_record_contains_flags_answers_or_sealed_scores": False,
    }


def test_dev5_v4_creator_evidence_binds_exact_uid_runtime_and_reported_digests() -> None:
    evidence, _ = _load()
    scope = evidence["scope"]
    assert scope == {
        "cluster": "dev",
        "kubernetes_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
        "namespace": "fleet-train-jobs",
        "gpus_used": 0,
    }
    assert evidence["source"] == {
        "commit": "0e6970c7f16f8199b2fa583cb19937aeecdfcfd9",
        "image_digest": ("sha256:d1d37c584e2aafdd47df1e1f3492ff3343eb3b83a5f658cf7ce2432ee8d6ef33"),
    }

    creator = evidence["creator"]
    assert creator["pod"] == {
        "uid": "d3974748-f78d-408f-8483-15e7589a2f9b",
        "phase": "Succeeded",
        "restart_count": 0,
        "gpus": 0,
    }
    assert creator["network_policy"]["uid"] == "63843340-df4d-4c3c-8f61-615facead7b8"
    assert creator["configmap"]["uid"] == "3f2aaaee-9fa5-4a54-bcf7-10beae55003e"

    report = evidence["creator_report"]
    assert report["status"] == "accepted"
    assert report["terminal"]["path"] == (
        "/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev5-v3/data-rebind-v4/TERMINAL.json"
    )
    assert report["terminal"]["reported_sha256"] == (
        "sha256:4291be0f88478f1b4b1811ebe462d521a87eed450bc8ea3a6f8454cb701680d9"
    )
    assert report["terminal"]["file_sha256"] == (
        "sha256:1901448efe9d13500d666037847a1e26141b7e16dcb5367065171badc2951810"
    )
    assert report["manifest"]["reported_sha256"] == (
        "sha256:073cbf43756b911f6d4fb6e390c677e6a3c3775a7d46940f255ed51fbaa5d171"
    )
    assert report["base_handoff_reported_sha256"] == (
        "sha256:5a21b445c875f68c3490e5b3eceb2e71f5c8d9bba644d0742e329c6bcf4eed88"
    )
    assert report["stage_receipt_reported_sha256"] == (
        "sha256:4266b9ec9fc164fdd6cc6ca442ccff8eafb2159ea8d4a4fc8b9bdf714dcb3764"
    )
    digests = [
        report["terminal"]["reported_sha256"],
        report["terminal"]["file_sha256"],
        report["manifest"]["reported_sha256"],
        report["base_handoff_reported_sha256"],
        report["stage_receipt_reported_sha256"],
    ]
    assert all(SHA256.fullmatch(value) for value in digests)
    assert report["manifest"]["rows"] == {"train": 1, "dev": 1}


def test_dev5_v4_data_pair_is_accepted_while_gpu_preview_remains_pending() -> None:
    evidence, _ = _load()

    assert evidence["evidence_status"] == "accepted"
    assert evidence["verification"] == {
        "terminal_receipt_identity": {
            "status": "accepted",
            "diagnostic_pod_uid": "6308ee81-0776-44f6-9d18-679aaf502e32",
            "restart_count": 0,
            "gpus": 0,
            "all_terminal_checks_passed": True,
            "self_digest_matches_computed": True,
        },
        "independent_data_pair_verification": {
            "status": "accepted",
            "pod_uid": "644cb02e-7753-47ce-81fe-33746d5c27fa",
            "restart_count": 0,
            "gpus": 0,
            "stage_receipt_sha256": (
                "sha256:42a5330af2a1d5cf47a29b955d638b64e8a8e6f7236e96f76f01441a0c1be897"
            ),
            "verifier_receipt_sha256": (
                "sha256:ffc6a3c999b6b35bc143aa721a52176ffcb6396903f43494d22fc666658c7ad6"
            ),
            "target_manifest_sha256": (
                "sha256:073cbf43756b911f6d4fb6e390c677e6a3c3775a7d46940f255ed51fbaa5d171"
            ),
            "target_manifest_file_sha256": (
                "sha256:7c745fa38d4f5c5c9c2c8a303865fe2eccfeeb0c5f549e61811c10486dc80679"
            ),
            "creator_receipt_status": "accepted",
            "rows": {"train": 1, "dev": 1},
            "approved_changes_per_row": 2,
            "unapproved_changes": 0,
            "policy_identity_sha256": (
                "sha256:721f49ae5df1fbd41fd408472b318766943eb2103b281b8dd28446b79d12a2bf"
            ),
            "failure": None,
        },
        "accepted_for_downstream_training": True,
        "gpu_preview": "pending",
        "next_gate": "run_the_exact_dev5_gpu_preview_before_any_production_submission",
    }
    assert evidence["cleanup"] == {
        "all_creator_cluster_objects_deleted_after_evidence_collection": True,
        "persistent_sfs_terminal_receipt_retained": True,
    }
