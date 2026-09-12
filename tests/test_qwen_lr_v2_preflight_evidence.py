"""Sanitized exact-image preparation evidence is not GPU training acceptance."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-12-lr-v2-cpu-preflight-preview-v1.json"


def test_cpu_evidence_binds_exact_helper_and_all_three_fresh_plans():
    proof = json.loads(EVIDENCE.read_bytes())
    assert proof["sha256"] == digest_json({k: v for k, v in proof.items() if k != "sha256"})
    authority = proof["authority_reference"]
    study = json.loads((ROOT / authority["path"]).read_bytes())
    assert authority["file_sha256"] == file_sha256(ROOT / authority["path"])
    assert authority["embedded_sha256"] == study["sha256"]
    stage = proof["staging_reference"]
    assert stage["embedded_sha256"] == json.loads((ROOT / stage["path"]).read_bytes())["sha256"]
    helper, summary = proof["helper"], proof["cpu_summary"]
    assert helper["uid"] == summary["pod_uid"]
    assert helper["phase"] == "Succeeded" and helper["exit_code"] == helper["restarts"] == 0
    assert helper["gpu_request"] == summary["gpus"] == summary["cuda_device_count"] == 0
    assert summary["cuda_available"] is False
    assert helper["exact_uid_rechecked_before_delete"] is True
    assert helper["delete_accepted"] is True and helper["api_object_absence_verified"] is True
    assert summary["sha256"] == digest({k: v for k, v in summary.items() if k != "sha256"})
    assert summary["source_manifest_sha256"] == digest(proof["source_file_manifest"])
    assert summary["corpus_manifest_sha256"] == stage["manifest_embedded_sha256"]
    assert summary["runtime_sha256"] == proof["source_file_manifest"]["training/sft_runtime.py"]
    assert len(proof["arms"]) == len(summary["prepared"]) == 3
    for arm, observed in zip(proof["arms"], summary["prepared"], strict=True):
        cpu, preview, prepared = arm["PREFLIGHT"], arm["PREVIEW"], arm["PREPARED"]
        for value in (cpu, preview):
            assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
            for key in ("plan_sha256", "request_sha256"):
                assert value[key] == prepared[key] == observed[key]
        assert cpu["sha256"] == observed["preflight_sha256"]
        assert cpu["status"] == "passed" and cpu["gpus"] == 0
        assert cpu["counts"]["train"] == {"rows": 602, "tasks": 29, "supervised_tokens": 700359}
        assert preview["api_base_url"] == "https://api.ft.dev.flt.build"
        assert preview["errors"] == preview["warnings"] == 0
        assert preview["submitted"] is False and preview["requeue"] is False
        assert preview["gpus"] == 8 and preview["nodes"] == 1
        assert preview["image"] == helper["image_id"]
        assert observed["output_absent"] and observed["runtime_subdirectory_absent"]


def test_snapshot_keeps_admission_title_and_production_limits_explicit():
    proof = json.loads(EVIDENCE.read_bytes())
    checks = proof["live_checks"]
    assert checks["jobs_api"]["exact_prefix_name_or_output_matches"] == 0
    assert checks["jobs_api"]["independent_title_absence_verified"] is False
    assert checks["jobs_api"]["title_field_available"] is False
    assert checks["jobs_api"]["requested_titles_equal_run_names"] is True
    assert checks["wandb"]["viewer_authenticated"] and checks["wandb"]["project_accessible"]
    assert len(checks["wandb"]["exact_run_ids_absent"]) == 3
    assert checks["capacity"]["whole_eight_gpu_node_currently_free"] is False
    assert all(node["free_gpus"] == 6 for node in checks["capacity"]["dev_nodes"])
    assert checks["local_journals"]["not_a_global_cross_operator_journal_census"] is True
    assert proof["safety"]["gpu_training_job_posts"] == 0
    assert proof["safety"]["production_authorized"] is False
    assert proof["priority_policy"]["pod_class"]["value"] == 10000
    assert proof["priority_policy"]["workload_class"]["value"] == 10000
