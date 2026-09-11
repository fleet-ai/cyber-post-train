from training import task_inventory as inventory
from training.io import digest_json


def row():
    return {
        "task_key": "cyber-example__blackbox_ctf_v1",
        "task_version_id": "11111111-1111-4111-8111-111111111111",
        "task_version": "1",
        "env_key": "cyber-env",
        "env_version": "v1",
        "environment_version_id": "22222222-2222-4222-8222-222222222222",
        "data_key": "cyber-data",
        "data_version": "v1",
        "split": "train",
        "resolution_authority": "synthetic",
    }


def response():
    return {
        "key": "cyber-example__blackbox_ctf_v1",
        "eval_task_version_id": "11111111-1111-4111-8111-111111111111",
        "environment_id": "cyber-env",
        "version": "v1",
        "environment_version_id": "22222222-2222-4222-8222-222222222222",
        "data_id": "cyber-data",
        "data_version": "v1",
        "prompt": "private synthetic prompt",
        "env_variables": {"PRIVATE": "synthetic"},
        "output_json_schema": {"type": "object"},
        "verifier_id": "synthetic-verifier",
        "verifier": {
            "verifier_version_id": "33333333-3333-4333-8333-333333333333",
            "version": "v1",
            "sha256": "sha256:" + "3" * 64,
        },
        "task_lifecycle_status": "production",
        "metadata": {
            "runtime_seed_manifest": {"content_sha256": "sha256:" + "4" * 64},
            "cyber_contract": {"verifier_contract": "3.0.0"},
            "task_graph_band": "medium",
            "cyber_subject": {
                "source_digest": "sha256:" + "5" * 64,
                "source_locator": {"artifact_key": "cyber/task-graphs/example"},
                "task_graph_id": "example",
                "atom_sources": [
                    {
                        "artifact_key": "cyber/atoms/app/vuln",
                        "version_index": 2,
                        "atom_id": "app/vuln",
                        "ignored": "private",
                    }
                ],
            },
        },
        "factual_answer": "must not persist",
    }


def test_audit_projects_only_allowlisted_metadata_and_seals_rows():
    result = inventory.audit_current_bindings(
        [row()], lambda *_: response(), observed_at="2026-09-11T00:00:00Z", workers=1
    )
    assert result["counts"] == {
        "total": 1,
        "valid": 1,
        "invalid": 0,
        "by_reason": {"valid:valid": 1},
    }
    audited = result["task_versions"][0]
    assert audited["sha256"] == digest_json({k: v for k, v in audited.items() if k != "sha256"})
    assert audited["private_fields_persisted"] is False
    assert audited["source"]["atom_sources"] == [
        {"artifact_key": "cyber/atoms/app/vuln", "version_index": 2, "atom_id": "app/vuln"}
    ]
    assert audited["taxonomy"]["application"]["value"] == ["app"]
    assert audited["taxonomy"]["difficulty"]["value"] == "medium"
    assert "private synthetic prompt" not in repr(result)
    assert "must not persist" not in repr(result)


def test_audit_classifies_known_failure_without_leaking_exception_text():
    broken = response()
    broken["data_id"] = broken["data_version"] = None
    result = inventory.audit_current_bindings(
        [row()], lambda *_: broken, observed_at="2026-09-11T00:00:00Z", workers=1
    )
    audited = result["task_versions"][0]
    assert audited["status"] == "invalid"
    assert audited["reason"] == "starting_data_absent"
    assert audited["task"] is None


def test_audit_rejects_unclassified_failure_text():
    result = inventory.audit_current_bindings(
        [row()],
        lambda *_: (_ for _ in ()).throw(RuntimeError("private server detail")),
        observed_at="2026-09-11T00:00:00Z",
        workers=1,
    )
    assert result["task_versions"][0]["reason"] == "unclassified_binding_failure"
