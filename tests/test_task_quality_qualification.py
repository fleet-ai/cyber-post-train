from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import task_quality_qualification as qualification

TASK_VERSION = "11111111-1111-4111-8111-111111111111"
TASK_ID = "22222222-2222-4222-8222-222222222222"
ENV_VERSION = "33333333-3333-4333-8333-333333333333"
VERIFIER_ID = "44444444-4444-4444-8444-444444444444"
VERIFIER_VERSION = "55555555-5555-4555-8555-555555555555"
EVIDENCE_RUN = "66666666-6666-4666-8666-666666666666"
VERIFIER_EXECUTION = "77777777-7777-4777-8777-777777777777"
SESSION_ID = EVIDENCE_RUN


def test_requirements_bind_exact_source_files_and_self_digest():
    root = Path(__file__).parents[1]
    path = (
        root
        / "configs/qualification/fleet-blackbox-unproven-task-quality-wave-v1.requirements.json"
    )
    value = json.loads(path.read_text())
    assert value["sha256"] == qualification.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    for reference in value["inputs"].values():
        source = root / reference["path"]
        assert qualification.file_digest(source) == reference["file_sha256"]
        logical = json.loads(source.read_text())
        assert logical["sha256"] == reference["logical_sha256"]


def _selected(key: str = "task-a", version: str = TASK_VERSION) -> dict:
    return {
        "task_key": key,
        "task_version_id": version,
        "task_id": TASK_ID,
        "task_shape": "blackbox",
        "qa_status": "clean",
    }


def _task(key: str = "task-a", version: str = TASK_VERSION) -> dict:
    return {
        "key": key,
        "eval_task_version_id": version,
        "environment_id": "cysec1-2-cmp-current-fakelook-fixture",
        "version": "v0.0.1",
        "environment_version_id": ENV_VERSION,
        "data_id": "commercial",
        "data_version": "v0.0.9",
        "seed_config": {
            "current": {
                "env_key": "current",
                "data_key": "commercial",
                "data_version": "v0.0.9",
            },
            "outlook": {
                "env_key": "outlook",
                "data_key": "mail",
                "data_version": "v0.0.2",
            },
        },
        "prompt": "private prompt",
        "env_variables": {"PRIVATE": "value"},
        "output_json_schema": {"type": "object"},
        "verifier_id": VERIFIER_ID,
        "verifier": {
            "verifier_version_id": VERIFIER_VERSION,
            "version": 1,
            "sha256": "a" * 64,
        },
        "metadata": {
            "projection_id": "blackbox_ctf_v1",
            "tools": ["bash", "submit_report"],
            "cyber_contract": qualification.REQUIRED_CONTRACT,
            "runtime_seed_manifest": {"content_sha256": "b" * 64},
            "task_graph_band": "medium",
            "cyber_subject": {
                "task_graph_id": "fixture-graph",
                "source_locator": "cyber/task-graphs/fixture-graph@2:task_graph_source",
                "atom_sources": [
                    {
                        "artifact_key": "cyber/atoms/current/fixture",
                        "atom_id": "current/fixture",
                        "locator": "cyber/atoms/current/fixture@3:atom_source",
                        "version_index": 3,
                    }
                ],
            },
        },
    }


def _binding(key: str = "task-a", family: str = "cyber/atoms/current/fixture@3") -> dict:
    value = qualification.safe_task_binding(_task(key), _selected(key))
    value["lineage"] = {**value["lineage"], "task_family": family}
    body = {name: item for name, item in value.items() if name != "binding_sha256"}
    return {**body, "binding_sha256": qualification.digest(body)}


def _plan(tasks: list[dict]) -> dict:
    source = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "origin_main_commit": "a" * 40,
        "merged_to_origin_main": True,
        "controller_path": "evals/fleet/task_quality_qualification.py",
        "controller_file_sha256": "sha256:" + "c" * 64,
    }
    return qualification.sealed(
        {
            "schema": qualification.PLAN_SCHEMA,
            "wave_id": "fixture-wave",
            "source": source,
            "inputs": {
                "inventory_file_sha256": "sha256:" + "1" * 64,
                "inventory_sha256": "sha256:" + "2" * 64,
                "coverage_file_sha256": "sha256:" + "3" * 64,
                "coverage_sha256": "sha256:" + "4" * 64,
                "protected_split_file_sha256": "sha256:" + "5" * 64,
                "protected_split_sha256": "sha256:" + "6" * 64,
            },
            "selection": {
                "protected_heldout_atom_keys_sha256": "sha256:" + "7" * 64,
            },
            "execution": {"concurrency": 1, "external_mutations_authorized": True},
            "tasks": tasks,
        }
    )


def test_safe_binding_is_composite_exact_and_content_free():
    binding = qualification.safe_task_binding(_task(), _selected())
    assert len(binding["environment"]["seed_config"]) == 2
    assert binding["runtime_seed_status"] == "exact"
    assert binding["lineage"]["task_family"] == "cyber/atoms/current/fixture"
    rendered = json.dumps(binding)
    assert "private prompt" not in rendered
    assert "PRIVATE" not in rendered
    assert "value" not in rendered


def test_safe_binding_allows_legacy_missing_tool_and_runtime_seed_for_live_probe():
    task = _task()
    task["metadata"].pop("tools")
    task["metadata"].pop("runtime_seed_manifest")
    binding = qualification.safe_task_binding(task, _selected())
    assert binding["tool_declaration_status"] == "absent_requires_live_probe"
    assert binding["runtime_seed_content_sha256"] is None
    assert binding["runtime_seed_status"].startswith("absent_")


def test_safe_binding_rejects_conflicting_tool_declaration():
    task = _task()
    task["metadata"]["tools"] = ["bash"]
    with pytest.raises(qualification.QualificationError, match="unexpected tool"):
        qualification.safe_task_binding(task, _selected())


def test_safe_binding_rejects_atom_id_artifact_key_mismatch():
    task = _task()
    task["metadata"]["cyber_subject"]["atom_sources"][0]["atom_id"] = "current/other"
    with pytest.raises(qualification.QualificationError, match="atom-source binding"):
        qualification.safe_task_binding(task, _selected())


@pytest.mark.parametrize(
    "source_locator",
    [
        "cyber/task-graphs/fixture-graph:task_graph_source",
        "cyber/task-graphs/unrelated@999:task_graph_source",
        "cyber/task-graphs/fixture-graph@2:atom_source",
    ],
)
def test_safe_binding_rejects_mutable_or_unrelated_task_graph_source(source_locator):
    task = _task()
    task["metadata"]["cyber_subject"]["source_locator"] = source_locator
    with pytest.raises(qualification.QualificationError, match="task graph source"):
        qualification.safe_task_binding(task, _selected())


def test_plan_excludes_protected_heldout_atom_and_duplicate_family(monkeypatch):
    heldout_rows = [
        {
            **_selected(f"heldout-{index}", f"00000000-0000-4000-8000-{index:012d}"),
            "qa_status": "known_broken",
        }
        for index in range(25)
    ]
    candidates = [_selected("candidate-a"), _selected("candidate-b"), _selected("candidate-c")]
    inventory = {
        "sha256": "sha256:" + "a" * 64,
        "task_count": 28,
        "tasks": [*heldout_rows, *candidates],
    }
    coverage = {
        "sha256": "sha256:" + "b" * 64,
        "no_exact_receipt_in_this_refresh_task_keys": [row["task_key"] for row in candidates],
    }
    split = {
        "sha256": "sha256:" + "c" * 64,
        "tasks": [
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "split": "dev" if index < 17 else "final_test",
            }
            for index, row in enumerate(heldout_rows)
        ],
    }

    def binding(_client, row, **_kwargs):
        if row["task_key"].startswith("heldout-"):
            result = _binding(row["task_key"], f"heldout-family-{row['task_key']}")
            result["atom_artifact_keys"] = ["cyber/atoms/final/protected"]
            return result
        result = _binding(row["task_key"], "shared" if row["task_key"] != "candidate-c" else "c")
        if row["task_key"] == "candidate-a":
            result["atom_artifact_keys"] = ["cyber/atoms/final/protected"]
        return result

    monkeypatch.setattr(qualification, "_fetch_binding", binding)
    plan = qualification.build_plan(
        inventory=inventory,
        coverage=coverage,
        split=split,
        inventory_sha256="sha256:" + "1" * 64,
        coverage_sha256="sha256:" + "2" * 64,
        split_sha256="sha256:" + "3" * 64,
        client=object(),
        wave_id="wave-1",
        qa_statuses={"clean"},
        limit=3,
        concurrency=2,
        source={"fixture": True, "merged_to_origin_main": True},
    )
    assert plan["selection"]["selected_task_versions"] == 2
    assert plan["selection"]["excluded_counts"]["protected_heldout_atom_overlap"] == 1
    assert plan["selection"]["zero_protected_heldout_atom_intersection"] is True


def test_probe_tools_calls_both_without_returning_content(monkeypatch):
    calls = []
    arguments = {}
    schemas = {
        "bash": {
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
            "additionalProperties": False,
        },
        "submit_report": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["no_flag"]},
                "explanation": {"type": "string", "minLength": 1},
            },
            "required": ["verdict", "explanation"],
            "additionalProperties": False,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            calls.append("DELETE")
            return httpx.Response(204, request=request)
        body = json.loads(request.content)
        method = body["method"]
        calls.append(method if method != "tools/call" else body["params"]["name"])
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": schemas[name]}
                    for name in ("bash", "submit_report")
                ]
            }
        else:
            arguments[body["params"]["name"]] = body["params"]["arguments"]
            result = {"content": [{"type": "text", "text": "FLAG{must-not-persist}"}]}
        return httpx.Response(
            200,
            headers={"mcp-session-id": "fixture"},
            json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            request=request,
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        qualification.httpx,
        "Client",
        lambda **_kwargs: real_client(transport=httpx.MockTransport(handler)),
    )
    receipt = qualification.probe_tools("https://fixture.invalid", "x-token", "secret")
    assert receipt["bash_reachable"] is True
    assert receipt["submit_report_reachable"] is True
    assert arguments == {
        "bash": {"script": "printf task-quality-runtime-ok"},
        "submit_report": {
            "verdict": "no_flag",
            "explanation": (
                "Automated task-runtime qualification probe; exploitability was not tested."
            ),
        },
    }
    assert "FLAG" not in json.dumps(receipt)
    assert calls == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "bash",
        "submit_report",
        "DELETE",
    ]


def test_probe_tools_rejects_schema_that_does_not_admit_negative_control(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            calls.append("DELETE")
            return httpx.Response(204, request=request)
        body = json.loads(request.content)
        method = body["method"]
        calls.append(method)
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "bash",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"script": {"type": "string"}},
                            "required": ["script"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "submit_report",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "flags": {"type": "array", "items": {"type": "string"}},
                                "explanation": {"type": "string"},
                            },
                            "required": ["flags", "explanation"],
                            "additionalProperties": False,
                        },
                    },
                ]
            }
        else:
            raise AssertionError("probe called a tool before validating the catalog schema")
        return httpx.Response(
            200,
            headers={"mcp-session-id": "fixture"},
            json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            request=request,
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        qualification.httpx,
        "Client",
        lambda **_kwargs: real_client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(qualification.QualificationError, match="does not admit"):
        qualification.probe_tools("https://fixture.invalid", "x-token", "secret")
    assert "tools/call" not in calls
    assert calls[-1] == "DELETE"


@pytest.mark.parametrize(
    ("created_new_session", "expected_status"),
    [(True, "qualified"), (False, "infrastructure_invalid")],
)
def test_qualify_one_requires_new_session_and_records_no_content(
    tmp_path, monkeypatch, created_new_session, expected_status
):
    binding = _binding()

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(qualification, "_client", lambda _key: Client())
    monkeypatch.setattr(qualification, "_account", lambda _client: None)
    monkeypatch.setattr(qualification, "_fetch_binding", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(
        self_hosted, "assert_authoritative_routes_deployed", lambda *_args: {"mode": "fixture"}
    )
    monkeypatch.setattr(
        qualification,
        "_assert_create_claim_routes_deployed",
        lambda *_args: {"mode": "openapi"},
    )

    def request(_client, method, path, **_kwargs):
        if method == "POST" and path.endswith("/instances"):
            return {
                "task_key": binding["task_key"],
                "task_version_id": binding["task_version_id"],
                "instance_id": "fixture-instance",
                "evidence_run_id": EVIDENCE_RUN,
            }
        if method == "GET" and path.startswith("/v1/env/instances/"):
            return {
                "instance_id": "fixture-instance",
                "env_key": binding["environment"]["id"],
                "version": binding["environment"]["version"],
                "status": "running",
                "terminated_at": None,
                "urls": {"root": "https://fixture.invalid"},
            }
        if method == "GET" and path == "/v1/runner-auth/token":
            return {"header": "x-token", "token": "secret"}
        if method == "POST":
            return {"private": "score response"}
        if method == "DELETE":
            return {"terminated_at": "2026-09-21T00:00:00Z"}
        raise AssertionError((method, path))

    monkeypatch.setattr(self_hosted, "_request", request)
    monkeypatch.setattr(
        qualification,
        "probe_tools",
        lambda *_args: {
            "tool_names": ["bash", "submit_report"],
            "tool_catalog_sha256": "sha256:" + "8" * 64,
            "bash_reachable": True,
            "submit_report_reachable": True,
            "tool_outputs_persisted": False,
        },
    )
    monkeypatch.setattr(
        self_hosted,
        "sanitize_authoritative_reward_response",
        lambda *_args, **_kwargs: {
            "reward": 0.0,
            "verifier_execution_id": VERIFIER_EXECUTION,
            "direct_authority_attestation": {"fixture": True},
        },
    )
    monkeypatch.setattr(
        self_hosted,
        "ingest_metadata_only_session",
        lambda *_args, **_kwargs: {
            "session_id": SESSION_ID,
            "created_new_session": created_new_session,
        },
    )
    receipt = qualification.qualify_one(
        binding, wave_id="fixture-wave", directory=tmp_path / "cell", api_key="secret"
    )
    assert receipt["qualification_status"] == expected_status
    if created_new_session:
        assert receipt["checks"]["metadata_only_session_ingested"] is True
    else:
        assert receipt["checks"]["metadata_only_session_ingested"] is False
        assert receipt["failure"]["phase"] == "session_ingest"
    rendered = "\n".join(path.read_text() for path in (tmp_path / "cell").iterdir())
    assert '"reward"' not in rendered
    assert "score response" not in rendered
    assert "FLAG{" not in rendered


def test_qualify_one_quarantines_ambiguous_provision_without_retry(tmp_path, monkeypatch):
    binding = _binding()

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(qualification, "_client", lambda _key: Client())
    monkeypatch.setattr(qualification, "_account", lambda _client: None)
    monkeypatch.setattr(qualification, "_fetch_binding", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(
        self_hosted, "assert_authoritative_routes_deployed", lambda *_args: {"mode": "fixture"}
    )
    monkeypatch.setattr(
        qualification,
        "_assert_create_claim_routes_deployed",
        lambda *_args: {"mode": "openapi"},
    )
    calls = 0

    def request(_client, method, path, **_kwargs):
        nonlocal calls
        if method == "POST" and path.endswith("/instances"):
            calls += 1
            raise httpx.ReadTimeout("ambiguous fixture response")
        raise AssertionError((method, path))

    monkeypatch.setattr(self_hosted, "_request", request)
    receipt = qualification.qualify_one(
        binding, wave_id="fixture-wave", directory=tmp_path / "cell", api_key="secret"
    )
    assert calls == 1
    assert receipt["qualification_status"] == "quarantined_ambiguous"
    assert receipt["ambiguous_external_mutation"] is True
    assert receipt["automatic_retry_performed"] is False


def test_qualify_one_fails_before_provision_when_create_claim_routes_are_missing(
    tmp_path, monkeypatch
):
    binding = _binding()

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(qualification, "_client", lambda _key: Client())
    monkeypatch.setattr(qualification, "_account", lambda _client: None)
    monkeypatch.setattr(qualification, "_fetch_binding", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(
        self_hosted, "assert_authoritative_routes_deployed", lambda *_args: {"mode": "fixture"}
    )
    post_calls = 0

    def request(_client, method, path, **_kwargs):
        nonlocal post_calls
        if method == "GET" and path == "/openapi.json":
            return {"paths": {}}
        if method == "POST":
            post_calls += 1
        raise AssertionError((method, path))

    monkeypatch.setattr(self_hosted, "_request", request)
    receipt = qualification.qualify_one(
        binding, wave_id="fixture-wave", directory=tmp_path / "cell", api_key="secret"
    )
    assert post_calls == 0
    assert receipt["qualification_status"] == "infrastructure_invalid"
    assert receipt["failure"]["phase"] == "preflight"
    assert receipt["checks"]["durable_create_claim_routes_deployed"] is False


def test_cleanup_resumes_only_the_exact_instance_after_transport_loss(tmp_path, monkeypatch):
    binding = _binding()
    config = qualification._config(binding, "fixture-wave")
    directory = tmp_path / "cell"
    directory.mkdir()
    qualification._write_once(
        directory / "PROVISION_INTENT.json",
        qualification.sealed(
            {
                "schema": "cyber_task_quality_provision_intent_v1",
                "run_id": config["run_id"],
                "task_version_id": binding["task_version_id"],
                "request_id": self_hosted.provisioning_request_id(config),
                "request_body_sha256": qualification.digest({}),
            }
        ),
    )
    qualification._write_once(
        directory / "PROVISION_RECEIPT.json",
        qualification.sealed(
            {
                "schema": "cyber_task_quality_provision_receipt_v1",
                "instance_id": "fixture-instance",
                "evidence_run_id": EVIDENCE_RUN,
                "task_version_id": binding["task_version_id"],
            }
        ),
    )

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(qualification, "_client", lambda _key: Client())
    monkeypatch.setattr(qualification, "_account", lambda _client: None)
    monkeypatch.setattr(
        qualification,
        "_optional_instance",
        lambda *_args: {
            "instance_id": "fixture-instance",
            "team_id": qualification.EXPECTED_TEAM_ID,
            "env_key": binding["environment"]["id"],
            "version": binding["environment"]["version"],
            "terminated_at": None,
        },
    )
    deletes = 0

    def request(_client, method, path, **_kwargs):
        nonlocal deletes
        assert method == "DELETE" and path == "/v1/env/instances/fixture-instance"
        deletes += 1
        if deletes == 1:
            raise httpx.ReadTimeout("ambiguous exact cleanup")
        return {"terminated_at": "2026-09-21T00:00:00Z"}

    monkeypatch.setattr(self_hosted, "_request", request)
    with pytest.raises(httpx.ReadTimeout):
        qualification.cleanup_one(
            binding, directory=directory, wave_id="fixture-wave", api_key="secret"
        )
    intent = (directory / "CLEANUP_RECOVERY_INTENT.json").read_bytes()
    receipt = qualification.cleanup_one(
        binding, directory=directory, wave_id="fixture-wave", api_key="secret"
    )
    assert deletes == 2
    assert (directory / "CLEANUP_RECOVERY_INTENT.json").read_bytes() == intent
    assert receipt["resolution"] == "recovery_delete_terminated"
    assert receipt["instance_id"] == "fixture-instance"


def test_not_analyzed_wave_requires_cumulative_attempted_catalog():
    with pytest.raises(qualification.QualificationError, match="cumulative attempted"):
        qualification.build_plan(
            inventory={"task_count": 0, "tasks": []},
            coverage={},
            split={},
            inventory_sha256="sha256:" + "1" * 64,
            coverage_sha256="sha256:" + "2" * 64,
            split_sha256="sha256:" + "3" * 64,
            client=object(),
            wave_id="wave-1",
            qa_statuses={"not_analyzed"},
            limit=1,
            concurrency=1,
            source={"fixture": True},
        )


def test_not_analyzed_wave_rejects_qualified_only_catalog():
    with pytest.raises(qualification.QualificationError, match="cumulative attempted"):
        qualification.build_plan(
            inventory={"task_count": 0, "tasks": []},
            coverage={},
            split={},
            inventory_sha256="sha256:" + "1" * 64,
            coverage_sha256="sha256:" + "2" * 64,
            split_sha256="sha256:" + "3" * 64,
            client=object(),
            wave_id="wave-1",
            qa_statuses={"not_analyzed"},
            limit=1,
            concurrency=1,
            source={"fixture": True},
            excluded_catalog={"schema": qualification.PRIVATE_CATALOG_SCHEMA},
        )


def test_execute_plan_writes_roster_inputs_and_aggregate_only(tmp_path, monkeypatch):
    binding = _binding()
    plan = _plan([binding])
    monkeypatch.setattr(
        qualification,
        "source_provenance",
        lambda **_kwargs: plan["source"],
    )

    class ContextClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(qualification, "_client", lambda _key: ContextClient())
    monkeypatch.setattr(qualification, "_account", lambda _client: None)

    def qualified(_binding, *, wave_id, directory, api_key):
        assert directory.parent == tmp_path / "cells"
        assert directory.parent.is_dir()
        directory.mkdir()
        receipt = qualification.sealed(
            {
                "schema": qualification.CELL_TERMINAL_SCHEMA,
                "wave_id": wave_id,
                "binding_sha256": _binding["binding_sha256"],
                "qualification_status": "qualified",
                "checks": {
                    name: True
                    for name in (
                        "exact_task_binding",
                        "durable_create_claim_routes_deployed",
                        "environment_started",
                        "bash_reachable",
                        "submit_report_reachable",
                        "verifier_completed",
                        "finite_authoritative_outcome",
                        "metadata_only_session_ingested",
                        "environment_cleanup_completed",
                    )
                },
            }
        )
        qualification._write_once(directory / "CELL_TERMINAL.json", receipt)
        return receipt

    monkeypatch.setattr(qualification, "qualify_one", qualified)
    aggregate = qualification.execute_plan(plan, tmp_path, api_key="secret")
    assert aggregate["counts"] == {
        "qualified": 1,
        "infrastructure_invalid": 0,
        "quarantined_ambiguous": 0,
    }
    assert (tmp_path / "SUPPLY_CATALOG.private.json").is_file()
    assert (tmp_path / "ATTEMPTED_CATALOG.private.json").is_file()
    assert (tmp_path / "QUALIFIED_CATALOG.private.json").is_file()
    assert binding["task_key"] not in json.dumps(aggregate)
    with pytest.raises(FileExistsError):
        qualification.execute_plan(plan, tmp_path, api_key="secret")
