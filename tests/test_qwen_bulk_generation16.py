from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from evals.fleet import qwen_bulk_generation16 as bulk
from evals.fleet import qwen_bulk_generation16_package as package
from evals.fleet import qwen_bulk_generation16_preflight as preflight
from evals.fleet import qwen_bulk_generation16_renderer as renderer
from evals.fleet import qwen_bulk_generation16_runtime as runtime
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
INVENTORY = Path("/private/tmp/exact100-inventory.XXXXXX.json")
PREFLIGHT_TOMBSTONES = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-generation16-preflight-v3-v8-terminal-tombstones-v1.json"
)
BULK_MISSING_SECRET_TOMBSTONE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-generation16-bulk-missing-secret-terminal-v1.json"
)
G17_BOOTSTRAP_TOMBSTONE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-generation17-bulk-bootstrap-terminal-v1.json"
)


def test_exact_hosted_partition_excludes_accepted_and_dedicated_cells() -> None:
    plans = bulk.validate_all(ROOT)
    assert {name: plan["new_session_count"] for name, plan in plans.items()} == {
        "qwen-a": 199,
        "qwen-b": 196,
    }
    rows = [row for plan in plans.values() for row in plan["attempts"]]
    identities = {(row["selection_rank"], row["attempt"]) for row in rows}
    assert len(rows) == len(identities) == 395
    assert not any(rank == 2 for rank, _attempt in identities)
    assert (4, 1) not in identities
    assert all(row["execution_generation"] == 17 for row in rows)
    assert {plan["job_name"] for plan in plans.values()} == {
        "chris-q38-ac-exact100-g17-a199-v1",
        "chris-q38-ac-exact100-g17-b196-v1",
    }
    assert all("-g17-" in plan["sfs_root"] for plan in plans.values())
    assert all(
        plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        for plan in plans.values()
    )


def test_attempts_are_sequential_within_complete_task_partitions() -> None:
    plans = bulk.validate_all(ROOT)
    rank_owner: dict[int, str] = {}
    for controller, plan in plans.items():
        assert plan["execution"]["attempts_per_task_sequential"] is True
        pairs = [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]]
        assert pairs == sorted(pairs)
        for rank, _attempt in pairs:
            assert rank_owner.setdefault(rank, controller) == controller


def test_g15_nullable_model_gate_and_package_are_valid() -> None:
    gate = bulk.load(ROOT / bulk.G15_GATE_PATH)
    runtime.validate_g15_gate(gate)
    assert gate["api_session"]["model_projection"] == "omitted"
    built = package.build_package(ROOT)
    assert set(built["controller_manifests"]) == {"qwen-a", "qwen-b"}
    assert all(
        size < package.prior.PACKAGE_OBJECT_LIMIT
        for size in built["object_json_bytes"].values()
    )
    packaged_paths = {
        entry["source_path"]
        for manifest in built["controller_manifests"].values()
        for obj in manifest["objects"]
        for entry in obj["entries"]
    }
    assert "evals/fleet/projected_runtime_plan.py" in packaged_paths
    run_script = (ROOT / bulk.RUN_PATH).read_text()
    assert "python3 -m evals.fleet.projected_runtime_plan" in run_script
    assert 'BULK_PLAN="$MATERIALIZED_PLAN"' in run_script
    assert 'test ! -L "$BULK_PLAN"' in run_script


def test_runtime_plans_preserve_exact_treatment() -> None:
    inventory = json.loads(INVENTORY.read_text())
    plans = {
        name: bulk.build_runtime_plan(name, inventory, ROOT) for name in bulk.CONTROLLERS
    }
    assert sum(len(plan["attempts"]) for plan in plans.values()) == 395
    assert all(plan["model"]["served_id"] == "qwen3.8-27b" for plan in plans.values())
    assert all(
        plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        for plan in plans.values()
    )
    assert all(plan["harness"]["context_window_size"] == 262144 for plan in plans.values())


def test_rendered_jobs_require_only_deployed_runtime_dependencies() -> None:
    inventory = json.loads(INVENTORY.read_text())
    rendered = renderer.render(ROOT, inventory, "a" * 40)
    jobs = [item for item in rendered["items"] if item["kind"] == "Job"]
    assert len(jobs) == 2
    secret_refs = {
        env["valueFrom"]["secretKeyRef"]["name"]
        for job in jobs
        for container in job["spec"]["template"]["spec"]["containers"]
        for env in container.get("env", [])
        if "secretKeyRef" in env.get("valueFrom", {})
    }
    assert secret_refs == {bulk.FLEET_API_KEY_SECRET}
    renderer.validate_runtime_dependencies(
        rendered,
        available_secrets={bulk.FLEET_API_KEY_SECRET},
        available_configmaps=set(),
    )
    with pytest.raises(ValueError, match="runtime dependencies are absent"):
        renderer.validate_runtime_dependencies(
            rendered,
            available_secrets=set(),
            available_configmaps=set(),
        )


def test_runtime_dependency_gate_rejects_missing_external_configmap() -> None:
    rendered = {
        "items": [
            {
                "kind": "Job",
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{"env": []}],
                            "volumes": [
                                {
                                    "name": "bootstrap",
                                    "projected": {
                                        "sources": [
                                            {"configMap": {"name": "runtime-core"}}
                                        ]
                                    },
                                }
                            ],
                        }
                    }
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="runtime-core"):
        renderer.validate_runtime_dependencies(
            rendered,
            available_secrets=set(),
            available_configmaps=set(),
        )
    renderer.validate_runtime_dependencies(
        rendered,
        available_secrets=set(),
        available_configmaps={"runtime-core"},
    )


def test_submit_path_cannot_create_when_runtime_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = json.loads(INVENTORY.read_text())
    rendered = renderer.render(ROOT, inventory, "a" * 40)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-3:] == ["secrets", "-o", "name"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[-3:] == ["configmaps", "-o", "name"]:
            names = "\n".join(
                f"configmap/{item['metadata']['name']}"
                for item in rendered["items"]
                if item["kind"] == "ConfigMap"
            )
            return subprocess.CompletedProcess(command, 0, names + "\n", "")
        pytest.fail("kubectl create was called after a missing dependency")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=bulk.FLEET_API_KEY_SECRET):
        renderer.submit_rendered_jobs(rendered)
    assert all("create" not in command for command in calls)


def test_projected_symlink_envelope_is_read_canonically(tmp_path: Path) -> None:
    payload = {"g15_gate_path": bulk.G15_GATE_PATH, "plans": [{}, {}]}
    target = tmp_path / "..data" / "plan.json"
    target.parent.mkdir()
    target.write_bytes(self_hosted.canonical_json(payload) + b"\n")
    projected = tmp_path / "preflight-plan.json"
    projected.symlink_to(target)
    assert preflight.load_projected_envelope(projected) == payload
    target.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="envelope drifted"):
        preflight.load_projected_envelope(projected)


def test_session_inventory_retries_transient_read_timeout() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("transient", request=request)
        return httpx.Response(200, json={"sessions": [], "has_more": False})

    sessions, requests = asyncio.run(
        preflight._collect_task_sessions_async(  # noqa: SLF001
            ["task-key"],
            {},
            transport=httpx.MockTransport(handler),
            retry_delay_seconds=0,
        )
    )
    assert sessions == []
    assert requests == calls == 2


def test_session_inventory_retry_exhaustion_is_terminal() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("unavailable", request=request)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(
            preflight._collect_task_sessions_async(  # noqa: SLF001
                ["task-key"],
                {},
                transport=httpx.MockTransport(handler),
                retry_delay_seconds=0,
            )
        )
    assert calls == preflight.SESSION_TRANSPORT_RETRIES + 1


def test_session_inventory_retries_connect_timeout_and_transient_http() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("connect", request=request)
        if calls == 2:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"sessions": [], "has_more": False})

    sessions, requests = asyncio.run(
        preflight._collect_task_sessions_async(  # noqa: SLF001
            ["task-key"],
            {},
            transport=httpx.MockTransport(handler),
            retry_delay_seconds=0,
        )
    )
    assert sessions == []
    assert requests == calls == 3
    assert {429, 502, 503, 504} == preflight.TRANSIENT_GET_STATUS_CODES
    assert issubclass(httpx.PoolTimeout, preflight.TRANSIENT_GET_ERRORS)


def test_session_inventory_paginates_with_exact_supported_params() -> None:
    calls: list[tuple[str, int, int]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        call = (params["task_key"], int(params["limit"]), int(params["offset"]))
        calls.append(call)
        if call[2] == 0:
            return httpx.Response(200, json={"sessions": [{"session_id": "one"}], "has_more": True})
        return httpx.Response(200, json={"sessions": [], "has_more": False})

    sessions, requests = asyncio.run(
        preflight._collect_task_sessions_async(  # noqa: SLF001
            ["task-key"], {}, transport=httpx.MockTransport(handler)
        )
    )
    assert sessions == [{"session_id": "one"}]
    assert requests == 2
    assert calls == [("task-key", 500, 0), ("task-key", 500, 1)]


def test_session_inventory_deadline_cancels_pending_requests() -> None:
    cancelled = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, json={"sessions": [], "has_more": False})

    with pytest.raises(RuntimeError, match="exceeded deadline"):
        asyncio.run(
            preflight._collect_task_sessions_async(  # noqa: SLF001
                [f"task-{index}" for index in range(16)],
                {},
                transport=httpx.MockTransport(handler),
                total_deadline_seconds=0.01,
            )
        )
    assert cancelled.is_set()


def test_session_inventory_rejects_empty_task_keys_and_is_bounded() -> None:
    with pytest.raises(RuntimeError, match="task-key inventory is invalid"):
        asyncio.run(preflight._collect_task_sessions_async([""], {}))  # noqa: SLF001
    assert preflight.SESSION_WORKERS == 8
    assert preflight.SESSION_REQUEST_TIMEOUT_SECONDS == 60
    assert preflight.SESSION_TOTAL_DEADLINE_SECONDS == 300


def test_preflight_uses_canonical_global_claim_filenames(tmp_path: Path) -> None:
    source = Path(preflight.__file__).read_text()
    assert "runtime.engine.claim_filename" in source
    execution_id = "sha256:" + "a" * 64
    expected = tmp_path / runtime.engine.claim_filename(execution_id)
    expected.write_text("{}\n")
    assert expected.exists()
    assert not (tmp_path / execution_id).exists()


def test_v3_v8_read_only_tombstones_are_digest_valid() -> None:
    receipt = json.loads(PREFLIGHT_TOMBSTONES.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert [row["generation"] for row in receipt["entries"]] == [
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
    ]
    assert all(row["read_only"] is True for row in receipt["entries"])
    assert all(row["scored_model_calls"] == 0 for row in receipt["entries"])
    assert receipt["bulk_jobs_submitted"] is False


def test_missing_secret_terminal_receipt_is_digest_valid_and_retry_safe() -> None:
    receipt = json.loads(BULK_MISSING_SECRET_TOMBSTONE.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert receipt["classification"] == "INFRASTRUCTURE_INVALID_PRE_EXECUTION"
    assert receipt["consumption_reconciliation"]["consumed_cells"] == 0
    assert all(row["container_started"] is False for row in receipt["jobs"])
    assert receipt["release"]["owned_jobs_deleted"] is True
    assert receipt["release"]["successor_state"] == "HELD"


def test_g17_bootstrap_terminal_receipt_is_digest_valid_and_retry_safe() -> None:
    receipt = json.loads(G17_BOOTSTRAP_TOMBSTONE.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert receipt["classification"] == "INFRASTRUCTURE_INVALID_PRE_EXECUTION"
    assert receipt["bootstrap_reconciliation"]["projected_package_valid"] is True
    assert receipt["bootstrap_reconciliation"]["deterministic_cause"] == (
        "projected_configmap_runtime_plan_is_symlink_rejected_by_strict_loader"
    )
    assert receipt["consumption_reconciliation"]["consumed_cells"] == 0
    assert receipt["release"] == {
        "owned_jobs_deleted": True,
        "configmaps_preserved": True,
        "successor_state": "HELD",
    }
