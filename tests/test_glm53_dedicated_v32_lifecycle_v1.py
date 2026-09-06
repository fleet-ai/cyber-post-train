import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v32_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v32_create_v1 as server
from evals.fleet import glm53_dedicated_v32_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v32_live_authorization_v1 as live_auth
from evals.fleet import glm53_dedicated_v32_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v32_watchdog_package_v1 as watchdog

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ba9493171aa1560d0a0ec7c4aee340a4371b5179"


class FakeBackend:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.exact: dict[str, dict[str, object]] = {}
        self.items: list[dict[str, object]] = []
        self.pages = 1
        self.preview_status = 200
        self.sfs_paths_override: list[str] | None = None
        self.sfs_name = live_auth.SFS_OBSERVER_POD_NAME
        self.sfs_uid = live_auth.SFS_OBSERVER_POD_UID
        self.sfs_mount = live_auth.SFS_OBSERVER_MOUNT_PATH

    def list_runs(self) -> tuple[list[dict[str, object]], int]:
        return copy.deepcopy(self.rows), self.pages

    def get_run(self, api_run_id: str) -> dict[str, object] | None:
        return copy.deepcopy(self.exact.get(api_run_id))

    def kubernetes_inventory(self) -> dict[str, object]:
        return {"items": copy.deepcopy(self.items)}

    def sfs_observation(
        self, *, absent_paths: tuple[str, ...]
    ) -> dict[str, object]:
        return {
            "observer_pod_name": self.sfs_name,
            "observer_pod_uid": self.sfs_uid,
            "observer_sfs_mount_path": self.sfs_mount,
            "absent_paths": self.sfs_paths_override or list(absent_paths),
        }

    def preview(self, _payload: object) -> int:
        return self.preview_status


def create_authorization(backend: FakeBackend | None = None) -> dict[str, object]:
    return live_auth.build_live_authorization(
        backend=backend or FakeBackend(),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=v24.PRIORITY_CLASS,
        now=time.time(),
    )


def _object(kind: str, name: str, uid: str, **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "apiVersion": "v1",
        "kind": kind,
        "metadata": {
            "name": name,
            "uid": uid,
            "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
        },
    }
    value.update(extra)
    return value


def binding() -> dict[str, object]:
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": "ft-run-1234abcd",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": (
            "http://ft-run-1234abcd-abcde-head-svc.fleet-train-jobs.svc:8000"
        ),
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def parity_authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": parity.SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_INCLUSTER_PARITY",
        "observed_at_epoch": time.time(),
        "server_binding": binding(),
        "server_ready_receipt_sha256": "sha256:" + "1" * 64,
        "watchdog_active_receipt_sha256": "sha256:" + "2" * 64,
        "job_identity_absent": True,
        "result_root_absent": True,
        "cpu_priority_class": parity.CPU_PRIORITY_CLASS,
        "cpu_preemption_policy": "Never",
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def test_v32_create_gate_requires_fresh_zero_state() -> None:
    value = create_authorization()
    server.validate_authorization(value)

    forged = copy.deepcopy(value)
    for field, changed in {
        "active_dedicated_nodes": 1,
        "active_dedicated_gpus": 6,
        "planned_nodes_after_create": 2,
        "planned_gpus_after_create": 14,
    }.items():
        forged[field] = changed
        forged["live_observation"][field] = changed
    forged["live_observation"]["receipt_sha256"] = crypto.digest_without(
        forged["live_observation"], "receipt_sha256"
    )
    forged["receipt_sha256"] = crypto.digest_without(forged, "receipt_sha256")
    with pytest.raises(server.CreateError, match="authorization_invalid"):
        server.validate_authorization(forged)
    assert server.TITLE.endswith("-v32")
    assert server.RUN_DIR.endswith("-v32")
    assert server.payload()["gpus_per_worker"] == 8
    assert server.payload()["priority_class"] == "fleet-infra-quiet"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("qwen_shaped_gpu", "requires_zero_project_server"),
        ("orphan_object", "requires_zero_project_server"),
        ("sfs", "sfs_observation_invalid"),
        ("sfs_name", "sfs_observation_invalid"),
        ("sfs_mount", "sfs_observation_invalid"),
        ("preview", "preview_invalid"),
    ],
)
def test_v32_live_builder_rejects_rehashed_drift_and_orphans(
    mutation: str, error: str
) -> None:
    backend = FakeBackend()
    if mutation == "qwen_shaped_gpu":
        backend.items.append(
            _object(
                "Pod",
                "qwen-shaped-project-gpu",
                "99999999-9999-4999-8999-999999999999",
                spec={
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "QWEN38_RUN_DIR",
                                    "value": (
                                        "/mnt/sfs/jobs/"
                                        "chris-cyber-evalserve-q38-dp6-z-v1"
                                    ),
                                }
                            ],
                            "resources": {
                                "requests": {"nvidia.com/gpu": "6"}
                            },
                        }
                    ]
                },
                status={"phase": "Running"},
            )
        )
    elif mutation == "orphan_object":
        backend.items.append(
            _object(
                "RayCluster",
                "orphan-project-cluster",
                "88888888-8888-4888-8888-888888888888",
            )
        )
    elif mutation == "sfs":
        backend.sfs_paths_override = [server.RUN_DIR]
    elif mutation == "sfs_name":
        backend.sfs_name = "arbitrary-ready-pod"
    elif mutation == "sfs_mount":
        backend.sfs_mount = "/tmp"
    elif mutation == "preview":
        backend.preview_status = 422
    with pytest.raises(live_auth.LiveAuthorizationError, match=error):
        create_authorization(backend)


def test_v32_live_builder_pages_and_rejects_hidden_active_server() -> None:
    backend = FakeBackend()
    backend.pages = 2
    hidden = {
        "name": "ft-run-deadbeef",
        "title": "another project server",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-hidden-v1",
        "status": "RUNNING",
    }
    backend.rows = [{"name": "history", "run_dir": "/tmp/history"}, hidden]
    backend.exact["ft-run-deadbeef"] = hidden
    with pytest.raises(
        live_auth.LiveAuthorizationError, match="requires_zero_project_server"
    ):
        create_authorization(backend)


def test_v32_live_builder_rejects_active_list_row_when_exact_get_is_404() -> None:
    backend = FakeBackend()
    backend.rows = [
        {
            "name": "ft-run-deadbeef",
            "title": "chris-cyber-evalserve-q38-dp6-z-v1",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-z-v1",
            "status": "RUNNING",
        }
    ]
    with pytest.raises(
        live_auth.LiveAuthorizationError, match="history_live_drift"
    ):
        create_authorization(backend)


@pytest.mark.parametrize("status", ["PENDING", "CREATED", "ADMITTED", "MYSTERY", None, 7])
def test_v32_live_builder_rejects_unknown_or_malformed_api_status(
    status: object,
) -> None:
    backend = FakeBackend()
    row = {
        "name": "ft-run-deadbeef",
        "title": "chris-cyber-evalserve-q38-dp6-z-v1",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-z-v1",
        "status": status,
    }
    backend.rows = [row]
    backend.exact["ft-run-deadbeef"] = row
    with pytest.raises(live_auth.LiveAuthorizationError, match="status_invalid"):
        create_authorization(backend)


def test_v32_live_builder_ignores_only_explicit_terminal_row_with_exact_404() -> None:
    backend = FakeBackend()
    backend.rows = [
        {
            "name": "ft-run-deadbeef",
            "title": "chris-cyber-evalserve-q38-dp6-z-v1",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-z-v1",
            "status": "FAILED",
        }
    ]
    server.validate_authorization(create_authorization(backend))


@pytest.mark.parametrize("mutation", ["control_path", "observer_name", "observer_mount"])
def test_v32_create_gate_rejects_rehashed_sfs_identity_drift(mutation: str) -> None:
    value = create_authorization()
    changed = copy.deepcopy(value)
    if mutation == "control_path":
        changed["live_observation"]["sfs_observation"]["absent_paths"][1] = (
            "/tmp/CREATED.json"
        )
    elif mutation == "observer_name":
        changed["live_observation"]["sfs_observation"]["observer_pod_name"] = (
            "arbitrary-ready-pod"
        )
    else:
        changed["live_observation"]["sfs_observation"][
            "observer_sfs_mount_path"
        ] = "/tmp"
    changed["live_observation"]["receipt_sha256"] = crypto.digest_without(
        changed["live_observation"], "receipt_sha256"
    )
    changed["receipt_sha256"] = crypto.digest_without(changed, "receipt_sha256")
    with pytest.raises(server.CreateError, match="authorization_invalid"):
        server.validate_authorization(changed)


def test_v32_live_builder_rejects_malformed_project_api_identity() -> None:
    backend = FakeBackend()
    backend.rows = [
        {
            "name": "ft-run-malformed",
            "title": "chris-cyber-evalserve-malformed",
            "run_dir": "/tmp/not-the-project-sfs-root",
            "status": "RUNNING",
        }
    ]
    with pytest.raises(
        live_auth.LiveAuthorizationError, match="project_identity_invalid"
    ):
        create_authorization(backend)


def test_v32_system_backend_pagination_progress_and_second_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(live_auth.SystemBackend)
    calls: list[str] = []

    def request(_method: str, route: str, _body: object = None):
        calls.append(route)
        if "offset=0" in route:
            return 200, {"items": [{"name": "first"}], "has_more": True}
        return 200, {"items": [{"name": "second"}], "has_more": False}

    monkeypatch.setattr(backend, "_request", request)
    rows, pages = backend.list_runs()
    assert [row["name"] for row in rows] == ["first", "second"]
    assert pages == 2
    assert calls[-1].endswith("offset=1")


def test_v32_parity_renderer_rejects_non_v30_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = {
        "objects": {
            "items": [
                {
                    "data": {
                        "run.sh": (
                            "python -m evals.fleet."
                            "glm53_dedicated_v31_incluster_parity_v1 run"
                        ),
                        "package.json": json.dumps({}),
                    }
                }
            ]
        }
    }
    monkeypatch.setattr(parity.engine, "render", lambda *_args: rendered)
    with pytest.raises(
        parity.ParityPackageError, match="entrypoint_template_invalid"
    ):
        parity.render(ROOT, COMMIT, parity_authorization())


def test_v32_materialized_parity_entrypoint_and_schema_are_generation_exact(
    tmp_path: Path,
) -> None:
    rendered = parity.render(ROOT, COMMIT, parity_authorization())
    source, auth, job = rendered["objects"]["items"]
    run = source["data"]["run.sh"]
    manifest = json.loads(source["data"]["package.json"])
    mounted_auth = json.loads(auth["data"]["authorization.json"])
    assert (
        "python -m evals.fleet.glm53_dedicated_v32_incluster_parity_v1 run"
        in run
    )
    assert "glm53_dedicated_v30_incluster_parity_v1 run" not in run
    assert "glm53_dedicated_v31_incluster_parity_v1 run" not in run
    assert manifest["run_sha256"] == crypto.sha256(run.encode())
    assert manifest["schema_version"] == parity.PACKAGE_SCHEMA
    assert manifest["server_title"] == server.TITLE
    assert manifest["server_run_dir"] == server.RUN_DIR
    assert mounted_auth["schema_version"] == parity.SCHEMA
    assert mounted_auth["receipt_sha256"] == crypto.digest_without(
        mounted_auth, "receipt_sha256"
    )
    assert job["metadata"]["name"] == parity.JOB_NAME
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    for relative in parity.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(auth["data"]["authorization.json"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.fleet.glm53_dedicated_v32_incluster_parity_v1",
            "plan",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    held = json.loads(result.stdout)
    assert held["entrypoint_module"].endswith("v32_incluster_parity_v1")
    assert held["authorization_schema"] == parity.SCHEMA
    assert held["scored_launch_authorized"] is False


def test_v32_controller_package_has_only_fresh_executable_identities(
    tmp_path: Path,
) -> None:
    rendered = package.render(ROOT, COMMIT, create_authorization())
    source, auth, job = rendered["objects"]["items"]
    manifest = json.loads(source["data"]["package.json"])
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "glm53_dedicated_v32_controller_v1" in command
    assert "glm53_dedicated_v30_controller_v1" not in command
    assert "glm53_dedicated_v31_controller_v1" not in command
    assert job["metadata"]["name"] == package.JOB_NAME
    assert manifest["schema_version"] == package.PACKAGE_SCHEMA
    assert manifest["server_title"] == server.TITLE
    assert manifest["server_run_dir"] == server.RUN_DIR
    assert auth["immutable"] is True
    assert len(json.dumps(source)) < 1_000_000
    for relative in package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v32_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v32_held_contract_is_score_free_and_idle_bounded() -> None:
    create = server.build_held()
    live = adapter.build_held(COMMIT)
    score_free = parity.build_held()
    assert watchdog.JOB_NAME.endswith("v32-request-watchdog-v1")
    assert live["idle_release_seconds"] == 600
    assert live["exactly_one_live_observation"] is True
    assert score_free["entrypoint_module"].endswith("v32_incluster_parity_v1")
    for receipt in (create, live, score_free):
        if "server_launch_authorized" in receipt:
            assert receipt["server_launch_authorized"] is False
        assert receipt["scored_launch_authorized"] is False
        assert receipt["protected_content_included"] is False
        assert receipt["receipt_sha256"] == crypto.digest_without(
            receipt, "receipt_sha256"
        )


def test_v32_held_and_v31_terminal_receipts_are_digest_valid() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v32-zero-state-lifecycle-held-v1.json"
        ).read_text()
    )
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-dedicated-v31-parity-package-failure-release-v1.json"
        ).read_text()
    )
    assert held == package.build_held(ROOT, COMMIT)
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")
    assert terminal["receipt_sha256"] == crypto.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["status"] == "RELEASED_ZERO_GPU_REMNANTS"
    assert terminal["retry_same_identity"] is False
    assert terminal["parity_result_present"] is False
    assert terminal["fleet_task_instance_calls"] == 0
    assert terminal["fleet_session_calls"] == 0
    assert terminal["verifier_calls"] == 0
    assert terminal["scoring_calls"] == 0
