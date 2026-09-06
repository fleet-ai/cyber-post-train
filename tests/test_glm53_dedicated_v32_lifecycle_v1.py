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
COMMIT = "c1f5301d1cf26f3b854e314e5d5f2c40b0169d04"


class FakeBackend:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.exact: dict[str, dict[str, object]] = {}
        self.items: list[dict[str, object]] = []
        self.pages = 1
        self.preview_status = 200
        self.qualification_present = False
        self.qualification_file_sha256: str | None = None

    def list_runs(self) -> tuple[list[dict[str, object]], int]:
        return copy.deepcopy(self.rows), self.pages

    def get_run(self, api_run_id: str) -> dict[str, object] | None:
        return copy.deepcopy(self.exact.get(api_run_id))

    def kubernetes_inventory(self) -> dict[str, object]:
        return {"items": copy.deepcopy(self.items)}

    def sfs_observation(
        self, *, absent_paths: tuple[str, ...], qualification_path: str | None
    ) -> dict[str, object]:
        if qualification_path is None:
            assert not self.qualification_present
        return {
            "observer_pod_name": "stable-sfs-observer",
            "observer_pod_uid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "observer_sfs_mount_path": "/shared",
            "absent_paths": list(absent_paths),
            "qualification_result_present": self.qualification_present,
            "qualification_result_file_sha256": self.qualification_file_sha256,
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


def qwen_authority() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": live_auth.QWEN_AUTHORITY_SCHEMA,
        "status": "QUALIFIED_SCORE_FREE_TERMINAL",
        "qualified_at_epoch": time.time(),
        "api_run_id": "ft-run-1234abcd",
        "server_title": "chris-cyber-evalserve-q38-dp6-z-v1",
        "server_run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-z-v1",
        "rayjob_name": "ft-run-1234abcd",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_name": "rayjob-ft-run-1234abcd-aaaaa",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "raycluster_name": "ft-run-1234abcd-abcde",
        "raycluster_uid": "33333333-3333-4333-8333-333333333333",
        "head_pod_name": "ft-run-1234abcd-abcde-head-fffff",
        "head_pod_uid": "44444444-4444-4444-8444-444444444444",
        "head_pod_node": "computeinstance-qualified-qwen",
        "service_name": "ft-run-1234abcd-abcde-head-svc",
        "service_uid": "55555555-5555-4555-8555-555555555555",
        "qualifier_job_name": "chris-cyber-q38-dp6-z-qualifier-v1",
        "qualifier_job_uid": "66666666-6666-4666-8666-666666666666",
        "qualifier_pod_name": "chris-cyber-q38-dp6-z-qualifier-v1-aaaaa",
        "qualifier_pod_uid": "77777777-7777-4777-8777-777777777777",
        "qualification_result_path": (
            "/mnt/sfs/jobs/chris-cyber-q38-dp6-z-qualifier-v1/RESULT.json"
        ),
        "qualification_result_file_sha256": "sha256:" + "8" * 64,
        "qualification_result_receipt_sha256": "sha256:" + "9" * 64,
        "requested_nodes": 1,
        "requested_gpus": 6,
        "head_pod_running_ready": True,
        "head_pod_restarts": 0,
        "qualified_score_free": True,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


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


def qualified_qwen_backend(authority: dict[str, object]) -> FakeBackend:
    backend = FakeBackend()
    api = {
        "name": authority["api_run_id"],
        "title": authority["server_title"],
        "run_dir": authority["server_run_dir"],
        "status": "RUNNING",
    }
    backend.rows = [api]
    backend.exact[str(authority["api_run_id"])] = api
    backend.qualification_present = True
    backend.qualification_file_sha256 = str(
        authority["qualification_result_file_sha256"]
    )
    backend.items = [
        _object(
            "RayJob",
            str(authority["rayjob_name"]),
            str(authority["rayjob_uid"]),
            spec={"run_dir": authority["server_run_dir"]},
            status={"jobStatus": "RUNNING"},
        ),
        _object(
            "Workload",
            str(authority["workload_name"]),
            str(authority["workload_uid"]),
        ),
        _object(
            "RayCluster",
            str(authority["raycluster_name"]),
            str(authority["raycluster_uid"]),
        ),
        _object(
            "Service",
            str(authority["service_name"]),
            str(authority["service_uid"]),
        ),
        _object(
            "Pod",
            str(authority["head_pod_name"]),
            str(authority["head_pod_uid"]),
            spec={
                "nodeName": authority["head_pod_node"],
                "containers": [
                    {
                        "env": [
                            {
                                "name": "QWEN38_RUN_DIR",
                                "value": authority["server_run_dir"],
                            }
                        ],
                        "resources": {"requests": {"nvidia.com/gpu": "6"}},
                    }
                ],
            },
            status={
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"restartCount": 0}],
            },
        ),
        _object(
            "Job",
            str(authority["qualifier_job_name"]),
            str(authority["qualifier_job_uid"]),
            status={"conditions": [{"type": "Complete", "status": "True"}]},
        ),
        _object(
            "Pod",
            str(authority["qualifier_pod_name"]),
            str(authority["qualifier_pod_uid"]),
            status={"phase": "Succeeded", "containerStatuses": [{"restartCount": 0}]},
        ),
    ]
    return backend


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
    authority = qwen_authority()
    changed = live_auth.build_live_authorization(
        backend=qualified_qwen_backend(authority),
        payload=server.payload(),
        title=server.TITLE,
        run_dir=server.RUN_DIR,
        control_result_path=server.RESULT_PATH,
        request_sha256=server.request_sha256(),
        priority_class=v24.PRIORITY_CLASS,
        qwen_authority=authority,
        now=time.time(),
    )
    server.validate_authorization(changed)
    drifted = copy.deepcopy(changed)
    drifted["coexisting_qwen_server"]["head_pod_restarts"] = 1
    drifted["receipt_sha256"] = crypto.digest_without(drifted, "receipt_sha256")
    with pytest.raises(server.CreateError, match="authorization_invalid"):
        server.validate_authorization(drifted)
    forged = copy.deepcopy(changed)
    forged["live_observation"] = value["live_observation"]
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
        ("qwen_restart", "live_qualification_invalid"),
        ("qwen_result_digest", "live_qualification_invalid"),
        ("orphan_gpu", "orphan_gpu_pod_detected"),
        ("preview", "preview_invalid"),
        ("authority_extra", "authority_invalid"),
    ],
)
def test_v32_live_builder_rejects_rehashed_drift_and_orphans(
    mutation: str, error: str
) -> None:
    authority = qwen_authority()
    backend = qualified_qwen_backend(authority)
    if mutation == "qwen_restart":
        backend.items[4]["status"]["containerStatuses"][0]["restartCount"] = 1
    elif mutation == "qwen_result_digest":
        backend.qualification_file_sha256 = "sha256:" + "0" * 64
    elif mutation == "orphan_gpu":
        backend.items.append(
            _object(
                "Pod",
                "orphan-project-gpu",
                "99999999-9999-4999-8999-999999999999",
                spec={
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "RUN_DIR",
                                    "value": (
                                        "/mnt/sfs/jobs/"
                                        "chris-cyber-evalserve-orphan"
                                    ),
                                }
                            ],
                            "resources": {
                                "requests": {"nvidia.com/gpu": "1"}
                            },
                        }
                    ]
                },
                status={"phase": "Running"},
            )
        )
    elif mutation == "preview":
        backend.preview_status = 422
    else:
        authority["unexpected"] = "must-fail"
        authority["receipt_sha256"] = crypto.digest_without(
            authority, "receipt_sha256"
        )
    with pytest.raises(live_auth.LiveAuthorizationError, match=error):
        live_auth.build_live_authorization(
            backend=backend,
            payload=server.payload(),
            title=server.TITLE,
            run_dir=server.RUN_DIR,
            control_result_path=server.RESULT_PATH,
            request_sha256=server.request_sha256(),
            priority_class=v24.PRIORITY_CLASS,
            qwen_authority=authority,
            now=time.time(),
        )


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
        live_auth.LiveAuthorizationError, match="zero_state_has_project_server"
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
