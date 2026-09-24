import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from cyber_post_train.direct_submit import SFT_PRODUCTION_CONTEXT, render_sft_rayjob
from cyber_post_train.jobs import digest
from tests.test_direct_submit import manifest as preview_manifest
from tests.test_direct_submit import preview
from training import qwen38_262k_release_supervisor as release
from training import sft_262k_4node_v1 as compiler
from training.sft import read_mapping

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json"
RUN_ID = "f2620000-0000-4000-8000-000000000001"
ROOT_UID = "00000000-0000-0000-0000-000000000101"
CLUSTER_UID = "00000000-0000-0000-0000-000000000102"
WORKLOAD_UID = "00000000-0000-0000-0000-000000000103"


def packet():
    plan = compiler.compile_sft(read_mapping(CONFIG), relative_to=CONFIG.parent)
    request = compiler.job_request(plan)
    rendered, _ = render_sft_rayjob(
        plan,
        request,
        preview(preview_manifest(request)),
        kubernetes_context=SFT_PRODUCTION_CONTEXT,
        run_id=RUN_ID,
    )
    return plan, request, rendered


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def owner(kind: str, name: str, uid: str) -> list[dict]:
    return [{"kind": kind, "name": name, "uid": uid, "controller": True}]


class Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class Backend:
    def __init__(self, rendered: dict, created: datetime) -> None:
        self.calls: list[tuple] = []
        self.expected_name = rendered["metadata"]["name"]
        self.root = deepcopy(rendered)
        self.root["metadata"].update(
            {"uid": ROOT_UID, "creationTimestamp": stamp(created), "resourceVersion": "1"}
        )
        self.root["status"] = {"jobStatus": "RUNNING", "rayClusterName": "candidate-cluster"}
        self.workloads: list[dict] = []
        self.cluster: dict | None = None
        self.pods: list[dict] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.raise_after_delete = False

    def get_rayjob(self, name: str) -> dict | None:
        self.calls.append(("get_rayjob", name))
        assert name == self.expected_name
        return deepcopy(self.root)

    def list_workloads_by_job_uid(self, uid: str) -> list[dict]:
        self.calls.append(("list_workloads_by_job_uid", uid))
        return deepcopy(self.workloads)

    def get_raycluster(self, name: str) -> dict | None:
        self.calls.append(("get_raycluster", name))
        return deepcopy(self.cluster)

    def list_pods(self, name: str, controller_uid: str) -> list[dict]:
        self.calls.append(("list_pods", name, controller_uid))
        assert controller_uid == CLUSTER_UID
        return deepcopy(self.pods)

    def delete_rayjob_uid_foreground(self, name: str, uid: str) -> None:
        self.calls.append(("delete_rayjob_uid_foreground", name, uid))
        self.delete_calls.append((name, uid))
        if self.raise_after_delete:
            raise RuntimeError("simulated uncertain delete response")


def resources(backend: Backend, allocated: datetime) -> None:
    backend.workloads = [
        {
            "kind": "Workload",
            "metadata": {
                "name": "candidate-workload",
                "uid": WORKLOAD_UID,
                "creationTimestamp": stamp(allocated),
                "ownerReferences": owner("RayJob", backend.root["metadata"]["name"], ROOT_UID),
            },
        }
    ]
    backend.cluster = {
        "kind": "RayCluster",
        "metadata": {
            "name": "candidate-cluster",
            "uid": CLUSTER_UID,
            "creationTimestamp": stamp(allocated),
            "ownerReferences": owner("RayJob", backend.root["metadata"]["name"], ROOT_UID),
        },
    }
    backend.pods = [pod(index, allocated) for index in range(4)]


def pod(index: int, created: datetime) -> dict:
    name = f"candidate-pod-{index}"
    return {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "uid": f"00000000-0000-0000-0000-{index + 200:012d}",
            "creationTimestamp": stamp(created),
            "ownerReferences": owner("RayCluster", "candidate-cluster", CLUSTER_UID),
        },
        "spec": {
            "nodeName": f"gpu-node-{index}",
            "initContainers": [{"name": "sfs", "resources": {}}],
            "containers": [
                {
                    "name": "trainer",
                    "image": release.IMAGE,
                    "resources": {
                        "requests": {"nvidia.com/gpu": 8},
                        "limits": {"nvidia.com/gpu": 8},
                    },
                }
            ],
        },
        "status": {
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "True",
                    "lastTransitionTime": stamp(created),
                }
            ],
            "containerStatuses": [
                {
                    "name": "trainer",
                    "imageID": "docker-pullable://" + release.IMAGE,
                    "restartCount": 0,
                }
            ],
        },
    }


def started(plan: dict, value: datetime) -> dict:
    body = {"plan_sha256": digest(plan), "started_at_unix": value.timestamp()}
    return {**body, "receipt_sha256": digest(body)}


def arm(tmp_path: Path, backend: Backend, clock: Clock, started_reader=lambda: None):
    plan, request, rendered = packet()
    backend.root = None
    supervisor = release.FourNodeReleaseSupervisor.arm(
        tmp_path / "operation",
        backend,
        plan=plan,
        request=request,
        manifest=rendered,
        observer_pid=1234,
        clock=clock,
        started_reader=started_reader,
        process_alive=lambda pid: pid == 1234,
    )
    return supervisor, plan, request, rendered


def bind(supervisor: release.FourNodeReleaseSupervisor, backend: Backend, rendered: dict):
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {
            "uid": ROOT_UID,
            "creationTimestamp": stamp(supervisor.clock()),
            "resourceVersion": "1",
        }
    )
    backend.root["status"] = {"jobStatus": "RUNNING", "rayClusterName": "candidate-cluster"}
    value = supervisor.reconcile_exact()
    assert value is not None
    return value


def authorize(supervisor: release.FourNodeReleaseSupervisor) -> None:
    assert supervisor.binding is not None
    release.persist_release_authorization(
        supervisor.operation_dir,
        supervisor.binding,
        authorized_at=supervisor.clock(),
    )
    assert (supervisor.operation_dir / "RELEASE_AUTHORIZATION.json").stat().st_mode & 0o777 == 0o600
    supervisor.reload_authorization()


def test_pre_arm_is_private_create_once_and_checks_only_the_exact_name(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, plan, request, manifest = arm(tmp_path, backend, Clock(now))

    operation = supervisor.operation_dir
    assert operation.stat().st_mode & 0o777 == 0o700
    assert backend.calls == [("get_rayjob", manifest["metadata"]["name"])]
    armed = json.loads((operation / "ARMED.json").read_text())
    assert armed["exact_name_absent_before_arm"] is True
    assert armed["plan_sha256"] == "sha256:" + digest(plan)
    assert armed["request_sha256"] == "sha256:" + digest(request)
    assert armed["manifest_sha256"] == "sha256:" + digest(manifest)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in operation.iterdir())

    with pytest.raises(FileExistsError):
        release.FourNodeReleaseSupervisor.arm(
            operation,
            backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=1234,
            clock=Clock(now),
            process_alive=lambda _: True,
        )


def test_uncertain_create_reconciles_the_journaled_exact_name_and_full_surface(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    intent = supervisor.write_create_intent()
    assert intent["observer_pid"] == 1234
    assert intent["observer_receipt_sha256"] == supervisor.armed["sha256"]
    assert supervisor.reconcile_exact() is None

    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "9"}
    )
    backend.root["status"] = {"jobStatus": "PENDING"}
    binding = supervisor.reconcile_exact()
    assert binding["rayjob_uid"] == ROOT_UID
    assert binding["rayjob_name"] == rendered["metadata"]["name"]
    assert not any(call[0].startswith("list") for call in backend.calls)

    other = tmp_path / "drift"
    other.mkdir()
    backend = Backend(rendered, now)
    drift, _, _, rendered = arm(other, backend, Clock(now))
    drift.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update({"uid": ROOT_UID, "creationTimestamp": stamp(now)})
    backend.root["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] = 2
    with pytest.raises(release.SupervisorError, match="surface changed"):
        drift.reconcile_exact()


def test_lost_create_response_is_recovered_after_launcher_process_restart(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    intent = supervisor.write_create_intent()

    # The create reached Kubernetes, but its response and the launcher process
    # were lost.  A fresh process may inspect only the one journaled exact name.
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "9"}
    )
    backend.root["status"] = {"jobStatus": "PENDING"}
    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=Clock(now),
        process_alive=lambda pid: pid == intent["observer_pid"],
    )
    binding = recovered.reconcile_exact()

    assert binding["rayjob_name"] == rendered["metadata"]["name"]
    assert binding["rayjob_uid"] == ROOT_UID
    assert binding["intent_sha256"] == intent["sha256"]
    assert backend.calls[-1] == ("get_rayjob", rendered["metadata"]["name"])
    assert not any(call[0].startswith("list") for call in backend.calls)


def test_candidate_and_live_inventory_require_exact_four_by_eight_and_image(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    plan, request, rendered = packet()
    broken = deepcopy(rendered)
    broken["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] = 4
    with pytest.raises(release.SupervisorError, match="four-node"):
        release.validate_exact_candidate(plan, request, broken)

    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.pods[0]["status"]["containerStatuses"][0]["imageID"] = (
        "containerd://" + release.IMAGE.rsplit("@", 1)[1]
    )
    supervisor.step()
    assert supervisor._topology_complete() is True
    assert len(supervisor.pods) == 4
    assert supervisor.peak_gpus == 32
    assert set(supervisor.runtime_images) == set(supervisor.pods)

    backend.pods[3]["status"]["containerStatuses"][0]["imageID"] = (
        "docker-pullable://registry/image@sha256:" + "f" * 64
    )
    with pytest.raises(release.SupervisorError, match="another image"):
        supervisor.step()


def test_exact_uid_ownership_chain_is_required(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.workloads[0]["metadata"]["ownerReferences"][0]["uid"] = str(UUID(int=999))
    with pytest.raises(release.SupervisorError, match="Workload is not owned"):
        supervisor.step()
    assert backend.delete_calls == []


def test_pending_pods_do_not_start_gpu_allocation_deadlines(tmp_path) -> None:
    created = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(created)
    _, _, rendered = packet()
    backend = Backend(rendered, created)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, created)
    authorize(supervisor)
    for item in backend.pods:
        item["spec"].pop("nodeName")
        item["status"]["conditions"] = [
            {"type": "PodScheduled", "status": "False", "lastTransitionTime": stamp(created)}
        ]

    clock.value += timedelta(seconds=release.ALLOCATION_TO_STARTED_SECONDS)
    assert supervisor.step() is None
    assert supervisor.allocation_at is None
    assert backend.delete_calls == []

    allocated = clock.value
    for item in backend.pods:
        item["spec"]["nodeName"] = "allocated-node"
        item["status"]["conditions"] = [
            {
                "type": "PodScheduled",
                "status": "True",
                "lastTransitionTime": stamp(allocated),
            }
        ]
    assert supervisor.step() is None
    assert supervisor.allocation_at == allocated
    clock.value += timedelta(seconds=release.ALLOCATION_TO_STARTED_SECONDS - 1)
    assert supervisor.step() is None
    assert backend.delete_calls == []
    clock.value += timedelta(seconds=1)
    assert supervisor.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]


def test_uid_chain_and_timing_state_survive_restart_and_terminal_ttl_cleanup(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root["status"]["jobStatus"] = "SUCCEEDED"
    supervisor.step()
    state_before = json.loads((supervisor.operation_dir / "STATE.json").read_text())

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.allocation_at == now
    assert recovered.terminal_status == "Succeeded"
    assert recovered.workload_uid == WORKLOAD_UID
    assert recovered.cluster_uid == CLUSTER_UID
    assert len(recovered.pods) == 4
    assert recovered.topology_observed is True
    assert recovered.state_revision == state_before["revision"]

    # The zero-TTL controller may remove the root and children between polls.
    backend.root = None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    assert recovered.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = recovered.step()
    assert result["release_confirmed"] is True
    assert result["ownership_chain"]["raycluster"]["uid"] == CLUSTER_UID
    assert len(result["ownership_chain"]["pods"]) == 4

    final_process = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert final_process.step() == result


def test_allocation_started_and_hard_deadlines_are_exact(tmp_path) -> None:
    allocated = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(allocated)
    _, _, rendered = packet()
    backend = Backend(rendered, allocated)
    supervisor, plan, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, allocated)
    authorize(supervisor)

    clock.value = allocated + timedelta(seconds=release.ALLOCATION_TO_STARTED_SECONDS - 1)
    assert supervisor.step() is None
    assert backend.delete_calls == []
    clock.value += timedelta(seconds=1)
    assert supervisor.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    assert supervisor.delete_reason == "allocation_started_receipt_deadline_elapsed"

    other = tmp_path / "started"
    other.mkdir()
    started_at = allocated + timedelta(seconds=300)
    clock = Clock(allocated)
    backend = Backend(rendered, allocated)
    receipt = started(plan, started_at)
    supervisor, _, _, rendered = arm(other, backend, clock, started_reader=lambda: receipt)
    bind(supervisor, backend, rendered)
    resources(backend, allocated)
    authorize(supervisor)
    clock.value = started_at
    assert supervisor.step() is None
    assert supervisor.started_at == started_at
    clock.value = started_at + timedelta(seconds=release.STARTED_TO_ACTION_SECONDS - 1)
    assert supervisor.step() is None
    assert backend.delete_calls == []
    clock.value += timedelta(seconds=1)
    assert supervisor.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    assert supervisor.delete_reason == "authenticated_started_deadline_elapsed"


def test_external_deadlines_match_the_immutable_runtime_contract() -> None:
    assert (
        release.ALLOCATION_TO_STARTED_SECONDS,
        release.STARTED_TO_ACTION_SECONDS,
        release.ALLOCATION_TO_ACTION_SECONDS,
        release.ALLOCATION_TO_RELEASE_SECONDS,
        release.TERMINAL_GRACE_SECONDS,
        release.DELETE_CONFIRM_SECONDS,
    ) == (1800, 29100, 30900, 31500, 300, 300)


def test_terminal_self_release_uses_fresh_full_relist_and_never_deletes(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root["status"]["jobStatus"] = "SUCCEEDED"
    assert supervisor.step() is None
    backend.root = None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    before = len(backend.calls)
    assert supervisor.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = supervisor.step()

    assert result["release_confirmed"] is True
    assert result["terminal_status"] == "Succeeded"
    assert result["delete_requested"] is False
    assert result["topology_and_images_observed"] is True
    assert result["fresh_final_relist"] == {
        "root": 0,
        "workloads": 0,
        "rayclusters": 0,
        "pods": 0,
        "active_gpus": 0,
    }
    assert result["active_gpus"] == 0
    assert backend.delete_calls == []
    assert [call[0] for call in backend.calls[before:]][-5:] == [
        "get_rayjob",
        "get_rayjob",
        "list_workloads_by_job_uid",
        "get_raycluster",
        "list_pods",
    ]


def test_final_zero_must_be_stable_across_fresh_relists(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    supervisor.step()

    backend.root = None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    assert supervisor.step() is None

    # A late controller-UID-selected Pod invalidates the first zero census.
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    backend.pods = [pod(0, now)]
    assert supervisor.step() is None
    assert supervisor.zero_scan_at is None

    backend.pods = []
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    assert supervisor.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = supervisor.step()
    assert result["release_confirmed"] is True
    assert result["fresh_final_relist"]["active_gpus"] == 0


def test_authorized_delete_is_root_only_uid_cas_and_waits_for_fresh_gpu_zero(tmp_path) -> None:
    allocated = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(allocated)
    _, _, rendered = packet()
    backend = Backend(rendered, allocated)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, allocated)
    authorize(supervisor)
    backend.root["status"]["jobStatus"] = "FAILED"
    supervisor.step()
    clock.value += timedelta(seconds=release.TERMINAL_GRACE_SECONDS)
    assert supervisor.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    assert (
        "delete_rayjob_uid_foreground",
        rendered["metadata"]["name"],
        ROOT_UID,
    ) in backend.calls
    assert all(call[0] != "delete_workload" for call in backend.calls)

    backend.root = None
    backend.workloads = []
    backend.cluster = None
    # One still-running owned Pod prevents a release claim.
    backend.pods = [pod(0, allocated)]
    assert supervisor.step() is None
    clock.value += timedelta(seconds=release.DELETE_CONFIRM_SECONDS)
    result = supervisor.step()
    assert result["release_confirmed"] is False
    assert result["fresh_final_relist"]["active_gpus"] == 8
    assert result["active_gpus"] == 8


def test_root_delete_is_impossible_without_exact_sealed_authorization(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root["status"]["jobStatus"] = "FAILED"
    supervisor.step()
    clock.value += timedelta(seconds=release.TERMINAL_GRACE_SECONDS)
    with pytest.raises(release.SupervisorError, match="lacks sealed authorization"):
        supervisor.step()
    assert backend.delete_calls == []


def test_run_survives_contract_fault_and_releases_only_exact_root(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    supervisor.step()
    authorize(supervisor)
    backend.pods[0]["metadata"]["ownerReferences"][0]["uid"] = str(UUID(int=999))

    def sleep(seconds: float) -> None:
        clock.value += timedelta(seconds=seconds)
        if backend.delete_calls:
            backend.root = None
            backend.workloads = []
            backend.cluster = None
            backend.pods = []

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        sleep=sleep,
        process_alive=lambda _: True,
    )
    result = recovered.run()

    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    assert result["release_confirmed"] is True
    assert result["delete_reason"] == "observation_contract_defect"
    state = json.loads((supervisor.operation_dir / "STATE.json").read_text())
    assert state["last_fault_class"] == "SupervisorError"


def test_uncertain_uid_delete_is_persisted_and_never_repeated(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    authorize(supervisor)
    backend.root["status"]["jobStatus"] = "FAILED"
    supervisor.step()
    clock.value += timedelta(seconds=release.TERMINAL_GRACE_SECONDS)
    backend.raise_after_delete = True
    with pytest.raises(RuntimeError, match="uncertain delete"):
        supervisor.step()
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]

    # A fresh process recovers the fsynced delete intent and only observes.
    backend.raise_after_delete = False
    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    backend.root = None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    assert recovered.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = recovered.step()
    assert result["release_confirmed"] is True
    assert result["ownership_chain"]["raycluster"]["uid"] == CLUSTER_UID
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
