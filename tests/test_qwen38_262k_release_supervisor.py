import json
import threading
from concurrent.futures import ThreadPoolExecutor
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
        self.delete_then_raise = False

    def get_rayjob(self, name: str) -> dict | None:
        self.calls.append(("get_rayjob", name))
        assert name == self.expected_name
        return deepcopy(self.root)

    def list_workloads_by_job_uid(self, uid: str) -> list[dict]:
        self.calls.append(("list_workloads_by_job_uid", uid))
        return deepcopy(self.workloads)

    def get_workload(self, name: str) -> dict | None:
        self.calls.append(("get_workload", name))
        return deepcopy(
            next(
                (item for item in self.workloads if item["metadata"]["name"] == name),
                None,
            )
        )

    def list_rayclusters_by_rayjob_uid(self, name: str, uid: str) -> list[dict]:
        self.calls.append(("list_rayclusters_by_rayjob_uid", name, uid))
        if self.cluster is None:
            return []
        return (
            [deepcopy(self.cluster)]
            if release._owned_by(self.cluster, kind="RayJob", name=name, uid=uid)
            else []
        )

    def get_raycluster(self, name: str) -> dict | None:
        self.calls.append(("get_raycluster", name))
        return deepcopy(self.cluster)

    def list_pods(self, name: str, controller_uid: str) -> list[dict]:
        self.calls.append(("list_pods", name, controller_uid))
        assert controller_uid == CLUSTER_UID
        return deepcopy(self.pods)

    def list_pods_by_run_identity(self, run_id: str, run_name: str) -> list[dict]:
        self.calls.append(("list_pods_by_run_identity", run_id, run_name))
        return deepcopy(
            [
                item
                for item in self.pods
                if item["metadata"].get("labels", {}).get("fleet.ai/run-id") == run_id
                and item["metadata"].get("labels", {}).get("fleet.ai/run-name") == run_name
            ]
        )

    def get_pod(self, name: str) -> dict | None:
        self.calls.append(("get_pod", name))
        return deepcopy(
            next((item for item in self.pods if item["metadata"]["name"] == name), None)
        )

    def delete_rayjob_uid_foreground(self, name: str, uid: str) -> None:
        self.calls.append(("delete_rayjob_uid_foreground", name, uid))
        self.delete_calls.append((name, uid))
        if self.delete_then_raise:
            self.root = None
            raise RuntimeError("simulated lost successful delete response")
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
            "labels": {
                "fleet.ai/run-id": RUN_ID,
                "fleet.ai/run-name": release.RUN_NAME,
            },
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
    delegation = json.loads((operation / "RELEASE_DELEGATION.json").read_text())
    assert delegation["armed_sha256"] == armed["sha256"]
    assert delegation["jobs_api_run_id"] == RUN_ID
    assert delegation["rayjob_name"] == manifest["metadata"]["name"]
    assert delegation["run_dir"] == release.RUN_DIR
    assert delegation["image"] == release.IMAGE
    assert (delegation["nodes"], delegation["gpus_per_node"], delegation["total_gpus"]) == (
        4,
        8,
        32,
    )
    assert delegation["plan_sha256"] == armed["plan_sha256"]
    assert delegation["request_sha256"] == armed["request_sha256"]
    assert delegation["manifest_sha256"] == armed["manifest_sha256"]
    assert delegation["derivation_rule"] == release.DELEGATION_DERIVATION_RULE
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


def test_sealed_delegation_cannot_change_the_exact_output_surface(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, _ = arm(tmp_path, backend, Clock(now))
    path = supervisor.operation_dir / "RELEASE_DELEGATION.json"
    value = json.loads(path.read_text())
    value["run_dir"] = "/mnt/sfs/jobs/another-run"
    value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
    path.write_text(json.dumps(value))

    with pytest.raises(release.SupervisorError, match="differs from the armed packet"):
        release.FourNodeReleaseSupervisor(
            supervisor.operation_dir,
            backend,
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
    assert intent["observer_receipt_sha256"] == supervisor.armed["observer_receipt_sha256"]
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


def test_binding_does_not_compare_api_server_and_launcher_wall_clocks(tmp_path) -> None:
    armed_at = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    server_time = armed_at - timedelta(days=1)
    _, _, rendered = packet()
    backend = Backend(rendered, server_time)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(armed_at))
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {
            "uid": ROOT_UID,
            "creationTimestamp": stamp(server_time),
            "resourceVersion": "9",
        }
    )
    backend.root["status"] = {"jobStatus": "PENDING"}

    binding = supervisor.reconcile_exact()

    assert binding is not None
    assert binding["rayjob_created_at"] == stamp(server_time)
    assert binding["bound_at"] == stamp(armed_at)


def test_exact_root_adoption_rejects_unreviewed_extra_runtime_surface(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "9"}
    )
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    backend.root["status"] = {"jobStatus": "PENDING"}

    with pytest.raises(release.SupervisorError, match="runtime surface changed"):
        supervisor.reconcile_exact()


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
    ttl = deepcopy(rendered)
    ttl["spec"]["ttlSecondsAfterFinished"] = 300
    with pytest.raises(release.SupervisorError, match="release contract changed"):
        release.validate_exact_candidate(plan, request, ttl)

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


def test_raycluster_is_discovered_by_exact_root_owner_when_status_name_is_absent(
    tmp_path,
) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root["status"].pop("rayClusterName")

    assert supervisor.step() is None
    assert supervisor.cluster_name == "candidate-cluster"
    assert supervisor.cluster_uid == CLUSTER_UID
    assert supervisor.allocation_at == now
    assert supervisor.current_gpus == 32


def test_zero_ownerref_clusters_with_absent_status_does_not_infer_a_name(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    backend.root["status"].pop("rayClusterName")
    backend.cluster = None
    backend.pods = [pod(0, now)]

    assert supervisor.step() is None
    assert supervisor.cluster_name == ""
    assert supervisor.allocation_at is None
    assert supervisor.current_gpus == 0


def test_status_cluster_must_agree_with_unique_ownerref_cluster(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root["status"]["rayClusterName"] = "stale-cluster"

    with pytest.raises(release.SupervisorError, match="names disagree"):
        supervisor.step()


def test_more_than_one_exact_ownerref_cluster_is_rejected(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    second = deepcopy(backend.cluster)
    second["metadata"]["name"] = "second-cluster"
    second["metadata"]["uid"] = str(UUID(int=909))
    backend.list_rayclusters_by_rayjob_uid = lambda *_: [
        deepcopy(backend.cluster),
        second,
    ]

    with pytest.raises(release.SupervisorError, match="more than one RayCluster"):
        supervisor.step()


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


def test_concurrent_observers_serialize_and_merge_state_publication(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    first, _, _, rendered = arm(tmp_path, backend, clock)
    bind(first, backend, rendered)
    authorize(first)
    resources(backend, now)
    second = release.FourNodeReleaseSupervisor(
        first.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    first.last_fault_class = "FirstObserverFault"
    first.last_fault_at = now
    second.last_fault_class = "SecondObserverFault"
    second.last_fault_at = now + timedelta(seconds=1)

    entered_write = threading.Event()
    release_write = threading.Event()
    second_started = threading.Event()
    original_write = release._write_atomic
    write_count = 0
    count_lock = threading.Lock()

    def delayed_first_state_write(path: Path, value: object) -> None:
        nonlocal write_count
        if path.name == "STATE.json":
            with count_lock:
                write_count += 1
                first_write = write_count == 1
            if first_write:
                entered_write.set()
                assert release_write.wait(timeout=5)
        original_write(path, value)

    monkeypatch.setattr(release, "_write_atomic", delayed_first_state_write)

    def run_second_step():
        second_started.set()
        return second.step()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.step)
        assert entered_write.wait(timeout=5)
        second_result = pool.submit(run_second_step)
        assert second_started.wait(timeout=5)
        assert second_result.done() is False
        release_write.set()
        assert first_result.result(timeout=5) is None
        assert second_result.result(timeout=5) is None

    state = json.loads((first.operation_dir / "STATE.json").read_text())
    assert state["revision"] == 2
    assert state["last_fault_class"] == "SecondObserverFault"
    assert state["last_fault_at"] == stamp(now + timedelta(seconds=1))
    assert state["peak_gpus"] == release.TOTAL_GPUS
    assert (first.operation_dir / "STATE.lock").stat().st_mode & 0o777 == 0o600


def test_concurrent_state_merge_rejects_impossible_pod_union_without_corrupting_state(
    tmp_path,
) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    first_backend = Backend(rendered, now)
    first, _, _, rendered = arm(tmp_path, first_backend, clock)
    bind(first, first_backend, rendered)
    authorize(first)
    resources(first_backend, now)

    second_backend = Backend(rendered, now)
    second_backend.root = deepcopy(first_backend.root)
    resources(second_backend, now)
    second_backend.pods = [pod(index, now) for index in range(4, 8)]
    second = release.FourNodeReleaseSupervisor(
        first.operation_dir,
        second_backend,
        clock=clock,
        process_alive=lambda _: True,
    )

    assert first.step() is None
    state_before = json.loads((first.operation_dir / "STATE.json").read_text())
    with pytest.raises(release.SupervisorError, match="concurrent Pod observation is malformed"):
        second.step()

    assert json.loads((first.operation_dir / "STATE.json").read_text()) == state_before
    recovered = release.FourNodeReleaseSupervisor(
        first.operation_dir,
        first_backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.pods == state_before["pods"]
    assert recovered.state_revision == 1


def test_concurrent_terminal_publisher_loads_winner_without_mutating_state(
    tmp_path, monkeypatch
) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    first, _, _, rendered = arm(tmp_path, backend, clock)
    bind(first, backend, rendered)
    authorize(first)
    second = release.FourNodeReleaseSupervisor(
        first.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    backend.root = None
    first.zero_scan_at = now - timedelta(seconds=release.POLL_SECONDS)
    second.zero_scan_at = now - timedelta(seconds=release.POLL_SECONDS)

    entered_result_write = threading.Event()
    release_result_write = threading.Event()
    second_started = threading.Event()
    original_write = release._write_once
    result_write_count = 0
    count_lock = threading.Lock()

    def delayed_first_result_write(path: Path, value: object) -> None:
        nonlocal result_write_count
        if path.name == "RESULT.json":
            with count_lock:
                result_write_count += 1
                first_write = result_write_count == 1
            if first_write:
                entered_result_write.set()
                assert release_result_write.wait(timeout=5)
        original_write(path, value)

    monkeypatch.setattr(release, "_write_once", delayed_first_result_write)

    def run_second_step():
        second_started.set()
        return second.step()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.step)
        assert entered_result_write.wait(timeout=5)
        second_result = pool.submit(run_second_step)
        assert second_started.wait(timeout=5)
        assert second_result.done() is False
        release_result_write.set()
        winner = first_result.result(timeout=5)
        loser = second_result.result(timeout=5)

    published = json.loads((first.operation_dir / "RESULT.json").read_text())
    state = json.loads((first.operation_dir / "STATE.json").read_text())
    assert winner == loser == published
    assert state["revision"] == 1
    assert winner["ownership_state_sha256"] == state["sha256"]


def test_terminal_result_winner_prevents_late_delete_intent_publication(tmp_path) -> None:
    allocated = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(allocated)
    _, _, rendered = packet()
    deleting_backend = Backend(rendered, allocated)
    deleting, _, _, rendered = arm(tmp_path, deleting_backend, clock)
    bind(deleting, deleting_backend, rendered)
    authorize(deleting)
    resources(deleting_backend, allocated)

    released_backend = Backend(rendered, allocated)
    released_backend.root = None
    finishing = release.FourNodeReleaseSupervisor(
        deleting.operation_dir,
        released_backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    clock.value += timedelta(seconds=release.ALLOCATION_TO_STARTED_SECONDS)
    finishing.zero_scan_at = clock.value - timedelta(seconds=release.POLL_SECONDS)

    delete_validation_started = threading.Event()
    release_delete_validation = threading.Event()
    original_get = deleting_backend.get_rayjob
    get_count = 0

    def block_delete_validation(name: str) -> dict | None:
        nonlocal get_count
        get_count += 1
        if get_count == 2:
            delete_validation_started.set()
            assert release_delete_validation.wait(timeout=5)
        return original_get(name)

    deleting_backend.get_rayjob = block_delete_validation
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting_result = pool.submit(deleting.step)
        assert delete_validation_started.wait(timeout=5)
        finishing_result = finishing.step()
        release_delete_validation.set()
        assert deleting_result.result(timeout=5) == finishing_result

    assert finishing_result["release_confirmed"] is True
    assert finishing_result["delete_requested"] is False
    assert not (deleting.operation_dir / "DELETE_INTENT.json").exists()
    assert deleting_backend.delete_calls == []
    recovered = release.FourNodeReleaseSupervisor(
        deleting.operation_dir,
        released_backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.step() == finishing_result


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
    final_calls = [call[0] for call in backend.calls[before:]]
    assert "get_rayjob" in final_calls
    assert "list_workloads_by_job_uid" in final_calls
    assert "list_rayclusters_by_rayjob_uid" in final_calls
    assert "get_raycluster" in final_calls
    assert "list_pods" in final_calls
    assert "get_pod" in final_calls


@pytest.mark.parametrize("defect", ["active_gpu", "state_digest"])
def test_correctly_sealed_malformed_terminal_result_is_rejected(tmp_path, defect) -> None:
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
    assert supervisor.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = supervisor.step()
    path = supervisor.operation_dir / "RESULT.json"
    path.unlink()
    malformed = {key: value for key, value in result.items() if key != "sha256"}
    if defect == "active_gpu":
        malformed["fresh_final_relist"] = {
            **malformed["fresh_final_relist"],
            "active_gpus": 8,
        }
    else:
        malformed["ownership_state_sha256"] = "sha256:" + "0" * 64
    release._write_once(path, release._seal(malformed))

    with pytest.raises(release.SupervisorError, match="terminal"):
        release.FourNodeReleaseSupervisor(
            supervisor.operation_dir,
            backend,
            clock=clock,
            process_alive=lambda _: True,
        )


def test_cleanup_only_release_requires_two_full_chain_gpu_zero_censuses(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    resources(backend, now)

    supervisor.cleanup_rejected_exact_root(deepcopy(backend.root))
    supervisor.drive_drift_cleanup_delete()
    backend.root = None
    backend.workloads = []
    backend.cluster = None
    # The controller can disappear before the first cleanup census. Exact
    # run-identity Pod discovery must still see all 32 live GPUs.
    assert supervisor.reconcile_drift_cleanup_absence() is None
    assert not (supervisor.operation_dir / "DRIFT_CLEANUP_RESULT.json").exists()

    backend.pods = []
    assert supervisor.reconcile_drift_cleanup_absence() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = supervisor.reconcile_drift_cleanup_absence()

    assert result["status"] == "cleanup_only_full_release_confirmed"
    assert result["accepted"] is False
    assert result["release_confirmed"] is True
    assert result["fresh_final_relist"] == {
        "root": 0,
        "workloads": 0,
        "rayclusters": 0,
        "pods": 0,
        "active_gpus": 0,
    }
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    assert not (supervisor.operation_dir / "BOUND.json").exists()
    assert not (supervisor.operation_dir / "RELEASE_AUTHORIZATION.json").exists()


def test_correctly_sealed_malformed_cleanup_result_is_rejected(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    supervisor.cleanup_rejected_exact_root(deepcopy(backend.root))
    supervisor.drive_drift_cleanup_delete()
    backend.root = None
    assert supervisor.reconcile_drift_cleanup_absence() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = supervisor.reconcile_drift_cleanup_absence()
    path = supervisor.operation_dir / "DRIFT_CLEANUP_RESULT.json"
    path.unlink()
    malformed = {key: value for key, value in result.items() if key != "sha256"}
    malformed["fresh_final_relist"] = {
        **malformed["fresh_final_relist"],
        "active_gpus": 8,
    }
    release._write_once(path, release._seal(malformed))
    with pytest.raises(release.SupervisorError, match="census"):
        supervisor.reconcile_drift_cleanup_absence()


def test_ordinary_and_cleanup_only_branches_are_mutually_exclusive(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()

    ordinary_backend = Backend(rendered, now)
    ordinary_dir = tmp_path / "ordinary"
    ordinary_dir.mkdir()
    ordinary, _, _, rendered = arm(ordinary_dir, ordinary_backend, Clock(now))
    bind(ordinary, ordinary_backend, rendered)
    ordinary_backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    with pytest.raises(release.SupervisorError, match="mutually exclusive"):
        ordinary.cleanup_rejected_exact_root(deepcopy(ordinary_backend.root))
    assert not (ordinary.operation_dir / "DRIFT_CLEANUP_BOUND.json").exists()

    cleanup_backend = Backend(rendered, now)
    cleanup_dir = tmp_path / "cleanup"
    cleanup_dir.mkdir()
    cleanup, _, _, rendered = arm(cleanup_dir, cleanup_backend, Clock(now))
    cleanup.write_create_intent()
    cleanup_backend.root = deepcopy(rendered)
    cleanup_backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    cleanup_backend.root["status"] = {
        "jobStatus": "RUNNING",
        "rayClusterName": "candidate-cluster",
    }
    strict_root = deepcopy(cleanup_backend.root)
    cleanup_backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    cleanup.cleanup_rejected_exact_root(deepcopy(cleanup_backend.root))
    cleanup_backend.root = strict_root
    with pytest.raises(release.SupervisorError, match="mutually exclusive"):
        cleanup.bind_created(strict_root)
    assert not (cleanup.operation_dir / "BOUND.json").exists()


def test_ordinary_atomic_branch_recovers_when_bound_derivation_was_interrupted(
    tmp_path,
) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    (supervisor.operation_dir / "BOUND.json").unlink()
    backend.root = None

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    binding = recovered.reconcile_and_authorize_exact()
    assert binding is not None and binding["rayjob_uid"] == ROOT_UID
    assert recovered.authorization is not None
    result = recovered.step()
    assert result is not None
    assert result["status"] == "release_uncertain"
    assert result["fresh_final_relist"]["active_gpus"] == 32


@pytest.mark.parametrize("crash_after", ["branch", "binding", "authorization"])
def test_cleanup_atomic_branch_resumes_every_local_receipt_prefix(tmp_path, crash_after) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    resources(backend, now)
    supervisor.cleanup_rejected_exact_root(deepcopy(backend.root))
    prefixes = {
        "branch": [
            "DRIFT_CLEANUP_BOUND.json",
            "DRIFT_CLEANUP_AUTHORIZATION.json",
            "DRIFT_CLEANUP_DELETE_INTENT.json",
        ],
        "binding": [
            "DRIFT_CLEANUP_AUTHORIZATION.json",
            "DRIFT_CLEANUP_DELETE_INTENT.json",
        ],
        "authorization": ["DRIFT_CLEANUP_DELETE_INTENT.json"],
    }
    for name in prefixes[crash_after]:
        (supervisor.operation_dir / name).unlink()
    backend.root = None

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    resumed = recovered.resume_selected_cleanup_branch()
    assert resumed is not None and resumed["accepted"] is False
    assert (supervisor.operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json").exists()
    assert recovered.drive_drift_cleanup_delete() is None
    assert recovered.reconcile_drift_cleanup_absence() is None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    assert recovered.reconcile_drift_cleanup_absence() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = recovered.reconcile_drift_cleanup_absence()
    assert result is not None and result["release_confirmed"] is True


def test_ordinary_binding_revalidates_the_live_persisted_surface(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    returned = deepcopy(backend.root)
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"

    with pytest.raises(release.SupervisorError, match="runtime surface changed"):
        supervisor.bind_created(returned)

    assert not (supervisor.operation_dir / "BRANCH_DECISION.json").exists()
    assert not (supervisor.operation_dir / "BOUND.json").exists()


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


def test_root_present_deadline_never_confirms_release_from_one_zero_scan(
    tmp_path, monkeypatch
) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    resources(backend, now)
    authorize(supervisor)
    supervisor.step()
    supervisor.journal_bound_root_release_after_observer_failure()
    clock.value += timedelta(seconds=release.DELETE_CONFIRM_SECONDS)
    original_fresh = supervisor._fresh_release_observation

    def root_vanishes_then_late_pod_appears() -> tuple[bool, dict]:
        backend.root = None
        backend.workloads = []
        backend.cluster = None
        backend.pods = []
        first_zero = original_fresh()
        backend.pods = [pod(0, now)]
        return first_zero

    monkeypatch.setattr(
        supervisor,
        "_fresh_release_observation",
        root_vanishes_then_late_pod_appears,
    )

    result = supervisor.step()

    assert result["status"] == "release_uncertain"
    assert result["release_confirmed"] is False
    assert result["reason"] == "release_confirmation_outer_bound_elapsed"
    assert result["fresh_final_relist"]["active_gpus"] == 0
    assert len(backend.pods) == 1


def test_final_absence_census_survives_selector_label_drift(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    supervisor.step()
    backend.root = None
    backend.cluster = None
    backend.pods = [backend.pods[0]]
    backend.list_workloads_by_job_uid = lambda _: []
    backend.list_pods = lambda *_: []

    released, final = supervisor.fresh_release_observation()

    assert released is False
    assert final["workloads"] == 1
    assert final["pods"] == 1
    assert final["active_gpus"] == 8


def test_final_census_finds_run_identity_pods_before_first_cluster_inventory(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    resources(backend, now)
    backend.root = None
    backend.workloads = []
    backend.cluster = None

    released, final = supervisor.fresh_release_observation()

    assert released is False
    assert final == {
        "root": 0,
        "workloads": 0,
        "rayclusters": 0,
        "pods": 4,
        "active_gpus": 32,
    }
    assert supervisor.cluster_name == ""
    assert supervisor.pods == {}


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


def test_authorized_emergency_delete_is_not_blocked_by_runtime_surface_drift(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    authorize(supervisor)
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"

    supervisor.journal_bound_root_release_after_observer_failure()

    assert backend.delete_calls == []
    supervisor._drive_bound_root_delete()
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]
    intent = json.loads((supervisor.operation_dir / "DELETE_INTENT.json").read_text())
    assert intent["reason"] == "observation_contract_defect"
    assert intent["rayjob_uid"] == ROOT_UID


def test_observer_recovers_crash_after_delete_intent_before_transport(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    bind(supervisor, backend, rendered)
    authorize(supervisor)

    supervisor.journal_bound_root_release_after_observer_failure()
    assert backend.delete_calls == []

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=Clock(now),
        process_alive=lambda _: True,
    )
    assert recovered.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]


def test_cleanup_observer_recovers_crash_after_intent_before_transport(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, Clock(now))
    supervisor.write_create_intent()
    backend.root = deepcopy(rendered)
    backend.root["metadata"].update(
        {"uid": ROOT_UID, "creationTimestamp": stamp(now), "resourceVersion": "1"}
    )
    backend.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"

    supervisor.cleanup_rejected_exact_root(deepcopy(backend.root))
    assert backend.delete_calls == []

    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=Clock(now),
        process_alive=lambda _: True,
    )
    assert recovered.drive_drift_cleanup_delete() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]


def test_delete_404_is_absent_and_uid_reuse_is_never_deleted(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    _, _, rendered = packet()

    absent_backend = Backend(rendered, now)
    absent_dir = tmp_path / "absent"
    absent_dir.mkdir()
    absent, _, _, rendered = arm(absent_dir, absent_backend, Clock(now))
    bind(absent, absent_backend, rendered)
    authorize(absent)
    absent.journal_bound_root_release_after_observer_failure()
    absent_backend.root = None
    assert absent._drive_bound_root_delete() == "absent"
    assert absent_backend.delete_calls == []

    reused_backend = Backend(rendered, now)
    reused_dir = tmp_path / "reused"
    reused_dir.mkdir()
    reused, _, _, rendered = arm(reused_dir, reused_backend, Clock(now))
    bind(reused, reused_backend, rendered)
    authorize(reused)
    reused.journal_bound_root_release_after_observer_failure()
    reused_backend.root["metadata"]["uid"] = str(UUID(int=999))
    result = reused.step()
    assert result["status"] == "release_uncertain"
    assert result["reason"] == "exact_name_uid_mismatch"
    assert reused_backend.delete_calls == []


def test_lost_successful_delete_response_reconciles_absence_without_reissue(tmp_path) -> None:
    now = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)
    clock = Clock(now)
    _, _, rendered = packet()
    backend = Backend(rendered, now)
    supervisor, _, _, rendered = arm(tmp_path, backend, clock)
    bind(supervisor, backend, rendered)
    authorize(supervisor)
    supervisor.journal_bound_root_release_after_observer_failure()
    backend.delete_then_raise = True

    with pytest.raises(RuntimeError, match="lost successful delete response"):
        supervisor._drive_bound_root_delete()
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]

    backend.delete_then_raise = False
    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.step() is None
    assert backend.delete_calls == [(rendered["metadata"]["name"], ROOT_UID)]


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


def test_uncertain_uid_delete_is_persisted_and_re_driven_by_exact_uid(tmp_path) -> None:
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

    # A fresh observer recovers the immutable intent and safely re-drives the
    # same exact-name + exact-UID preconditioned deletion.
    backend.raise_after_delete = False
    recovered = release.FourNodeReleaseSupervisor(
        supervisor.operation_dir,
        backend,
        clock=clock,
        process_alive=lambda _: True,
    )
    assert recovered.step() is None
    assert backend.delete_calls == [
        (rendered["metadata"]["name"], ROOT_UID),
        (rendered["metadata"]["name"], ROOT_UID),
    ]
    backend.root = None
    backend.workloads = []
    backend.cluster = None
    backend.pods = []
    assert recovered.step() is None
    clock.value += timedelta(seconds=release.POLL_SECONDS)
    result = recovered.step()
    assert result["release_confirmed"] is True
    assert result["ownership_chain"]["raycluster"]["uid"] == CLUSTER_UID
    assert backend.delete_calls == [
        (rendered["metadata"]["name"], ROOT_UID),
        (rendered["metadata"]["name"], ROOT_UID),
    ]
