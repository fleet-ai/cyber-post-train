import json
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train import direct_submit as direct_submit_module
from cyber_post_train.direct_submit import (
    SFT_PRODUCTION_CONTEXT,
    direct_submit_sft_once,
)
from cyber_post_train.jobs import JobsError, digest
from tests.test_direct_submit import FakeJobs, FakeKubectl, preview
from tests.test_direct_submit import manifest as preview_manifest
from tests.test_direct_submit import plan as generic_plan
from tests.test_direct_submit import request as generic_request
from tests.test_qwen38_262k_release_supervisor import (
    ROOT_UID,
    RUN_ID,
    Clock,
    packet,
    stamp,
)
from training import qwen38_262k_release_kubernetes as release_k8s
from training import qwen38_262k_release_supervisor as release
from training import sft
from training.qwen38_262k_release_kubernetes import (
    ExactCandidateDirectCreateGuard,
    ProductionReleaseKubernetesBackend,
    ReleaseKubernetesError,
)

NOW = datetime(2026, 9, 24, 18, 0, tzinfo=UTC)
CLUSTER_UID = "00000000-0000-0000-0000-000000000202"


class KubectlRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, self.responses.pop(0), "")


def test_backend_uses_only_exact_context_namespace_uid_surfaces() -> None:
    pod = {
        "kind": "Pod",
        "metadata": {
            "name": "candidate-pod",
            "ownerReferences": [
                {
                    "kind": "RayCluster",
                    "name": "candidate-cluster",
                    "uid": CLUSTER_UID,
                    "controller": True,
                }
            ],
        },
    }
    runner = KubectlRunner(
        [
            "",
            json.dumps({"kind": "WorkloadList", "items": []}),
            json.dumps({"kind": "RayCluster", "metadata": {"name": "candidate-cluster"}}),
            json.dumps({"kind": "PodList", "items": [pod]}),
            json.dumps({"kind": "Status", "status": "Success"}),
        ]
    )
    backend = ProductionReleaseKubernetesBackend(runner=runner)

    assert backend.get_rayjob("candidate") is None
    assert backend.list_workloads_by_job_uid(ROOT_UID) == []
    assert backend.get_raycluster("candidate-cluster")["kind"] == "RayCluster"
    assert backend.list_pods("candidate-cluster", CLUSTER_UID) == [pod]
    backend.delete_rayjob_uid_foreground("candidate", ROOT_UID)

    commands = [command for command, _ in runner.calls]
    assert all(
        command[:5]
        == [
            "kubectl",
            "--context",
            release.CONTEXT,
            "--request-timeout=60s",
            command[4],
        ]
        for command in commands
    )
    assert all(release.NAMESPACE in command for command in commands[:-1])
    assert commands[1][4:6] == ["get", "workloads.kueue.x-k8s.io"]
    assert not any(item.startswith("--selector=") for item in commands[1])
    assert commands[3][4:6] == ["get", "pods"]
    assert not any(item.startswith("--selector=") for item in commands[3])
    delete_command, delete_kwargs = runner.calls[-1]
    assert delete_command[-4:] == [
        "--raw",
        f"/apis/ray.io/v1/namespaces/{release.NAMESPACE}/rayjobs/candidate",
        "-f",
        "-",
    ]
    assert json.loads(delete_kwargs["input"]) == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": ROOT_UID},
        "propagationPolicy": "Foreground",
    }
    assert not hasattr(backend, "delete_workload")
    assert not hasattr(backend, "delete_raycluster")
    assert not hasattr(backend, "delete_pod")


def test_backend_unfiltered_census_ignores_a_pod_from_another_controller_uid() -> None:
    runner = KubectlRunner(
        [
            json.dumps(
                {
                    "kind": "PodList",
                    "items": [
                        {
                            "metadata": {
                                "name": "foreign",
                                "ownerReferences": [
                                    {
                                        "kind": "RayCluster",
                                        "name": "candidate-cluster",
                                        "uid": ROOT_UID,
                                        "controller": True,
                                    }
                                ],
                            }
                        }
                    ],
                }
            )
        ]
    )
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    assert backend.list_pods("candidate-cluster", CLUSTER_UID) == []


def test_backend_discovers_raycluster_only_by_exact_controller_ownerref() -> None:
    owned = {
        "kind": "RayCluster",
        "metadata": {
            "name": "owned",
            "ownerReferences": [
                {
                    "kind": "RayJob",
                    "name": "candidate",
                    "uid": ROOT_UID,
                    "controller": True,
                }
            ],
        },
    }
    foreign = deepcopy(owned)
    foreign["metadata"]["name"] = "foreign"
    foreign["metadata"]["ownerReferences"][0]["uid"] = CLUSTER_UID
    runner = KubectlRunner([json.dumps({"kind": "RayClusterList", "items": [owned, foreign]})])
    backend = ProductionReleaseKubernetesBackend(runner=runner)

    assert backend.list_rayclusters_by_rayjob_uid("candidate", ROOT_UID) == [owned]
    command = runner.calls[0][0]
    assert command[4:6] == ["get", "rayclusters.ray.io"]
    assert not any(item.startswith("--selector=") for item in command)


def test_backend_workload_census_survives_missing_job_uid_label() -> None:
    owned = {
        "kind": "Workload",
        "metadata": {
            "name": "owned",
            "ownerReferences": [
                {
                    "kind": "RayJob",
                    "name": "candidate",
                    "uid": ROOT_UID,
                    "controller": True,
                }
            ],
        },
    }
    foreign = deepcopy(owned)
    foreign["metadata"]["name"] = "foreign"
    foreign["metadata"]["ownerReferences"][0]["uid"] = CLUSTER_UID
    runner = KubectlRunner([json.dumps({"kind": "WorkloadList", "items": [owned, foreign]})])
    backend = ProductionReleaseKubernetesBackend(runner=runner)

    assert backend.list_workloads_by_job_uid(ROOT_UID) == [owned]
    command = runner.calls[0][0]
    assert command[4:6] == ["get", "workloads.kueue.x-k8s.io"]
    assert not any(item.startswith("--selector=") for item in command)


def test_backend_pod_census_uses_exact_sealed_run_identity_without_selector() -> None:
    owned = {
        "kind": "Pod",
        "metadata": {
            "name": "owned",
            "labels": {
                "fleet.ai/run-id": RUN_ID,
                "fleet.ai/run-name": release.RUN_NAME,
            },
        },
    }
    foreign = deepcopy(owned)
    foreign["metadata"]["name"] = "foreign"
    foreign["metadata"]["labels"]["fleet.ai/run-id"] = str(UUID(int=999))
    runner = KubectlRunner([json.dumps({"kind": "PodList", "items": [owned, foreign]})])
    backend = ProductionReleaseKubernetesBackend(runner=runner)

    assert backend.list_pods_by_run_identity(RUN_ID, release.RUN_NAME) == [owned]
    command = runner.calls[0][0]
    assert command[4:6] == ["get", "pods"]
    assert not any(item.startswith("--selector=") for item in command)


class ExactBackendRunner:
    def __init__(self) -> None:
        self.root: dict | None = None
        self.get_names: list[str] = []
        self.calls: list[list[str]] = []
        self.fail_gets = 0
        self.deleted: list[tuple[str, str]] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        args = command[4:]
        if args[:2] == ["get", "rayjobs.ray.io"]:
            name = args[2]
            self.get_names.append(name)
            if self.fail_gets:
                self.fail_gets -= 1
                raise ReleaseKubernetesError("synthetic transient read failure")
            output = "" if self.root is None else json.dumps(self.root)
            return subprocess.CompletedProcess(command, 0, output, "")
        if args[:2] == ["get", "workloads.kueue.x-k8s.io"]:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"kind": "WorkloadList", "items": []}), ""
            )
        if args[:2] == ["get", "rayclusters.ray.io"]:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"kind": "RayClusterList", "items": []}), ""
            )
        if args[:2] == ["get", "pods"]:
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"kind": "PodList", "items": []}), ""
            )
        if args[:2] == ["delete", "--raw"]:
            body = json.loads(kwargs["input"])
            name = args[2].rsplit("/", 1)[1]
            uid = body["preconditions"]["uid"]
            self.deleted.append((name, uid))
            self.root = None
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"kind": "Status", "status": "Success"}), ""
            )
        raise AssertionError(f"unexpected Kubernetes operation: {args}")


class ExactKubectl(FakeKubectl):
    context = SFT_PRODUCTION_CONTEXT

    def __init__(self, runner: ExactBackendRunner, mode: str = "success") -> None:
        super().__init__()
        self.runner = runner
        self.mode = mode

    def create_once(self, obj: dict) -> dict:
        self.calls.append(("create", digest(obj)))
        if self.mode == "raise_without_store":
            raise JobsError("synthetic create uncertainty")
        result = deepcopy(obj)
        result["metadata"].update(
            {"uid": ROOT_UID, "creationTimestamp": stamp(NOW), "resourceVersion": "1"}
        )
        if self.mode == "mutated_surface":
            result["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
        if self.mode == "mismatched_identity":
            result["metadata"]["annotations"]["fleet.ai/run-id"] = str(UUID(int=999))
        self.created = deepcopy(result)
        self.runner.root = deepcopy(result)
        if self.mode == "raise_after_store":
            raise JobsError("synthetic lost create response")
        if self.mode == "crash_after_store":
            raise SystemExit("synthetic launcher death")
        return result

    def get_rayjob(self, name: str) -> dict:
        self.calls.append(("get", name))
        assert self.runner.root is not None
        assert self.runner.root["metadata"]["name"] == name
        return deepcopy(self.runner.root)


def exact_inputs() -> tuple[dict, dict, dict]:
    plan, request, rendered = packet()
    return plan, request, preview(preview_manifest(request))


def guard(
    operation_dir: Path,
    runner: ExactBackendRunner,
    *,
    alive=lambda _: True,
) -> ExactCandidateDirectCreateGuard:
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    holder: dict[str, ExactCandidateDirectCreateGuard] = {}
    clock = Clock(NOW)
    monotonic_value = 0.0

    def launch(
        operation: Path,
        plan: dict,
        request: dict,
        manifest: dict,
    ) -> dict:
        receipt = release_k8s._ready_receipt(
            operation,
            plan,
            request,
            manifest,
            pid=1234,
            now=NOW,
        )
        release._write_once(release_k8s._ready_path(operation), receipt)
        return receipt

    def simulate_detached_observer_tick(seconds: float) -> None:
        nonlocal monotonic_value
        monotonic_value += seconds
        clock.value += timedelta(seconds=seconds)
        candidate = holder["guard"]
        supervisor = candidate._armed()
        binding_path = operation_dir / "BOUND.json"
        authorization_path = operation_dir / "RELEASE_AUTHORIZATION.json"
        takeover_path = release_k8s._takeover_path(operation_dir, candidate.recovery_token)
        # This callback represents the independently running observer. The
        # production direct guard never invokes either delete driver itself.
        if (operation_dir / "DRIFT_CLEANUP_DELETE_INTENT.json").exists():
            supervisor.drive_drift_cleanup_delete()
        if binding_path.exists() and authorization_path.exists() and not takeover_path.exists():
            binding = json.loads(binding_path.read_text())
            authorization = json.loads(authorization_path.read_text())
            receipt = release_k8s._takeover_receipt(
                supervisor,
                binding,
                authorization,
                pid=candidate.observer_pid,
                now=NOW,
            )
            release._write_once(takeover_path, receipt)

    def launch_recovery(operation: Path, supervisor: release.FourNodeReleaseSupervisor) -> dict:
        token = "00000000-0000-4000-8000-000000005678"
        receipt = release_k8s._recovery_ready_receipt(
            supervisor,
            token=token,
            pid=5678,
            now=NOW,
        )
        path = release_k8s._recovery_ready_path(operation, token)
        if not path.exists():
            release._write_once(path, receipt)
        return receipt

    candidate = ExactCandidateDirectCreateGuard(
        operation_dir,
        backend,
        launch_observer=launch,
        launch_recovery_observer=launch_recovery,
        clock=clock,
        process_alive=alive,
        monotonic=lambda: monotonic_value,
        sleep=simulate_detached_observer_tick,
        started_visible=lambda: True,
    )
    holder["guard"] = candidate
    return candidate


def submit(
    tmp_path: Path,
    runner: ExactBackendRunner,
    kube: ExactKubectl,
    release_guard: ExactCandidateDirectCreateGuard,
) -> dict:
    plan, request, preview_value = exact_inputs()
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    return direct_submit_sft_once(
        plan=plan,
        request=request,
        jobs=FakeJobs(preview_value=preview_value),
        kubectl=kube,
        journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
        run_id=RUN_ID,
        jobs_root=jobs_root,
        qwen38_262k_release_guard=release_guard,
    )


def test_exact_candidate_requires_guard_before_any_external_call(tmp_path) -> None:
    plan, request, _ = exact_inputs()
    jobs, kube = FakeJobs(), FakeKubectl()
    with pytest.raises(JobsError, match="dedicated release supervisor"):
        direct_submit_sft_once(
            plan=plan,
            request=request,
            jobs=jobs,
            kubectl=kube,
            journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
            run_id=RUN_ID,
        )
    assert jobs.calls == [] and kube.calls == []


def test_exact_candidate_default_identity_uses_a_fresh_local_random_uuid() -> None:
    plan, request, rendered = packet()
    preview_value = preview(preview_manifest(request))
    first, _ = direct_submit_module.render_sft_rayjob(
        plan,
        request,
        preview_value,
        kubernetes_context=SFT_PRODUCTION_CONTEXT,
    )
    second, _ = direct_submit_module.render_sft_rayjob(
        plan,
        request,
        preview_value,
        kubernetes_context=SFT_PRODUCTION_CONTEXT,
    )
    first_id = UUID(first["metadata"]["annotations"]["fleet.ai/run-id"])
    second_id = UUID(second["metadata"]["annotations"]["fleet.ai/run-id"])

    assert first_id.version == 4 and second_id.version == 4
    assert first_id != second_id
    assert first["metadata"]["name"] != second["metadata"]["name"]


def test_exact_candidate_rejects_remote_output_absence_instead_of_direct_sfs(tmp_path) -> None:
    plan, request, _ = exact_inputs()
    runner = ExactBackendRunner()
    jobs, kube = FakeJobs(), ExactKubectl(runner)

    with pytest.raises(JobsError, match="cannot use a remote output-absence receipt"):
        direct_submit_sft_once(
            plan=plan,
            request=request,
            jobs=jobs,
            kubectl=kube,
            journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
            run_id=RUN_ID,
            output_absence_receipt={"status": "remote-only"},
            qwen38_262k_release_guard=guard(tmp_path / "operation", runner),
        )

    assert jobs.calls == [] and kube.calls == []


def test_guard_cannot_be_attached_to_another_sft_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sft, "job_request", lambda _: generic_request())
    with pytest.raises(JobsError, match="cannot guard another"):
        direct_submit_sft_once(
            plan=generic_plan(),
            request=generic_request(),
            jobs=FakeJobs(),
            kubectl=FakeKubectl(),
            journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
            qwen38_262k_release_guard=object(),
        )


def test_public_cli_wires_fresh_exact_guard_before_the_candidate_create(
    tmp_path, monkeypatch
) -> None:
    plan, request, _ = packet()
    calls: list[dict] = []
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_submission_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_external_action_gate", lambda *args: None)
    monkeypatch.setattr(cli, "_require_preflight", lambda *args: None)
    monkeypatch.setattr(cli, "_client_for_plan", lambda *_: nullcontext("jobs"))
    monkeypatch.setattr(
        direct_submit_module,
        "direct_submit_sft_once",
        lambda **kwargs: calls.append(kwargs) or {"submitted": False},
    )

    result = CliRunner().invoke(
        cli.app,
        ["direct-submit-sft", str(tmp_path), "--context", release.CONTEXT],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    wired = calls[0]["qwen38_262k_release_guard"]
    assert type(wired) is ExactCandidateDirectCreateGuard
    assert wired.operation_dir == tmp_path / "RELEASE_SUPERVISION"
    assert type(wired.backend) is ProductionReleaseKubernetesBackend


def test_direct_create_arms_journals_checks_liveness_binds_returned_uid(tmp_path) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner)
    operation = tmp_path / "operation"
    release_guard = guard(operation, runner)
    result = submit(tmp_path, runner, kube, release_guard)

    assert result["uid"] == ROOT_UID
    assert result["release_binding_sha256"].startswith("sha256:")
    assert result["release_authorization_sha256"].startswith("sha256:")
    assert result["release_takeover_sha256"].startswith("sha256:")
    assert [call[0] for call in kube.calls].count("create") == 1
    direct_intent = json.loads((tmp_path / "DIRECT_SUBMISSION.jsonl").read_text().splitlines()[0])
    assert direct_intent["release_supervisor_pid"] == 1234
    assert (
        direct_intent["release_supervisor_receipt_sha256"] == release_guard.observer_receipt_sha256
    )
    create_intent = json.loads((operation / "CREATE_INTENT.json").read_text())
    assert create_intent["observer_receipt_sha256"] == release_guard.observer_receipt_sha256
    assert json.loads((operation / "BOUND.json").read_text())["rayjob_uid"] == ROOT_UID
    assert (
        json.loads((operation / "RELEASE_AUTHORIZATION.json").read_text())["delegation_sha256"]
        == json.loads((operation / "RELEASE_DELEGATION.json").read_text())["sha256"]
    )
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in operation.iterdir())


def test_observer_liveness_is_the_last_check_before_the_sole_create(tmp_path) -> None:
    events: list[str] = []

    class OrderedRunner(ExactBackendRunner):
        def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            events.append("exact-get")
            return super().__call__(command, **kwargs)

    class OrderedKubectl(ExactKubectl):
        def create_once(self, obj: dict) -> dict:
            events.append("create")
            return super().create_once(obj)

    def alive(_: int) -> bool:
        events.append("alive")
        return True

    runner = OrderedRunner()
    submit(
        tmp_path,
        runner,
        OrderedKubectl(runner),
        guard(tmp_path / "operation", runner, alive=alive),
    )
    create_index = events.index("create")
    assert events[create_index - 1 : create_index + 1] == ["alive", "create"]


def test_lost_create_response_reconciles_exact_name_once_without_retry(tmp_path) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner, "raise_after_store")
    result = submit(tmp_path, runner, kube, guard(tmp_path / "operation", runner))
    assert result["create_response_reconciled"] is True
    assert result["uid"] == ROOT_UID
    assert [call[0] for call in kube.calls].count("create") == 1
    assert set(runner.get_names) == {runner.root["metadata"]["name"]}


def test_ambiguous_create_boundedly_recovers_a_late_exact_name_without_retry(
    tmp_path,
) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner, "raise_without_store")
    candidate = guard(tmp_path / "operation", runner)
    original_sleep = candidate.sleep
    reads = 0

    def reveal(seconds: float) -> None:
        nonlocal reads
        reads += 1
        if reads == 2:
            root = deepcopy(candidate._armed().manifest)
            root["metadata"].update(
                {"uid": ROOT_UID, "creationTimestamp": stamp(NOW), "resourceVersion": "1"}
            )
            runner.root = root
        original_sleep(seconds)

    candidate.sleep = reveal
    result = submit(tmp_path, runner, kube, candidate)

    assert result["create_response_reconciled"] is True
    assert result["uid"] == ROOT_UID
    assert [call[0] for call in kube.calls].count("create") == 1
    assert len(runner.get_names) >= 3


def test_created_surface_drift_uses_cleanup_only_uid_delete_and_is_never_accepted(
    tmp_path,
) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner, "mutated_surface")
    operation = tmp_path / "operation"
    release_guard = guard(operation, runner)

    with pytest.raises(JobsError, match="rejected and exact UID cleanup was requested"):
        submit(tmp_path, runner, kube, release_guard)

    expected_name = kube.created["metadata"]["name"]
    assert runner.deleted == [(expected_name, ROOT_UID)]
    assert [call[0] for call in kube.calls].count("create") == 1
    assert not (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()
    assert json.loads((operation / "DRIFT_CLEANUP_BOUND.json").read_text())["status"] == (
        "cleanup_only_exact_uid_not_accepted"
    )
    assert (
        json.loads((operation / "DRIFT_CLEANUP_AUTHORIZATION.json").read_text())[
            "acceptance_forbidden"
        ]
        is True
    )
    assert not hasattr(release_guard.backend, "delete_pod")


def test_cleanup_rejection_ignores_response_and_rereads_only_the_exact_name(
    tmp_path,
) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    persisted = ExactKubectl(runner).create_once(manifest)
    persisted["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    runner.root = deepcopy(persisted)
    untrusted_response = deepcopy(persisted)
    untrusted_response["metadata"]["uid"] = str(UUID(int=999))

    evidence = candidate.cleanup_rejected_created(untrusted_response)

    assert evidence["accepted"] is False
    assert runner.deleted == [(manifest["metadata"]["name"], ROOT_UID)]
    assert set(runner.get_names) == {manifest["metadata"]["name"]}


def test_cleanup_recovery_launch_failure_stays_read_only_then_seals_uncertainty(
    tmp_path,
) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    persisted = ExactKubectl(runner).create_once(manifest)
    persisted["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    runner.root = deepcopy(persisted)

    class Timer:
        value = 0.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += max(seconds, release.DELETE_CONFIRM_SECONDS + 1)

    attempts: list[bool] = []

    def fail_recovery(*_) -> dict:
        attempts.append(True)
        raise release.SupervisorError("synthetic recovery launch failure")

    timer = Timer()
    candidate.process_alive = lambda _: False
    candidate.launch_recovery_observer = fail_recovery
    candidate.monotonic = timer.monotonic
    candidate.sleep = timer.sleep

    with pytest.raises(release.SupervisorError, match="possible resource leak"):
        candidate.cleanup_rejected_created({"untrusted": True})

    result = json.loads((operation / "DRIFT_CLEANUP_RESULT.json").read_text())
    assert result["status"] == "cleanup_only_release_uncertain"
    assert result["release_confirmed"] is False
    assert result["fresh_final_relist"]["root"] == 1
    assert runner.deleted == []
    assert attempts


def test_cleanup_rejection_hands_dead_observer_to_same_intent_recovery(tmp_path) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    persisted = ExactKubectl(runner).create_once(manifest)
    persisted["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
    runner.root = deepcopy(persisted)
    candidate.process_alive = lambda pid: pid == 5678

    evidence = candidate.cleanup_rejected_created(persisted)

    assert candidate.observer_pid == 5678
    assert candidate.recovery_token == "00000000-0000-4000-8000-000000005678"
    assert evidence["release_result_sha256"].startswith("sha256:")
    assert runner.deleted == [(manifest["metadata"]["name"], ROOT_UID)]


def test_ambiguous_create_recovery_failure_keeps_exact_name_reads_until_bound(
    tmp_path,
) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    candidate.process_alive = lambda _: False
    candidate.launch_recovery_observer = lambda *_: (_ for _ in ()).throw(
        release.SupervisorError("synthetic recovery launch failure")
    )

    class Timer:
        value = 0.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    timer = Timer()
    candidate.monotonic = timer.monotonic
    candidate.sleep = timer.sleep
    runner.fail_gets = 2

    with pytest.raises(release.SupervisorError, match="possible resource leak"):
        candidate.reconcile_ambiguous_create()

    assert len(runner.get_names) > 2
    assert set(runner.get_names) == {manifest["metadata"]["name"]}
    assert runner.deleted == []


def test_created_identity_mismatch_never_derives_cleanup_authority_or_deletes(tmp_path) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner, "mismatched_identity")
    operation = tmp_path / "operation"

    with pytest.raises(JobsError, match="cleanup is uncertain"):
        submit(tmp_path, runner, kube, guard(operation, runner))

    assert runner.deleted == []
    assert not (operation / "DRIFT_CLEANUP_BOUND.json").exists()
    assert not (operation / "DRIFT_CLEANUP_AUTHORIZATION.json").exists()


def test_uncertain_create_without_exact_root_never_retries_or_discovers_prefixes(tmp_path) -> None:
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner, "raise_without_store")
    operation = tmp_path / "operation"
    with pytest.raises(JobsError, match="synthetic create uncertainty"):
        submit(tmp_path, runner, kube, guard(operation, runner))
    assert [call[0] for call in kube.calls].count("create") == 1
    assert not (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()
    assert len(set(runner.get_names)) == 1


def test_dead_observer_immediately_before_post_fails_closed(tmp_path) -> None:
    states = iter([True, True, True, False])
    runner = ExactBackendRunner()
    kube = ExactKubectl(runner)
    with pytest.raises(release.SupervisorError, match="died immediately before create"):
        submit(
            tmp_path,
            runner,
            kube,
            guard(tmp_path / "operation", runner, alive=lambda _: next(states)),
        )
    assert [call[0] for call in kube.calls].count("create") == 0


def test_hard_crash_after_post_is_recovered_from_exact_persisted_name(tmp_path) -> None:
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    kube = ExactKubectl(runner, "crash_after_store")
    operation = tmp_path / "operation"
    with pytest.raises(SystemExit, match="launcher death"):
        submit(tmp_path, runner, kube, guard(operation, runner))

    recovered = release.FourNodeReleaseSupervisor(
        operation,
        backend,
        clock=Clock(NOW),
        process_alive=lambda _: True,
    )
    binding = recovered.reconcile_and_authorize_exact()
    assert binding is not None and binding["rayjob_uid"] == ROOT_UID
    assert recovered.authorization is not None
    assert [call[0] for call in kube.calls].count("create") == 1


def test_hard_crash_after_uid_binding_derives_write_once_authorization(tmp_path) -> None:
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    kube = ExactKubectl(runner)
    operation = tmp_path / "operation"
    release_guard = guard(operation, runner)

    def crash_before_authorization(root: dict) -> dict:
        release_guard._armed().bind_created(root)
        raise SystemExit("synthetic launcher death before authorization")

    release_guard.accept_created = crash_before_authorization
    with pytest.raises(SystemExit, match="before authorization"):
        submit(
            tmp_path,
            runner,
            kube,
            release_guard,
        )
    assert (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()

    recovered = release.FourNodeReleaseSupervisor(
        operation,
        backend,
        clock=Clock(NOW),
        process_alive=lambda _: True,
    )
    recovered.reconcile_and_authorize_exact()
    first = (operation / "RELEASE_AUTHORIZATION.json").read_bytes()
    recovered.authorize_bound_uid()
    assert (operation / "RELEASE_AUTHORIZATION.json").read_bytes() == first


def test_default_detached_launcher_waits_for_private_candidate_bound_readiness(tmp_path) -> None:
    plan, request, manifest = packet()
    operation = tmp_path / "operation"
    seen: dict = {}

    class Process:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

    def popen(command: list[str], **kwargs) -> Process:
        seen["command"] = command
        seen["kwargs"] = kwargs
        receipt = release_k8s._ready_receipt(
            operation,
            plan,
            request,
            manifest,
            pid=Process.pid,
            now=NOW,
        )
        release._write_once(release_k8s._ready_path(operation), receipt)
        return Process()

    receipt = release_k8s.launch_detached_release_observer(
        operation,
        plan,
        request,
        manifest,
        popen=popen,
        monotonic=lambda: 0,
        sleep=lambda _: None,
    )

    assert seen["command"][:4] == [
        release_k8s.sys.executable,
        "-m",
        "training.qwen38_262k_release_kubernetes",
        "observe",
    ]
    assert seen["command"][-2:] == [manifest["metadata"]["name"], RUN_ID]
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["close_fds"] is True
    assert receipt["pid"] == Process.pid
    assert receipt["operation_dir"] == str(operation.resolve())
    assert release_k8s._ready_path(operation).stat().st_mode & 0o777 == 0o600


def test_default_detached_launcher_fails_if_observer_dies_before_readiness(tmp_path) -> None:
    plan, request, manifest = packet()

    class DeadProcess:
        pid = 4321

        @staticmethod
        def poll() -> int:
            return 1

    with pytest.raises(release.SupervisorError, match="exited before readiness"):
        release_k8s.launch_detached_release_observer(
            tmp_path / "operation",
            plan,
            request,
            manifest,
            popen=lambda *args, **kwargs: DeadProcess(),
            monotonic=lambda: 0,
            sleep=lambda _: None,
        )


def test_process_liveness_reaps_a_completed_detached_observer() -> None:
    class CompletedProcess:
        @staticmethod
        def poll() -> int:
            return 1

    release_k8s._OBSERVER_PROCESSES[4321] = CompletedProcess()

    assert release_k8s._process_alive(4321) is False
    assert 4321 not in release_k8s._OBSERVER_PROCESSES


def test_direct_guard_requires_shared_sfs_before_starting_observer(tmp_path) -> None:
    plan, request, manifest = packet()
    runner = ExactBackendRunner()
    launched: list[Path] = []
    candidate = ExactCandidateDirectCreateGuard(
        tmp_path / "operation",
        ProductionReleaseKubernetesBackend(runner=runner),
        launch_observer=lambda operation, *_: launched.append(operation),
        started_visible=lambda: False,
    )

    with pytest.raises(release.SupervisorError, match="direct shared-SFS visibility"):
        candidate.arm(plan, request, manifest)

    assert launched == []
    assert runner.calls == []


def test_failed_observer_takeover_reports_possible_leak_without_false_confirmation(
    tmp_path, monkeypatch
) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"

    class Timer:
        value = 0.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, _: float) -> None:
            self.value += release.DELETE_CONFIRM_SECONDS + 1

    timer = Timer()
    candidate = guard(operation, runner)
    candidate.monotonic = timer.monotonic
    candidate.sleep = timer.sleep
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    created = ExactKubectl(runner).create_once(manifest)
    supervisor = candidate._armed()
    delete_calls: list[bool] = []
    original_journal = supervisor.journal_bound_root_release_after_observer_failure

    def journal_only() -> None:
        delete_calls.append(True)
        original_journal()

    monkeypatch.setattr(
        supervisor,
        "journal_bound_root_release_after_observer_failure",
        journal_only,
    )
    monkeypatch.setattr(
        supervisor,
        "fresh_release_observation",
        lambda: (False, {"reason": "complete_UID_chain_was_not_observed"}),
    )
    candidate.process_alive = lambda _: False

    with pytest.raises(release.SupervisorError, match="possible resource leak"):
        candidate.accept_created(created)

    assert delete_calls == [True]
    uncertainty = json.loads((operation / "GUARD_RELEASE_UNCERTAINTY.json").read_text())
    assert uncertainty["status"] == "release_uncertain_after_observer_handoff_failure"
    assert uncertainty["rayjob_uid"] == ROOT_UID
    assert uncertainty["fresh_final_relist"] == {"reason": "complete_UID_chain_was_not_observed"}
    uncertainty["status"] = "released"
    uncertainty["sha256"] = "sha256:" + digest(
        {key: value for key, value in uncertainty.items() if key != "sha256"}
    )
    (operation / "GUARD_RELEASE_UNCERTAINTY.json").write_text(json.dumps(uncertainty))
    with pytest.raises(release.SupervisorError, match="receipt is invalid"):
        candidate._persist_guard_release_uncertainty(
            {"reason": "complete_UID_chain_was_not_observed"}, "synthetic"
        )


def test_post_create_binding_failure_retries_recovery_with_read_only_exact_gets(
    tmp_path, monkeypatch
) -> None:
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    created = ExactKubectl(runner).create_once(manifest)
    candidate.process_alive = lambda _: False

    attempts: list[bool] = []

    def fail_recovery(*_) -> dict:
        attempts.append(True)
        raise release.SupervisorError("synthetic recovery launch failure")

    candidate.launch_recovery_observer = fail_recovery

    class Timer:
        value = 0.0

        def monotonic(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    timer = Timer()
    candidate.monotonic = timer.monotonic
    candidate.sleep = timer.sleep
    monkeypatch.setattr(
        candidate._armed(),
        "bind_created",
        lambda _: (_ for _ in ()).throw(OSError("synthetic local binding failure")),
    )

    with pytest.raises(release.SupervisorError, match="possible resource leak"):
        candidate.accept_created(created)

    assert attempts
    assert len(runner.get_names) > 1
    assert set(runner.get_names) == {manifest["metadata"]["name"]}
    assert runner.deleted == []
    assert (operation / "CREATE_INTENT.json").exists()
    assert not (operation / "BOUND.json").exists()


def test_dead_initial_observer_hands_off_to_sealed_same_intent_recovery(
    tmp_path, monkeypatch
) -> None:
    class Process:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    initial = Process()
    recovery = Process()
    monkeypatch.setitem(release_k8s._OBSERVER_PROCESSES, 1234, initial)
    monkeypatch.setitem(release_k8s._OBSERVER_PROCESSES, 5678, recovery)
    runner = ExactBackendRunner()
    operation = tmp_path / "operation"
    candidate = guard(operation, runner, alive=release_k8s._process_alive)
    plan, request, manifest = packet()
    candidate.arm(plan, request, manifest)
    candidate.prepare_immediately_before_post()
    created = ExactKubectl(runner).create_once(manifest)
    initial.returncode = 1

    evidence = candidate.accept_created(created)

    assert 1234 not in release_k8s._OBSERVER_PROCESSES
    assert candidate.observer_pid == 5678
    assert candidate.recovery_token == "00000000-0000-4000-8000-000000005678"
    assert evidence["release_takeover_sha256"].startswith("sha256:")
    assert release_k8s._takeover_path(operation, candidate.recovery_token).exists()


def test_detached_observer_retries_exact_get_after_transient_prebinding_failure(
    tmp_path, monkeypatch
) -> None:
    plan, request, manifest = packet()
    operation = tmp_path / "operation"
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    armed = False

    def parent_progress(_: float) -> None:
        nonlocal armed
        if armed:
            return
        ready = release_k8s._read_ready_receipt(release_k8s._ready_path(operation))
        ready_at = datetime.fromisoformat(ready["ready_at"].replace("Z", "+00:00"))
        supervisor = release.FourNodeReleaseSupervisor.arm(
            operation,
            backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=ready["pid"],
            observer_receipt_sha256=ready["sha256"],
            clock=Clock(ready_at),
            process_alive=lambda _: True,
        )
        supervisor.write_create_intent()
        runner.root = deepcopy(manifest)
        runner.root["metadata"].update(
            {
                "uid": ROOT_UID,
                "creationTimestamp": stamp(ready_at),
                "resourceVersion": "1",
            }
        )
        runner.fail_gets = 1
        armed = True

    monkeypatch.setattr(
        release_k8s,
        "ProductionReleaseKubernetesBackend",
        lambda: backend,
    )
    monkeypatch.setattr(release_k8s.time, "sleep", parent_progress)
    monkeypatch.setattr(
        release_k8s.FourNodeReleaseSupervisor,
        "run",
        lambda self: {
            "status": "synthetic-supervised",
            "uid": self.binding["rayjob_uid"],
        },
    )

    result = release_k8s.run_detached_release_observer(
        operation,
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(manifest),
        manifest["metadata"]["name"],
        RUN_ID,
    )

    assert result == {"status": "synthetic-supervised", "uid": ROOT_UID}
    assert runner.fail_gets == 0
    assert (operation / "BOUND.json").exists()
    assert (operation / "RELEASE_AUTHORIZATION.json").exists()
    assert (operation / "OBSERVER_TAKEOVER.json").exists()


def test_detached_observer_never_accepts_drifted_root_and_uses_cleanup_only_path(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(release, "POLL_SECONDS", 0)
    plan, request, manifest = packet()
    operation = tmp_path / "operation"
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    armed = False

    def parent_progress(_: float) -> None:
        nonlocal armed
        if armed:
            return
        ready = release_k8s._read_ready_receipt(release_k8s._ready_path(operation))
        ready_at = datetime.fromisoformat(ready["ready_at"].replace("Z", "+00:00"))
        supervisor = release.FourNodeReleaseSupervisor.arm(
            operation,
            backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=ready["pid"],
            observer_receipt_sha256=ready["sha256"],
            clock=Clock(ready_at),
            process_alive=lambda _: True,
        )
        supervisor.write_create_intent()
        runner.root = deepcopy(manifest)
        runner.root["metadata"].update(
            {
                "uid": ROOT_UID,
                "creationTimestamp": stamp(ready_at),
                "resourceVersion": "1",
            }
        )
        runner.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
        armed = True

    monkeypatch.setattr(release_k8s, "ProductionReleaseKubernetesBackend", lambda: backend)
    monkeypatch.setattr(release_k8s.time, "sleep", parent_progress)

    result = release_k8s.run_detached_release_observer(
        operation,
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(manifest),
        manifest["metadata"]["name"],
        RUN_ID,
    )

    assert result["accepted"] is False
    assert runner.deleted == [(manifest["metadata"]["name"], ROOT_UID)]
    assert not (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()


def test_detached_observer_recovers_launcher_cleanup_without_repeating_delete(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(release, "POLL_SECONDS", 0)
    plan, request, manifest = packet()
    operation = tmp_path / "operation"
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    cleaned = False

    def parent_progress(_: float) -> None:
        nonlocal cleaned
        if cleaned:
            return
        ready = release_k8s._read_ready_receipt(release_k8s._ready_path(operation))
        ready_at = datetime.fromisoformat(ready["ready_at"].replace("Z", "+00:00"))
        supervisor = release.FourNodeReleaseSupervisor.arm(
            operation,
            backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=ready["pid"],
            observer_receipt_sha256=ready["sha256"],
            clock=Clock(ready_at),
            process_alive=lambda _: True,
        )
        supervisor.write_create_intent()
        runner.root = deepcopy(manifest)
        runner.root["metadata"].update(
            {
                "uid": ROOT_UID,
                "creationTimestamp": stamp(ready_at),
                "resourceVersion": "1",
            }
        )
        runner.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
        supervisor.cleanup_rejected_exact_root(deepcopy(runner.root))
        runner.fail_gets = 1
        cleaned = True

    monkeypatch.setattr(release_k8s, "ProductionReleaseKubernetesBackend", lambda: backend)
    monkeypatch.setattr(release_k8s.time, "sleep", parent_progress)

    result = release_k8s.run_detached_release_observer(
        operation,
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(manifest),
        manifest["metadata"]["name"],
        RUN_ID,
    )

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
    assert runner.deleted == [(manifest["metadata"]["name"], ROOT_UID)]
    assert not (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()


@pytest.mark.parametrize("race_stage", ["before_catch_get", "inside_cleanup"])
def test_detached_observer_joins_concurrent_direct_cleanup_races(
    tmp_path, monkeypatch, race_stage
) -> None:
    monkeypatch.setattr(release, "POLL_SECONDS", 0)
    plan, request, manifest = packet()
    operation = tmp_path / "operation"
    runner = ExactBackendRunner()
    backend = ProductionReleaseKubernetesBackend(runner=runner)
    armed = False

    def parent_progress(_: float) -> None:
        nonlocal armed
        if armed:
            return
        ready = release_k8s._read_ready_receipt(release_k8s._ready_path(operation))
        ready_at = datetime.fromisoformat(ready["ready_at"].replace("Z", "+00:00"))
        supervisor = release.FourNodeReleaseSupervisor.arm(
            operation,
            backend,
            plan=plan,
            request=request,
            manifest=manifest,
            observer_pid=ready["pid"],
            observer_receipt_sha256=ready["sha256"],
            clock=Clock(ready_at),
            process_alive=lambda _: True,
        )
        supervisor.write_create_intent()
        runner.root = deepcopy(manifest)
        runner.root["metadata"].update(
            {
                "uid": ROOT_UID,
                "creationTimestamp": stamp(ready_at),
                "resourceVersion": "1",
            }
        )
        runner.root["metadata"]["annotations"]["unreviewed.example/behavior"] = "on"
        armed = True

    original_reconcile = release.FourNodeReleaseSupervisor.reconcile_and_authorize_exact
    original_cleanup = release.FourNodeReleaseSupervisor.cleanup_rejected_exact_root
    raced = False

    if race_stage == "before_catch_get":

        def reconcile(self):
            nonlocal raced
            if not raced:
                raced = True
                original_cleanup(self, deepcopy(runner.root))
                raise release.SupervisorError("synthetic direct cleanup won the race")
            return original_reconcile(self)

        monkeypatch.setattr(
            release.FourNodeReleaseSupervisor,
            "reconcile_and_authorize_exact",
            reconcile,
        )
    else:

        def cleanup(self, root):
            nonlocal raced
            result = original_cleanup(self, root)
            if not raced:
                raced = True
                raise release.SupervisorError("synthetic root vanished during cleanup join")
            return result

        monkeypatch.setattr(
            release.FourNodeReleaseSupervisor,
            "cleanup_rejected_exact_root",
            cleanup,
        )

    monkeypatch.setattr(release_k8s, "ProductionReleaseKubernetesBackend", lambda: backend)
    monkeypatch.setattr(release_k8s.time, "sleep", parent_progress)

    result = release_k8s.run_detached_release_observer(
        operation,
        "sha256:" + digest(plan),
        "sha256:" + digest(request),
        "sha256:" + digest(manifest),
        manifest["metadata"]["name"],
        RUN_ID,
    )

    assert raced is True
    assert result["status"] == "cleanup_only_full_release_confirmed"
    assert result["release_confirmed"] is True
    assert runner.deleted == [(manifest["metadata"]["name"], ROOT_UID)]
    assert not (operation / "BOUND.json").exists()
    assert not (operation / "RELEASE_AUTHORIZATION.json").exists()


def test_observer_entrypoint_passes_only_the_exact_journaled_identity(
    monkeypatch, tmp_path
) -> None:
    values: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        release_k8s,
        "run_detached_release_observer",
        lambda *args, **kwargs: values.append((args, kwargs)),
    )
    argv = [
        "module",
        "observe",
        str(tmp_path / "operation"),
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
        "candidate-name",
        RUN_ID,
    ]
    assert release_k8s._main(argv) == 0
    assert values == [
        (
            (
                Path(argv[2]),
                argv[3],
                argv[4],
                argv[5],
                argv[6],
                argv[7],
            ),
            {"recovery": False, "recovery_token": ""},
        )
    ]
    token = "00000000-0000-4000-8000-000000009999"
    recovery_argv = [*argv]
    recovery_argv[1] = "recover"
    recovery_argv.append(token)
    assert release_k8s._main(recovery_argv) == 0
    assert values[-1] == (
        (
            Path(argv[2]),
            argv[3],
            argv[4],
            argv[5],
            argv[6],
            argv[7],
        ),
        {"recovery": True, "recovery_token": token},
    )


def test_started_receipt_reader_treats_missing_as_unknown_and_rejects_symlink(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(release_k8s, "RUN_DIR", str(tmp_path))
    assert release_k8s._started_receipt() is None
    started_path = tmp_path / "STARTED.json"
    started_path.write_text('{"plan_sha256":')
    assert release_k8s._started_receipt() is None
    started_path.unlink()
    target = tmp_path / "actual.json"
    target.write_text("{}")
    started_path.symlink_to(target)
    with pytest.raises(release.SupervisorError, match="not a regular file"):
        release_k8s._started_receipt()


def test_concurrent_binding_collision_accepts_only_the_same_exact_uid(
    tmp_path, monkeypatch
) -> None:
    runner = ExactBackendRunner()
    release_guard = guard(tmp_path / "operation", runner)
    plan, request, manifest = packet()
    kube = ExactKubectl(runner)
    release_guard.arm(plan, request, manifest)
    release_guard.prepare_immediately_before_post()
    created = kube.create_once(manifest)
    original = release._write_once
    injected = False

    def racing_write(path: Path, value: object) -> None:
        nonlocal injected
        if path.name == "BOUND.json" and not injected:
            injected = True
            original(path, value)
            raise FileExistsError(path)
        original(path, value)

    monkeypatch.setattr(release, "_write_once", racing_write)
    binding = release_guard._armed().bind_created(created)
    assert binding["rayjob_uid"] == ROOT_UID


def test_authorization_revalidates_full_binding_surface(tmp_path) -> None:
    runner = ExactBackendRunner()
    release_guard = guard(tmp_path / "operation", runner)
    plan, request, manifest = packet()
    release_guard.arm(plan, request, manifest)
    release_guard.prepare_immediately_before_post()
    created = ExactKubectl(runner).create_once(manifest)
    binding = release_guard._armed().bind_created(created)
    forged = {**binding, "rayjob_name": "another-root"}
    forged["sha256"] = "sha256:" + digest(
        {key: value for key, value in forged.items() if key != "sha256"}
    )
    with pytest.raises(release.SupervisorError, match="differs from release delegation"):
        release.persist_release_authorization(
            release_guard.operation_dir,
            forged,
            authorized_at=NOW,
        )
