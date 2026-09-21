"""Offline safety checks for the one-episode SkyRL collector diagnostic.

These tests deliberately exercise only plan construction and supplied-render
validation.  They never instantiate a Jobs client, preview, create a workload,
or contact a Fleet service.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train.jobs import API_URLS, Jobs, JobsError
from scripts import prepare_skyrl_collector_diagnostic as packet_builder
from scripts.audit_qwen38_skyrl_launch_readiness import (
    CANARY_RUN,
    NEXT_GATES,
    compile_prod8,
    load,
    prod8_metadata,
)
from training import rl_episode
from training import skyrl_collector_diagnostic as diagnostic


def _forbidden_network(*_args, **_kwargs):
    raise AssertionError("offline collector-diagnostic test attempted network I/O")


@pytest.fixture
def diagnostic_inputs(tmp_path):
    """Write a locally compiled, current qualified one-node source plan."""
    run = load(CANARY_RUN)
    source, _ = compile_prod8(run, prod8_metadata(run, load(NEXT_GATES)))
    assert source["execution"]["image"] == diagnostic.IMAGE
    root = tmp_path
    source_path = root / "source-plan.json"
    source_path.write_text(json.dumps(source, sort_keys=True))
    config = {
        "schema": diagnostic.CONFIG_SCHEMA,
        "source_plan": source_path.name,
        "name": "collector-diagnostic",
        "output_root": "/mnt/sfs/jobs/collector-diagnostic",
        "selection": {
            "split": "train",
            "row_index": 0,
            "line_sha256": "sha256:" + "1" * 64,
            "source_config_sha256": "sha256:" + "2" * 64,
        },
    }
    return NS(root=root, source=source, config=config)


def _container(request: dict) -> dict:
    return {
        "image": request["image"],
        "resources": {
            "requests": {
                "cpu": request["resources"]["cpu_request"],
                "memory": request["resources"]["memory_request"],
                "nvidia.com/gpu": request["gpus_per_worker"],
            },
            "limits": {
                "cpu": request["resources"]["cpu_limit"],
                "memory": request["resources"]["memory_limit"],
                "nvidia.com/gpu": request["gpus_per_worker"],
            },
        },
        "env": [
            {"name": name, "value": value}
            for name, value in {**request["env"], "RUN_DIR": request["run_dir"]}.items()
        ],
        "envFrom": [
            {"secretRef": {"name": name, "optional": False}} for name in request["secrets"]
        ],
        "securityContext": {
            "privileged": False,
            "runAsUser": 1000,
            "runAsGroup": 100,
            "runAsNonRoot": True,
        },
    }


def _preview(request: dict) -> dict:
    template = {
        "spec": {
            "priorityClassName": request["priority_class"],
            "securityContext": {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True},
            "containers": [_container(request)],
        }
    }
    rendered = {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {
                "fleet.ai/run-dir": request["run_dir"],
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "backoffLimit": 0,
            "entrypoint": request["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": template},
                "workerGroupSpecs": [],
            },
        },
    }
    return {"manifest_yaml": yaml.safe_dump(rendered, sort_keys=True), "warnings": []}


def _rendered(preview: dict) -> dict:
    return yaml.safe_load(preview["manifest_yaml"])


def _with_rendered(rendered: dict) -> dict:
    return {"manifest_yaml": yaml.safe_dump(rendered, sort_keys=True), "warnings": []}


def test_compile_request_and_offline_packets_are_network_free(diagnostic_inputs, monkeypatch):
    """The offline preparation path cannot silently turn into a preview/create."""
    monkeypatch.setattr(diagnostic.httpx, "Client", _forbidden_network)
    monkeypatch.setattr(diagnostic.httpx, "AsyncClient", _forbidden_network)
    monkeypatch.setattr(Jobs, "preview", _forbidden_network)
    monkeypatch.setattr(Jobs, "raw_preview", _forbidden_network)
    monkeypatch.setattr(Jobs, "submit_once", _forbidden_network)

    plan = diagnostic.compile_diagnostic(
        diagnostic_inputs.config, relative_to=diagnostic_inputs.root
    )
    request = diagnostic.job_request(plan)
    local = diagnostic.offline_preview(plan, request)
    observer = diagnostic.release_observer_contract(plan, request)

    assert plan["execution"] == {
        "cluster_target": "dev",
        "jobs_api_base_url": API_URLS["dev"],
        "image": diagnostic.IMAGE,
        "priority": "c1",
        "resources": plan["execution"]["resources"],
        "maximum_seconds": diagnostic.MAXIMUM_SECONDS,
    }
    assert request["workers"] * request["gpus_per_worker"] == 8
    assert request["failureAlerts"] is False
    assert request["secrets"] == ["fleet-api"]
    assert "wandb-api" not in request["secrets"]
    assert {key: value for key, value in request["env"].items() if key.startswith("WANDB_")} == {
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_CONSOLE": "off",
    }
    assert local["status"] == "locally_rendered_not_server_previewed"
    assert local["external_reads"] == local["external_mutations"] == 0
    assert local["preview_authorized"] is local["create_authorized"] is False
    assert observer["arm_before_jobs_post"] is True
    assert observer["maximum_seconds"] == diagnostic.MAXIMUM_SECONDS
    assert observer["pre_post_prefix_guard_schema"] == "cyber_jobs_api_prefix_guard_armed_v1"
    assert observer["post_post_exact_binding_schema"] == "cyber_jobs_api_exact_rayjob_binding_v1"
    assert observer["exact_observer_schema"] == "cyber_jobs_api_exact_uid_observer_result_v1"
    assert observer["release_acceptance"]["raw_delete_requires_separate_creator_contract"] is True


def test_compile_rejects_unattestable_image_module_hash_input(diagnostic_inputs):
    config = {
        **diagnostic_inputs.config,
        "image_native_sources": {"setup": "sha256:" + "a" * 64},
    }
    with pytest.raises(JobsError, match="config fields changed"):
        diagnostic.compile_diagnostic(config, relative_to=diagnostic_inputs.root)


def test_post_start_setup_identity_excludes_private_module_path(monkeypatch):
    module = NS(__name__=diagnostic.SETUP_MODULE, __file__="/private/image/setup.py")
    monkeypatch.setattr(diagnostic.importlib.metadata, "version", lambda package: "0.4.1")

    assert diagnostic._setup_identity(module) == {
        "module": diagnostic.SETUP_MODULE,
        "package": "skyrl",
        "version": "0.4.1",
    }


def test_success_receipt_requires_sanitized_post_start_setup_identity():
    receipt = {
        "schema": diagnostic.RECEIPT_SCHEMA,
        "status": "collection_completed_no_training",
        "diagnostic_plan_sha256": "sha256:" + "1" * 64,
        "legacy_runtime_sha256": "sha256:" + "2" * 64,
        "diagnostic_runtime_sha256": "sha256:" + "3" * 64,
        "image_digest": "sha256:" + "4" * 64,
        "source_row_sha256": "sha256:" + "5" * 64,
        "sampling_sha256": "sha256:" + "6" * 64,
        "episode_attempted": True,
        "collection_completed": True,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "wandb_events": 0,
        "fleet_instance_release_confirmed": True,
        "engine_cleanup_confirmed": True,
        "external_job_cleanup_required": True,
        "leaf_reason_count": 0,
        "native_setup": {
            "module": diagnostic.SETUP_MODULE,
            "package": "skyrl",
            "version": "0.4.1",
        },
    }
    assert diagnostic._validate_receipt(receipt) == receipt

    del receipt["native_setup"]
    with pytest.raises(JobsError, match="native setup identity"):
        diagnostic._validate_receipt(receipt)

    receipt["native_setup"] = {
        "module": diagnostic.SETUP_MODULE,
        "package": "skyrl",
        "version": "0.4.1",
        "file": "/private/image/setup.py",
    }
    with pytest.raises(JobsError, match="native setup receipt changed"):
        diagnostic._validate_receipt(receipt)


def test_prepare_command_writes_only_a_local_nonlaunchable_packet(diagnostic_inputs, monkeypatch):
    monkeypatch.setattr(diagnostic.httpx, "Client", _forbidden_network)
    monkeypatch.setattr(diagnostic.httpx, "AsyncClient", _forbidden_network)
    config_path = diagnostic_inputs.root / "collector-config.json"
    config_path.write_text(json.dumps(diagnostic_inputs.config, sort_keys=True))
    output = diagnostic_inputs.root / "packet"

    packet = packet_builder.prepare(config_path, output)

    assert packet["status"] == "prepared_locally_not_server_previewed"
    assert packet["preview_authorized"] is packet["create_authorized"] is False
    assert packet["external_reads"] == packet["external_mutations"] == 0
    assert {path.name for path in output.iterdir()} == {
        "PLAN.json",
        "REQUEST.json",
        "OFFLINE_PREVIEW.json",
        "RELEASE_OBSERVER_CONTRACT.json",
        "PACKET.json",
    }
    with pytest.raises(JobsError):
        packet_builder.prepare(config_path, output)


def test_server_preview_requires_root_alert_opt_out_and_disables_wandb(
    diagnostic_inputs, monkeypatch
):
    # This test owns a sealed synthetic plan. Freeze the in-process runtime
    # snapshot before compiling it, so a concurrent source edit cannot obscure
    # endpoint-binding assertions with an unrelated byte-drift rejection.
    runtime = diagnostic._runtime_files()
    monkeypatch.setattr(diagnostic, "_runtime_files", lambda: runtime)
    plan = diagnostic.compile_diagnostic(
        diagnostic_inputs.config, relative_to=diagnostic_inputs.root
    )
    request = diagnostic.job_request(plan)
    preview = _preview(request)

    proof = diagnostic.validate_server_preview(plan, request, preview, api_base_url=API_URLS["dev"])
    assert proof["failure_alert_annotation"] == "off"
    assert proof["gpus"] == 8
    assert proof["runtime_user"] == {"uid": 1000, "gid": 100}
    assert proof["create_authorized"] is False

    def mutate_alert(rendered):
        rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "false"

    def mutate_root_alert_to_template_only(rendered):
        rendered["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["metadata"] = {
            "annotations": {"fleet.ai/failure-alerts": "off"}
        }

    def mutate_extra_wandb(rendered):
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
            "env"
        ].append({"name": "WANDB_PROJECT", "value": "leak"})

    def mutate_wandb_secret(rendered):
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
            "envFrom"
        ].append({"secretRef": {"name": "wandb-api", "optional": False}})

    def mutate_optional_fleet_secret(rendered):
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
            "envFrom"
        ][0]["secretRef"]["optional"] = True

    def mutate_duplicate_fleet_secret(rendered):
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
            "envFrom"
        ].append({"secretRef": {"name": "fleet-api", "optional": False}})

    def mutate_wandb_value(rendered):
        entries = rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
            "containers"
        ][0]["env"]
        next(item for item in entries if item["name"] == "WANDB_DISABLED")["value"] = "false"

    def mutate_wandb_init_container(rendered):
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
            "initContainers"
        ] = [{"name": "setup", "env": [{"name": "WANDB_PROJECT", "value": "leak"}]}]

    for mutate in (
        mutate_alert,
        mutate_root_alert_to_template_only,
        mutate_extra_wandb,
        mutate_wandb_secret,
        mutate_optional_fleet_secret,
        mutate_duplicate_fleet_secret,
        mutate_wandb_value,
        mutate_wandb_init_container,
    ):
        rendered = _rendered(preview)
        mutate(rendered)
        with pytest.raises(JobsError):
            diagnostic.validate_server_preview(
                plan, request, _with_rendered(rendered), api_base_url=API_URLS["dev"]
            )

    with pytest.raises(JobsError, match="request changed before server preview"):
        diagnostic.validate_server_preview(plan, request, preview, api_base_url=API_URLS["prod"])


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            rl_episode.InvalidEpisode("generation_incomplete_context_full"),
            "generation_incomplete_context_full",
        ),
        (rl_episode.InvalidEpisode("not_an_allowlisted_reason"), None),
        (ValueError("unclassified"), None),
        (
            ExceptionGroup(
                "mixed",
                [rl_episode.InvalidEpisode("generation_incomplete"), ValueError("other")],
            ),
            None,
        ),
        (
            ExceptionGroup(
                "duplicate",
                [
                    rl_episode.InvalidEpisode("generation_incomplete"),
                    rl_episode.InvalidEpisode("generation_incomplete"),
                ],
            ),
            None,
        ),
    ],
)
def test_leaf_reason_is_single_and_unambiguous(error, expected):
    assert diagnostic._allowlisted_leaf_reason(error) == expected


def test_leaf_reason_rejects_generic_wrapper_around_safe_cause():
    """A generic wrapper makes the root cause ambiguous rather than safe evidence."""
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = rl_episode.InvalidEpisode("generation_incomplete")
    assert diagnostic._allowlisted_leaf_reason(wrapper) is None


class _Remote:
    def __init__(self, calls: list[str], name: str, *, fail: bool = False):
        self.calls, self.name, self.fail = calls, name, fail

    def remote(self):
        self.calls.append("remote:" + self.name)
        if self.fail:
            raise RuntimeError("synthetic remote failure")
        return self.name


class _Actor:
    def __init__(self, calls: list[str], name: str, *, fail: bool = False):
        self.shutdown = _Remote(calls, name, fail=fail)
        self.name = name


class _Group:
    def __init__(self, actors):
        self._actors = actors

    def get_actors(self):
        return tuple(self._actors)


def test_engine_cleanup_runs_all_bounded_teardown_steps():
    calls: list[object] = []
    actor = _Actor(calls, "actor")
    engine_setup = NS(
        router=NS(shutdown=lambda: calls.append("router")), server_groups=(_Group([actor]),)
    )
    ray = NS(
        get=lambda refs, timeout: calls.append(("get", tuple(refs), timeout)),
        kill=lambda item, no_restart: calls.append(("kill", item.name, no_restart)),
    )

    assert diagnostic._stop_setup(engine_setup, ray) is True
    assert calls == ["router", "remote:actor", ("get", ("actor",), 30), ("kill", "actor", True)]


def test_engine_cleanup_never_claims_success_after_teardown_error():
    calls: list[object] = []

    def broken_router():
        calls.append("router")
        raise RuntimeError("synthetic router failure")

    actor = _Actor(calls, "actor", fail=True)
    engine_setup = NS(router=NS(shutdown=broken_router), server_groups=(_Group([actor]),))
    ray = NS(
        get=lambda *_args, **_kwargs: pytest.fail("no references should survive a remote failure"),
        kill=lambda item, no_restart: calls.append(("kill", item.name, no_restart)),
    )

    assert diagnostic._stop_setup(engine_setup, ray) is False
    assert calls == ["router", "remote:actor", ("kill", "actor", True)]


def test_ray_shutdown_failure_is_not_a_cleanup_success():
    class _Ray:
        def shutdown(self):
            raise RuntimeError("synthetic shutdown failure")

    assert diagnostic._shutdown_ray(_Ray()) is False
