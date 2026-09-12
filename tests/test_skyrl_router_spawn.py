"""CPU-only process-boundary tests; no model, Ray actor or task execution."""

import hashlib
import importlib.util
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from training import skyrl_training as train


def _synthetic_router_target(result, increment=1):
    result.value += increment


def _native_signature_synthetic_target(router_args, log_file):
    assert log_file is None
    router_args.synthetic_probe.value += 1


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_real_child_boundary_is_picklable_and_restores_parent_environment(monkeypatch, method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip("process context unavailable")
    native = multiprocessing.get_context(method)
    probe, result = native.RawValue("b", 0), native.RawValue("i", 0)
    monkeypatch.setenv("SYNTHETIC_ROUTER_SECRET", "synthetic-only")
    names = train._credential_names(os.environ)
    boundary = train._RouterMultiprocessing(native, _synthetic_router_target, names, probe, method)
    process = boundary.Process(
        target=_synthetic_router_target,
        args=(result,),
        kwargs={"increment": 3},
        daemon=True,
        name="synthetic-router",
    )
    try:
        with train._scrubbed_process_environment(names):
            process.start()
        process.join(timeout=30)
        assert not process.is_alive() and process.exitcode == 0
        assert probe.value == 1 and result.value == 3
        assert os.environ["SYNTHETIC_ROUTER_SECRET"] == "synthetic-only"
        assert process.daemon and process.name == "synthetic-router"
        assert train._router_child_entry.__module__ == "training.skyrl_training"
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_real_child_rejects_nonempty_credentials_before_target(monkeypatch, method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip("process context unavailable")
    native = multiprocessing.get_context(method)
    probe, result = native.RawValue("b", 0), native.RawValue("i", 0)
    # Deliberately not in the parent-supplied list: a newly introduced secret is
    # independently caught by the child's name-only inventory.
    monkeypatch.setenv("SYNTHETIC_ROUTER_SECRET", "synthetic-only")
    process = train._RouterMultiprocessing(
        native, _synthetic_router_target, (), probe, method
    ).Process(target=_synthetic_router_target, args=(result,))
    try:
        process.start()
        process.join(timeout=30)
        assert not process.is_alive() and process.exitcode != 0
        assert probe.value == -1 and result.value == 0
        assert os.environ["SYNTHETIC_ROUTER_SECRET"] == "synthetic-only"
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


def test_process_constructor_rejects_changed_native_target_and_context():
    native = NS(
        get_start_method=lambda: "spawn",
        Process=lambda **kwargs: pytest.fail("unexpected child launch"),
    )
    boundary = train._RouterMultiprocessing(
        native, _synthetic_router_target, (), NS(value=0), "spawn"
    )
    with pytest.raises(train._DiagnosticContractRejected, match="target changed"):
        boundary.Process(target=lambda: None)
    native.get_start_method = lambda: "forkserver"
    with pytest.raises(train._DiagnosticContractRejected, match="context changed"):
        boundary.Process(target=_synthetic_router_target)


def test_child_context_mismatch_never_executes_target(monkeypatch):
    probe, result = NS(value=0), NS(value=0)
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda: "forkserver")
    with pytest.raises(RuntimeError, match="context changed"):
        train._router_child_entry(_synthetic_router_target, (), probe, "spawn", (result,), {})
    assert probe.value == -2 and result.value == 0


@pytest.mark.skipif(importlib.util.find_spec("skyrl") is None, reason="pinned SkyRL image only")
def test_pinned_native_router_start_uses_spawn_with_synthetic_child_only(monkeypatch):
    # CPU qualification of the actual native start/shutdown methods and our
    # wrapper, NOT Rust-router health, vLLM engines, actors, model or GPU work.
    train.native_source()
    router_module = importlib.import_module(
        "skyrl.backends.skyrl_train.inference_servers.vllm_router"
    )
    assert hashlib.sha256(Path(router_module.__file__).read_bytes()).hexdigest() == (
        "f39f7b00125d670773cf1a461f2bc2e13e32dc40a84936da4d96b2db8e315a34"
    )
    assert multiprocessing.get_start_method() == "spawn"
    import ray

    monkeypatch.setattr(ray, "init", lambda **kwargs: pytest.fail("CPU probe initialized Ray"))
    monkeypatch.setattr(router_module, "get_node_ip", lambda: "127.0.0.1")
    # The synthetic replacement is itself a global importable function, exactly
    # like the unchanged production target; no closure enters spawn's pickle.
    monkeypatch.setattr(
        router_module, "_run_router_with_logging", _native_signature_synthetic_target
    )
    result = multiprocessing.RawValue("i", 0)

    def child_finished(self, url):
        self._process.join(timeout=30)
        assert not self._process.is_alive() and self._process.exitcode == 0
        assert result.value == 1

    monkeypatch.setattr(router_module.VLLMRouter, "_wait_until_healthy", child_finished)
    monkeypatch.setenv("SYNTHETIC_ROUTER_SECRET", "synthetic-only")
    original_target = router_module._run_router_with_logging
    original_multiprocessing = router_module.multiprocessing
    ownership = train._DiagnosticOwnership()
    router = None
    try:
        with train._instrument_engine_ownership(ownership, {}, train._credential_names(os.environ)):
            router = router_module.VLLMRouter(
                NS(
                    port=0,
                    prometheus_port=0,
                    vllm_pd_disaggregation=False,
                    pd_disaggregation=False,
                    worker_urls=[],
                    synthetic_probe=result,
                )
            )
            assert router.start().startswith("http://127.0.0.1:")
            assert router_module._run_router_with_logging is original_target
        assert ownership.routers == [router]
        assert ownership.router_start_attempts == ownership.router_credential_probes_passed == 1
        assert ownership.router_credential_probe_failures == 0
        assert ownership.router_multiprocessing_start_method == "spawn"
        assert not any((ownership.actors, ownership.groups, ownership.placement_groups))
        assert router_module.multiprocessing is original_multiprocessing
        assert os.environ["SYNTHETIC_ROUTER_SECRET"] == "synthetic-only"
    finally:
        if router is not None:
            router.shutdown()
            assert train._router_released(router, True)
            if router._process is not None:
                router._process.close()


@pytest.mark.parametrize(
    "error_kind", ["known-contract", "arbitrary-value", "import", "cleanup-defect", "output-defect"]
)
def test_pre_ray_contract_rejection_is_truthful_and_other_defects_still_fail(
    tmp_path, monkeypatch, error_kind
):
    # This deliberately small plan exercises only diagnostic lifecycle; the
    # ordinary suite separately validates the immutable plan/input contracts.
    root = tmp_path / "diagnostic"
    root.mkdir()
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_UID", "1000")
    monkeypatch.setenv("CYBER_EXPECTED_RUNTIME_GID", "100")
    monkeypatch.setattr(train.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train.os, "getegid", lambda: 100)
    monkeypatch.setattr(train, "_require_engine_start_qualified_image", lambda _: None)
    plan = {"output_root": str(root), "arguments": {}}
    monkeypatch.setattr(
        train.skyrl,
        "SkyRLConfig",
        lambda **kwargs: NS(engine_start_timeout_seconds=10, engine_cleanup_timeout_seconds=10),
    )
    monkeypatch.setattr(train, "job_request", lambda _: None)
    monkeypatch.setattr(train, "check_inputs", lambda _: None)
    monkeypatch.setattr(train, "native_source", lambda: {})
    monkeypatch.setattr(train.skyrl, "diagnostic_native_config", lambda _: NS())
    monkeypatch.setattr(
        train,
        "_diagnostic_shape",
        lambda *args: {"diagnostic_workers": 2, "diagnostic_gpus_per_worker": 4},
    )
    events = []

    def output(*args, started):
        if started and error_kind == "output-defect":
            raise ValueError("synthetic unexpected output")
        events.append("output-final" if started else "output-initial")
        return {"runtime_files_unchanged": True, "unexpected_output_artifacts": 0}

    monkeypatch.setattr(train, "_diagnostic_output_evidence", output)
    log = root / "infra.log"
    log.write_text("")
    monkeypatch.setattr(train, "_ray_environment", lambda *args, **kwargs: ({}, log, ()))
    monkeypatch.setattr(train, "_infra_log_evidence", lambda _: {"sha256": "0" * 64})
    monkeypatch.setitem(
        sys.modules,
        "ray",
        NS(init=lambda **kwargs: pytest.fail("Ray initialized before rejection")),
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=lambda _: pytest.fail("built engine arguments")),
    )

    def reject(*args):
        if error_kind in {"known-contract", "cleanup-defect", "output-defect"}:
            raise train._DiagnosticContractRejected("unsupported native router process context")
        if error_kind == "arbitrary-value":
            raise ValueError("synthetic unexpected bug")
        raise ImportError("synthetic broken installation")

    monkeypatch.setattr(train, "_instrument_engine_ownership", reject)

    def cleanup(setup, ray, ownership, timeout):
        assert setup is None and ownership.job_id is None
        assert not any(
            (ownership.actors, ownership.groups, ownership.routers, ownership.placement_groups)
        )
        events.append("cleanup")
        if error_kind == "cleanup-defect":
            raise ValueError("synthetic unproven cleanup")
        return {
            "cleanup_proven": True,
            "active_owned_actors": 0,
            "active_owned_placement_groups": 0,
        }

    monkeypatch.setattr(train, "_cleanup_engine_diagnostic", cleanup)
    if error_kind != "known-contract":
        with pytest.raises((ValueError, ImportError, ExceptionGroup)):
            train.engine_diagnostic(plan)
        assert not (root / "ENGINE_DIAGNOSTIC.json").exists()
        return

    result = train.engine_diagnostic(plan)
    train.sealed(result, train.ENGINE_DIAGNOSTIC_SCHEMA)
    assert events == ["output-initial", "cleanup", "output-final"]
    assert result["status"] == "diagnostic_contract_rejected"
    assert result["startup_phase"] == "ownership_instrumentation"
    assert result["engine_start_state"] == "none"
    assert result["cleanup"]["cleanup_proven"]
    assert result["diagnostic_completed"]
    assert not any(
        result[key]
        for key in (
            "ray_initialization_attempted",
            "engine_start_qualified",
            "training_qualified",
            "production_training_shape_qualified",
            "credential_environment_isolation_proven",
            "engine_started",
            "task_rows_read",
            "verifier_calls",
            "optimizer_steps",
            "rollouts",
            "checkpoints_created",
            "checkpoint_created",
            "wandb_initialized",
        )
    )
    assert not (root / "ACCEPTED.json").exists()
