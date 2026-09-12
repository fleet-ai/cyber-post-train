"""Offline launch/lifecycle tests; synthetic task data and no paid requests."""

import base64
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_rl_data import build, setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import rl_data, sft_runtime
from training import skyrl_training as train

ROOT = Path(__file__).resolve().parents[1]


def _without_diagnostic_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def _materialize_diagnostic_runtime(plan):
    root = Path(plan["output_root"])
    runtime = root / ".runtime"
    runtime.mkdir()
    for name, content in train._diagnostic_runtime_files(plan).items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


class _Ref:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error


class _RemoteMethod:
    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        try:
            return _Ref(self.method(*args, **kwargs))
        except BaseException as exc:
            return _Ref(error=exc)


class _Actor:
    def __init__(self, value):
        self.value, self.dead = value, False

    def __getattr__(self, name):
        if name == "shutdown":
            return _RemoteMethod(lambda: None)
        return _RemoteMethod(getattr(self.value, name))


class _RemoteClass:
    def __init__(self, value, *, options_kwargs=None, factory=None):
        self.value = value
        self.options_kwargs = {} if options_kwargs is None else options_kwargs
        self.factory = factory

    def options(self, **kwargs):
        return type(self)(
            self.value,
            options_kwargs={**self.options_kwargs, **kwargs},
            factory=self.factory,
        )

    def remote(self, *args, **kwargs):
        if self.factory is not None:
            return self.factory(self.options_kwargs, *args, **kwargs)
        return _Actor(self.value(*args, **kwargs))


class _JobID:
    """Match the pinned Ray 2.56 identity/display split."""

    def __init__(self, value="a1"):
        self.value = value

    def hex(self):
        return self.value

    def __str__(self):
        return f"JobID({self.value})"


def _fake_ray(calls, *, credential_count=0, nodes=None):
    actors, actor_options = [], []

    class NodeAffinitySchedulingStrategy:
        def __init__(self, *, node_id, soft):
            self.node_id, self.soft = node_id, soft

    def remote(value):
        def create(options, *args, **kwargs):
            options = dict(options)
            actor_options.append(options)
            strategy = options.get("scheduling_strategy")

            def inspect(names):
                count = (
                    credential_count(options, names)
                    if callable(credential_count)
                    else credential_count
                )
                return {"node_id": strategy.node_id, "nonempty": count}

            actor = (
                _Actor(NS(inspect=inspect))
                if value is train._CredentialProbe
                else _Actor(value(*args, **kwargs))
            )
            actors.append(actor)
            return actor

        return _RemoteClass(value, factory=create)

    def get(refs, *, timeout=None, **kwargs):
        calls.append(("ray-get", timeout))

        def resolve(ref):
            if ref.error is not None:
                raise ref.error
            return ref.value

        return [resolve(ref) for ref in refs] if isinstance(refs, list) else resolve(refs)

    if nodes is None:
        nodes = [
            {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
            {"Alive": True, "NodeID": "bb", "Resources": {"GPU": 4.0}},
            {"Alive": True, "NodeID": "dd", "Resources": {"CPU": 16.0}},
            {"Alive": False, "NodeID": "cc", "Resources": {"GPU": 4.0}},
        ]
    ray = NS(
        init=lambda **kwargs: calls.append(("ray", kwargs)),
        get=get,
        get_runtime_context=lambda: NS(get_job_id=lambda: _JobID()),
        nodes=lambda: nodes() if callable(nodes) else nodes,
        remote=remote,
        kill=lambda actor, **kwargs: (setattr(actor, "dead", True), calls.append("ray-kill")),
        shutdown=lambda: calls.append("ray-shutdown"),
        util=NS(
            scheduling_strategies=NS(NodeAffinitySchedulingStrategy=NodeAffinitySchedulingStrategy)
        ),
        actor_options=actor_options,
    )
    return ray, actors


def _exact_engine_config():
    return NS(
        num_engines=2,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )


def _install_engine_modules(monkeypatch, calls, create):
    class Group:
        def __init__(self):
            self.actor = _Actor(NS())

        def get_actors(self):
            return [self.actor]

        def _create_actor_class(self, *args, **kwargs):
            module = sys.modules["skyrl.backends.skyrl_train.inference_servers.server_group"]
            runtime_env = module.build_engine_runtime_env()
            return _RemoteClass(train._CredentialProbe).options(runtime_env=runtime_env)

    class Router:
        def start(self):
            module = sys.modules["skyrl.backends.skyrl_train.inference_servers.vllm_router"]
            module.multiprocessing.Process(
                target=module._run_router_with_logging, args=(), daemon=True, name="vllm-router"
            ).start()
            return "http://router"

        def shutdown(self):
            calls.append("router")

    placement = NS(remove_placement_group=lambda group: calls.append("remove-pg"))
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement)
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.server_group",
        NS(
            ServerGroup=Group,
            build_engine_runtime_env=lambda **kwargs: None,
            placement_group=lambda *args, **kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.vllm_router",
        NS(
            VLLMRouter=Router,
            _run_router_with_logging=lambda: calls.append("router-target"),
            multiprocessing=NS(
                get_start_method=multiprocessing.get_start_method,
                RawValue=lambda kind, value: NS(value=value),
                Process=lambda *, target, args=(), **kwargs: NS(start=lambda: target(*args)),
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.setup",
        NS(create_inference_servers=create, ray_placement_group=lambda *args, **kwargs: object()),
    )


@pytest.fixture
def prepared(data_setup):  # noqa: F811
    lock = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
    data_setup.lock.update(json.loads(lock.read_bytes()))
    build(data_setup)
    config = {
        "backend": "skyrl",
        "name": "synthetic-rl",
        "output_root": "/mnt/sfs/jobs/synthetic-rl",
        "model": {
            "lock": str(lock),
            "weights": str(ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
            "root": data_setup.config["model_root"],
        },
        "data": {"manifest": "out/manifest.json", "root": "/mnt/sfs/data/synthetic-rl"},
        "recipe": {"groups": 1, "samples_per_prompt": 8, "steps": 2, "lr": 1e-6},
        "wandb": {"entity": "synthetic", "project": "synthetic", "run_id": "synthetic-rl"},
    }
    plan = train.compile_rl(config, relative_to=data_setup.tmp)
    return NS(config=config, plan=plan, state=data_setup)


def test_prepare_cli_and_portable_runtime_are_offline(prepared, monkeypatch):
    from training.sft import IMAGE

    tmp, plan = prepared.state.tmp, prepared.plan
    request = train.job_request(plan)
    assert IMAGE == train.IMAGE == request["image"]
    assert request["priority_class"] == "c1" and not request["requeueIfPreempted"]
    assert (request["workers"], request["gpus_per_worker"]) == (1, 8)
    assert request["secrets"] == ["fleet-api", "wandb-api"]
    assert "API_KEY" not in str(request["env"])
    assert plan["arguments"]["steps"] == plan["native_overrides"]["trainer.max_training_steps"] == 2
    source = tmp / "launch.json"
    source.write_text(json.dumps(prepared.config))
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("offline command used network"))
    # A staged SkyRL runtime does not contain the unrelated Miles launcher.
    monkeypatch.setitem(sys.modules, "training.miles_training", None)
    result = CliRunner().invoke(cli.app, ["rl", str(source), "--output", str(tmp / "prepared")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["submitted"] is False
    assert cli._prepared(tmp / "prepared") == (plan, request)
    assert (
        CliRunner()
        .invoke(cli.app, ["rl", str(source), "--output", str(tmp / "prepared")])
        .exit_code
        == 2
    )
    # The immutable GPU bundle must be importable without the source checkout.
    bundle = tmp / "bundle"
    for name, content in train._runtime().items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for directory in ("training", "cyber_post_train", "evals", "evals/fleet"):
        (bundle / directory / "__init__.py").write_text("")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from training import skyrl_training as s; from training.rl_data import selection; "
            "assert s.MODULE == 'training.skyrl_training'",
        ],
        cwd=tmp,
        env={**os.environ, "PYTHONPATH": str(bundle)},
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_engine_diagnostic_request_is_exact_image_no_secret_and_no_training(prepared):
    # The retained failure used Mamba cache mode ``none`` and its exact vLLM
    # EngineConfig accepted 8192.  The unrelated 2096/align-mode workaround is
    # not an admitted correction for this model/runtime.
    overrides = prepared.plan["native_overrides"]
    assert overrides["generator.inference_engine.enable_prefix_caching"] is False
    assert "generator.inference_engine.max_num_batched_tokens" not in overrides
    request = train.engine_diagnostic_request(prepared.plan)
    assert request["image"] == train.IMAGE
    assert request["secrets"] == []
    assert (request["workers"], request["gpus_per_worker"]) == (2, 4)
    assert request["workers"] * request["gpus_per_worker"] == 8
    assert train._diagnostic_shape(prepared.plan) == {
        "diagnostic_workers": 2,
        "diagnostic_gpus_per_worker": 4,
        "diagnostic_total_gpus": 8,
        "num_engines": 2,
        "tensor_parallel_size": 4,
    }
    assert not request["requeueIfPreempted"]
    assert request["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in request["env"]
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in train.job_request(prepared.plan)["env"]
    assert prepared.plan["arguments"]["engine_start_timeout_seconds"] == 1800
    assert prepared.plan["arguments"]["engine_cleanup_timeout_seconds"] == 300
    assert "WANDB" not in json.dumps(request["env"])
    keys = sorted(k for k in request["env"] if k.startswith("CYBER_RUNTIME_BUNDLE"))
    payload = json.loads(
        gzip.decompress(base64.b64decode("".join(request["env"][k] for k in keys)))
    )
    assert payload["argv"][-1] == "--engine-diagnostic"


def test_engine_diagnostic_native_config_disables_tracking_and_gpu_monitor(prepared, monkeypatch):
    from training import skyrl_episode

    result = NS(trainer=NS(logger="console", enable_ray_gpu_monitor=False))
    seen = {}

    def module(name, sha):
        assert train.skyrl.NATIVE_SOURCES[name] == sha
        if name.endswith(".config"):
            return NS(
                SkyRLTrainConfig=NS(
                    from_cli_overrides=lambda values: (seen.update(values=values), result)[1]
                )
            )
        return NS(validate_cfg=lambda cfg: seen.update(validated=cfg))

    monkeypatch.setattr(skyrl_episode, "_module", module)
    config = train.skyrl.SkyRLConfig(**prepared.plan["arguments"])
    assert train.skyrl.diagnostic_native_config(config) is result
    assert seen["values"]["trainer.logger"] == "console"
    assert seen["values"]["trainer.enable_ray_gpu_monitor"] is False
    assert seen["validated"] is result
    assert prepared.plan["native_overrides"]["trainer.logger"] == "wandb"
    result.trainer.logger = "wandb"
    with pytest.raises(ValueError, match="telemetry controls changed"):
        train.skyrl.diagnostic_native_config(config)


def test_engine_diagnostic_shape_rejects_training_topology_substitution(prepared):
    prepared.plan["arguments"]["nodes"] = 2
    prepared.plan["native_overrides"]["generator.inference_engine.num_engines"] = 4

    with pytest.raises(ValueError, match="exact one-node"):
        train._diagnostic_shape(prepared.plan)


def test_engine_diagnostic_preflight_never_reads_task_rows(prepared, monkeypatch):
    _without_diagnostic_wandb(monkeypatch)
    monkeypatch.setattr(train, "_require_engine_start_qualified_image", lambda plan: None)
    monkeypatch.setattr(train.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train.os, "getegid", lambda: 100)
    monkeypatch.setattr(train, "check_inputs", lambda plan: None)
    monkeypatch.setattr(train, "native_source", lambda: {})
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=lambda cfg: "exact-vllm-args"),
    )
    monkeypatch.setattr(
        train.skyrl, "native_config", lambda config: pytest.fail("used training config")
    )
    cfg = NS(generator=NS(inference_engine=_exact_engine_config()))
    monkeypatch.setattr(train.skyrl, "diagnostic_native_config", lambda config: cfg)
    monkeypatch.setattr(train, "check_artifacts", lambda plan: pytest.fail("read task rows"))

    receipt = train.engine_diagnostic_preflight(prepared.plan)

    assert receipt["schema"] == "cyber_skyrl_engine_diagnostic_cpu_preflight_v1"
    assert receipt["request_sha256"] == digest(train.engine_diagnostic_request(prepared.plan))
    assert receipt["engine_cli_args_checked"]
    assert {
        key: receipt[key]
        for key in (
            "diagnostic_workers",
            "diagnostic_gpus_per_worker",
            "diagnostic_total_gpus",
            "num_engines",
            "tensor_parallel_size",
        )
    } == train._diagnostic_shape(prepared.plan, cfg)
    assert receipt["task_rows_read"] == 0
    assert "vllm_v1_multiprocessing_disabled" not in receipt
    assert not any(
        receipt[key]
        for key in ("rollouts", "verifier_calls", "optimizer_updates", "checkpoints", "wandb")
    )


def test_engine_diagnostic_preflight_rejects_native_cli_arg_drift(prepared, monkeypatch):
    monkeypatch.setattr(train, "_require_engine_start_qualified_image", lambda plan: None)
    monkeypatch.setattr(train.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train.os, "getegid", lambda: 100)
    monkeypatch.setattr(train, "check_inputs", lambda plan: None)
    monkeypatch.setattr(train, "native_source", lambda: {})
    cfg = NS(generator=NS(inference_engine=_exact_engine_config()))
    monkeypatch.setattr(train.skyrl, "diagnostic_native_config", lambda config: cfg)
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(
            build_vllm_cli_args=lambda cfg: (_ for _ in ()).throw(
                ValueError("synthetic native arg drift")
            )
        ),
    )

    with pytest.raises(ValueError, match="native arg drift"):
        train.engine_diagnostic_preflight(prepared.plan)


def test_engine_diagnostic_scrubs_ambient_credentials_but_keeps_them_for_driver(
    prepared, monkeypatch
):
    plan = prepared.plan
    root = prepared.state.tmp / "credential-rejection"
    root.mkdir()
    plan["output_root"] = str(root)
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-platform-plumbing")
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-forbidden-tracking-key")
    native = {
        "skyrl.train.utils.utils": NS(
            prepare_runtime_environment=lambda _: {
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
                "SAFE_NATIVE_SETTING": "yes",
            }
        )
    }

    env, _, scrubbed = train._ray_environment(plan, object(), native, diagnostic=True)

    assert os.environ["FLEET_API_KEY"] == "synthetic-platform-plumbing"
    assert os.environ["WANDB_API_KEY"] == "synthetic-forbidden-tracking-key"
    assert env["FLEET_API_KEY"] == env["WANDB_API_KEY"] == ""
    assert env["SAFE_NATIVE_SETTING"] == "yes"
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in env
    assert {"FLEET_API_KEY", "WANDB_API_KEY"}.issubset(scrubbed)


@pytest.mark.parametrize(
    "key",
    [
        "FLEET_API_KEY",
        "WANDB_API_KEY",
        "SERVICE_TOKEN",
        "DATABASE_PASSWORD",
        "CLIENT_SECRET",
        "AWS_ACCESS_KEY_ID",
        "WORKER_CREDENTIAL",
        "lowercase_api_key",
    ],
)
def test_engine_diagnostic_scrubs_credential_like_ray_environment(prepared, monkeypatch, key):
    _without_diagnostic_wandb(monkeypatch)
    root = prepared.state.tmp / "credential-ray-environment"
    root.mkdir()
    prepared.plan["output_root"] = str(root)
    native = {
        "skyrl.train.utils.utils": NS(
            prepare_runtime_environment=lambda _: {
                key: "" if key == "WORKER_CREDENTIAL" else "synthetic-private-value"
            }
        )
    }

    env, _, scrubbed = train._ray_environment(prepared.plan, object(), native, diagnostic=True)

    assert env[key] == "" and key in scrubbed


def test_ray_credential_probe_rejects_and_tracks_nonempty_worker_secret():
    ray, actors = _fake_ray([], credential_count=1)
    ownership = train._DiagnosticOwnership()
    environment = {"FLEET_API_KEY": "", "SAFE_NATIVE_SETTING": "yes"}

    with pytest.raises(ValueError, match="retained a scrubbed credential"):
        train._probe_worker_credentials(
            ray,
            ownership,
            environment,
            ("FLEET_API_KEY",),
            expected_nodes=2,
            expected_gpus_per_node=4,
        )

    assert ownership.actors == actors and len(actors) == 2
    assert ownership.ray_gpu_nodes_discovered == ownership.ray_gpu_nodes_probed == 2
    assert ownership.ray_actor_environment_probes_passed == 0
    assert ownership.ray_actor_environment_probe_failures == 2
    assert ownership.ray_actor_nonempty_scrubbed_credentials == 2
    assert {option["scheduling_strategy"].node_id for option in ray.actor_options} == {
        "aa",
        "bb",
    }
    assert all(
        option["runtime_env"]["env_vars"] == {**environment, "WANDB_API_KEY": ""}
        for option in ray.actor_options
    )


def test_synthetic_ray_jobs_precedence_cannot_bypass_explicit_actor_scrub():
    job_environment = {
        "FLEET_API_KEY": "synthetic-job-secret",
        "SAFE_JOB_SETTING": "job",
    }

    def inherited_nonempty(options, names):
        # Model Ray Jobs precedence: the driver-level ``ray.init`` environment is
        # ignored, while an actor-level runtime environment overrides the job.
        effective = {**job_environment, **options["runtime_env"]["env_vars"]}
        return sum(bool(effective.get(name)) for name in names)

    calls = []
    ray, actors = _fake_ray(calls, credential_count=inherited_nonempty)
    ownership = train._DiagnosticOwnership()
    environment = {
        "FLEET_API_KEY": "synthetic-must-be-blanked",
        "SAFE_NATIVE_SETTING": "native",
    }
    ray.init(runtime_env={"env_vars": {"FLEET_API_KEY": ""}})

    assert (
        train._probe_worker_credentials(
            ray,
            ownership,
            environment,
            ("FLEET_API_KEY", "WANDB_API_KEY"),
            expected_nodes=2,
            expected_gpus_per_node=4,
        )
        == 0
    )
    assert ownership.actors == actors and len(actors) == 2
    assert ownership.ray_gpu_nodes_discovered == ownership.ray_gpu_nodes_probed == 2
    assert ownership.ray_actor_environment_probes_passed == 2
    assert ownership.ray_actor_environment_probe_failures == 0
    assert ownership.ray_actor_nonempty_scrubbed_credentials == 0
    for option in ray.actor_options:
        assert option["num_cpus"] == option["num_gpus"] == option["max_restarts"] == 0
        assert not option["scheduling_strategy"].soft
        assert option["runtime_env"]["env_vars"] == {
            "FLEET_API_KEY": "",
            "SAFE_NATIVE_SETTING": "native",
            "WANDB_API_KEY": "",
        }


def test_pinned_ray_actor_environment_overrides_inherited_synthetic_credential():
    """Opt-in exact-image proof; uses logical GPU resources, never real GPUs."""
    if os.environ.get("CYBER_SKYRL_PINNED_RAY_PROOF") != train.IMAGE:
        pytest.skip("requires an explicit run inside the exact pinned SkyRL image")

    ray = pytest.importorskip("ray")
    assert ray.__version__ == "2.56.0"
    from ray.cluster_utils import Cluster

    key = "SYNTHETIC_DIAGNOSTIC_API_KEY"
    missing = object()
    previous = os.environ.get(key, missing)
    os.environ[key] = "synthetic-inherited-value"
    cluster = Cluster(initialize_head=False)
    baseline_actors = []
    ownership = train._DiagnosticOwnership()

    @ray.remote
    class InheritedCredentialProbe:
        def inspect(self):
            import os

            import ray

            return {
                "node_id": str(ray.get_runtime_context().get_node_id()),
                "nonempty": int(bool(os.environ.get(key))),
            }

    try:
        cluster.add_node(num_cpus=1, num_gpus=4, include_dashboard=False)
        cluster.add_node(num_cpus=1, num_gpus=4, include_dashboard=False)
        ray.init(address=cluster.address, log_to_driver=False)
        node_ids = train._live_gpu_node_ids(ray, 2, 4)
        for node_id in node_ids:
            baseline_actors.append(
                InheritedCredentialProbe.options(
                    num_cpus=0,
                    num_gpus=0,
                    max_restarts=0,
                    scheduling_strategy=train._node_affinity_strategy(ray, node_id),
                ).remote()
            )
        inherited = ray.get([actor.inspect.remote() for actor in baseline_actors])
        assert tuple(sorted(item["node_id"].lower() for item in inherited)) == node_ids
        assert [item["nonempty"] for item in inherited] == [1, 1]

        assert (
            train._probe_worker_credentials(
                ray,
                ownership,
                {key: "", "SAFE_NATIVE_SETTING": "native"},
                train._credential_names({key: ""}),
                expected_nodes=2,
                expected_gpus_per_node=4,
            )
            == 0
        )
        assert ownership.ray_gpu_nodes_discovered == ownership.ray_gpu_nodes_probed == 2
        assert ownership.ray_actor_environment_probes_passed == 2
        assert ownership.ray_actor_environment_probe_failures == 0
        assert ownership.ray_actor_nonempty_scrubbed_credentials == 0
    finally:
        if ray.is_initialized():
            for actor in baseline_actors + ownership.actors:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
        cluster.shutdown()
        if previous is missing:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@pytest.mark.parametrize(
    "nodes",
    [
        [{"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}}],
        [
            {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
            {"Alive": True, "NodeID": "bb", "Resources": {"GPU": 8.0}},
        ],
        [
            {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
            {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
        ],
    ],
)
def test_ray_credential_probe_rejects_inexact_gpu_topology_before_actor_creation(nodes):
    ray, actors = _fake_ray([], nodes=nodes)
    ownership = train._DiagnosticOwnership()

    with pytest.raises(ValueError, match="GPU node topology differs"):
        train._probe_worker_credentials(
            ray,
            ownership,
            {"FLEET_API_KEY": ""},
            ("FLEET_API_KEY",),
            expected_nodes=2,
            expected_gpus_per_node=4,
        )

    assert actors == ownership.actors == []


def test_ray_credential_probe_rejects_gpu_node_churn_after_tracking_actors():
    snapshots = iter(
        [
            [
                {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
                {"Alive": True, "NodeID": "bb", "Resources": {"GPU": 4.0}},
            ],
            [
                {"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}},
                {"Alive": True, "NodeID": "cc", "Resources": {"GPU": 4.0}},
            ],
        ]
    )
    ray, actors = _fake_ray([], nodes=lambda: next(snapshots))
    ownership = train._DiagnosticOwnership()

    with pytest.raises(ValueError, match="topology changed"):
        train._probe_worker_credentials(
            ray,
            ownership,
            {"FLEET_API_KEY": ""},
            ("FLEET_API_KEY",),
            expected_nodes=2,
            expected_gpus_per_node=4,
        )

    assert ownership.actors == actors and len(actors) == 2
    assert ownership.ray_gpu_nodes_discovered == ownership.ray_gpu_nodes_probed == 2
    assert ownership.ray_actor_environment_probes_passed == 2
    assert ownership.ray_actor_environment_probe_failures == 0


def test_private_log_evidence_streams_and_detects_cross_chunk_signature(tmp_path):
    signature = b"Engine core initialization failed. See root cause above."
    path = tmp_path / "infra.log"
    payload = b"x" * (1024 * 1024 - len(signature) // 2) + signature + b"tail"
    path.write_bytes(payload)

    evidence = train._infra_log_evidence(path)

    assert evidence == {
        "path": "private-native-logs/infra.log",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "engine_failure_signature_present": True,
    }


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "backend",
        "checkpoint",
        "priority",
        "resource",
        "groups",
        "start_deadline",
        "cleanup_deadline",
        "model",
        "file",
        "data_name",
        "digest",
        "runtime",
        "native",
        "output",
    ],
)
def test_invalid_launch_stops_before_gpu_import(prepared, fault):
    config, tmp = prepared.config, prepared.state.tmp
    if fault == "unknown":
        config["extra"] = True
    elif fault == "backend":
        config["backend"] = "miles"
    elif fault == "checkpoint":
        config["checkpoint"] = {"root": "unreviewed"}
    elif fault == "priority":
        config["cluster"] = {"priority": "c0"}
    elif fault == "resource":
        config["cluster"] = {"resources": {"memory_request": "2Gi"}}
    elif fault == "groups":
        config["recipe"]["groups"] = 2
    elif fault == "start_deadline":
        config["recipe"]["engine_start_timeout_seconds"] = 3601
    elif fault == "cleanup_deadline":
        config["recipe"]["engine_cleanup_timeout_seconds"] = 0
    elif fault in {"model", "file", "data_name", "digest"}:
        path = tmp / "out/manifest.json"
        value = json.loads(path.read_bytes())
        if fault == "model":
            value["tokenizer"]["repo"] = "zai-org/GLM-5.3"
        elif fault == "file":
            value["files"]["train"]["path"] = "../train.jsonl"
        elif fault == "data_name":
            value["name"] = "other-run"
        value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
        if fault == "digest":
            value["sha256"] = "wrong"
        path.write_text(json.dumps(value))
    else:
        plan = prepared.plan
        if fault == "runtime":
            plan["runtime_sha256"] = "a" * 64
        elif fault == "native":
            plan["native_overrides"]["trainer.max_training_steps"] = 999
        else:
            plan["output_root"] = "/mnt/sfs/jobs/other"
        with pytest.raises(ValueError):
            train.job_request(plan)
        return
    with pytest.raises(ValueError):
        train.compile_rl(config, relative_to=tmp)


@pytest.fixture
def artifacts(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp / "out"
    monkeypatch.setattr(train, "check_inputs", lambda _: None)
    plan["arguments"]["data_manifest"] = str(root / "manifest.json")
    for split in ("train", "dev"):
        plan["arguments"][split + "_data"] = str(root / (split + ".jsonl"))
    return plan, root


def test_artifact_and_dataset_boundary_preserve_rows(artifacts, monkeypatch):
    plan, root = artifacts
    before = {p: p.read_bytes() for p in root.iterdir()}
    rows = train.check_artifacts(plan)
    assert {k: len(v) for k, v in rows.items()} == {"train": 1, "dev": 1}
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    for split in rows:
        assert len(train.dataset(plan, tokenizer, split, rows[split])) == 1
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "bytes",
        "split_digest",
        "count",
        "prompt",
        "split",
        "run",
        "env",
        "template",
        "duplicate",
        "task",
    ],
)
def test_artifact_drift_is_fatal(artifacts, fault):
    plan, root = artifacts
    if fault == "manifest":
        (root / "manifest.json").write_text("{}")
    elif fault == "bytes":
        (root / "train.jsonl").write_text("changed")
    elif fault == "split_digest":
        split = json.loads((root / "split.json").read_bytes())
        split["sha256"] = "changed"
        (root / "split.json").write_text(json.dumps(split))
    elif fault == "count":
        plan["data"]["files"]["train"]["rows"] = 2
        (root / "manifest.json").write_text(json.dumps(plan["data"]))
    else:
        path = root / "train.jsonl"
        row = json.loads(path.read_bytes())
        cfg = json.loads(row["cyber_config_json"])
        if fault == "prompt":
            row["prompt"][0]["content"] = "changed"
        elif fault == "split":
            row["split"] = "dev"
        elif fault == "run":
            cfg["run_id"] = "changed"
        elif fault == "env":
            row["env_class"] = "changed"
        elif fault == "template":
            cfg["model"]["runtime_chat_template_sha256"] = "changed"
        elif fault == "task":
            cfg["task"]["key"] = "changed"
        cfg["config_sha256"] = fleet.digest_without(cfg, "config_sha256")
        row["cyber_config_json"] = json.dumps(cfg)
        path.write_text((json.dumps(row) + "\n") * (2 if fault == "duplicate" else 1))
        item = plan["data"]["files"]["train"]
        item["sha256"] = "sha256:" + train._hash(path)
        item["rows"] = 2 if fault == "duplicate" else 1
        (root / "manifest.json").write_text(json.dumps(plan["data"]))
    with pytest.raises(ValueError):
        train.check_artifacts(plan)


@pytest.mark.parametrize("drift", ["drop", "prompt", "binding", "env"])
def test_native_dataset_cannot_filter_or_edit(artifacts, prepared, monkeypatch, drift):
    plan, _ = artifacts
    rows = train.check_artifacts(plan)
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    prepared.state.drift = drift
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    with pytest.raises(ValueError, match="silently"):
        train.dataset(plan, tokenizer, "train", rows["train"])


@pytest.mark.parametrize("fault", [None, "gpu", "output", "template"])
def test_cpu_preflight_dispatch_never_starts_ray(artifacts, monkeypatch, fault):
    monkeypatch.setattr(train, "_require_engine_start_qualified_image", lambda plan: None)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "getegid", lambda: 100)
    plan, root = artifacts
    plan["output_root"] = str(root / "new-output")
    if fault == "output":
        Path(plan["output_root"]).mkdir()
    tokenizer, _, Dataset, _ = rl_data._native_skyrl(None, None)
    if fault == "template":
        tokenizer.chat_template = "changed"
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: fault == "gpu")))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: tokenizer)),
    )
    monkeypatch.setattr(train, "job_request", lambda _: {"fixture": True})
    monkeypatch.setattr(train, "native_source", lambda: {})
    monkeypatch.setattr(train.skyrl, "native_config", lambda _: NS())
    monkeypatch.setattr(train, "_module", lambda *a: NS(PromptDataset=Dataset))
    if fault:
        with pytest.raises((ValueError, FileExistsError)):
            train.preflight(plan)
    else:
        proof = train.preflight(plan)
        assert proof["status"] == "passed" and proof["gpus"] == 0
        assert proof["native_parser_checked"] and not proof["rl_qualified"]
        assert proof["runtime_user"] == {"uid": 1000, "gid": 100}


@pytest.mark.parametrize("uid,gid", [(0, 0), (0, 100), (1000, 0)])
def test_preflight_rejects_privileged_or_different_file_access(monkeypatch, uid, gid):
    monkeypatch.setattr(os, "geteuid", lambda: uid)
    monkeypatch.setattr(os, "getegid", lambda: gid)
    monkeypatch.setattr(
        train, "job_request", lambda _: pytest.fail("read artifacts before user gate")
    )
    with pytest.raises(ValueError, match="1000:100"):
        train.preflight({})


def test_cpu_preflight_rejects_only_the_disqualified_qwen38_image(prepared, monkeypatch):
    monkeypatch.setattr(train.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train.os, "getegid", lambda: 100)
    with pytest.raises(ValueError, match="dev5/dev6"):
        train.preflight(prepared.plan)

    replacement = {
        **prepared.plan,
        "execution": {
            **prepared.plan["execution"],
            "image": "registry.invalid/skyrl@sha256:" + "1" * 64,
        },
    }
    train._require_engine_start_qualified_image(replacement)
    other_model = {
        **prepared.plan,
        "model": {**prepared.plan["model"], "repo": "Qwen/Qwen3.6-27B"},
    }
    train._require_engine_start_qualified_image(other_model)


@pytest.fixture
def completed(prepared):
    plan, root = prepared.plan, prepared.state.tmp / "result"
    plan["output_root"] = str(root)
    args = plan["arguments"]
    for phase, step in (("eval", 0), ("train", 1), ("eval", 1), ("train", 2), ("eval", 2)):
        directory = root / f"episodes/batches/{phase}-{step}"
        directory.mkdir(parents=True)
        value = {
            "schema": "cyber_skyrl_batch_v1",
            "phase": phase,
            "global_step": step,
            "data_sha256": plan["data"]["sha256"],
        }
        value["sha256"] = "sha256:" + digest(value)
        (directory / "COLLECTED.json").write_text(json.dumps(value))
    checkpoint = root / f"checkpoints/global_step_{args['steps']}"
    (checkpoint / "policy").mkdir(parents=True)
    for name in ("data.pt", "trainer_state.pt", "policy/synthetic.distcp"):
        (checkpoint / name).write_bytes(b"synthetic state, not a checkpoint qualification")
    (root / "checkpoints/latest_ckpt_global_step.txt").write_text(str(args["steps"]))
    return plan, root


@pytest.mark.parametrize(
    "fault", [None, "failed", "batch", "digest", "missing", "pointer", "sampler", "policy"]
)
def test_terminal_checks_do_not_fabricate_acceptance(completed, fault):
    plan, root = completed
    path = root / "episodes/batches/train-1"
    if fault == "failed":
        (path / "FAILED.json").write_text("{}")
    elif fault in {"batch", "digest"}:
        value = json.loads((path / "COLLECTED.json").read_bytes())
        value["global_step"] = 3
        if fault == "batch":
            value["sha256"] = "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
        (path / "COLLECTED.json").write_text(json.dumps(value))
    elif fault == "missing":
        (path / "COLLECTED.json").unlink()
        path.rmdir()
    elif fault == "pointer":
        (root / "checkpoints/latest_ckpt_global_step.txt").write_text("1")
    elif fault == "sampler":
        (root / "checkpoints/global_step_2/data.pt").unlink()
    elif fault == "policy":
        (root / "checkpoints/global_step_2/policy/synthetic.distcp").unlink()
    if fault:
        with pytest.raises(ValueError):
            train.native_result(plan)
    else:
        proof = train.native_result(plan)
        assert proof["checkpoint_global_step"] == 2 and proof["completed_batches"] == 5
        assert (
            not proof["optimizer_update_independently_verified"]
            and not proof["checkpoint_reload_verified"]
        )
    assert not (root / "ACCEPTED.json").exists()


def test_scalar_tracking_does_not_upload_private_exceptions(prepared, monkeypatch):
    plan = prepared.plan
    plan["output_root"] = str(prepared.state.tmp)
    calls = []
    monkeypatch.setenv("WANDB_API_KEY", "synthetic")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        NS(
            init=lambda **kw: (
                calls.append(kw),
                NS(id="synthetic-rl", entity="synthetic", project="synthetic"),
            )[1],
            log=lambda **kw: calls.append(kw),
            finish=lambda **kw: pytest.fail("premature finish"),
        ),
    )
    tracker = train.ScalarTracking(plan)
    tracker.log({"policy/loss": 0.5, "eval/reward": 0.0}, step=1, commit=True)
    tracker.log_exception(RuntimeError("private secret task text"), step=1)
    tracker.finish()
    assert len(calls) == 2 and "private secret" not in str(calls)
    assert os.environ["WANDB_CONSOLE"] == "off" and os.environ["WANDB_RESUME"] == "never"
    for value in (True, "text", {}, [0.5], float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tracker.log({"value": value}, 2)
    with pytest.raises(ValueError):
        tracker.log({"bad\nkey": 0.0}, 2)
    with pytest.raises(ValueError):
        tracker.log({}, -1)
    with pytest.raises(ValueError):
        tracker.log_samples_to_table("text", [], [], 1)
    sys.modules["wandb"].init = lambda **kw: NS(id="other", entity="synthetic", project="synthetic")
    with pytest.raises(ValueError, match="identity mismatch"):
        train.ScalarTracking(plan)


def test_native_sources_are_checked_without_inventing_a_driver(monkeypatch):
    calls = []
    monkeypatch.setattr(sft_runtime, "validate_runtime_sources", lambda: calls.append("runtime"))
    monkeypatch.setattr(train, "_module", lambda name, sha: (calls.append((name, sha)), name)[1])
    assert train.native_source() == {name: name for name in train.NATIVE}
    assert calls == ["runtime", *train.NATIVE.items()]


@pytest.mark.parametrize(
    "fault", [None, "setup", "native", "step", "checkpoint", "tracking", "no_tracking"]
)
def test_native_wrapper_keeps_native_loop_and_truthful_finalization(prepared, monkeypatch, fault):
    from training import skyrl_rollout

    plan, calls = prepared.plan, []
    result_rows = {"train": ["train"], "dev": ["dev"]}
    monkeypatch.setattr(train, "check_artifacts", lambda _: result_rows)
    monkeypatch.setattr(train.skyrl, "native_config", lambda _: "native-config")
    infra_log = prepared.state.tmp / "synthetic-infra.log"
    infra_log.write_text("")
    monkeypatch.setattr(train, "_prepare_infra_log", lambda _: infra_log)
    monkeypatch.setattr(
        train, "dataset", lambda p, t, split, rows: (calls.append((split, rows)), rows)[1]
    )
    monkeypatch.setattr(train, "ScalarTracking", lambda p: "scalar-tracker")
    monkeypatch.setattr(
        skyrl_rollout, "Generator", lambda *a, **kw: (calls.append((a, kw)), "generator")[1]
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        NS(init=lambda **kw: calls.append(("ray", kw)), shutdown=lambda: calls.append("shutdown")),
    )

    def finish(**kw):
        calls.append(("finish", kw))
        if fault == "tracking":
            raise RuntimeError("synthetic tracking failure")

    monkeypatch.setitem(
        sys.modules, "wandb", NS(run=None if fault == "no_tracking" else NS(), finish=finish)
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.utils.ppo_utils",
        NS(sync_registries=lambda: calls.append("sync")),
    )

    class Native:
        def __init__(self, cfg):
            assert cfg == "native-config"
            if fault == "setup":
                raise RuntimeError("synthetic setup failure")
            self.tokenizer = "native-tokenizer"
            assert self.get_train_dataset() == ["train"]
            assert self.get_eval_dataset() == ["dev"]
            assert self.get_generator(cfg, self.tokenizer, "engine") == "generator"
            assert self.get_tracker() == "scalar-tracker"
            assert self.get_trajectory_logger() is None
            self.trainer = NS(global_step=1 if fault == "step" else 2)

        def run(self):
            calls.append("native-loop")
            if fault == "native":
                raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(
        train,
        "native_source",
        lambda: {
            "skyrl.train.entrypoints.main_base": NS(BasePPOExp=Native),
            "skyrl.train.utils.utils": NS(
                prepare_runtime_environment=lambda cfg: {"PINNED": "yes"}
            ),
        },
    )

    def result(p):
        assert p == plan
        calls.append("checkpoint-check")
        if fault == "checkpoint":
            raise ValueError("synthetic missing checkpoint")

    monkeypatch.setattr(train, "native_result", result)
    if fault and fault != "no_tracking":
        with pytest.raises((RuntimeError, ValueError)):
            train._native(plan)
    else:
        train._native(plan)
    assert calls[-1] == "shutdown"
    if fault == "no_tracking":
        assert not any(isinstance(call, tuple) and call[0] == "finish" for call in calls)
    else:
        assert ("finish", {"exit_code": 0 if fault in (None, "tracking") else 1}) in calls
    assert calls[0][0] == "ray" and calls[0][1]["address"] == "auto"
    assert not calls[0][1]["log_to_driver"]
    assert calls[0][1]["runtime_env"]["env_vars"]["PINNED"] == "yes"
    assert calls[0][1]["runtime_env"]["env_vars"]["SKYRL_LOG_FILE"] == str(infra_log)


@pytest.mark.parametrize("fault", [None, "pre-engine", "create", "partial", "timeout", "router"])
def test_engine_diagnostic_is_bounded_private_clean_and_truthful(prepared, monkeypatch, fault):
    plan, calls = prepared.plan, []
    root = prepared.state.tmp / "engine-diagnostic"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setenv("FLEET_API_KEY", "outer-platform-key")
    monkeypatch.setattr(train, "job_request", lambda _: None)
    checked = []

    def check_inputs(value):
        assert value is plan and not (root / "ENGINE_DIAGNOSTIC_STARTED.json").exists()
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        checked.append(True)

    monkeypatch.setattr(train, "check_inputs", check_inputs)
    engine_config = _exact_engine_config()
    cfg = NS(
        generator=NS(inference_engine=engine_config),
        trainer=NS(log_path=str(root / "private-native-logs")),
    )
    monkeypatch.setattr(train.skyrl, "diagnostic_native_config", lambda _: cfg)
    monkeypatch.setattr(
        train,
        "native_source",
        lambda: {
            "skyrl.train.utils.utils": NS(
                prepare_runtime_environment=lambda _: {"FLEET_API_KEY": "must-be-scrubbed"}
            )
        },
    )
    ray, _ = _fake_ray(calls)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.utils.ppo_utils",
        NS(sync_registries=lambda: pytest.fail("diagnostic created registry actors")),
    )

    def build_vllm_cli_args(_):
        if fault == "pre-engine":
            raise RuntimeError("private engine argument failure")
        return "exact-vllm-args"

    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=build_vllm_cli_args),
    )

    def create(engine, args, log_path):
        assert (engine, args, log_path) == (
            engine_config,
            "exact-vllm-args",
            str(root / "private-native-logs"),
        )
        group_type = sys.modules[
            "skyrl.backends.skyrl_train.inference_servers.server_group"
        ].ServerGroup
        if fault in {"partial", "timeout"}:
            group_type()
        if fault == "create":
            raise PermissionError("private unrelated diagnostic")
        if fault == "partial":
            raise RuntimeError("private partial engine failure")
        if fault == "timeout":
            ray.get(_Ref(error=TimeoutError("synthetic bare wait")))
        (root / "private-native-logs/infra.log").write_text(
            "engine initialized without task rows\n"
        )
        groups = [group_type(), group_type()]
        router_type = sys.modules[
            "skyrl.backends.skyrl_train.inference_servers.vllm_router"
        ].VLLMRouter
        router = router_type()
        proxy_url = router.start()
        if fault == "router":
            raise RuntimeError("private post-probe router failure")
        return NS(
            router=router,
            server_groups=groups,
            server_urls=["http://engine-0", "http://engine-1"],
            proxy_url=proxy_url,
        )

    _install_engine_modules(monkeypatch, calls, create)
    monkeypatch.setattr(train, "_active_owned_resources", lambda *args: (0, 0))

    result = train.engine_diagnostic(plan)

    train.sealed(result, train.ENGINE_DIAGNOSTIC_SCHEMA)
    expected = {
        None: ("passed", "all", True),
        "pre-engine": ("pre_engine_rejected", "none", False),
        "create": ("engine_start_rejected", "none", False),
        "partial": ("engine_start_rejected", "partial_or_unknown", None),
        "timeout": ("engine_start_timeout", "partial_or_unknown", None),
        "router": ("engine_start_rejected", "partial_or_unknown", None),
    }[fault]
    assert (result["status"], result["engine_start_state"], result["engine_started"]) == expected
    assert checked == [True]
    assert result["cleanup"]["cleanup_proven"]
    assert result["cleanup"]["active_owned_actors"] == 0
    assert result["cleanup"]["active_owned_placement_groups"] == 0
    assert result["ray_gpu_nodes_expected"] == 2
    assert result["ray_gpu_nodes_discovered"] == 2
    assert result["ray_gpu_nodes_probed"] == 2
    assert result["ray_actor_environment_probes_passed"] == 2
    assert result["ray_actor_environment_probe_failures"] == 0
    assert result["ray_actor_nonempty_scrubbed_credentials"] == 0
    assert len(ray.actor_options) == 2
    assert {option["scheduling_strategy"].node_id for option in ray.actor_options} == {"aa", "bb"}
    assert all(
        option["runtime_env"]["env_vars"]["FLEET_API_KEY"] == ""
        and option["runtime_env"]["env_vars"]["WANDB_API_KEY"] == ""
        and "VLLM_ENABLE_V1_MULTIPROCESSING" not in option["runtime_env"]["env_vars"]
        and option["runtime_env"]["env_vars"]["SKYRL_LOG_FILE"]
        == str(root / "private-native-logs/infra.log")
        for option in ray.actor_options
    )
    assert result["credential_environment_isolation_proven"]
    assert result["ray_actor_environment_isolation_proven"]
    assert not result["service_account_token_isolation_proven"]
    assert not result["service_account_rbac_write_access_tested"]
    assert result["credential_variables_scrubbed"] >= 2
    assert result["registry_actors_created"] == 0
    assert result["router_child_credential_environment_isolation"] == (
        "proven" if fault in {None, "router"} else "not_started"
    )
    assert (
        result["router_start_attempts"]
        == result["router_environment_probes_passed"]
        == (1 if fault in {None, "router"} else 0)
    )
    assert result["router_environment_probe_failures"] == 0
    assert {
        key: result[key]
        for key in (
            "diagnostic_workers",
            "diagnostic_gpus_per_worker",
            "diagnostic_total_gpus",
            "num_engines",
            "tensor_parallel_size",
        )
    } == train._diagnostic_shape(plan, cfg)
    assert result["output_postconditions"] == {
        "runtime_files_unchanged": True,
        "unexpected_output_artifacts": 0,
        "checkpoint_artifacts": 0,
        "episode_artifacts": 0,
        "task_artifacts": 0,
    }
    assert (
        result["task_rows_read"]
        == result["verifier_calls"]
        == result["optimizer_steps"]
        == result["rollouts"]
        == result["checkpoints_created"]
        == 0
    )
    assert not any(
        result[key]
        for key in (
            "checkpoint_created",
            "wandb_initialized",
            "training_qualified",
            "production_training_shape_qualified",
        )
    )
    assert result["engine_start_qualified"] is (fault is None)
    assert "vllm_v1_multiprocessing_disabled" not in result
    assert (
        result["startup_phase"]
        == {
            None: "complete",
            "pre-engine": "engine_argument_build",
            "create": "engine_creation",
            "partial": "engine_creation",
            "timeout": "engine_creation",
            "router": "engine_creation",
        }[fault]
    )
    serialized = json.dumps(result)
    assert "outer-platform-key" not in serialized
    assert "must-be-scrubbed" not in serialized
    assert "private unrelated diagnostic" not in serialized
    assert "private partial engine failure" not in serialized
    assert "private engine argument failure" not in serialized
    assert "private post-probe router failure" not in serialized
    assert (root / "ENGINE_DIAGNOSTIC.json").is_file()
    started_receipt = json.loads((root / "ENGINE_DIAGNOSTIC_STARTED.json").read_text())
    train.sealed(started_receipt, train.ENGINE_DIAGNOSTIC_SCHEMA)
    assert all(started_receipt[key] == result[key] for key in train._diagnostic_shape(plan, cfg))
    assert "vllm_v1_multiprocessing_disabled" not in started_receipt
    assert calls[-1] == "ray-shutdown"


def test_engine_diagnostic_input_recheck_precedes_all_output(prepared, monkeypatch):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-input-drift"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(train, "job_request", lambda _: None)

    def reject_inputs(_):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise ValueError("staged model changed")

    monkeypatch.setattr(train, "check_inputs", reject_inputs)
    monkeypatch.setattr(train, "native_source", lambda: pytest.fail("loaded native source"))

    with pytest.raises(ValueError, match="staged model changed"):
        train.engine_diagnostic(plan)

    assert {path.name for path in root.iterdir()} == {".runtime"}
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_engine_diagnostic_plan_validation_uses_total_startup_deadline(prepared, monkeypatch):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-plan-drift"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    monkeypatch.setenv("RUN_DIR", str(root))

    def reject_plan(_):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise ValueError("plan runtime changed")

    monkeypatch.setattr(train, "job_request", reject_plan)
    monkeypatch.setattr(
        train,
        "_diagnostic_output_evidence",
        lambda *args, **kwargs: pytest.fail("output inspected before plan validation"),
    )

    with pytest.raises(ValueError, match="plan runtime changed"):
        train.engine_diagnostic(plan)

    assert list(root.iterdir()) == []
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_engine_diagnostic_input_recheck_uses_total_startup_deadline(prepared, monkeypatch):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-input-timeout"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    plan["arguments"]["engine_start_timeout_seconds"] = 1
    _materialize_diagnostic_runtime(plan)
    (root / ".runtime/plan.json").write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":"))
    )
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(train, "job_request", lambda _: None)
    monkeypatch.setattr(train, "check_inputs", lambda _: time.sleep(10))
    monkeypatch.setattr(train, "native_source", lambda: pytest.fail("deadline was not enforced"))

    started = time.monotonic()
    with pytest.raises(train.HardDeadlineExceeded, match="engine startup"):
        train.engine_diagnostic(plan)

    assert time.monotonic() - started < 3
    assert {path.name for path in root.iterdir()} == {".runtime"}
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("name", ["checkpoints", "episodes", "task-artifact.json"])
def test_engine_diagnostic_rejects_every_nonruntime_output_before_start(prepared, name):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-forbidden-output"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    path = root / name
    path.mkdir() if "." not in name else path.write_text("synthetic task data")

    with pytest.raises(ValueError, match="unexpected artifact"):
        train._diagnostic_output_evidence(plan, started=False)


def test_engine_diagnostic_rejects_runtime_bytecode_artifact(prepared):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-runtime-bytecode"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    bytecode = root / ".runtime/training/__pycache__/skyrl_training.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"synthetic bytecode")

    with pytest.raises(ValueError, match="runtime bundle changed"):
        train._diagnostic_output_evidence(plan, started=False)


def test_engine_diagnostic_rejects_empty_runtime_cache_directory(prepared):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-runtime-empty-cache"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    (root / ".runtime/training/__pycache__").mkdir()

    with pytest.raises(ValueError, match="runtime bundle changed"):
        train._diagnostic_output_evidence(plan, started=False)


def test_engine_diagnostic_runtime_proof_is_bound_to_plan_digest(prepared):
    plan = prepared.plan
    root = prepared.state.tmp / "diagnostic-runtime-digest"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    plan["runtime_sha256"] = "0" * 64
    _materialize_diagnostic_runtime(plan)

    with pytest.raises(ValueError, match="runtime sources changed"):
        train._diagnostic_output_evidence(plan, started=False)


@pytest.mark.parametrize(
    "fault",
    [
        "startup",
        "topology",
        "credential-isolation",
        "router-credential-isolation",
        "cleanup-proof",
        "cleanup-deadline",
        "artifact",
    ],
)
def test_engine_diagnostic_failures_only_write_nonqualifying_proven_receipts(
    prepared, monkeypatch, fault
):
    plan, calls = prepared.plan, []
    root = prepared.state.tmp / "engine-diagnostic-failure"
    root.mkdir()
    plan["output_root"] = plan["arguments"]["output_root"] = str(root)
    _materialize_diagnostic_runtime(plan)
    if fault == "cleanup-deadline":
        plan["arguments"]["engine_cleanup_timeout_seconds"] = 1
        # The plan changed, so its exact bundled plan.json changes with it.
        (root / ".runtime/plan.json").write_text(
            json.dumps(plan, sort_keys=True, separators=(",", ":"))
        )
    monkeypatch.setenv("RUN_DIR", str(root))
    if fault == "router-credential-isolation":
        monkeypatch.setenv("FLEET_API_KEY", "synthetic-router-leak")
        monkeypatch.setattr(train, "_scrubbed_process_environment", lambda names: nullcontext())
    monkeypatch.setattr(train, "job_request", lambda _: None)
    monkeypatch.setattr(train, "check_inputs", lambda _: None)
    engine_config = _exact_engine_config()
    cfg = NS(
        generator=NS(inference_engine=engine_config),
        trainer=NS(log_path=str(root / "private-native-logs")),
    )
    monkeypatch.setattr(train.skyrl, "diagnostic_native_config", lambda _: cfg)
    monkeypatch.setattr(
        train,
        "native_source",
        lambda: {"skyrl.train.utils.utils": NS(prepare_runtime_environment=lambda _: {})},
    )
    ray, _ = _fake_ray(
        calls,
        credential_count=1 if fault == "credential-isolation" else 0,
        nodes=(
            [{"Alive": True, "NodeID": "aa", "Resources": {"GPU": 4.0}}]
            if fault == "topology"
            else None
        ),
    )
    if fault == "startup":
        ray.init = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("private Ray initialization failure")
        )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.utils.ppo_utils",
        NS(sync_registries=lambda: pytest.fail("diagnostic created registry actors")),
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=lambda _: "args"),
    )

    def create(*args, **kwargs):
        if fault == "artifact":
            (root / "episodes").mkdir()
        group_type = sys.modules[
            "skyrl.backends.skyrl_train.inference_servers.server_group"
        ].ServerGroup
        if fault == "router-credential-isolation":
            group_type()
            router_type = sys.modules[
                "skyrl.backends.skyrl_train.inference_servers.vllm_router"
            ].VLLMRouter
            router_type().start()
        return NS(
            router=None,
            server_groups=[group_type(), group_type()],
            server_urls=["http://engine-0", "http://engine-1"],
            proxy_url="http://router",
        )

    _install_engine_modules(monkeypatch, calls, create)
    if fault == "cleanup-proof":
        monkeypatch.setattr(
            train,
            "_active_owned_resources",
            lambda *args: (_ for _ in ()).throw(TimeoutError("state unavailable")),
        )
    elif fault == "cleanup-deadline":
        monkeypatch.setattr(train, "_active_owned_resources", lambda *args: (1, 0))
    else:
        monkeypatch.setattr(train, "_active_owned_resources", lambda *args: (0, 0))

    events = []
    receipt_faults = {"topology", "credential-isolation", "router-credential-isolation"}
    if fault in receipt_faults:
        original_cleanup = train._cleanup_engine_diagnostic
        original_output = train._diagnostic_output_evidence
        original_write = train._write

        def cleanup(*args, **kwargs):
            result = original_cleanup(*args, **kwargs)
            events.append("cleanup-proven")
            return result

        def output(*args, **kwargs):
            result = original_output(*args, **kwargs)
            if kwargs.get("started"):
                events.append("output-proven")
            return result

        def write(path, value):
            result = original_write(path, value)
            if path.name == "ENGINE_DIAGNOSTIC.json":
                events.append("receipt-written")
            return result

        monkeypatch.setattr(train, "_cleanup_engine_diagnostic", cleanup)
        monkeypatch.setattr(train, "_diagnostic_output_evidence", output)
        monkeypatch.setattr(train, "_write", write)

    with pytest.raises((RuntimeError, ValueError, TimeoutError, BaseExceptionGroup)):
        train.engine_diagnostic(plan)
    events.append("error-propagated")

    receipt = root / "ENGINE_DIAGNOSTIC.json"
    if fault in receipt_faults:
        assert events == [
            "cleanup-proven",
            "output-proven",
            "receipt-written",
            "error-propagated",
        ]
        assert receipt.is_file()
        result = json.loads(receipt.read_text())
        train.sealed(result, train.ENGINE_DIAGNOSTIC_SCHEMA)
        assert result["status"] == "environment_isolation_rejected"
        assert not result["engine_start_qualified"]
        assert not result["credential_environment_isolation_proven"]
        assert result["cleanup"]["cleanup_proven"]
        assert result["cleanup"]["active_owned_actors"] == 0
        assert result["cleanup"]["active_owned_placement_groups"] == 0
        assert result["output_postconditions"] == {
            "runtime_files_unchanged": True,
            "unexpected_output_artifacts": 0,
            "checkpoint_artifacts": 0,
            "episode_artifacts": 0,
            "task_artifacts": 0,
        }
        assert (
            result["task_rows_read"]
            == result["verifier_calls"]
            == result["optimizer_steps"]
            == result["rollouts"]
            == result["checkpoints_created"]
            == 0
        )
        assert not any(
            result[key]
            for key in (
                "engine_start_qualified",
                "training_qualified",
                "production_training_shape_qualified",
                "checkpoint_created",
                "wandb_initialized",
            )
        )
        assert result["startup_phase"] in {
            "ray_actor_environment_probe",
            "engine_creation",
        }
        if fault == "topology":
            assert result["ray_gpu_nodes_discovered"] is None
            assert result["ray_gpu_nodes_probed"] == 0
            assert result["ray_actor_environment_probes_passed"] is None
            assert result["ray_actor_environment_probe_failures"] is None
            assert result["ray_actor_nonempty_scrubbed_credentials"] is None
        elif fault == "credential-isolation":
            assert result["ray_gpu_nodes_discovered"] == result["ray_gpu_nodes_probed"] == 2
            assert result["ray_actor_environment_probes_passed"] == 0
            assert result["ray_actor_environment_probe_failures"] == 2
            assert result["ray_actor_nonempty_scrubbed_credentials"] == 2
        else:
            assert result["ray_actor_environment_isolation_proven"]
            assert result["router_environment_probe_failures"] == 1
        assert "synthetic-router-leak" not in json.dumps(result)
    else:
        assert not receipt.exists()


def test_engine_instrumentation_tracks_partial_native_objects_and_scrubs_actor_env(
    monkeypatch,
):
    calls = []
    _install_engine_modules(monkeypatch, calls, lambda *args, **kwargs: None)
    ownership = train._DiagnosticOwnership()
    group_module = sys.modules["skyrl.backends.skyrl_train.inference_servers.server_group"]
    router_module = sys.modules["skyrl.backends.skyrl_train.inference_servers.vllm_router"]
    setup_module = sys.modules["skyrl.backends.skyrl_train.inference_servers.setup"]
    monkeypatch.setenv("FLEET_API_KEY", "outer-driver-value")
    router_module._run_router_with_logging = lambda: calls.append(
        ("router-child-key", os.environ.get("FLEET_API_KEY"))
    )
    group_module.build_engine_runtime_env = lambda **kwargs: {
        "env_vars": {
            "ACTOR_SECRET": "synthetic-private-value",
            "FLEET_API_KEY": "synthetic-overlay-value",
            "SAFE_SETTING": "yes",
        }
    }
    originals = (
        group_module.ServerGroup,
        router_module.VLLMRouter,
        router_module._run_router_with_logging,
        router_module.multiprocessing,
    )
    scrubbed = train._credential_names(os.environ)

    with train._instrument_engine_ownership(
        ownership,
        {
            "FLEET_API_KEY": "synthetic-must-be-blanked",
            "NATIVE_BASE_SETTING": "base",
        },
        scrubbed,
    ):
        assert router_module._run_router_with_logging is originals[2]
        assert router_module.multiprocessing is not originals[3]
        group = group_module.ServerGroup()
        actor_class = group._create_actor_class()
        actor_class.remote()
        router = router_module.VLLMRouter()
        assert router.start() == "http://router"
        setup_module.ray_placement_group([])
        group_module.placement_group([])
        runtime_env = actor_class._actor_class.options_kwargs["runtime_env"]
        assert runtime_env["env_vars"] == {
            **dict.fromkeys(scrubbed, ""),
            "ACTOR_SECRET": "",
            "FLEET_API_KEY": "",
            "NATIVE_BASE_SETTING": "base",
            "SAFE_SETTING": "yes",
            "WANDB_API_KEY": "",
        }

    assert len(ownership.groups) == len(ownership.routers) == 1
    assert len(ownership.actors) == len(ownership.engine_actors) == 1
    assert len(ownership.placement_groups) == 2
    assert ownership.router_start_attempts == ownership.router_credential_probes_passed == 1
    assert ownership.router_credential_probe_failures == 0
    assert ("router-child-key", "") in calls
    assert os.environ["FLEET_API_KEY"] == "outer-driver-value"
    assert (
        group_module.ServerGroup,
        router_module.VLLMRouter,
        router_module._run_router_with_logging,
        router_module.multiprocessing,
    ) == originals


def test_engine_instrumentation_rejects_forkserver_before_patching_or_starting(monkeypatch):
    _install_engine_modules(monkeypatch, [], lambda *args, **kwargs: None)
    router_module = sys.modules["skyrl.backends.skyrl_train.inference_servers.vllm_router"]
    originals = (router_module.VLLMRouter, router_module._run_router_with_logging)
    router_module.multiprocessing.get_start_method = lambda: "forkserver"
    ownership = train._DiagnosticOwnership()
    with (
        pytest.raises(train._DiagnosticContractRejected, match="unsupported native router"),
        train._instrument_engine_ownership(ownership, {}, ()),
    ):
        pytest.fail("unsupported context entered instrumentation")
    assert ownership.router_multiprocessing_start_method == "forkserver"
    assert not ownership.actors and not ownership.groups and not ownership.routers
    assert (router_module.VLLMRouter, router_module._run_router_with_logging) == originals


@pytest.mark.parametrize(
    "location", ["absorb", "router", "actor-remote", "actor-kill", "placement-group"]
)
def test_cleanup_never_swallows_hard_deadline(monkeypatch, location):
    calls = []
    ray, _ = _fake_ray(calls)
    original_get = ray.get
    ownership = train._DiagnosticOwnership()

    def expired(*args, **kwargs):
        raise train.HardDeadlineExceeded("SkyRL engine cleanup")

    if location == "absorb":
        ownership.groups.append(NS(get_actors=expired))
    elif location == "router":
        ownership.routers.append(NS(shutdown=expired))
    elif location in {"actor-remote", "actor-kill"}:
        actor = _Actor(NS())
        if location == "actor-remote":
            actor.shutdown = NS(remote=expired)
        else:
            ray.kill = expired
        ownership.actors.append(actor)
    else:
        ownership.placement_groups.append(object())
        monkeypatch.setitem(
            sys.modules,
            "ray.util.placement_group",
            NS(remove_placement_group=expired),
        )

    with pytest.raises(train.HardDeadlineExceeded, match="plan-bound hard deadline"):
        train._cleanup_engine_diagnostic(None, ray, ownership, 10)

    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert ray.get is original_get


def test_bare_ray_get_timeout_becomes_plan_deadline_and_restores_ray(monkeypatch):
    from training.rl_runtime import HardDeadline

    def wait(refs, *, timeout=None):
        assert 0 < timeout <= 10
        raise TimeoutError("synthetic Ray wait")

    ray = NS(get=wait)
    original = ray.get
    deadline = HardDeadline(10, "synthetic startup")
    with (
        train._bounded_ray_get(ray, deadline),
        pytest.raises(train.HardDeadlineExceeded, match="plan-bound"),
    ):
        ray.get(object())
    assert deadline.expired and ray.get is original


def test_dashboard_free_ray_state_proves_exact_current_job_ownership(monkeypatch):
    job_id = "a1"
    actor_table = {
        "own-alive": {"JobID": job_id, "State": "ALIVE"},
        "own-restarting": {"JobID": job_id.upper(), "State": "RESTARTING"},
        "own-dead": {"JobID": job_id, "State": "DEAD"},
        "other-alive": {"JobID": "b2", "State": "ALIVE"},
    }
    rows = [
        NS(creator_job_id=b"\xa1", state=1),
        NS(creator_job_id=b"\xa1", state=2),
        NS(creator_job_id=b"\xa1", state=3),
        NS(creator_job_id=b"\xb2", state=1),
    ]

    class PlacementGroupTableData:
        class PlacementGroupState:
            @staticmethod
            def Name(value):
                return {1: "CREATED", 2: "RESCHEDULING", 3: "REMOVED"}[value]

        @staticmethod
        def FromString(value):
            return value

    accessor = NS(get_placement_group_table=lambda: rows)
    monkeypatch.setitem(
        sys.modules,
        "ray._private.state",
        NS(
            actors=lambda: actor_table,
            state=NS(_connect_and_get_accessor=lambda: accessor),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.core.generated.gcs_pb2",
        NS(PlacementGroupTableData=PlacementGroupTableData),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._common.utils",
        NS(binary_to_hex=lambda value: value.hex()),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.util.state",
        NS(
            list_actors=lambda **kwargs: pytest.fail("dashboard actor API used"),
            list_placement_groups=lambda **kwargs: pytest.fail("dashboard PG API used"),
        ),
    )

    assert train._active_owned_resources(job_id, 10) == (2, 2)

    actor_table["own-alive"]["State"] = "DEAD"
    actor_table["own-restarting"]["State"] = "DEAD"
    rows[0].state = rows[1].state = 3
    assert train._active_owned_resources(job_id, 10) == (0, 0)


def test_ray_job_id_uses_raw_hex_not_decorated_display_string():
    value = _JobID("A1")
    assert str(value) == "JobID(A1)"
    assert train._ray_job_id_hex(value) == "a1"
    assert train._ray_job_id_hex("A1") == "a1"
    for invalid in ("JobID(A1)", "a", "", object()):
        with pytest.raises(ValueError, match="Ray job identity changed"):
            train._ray_job_id_hex(invalid)


def test_ray_state_uncertainty_cannot_prove_cleanup(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ray._private.state",
        NS(
            actors=lambda: {"actor": {"JobID": "a1"}},
            state=NS(_connect_and_get_accessor=lambda: pytest.fail("must fail first")),
        ),
    )

    with pytest.raises(ValueError, match="actor ownership record changed"):
        train._active_owned_resources("a1", 10)


@pytest.mark.parametrize(
    "mode",
    ["parent", "native", "diagnostic", "diagnostic_failure", "digest", "runtime", "execution"],
)
def test_main_dispatch_is_digest_bound_and_sanitized(prepared, monkeypatch, capsys, mode):
    plan, tmp = prepared.plan, prepared.state.tmp
    path = tmp / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []
    argv = [
        train.MODULE,
        "--plan",
        str(path),
        "--sha256",
        "wrong" if mode == "digest" else digest(plan),
    ]
    if mode == "native":
        argv.append("--native")
    elif mode in {"diagnostic", "diagnostic_failure"}:
        argv.append("--engine-diagnostic")
    monkeypatch.setattr(sys, "argv", argv)

    def request(p):
        if mode == "runtime":
            raise ValueError("synthetic secret not for stdout")

    def run(p, path):
        calls.append("parent")
        if mode == "execution":
            raise RuntimeError("synthetic private exception")
        return {"status": "native_loop_returned", "sha256": "synthetic"}

    monkeypatch.setattr(train, "job_request", request)
    monkeypatch.setattr(train, "run", run)
    monkeypatch.setattr(train, "_native", lambda p: calls.append("native"))

    def diagnostic(p):
        calls.append("diagnostic")
        if mode == "diagnostic_failure":
            raise RuntimeError("private diagnostic failure")
        return {"status": "engine_start_rejected", "sha256": "safe"}

    monkeypatch.setattr(train, "engine_diagnostic", diagnostic)
    if mode in {"diagnostic_failure", "digest", "runtime", "execution"}:
        with pytest.raises(SystemExit) as error:
            train.main()
        assert error.value.code == 1
    else:
        train.main()
        assert calls == [mode]
    output = capsys.readouterr().out
    assert "secret" not in output and "private" not in output


def test_shared_supervisor_uses_importable_skyrl_module(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp
    plan["output_root"] = str(root)
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(train, "native_source", lambda: None)
    monkeypatch.setattr(train, "native_result", lambda p: {"status": "native_loop_returned"})
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    calls = []

    def spawn(argv, **kw):
        calls.append(argv)
        return NS(pid=12345, poll=lambda: 0, returncode=0, wait=lambda **kw: 0)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = train.run(plan, root / "plan.json")
    assert result["status"] == "native_loop_returned"
    assert calls[0][2] == "training.skyrl_training" and calls[0][-1] == "--native"
    assert (root / "private-skyrl.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("receipt_io_failure", [False, True])
def test_native_main_preserves_sanitized_failure(prepared, monkeypatch, capsys, receipt_io_failure):
    from training import rl_runtime

    plan, root = prepared.plan, prepared.state.tmp
    plan["output_root"] = str(root)
    path = root / "native-plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(train, "job_request", lambda _: None)
    monkeypatch.setattr(
        sys, "argv", [train.MODULE, "--plan", str(path), "--sha256", digest(plan), "--native"]
    )

    def fail(_):
        raise PermissionError("private input contents")

    monkeypatch.setattr(train, "_native", fail)
    if receipt_io_failure:
        monkeypatch.setattr(rl_runtime, "native_failure", lambda *a: fail(None))
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "error_class": "PermissionError",
    }
    if not receipt_io_failure:
        receipt = json.loads((root / "NATIVE_FAILURE.json").read_text())
        assert receipt["causes"][0]["error_class"] == "PermissionError"
        assert "private input contents" not in json.dumps(receipt)


def test_native_main_records_budget_rejection_without_acceptance(prepared, monkeypatch, capsys):
    from training.rl_episode import EpisodeBudgetExceeded

    plan, root = prepared.plan, prepared.state.tmp
    plan["output_root"] = str(root)
    path = root / "native-plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(train, "job_request", lambda _: None)
    monkeypatch.setattr(
        sys, "argv", [train.MODULE, "--plan", str(path), "--sha256", digest(plan), "--native"]
    )

    def reject(_):
        raise EpisodeBudgetExceeded("generation_incomplete_length")

    monkeypatch.setattr(train, "_native", reject)
    train.main()
    assert (root / "NATIVE_REJECTED.json").exists()
    assert not (root / "NATIVE_FAILURE.json").exists() and not (root / "ACCEPTED.json").exists()
    assert capsys.readouterr().out == ""


def test_module_entry_rejects_missing_plan_without_private_path(tmp_path, monkeypatch, capsys):
    import runpy

    monkeypatch.delitem(sys.modules, train.MODULE)
    monkeypatch.setattr(
        sys, "argv", [train.MODULE, "--plan", str(tmp_path / "private-plan"), "--sha256", "0" * 64]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_module(train.MODULE, run_name="__main__")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "error_class": "FileNotFoundError",
    }


def test_skyrl_preflight_cli_and_submission_proof_dispatch(prepared, monkeypatch):
    plan, root = prepared.plan, prepared.state.tmp / "prepared"
    request = train.job_request(plan)
    cli._prepare(root, plan, request)
    monkeypatch.setattr(
        train,
        "preflight",
        lambda p: {
            "schema": "cyber_skyrl_training_cpu_preflight_v1",
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(p),
            "request_sha256": digest(request),
        },
    )
    result = CliRunner().invoke(cli.app, ["preflight", str(root)])
    assert result.exit_code == 0, result.output
    proof = json.loads((root / "PREFLIGHT.json").read_bytes())
    assert proof["schema"] == "cyber_skyrl_training_cpu_preflight_v1"
    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit_once(self, value, path):
            calls.append((value, path))
            return {"status": "synthetic-only"}

    monkeypatch.setattr(cli, "_client", lambda cluster: Client())
    assert CliRunner().invoke(cli.app, ["submit", str(root)]).exit_code == 0
    assert calls == [(request, root / "SUBMISSION.jsonl")]
    proof["schema"] = "cyber_miles_training_cpu_preflight_v1"
    proof["sha256"] = digest({k: v for k, v in proof.items() if k != "sha256"})
    (root / "PREFLIGHT.json").write_text(json.dumps(proof))
    assert CliRunner().invoke(cli.app, ["submit", str(root)]).exit_code == 2
    assert len(calls) == 1


@pytest.mark.skipif(importlib.util.find_spec("skyrl") is None, reason="pinned SkyRL image only")
def test_real_native_driver_and_dataset_source_identities():
    modules = train.native_source()
    assert modules["skyrl.train.entrypoints.main_base"].BasePPOExp is not None
