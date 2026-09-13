"""Fresh dev9 identity and terminal-dev8 replay gates for the SkyRL smoke."""
# ruff: noqa: F811

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import yaml
from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared as skyrl_prepared  # noqa: F401

from cyber_post_train import cli
from cyber_post_train.jobs import JobsError
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_V8 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v8.json"
)
DATA_V9 = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v9.json"
)
RUN_V8 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v8.json"
RUN_V9 = ROOT / "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v9.json"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
DEFAULT_USER_EVIDENCE = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-12-skyrl-worker-rpc-relay-image-default-user-qualification-v1.json"
)
_OMITTED = object()


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def preview_with_contexts(*, pod_context=_OMITTED, container_context=_OMITTED) -> dict:
    spec = {"containers": [{}]}
    if pod_context is not _OMITTED:
        spec["securityContext"] = pod_context
    if container_context is not _OMITTED:
        spec["containers"][0]["securityContext"] = container_context
    template = {"spec": spec}
    return {
        "manifest_yaml": yaml.safe_dump(
            {
                "metadata": {"namespace": skyrl_training.DEV_KUBERNETES_NAMESPACE},
                "spec": {
                    "rayClusterSpec": {
                        "headGroupSpec": {"template": copy.deepcopy(template)},
                        "workerGroupSpecs": [
                            {"replicas": 1, "template": copy.deepcopy(template)}
                        ],
                    }
                },
            }
        )
    }


def fallback_identity() -> tuple[dict, dict]:
    plan = {
        "run_name": skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME,
        "output_root": skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_OUTPUT_ROOT,
        "execution": {
            "cluster_target": "dev",
            "image": IMAGE,
            "image_cpu_qualification": copy.deepcopy(
                skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION
            ),
        },
    }
    request = {
        "image": IMAGE,
        "workers": skyrl_training.ENGINE_DIAGNOSTIC_WORKERS,
        "gpus_per_worker": skyrl_training.ENGINE_DIAGNOSTIC_GPUS_PER_WORKER,
        "env": {
            "CYBER_EXPECTED_RUNTIME_UID": "1000",
            "CYBER_EXPECTED_RUNTIME_GID": "100",
        },
    }
    return plan, request


def force_absent_context_fallback(monkeypatch) -> None:
    def missing_context(*_args):
        raise JobsError("engine diagnostic preview runtime user differs from 1000:100")

    monkeypatch.setattr(skyrl_training, "validate_engine_diagnostic_preview", missing_context)


def test_dev9_is_a_fresh_dev_only_identity_with_unchanged_science():
    data_v8, data_v9 = load(DATA_V8), load(DATA_V9)
    run_v8, run_v9 = load(RUN_V8), load(RUN_V9)

    assert RUN_V9.name == skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_PATH
    assert sha256(RUN_V9) == skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_FILE_SHA256
    assert sha256(DATA_V9) == (skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_CONFIG_FILE_SHA256)
    assert run_v9["name"] == data_v9["name"] == run_v9["wandb"]["run_id"]
    assert run_v9["name"] == skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME
    assert run_v9["output_root"] == skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_OUTPUT_ROOT
    assert data_v9["output"] == skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_ROOT
    assert run_v9["data"]["root"] == data_v9["output"]
    assert run_v9["data"]["manifest"] == data_v9["output"] + "/manifest.json"
    assert run_v9["cluster"]["target"] == "dev"

    assert data_v9["name"] != data_v8["name"]
    assert data_v9["output"] != data_v8["output"]
    assert run_v9["output_root"] != run_v8["output_root"]
    assert run_v9["wandb"]["run_id"] != run_v8["wandb"]["run_id"]
    prepared = Path(skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_PREPARED_ROOT)
    assert prepared.is_absolute()
    assert prepared != Path(run_v9["output_root"])
    assert not prepared.is_relative_to(Path(run_v9["output_root"]))
    assert not Path(run_v9["output_root"]).is_relative_to(prepared)
    assert prepared != Path(data_v9["output"])

    for key in ("backend", "task_set", "split", "tool_catalog", "model_lock", "model_root"):
        assert data_v9[key] == data_v8[key]
    assert data_v9["limits"] == data_v8["limits"]
    for key in ("backend", "model", "recipe"):
        assert run_v9[key] == run_v8[key]
    assert {key: value for key, value in run_v9["cluster"].items() if key != "target"} == (
        run_v8["cluster"]
    )


def test_dev9_uses_the_new_qualified_relay_image(skyrl_prepared):
    plan = copy.deepcopy(skyrl_prepared.plan)
    request = skyrl_training.engine_diagnostic_request(plan)

    assert skyrl_training.IMAGE == request["image"] == IMAGE
    assert skyrl_training.ENGINE_IMAGE_CPU_QUALIFICATION["status"] == "qualified"
    assert request["workers"] == 2
    assert request["gpus_per_worker"] == 4
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["env"]["CYBER_EXPECTED_RUNTIME_UID"] == "1000"
    assert request["env"]["CYBER_EXPECTED_RUNTIME_GID"] == "100"
    assert "WANDB" not in json.dumps(request["env"])


def test_absent_context_exception_is_bound_to_the_exact_default_user_receipt():
    payload = DEFAULT_USER_EVIDENCE.read_bytes()
    evidence = json.loads(payload)
    unsigned = {key: value for key, value in evidence.items() if key != "receipt_sha256"}
    binding = cli._DEV9_ABSENT_RUNTIME_CONTEXT_QUALIFICATION

    assert hashlib.sha256(payload).hexdigest() == binding["default_user_evidence_file_sha256"]
    assert (
        hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        ).hexdigest()
        == evidence["receipt_sha256"]
        == binding["default_user_evidence_receipt_sha256"]
    )
    assert evidence["requested_image"] == evidence["runtime"]["runtime_image_id"] == IMAGE
    assert evidence["runtime"]["pod_security_context"] == {}
    assert evidence["runtime"]["container_security_context"] is None
    assert (evidence["runtime"]["effective_uid"], evidence["runtime"]["effective_gid"]) == (
        1000,
        100,
    )
    assert evidence["cleanup"]["pod_absent_after_delete"] is True


@pytest.mark.parametrize("missing", [None, "runAsUser", "runAsGroup", "runAsNonRoot"])
def test_dev9_preview_accepts_only_truly_absent_identity_fields(monkeypatch, missing):
    force_absent_context_fallback(monkeypatch)
    plan, request = fallback_identity()
    expected = {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
    context = (
        {}
        if missing is None
        else {key: value for key, value in expected.items() if key != missing}
    )

    result = cli._validate_engine_diagnostic_preview(
        plan, request, preview_with_contexts(pod_context=context)
    )

    assert result["runtime_user"] == {"uid": 1000, "gid": 100}
    assert result["runtime_user_evidence"] == {
        "mode": "qualified_image_default_with_gpu_entrypoint_recheck",
        "qualification_receipt_sha256": (
            "d1080784334becc76edb698ce363ecc513e5c98f9fce6c9510de59136192129f"
        ),
        "omitted_preview_fields": 6 if missing is None else 2,
        "explicit_conflicts_rejected": True,
    }
    assert result["pods"] == 2


@pytest.mark.parametrize(
    ("pod_context", "container_context"),
    [
        ({"runAsUser": 0}, _OMITTED),
        (_OMITTED, {"runAsGroup": 0}),
        ({"runAsNonRoot": False}, _OMITTED),
        (_OMITTED, {"runAsUser": None}),
        ({"runAsUser": True}, _OMITTED),
        (_OMITTED, {"runAsNonRoot": 1}),
        ({"runAsUser": 0}, {"runAsUser": 1000}),
    ],
)
def test_dev9_preview_rejects_every_explicit_identity_conflict(
    monkeypatch, pod_context, container_context
):
    force_absent_context_fallback(monkeypatch)
    plan, request = fallback_identity()

    with pytest.raises(JobsError, match="explicit runtime user conflict"):
        cli._validate_engine_diagnostic_preview(
            plan,
            request,
            preview_with_contexts(
                pod_context=pod_context,
                container_context=container_context,
            ),
        )


@pytest.mark.parametrize("drift", ["image", "qualification"])
def test_absent_context_exception_rejects_any_other_image_or_proof(monkeypatch, drift):
    force_absent_context_fallback(monkeypatch)
    plan, request = fallback_identity()
    if drift == "image":
        request["image"] = request["image"].replace("89758df2", "09758df2")
    else:
        plan["execution"]["image_cpu_qualification"]["status"] = "unqualified"

    with pytest.raises(JobsError, match="runtime user differs"):
        cli._validate_engine_diagnostic_preview(
            plan, request, preview_with_contexts()
        )


def test_gpu_entrypoint_rechecks_actual_identity_before_ray_import():
    source = inspect.getsource(skyrl_training.engine_diagnostic)
    assert source.index("_require_engine_diagnostic_runtime_identity(plan)") < source.index(
        "import ray as ray_module"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_name", "chris-q38-rldiag-dev8"),
        ("output_root", "/mnt/sfs/jobs/chris-q38-rldiag-dev8"),
        (
            "data_root",
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev8/data",
        ),
        ("wandb_run_id", "chris-q38-rldiag-dev8"),
    ],
)
def test_terminal_dev8_identity_cannot_be_prepared_or_replayed(skyrl_prepared, field, value):
    plan = copy.deepcopy(skyrl_prepared.plan)
    if field == "run_name":
        plan["run_name"] = value
        plan["arguments"]["name"] = value
    elif field == "output_root":
        plan["output_root"] = value
        plan["arguments"]["output_root"] = value
    elif field == "data_root":
        plan["arguments"]["train_data"] = value + "/train.jsonl"
        plan["arguments"]["dev_data"] = value + "/dev.jsonl"
        plan["arguments"]["data_manifest"] = value + "/manifest.json"
    else:
        plan["arguments"]["wandb_run_id"] = value

    with pytest.raises(ValueError, match="terminal dev8 identity cannot be replayed"):
        skyrl_training.job_request(plan)


def test_partial_dev9_identity_is_rejected_before_request_render(skyrl_prepared):
    plan = copy.deepcopy(skyrl_prepared.plan)
    plan["run_name"] = skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME
    plan["arguments"]["name"] = skyrl_training.ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME

    with pytest.raises(ValueError, match="dev9 identity is incomplete or mixed"):
        skyrl_training.job_request(plan)


def test_historical_dev8_embedded_proof_cannot_be_relabelled_as_dev9():
    forged = {
        "config_path": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_PATH,
        "config_file_sha256": skyrl_training.ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256,
        "image_sha256": skyrl_training.IMAGE.rsplit("@sha256:", 1)[1],
    }

    with pytest.raises(ValueError, match="historical and privacy-disqualified"):
        skyrl_training._validate_embedded_engine_prerequisite(forged, required=True)
