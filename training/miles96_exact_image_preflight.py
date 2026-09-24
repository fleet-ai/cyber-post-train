"""CPU-only proof that the pinned Miles image can open the repaired Fleet session."""

from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import importlib
import json
import os
import pathlib
import sys
from copy import deepcopy

NAME = "chris-q38-m96-image-preflight-dual-v1"
CONFIG_MAP = NAME + "-code"
ADAPTER_COMMIT = "5bfff503224ea42ee9c3ae1da9cf5fda2ba6043c"
KUBERNETES_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
PREVIEW_SCHEMA = "cyber_qwen38_miles96_exact_image_server_preview_v1"
ABSENCE_SCHEMA = "cyber_qwen38_miles96_exact_image_precreate_absence_v1"

EXPECTED_SOURCE_HASHES = {
    "fti.fleet.v1": "0524f19dcc886b20d17b39c21bd6417f537423eec6ef2359487fc3911dad441c",
    "fti.miles.v1.client_recording": (
        "41292533ec356a51a722c2c98f92bdfd98c56fdce429537a816957bf7172e9dc"
    ),
    "fti.miles.v1.common": "0a6801afe0cea5e4c0b6a53ffe07c083cc7e3691cf231dff460889e79c1626b0",
    "fti.trainers.miles.run_fleet": (
        "ae82e3f03d14c81e52009bc2cbff857d9e844baf07a9d82d8266608cb8cba2b1"
    ),
    "miles.backends.megatron_utils.actor": (
        "eecd72a4387511916add2c97d9e2dad6716db9097fd6ec471468c1f7edc074b9"
    ),
    "miles.backends.megatron_utils.hf_export": (
        "4986684bb62acf2ccd0e18a5ab7cf6bd50391f42d42be35ad3a377b36bbd4a34"
    ),
    "miles.rollout.inference_rollout.inference_rollout_common": (
        "96e3cba12ae033527e823ed3dd8cb43c31d244775756ecf0bab4ad81eb4f06a4"
    ),
    "miles.rollout.inference_rollout.inference_rollout_eval": (
        "7c13e0e1ab49cc1cb7224c3f9f91245d4c0bd46c0b7c1785cba8b633ca587a26"
    ),
    "miles.utils.http_utils": "da630d6594c86d76e89a262d1060c7f81238da9659899dea917987c5f39fab86",
    "miles.utils.tracking_utils.wandb_utils": (
        "d2a2bb4463b0a2158b2cd31e72e6e209a7f0092e182bf358ac07aea7cc68d5fe"
    ),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sealed(body: dict) -> dict:
    return {**body, "sha256": "sha256:" + canonical_digest(body)}


def normalize_server_manifest(value: dict) -> dict:
    """Remove only API-server identity fields that change across dry runs."""
    result = deepcopy(value)
    result.pop("status", None)
    metadata = result.setdefault("metadata", {})
    for key in ("creationTimestamp", "generation", "managedFields", "resourceVersion", "uid"):
        metadata.pop(key, None)
    if result.get("kind") == "Job":
        spec = result.setdefault("spec", {})
        spec.pop("selector", None)
        labels = spec.setdefault("template", {}).setdefault("metadata", {}).setdefault("labels", {})
        for key in (
            "batch.kubernetes.io/controller-uid",
            "batch.kubernetes.io/job-name",
            "controller-uid",
            "job-name",
        ):
            labels.pop(key, None)
    return result


def _contains(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_contains(left, right) for left, right in zip(expected, actual, strict=True))
        )
    return expected == actual


def validate_server_manifest(rendered: dict, expected: dict) -> dict:
    """Prove that a server-rendered preview preserves every safety-critical field."""
    normalized = normalize_server_manifest(rendered)
    if (
        normalized.get("apiVersion") != expected["apiVersion"]
        or normalized.get("kind") != expected["kind"]
        or normalized.get("metadata", {}).get("name") != expected["metadata"]["name"]
        or normalized.get("metadata", {}).get("namespace") != expected["metadata"]["namespace"]
        or not _contains(normalize_server_manifest(expected), normalized)
    ):
        raise ValueError("server preview differs from the local manifest")
    if expected["kind"] == "Job":
        pod = normalized["spec"]["template"]["spec"]
        if (
            normalized["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
            or normalized["spec"].get("suspend") is not True
            or normalized["spec"].get("backoffLimit") != 0
            or pod.get("restartPolicy") != "Never"
            or pod.get("automountServiceAccountToken") is not False
            or pod.get("priorityClassName") != "c1"
            or len(pod.get("containers", [])) != 1
            or pod.get("initContainers") not in (None, [])
            or pod.get("ephemeralContainers") not in (None, [])
            or len(pod.get("volumes", [])) != 1
            or any(
                "nvidia.com/gpu" in (container.get("resources", {}).get(section, {}))
                for container in pod["containers"]
                for section in ("requests", "limits")
            )
        ):
            raise ValueError("server preview violates the zero-GPU safety contract")
    elif normalized.get("immutable") is not True:
        raise ValueError("server preview ConfigMap is mutable")
    return normalized


def preview_receipt(rendered: dict, expected: dict, *, observed_at: str) -> dict:
    normalized = validate_server_manifest(rendered, expected)
    body = {
        "schema": PREVIEW_SCHEMA,
        "kind": expected["kind"],
        "name": expected["metadata"]["name"],
        "namespace": expected["metadata"]["namespace"],
        "local_manifest_sha256": "sha256:" + canonical_digest(expected),
        "normalized_manifest": normalized,
        "normalized_manifest_sha256": "sha256:" + canonical_digest(normalized),
        "observed_at": observed_at,
    }
    return sealed(body)


def validate_preview_receipt(receipt: dict, expected: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    if (
        receipt.get("sha256") != "sha256:" + canonical_digest(body)
        or body.get("schema") != PREVIEW_SCHEMA
        or body.get("kind") != expected["kind"]
        or body.get("name") != expected["metadata"]["name"]
        or body.get("namespace") != expected["metadata"]["namespace"]
        or body.get("local_manifest_sha256") != "sha256:" + canonical_digest(expected)
        or body.get("normalized_manifest_sha256")
        != "sha256:" + canonical_digest(body.get("normalized_manifest"))
    ):
        raise ValueError("server preview receipt is not exact")
    validate_server_manifest(body["normalized_manifest"], expected)
    dt.datetime.fromisoformat(body["observed_at"].replace("Z", "+00:00"))
    return receipt


def absence_receipt(
    *,
    job_absent: bool,
    config_map_absent: bool,
    pod_prefix_collision_count: int,
    workload_prefix_collision_count: int,
    observed_at: str,
) -> dict:
    return sealed(
        {
            "schema": ABSENCE_SCHEMA,
            "job_name": NAME,
            "config_map_name": CONFIG_MAP,
            "namespace": "fleet-train-jobs",
            "kubernetes_context": KUBERNETES_CONTEXT,
            "job_absent": job_absent,
            "config_map_absent": config_map_absent,
            "pod_prefix_collision_count": pod_prefix_collision_count,
            "workload_prefix_collision_count": workload_prefix_collision_count,
            "observed_at": observed_at,
        }
    )


def validate_absence_receipt(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    if (
        receipt.get("sha256") != "sha256:" + canonical_digest(body)
        or body.get("schema") != ABSENCE_SCHEMA
        or body.get("job_name") != NAME
        or body.get("config_map_name") != CONFIG_MAP
        or body.get("namespace") != "fleet-train-jobs"
        or body.get("kubernetes_context") != KUBERNETES_CONTEXT
        or body.get("job_absent") is not True
        or body.get("config_map_absent") is not True
        or body.get("pod_prefix_collision_count") != 0
        or body.get("workload_prefix_collision_count") != 0
    ):
        raise ValueError("precreate absence receipt is incomplete")
    dt.datetime.fromisoformat(body["observed_at"].replace("Z", "+00:00"))
    return receipt


def session_open_checks(mechanics, runtime: pathlib.Path, plan: dict) -> None:
    """Exercise the repaired session gate with one raw tool read per open."""
    from fti.fleet.v1 import openai_tools

    raw = json.loads((runtime / mechanics.TOOL_CATALOG_PATH).read_text())
    session_type = mechanics._evidence_session_class()
    base = session_type.__mro__[1]

    class FakeInstance:
        def __init__(self) -> None:
            self.calls = 0

        def list_tools(self):
            self.calls += 1
            return deepcopy(raw)

    def fake_base_open(self) -> None:
        self.instance = FakeInstance()
        self.tools = openai_tools(self.instance.list_tools())
        if getattr(self, "_visible_drift", False):
            self.tools[0]["function"]["name"] = "drifted"

    def fake_close(self) -> None:
        self.close_calls += 1

    def new_session(*, visible_drift: bool = False):
        session = session_type.__new__(session_type)
        session.task_key = plan["task_binding"]["task_key"]
        session.task_version_id = plan["task_binding"]["task_version_id"]
        session.verifier_version_id = plan["task_binding"]["verifier_version_id"]
        session.instance = None
        session.tools = []
        session.close_calls = 0
        session._visible_drift = visible_drift
        return session

    base.open = fake_base_open
    session_type.close = fake_close

    exact = new_session()
    exact.open()
    if exact.instance.calls != 1 or exact.close_calls != 0:
        raise ValueError("exact session-open contract did not pass once")

    visible_drift = new_session(visible_drift=True)
    try:
        visible_drift.open()
    except ValueError:
        pass
    else:
        raise ValueError("visible tool drift did not fail closed")
    if visible_drift.instance.calls != 1 or visible_drift.close_calls != 1:
        raise ValueError("visible tool drift cleanup was not exact")

    plan_path = runtime / "plan.json"
    original_plan = plan_path.read_text()
    changed = deepcopy(plan)
    changed["tool_contract"]["openai_tool_catalog_sha256"] = "sha256:" + "0" * 64
    plan_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")))
    plan_drift = new_session()
    try:
        plan_drift.open()
    except ValueError:
        pass
    else:
        raise ValueError("plan tool-contract drift did not fail closed")
    finally:
        plan_path.write_text(original_plan)
    if plan_drift.instance.calls != 1 or plan_drift.close_calls != 1:
        raise ValueError("plan drift cleanup was not exact")

    runtime_dir = os.environ["CYBER_RUNTIME_DIR"]
    os.environ["CYBER_RUNTIME_DIR"] = "/tmp/absent-miles96-preflight-runtime"
    plan_load_error = new_session()
    try:
        plan_load_error.open()
    except ValueError:
        pass
    else:
        raise ValueError("plan-load error did not fail closed")
    finally:
        os.environ["CYBER_RUNTIME_DIR"] = runtime_dir
    if plan_load_error.instance.calls != 1 or plan_load_error.close_calls != 1:
        raise ValueError("plan-load cleanup was not exact")


def build_packet(operator_commit: str) -> tuple[dict, dict, dict]:
    """Rebuild the exact ConfigMap, Kueue-managed Job, and packet binding."""
    from cyber_post_train.jobs import digest
    from training import miles96_mechanics_canary as mechanics
    from training import miles96_signal_qualification as signal
    from training import miles_signal_wave

    wave = miles_signal_wave.load()
    candidate = wave["candidates"][0]
    plan = signal.build_plan(
        name=candidate["identity"]["name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=signal.HF_MODEL_BINDING_SHA256,
        task_binding={
            **miles_signal_wave.task_binding(wave, candidate),
            "authority_receipt_sha256": candidate["authority_receipt_sha256"],
        },
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=candidate["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )
    request = signal.job_request(plan)
    chunks = sorted(
        (
            (int(key.rsplit("_", 1)[1]), value)
            for key, value in request["env"].items()
            if key.startswith("CYBER_RUNTIME_BUNDLE_")
            and key.removeprefix("CYBER_RUNTIME_BUNDLE_").isdigit()
        )
    )
    encoded = request["env"].get("CYBER_RUNTIME_BUNDLE") or "".join(value for _, value in chunks)
    blob = base64.b64decode(encoded, validate=True)
    source_closure = "sha256:" + mechanics.digest(plan["runtime_sources"])
    driver = pathlib.Path(__file__).read_text()
    driver_sha = "sha256:" + hashlib.sha256(driver.encode()).hexdigest()
    labels = {
        "app.kubernetes.io/name": "q38-m96-image-preflight",
        "fleet.ai/owner": "christopher",
    }
    annotations = {
        "fleet.ai/adapter-commit": ADAPTER_COMMIT,
        "fleet.ai/operator-commit": operator_commit,
        "fleet.ai/driver-sha256": driver_sha,
        "fleet.ai/source-closure-sha256": source_closure,
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": CONFIG_MAP,
            "namespace": mechanics.NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "immutable": True,
        "data": {"driver.py": driver, "runtime_bundle.b64": encoded},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": mechanics.NAMESPACE,
            "labels": {
                **labels,
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {**annotations, "fleet.ai/failure-alerts": "off"},
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "automountServiceAccountToken": False,
                    "restartPolicy": "Never",
                    "priorityClassName": "c1",
                    "containers": [
                        {
                            "name": "preflight",
                            "image": mechanics.IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "/preflight/driver.py"],
                            "env": [
                                {
                                    "name": "EXPECTED_BUNDLE_SHA256",
                                    "value": "sha256:" + hashlib.sha256(blob).hexdigest(),
                                },
                                {"name": "EXPECTED_DRIVER_SHA256", "value": driver_sha},
                                {
                                    "name": "EXPECTED_PLAN_SHA256",
                                    "value": "sha256:" + mechanics.digest(plan),
                                },
                                {
                                    "name": "EXPECTED_SOURCE_CLOSURE_SHA256",
                                    "value": source_closure,
                                },
                                {"name": "EXPECTED_RUNTIME_IMAGE", "value": mechanics.IMAGE},
                                {
                                    "name": "EXPECTED_IMAGE_DIGEST",
                                    "value": "sha256:" + mechanics.IMAGE.rsplit("@sha256:", 1)[1],
                                },
                                {"name": "RECEIPT_PATH", "value": "/tmp/preflight.json"},
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"},
                            },
                            "volumeMounts": [
                                {"name": "preflight", "mountPath": "/preflight", "readOnly": True}
                            ],
                        }
                    ],
                    "volumes": [{"name": "preflight", "configMap": {"name": CONFIG_MAP}}],
                },
            },
        },
    }
    body = {
        "schema": "cyber_qwen38_miles96_exact_image_preflight_packet_v1",
        "name": NAME,
        "config_map": CONFIG_MAP,
        "adapter_commit": ADAPTER_COMMIT,
        "operator_commit": operator_commit,
        "driver_sha256": driver_sha,
        "source_closure_sha256": source_closure,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "runtime_bundle_sha256": "sha256:" + hashlib.sha256(blob).hexdigest(),
        "runtime_image": mechanics.IMAGE,
        "job_manifest_sha256": "sha256:" + digest(job),
        "config_map_manifest_sha256": "sha256:" + digest(config_map),
        "requested_gpus": 0,
        "external_post_count": 0,
    }
    return {**body, "sha256": "sha256:" + digest(body)}, config_map, job


def main() -> None:
    encoded = pathlib.Path("/preflight/runtime_bundle.b64").read_text()
    blob = base64.b64decode(encoded, validate=True)
    if "sha256:" + hashlib.sha256(blob).hexdigest() != os.environ["EXPECTED_BUNDLE_SHA256"]:
        raise ValueError("runtime bundle digest changed")
    driver_sha256 = "sha256:" + sha256(pathlib.Path(__file__))
    if driver_sha256 != os.environ["EXPECTED_DRIVER_SHA256"]:
        raise ValueError("preflight driver changed")
    package = json.loads(gzip.decompress(blob))
    runtime = pathlib.Path("/tmp/runtime")
    runtime.mkdir(mode=0o700)
    for name, content in package["files"].items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    sys.path.insert(0, str(runtime))
    os.environ["CYBER_RUNTIME_DIR"] = str(runtime)
    importlib.invalidate_caches()

    from training import miles96_mechanics_canary as mechanics
    from training import miles96_signal_qualification as signal

    plan = signal.validate_plan(json.loads((runtime / "plan.json").read_text()))
    if "sha256:" + mechanics.digest(plan) != os.environ["EXPECTED_PLAN_SHA256"]:
        raise ValueError("signal plan digest changed")
    if plan["qualification"]["optimizer_steps"] != 0 or plan["acceptance"]["optimizer_steps"] != 0:
        raise ValueError("signal plan is not zero-update")
    if plan["runtime_sources"] != signal.runtime_source_manifest():
        raise ValueError("runtime source manifest changed")
    if (
        "sha256:" + mechanics.digest(plan["runtime_sources"])
        != os.environ["EXPECTED_SOURCE_CLOSURE_SHA256"]
    ):
        raise ValueError("runtime source closure changed")

    binding = signal._runtime_signal_binding(plan)
    session_open_checks(mechanics, runtime, plan)
    arguments = signal.native_arguments(plan)
    positions = {value: index for index, value in enumerate(arguments)}
    if (
        arguments[:2] != ["-m", "fti.trainers.miles.run_fleet"]
        or arguments[positions["--mode"] + 1] != "eval"
        or arguments[positions["--n-samples-per-prompt"] + 1] != "8"
        or arguments[positions["--rollout-batch-size"] + 1] != "1"
        or any(value in arguments for value in ("--save", "--save-interval", "--post-save"))
        or binding["sample_count"] != 8
        or binding["max_concurrent_envs"] != 2
        or binding["outer_episode_replacements"] != 0
    ):
        raise ValueError("zero-update evaluation entrypoint changed")

    import fti

    if fti.__version__ != "0.10.27":
        raise ValueError("installed FTI version changed")
    observed = {}
    for module_name, expected in EXPECTED_SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        value = sha256(pathlib.Path(module.__file__))
        if value != expected:
            raise ValueError(f"installed source drift: {module_name}")
        observed[module_name] = "sha256:" + value

    body = {
        "schema": "cyber_qwen38_miles96_exact_image_preflight_v1",
        "status": "passed",
        "runtime_image": os.environ["EXPECTED_RUNTIME_IMAGE"],
        "image_digest": os.environ["EXPECTED_IMAGE_DIGEST"],
        "source_closure_sha256": os.environ["EXPECTED_SOURCE_CLOSURE_SHA256"],
        "driver_sha256": driver_sha256,
        "runtime_bundle_sha256": os.environ["EXPECTED_BUNDLE_SHA256"],
        "plan_sha256": os.environ["EXPECTED_PLAN_SHA256"],
        "raw_tool_catalog_sha256": mechanics.RAW_TOOL_CATALOG_SHA256,
        "openai_tool_catalog_sha256": mechanics.OPENAI_TOOL_CATALOG_SHA256,
        "tool_transform_source_sha256": "sha256:" + mechanics.FTI_V1_SHA256,
        "source_hashes": observed,
        "checks": {
            "image_digest_exact": True,
            "pinned_fti_imports": True,
            "pinned_miles_sources": True,
            "zero_update_entrypoint": True,
            "sealed_raw_tool_catalog_exact": True,
            "pinned_openai_projection_exact": True,
            "session_open_exact_catalog_passed": True,
            "session_open_visible_drift_rejected_and_closed": True,
            "session_open_plan_drift_rejected_and_closed": True,
            "session_open_plan_load_error_rejected_and_closed": True,
        },
        "observed_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    receipt = {**body, "sha256": "sha256:" + canonical_digest(body)}
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    pathlib.Path(os.environ["RECEIPT_PATH"]).write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
