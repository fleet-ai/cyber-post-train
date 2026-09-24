"""ConfigMap-free CPU proof for the maintained Miles96 signal runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest
from cyber_post_train.sft_cpu_preflight_job import (
    render_cpu_preflight_job,
    validate_completed_cpu_preflight_job,
    validate_cpu_preflight_job_response,
)
from training import miles96_signal_qualification as signal
from training import miles_signal_wave
from training.miles96_exact_image_preflight_driver import (
    ENV_DRIVER_SHA256,
    ENV_PLAN_SHA256,
    ENV_RECEIPT_PATH,
    ENV_REQUEST_SHA256,
    ENV_RUNTIME_IMAGE,
    ENV_SOURCE_CLOSURE_SHA256,
    LOG_PREFIX,
    RUNTIME_BUNDLE,
    RUNTIME_BUNDLE_SHA256,
    SCHEMA,
)

NAME = "chris-q38-m96-image-pre-v3"
ROLE = "miles96-image-preflight"
ADAPTER_COMMIT = "978df19a1f6b344e2f88d9502060700a59294681"
SOURCE_CLOSURE_SHA256 = "sha256:de52b39f55e92a079bef1c0ea14b9e823e4af5096318dca5cf48cc800225fd15"
DRIVER_PATH = Path(__file__).with_name("miles96_exact_image_preflight_driver.py")
DRIVER_ENV = "CYBER_MILES96_PREFLIGHT_DRIVER_B64"
PLAN_ANNOTATION = "cyber-post-train.fleet.ai/plan-sha256"
REQUEST_ANNOTATION = "cyber-post-train.fleet.ai/request-sha256"
BUNDLE_ANNOTATION = "cyber-post-train.fleet.ai/runtime-bundle-sha256"
DRIVER_ANNOTATION = "cyber-post-train.fleet.ai/preflight-driver-sha256"
SOURCE_COMMIT_ANNOTATION = "cyber-post-train.fleet.ai/source-commit"
ADAPTER_COMMIT_ANNOTATION = "cyber-post-train.fleet.ai/adapter-commit"
CREATE_AUTHORIZATION_SCHEMA = "cyber_qwen38_miles96_image_preflight_create_auth_v1"
RECONCILED_CREATE_SCHEMA = "cyber_qwen38_miles96_image_preflight_reconcile_v1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class MilesImagePreflightPackage:
    job: dict[str, Any]
    plan: dict[str, Any]
    request: dict[str, Any]
    source_commit: str
    driver_sha256: str
    runtime_bundle_sha256: str


def _driver() -> tuple[str, str]:
    raw = DRIVER_PATH.read_bytes()
    try:
        source = raw.decode()
    except UnicodeDecodeError:
        raise ValueError("Miles image-preflight driver is not UTF-8") from None
    if source.encode() != raw:
        raise ValueError("Miles image-preflight driver is not byte-stable")
    return source, "sha256:" + hashlib.sha256(raw).hexdigest()


def _runtime_transport(request: dict[str, Any]) -> tuple[dict[str, str], bytes]:
    env = request["env"]
    if RUNTIME_BUNDLE in env:
        transport = {RUNTIME_BUNDLE: env[RUNTIME_BUNDLE]}
        encoded = env[RUNTIME_BUNDLE]
    else:
        transport = {
            key: value
            for key, value in env.items()
            if key.startswith(RUNTIME_BUNDLE + "_")
            and key.removeprefix(RUNTIME_BUNDLE + "_").isdigit()
        }
        indexes = sorted(int(key.removeprefix(RUNTIME_BUNDLE + "_")) for key in transport)
        if indexes != list(range(len(indexes))):
            raise ValueError("Miles runtime transport chunks are incomplete")
        encoded = "".join(transport[f"{RUNTIME_BUNDLE}_{index}"] for index in indexes)
    try:
        blob = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Miles runtime transport is not canonical base64") from exc
    observed = "sha256:" + hashlib.sha256(blob).hexdigest()
    if env.get(RUNTIME_BUNDLE_SHA256) != observed:
        raise ValueError("Miles runtime transport digest changed")
    return {**transport, RUNTIME_BUNDLE_SHA256: observed}, blob


def _plan_and_request() -> tuple[dict[str, Any], dict[str, Any]]:
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
    return plan, signal.job_request(plan)


def _bootstrap(driver_sha256: str) -> list[str]:
    source = (
        "import base64,hashlib,os;"
        f"s=base64.b64decode(os.environ.pop({DRIVER_ENV!r}),validate=True);"
        f"assert 'sha256:'+hashlib.sha256(s).hexdigest()=={driver_sha256!r};"
        "exec(compile(s,'<miles96-image-preflight>','exec'))"
    )
    return ["python", "-u", "-c", source]


def _build(source_commit: str) -> MilesImagePreflightPackage:
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("Miles image-preflight source commit must be exact")
    plan, request = _plan_and_request()
    transport, blob = _runtime_transport(request)
    driver, driver_sha256 = _driver()
    plan_sha256 = "sha256:" + digest(plan)
    request_sha256 = "sha256:" + digest(request)
    bundle_sha256 = "sha256:" + hashlib.sha256(blob).hexdigest()
    job = render_cpu_preflight_job(
        name=NAME,
        image=request["image"],
        role=ROLE,
        command=_bootstrap(driver_sha256),
        environment={
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "WANDB_MODE": "disabled",
            "PYTHONDONTWRITEBYTECODE": "1",
            DRIVER_ENV: base64.b64encode(driver.encode()).decode(),
            ENV_DRIVER_SHA256: driver_sha256,
            ENV_PLAN_SHA256: plan_sha256,
            ENV_REQUEST_SHA256: request_sha256,
            ENV_SOURCE_CLOSURE_SHA256: SOURCE_CLOSURE_SHA256,
            ENV_RUNTIME_IMAGE: request["image"],
            ENV_RECEIPT_PATH: "/tmp/preflight.json",
            **transport,
        },
        annotations={
            PLAN_ANNOTATION: plan_sha256,
            REQUEST_ANNOTATION: request_sha256,
            BUNDLE_ANNOTATION: bundle_sha256,
            DRIVER_ANNOTATION: driver_sha256,
            SOURCE_COMMIT_ANNOTATION: source_commit,
            ADAPTER_COMMIT_ANNOTATION: ADAPTER_COMMIT,
        },
        resources={
            "requests": {"cpu": "1", "memory": "2Gi", "ephemeral-storage": "1Gi"},
            "limits": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "2Gi"},
        },
        image_pull_secrets=request.get("image_pull_secrets", []),
        termination_message_path="/tmp/preflight.json",
    )
    return MilesImagePreflightPackage(
        job=job,
        plan=plan,
        request=request,
        source_commit=source_commit,
        driver_sha256=driver_sha256,
        runtime_bundle_sha256=bundle_sha256,
    )


def build_package(source_commit: str) -> MilesImagePreflightPackage:
    package = _build(source_commit)
    validate_package(package)
    return package


def validate_package(package: MilesImagePreflightPackage) -> dict[str, Any]:
    if not isinstance(package, MilesImagePreflightPackage) or package != _build(
        package.source_commit
    ):
        raise ValueError("Miles image-preflight package differs from rebuilt source bytes")
    return {
        "name": NAME,
        "plan_sha256": "sha256:" + digest(package.plan),
        "request_sha256": "sha256:" + digest(package.request),
        "manifest_sha256": "sha256:" + digest(package.job),
        "driver_sha256": package.driver_sha256,
        "runtime_bundle_sha256": package.runtime_bundle_sha256,
        "source_commit": package.source_commit,
        "gpus": 0,
    }


def validate_receipt(
    receipt: dict[str, Any], package: MilesImagePreflightPackage
) -> dict[str, Any]:
    proof = validate_package(package)
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    tools = package.plan["tool_contract"]
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "gpus",
            "job_name",
            "runtime_image",
            "image_digest",
            "source_closure_sha256",
            "driver_sha256",
            "runtime_bundle_sha256",
            "plan_sha256",
            "request_sha256",
            "runtime_binding_sha256",
            "raw_tool_catalog_sha256",
            "openai_tool_catalog_sha256",
            "tool_transform_source_sha256",
            "checks",
            "observed_at_unix",
            "sha256",
        }
        or receipt.get("sha256") != "sha256:" + digest(body)
        or receipt.get("schema") != SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("gpus") != 0
        or receipt.get("job_name") != NAME
        or receipt.get("runtime_image") != package.request["image"]
        or receipt.get("image_digest") != package.request["image"].rsplit("@", 1)[-1]
        or receipt.get("source_closure_sha256") != SOURCE_CLOSURE_SHA256
        or receipt.get("driver_sha256") != proof["driver_sha256"]
        or receipt.get("runtime_bundle_sha256") != proof["runtime_bundle_sha256"]
        or receipt.get("plan_sha256") != proof["plan_sha256"]
        or receipt.get("request_sha256") != proof["request_sha256"]
        or _SHA256.fullmatch(str(receipt.get("runtime_binding_sha256"))) is None
        or receipt.get("raw_tool_catalog_sha256") != tools["raw_tool_catalog_sha256"]
        or receipt.get("openai_tool_catalog_sha256") != tools["openai_tool_catalog_sha256"]
        or receipt.get("tool_transform_source_sha256") != tools["transform_source_sha256"]
        or receipt.get("checks")
        != {
            "pinned_runtime_binding": True,
            "zero_update_entrypoint": True,
            "session_open_exact_catalog": True,
            "session_open_raw_drift_rejected_and_closed": True,
        }
        or type(receipt.get("observed_at_unix")) not in {int, float}
    ):
        raise ValueError("Miles exact-image preflight receipt is incomplete")
    return receipt


def validate_completed_preflight(
    package: MilesImagePreflightPackage,
    job: dict[str, Any],
    workloads: dict[str, Any],
    pods: dict[str, Any],
    service_account: dict[str, Any],
    logs: str,
) -> dict[str, Any]:
    validate_package(package)
    validate_completed_cpu_preflight_job(package.job, job, workloads, pods, service_account)
    lines = logs.splitlines()
    if len(lines) != 1 or not lines[0].startswith(LOG_PREFIX):
        raise ValueError("Miles image-preflight logs do not contain one sanitized receipt")
    try:
        receipt = json.loads(lines[0].removeprefix(LOG_PREFIX))
    except json.JSONDecodeError as exc:
        raise ValueError("Miles image-preflight receipt is invalid JSON") from exc
    return validate_receipt(receipt, package)


def validate_response(
    actual: dict[str, Any],
    package: MilesImagePreflightPackage,
    *,
    require_uid: bool,
    admitted: bool = False,
    admitted_workload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_package(package)
    return validate_cpu_preflight_job_response(
        actual,
        package.job,
        require_uid=require_uid,
        admitted=admitted,
        admitted_workload=admitted_workload,
    )


def authorize_create_once(
    package: MilesImagePreflightPackage,
    *,
    prod_previews: list[dict[str, Any]],
    observer: dict[str, Any],
    operation_root: Path,
    parent_review_sha256: str,
) -> dict[str, Any]:
    """Wrap the proven stage create core with a Miles/root-review binding."""
    from training import skyrl_prod9_direct as direct

    proof = validate_package(package)
    if (
        len(prod_previews) != 2
        or _SHA256.fullmatch(parent_review_sha256) is None
        or operation_root.is_symlink()
        or not operation_root.is_absolute()
        or not operation_root.is_dir()
    ):
        raise ValueError("Miles image-preflight create authority is incomplete")
    previews = [
        direct.validate_cpu_preview_proof(
            package.job,
            preview,
            purpose="preflight",
            context=direct.PROD_CONTEXT,
            fresh=True,
        )
        for preview in prod_previews
    ]
    if previews[0].get("server_render_sha256") != previews[1].get("server_render_sha256"):
        raise ValueError("Miles image-preflight production previews differ")
    armed = direct._observer_pid(
        observer,
        kind="job",
        name=NAME,
        plan_sha256=proof["plan_sha256"],
        manifest_sha256=proof["manifest_sha256"],
        gpus=0,
        maximum_seconds=direct.CPU_MAXIMUM_SECONDS,
    )
    core = direct._seal(
        {
            "schema": direct.STAGE_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "stage_spec_sha256": proof["plan_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            # The create core only freshness-checks these two reviewed proofs;
            # the outer Miles receipt above proves both are production previews.
            "dev_preview": previews[0],
            "prod_preview": previews[1],
            "observer": armed,
            "operation_root": str(operation_root.resolve()),
        }
    )
    return direct._seal(
        {
            "schema": CREATE_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "parent_review_sha256": parent_review_sha256,
            "plan_sha256": proof["plan_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "source_commit": package.source_commit,
            "core_authorization": core,
        }
    )


def create_once(
    package: MilesImagePreflightPackage,
    operation_root: Path,
    authorization: dict[str, Any],
    *,
    expected_parent_review_sha256: str,
    runner=subprocess.run,
    dev_duplicate_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Issue the shared rail's only mutating call for the exact Miles package."""
    from training import skyrl_prod9_direct as direct

    auth = direct._validate_seal(authorization, CREATE_AUTHORIZATION_SCHEMA)
    if auth.get("parent_review_sha256") != expected_parent_review_sha256:
        raise ValueError("Miles image-preflight parent review changed")
    expected = authorize_create_once(
        package,
        prod_previews=[
            auth["core_authorization"]["dev_preview"],
            auth["core_authorization"]["prod_preview"],
        ],
        observer=auth["core_authorization"]["observer"],
        operation_root=operation_root,
        parent_review_sha256=expected_parent_review_sha256,
    )
    if auth != expected:
        raise ValueError("Miles image-preflight create authorization changed")
    proof = validate_package(package)
    return direct._create_cpu_once(
        operation_root,
        package.job,
        auth["core_authorization"],
        purpose="stage",
        name=NAME,
        plan_sha256=proof["plan_sha256"],
        runner=runner,
        dev_duplicate_proof=dev_duplicate_proof,
    )


def reconcile_create_intent(
    package: MilesImagePreflightPackage,
    operation_root: Path,
    authorization: dict[str, Any],
    *,
    expected_parent_review_sha256: str,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Read back an ambiguous create by exact name/UID; never issue a create."""
    from types import SimpleNamespace

    from cyber_post_train.sfs_output_job import _validate_admitted_workload
    from training import skyrl_prod9_direct as direct

    proof = validate_package(package)
    auth = direct._validate_seal(authorization, CREATE_AUTHORIZATION_SCHEMA)
    core = direct._validate_seal(auth.get("core_authorization"), direct.STAGE_AUTHORIZATION_SCHEMA)
    journal = operation_root / "PROD9_STAGE_CREATE.jsonl"
    if (
        auth.get("parent_review_sha256") != expected_parent_review_sha256
        or auth.get("plan_sha256") != proof["plan_sha256"]
        or auth.get("manifest_sha256") != proof["manifest_sha256"]
        or auth.get("source_commit") != package.source_commit
        or core.get("stage_spec_sha256") != proof["plan_sha256"]
        or core.get("manifest_sha256") != proof["manifest_sha256"]
        or operation_root.resolve() != Path(core.get("operation_root", ""))
        or journal.is_symlink()
        or not journal.is_file()
    ):
        raise ValueError("Miles image-preflight reconciliation authority is incomplete")
    lines = journal.read_text().splitlines()
    if len(lines) not in {1, 2}:
        raise ValueError("Miles image-preflight create journal is invalid")
    try:
        intent = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Miles image-preflight create intent is invalid") from exc
    if (
        intent.get("state") != "CREATE_INTENT_DO_NOT_RETRY"
        or intent.get("purpose") != "stage"
        or intent.get("plan_sha256") != proof["plan_sha256"]
        or intent.get("manifest_sha256") != proof["manifest_sha256"]
        or intent.get("authorization_sha256") != core["sha256"]
        or not isinstance(intent.get("duplicate_checks"), dict)
    ):
        raise ValueError("Miles image-preflight create intent changed")
    if len(lines) == 2:
        try:
            return direct._cpu_created(
                json.loads(lines[1]),
                purpose="stage",
                name=NAME,
                plan_sha256=proof["plan_sha256"],
                manifest_sha256=proof["manifest_sha256"],
                authorization_sha256=core["sha256"],
            )
        except json.JSONDecodeError as exc:
            raise ValueError("Miles image-preflight create receipt is invalid") from exc

    result = direct._kubectl(
        runner,
        direct.PROD_CONTEXT,
        "get",
        "job",
        NAME,
        "--ignore-not-found",
        "-o",
        "json",
    )
    if result.returncode:
        raise ValueError("Miles image-preflight exact-name reconciliation failed")
    if not result.stdout.strip():
        return direct._seal(
            {
                "schema": RECONCILED_CREATE_SCHEMA,
                "status": "not_observed_after_consumed_intent",
                "name": NAME,
                "plan_sha256": proof["plan_sha256"],
                "manifest_sha256": proof["manifest_sha256"],
                "recovery_observer_required": False,
            }
        )
    try:
        job = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Miles image-preflight exact-name response is invalid") from exc
    admitted = job.get("spec", {}).get("suspend") is False
    workload = None
    if admitted:
        workload_result = direct._kubectl(
            runner,
            direct.PROD_CONTEXT,
            "get",
            "workload",
            "--selector",
            f"kueue.x-k8s.io/job-uid={job.get('metadata', {}).get('uid')}",
            "-o",
            "json",
        )
        if workload_result.returncode:
            raise ValueError("Miles image-preflight Workload reconciliation failed")
        try:
            workload = _validate_admitted_workload(
                SimpleNamespace(job=package.job), job, json.loads(workload_result.stdout)
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Miles image-preflight admitted identity is invalid") from exc
    observed = validate_response(
        job,
        package,
        require_uid=True,
        admitted=admitted,
        admitted_workload=workload,
    )
    created_at = job.get("metadata", {}).get("creationTimestamp")
    direct._timestamp(created_at)
    return direct._seal(
        {
            "schema": RECONCILED_CREATE_SCHEMA,
            "status": "observed_exact_job_requires_recovery_observer",
            "name": NAME,
            "plan_sha256": proof["plan_sha256"],
            "manifest_sha256": proof["manifest_sha256"],
            "job_uid": observed["job_uid"],
            "created_at": created_at,
            "admitted": admitted,
            "recovery_observer_required": True,
        }
    )


def validate_release(
    release: dict[str, Any],
    package: MilesImagePreflightPackage,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Validate direct or exact-UID recovery cleanup through one shared contract."""
    from training import dev_cleanup_observer as cleanup
    from training import skyrl_prod9_direct as direct

    proof = validate_package(package)
    schema = release.get("schema") if isinstance(release, dict) else None
    if schema not in {cleanup.DIRECT_RESULT_SCHEMA, cleanup.RECOVERY_RESULT_SCHEMA}:
        raise ValueError("Miles image-preflight release schema changed")
    direct._validate_seal(release, schema)
    normalized = release
    if schema == cleanup.RECOVERY_RESULT_SCHEMA:
        normalized = direct._seal(
            {
                **{key: value for key, value in release.items() if key != "sha256"},
                "schema": cleanup.DIRECT_RESULT_SCHEMA,
            }
        )
    direct._cpu_release(
        normalized,
        receipt,
        name=NAME,
        plan_sha256=proof["plan_sha256"],
        manifest_sha256=proof["manifest_sha256"],
        fresh=False,
    )
    if schema == cleanup.RECOVERY_RESULT_SCHEMA and release.get(
        "recovered_existing_target_uid"
    ) != release.get("uid"):
        raise ValueError("Miles recovery release UID changed")
    return release
