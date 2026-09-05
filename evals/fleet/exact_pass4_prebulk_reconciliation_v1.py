"""Sealed-safe reconciliation for the exact easiest-100 pass@4 bulk release.

The observer never reads a transcript or score.  It validates the immutable
task inventory and both generation-5 canary chains, then checks only sanitized
session metadata, Kubernetes state, global execution claims, and output-root
absence.  A source observation is accepted by a second create-once CPU Job so
the final receipt can truthfully bind a terminal source Job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import stat
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v1 as bulk
from evals.fleet import self_hosted

OBSERVATION_SCHEMA = "fleet-exact-pass4-prebulk-observation-v1"
TERMINAL_SCHEMA = bulk.RECONCILIATION_GATE_SCHEMA
CANARY_GATE_SCHEMA = bulk.CANARY_GATE_SCHEMA
PACKAGE_SCHEMA = "fleet-exact-pass4-prebulk-reconciliation-package-v1"
RELEASE_SCHEMA = "fleet-exact-pass4-prebulk-reconciliation-release-v1"
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-prebulk-reconciliation-held-v1.json"
)
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-exact-pass4-prebulk-reconciliation-release-v1.json"
)
NAMESPACE = "fleet-train-jobs"
SOURCE_JOB = "chris-cyber-exact100-prebulk-reconcile-source-v1"
ACCEPT_JOB = "chris-cyber-exact100-prebulk-reconcile-accept-v1"
OUTPUT_ROOT = Path("/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v1")
INVENTORY_TERMINAL = Path("/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json")
INVENTORY_JOB = "chris-cyber-exact100-pass4-inventory-v2"
INVENTORY_CONFIGMAP = "chris-cyber-exact100-pass4-inventory-bootstrap-v2"
INVENTORY_INTENT = "chris-cyber-exact100-pass4-inventory-intent-v2"
INVENTORY_PACKAGE_FILENAME = "exact100-inventory-package.json"
SFS_ROOT = Path("/mnt/sfs")
G5_OUTPUTS = {
    "qwen3.8-27b": Path("/mnt/sfs/jobs/chris-q38-ac-r004-a1-g5-v1"),
    "glm-5.3": Path("/mnt/sfs/jobs/chris-glm53-ac-r013-a1-g5-v1"),
}
G5_RELEASE_PROJECTED = {
    "qwen3.8-27b": Path("/bootstrap/qwen-g5-release.json"),
    "glm-5.3": Path("/bootstrap/glm-g5-release.json"),
}
G5_RELEASE_REPO_PATHS = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-autocontinue-generation5-scoring-release-v1.json"
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-glm53-autocontinue-generation5-scoring-release-v1.json"
    ),
}
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ReconciliationError(RuntimeError):
    """A content-free reconciliation failure."""


def _evidence_path(path: Path, evidence_root: Path | None) -> Path:
    """Map one fixed logical SFS path into a sanitized local mirror."""
    if not path.is_absolute() or not path.is_relative_to(SFS_ROOT) or ".." in path.parts:
        raise ReconciliationError("evidence_logical_path_unsafe")
    if evidence_root is None:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(SFS_ROOT):
            raise ReconciliationError("evidence_path_escaped_sfs")
        return resolved
    root = evidence_root.resolve(strict=True)
    mapped = (root / path.relative_to(SFS_ROOT)).resolve(strict=True)
    if not mapped.is_relative_to(root):
        raise ReconciliationError("evidence_path_escaped_mirror")
    return mapped


def fixed_evidence_paths(root: Path) -> dict[str, Path]:
    """Return the complete content-safe SFS evidence allowlist for release rendering."""
    from evals.fleet import autocontinue_generation5_canary as generation5

    paths = {"inventory_terminal": INVENTORY_TERMINAL}
    for model in sorted(G5_OUTPUTS):
        spec = bulk.load(root / bulk.CANARY_SPEC_PATHS[model])
        execution_id = spec.get("execution", {}).get("execution_id")
        if not isinstance(execution_id, str) or SHA_RE.fullmatch(execution_id) is None:
            raise ReconciliationError("generation5_execution_id_invalid")
        paths[f"{model}_terminal"] = G5_OUTPUTS[model] / "CANARY-TERMINAL.json"
        paths[f"{model}_claim"] = Path(generation5.CLAIM_ROOT) / (
            execution_id.removeprefix("sha256:") + ".json"
        )
    if len(paths) != 5 or len(set(paths.values())) != 5:
        raise ReconciliationError("fixed_evidence_path_set_invalid")
    if any(not path.is_relative_to(SFS_ROOT) for path in paths.values()):
        raise ReconciliationError("fixed_evidence_path_outside_sfs")
    return paths


def validate_template_bindings(root: Path) -> None:
    """Cross-check every executable template against the frozen treatment identity."""
    campaign = bulk.exact.read_object(root / bulk.CAMPAIGN_PATH)
    bulk.exact.validate_campaign(campaign, root)
    expected_treatment = bulk.exact.EXPECTED_TREATMENT
    expected_endpoint = "https://inference.flt.build"
    plans = bulk.validate_all(root)
    for model, template_path in bulk.RUNTIME_TEMPLATE_PATHS.items():
        template = bulk.load(root / template_path)
        expected_model = {
            "endpoint_origin": expected_endpoint,
            **bulk.exact.EXPECTED_MODELS[model],
        }
        harness = template.get("harness")
        treatment = template.get("treatment_block")
        if not isinstance(harness, dict) or not isinstance(treatment, dict):
            raise ReconciliationError("runtime_template_treatment_absent")
        expected_harness = {
            "name": expected_treatment["harness"],
            "version": expected_treatment["harness_version"],
            "release_asset_sha256": expected_treatment["release_asset_sha256"],
            "provider_adapter": expected_treatment["provider_adapter"],
            "context_management": expected_treatment["context_management"],
            "context_window_size": expected_treatment["context_window_size"],
            "max_model_requests": expected_treatment["max_model_requests"],
            "max_output_tokens": expected_treatment["max_output_tokens"],
            "timeout_seconds": expected_treatment["timeout_seconds"],
            "settings_canonical_sha256": harness.get("settings_canonical_sha256"),
            "settings_file_sha256": harness.get("settings_file_sha256"),
        }
        render_input = json.loads(json.dumps(template))
        render_input["harness"]["compaction_headroom_tokens"] = expected_treatment[
            "compaction_headroom_tokens"
        ]
        settings = self_hosted.opencode_settings(render_input)
        canonical = self_hosted.canonical_json(settings)
        model_plans = [value for value in plans.values() if value["model"]["served_id"] == model]
        if (
            template.get("model") != expected_model
            or template.get("serving_block") != treatment.get("serving_block")
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("campaign_sha256")
            != (template.get("source") or {}).get("campaign_sha256")
            or treatment.get("endpoint_origin") != expected_endpoint
            or treatment.get("served_id") != expected_model["served_id"]
            or treatment.get("model_revision") != expected_model["revision"]
            or treatment.get("session_model") != expected_model["session_model"]
            or treatment.get("harness") != harness
            or harness != expected_harness
            or treatment.get("required_task_tools") != expected_treatment["tools"]
            or treatment.get("required_task_tool_catalog_sha256")
            != expected_treatment["tool_catalog_sha256"]
            or harness.get("settings_canonical_sha256") != self_hosted.sha256(canonical)
            or harness.get("settings_file_sha256") != self_hosted.sha256(canonical + b"\n")
            or settings.get("compaction")
            != {"auto": True, "reserved": expected_treatment["compaction_headroom_tokens"]}
            or len(model_plans) != 2
            or any(
                plan["model"] != bulk.exact.EXPECTED_MODELS[model]
                or plan["serving_block"] != template["serving_block"]
                or plan["treatment"] != expected_treatment
                for plan in model_plans
            )
        ):
            raise ReconciliationError("runtime_template_cross_field_drifted")


def _strict_json_loads(raw: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ReconciliationError("duplicate_json_key")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationError("invalid_json_response") from exc


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise ReconciliationError("cross_origin_redirect_rejected")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ReconciliationError(f"{label}_invalid")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ReconciliationError(f"{label}_invalid") from exc


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReconciliationError("evidence_path_unsafe_or_absent")
    return bulk.load(path)


def _file_sha(path: Path) -> str:
    return bulk.file_sha256(path)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": bulk.digest(value, "receipt_sha256")}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    _write_bytes_once(path, self_hosted.canonical_json(value) + b"\n")


def _write_bytes_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReconciliationError("receipt_target_unsafe")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def remaining_cells(root: Path) -> list[dict[str, Any]]:
    """Return the exact 798 sanitized cells, independent of live state."""
    plans = bulk.validate_all(root)
    rows = [
        {
            "model": plan["model"]["served_id"],
            "controller": controller,
            "selection_rank": item["selection_rank"],
            "attempt": item["attempt"],
            "task_version_id": item["task_version_id"],
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "run_id": item["run_id"],
        }
        for controller, plan in plans.items()
        for item in plan["attempts"]
    ]
    rows.sort(key=lambda row: (row["model"], row["selection_rank"], row["attempt"]))
    if len(rows) != 798 or len({row["cell_id"] for row in rows}) != 798:
        raise ReconciliationError("remaining_cell_partition_invalid")
    return rows


def validate_held(root: Path) -> dict[str, Any]:
    receipt = _load(root / HELD_PATH)
    if (
        receipt.get("schema_version") != "fleet-exact-pass4-prebulk-reconciliation-held-v1"
        or receipt.get("status") != "HELD"
        or receipt.get("planned_cell_count") != 800
        or receipt.get("required_generation5_accepted_cells") != 2
        or receipt.get("expected_remaining_cells") != 798
        or receipt.get("release", {}).get("launch_authorized") is not False
        or receipt.get("privacy")
        != {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        or receipt.get("receipt_sha256") != bulk.digest(receipt, "receipt_sha256")
    ):
        raise ReconciliationError("prebulk_held_receipt_invalid")
    return receipt


def build_release(
    root: Path, package_commit: str, *, evidence_root: Path | None = None
) -> dict[str, Any]:
    """Render a deterministic observer-only release after prerequisites exist."""
    from evals.fleet import exact_pass4_prebulk_reconciliation_package_v1 as package

    if bulk.COMMIT_RE.fullmatch(package_commit) is None:
        raise ReconciliationError("prebulk_package_commit_invalid")
    held = validate_held(root)
    validate_template_bindings(root)
    inventory_path = _evidence_path(INVENTORY_TERMINAL, evidence_root)
    inventory_receipt = _load(inventory_path)
    bulk.validate_inventory_gate(inventory_receipt, root)
    canaries = {
        model: _canary_chain(model, root, evidence_root=evidence_root) for model in G5_OUTPUTS
    }
    built = package.build_package(root)
    value = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "launch_authorized": True,
        "bulk_launch_authorized": False,
        "campaign_id": bulk.exact.EXPECTED_CAMPAIGN_ID,
        "package_commit": package_commit,
        "package": {
            "schema_version": package.SCHEMA,
            "aggregate_sha256": built["aggregate_sha256"],
            "object_json_bytes": built["object_json_bytes"],
            "objects": sorted(built["configmaps"]),
            "release_included": False,
        },
        "held_authority": {
            "path": str(HELD_PATH),
            "file_sha256": _file_sha(root / HELD_PATH),
            "receipt_sha256": held["receipt_sha256"],
        },
        "exact100_inventory": {
            "path": str(INVENTORY_TERMINAL),
            "file_sha256": _file_sha(inventory_path),
            "receipt_sha256": inventory_receipt["receipt_sha256"],
        },
        "generation5": {
            model: {
                "cell_id": chain["spec"]["statistical_cell"]["cell_id"],
                "execution_id": chain["spec"]["execution"]["execution_id"],
                "claim_file_sha256": _file_sha(chain["claim_path"]),
                "claim_receipt_sha256": chain["claim"]["receipt_sha256"],
                "terminal_file_sha256": _file_sha(chain["terminal_path"]),
                "terminal_receipt_sha256": chain["terminal"]["receipt_sha256"],
                "scoring_release_file_sha256": _file_sha(chain["release_path"]),
                "scoring_release_receipt_sha256": chain["release"]["receipt_sha256"],
                "session_id": chain["session_id"],
                "verifier_execution_id": chain["verifier_execution_id"],
            }
            for model, chain in canaries.items()
        },
        "authorization": {
            "source_job": SOURCE_JOB,
            "accept_job": ACCEPT_JOB,
            "create_once": True,
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "automatic_retry": False,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    return _seal(value)


def render_release_from_commit(
    root: Path, package_commit: str, *, evidence_root: Path | None = None
) -> dict[str, Any]:
    """Render using only bytes from an exact clean checked-out package commit."""
    from evals.fleet import immutable_submission_snapshot as snapshot

    snapshot.assert_stable(root, package_commit)
    with tempfile.TemporaryDirectory(prefix="prebulk-release-") as directory:
        committed_root = Path(directory) / "package"
        snapshot.materialize(root, package_commit, committed_root)
        if (committed_root / RELEASE_PATH).exists() or (committed_root / RELEASE_PATH).is_symlink():
            raise ReconciliationError("prebulk_release_is_circularly_packaged")
        return build_release(committed_root, package_commit, evidence_root=evidence_root)


def validate_release(
    receipt: dict[str, Any], root: Path, *, evidence_root: Path | None = None
) -> None:
    package_commit = receipt.get("package_commit")
    if not isinstance(package_commit, str) or receipt != build_release(
        root, package_commit, evidence_root=evidence_root
    ):
        raise ReconciliationError("prebulk_release_invalid")


def _install_projected_releases(root: Path) -> None:
    """Install the exact raw G5 releases projected from immutable runtime CMs."""
    for model, projected in G5_RELEASE_PROJECTED.items():
        raw = projected.read_bytes()
        declared_path = projected.with_name(projected.stem + "-file-sha256")
        package_path = projected.with_name(projected.stem + "-package-commit")
        declared = declared_path.read_text().strip()
        release = _load(projected)
        if not isinstance(release, dict) or _file_sha(projected) != declared:
            raise ReconciliationError("projected_generation5_release_drifted")
        if release.get("package_commit") != package_path.read_text().strip():
            raise ReconciliationError("projected_generation5_package_commit_drifted")
        target = root / G5_RELEASE_REPO_PATHS[model]
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)


def _canary_chain(
    model: str, root: Path, *, evidence_root: Path | None = None
) -> dict[str, Any]:
    from evals.fleet import autocontinue_generation5_authority_v1 as authority
    from evals.fleet import autocontinue_generation5_canary as generation5

    spec = authority.load(root / bulk.CANARY_SPEC_PATHS[model])
    plan = generation5.validate_spec(spec, root)
    logical_terminal_path = G5_OUTPUTS[model] / "CANARY-TERMINAL.json"
    terminal_path = _evidence_path(logical_terminal_path, evidence_root)
    terminal = _load(terminal_path)
    release_path = root / G5_RELEASE_REPO_PATHS[model]
    release = authority.load(release_path)
    package_commit = release.get("package_commit")
    if not isinstance(package_commit, str):
        raise ReconciliationError("generation5_package_commit_invalid")
    logical_claim_path = Path(generation5.CLAIM_ROOT) / (
        spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
    )
    claim_path = _evidence_path(logical_claim_path, evidence_root)
    claim = _load(claim_path)
    root_authority = authority.load(root / authority.AUTH_PATH)
    authority.validate_terminal(
        terminal,
        spec,
        plan,
        claim,
        release,
        _file_sha(release_path),
        root_authority,
        root,
        package_commit,
    )
    result = terminal.get("result") or {}
    if any(
        (
            result.get("accepted") is not True,
            result.get("quarantined") is not False,
            result.get("session_ingest_completed") is not True,
            result.get("cleanup_completed") is not True,
        )
    ):
        raise ReconciliationError("generation5_canary_not_accepted")
    return {
        "model": model,
        "spec": spec,
        "plan": plan,
        "claim": claim,
        "claim_path": claim_path,
        "logical_claim_path": logical_claim_path,
        "terminal": terminal,
        "terminal_path": terminal_path,
        "logical_terminal_path": logical_terminal_path,
        "release_path": release_path,
        "release": release,
        "package_commit": package_commit,
        "session_id": _uuid(result.get("session_id"), "generation5_session_id"),
        "verifier_execution_id": _uuid(
            result.get("verifier_execution_id"), "generation5_verifier_execution_id"
        ),
    }


def _default_sessions(task_key: str, api_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode({"task_key": task_key, "limit": 500, "offset": offset})
        request = urllib.request.Request(
            f"https://orchestrator.fleetai.com/v1/sessions?{params}",
            method="GET",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        with urllib.request.build_opener(_RejectRedirects()).open(request, timeout=60) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            raise ReconciliationError("fleet_session_page_too_large")
        value = _strict_json_loads(payload)
        page = value.get("sessions") if isinstance(value, dict) else None
        if not isinstance(page, list) or not all(isinstance(row, dict) for row in page):
            raise ReconciliationError("fleet_session_page_invalid")
        rows.extend(page)
        if value.get("has_more") is False:
            return rows
        if not page:
            raise ReconciliationError("fleet_session_pagination_stalled")
        offset += len(page)


def _default_kubernetes(kind: str, name: str | None = None) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host:
        raise ReconciliationError("kubernetes_service_unavailable")
    if kind == "job":
        path = f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{name}"
    elif kind == "pods":
        selector = urllib.parse.quote(f"job-name={name}", safe="")
        path = f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}"
    elif kind == "configmap":
        path = f"/api/v1/namespaces/{NAMESPACE}/configmaps/{name}"
    else:
        raise ReconciliationError("kubernetes_query_invalid")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    request = urllib.request.Request(
        f"https://{host}:{port}{path}",
        method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    context = ssl.create_default_context(
        cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    opener = urllib.request.build_opener(
        _RejectRedirects(), urllib.request.HTTPSHandler(context=context)
    )
    with opener.open(request, timeout=30) as response:
        payload = response.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise ReconciliationError("kubernetes_response_too_large")
    value = _strict_json_loads(payload)
    if not isinstance(value, dict):
        raise ReconciliationError("kubernetes_response_invalid")
    return value


def _succeeded_job(
    job_name: str,
    expected_job_uid: str,
    expected_pod_uid: str,
    kubernetes: Callable[[str, str | None], dict[str, Any]],
) -> dict[str, Any]:
    job = kubernetes("job", job_name)
    pods = kubernetes("pods", job_name).get("items")
    conditions = (job.get("status") or {}).get("conditions") or []
    if (
        _uuid((job.get("metadata") or {}).get("uid"), "job_uid") != expected_job_uid
        or not isinstance(pods, list)
        or len(pods) != 1
        or _uuid((pods[0].get("metadata") or {}).get("uid"), "pod_uid") != expected_pod_uid
        or (pods[0].get("status") or {}).get("phase") != "Succeeded"
        or not any(
            row.get("type") == "Complete" and row.get("status") == "True"
            for row in conditions
            if isinstance(row, dict)
        )
        or any(
            row.get("type") == "Failed" and row.get("status") == "True"
            for row in conditions
            if isinstance(row, dict)
        )
    ):
        raise ReconciliationError("required_job_not_exclusively_complete")
    pod_status = pods[0].get("status") or {}
    statuses = [
        *(pod_status.get("initContainerStatuses") or []),
        *(pod_status.get("containerStatuses") or []),
    ]
    restarts = sum(row.get("restartCount", 0) for row in statuses if isinstance(row, dict))
    terminated = [
        (row.get("state") or {}).get("terminated") for row in statuses if isinstance(row, dict)
    ]
    if (
        restarts != 0
        or not terminated
        or any(not isinstance(state, dict) or state.get("exitCode") != 0 for state in terminated)
    ):
        raise ReconciliationError("required_pod_not_clean_success")
    return {"job_uid": expected_job_uid, "pod_uid": expected_pod_uid, "pod_restarts": 0}


def _inventory_package_authority(
    inventory_receipt: dict[str, Any],
    kubernetes: Callable[[str, str | None], dict[str, Any]],
) -> dict[str, Any]:
    """Validate immutable live inventory-v2 producer bytes and terminal runtime."""
    from evals.fleet import exact_pass4_task_inventory_package as package

    runtime = inventory_receipt.get("runtime")
    if not isinstance(runtime, dict):
        raise ReconciliationError("inventory_runtime_absent")
    job_state = _succeeded_job(
        INVENTORY_JOB,
        _uuid(runtime.get("job_uid"), "inventory_job_uid"),
        _uuid(runtime.get("pod_uid"), "inventory_pod_uid"),
        kubernetes,
    )
    configmap = kubernetes("configmap", INVENTORY_CONFIGMAP)
    intent = kubernetes("configmap", INVENTORY_INTENT)
    config_metadata = configmap.get("metadata") or {}
    intent_metadata = intent.get("metadata") or {}
    data = configmap.get("data")
    package_text = data.get("package.json") if isinstance(data, dict) else None
    if (
        config_metadata.get("name") != INVENTORY_CONFIGMAP
        or config_metadata.get("namespace") != NAMESPACE
        or configmap.get("immutable") is not True
        or not isinstance(package_text, str)
    ):
        raise ReconciliationError("inventory_bootstrap_configmap_invalid")
    package_receipt = package._load_object(package_text)  # noqa: SLF001
    package.validate_package_manifest(package_receipt)
    package_commit = package_receipt["package_commit"]
    for key, binding in package_receipt["files"].items():
        raw = data.get(key)
        if not isinstance(raw, str) or package.sha256(raw.encode()) != binding["sha256"]:
            raise ReconciliationError("inventory_bootstrap_payload_drifted")
    if data.get("package_commit") != package_commit:
        raise ReconciliationError("inventory_bootstrap_commit_drifted")
    expected_intent = package.build_intent(configmap)
    if (
        intent_metadata.get("name") != INVENTORY_INTENT
        or intent_metadata.get("namespace") != NAMESPACE
        or intent.get("immutable") is not True
        or intent.get("data") != expected_intent["data"]
    ):
        raise ReconciliationError("inventory_intent_configmap_invalid")
    if package_receipt["runtime_contract"]["sfs_terminal_path"] != str(INVENTORY_TERMINAL):
        raise ReconciliationError("inventory_package_terminal_binding_drifted")
    raw_package = package_text.encode()
    return {
        "raw": raw_package,
        "binding": {
            "path": str(OUTPUT_ROOT / INVENTORY_PACKAGE_FILENAME),
            "file_sha256": package.sha256(raw_package),
            "package_sha256": package_receipt["package_sha256"],
            "package_commit": package_commit,
            "bootstrap_configmap": INVENTORY_CONFIGMAP,
            "bootstrap_configmap_uid": _uuid(config_metadata.get("uid"), "inventory_configmap_uid"),
            "intent_configmap": INVENTORY_INTENT,
            "intent_configmap_uid": _uuid(intent_metadata.get("uid"), "inventory_intent_uid"),
            "producer_runtime": job_state,
        },
    }


def _assert_absent_jobs(
    plans: dict[str, dict[str, Any]],
    kubernetes: Callable[[str, str | None], dict[str, Any]],
) -> None:
    for plan in plans.values():
        try:
            job = kubernetes("job", plan["job_name"])
        except Exception as exc:
            if getattr(exc, "code", None) == 404 or "not_found" in str(exc).lower():
                job = {}
            else:
                raise
        pods = kubernetes("pods", plan["job_name"]).get("items")
        if job or pods:
            raise ReconciliationError("bulk_kubernetes_collision")


def collect(
    root: Path,
    *,
    job_uid: str,
    pod_uid: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] = _default_sessions,
    kubernetes: Callable[[str, str | None], dict[str, Any]] = _default_kubernetes,
    api_key: str | None = None,
    observed_at: str | None = None,
    evidence_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect a score-blind snapshot; raise instead of emitting an unsafe CLEAR."""
    job_uid = _uuid(job_uid, "observer_job_uid")
    pod_uid = _uuid(pod_uid, "observer_pod_uid")
    key = api_key or os.environ.get("FLEET_API_KEY")
    if not key:
        raise ReconciliationError("fleet_api_key_absent")
    validate_template_bindings(root)
    inventory_path = _evidence_path(INVENTORY_TERMINAL, evidence_root)
    inventory_receipt = _load(inventory_path)
    bulk.validate_inventory_gate(inventory_receipt, root)
    inventory_package = _inventory_package_authority(inventory_receipt, kubernetes)
    canaries = {
        model: _canary_chain(model, root, evidence_root=evidence_root) for model in G5_OUTPUTS
    }
    plans = bulk.validate_all(root)
    rows = remaining_cells(root)
    run_ids = {row["run_id"] for row in rows}
    execution_ids = {row["execution_id"] for row in rows}
    cell_ids = {row["cell_id"] for row in rows}

    for chain in canaries.values():
        _succeeded_job(
            chain["plan"]["scored_job_name"],
            chain["claim"]["job_uid"],
            chain["claim"]["pod_uid"],
            kubernetes,
        )
    _assert_absent_jobs(plans, kubernetes)
    for plan in plans.values():
        path = Path(plan["sfs_root"])
        if path.exists() or path.is_symlink():
            raise ReconciliationError("bulk_sfs_output_collision")
    for execution_id in execution_ids:
        claim = Path(bulk.CLAIM_ROOT) / (execution_id.removeprefix("sha256:") + ".json")
        if claim.exists() or claim.is_symlink():
            raise ReconciliationError("bulk_global_claim_collision")

    tasks = {row["task"]["key"] for row in inventory_receipt["tasks"]}
    canary_sessions = {chain["session_id"]: chain for chain in canaries.values()}
    seen_canaries: set[str] = set()
    collision_ids: list[str] = []
    page_rows = 0
    for task_key in sorted(tasks):
        for row in sessions(task_key, key):
            page_rows += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            session_id = row.get("session_id")
            if session_id in canary_sessions:
                chain = canary_sessions[session_id]
                verifier = row.get("verifier_execution") or {}
                expected_attempt = chain["plan"]["attempts"][0]
                projected_version = row.get("eval_task_version_id") or row.get("task_version_id")
                projected_run = metadata.get("run_id")
                if (
                    row.get("status") != "completed"
                    or row.get("model")
                    != bulk.exact.EXPECTED_MODELS[chain["model"]]["session_model"]
                    or verifier.get("id") != chain["verifier_execution_id"]
                    or projected_version != chain["spec"]["statistical_cell"]["task_version_id"]
                    or projected_run != expected_attempt["run_id"]
                ):
                    raise ReconciliationError("generation5_authoritative_session_drifted")
                seen_canaries.add(session_id)
            if (
                metadata.get("run_id") in run_ids
                or metadata.get("execution_id") in execution_ids
                or metadata.get("cell_id") in cell_ids
            ):
                collision_ids.append(str(session_id))
    if set(canary_sessions) != seen_canaries:
        raise ReconciliationError("generation5_authoritative_session_absent")
    if collision_ids:
        raise ReconciliationError("bulk_fleet_session_collision")

    timestamp = observed_at or _now()
    if UTC_RE.fullmatch(timestamp) is None:
        raise ReconciliationError("observation_timestamp_invalid")
    execution_id_list = sorted(execution_ids)
    receipt = _seal(
        {
            "schema_version": OBSERVATION_SCHEMA,
            "status": "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE",
            "observed_at_utc": timestamp,
            "campaign_id": bulk.exact.EXPECTED_CAMPAIGN_ID,
            "universe_sha256": bulk._universe(root)["universe_sha256"],
            "exact100_inventory": {
                "path": str(INVENTORY_TERMINAL),
                "file_sha256": _file_sha(inventory_path),
                "receipt_sha256": inventory_receipt["receipt_sha256"],
            },
            "exact100_inventory_package": inventory_package["binding"],
            "generation5": {
                model: {
                    "cell_id": chain["spec"]["statistical_cell"]["cell_id"],
                    "execution_id": chain["spec"]["execution"]["execution_id"],
                    "session_id": chain["session_id"],
                    "verifier_execution_id": chain["verifier_execution_id"],
                    "claim_receipt_sha256": chain["claim"]["receipt_sha256"],
                    "terminal_receipt_sha256": chain["terminal"]["receipt_sha256"],
                    "scoring_release_file_sha256": _file_sha(chain["release_path"]),
                    "scoring_release_receipt_sha256": chain["release"]["receipt_sha256"],
                }
                for model, chain in canaries.items()
            },
            "remaining_cells": rows,
            "remaining_cells_sha256": self_hosted.sha256(self_hosted.canonical_json(rows)),
            "planned_execution_count": 798,
            "planned_execution_ids_sha256": self_hosted.sha256(
                self_hosted.canonical_json(execution_id_list)
            ),
            "collisions": {
                "fleet_api": 0,
                "kubernetes_job_or_pod": 0,
                "sfs_output": 0,
                "global_claim": 0,
                "active_or_accepted_cell": 0,
            },
            "request_counts": {
                "task_session_queries": len(tasks),
                "session_rows_examined": page_rows,
                "transcript_queries": 0,
                "mutation_calls": 0,
            },
            "checked_job_names": sorted(plan["job_name"] for plan in plans.values()),
            "checked_output_roots": sorted(plan["sfs_root"] for plan in plans.values()),
            "runtime": {"job_uid": job_uid, "pod_uid": pod_uid},
            "privacy": {
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            },
        }
    )
    return receipt, canaries, inventory_package


def validate_observation(receipt: dict[str, Any], root: Path) -> None:
    rows = remaining_cells(root)
    plans = bulk.validate_all(root)
    execution_ids = sorted(row["execution_id"] for row in rows)
    inventory = receipt.get("exact100_inventory")
    inventory_package = receipt.get("exact100_inventory_package")
    generation5 = receipt.get("generation5")
    request_counts = receipt.get("request_counts")
    if (
        receipt.get("schema_version") != OBSERVATION_SCHEMA
        or receipt.get("status") != "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE"
        or receipt.get("campaign_id") != bulk.exact.EXPECTED_CAMPAIGN_ID
        or receipt.get("universe_sha256") != bulk._universe(root)["universe_sha256"]
        or receipt.get("remaining_cells") != rows
        or receipt.get("remaining_cells_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(rows))
        or receipt.get("planned_execution_count") != 798
        or receipt.get("planned_execution_ids_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(execution_ids))
        or receipt.get("checked_job_names") != sorted(plan["job_name"] for plan in plans.values())
        or receipt.get("checked_output_roots")
        != sorted(plan["sfs_root"] for plan in plans.values())
        or not isinstance(inventory, dict)
        or inventory.get("path") != str(INVENTORY_TERMINAL)
        or SHA_RE.fullmatch(str(inventory.get("file_sha256"))) is None
        or SHA_RE.fullmatch(str(inventory.get("receipt_sha256"))) is None
        or not isinstance(inventory_package, dict)
        or set(inventory_package)
        != {
            "path",
            "file_sha256",
            "package_sha256",
            "package_commit",
            "bootstrap_configmap",
            "bootstrap_configmap_uid",
            "intent_configmap",
            "intent_configmap_uid",
            "producer_runtime",
        }
        or inventory_package.get("path") != str(OUTPUT_ROOT / INVENTORY_PACKAGE_FILENAME)
        or SHA_RE.fullmatch(str(inventory_package.get("file_sha256"))) is None
        or SHA_RE.fullmatch(str(inventory_package.get("package_sha256"))) is None
        or bulk.COMMIT_RE.fullmatch(str(inventory_package.get("package_commit"))) is None
        or inventory_package.get("bootstrap_configmap") != INVENTORY_CONFIGMAP
        or inventory_package.get("intent_configmap") != INVENTORY_INTENT
        or not isinstance(generation5, dict)
        or set(generation5) != set(G5_OUTPUTS)
        or any(
            not isinstance(value, dict)
            or set(value)
            != {
                "cell_id",
                "execution_id",
                "session_id",
                "verifier_execution_id",
                "claim_receipt_sha256",
                "terminal_receipt_sha256",
                "scoring_release_file_sha256",
                "scoring_release_receipt_sha256",
            }
            or SHA_RE.fullmatch(str(value.get("cell_id"))) is None
            or SHA_RE.fullmatch(str(value.get("execution_id"))) is None
            or SHA_RE.fullmatch(str(value.get("claim_receipt_sha256"))) is None
            or SHA_RE.fullmatch(str(value.get("terminal_receipt_sha256"))) is None
            or SHA_RE.fullmatch(str(value.get("scoring_release_file_sha256"))) is None
            or SHA_RE.fullmatch(str(value.get("scoring_release_receipt_sha256"))) is None
            for value in generation5.values()
        )
        or not isinstance(request_counts, dict)
        or request_counts.get("task_session_queries") != 100
        or not isinstance(request_counts.get("session_rows_examined"), int)
        or request_counts.get("session_rows_examined", -1) < 2
        or request_counts.get("transcript_queries") != 0
        or request_counts.get("mutation_calls") != 0
        or receipt.get("collisions")
        != {
            "fleet_api": 0,
            "kubernetes_job_or_pod": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "active_or_accepted_cell": 0,
        }
        or receipt.get("privacy")
        != {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        or receipt.get("receipt_sha256") != bulk.digest(receipt, "receipt_sha256")
        or UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None
    ):
        raise ReconciliationError("source_observation_invalid")
    _uuid((receipt.get("runtime") or {}).get("job_uid"), "source_job_uid")
    _uuid((receipt.get("runtime") or {}).get("pod_uid"), "source_pod_uid")
    _uuid(inventory_package.get("bootstrap_configmap_uid"), "inventory_configmap_uid")
    _uuid(inventory_package.get("intent_configmap_uid"), "inventory_intent_uid")
    inventory_runtime = inventory_package.get("producer_runtime") or {}
    _uuid(inventory_runtime.get("job_uid"), "inventory_job_uid")
    _uuid(inventory_runtime.get("pod_uid"), "inventory_pod_uid")
    if inventory_runtime.get("pod_restarts") != 0:
        raise ReconciliationError("inventory_producer_restarted")
    for value in generation5.values():
        _uuid(value.get("session_id"), "generation5_session_id")
        _uuid(value.get("verifier_execution_id"), "generation5_verifier_execution_id")


def _canary_gate(
    chain: dict[str, Any], collector_job_uid: str, collector_pod_uid: str, timestamp: str
) -> dict[str, Any]:
    model = chain["model"]
    copied_claim = OUTPUT_ROOT / f"{model}-generation5-claim.json"
    copied_release = OUTPUT_ROOT / f"{model}-generation5-release.json"
    _write_once(copied_claim, chain["claim"])
    _write_bytes_once(copied_release, chain["release_path"].read_bytes())
    if _file_sha(copied_release) != _file_sha(chain["release_path"]):
        raise ReconciliationError("generation5_release_copy_drifted")
    value = {
        "schema_version": CANARY_GATE_SCHEMA,
        "status": "ACCEPTED",
        "model": model,
        "generation5_spec_sha256": chain["spec"]["generation5_spec_sha256"],
        "plan_sha256": chain["plan"]["plan_sha256"],
        "execution_id": chain["spec"]["execution"]["execution_id"],
        "job_name": chain["plan"]["scored_job_name"],
        "run_id": chain["plan"]["attempts"][0]["run_id"],
        "terminal_file_sha256": _file_sha(chain["terminal_path"]),
        "terminal_receipt_sha256": chain["terminal"]["receipt_sha256"],
        "session_id": chain["session_id"],
        "verifier_execution_id": chain["verifier_execution_id"],
        "job_succeeded": True,
        "pod_restarts": 0,
        "lease_probe_acquired_and_released": True,
        "claim_job_uid": chain["claim"]["job_uid"],
        "claim_pod_uid": chain["claim"]["pod_uid"],
        "observer_runtime": {
            "job_uid": collector_job_uid,
            "pod_uid": collector_pod_uid,
        },
        "observed_at_utc": timestamp,
        "evidence": {
            "claim_path": str(copied_claim),
            "terminal_path": str(chain["logical_terminal_path"]),
            "release_path": str(copied_release),
            "authority_path": (
                "docs/evidence/qwen38-study/"
                "2026-09-05-opencode-autocontinue-generation5-root-authorization-v1.json"
            ),
            "package_commit": chain["package_commit"],
        },
        "prompts_traces_flags_or_scores_included": False,
    }
    return _seal(value)


def accept(
    root: Path,
    observation: dict[str, Any],
    *,
    collector_job_uid: str,
    collector_pod_uid: str,
    package_commit: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] = _default_sessions,
    kubernetes: Callable[[str, str | None], dict[str, Any]] = _default_kubernetes,
    api_key: str | None = None,
    observed_at: str | None = None,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Recheck live state and accept only an exclusively Complete source Job."""
    validate_observation(observation, root)
    source_runtime = observation["runtime"]
    source_state = _succeeded_job(
        SOURCE_JOB, source_runtime["job_uid"], source_runtime["pod_uid"], kubernetes
    )
    fresh, canaries, inventory_package = collect(
        root,
        job_uid=collector_job_uid,
        pod_uid=collector_pod_uid,
        sessions=sessions,
        kubernetes=kubernetes,
        api_key=api_key,
        observed_at=observed_at,
        evidence_root=evidence_root,
    )
    if (
        fresh["exact100_inventory"] != observation["exact100_inventory"]
        or fresh["exact100_inventory_package"] != observation["exact100_inventory_package"]
        or fresh["generation5"] != observation["generation5"]
    ):
        raise ReconciliationError("source_observation_authority_drifted")
    timestamp = fresh["observed_at_utc"]
    inventory_package_path = OUTPUT_ROOT / INVENTORY_PACKAGE_FILENAME
    _write_bytes_once(inventory_package_path, inventory_package["raw"])
    if _file_sha(inventory_package_path) != fresh["exact100_inventory_package"]["file_sha256"]:
        raise ReconciliationError("inventory_package_copy_drifted")
    gates = {
        model: _canary_gate(chain, collector_job_uid, collector_pod_uid, timestamp)
        for model, chain in canaries.items()
    }
    for model, value in gates.items():
        _write_once(OUTPUT_ROOT / f"{model}-generation5-gate.json", value)
    terminal = _seal(
        {
            "schema_version": TERMINAL_SCHEMA,
            "status": "CLEAR",
            "observed_at_utc": timestamp,
            "observer_package_commit": package_commit,
            "planned_execution_count": 798,
            "planned_execution_ids_sha256": fresh["planned_execution_ids_sha256"],
            "remaining_cells": fresh["remaining_cells"],
            "remaining_cells_sha256": fresh["remaining_cells_sha256"],
            "fleet_api_collisions": 0,
            "kubernetes_job_or_pod_collisions": 0,
            "sfs_output_collisions": 0,
            "global_claim_collisions": 0,
            "active_or_accepted_cell_collisions": 0,
            "checked_immediately_before_release": True,
            "observer_job_succeeded": True,
            "observer_pod_restarts": source_state["pod_restarts"],
            "methods": ["GET"],
            "mutation_calls": 0,
            "checked_job_names": fresh["checked_job_names"],
            "checked_output_roots": fresh["checked_output_roots"],
            "runtime": source_runtime,
            "collector_runtime": {
                "job_uid": _uuid(collector_job_uid, "collector_job_uid"),
                "pod_uid": _uuid(collector_pod_uid, "collector_pod_uid"),
            },
            "source_observation": {
                "path": str(OUTPUT_ROOT / "OBSERVATION.json"),
                "file_sha256": _file_sha(OUTPUT_ROOT / "OBSERVATION.json"),
                "receipt_sha256": observation["receipt_sha256"],
            },
            "exact100_inventory": fresh["exact100_inventory"],
            "exact100_inventory_package": fresh["exact100_inventory_package"],
            "generation5_gate_receipts": {
                model: {
                    "path": str(OUTPUT_ROOT / f"{model}-generation5-gate.json"),
                    "receipt_sha256": value["receipt_sha256"],
                }
                for model, value in gates.items()
            },
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    bulk.validate_reconciliation_gate(terminal, root)
    return terminal


def main() -> int:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("render-release", "validate-release", "source", "accept", "validate"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="local directory mirroring logical /mnt/sfs paths for sanitized evidence",
    )
    parser.add_argument("--package-commit")
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root
    if args.command == "render-release":
        if not args.package_commit:
            parser.error("render-release requires --package-commit")
        print(
            json.dumps(
                render_release_from_commit(
                    args.root, args.package_commit, evidence_root=args.evidence_root
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "validate-release":
        validate_release(
            _load(args.root / RELEASE_PATH), args.root, evidence_root=args.evidence_root
        )
        return 0
    if args.command == "validate":
        validate_observation(_load(args.output_root / "OBSERVATION.json"), args.root)
        bulk.validate_reconciliation_gate(_load(args.output_root / "TERMINAL.json"), args.root)
        return 0
    _install_projected_releases(args.root)
    validate_release(_load(args.root / RELEASE_PATH), args.root, evidence_root=args.evidence_root)
    if args.command == "source":
        receipt, _, _ = collect(
            args.root,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            evidence_root=args.evidence_root,
        )
        _write_once(args.output_root / "OBSERVATION.json", receipt)
        return 0
    if not args.package_commit:
        parser.error("accept requires --package-commit")
    observation = _load(args.output_root / "OBSERVATION.json")
    terminal = accept(
        args.root,
        observation,
        collector_job_uid=os.environ["JOB_UID"],
        collector_pod_uid=os.environ["POD_UID"],
        package_commit=args.package_commit,
        evidence_root=args.evidence_root,
    )
    _write_once(args.output_root / "TERMINAL.json", terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
