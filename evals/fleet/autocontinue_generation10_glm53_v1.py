"""Create-once GLM Generation-10 canary runtime.

This is an independently releasable, single-cell successor.  The checked-in
state is held: a release and a fresh duplicate preflight must be supplied by a
reviewer before the runtime can execute.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation10_joint_preparer_v1 as preparer
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

MODEL = "glm-5.3"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-glm53-ac-r013-a1-g10-v1"
CONFIGMAP_NAME = "chris-glm53-ac-r013-a1-g10-v1-run-v1"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
SECRET_KEY = "FLEET_API_KEY"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
CLAIM_ROOT = Path(preparer.CLAIM_ROOT)
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-glm53-autocontinue-generation10-executable-held-v1.json"
)
TOMBSTONE_PATH = preparer.MODELS[MODEL]["tombstone"]
DIAGNOSIS_PATH = preparer.DIAGNOSIS_PATH
MODULE_PATH = "evals/fleet/autocontinue_generation10_glm53_v1.py"
PACKAGE_PATH = "evals/fleet/autocontinue_generation10_glm53_package_v1.py"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation10_glm53_v1.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation10_glm53_v1.sh"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation10-glm53-held-v1.yaml"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation10-glm53-release-v1"
DUPLICATE_SCHEMA = "fleet-opencode-autocontinue-generation10-glm53-duplicate-preflight-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v10"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation10-glm53-terminal-v1"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: Mapping[str, Any], field: str = "receipt_sha256") -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field}))


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent immutable input: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise ValueError(f"non-canonical immutable input: {path}")
    return value


def static(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the GLM treatment without requiring any Qwen tombstone."""
    preparer.assert_route_gate_implementation()
    preparer.assert_optimized_runtime_implementation(root)
    prepared = preparer.prepare_model(root, MODEL)
    if prepared.get("status") != "HELD_RELEASE_AND_DUPLICATE_PREFLIGHT_REQUIRED":
        raise ValueError("GLM Generation-10 pre-instance authority is not ready")
    spec = prepared["spec"]
    plan = prepared["plan"]
    held = load(root / HELD_PATH)
    expected_held = {
        "schema_version": "fleet-opencode-autocontinue-generation10-glm53-executable-held-v1",
        "status": "HELD",
        "launch_authorized": False,
        "objects_created": False,
        "model": MODEL,
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "output_root": OUTPUT_ROOT,
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation10_spec_sha256": spec["generation10_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "generation7_tombstone_receipt_sha256": prepared["tombstone_receipt_sha256"],
        "authority_diagnosis_receipt_sha256": spec["required_authority_gate"][
            "diagnosis_receipt_sha256"
        ],
        "runtime": {
            "validation_mode": "generation8_single_pass_immutable_validation",
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "create_once": True,
            "one_canary_only": True,
        },
        "secret": {
            "name": SECRET_NAME,
            "uid": SECRET_UID,
            "key": SECRET_KEY,
            "value_persisted": False,
        },
        "fleet_team_id": FLEET_TEAM_ID,
        "release_present": False,
        "fresh_duplicate_preflight_present": False,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    expected_held["receipt_sha256"] = digest(expected_held)
    if held != expected_held:
        raise ValueError("GLM Generation-10 executable held receipt drifted")
    return spec, plan, held


def validate_duplicate(
    value: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    now: datetime | None = None,
    maximum_age_seconds: int = 300,
) -> None:
    identities = value.get("identities", {})
    sfs = value.get("sfs", {})
    fleet = value.get("fleet", {})
    kubernetes = value.get("kubernetes", {})
    secret = value.get("secret", {})
    account = value.get("account", {})
    try:
        observed = datetime.fromisoformat(
            str(value.get("observed_at_utc", "")).removesuffix("Z") + "+00:00"
        )
        age = ((now or datetime.now(UTC)) - observed).total_seconds()
    except (TypeError, ValueError) as exc:
        raise ValueError("GLM Generation-10 duplicate preflight timestamp drifted") from exc
    expected_fleet = {
        f"generation{generation}_matching_session_rows": 0
        for generation in (8, 9, 10)
    }
    if (
        value.get("schema_version") != DUPLICATE_SCHEMA
        or value.get("status") != "FRESH_ABSENT"
        or value.get("model") != MODEL
        or value.get("cell_id") != spec["statistical_cell"]["cell_id"]
        or value.get("execution_id") != spec["execution"]["execution_id"]
        or identities
        != {
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "output_root": OUTPUT_ROOT,
        }
        or kubernetes
        != {
            "job_absent": True,
            "pods_absent": True,
            "configmap_absent": True,
            "checked_namespace": NAMESPACE,
        }
        or sfs.get("output_root_absent") is not True
        or sfs.get("generation8_claim_absent") is not True
        or sfs.get("generation9_claim_absent") is not True
        or sfs.get("generation10_claim_absent") is not True
        or sfs.get("generation7_claim_present") is not True
        or not isinstance(sfs.get("generation7_claim_receipt_sha256"), str)
        or SHA_RE.fullmatch(sfs["generation7_claim_receipt_sha256"]) is None
        or not isinstance(fleet.get("session_rows_examined"), int)
        or fleet.get("session_rows_examined", -1) < 0
        or {key: item for key, item in fleet.items() if key != "session_rows_examined"}
        != expected_fleet
        or secret
        != {
            "name": SECRET_NAME,
            "uid": SECRET_UID,
            "key_names": [SECRET_KEY],
            "value_read": False,
            "value_persisted": False,
        }
        or account != {"team_name": "fleet", "team_id": FLEET_TEAM_ID}
        or value.get("response_bodies_persisted") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
        or age < 0
        or age > maximum_age_seconds
    ):
        raise ValueError("GLM Generation-10 duplicate preflight drifted")


def _command(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("read-only duplicate observation command failed")
    return result.stdout.strip()


def _allie_path(path: str | Path) -> str:
    value = str(path)
    if not value.startswith("/mnt/sfs/"):
        raise ValueError("SFS path is outside the allowed observation root")
    return "/shared/" + value.removeprefix("/mnt/sfs/")


def _allie_exists(path: str | Path) -> bool:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            "allie-dev",
            "--",
            "sh",
            "-c",
            'test -e "$1" || test -L "$1"',
            "_",
            _allie_path(path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("read-only SFS existence observation failed")
    return result.returncode == 0


def _allie_claim(path: Path) -> dict[str, Any]:
    raw = _command(
        ["kubectl", "-n", NAMESPACE, "exec", "allie-dev", "--", "cat", _allie_path(path)]
    ).encode()
    if len(raw) > 64 * 1024:
        raise RuntimeError("global claim exceeds sealed-safe size")
    value = json.loads(raw)
    forbidden = {"prompt", "prompts", "trace", "traces", "flag", "flags", "score", "scores"}

    def reject(item: Any) -> None:
        if isinstance(item, dict):
            if forbidden.intersection(item):
                raise RuntimeError("protected key in global claim")
            for child in item.values():
                reject(child)
        elif isinstance(item, list):
            for child in item:
                reject(child)

    reject(value)
    if not isinstance(value, dict) or value.get("receipt_sha256") != digest(value):
        raise RuntimeError("global claim self-digest drifted")
    return value


def _validate_g7_claim(value: Mapping[str, Any], tombstone: Mapping[str, Any]) -> None:
    if (
        value.get("receipt_sha256") != digest(value)
        or value.get("receipt_sha256")
        != tombstone.get("generation7_global_claim", {}).get("receipt_sha256")
    ):
        raise RuntimeError("Generation-7 global claim drifted")


def observe_duplicate(root: Path, key: str) -> dict[str, Any]:
    """Render a GET/read-only duplicate receipt without protected content."""
    spec, plan, _held = static(root)
    row = preparer.MODELS[MODEL]
    if _command(
        ["kubectl", "-n", NAMESPACE, "get", "pod", "allie-dev", "-o", "jsonpath={.metadata.uid}"]
    ) != os.environ.get("SFS_OBSERVER_UID"):
        raise RuntimeError("SFS observer UID drifted")
    if (
        _command(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                "pod",
                "allie-dev",
                "-o",
                "jsonpath={.status.phase}",
            ]
        )
        != "Running"
    ):
        raise RuntimeError("SFS observer is not Running")
    job_absent = not bool(
        _command(
            ["kubectl", "-n", NAMESPACE, "get", "job", JOB_NAME, "--ignore-not-found", "-o", "name"]
        )
    )
    pods_absent = not bool(
        _command(
            ["kubectl", "-n", NAMESPACE, "get", "pod", "-l", f"job-name={JOB_NAME}", "-o", "name"]
        )
    )
    configmap_absent = not bool(
        _command(
            [
                "kubectl",
                "-n",
                NAMESPACE,
                "get",
                "configmap",
                CONFIGMAP_NAME,
                "--ignore-not-found",
                "-o",
                "name",
            ]
        )
    )
    secret_uid = _command(
        ["kubectl", "-n", NAMESPACE, "get", "secret", SECRET_NAME, "-o", "jsonpath={.metadata.uid}"]
    )
    secret_keys = _command(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "secret",
            SECRET_NAME,
            "-o",
            'go-template={{range $k,$v := .data}}{{$k}}{{"\\n"}}{{end}}',
        ]
    ).splitlines()
    generation_ids = {
        generation: preparer.generation8.generation7.exact.execution_for(
            row["cell_id"], generation
        )["execution_id"]
        for generation in (8, 9, 10)
    }
    claim_paths = {
        generation: _claim_path(execution) for generation, execution in generation_ids.items()
    }
    g7_claim = _allie_claim(_claim_path(row["g7_execution_id"]))
    tombstone = preparer.read_canonical(root / TOMBSTONE_PATH)
    preparer.validate_preinstance_tombstone(MODEL, tombstone)
    _validate_g7_claim(g7_claim, tombstone)

    from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as reconciliation

    rows = reconciliation._default_sessions(plan["tasks"][0]["task"]["key"], key)  # noqa: SLF001
    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        raise RuntimeError("Fleet session metadata response is invalid")
    run_ids = {
        8: preparer.load_static(root)[MODEL]["plan"]["attempts"][0]["run_id"],
        9: f"chris-glm53-ac-g9-r013-a1-{generation_ids[9].removeprefix('sha256:')[:8]}",
        10: plan["attempts"][0]["run_id"],
    }
    fleet: dict[str, int] = {}
    for generation in (8, 9, 10):
        expected = {row["cell_id"], generation_ids[generation], run_ids[generation]}
        matches = sum(
            bool(
                preparer.generation8._identity_values(  # noqa: SLF001
                    item, {"cell_id", "execution_id", "run_id"}
                )
                & expected
            )
            for item in rows
        )
        # Session metadata is the single authoritative duplicate carrier used
        # here.  Do not label this count as an independent ingestion or
        # verifier-endpoint observation.
        fleet[f"generation{generation}_matching_session_rows"] = matches
    fleet["session_rows_examined"] = len(rows)
    with hosted._client(key) as client:
        account_value = self_hosted._request(client, "GET", "/v1/account")
    account = {
        "team_name": str(account_value.get("team_name")),
        "team_id": str(account_value.get("team_id")),
    }
    sfs = {
        "output_root_absent": not _allie_exists(OUTPUT_ROOT),
        "generation7_claim_present": True,
        "generation7_claim_receipt_sha256": g7_claim["receipt_sha256"],
        **{
            f"generation{generation}_claim_absent": not _allie_exists(path)
            for generation, path in claim_paths.items()
        },
    }
    clear = (
        job_absent
        and pods_absent
        and configmap_absent
        and all(
            value == 0 for key_name, value in fleet.items() if key_name != "session_rows_examined"
        )
        and all(value is True for key_name, value in sfs.items() if key_name.endswith("_absent"))
        and secret_uid == SECRET_UID
        and secret_keys == [SECRET_KEY]
        and account == {"team_name": "fleet", "team_id": FLEET_TEAM_ID}
    )
    body = {
        "schema_version": DUPLICATE_SCHEMA,
        "status": "FRESH_ABSENT" if clear else "COLLISION_OR_AUTHORITY_DRIFT",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": MODEL,
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "identities": {
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "output_root": OUTPUT_ROOT,
        },
        "kubernetes": {
            "job_absent": job_absent,
            "pods_absent": pods_absent,
            "configmap_absent": configmap_absent,
            "checked_namespace": NAMESPACE,
        },
        "sfs": sfs,
        "fleet": fleet,
        "secret": {
            "name": SECRET_NAME,
            "uid": secret_uid,
            "key_names": secret_keys,
            "value_read": False,
            "value_persisted": False,
        },
        "account": account,
        "response_bodies_persisted": False,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def build_release(
    root: Path,
    package: Mapping[str, Any],
    package_commit: str,
    duplicate: Mapping[str, Any],
    launch_route: Mapping[str, Any],
) -> dict[str, Any]:
    spec, plan, held = static(root)
    if COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("package commit is invalid")
    validate_duplicate(duplicate, spec)
    hosted_runtime.validate_live_route(
        dict(launch_route),
        caller=hosted_runtime.LAUNCHER_CALLER,
        maximum_age_seconds=300,
        candidate_scored_job_names=(JOB_NAME,),
    )
    if (
        package.get("schema_version")
        != "fleet-opencode-autocontinue-generation10-glm53-split-package-v1"
        or package.get("model") != MODEL
        or package.get("generation10_spec_sha256") != spec["generation10_spec_sha256"]
        or package.get("rendered_plan_sha256") != plan["plan_sha256"]
        or package.get("held_receipt_sha256") != held["receipt_sha256"]
        or package.get("launch_authorized") is not False
        or package.get("release_included") is not False
    ):
        raise ValueError("GLM Generation-10 package drifted")
    body = {
        "schema_version": RELEASE_SCHEMA,
        "status": "RELEASED",
        "append_only": True,
        "launch_authorized": True,
        "model": MODEL,
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation10_spec_sha256": spec["generation10_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "package_commit": package_commit,
        "package_aggregate_sha256": package["aggregate_sha256"],
        "held_receipt_sha256": held["receipt_sha256"],
        "generation7_tombstone_receipt_sha256": spec["predecessor_generation7"][
            "tombstone_receipt_sha256"
        ],
        "authority_diagnosis_receipt_sha256": spec["required_authority_gate"][
            "diagnosis_receipt_sha256"
        ],
        "fresh_duplicate_preflight_receipt_sha256": duplicate["receipt_sha256"],
        "launch_route_receipt_sha256": launch_route["receipt_sha256"],
        "secret": {"name": SECRET_NAME, "uid": SECRET_UID, "key": SECRET_KEY},
        "fleet_team_id": FLEET_TEAM_ID,
        "authorization": {
            "execution_generation": 10,
            "same_statistical_cell": True,
            "hosted_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
            "create_once": True,
            "bulk_release_authorized": False,
        },
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    return {**body, "receipt_sha256": digest(body)}


def validate_release(
    value: Mapping[str, Any],
    root: Path,
    package: Mapping[str, Any],
    package_commit: str,
    duplicate: Mapping[str, Any],
    launch_route: Mapping[str, Any],
) -> None:
    if value != build_release(root, package, package_commit, duplicate, launch_route):
        raise ValueError("GLM Generation-10 release drifted")


def _claim_path(execution_id: str) -> Path:
    return CLAIM_ROOT / f"{execution_id.removeprefix('sha256:')}.json"


def _validate_fresh_sfs(spec: Mapping[str, Any], tombstone: Mapping[str, Any]) -> None:
    row = preparer.MODELS[MODEL]
    g7 = _claim_path(row["g7_execution_id"])
    if g7.is_symlink() or not g7.is_file():
        raise RuntimeError("exact Generation-7 global claim is absent")
    g7_value = load(g7)
    _validate_g7_claim(g7_value, tombstone)
    for execution_id in (
        row["g8_execution_id"],
        preparer.generation8.generation7.exact.execution_for(row["cell_id"], 9)["execution_id"],
        spec["execution"]["execution_id"],
    ):
        path = _claim_path(execution_id)
        if path.exists() or path.is_symlink():
            raise RuntimeError("a successor generation claim already exists")
    root = Path(OUTPUT_ROOT)
    if root.exists() or root.is_symlink():
        raise RuntimeError("Generation-10 output root already exists")


def claim_execution(
    spec: Mapping[str, Any], plan: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    job_uid, pod_uid = os.environ.get("JOB_UID"), os.environ.get("POD_UID")
    if not legacy._is_uuid(job_uid) or not legacy._is_uuid(pod_uid):
        raise RuntimeError("Generation-10 claim requires downward API UIDs")
    root_fd = legacy._open_directory_nofollow(CLAIM_ROOT)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            tombstone = load(Path(TOMBSTONE_PATH))
            preparer.validate_preinstance_tombstone(MODEL, tombstone)
            _validate_fresh_sfs(spec, tombstone)
            body = {
                "schema_version": CLAIM_SCHEMA,
                "generation10_spec_sha256": spec["generation10_spec_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "cell_id": spec["statistical_cell"]["cell_id"],
                "execution_id": spec["execution"]["execution_id"],
                "execution_generation": 10,
                "run_id": plan["attempts"][0]["run_id"],
                "job_uid": job_uid,
                "pod_uid": pod_uid,
                "claimed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "scoring_release_receipt_sha256": release["receipt_sha256"],
                "generation7_tombstone_receipt_sha256": tombstone["receipt_sha256"],
                "prior_claims_preserved": True,
                "immutable": True,
                "automatic_retry": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            receipt = {**body, "receipt_sha256": digest(body)}
            name = _claim_path(spec["execution"]["execution_id"]).name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            with os.fdopen(fd, "wb") as stored:
                stored.write(canonical(receipt) + b"\n")
                stored.flush()
                os.fsync(stored.fileno())
            os.fsync(root_fd)
            return receipt
    finally:
        os.close(root_fd)


def run(
    root: Path,
    package: Mapping[str, Any],
    package_commit: str,
    duplicate: Mapping[str, Any],
    launch_route: Mapping[str, Any],
    release: Mapping[str, Any],
    out: Path,
    proxy: Path,
) -> dict[str, Any]:
    spec, plan, _held = static(root)
    validate_release(release, root, package, package_commit, duplicate, launch_route)
    if out != Path(OUTPUT_ROOT):
        raise RuntimeError("Generation-10 output root drifted")
    if os.environ.get("GENERATION10_SECRET_UID") != SECRET_UID:
        raise RuntimeError("Generation-10 Secret UID binding drifted")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    hosted_runtime.validate_live_route(
        dict(launch_route),
        caller=hosted_runtime.LAUNCHER_CALLER,
        maximum_age_seconds=None,
        candidate_scored_job_names=(JOB_NAME,),
    )
    lease = plan["execution"]["endpoint_lease"]
    with legacy.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        live_route = hosted_runtime.observe_live_route(
            key,
            caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=(JOB_NAME,),
        )
        hosted_runtime.validate_live_route(
            live_route,
            caller=hosted_runtime.RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=(JOB_NAME,),
        )
        hosted._validate_plan_identity_absence(copy.deepcopy(plan), out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(copy.deepcopy(plan), out, plan["tasks"][0], key)
        claim = claim_execution(spec, plan, release)
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", copy.deepcopy(plan))
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", copy.deepcopy(release))
        self_hosted.write_json_once(out / "DUPLICATE-PREFLIGHT.json", copy.deepcopy(duplicate))
        result = legacy._run_cell(copy.deepcopy(plan), out, proxy, key)
        body = {
            "schema_version": TERMINAL_SCHEMA,
            "generation10_spec_sha256": spec["generation10_spec_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "cell_id": spec["statistical_cell"]["cell_id"],
            "execution_id": spec["execution"]["execution_id"],
            "execution_generation": 10,
            "generation_claim_receipt_sha256": claim["receipt_sha256"],
            "job_uid": claim["job_uid"],
            "pod_uid": claim["pod_uid"],
            "terminal_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "scoring_release_receipt_sha256": release["receipt_sha256"],
            "result": generation2._validated_result(result, copy.deepcopy(plan)),
            "retry_allowed": False,
            "bulk_release_authorized": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        terminal = {**body, "receipt_sha256": digest(body)}
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preview", "observe-duplicate", "render-release", "run")
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--package-commit")
    parser.add_argument("--duplicate-preflight", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    if args.command == "preview":
        spec, plan, held = static(root)
        print(
            json.dumps(
                {
                    "status": "HELD",
                    "launch_authorized": False,
                    "job_name": JOB_NAME,
                    "execution_id": spec["execution"]["execution_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "held_receipt_sha256": held["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "observe-duplicate":
        if not args.output:
            parser.error("observe-duplicate requires an unused output")
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        value = observe_duplicate(root, key)
        self_hosted.write_json_once(args.output, value)
        return 0 if value["status"] == "FRESH_ABSENT" else 1
    required = (
        args.package_manifest,
        args.package_commit,
        args.duplicate_preflight,
        args.launch_route,
    )
    if any(item is None for item in required):
        parser.error("release/run require package, commit, duplicate preflight, and launch route")
    package = load(args.package_manifest)
    duplicate = load(args.duplicate_preflight)
    route = load(args.launch_route)
    if args.command == "render-release":
        if not args.output:
            parser.error("render-release requires an unused output")
        self_hosted.write_json_once(
            args.output,
            build_release(root, package, args.package_commit, duplicate, route),
        )
        return 0
    if not args.release or not args.out_dir or not args.proxy:
        parser.error("run requires release, out-dir, and proxy")
    run(
        root,
        package,
        args.package_commit,
        duplicate,
        route,
        load(args.release),
        args.out_dir,
        args.proxy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
