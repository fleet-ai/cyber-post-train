"""Reconcile stale submitted Jobs API history without weakening v32 zero-state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from evals.fleet import exact_pass4_crypto as crypto

SCHEMA = "fleet-glm53-dedicated-v32-stale-run-reconciliation-v1"
HELD_SCHEMA = "fleet-glm53-dedicated-v32-stale-run-reconciliation-held-v1"
ACTIVE_STATUSES = {"SUBMITTED", "SUSPENDED", "RUNNING"}
TERMINAL_STATUSES = {"CANCELLED", "COMPLETED", "FAILED", "STOPPED", "SUCCEEDED"}
MAX_AGE_SECONDS = 300
RECEIPT_BASENAMES = {
    "ACTIVE.json",
    "CREATED.json",
    "FAILED.json",
    "QUALIFIED.json",
    "READY.json",
    "RELEASE.json",
    "RESULT.json",
    "TERMINAL.json",
}
RESULT_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "jobs_api_pages",
    "project_list_snapshot_sha256",
    "project_rows_seen",
    "reconciled_rows",
    "kubernetes_snapshot_sha256",
    "project_object_count",
    "terminal_project_object_count",
    "terminal_project_objects",
    "terminal_project_object_snapshot_sha256",
    "project_gpu_pod_count",
    "sfs_observation",
    "api_mutation_calls",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}
ROW_KEYS = {
    "api_run_id",
    "title",
    "run_dir",
    "listed_status",
    "exact_get_http_status",
    "kubernetes_reference_count",
    "terminal_basis",
    "terminal_kubernetes_evidence",
    "sfs_root_exists",
    "sfs_terminal_evidence",
}


class ReconciliationError(RuntimeError):
    """The stale history could not be proven terminal and score-free."""


class Backend(Protocol):
    def list_runs(self) -> tuple[list[dict[str, Any]], int]: ...

    def get_run(self, api_run_id: str) -> dict[str, Any] | None: ...

    def kubernetes_inventory(self) -> dict[str, Any]: ...

    def sfs_evidence(self, run_dirs: tuple[str, ...]) -> dict[str, Any]: ...


def _project_run_dir(value: object) -> bool:
    return isinstance(value, str) and value.startswith(
        "/mnt/sfs/jobs/chris-cyber-evalserve-"
    )


def _project_title(value: object) -> bool:
    return isinstance(value, str) and value.startswith("chris-cyber-evalserve-")


def _status(value: object) -> str:
    if not isinstance(value, str):
        raise ReconciliationError("v32_stale_run_status_invalid")
    result = value.upper()
    if result not in ACTIVE_STATUSES | TERMINAL_STATUSES:
        raise ReconciliationError("v32_stale_run_status_invalid")
    return result


def normalize_project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not (_project_run_dir(row.get("run_dir")) or _project_title(row.get("title"))):
            continue
        if not isinstance(row.get("name"), str) or not _project_run_dir(row.get("run_dir")):
            raise ReconciliationError("v32_stale_run_identity_invalid")
        result.append(
            {
                "api_run_id": row["name"],
                "title": row.get("title"),
                "run_dir": row["run_dir"],
                "listed_status": _status(row.get("status")),
            }
        )
    return sorted(result, key=lambda row: (row["api_run_id"], row["run_dir"]))


def project_snapshot_sha256(rows: list[dict[str, Any]]) -> str:
    return crypto.sha256(crypto.canonical_json(normalize_project_rows(rows)))


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _pod_gpu_requests(pod: Mapping[str, Any]) -> int:
    try:
        return sum(
            int(
                ((container.get("resources") or {}).get("requests") or {}).get(
                    "nvidia.com/gpu", 0
                )
            )
            for container in (pod.get("spec") or {}).get("containers") or []
        )
    except (TypeError, ValueError) as exc:
        raise ReconciliationError("v32_stale_run_gpu_shape_invalid") from exc


def _safe_kubernetes_projection(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = inventory.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ReconciliationError("v32_stale_run_kubernetes_inventory_invalid")
    result = []
    for item in items:
        metadata = _metadata(item)
        result.append(
            {
                "kind": item.get("kind"),
                "name": metadata.get("name"),
                "uid": metadata.get("uid"),
                "phase": (item.get("status") or {}).get("phase"),
                "job_status": (item.get("status") or {}).get("jobStatus"),
                "owner": (metadata.get("labels") or {}).get(
                    "cyber-post-train.fleet.ai/owner"
                ),
                "gpu_requests": _pod_gpu_requests(item) if item.get("kind") == "Pod" else 0,
            }
        )
    return sorted(result, key=lambda row: (str(row["kind"]), str(row["name"])))


def _terminal_kubernetes_object(item: Mapping[str, Any]) -> bool:
    if item.get("kind") == "RayJob":
        return str((item.get("status") or {}).get("jobStatus") or "").upper() in {
            "FAILED",
            "STOPPED",
            "SUCCEEDED",
        }
    if item.get("kind") == "Workload":
        return any(
            condition.get("type") == "Finished"
            and condition.get("status") == "True"
            and condition.get("reason") in {"Failed", "Succeeded"}
            for condition in (item.get("status") or {}).get("conditions") or []
            if isinstance(condition, dict)
        )
    return False


def _terminal_object_projection(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata(item)
    owners = metadata.get("ownerReferences") or []
    if not isinstance(owners, list) or not all(isinstance(row, dict) for row in owners):
        raise ReconciliationError("v32_stale_run_kubernetes_inventory_invalid")
    return {
        "kind": item.get("kind"),
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "job_status": (item.get("status") or {}).get("jobStatus"),
        "finished_reason": next(
            (
                condition.get("reason")
                for condition in (item.get("status") or {}).get("conditions") or []
                if isinstance(condition, dict)
                and condition.get("type") == "Finished"
                and condition.get("status") == "True"
            ),
            None,
        ),
        "owners": sorted(
            [
                {
                    "kind": owner.get("kind"),
                    "name": owner.get("name"),
                    "uid": owner.get("uid"),
                }
                for owner in owners
            ],
            key=lambda row: (str(row["kind"]), str(row["name"]), str(row["uid"])),
        ),
    }


def build_reconciliation(
    *,
    backend: Backend,
    now: float | None = None,
    rows_snapshot: tuple[list[dict[str, Any]], int] | None = None,
) -> dict[str, Any]:
    observed = time.time() if now is None else now
    rows, pages = backend.list_runs() if rows_snapshot is None else rows_snapshot
    normalized = normalize_project_rows(rows)
    stale: list[dict[str, Any]] = []
    for row in normalized:
        current = backend.get_run(row["api_run_id"])
        if current is None:
            if row["listed_status"] in ACTIVE_STATUSES:
                stale.append(row)
            continue
        if current.get("run_dir") != row["run_dir"]:
            raise ReconciliationError("v32_stale_run_exact_get_drift")
        current_status = _status(current.get("status"))
        if row["listed_status"] in ACTIVE_STATUSES:
            if current_status in ACTIVE_STATUSES:
                raise ReconciliationError("v32_stale_run_current_server_present")
            raise ReconciliationError("v32_stale_run_exact_get_status_drift")

    inventory = backend.kubernetes_inventory()
    projection = _safe_kubernetes_projection(inventory)
    all_project_objects = [
        item
        for item in inventory["items"]
        if item.get("kind") in {"RayJob", "RayCluster", "Workload", "Service"}
        and (
            (_metadata(item).get("labels") or {}).get(
                "cyber-post-train.fleet.ai/owner"
            )
            == "chris"
            or "/mnt/sfs/jobs/chris-cyber-evalserve-"
            in json.dumps(item, sort_keys=True)
        )
    ]
    gpu_pods = [
        item
        for item in inventory["items"]
        if item.get("kind") == "Pod"
        and (item.get("status") or {}).get("phase") in {"Pending", "Running"}
        and _pod_gpu_requests(item) > 0
        and (
            (_metadata(item).get("labels") or {}).get(
                "cyber-post-train.fleet.ai/owner"
            )
            == "chris"
            or "chris-cyber-evalserve-" in json.dumps(item, sort_keys=True)
        )
    ]
    terminal_project_objects = [
        item for item in all_project_objects if _terminal_kubernetes_object(item)
    ]
    blocking_project_objects = [
        item for item in all_project_objects if not _terminal_kubernetes_object(item)
    ]
    terminal_projection = sorted(
        [_terminal_object_projection(item) for item in terminal_project_objects],
        key=lambda row: (str(row["kind"]), str(row["name"])),
    )
    if blocking_project_objects or gpu_pods:
        raise ReconciliationError("v32_stale_run_kubernetes_or_gpu_remnant")

    sfs = backend.sfs_evidence(tuple(row["run_dir"] for row in stale))
    roots = sfs.get("roots") if isinstance(sfs, dict) else None
    if (
        set(sfs) != {
            "observer_pod_name",
            "observer_pod_uid",
            "observer_sfs_mount_path",
            "roots",
        }
        or not isinstance(roots, dict)
        or set(roots) != {row["run_dir"] for row in stale}
    ):
        raise ReconciliationError("v32_stale_run_sfs_evidence_invalid")
    reconciled = []
    for row in stale:
        evidence = roots[row["run_dir"]]
        receipts = evidence.get("terminal_evidence") if isinstance(evidence, dict) else None
        if (
            set(evidence) != {"root_exists", "unsafe_symlink", "terminal_evidence"}
            or evidence.get("unsafe_symlink") is not False
            or not isinstance(receipts, list)
            or any(
                not isinstance(receipt, dict)
                or set(receipt) != {"relative_path", "size", "sha256"}
                or Path(str(receipt["relative_path"])).name not in RECEIPT_BASENAMES
                or not str(receipt["sha256"]).startswith("sha256:")
                for receipt in receipts
            )
        ):
            raise ReconciliationError("v32_stale_run_sfs_evidence_invalid")
        references = [
            _terminal_object_projection(item)
            for item in terminal_project_objects
            if row["api_run_id"] in json.dumps(item, sort_keys=True)
            or row["run_dir"] in json.dumps(item, sort_keys=True)
        ]
        if any(
            row["api_run_id"] in json.dumps(item, sort_keys=True)
            or row["run_dir"] in json.dumps(item, sort_keys=True)
            for item in blocking_project_objects
        ):
            raise ReconciliationError("v32_stale_run_kubernetes_reference_present")
        reconciled.append(
            {
                **row,
                "exact_get_http_status": 404,
                "kubernetes_reference_count": len(references),
                "terminal_basis": (
                    "EXACT_GET_404_AND_TERMINAL_KUBERNETES_UID_CHAIN"
                    if references
                    else "EXACT_GET_404_AND_ZERO_KUBERNETES_REFERENCES"
                ),
                "terminal_kubernetes_evidence": sorted(
                    references,
                    key=lambda item: (str(item["kind"]), str(item["name"])),
                ),
                "sfs_root_exists": evidence["root_exists"],
                "sfs_terminal_evidence": receipts,
            }
        )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASSED_SCORE_BLIND_STALE_RUN_RECONCILIATION",
        "observed_at_epoch": observed,
        "jobs_api_pages": pages,
        "project_list_snapshot_sha256": crypto.sha256(crypto.canonical_json(normalized)),
        "project_rows_seen": len(normalized),
        "reconciled_rows": reconciled,
        "kubernetes_snapshot_sha256": crypto.sha256(crypto.canonical_json(projection)),
        "project_object_count": 0,
        "terminal_project_object_count": len(terminal_projection),
        "terminal_project_objects": terminal_projection,
        "terminal_project_object_snapshot_sha256": crypto.sha256(
            crypto.canonical_json(terminal_projection)
        ),
        "project_gpu_pod_count": 0,
        "sfs_observation": {
            "observer_pod_name": sfs["observer_pod_name"],
            "observer_pod_uid": sfs["observer_pod_uid"],
            "observer_sfs_mount_path": sfs["observer_sfs_mount_path"],
        },
        "api_mutation_calls": 0,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def validate_reconciliation(
    value: Mapping[str, Any], *, rows: list[dict[str, Any]], now: float | None = None
) -> None:
    observed = value.get("observed_at_epoch")
    current_time = time.time() if now is None else now
    reconciled = value.get("reconciled_rows")
    normalized = normalize_project_rows(rows)
    expected = [row for row in normalized if row["listed_status"] in ACTIVE_STATUSES]
    if (
        set(value) != RESULT_KEYS
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "PASSED_SCORE_BLIND_STALE_RUN_RECONCILIATION"
        or value.get("receipt_sha256")
        != crypto.digest_without(dict(value), "receipt_sha256")
        or not isinstance(observed, (int, float))
        or isinstance(observed, bool)
        or observed > current_time + 5
        or current_time - observed > MAX_AGE_SECONDS
        or value.get("project_list_snapshot_sha256") != project_snapshot_sha256(rows)
        or value.get("project_rows_seen") != len(normalized)
        or not isinstance(reconciled, list)
        or any(set(row) != ROW_KEYS for row in reconciled if isinstance(row, dict))
        or any(not isinstance(row, dict) for row in reconciled)
        or value.get("project_object_count") != 0
        or value.get("project_gpu_pod_count") != 0
        or any(
            value.get(field) != 0
            for field in (
                "api_mutation_calls",
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or value.get("protected_content_included") is not False
    ):
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    if value.get("jobs_api_pages") is None or not isinstance(
        value.get("jobs_api_pages"), int
    ) or value["jobs_api_pages"] < 1:
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    expected_identities = [
        (row["api_run_id"], row["title"], row["run_dir"], row["listed_status"])
        for row in expected
    ]
    actual_identities = [
        (row["api_run_id"], row["title"], row["run_dir"], row["listed_status"])
        for row in reconciled
    ]
    if actual_identities != expected_identities:
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    sfs = value.get("sfs_observation")
    try:
        from evals.fleet import glm53_dedicated_v32_live_authorization_v1 as live

        expected_sfs = {
            "observer_pod_name": live.SFS_OBSERVER_POD_NAME,
            "observer_pod_uid": live.SFS_OBSERVER_POD_UID,
            "observer_sfs_mount_path": live.SFS_OBSERVER_MOUNT_PATH,
        }
    except (AttributeError, ImportError) as exc:
        raise ReconciliationError("v32_stale_run_reconciliation_invalid") from exc
    if sfs != expected_sfs:
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    for row in reconciled:
        receipts = row["sfs_terminal_evidence"]
        if (
            row["exact_get_http_status"] != 404
            or not isinstance(row["kubernetes_reference_count"], int)
            or isinstance(row["kubernetes_reference_count"], bool)
            or row["kubernetes_reference_count"] < 0
            or row["kubernetes_reference_count"]
            != len(row["terminal_kubernetes_evidence"])
            or row["terminal_basis"]
            not in {
                "EXACT_GET_404_AND_ZERO_KUBERNETES_REFERENCES",
                "EXACT_GET_404_AND_TERMINAL_KUBERNETES_UID_CHAIN",
            }
            or (row["kubernetes_reference_count"] == 0)
            != (
                row["terminal_basis"]
                == "EXACT_GET_404_AND_ZERO_KUBERNETES_REFERENCES"
            )
            or not isinstance(row["sfs_root_exists"], bool)
            or not isinstance(receipts, list)
            or not isinstance(row["terminal_kubernetes_evidence"], list)
            or any(
                not isinstance(item, dict)
                or set(item)
                != {
                    "kind",
                    "name",
                    "uid",
                    "job_status",
                    "finished_reason",
                    "owners",
                }
                or item["kind"] not in {"RayJob", "Workload"}
                or not isinstance(item["name"], str)
                or not isinstance(item["uid"], str)
                or not item["uid"]
                or (
                    item["kind"] == "RayJob"
                    and item["job_status"] not in {"FAILED", "STOPPED", "SUCCEEDED"}
                )
                or (
                    item["kind"] == "Workload"
                    and item["finished_reason"] not in {"Failed", "Succeeded"}
                )
                or not isinstance(item["owners"], list)
                for item in row["terminal_kubernetes_evidence"]
            )
            or any(
                not isinstance(receipt, dict)
                or set(receipt) != {"relative_path", "size", "sha256"}
                or not isinstance(receipt["relative_path"], str)
                or Path(receipt["relative_path"]).name not in RECEIPT_BASENAMES
                or Path(receipt["relative_path"]).is_absolute()
                or ".." in Path(receipt["relative_path"]).parts
                or not isinstance(receipt["size"], int)
                or isinstance(receipt["size"], bool)
                or receipt["size"] < 0
                or not isinstance(receipt["sha256"], str)
                or len(receipt["sha256"]) != 71
                or not receipt["sha256"].startswith("sha256:")
                for receipt in receipts
            )
        ):
            raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    if (
        not isinstance(value.get("terminal_project_object_count"), int)
        or isinstance(value.get("terminal_project_object_count"), bool)
        or value["terminal_project_object_count"] < 0
        or not isinstance(value.get("terminal_project_object_snapshot_sha256"), str)
        or len(value["terminal_project_object_snapshot_sha256"]) != 71
        or not value["terminal_project_object_snapshot_sha256"].startswith("sha256:")
        or not isinstance(value.get("terminal_project_objects"), list)
        or value["terminal_project_object_count"]
        != len(value["terminal_project_objects"])
        or value["terminal_project_object_snapshot_sha256"]
        != crypto.sha256(crypto.canonical_json(value["terminal_project_objects"]))
        or any(
            not isinstance(item, dict)
            or set(item)
            != {
                "kind",
                "name",
                "uid",
                "job_status",
                "finished_reason",
                "owners",
            }
            or item["kind"] not in {"RayJob", "Workload"}
            or not isinstance(item["name"], str)
            or not isinstance(item["uid"], str)
            or not item["uid"]
            or (
                item["kind"] == "RayJob"
                and item["job_status"] not in {"FAILED", "STOPPED", "SUCCEEDED"}
            )
            or (
                item["kind"] == "Workload"
                and item["finished_reason"] not in {"Failed", "Succeeded"}
            )
            or not isinstance(item["owners"], list)
            for item in value["terminal_project_objects"]
        )
    ):
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")
    terminal_keys = {
        crypto.sha256(crypto.canonical_json(item))
        for item in value["terminal_project_objects"]
    }
    if any(
        crypto.sha256(crypto.canonical_json(item)) not in terminal_keys
        for row in reconciled
        for item in row["terminal_kubernetes_evidence"]
    ):
        raise ReconciliationError("v32_stale_run_reconciliation_invalid")


class SystemBackend:
    def __init__(self) -> None:
        from evals.fleet import glm53_dedicated_v32_live_authorization_v1 as live

        self.live = live
        self.backend = live.SystemBackend()

    def list_runs(self) -> tuple[list[dict[str, Any]], int]:
        return self.backend.list_runs()

    def get_run(self, api_run_id: str) -> dict[str, Any] | None:
        return self.backend.get_run(api_run_id)

    def kubernetes_inventory(self) -> dict[str, Any]:
        return self.backend.kubernetes_inventory()

    def sfs_evidence(self, run_dirs: tuple[str, ...]) -> dict[str, Any]:
        name, uid, mount = self.backend._observer()  # noqa: SLF001
        paths = [self.backend._observed_path(mount, path) for path in run_dirs]  # noqa: SLF001
        script = """
import hashlib,json,os,sys
allowed={'ACTIVE.json','CREATED.json','FAILED.json','QUALIFIED.json','READY.json','RELEASE.json','RESULT.json','TERMINAL.json'}
result={}
for canonical,root in zip(sys.argv[1::2],sys.argv[2::2],strict=True):
    unsafe=os.path.islink(root)
    receipts=[]
    exists=os.path.exists(root)
    if exists and not unsafe:
        for current,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if not os.path.islink(os.path.join(current,d))]
            for filename in files:
                path=os.path.join(current,filename)
                if filename in allowed:
                    unsafe=unsafe or os.path.islink(path)
                    if not os.path.islink(path):
                        raw=open(path,'rb').read()
                        receipts.append({'relative_path':os.path.relpath(path,root),'size':len(raw),'sha256':'sha256:'+hashlib.sha256(raw).hexdigest()})
    result[canonical]={
        'root_exists':exists,
        'unsafe_symlink':unsafe,
        'terminal_evidence':sorted(receipts,key=lambda row:row['relative_path']),
    }
print(json.dumps(result,sort_keys=True,separators=(',',':')))
"""
        args = [value for pair in zip(run_dirs, paths, strict=True) for value in pair]
        completed = subprocess.run(
            ["kubectl", "-n", self.live.NAMESPACE, "exec", "-i", name, "--", "python", "-", *args],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ReconciliationError("v32_stale_run_sfs_observer_failed")
        roots = json.loads(completed.stdout)
        return {
            "observer_pod_name": name,
            "observer_pod_uid": uid,
            "observer_sfs_mount_path": mount,
            "roots": roots,
        }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(crypto.canonical_json(dict(value)) + b"\n")


def build_held(package_commit: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": HELD_SCHEMA,
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "reconciliation_module": __name__,
        "maximum_age_seconds": MAX_AGE_SECONDS,
        "submitted_exact_404_requires_per_row_evidence": True,
        "unknown_statuses_block": True,
        "kubernetes_and_gpu_remnants_block": True,
        "sfs_terminal_files_hashed_not_read_into_receipt": True,
        "server_launch_authorized": False,
        "api_mutation_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    backend = SystemBackend()
    rows, pages = backend.list_runs()
    value = build_reconciliation(backend=backend, rows_snapshot=(rows, pages))
    validate_reconciliation(value, rows=rows)
    _write_once(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "project_rows_seen": value["project_rows_seen"],
                "reconciled_rows": len(value["reconciled_rows"]),
                "receipt_sha256": value["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
