#!/usr/bin/env python3
"""Render and server-preview one read-only Fleet heldout launch gate.

The gate packages sixteen sealed source packets separately so every ConfigMap
stays below Kubernetes' object limit.  When later created, one bounded CPU Job
runs the existing duplicate census for each packet and a score-blind PostgreSQL
daily-cap census.  This renderer never creates cluster objects.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import heldout_launch
from evals.fleet.evaluate import stable_job_preview

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "5b31f910de81809811c61118b7ec0d11d353eb61"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
NAME = "chris-q38-h20-budget-duplicate-gate-v2"
NAMESPACE = heldout_launch.NAMESPACE
PACKETS = 16
SESSIONS = 160
CAP = 500

GATE = r'''from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import heldout_launch

CAP = 500
RESERVATION_TABLE = "public.cyber_fleet_daily_rollout_reservations_v1"


def digest(value):
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def extract_bundle(source, destination):
    values = json.loads(gzip.decompress(source.read_bytes()))
    if not isinstance(values, dict):
        raise RuntimeError("invalid_bundle")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, text in values.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not isinstance(text, str):
            raise RuntimeError("invalid_bundle_path")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def control_dsn():
    value = os.environ.get("ROLLOUT_DATABASE_URL", "")
    parsed = urlsplit(value)
    query = {key.casefold() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.fragment
        or query & {"database", "dbname"}
    ):
        raise RuntimeError("invalid_database_environment")
    return value


def database_dsn(base, database):
    parsed = urlsplit(base)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, "/" + quote(database, safe=""), parsed.query, "")
    )


def connect(dsn):
    return psycopg.connect(
        dsn,
        autocommit=False,
        row_factory=dict_row,
        connect_timeout=5,
        application_name="fleet-heldout20-budget-duplicate-gate-v2",
    )


def readonly(connection):
    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
    connection.execute("SET LOCAL TIME ZONE 'UTC'")
    connection.execute("SET LOCAL search_path = pg_catalog, public")
    connection.execute("SET LOCAL statement_timeout = '30s'")
    connection.execute("SET LOCAL lock_timeout = '5s'")


def cell_id(row):
    identity = "\0".join(
        str(row[field])
        for field in ("experiment_id", "task_version_id", "model_id", "attempt")
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-cyber-rollout-cell-v1:{identity}"))


def reservation_digest(row):
    body = {
        "schema": "cyber_fleet_daily_rollout_reservation_v1",
        "created_utc_day": row["created_utc_day"].isoformat(),
        "reservation_key": row["reservation_key"],
        "database": row["database_name"],
        "rollouts": row["rollouts"],
        "baseline_materialized_rollouts": row["baseline_materialized_rollouts"],
        "authorization_cap": row["authorization_cap"],
    }
    return crypto.digest_without(body, "sha256")


def count_database(base, database, utc_day, planned_ids):
    with connect(database_dsn(base, database)) as connection, connection.transaction():
        readonly(connection)
        tables = connection.execute(
            "SELECT to_regclass('public.rollout_cells') AS cells, "
            "to_regclass('public.rollout_events') AS events"
        ).fetchone()
        if tables["cells"] is None and tables["events"] is None:
            return None
        if tables["cells"] is None or tables["events"] is None:
            raise RuntimeError("incomplete_rollout_database")
        row = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM public.rollout_events
               WHERE event = 'started' AND recorded_at >= %s::date
                 AND recorded_at < (%s::date + INTERVAL '1 day')) AS used,
              (SELECT COUNT(*) FROM public.rollout_cells
               WHERE state IN ('pending', 'claimed')) AS pending,
              ((SELECT COUNT(*) FROM public.rollout_events WHERE event = 'started') +
               (SELECT COUNT(*) FROM public.rollout_cells
                WHERE state IN ('pending', 'claimed'))) AS materialized,
              (SELECT COUNT(*) FROM public.rollout_cells
               WHERE cell_id = ANY(%s::text[])) AS duplicate_cells,
              (SELECT COUNT(*) FROM public.rollout_cells
               WHERE cell_id = ANY(%s::text[]) AND session_id IS NOT NULL) AS duplicate_sessions
            """,
            (utc_day, utc_day, planned_ids, planned_ids),
        ).fetchone()
        return {key: int(row[key]) for key in row}


def budget_census(planned_ids, planned_sessions):
    base = control_dsn()
    with connect(base) as control, control.transaction():
        readonly(control)
        utc_day = control.execute(
            "SELECT (clock_timestamp() AT TIME ZONE 'UTC')::date AS utc_day"
        ).fetchone()["utc_day"]
        catalog = control.execute(
            """
            SELECT datname, datallowconn,
                   has_database_privilege(datname, 'CONNECT') AS can_connect
            FROM pg_database WHERE NOT datistemplate ORDER BY datname
            """
        ).fetchall()
        table = control.execute(
            "SELECT to_regclass(%s) AS table_name", (RESERVATION_TABLE,)
        ).fetchone()["table_name"]
        reservations = []
        if table is not None:
            reservations = control.execute(
                f"""
                SELECT reservation_key, database_name, rollouts,
                       baseline_materialized_rollouts, authorization_cap,
                       created_utc_day, reservation_sha256
                FROM {RESERVATION_TABLE} ORDER BY reservation_key
                """
            ).fetchall()

    required = {row["database_name"] for row in reservations}
    counts = {}
    unreadable = 0
    for row in catalog:
        if not row["datallowconn"]:
            continue
        if not row["can_connect"]:
            unreadable += 1
            continue
        value = count_database(base, row["datname"], utc_day, planned_ids)
        if value is not None:
            counts[row["datname"]] = value
    if unreadable:
        raise RuntimeError("unreadable_connectable_database")
    if required - counts.keys():
        raise RuntimeError("reserved_database_unreadable")

    for row in reservations:
        if (
            not 0 < int(row["rollouts"]) <= CAP
            or not 0 < int(row["authorization_cap"]) <= CAP
            or row["reservation_sha256"] != reservation_digest(row)
        ):
            raise RuntimeError("invalid_daily_reservation")

    used = sum(row["used"] for row in counts.values())
    pending = sum(row["pending"] for row in counts.values())
    duplicates = sum(row["duplicate_cells"] for row in counts.values())
    duplicate_sessions = sum(row["duplicate_sessions"] for row in counts.values())
    reserved = sum(
        max(
            int(row["baseline_materialized_rollouts"])
            + int(row["rollouts"])
            - counts[row["database_name"]]["materialized"],
            0,
        )
        for row in reservations
    )
    committed_before = used + pending + reserved
    committed_after = committed_before + planned_sessions
    with connect(base) as control, control.transaction():
        readonly(control)
        final_day = control.execute(
            "SELECT (clock_timestamp() AT TIME ZONE 'UTC')::date AS utc_day"
        ).fetchone()["utc_day"]
    if final_day != utc_day:
        raise RuntimeError("utc_day_changed")
    if duplicates or duplicate_sessions:
        raise RuntimeError("postgres_exact_duplicate")
    if committed_after > CAP:
        raise RuntimeError("daily_rollout_budget_exhausted")
    return {
        "utc_day": utc_day.isoformat(),
        "cap": CAP,
        "used": used,
        "pending_or_claimed": pending,
        "durably_reserved_not_materialized": reserved,
        "committed_before": committed_before,
        "planned": planned_sessions,
        "committed_after": committed_after,
        "remaining_after": CAP - committed_after,
        "rollout_databases": len(counts),
        "reservation_rows": len(reservations),
        "exact_duplicate_cells": duplicates,
        "exact_duplicate_sessions": duplicate_sessions,
        "authorization_kind": "read_only_preflight_not_reservation",
    }


def write_receipt(receipt):
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    print(encoded, flush=True)
    Path("/dev/termination-log").write_text(encoded + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    plan = json.loads((root / "gate-plan.json").read_text(encoding="utf-8"))
    if plan["schema"] != "cyber_fleet_heldout_budget_duplicate_gate_plan_v1":
        raise RuntimeError("invalid_plan")
    if plan["sha256"] != digest({key: value for key, value in plan.items() if key != "sha256"}):
        raise RuntimeError("invalid_plan_digest")
    packets_root = root / "packets"
    packets_root.mkdir()
    planned_ids = []
    census = []
    for item in plan["packets"]:
        destination = packets_root / item["id"]
        extract_bundle(Path("/gate-input/packets") / f"{item['id']}.json.gz", destination)
        packet_path = destination / "LAUNCH_PACKET.json"
        if file_digest(packet_path) != item["packet_sha256"]:
            raise RuntimeError("packet_digest_changed")
        package = heldout_launch.build_package(packet_path)
        if (
            package.packet.job_name != item["job_name"]
            or package.packet.config_map_name != item["config_map_name"]
            or package.packet.output_root != item["output_root"]
            or package.packet.database != item["database"]
            or package.packet.identity_sha256 != item["evaluation_identity_sha256"]
        ):
            raise RuntimeError("packet_binding_changed")
        rows = heldout_launch.sealed_evaluation(package).rows
        if len(rows) != item["cells"]:
            raise RuntimeError("packet_cell_count_changed")
        planned_ids.extend(cell_id(row) for row in rows)
        census.append(
            heldout_launch.duplicate_census(
                package,
                cluster=heldout_launch.KubectlCluster("incluster"),
                database=heldout_launch.PostgresDatabase(),
            )
        )
    if len(planned_ids) != plan["planned_sessions"] or len(set(planned_ids)) != len(planned_ids):
        raise RuntimeError("planned_cell_identity_changed")
    if any(any(value != 0 for value in row.values()) for row in census):
        raise RuntimeError("kubernetes_duplicate")
    budget = budget_census(planned_ids, plan["planned_sessions"])
    receipt = {
        "schema": "cyber_fleet_heldout_budget_duplicate_gate_receipt_v1",
        "status": "accepted",
        "plan_sha256": plan["sha256"],
        "packet_set_sha256": plan["packet_set_sha256"],
        "packet_count": len(plan["packets"]),
        "kubernetes_sfs_database_duplicate_checks": len(census),
        "duplicate_counts": {"jobs": 0, "config_maps": 0, "pods": 0, "outputs": 0,
                             "databases": 0, "ledger_identities": 0,
                             "postgres_cells": 0, "postgres_sessions": 0},
        "budget": budget,
        "privacy": {"score_blind": True, "local_results_read": False,
                    "scores_or_sealed_outputs_read": False, "credentials_included": False},
        "external_mutations": 0,
    }
    receipt["sha256"] = digest(receipt)
    write_receipt(receipt)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "schema": "cyber_fleet_heldout_budget_duplicate_gate_receipt_v1",
            "status": "blocked",
            "reason_class": type(exc).__name__,
            "reason_sha256": "sha256:" + hashlib.sha256(str(exc).encode()).hexdigest(),
            "privacy": {"score_blind": True, "local_results_read": False,
                        "scores_or_sealed_outputs_read": False, "credentials_included": False},
            "external_mutations": 0,
        }
        failure["sha256"] = digest(failure)
        write_receipt(failure)
        raise SystemExit(1)
'''


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _gzip_json(value: Any) -> bytes:
    compressed = gzip.compress(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    return compressed[:9] + b"\xff" + compressed[10:]


def _files(path: Path) -> dict[str, str]:
    result = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and not item.is_symlink():
            result[item.name] = item.read_text(encoding="utf-8")
    return result


def _runtime(plan: dict[str, Any]) -> bytes:
    files = {"gate.py": GATE, "gate-plan.json": json.dumps(plan, sort_keys=True)}
    for source in (
        ROOT / "cyber_post_train" / "__init__.py",
        ROOT / "cyber_post_train" / "jobs.py",
        ROOT / "evals" / "__init__.py",
        ROOT / "evals" / "fleet" / "__init__.py",
    ):
        files[str(source.relative_to(ROOT))] = source.read_text(encoding="utf-8")
    for source in sorted((ROOT / "evals" / "fleet").glob("*.py")):
        files[str(source.relative_to(ROOT))] = source.read_text(encoding="utf-8")
    return _gzip_json(files)


def _assert_commit() -> None:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if value != EXPECTED_COMMIT:
        raise ValueError("integration checkout is not the authorized commit")


def _packet_plan(render: Path) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    bindings_path = render / "static-bindings.json"
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in bindings.items() if key != "sha256"}
    if (
        bindings.get("schema") != "cyber_fleet_campaign_static_bindings_v1"
        or bindings.get("sha256") != _digest(unsigned)
        or len(bindings.get("groups", {})) != PACKETS
        or len(bindings.get("cells", {})) != SESSIONS
    ):
        raise ValueError("static bindings are invalid")
    packets = []
    bundles = []
    all_cells = set()
    common_ledger = None
    for index, (group_id, group) in enumerate(sorted(bindings["groups"].items())):
        packet_path = (render / group["packet"]).resolve()
        if render.resolve() not in packet_path.parents or _file_digest(packet_path) != group[
            "packet_sha256"
        ]:
            raise ValueError("source packet differs from static binding")
        package = heldout_launch.build_package(packet_path)
        sealed = heldout_launch.sealed_evaluation(package)
        members = set(group["cells"])
        if len(sealed.rows) != len(members) or all_cells & members:
            raise ValueError("source group does not form an exact cell partition")
        all_cells.update(members)
        ledger_digest = _file_digest(package.packet.files["evaluation_ledger"])
        common_ledger = common_ledger or ledger_digest
        if ledger_digest != common_ledger:
            raise ValueError("source packets do not share one evaluation ledger")
        item_id = f"p{index:02d}"
        packets.append(
            {
                "id": item_id,
                "group_id": group_id,
                "packet_sha256": group["packet_sha256"],
                "job_name": package.packet.job_name,
                "config_map_name": package.packet.config_map_name,
                "output_root": package.packet.output_root,
                "database": package.packet.database,
                "evaluation_identity_sha256": package.packet.identity_sha256,
                "cells": len(sealed.rows),
            }
        )
        bundles.append((item_id, _gzip_json(_files(packet_path.parent))))
    if all_cells != set(bindings["cells"]):
        raise ValueError("source groups do not cover all bound cells")
    packet_set = [
        {key: value for key, value in item.items() if key not in {"job_name", "config_map_name",
                                                                  "output_root", "database"}}
        for item in packets
    ]
    plan = {
        "schema": "cyber_fleet_heldout_budget_duplicate_gate_plan_v1",
        "repo_commit": EXPECTED_COMMIT,
        "static_bindings_file_sha256": _file_digest(bindings_path),
        "static_bindings_sha256": bindings["sha256"],
        "packet_set_sha256": _digest(packet_set),
        "planned_sessions": SESSIONS,
        "daily_cap": CAP,
        "packets": packets,
    }
    plan["sha256"] = _digest(plan)
    return plan, bundles


def _config_map(name: str, key: str, value: bytes) -> dict[str, Any]:
    if len(base64.b64encode(value)) > 900_000:
        raise ValueError(f"ConfigMap {name} exceeds the conservative size ceiling")
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"cyber-post-train.fleet.ai/experiment": NAME,
                       "cyber-post-train.fleet.ai/owner": "chris"},
            "annotations": {heldout_launch.CREATE_ONCE_ANNOTATION: "true"},
        },
        "immutable": True,
        "binaryData": {key: base64.b64encode(value).decode()},
    }


def _job(config_maps: list[str]) -> dict[str, Any]:
    projections = [
        {"configMap": {"name": config_maps[0], "items": [{"key": "runtime.json.gz",
                                                             "path": "runtime.json.gz"}]}}
    ]
    projections.extend(
        {"configMap": {"name": name, "items": [{"key": "packet.json.gz",
                                                   "path": f"packets/p{index:02d}.json.gz"}]}}
        for index, name in enumerate(config_maps[1:])
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": NAMESPACE,
            "labels": {"cyber-post-train.fleet.ai/experiment": NAME,
                       "cyber-post-train.fleet.ai/owner": "chris",
                       "kueue.x-k8s.io/queue-name": "training-lq"},
            "annotations": {heldout_launch.FAILURE_ALERT_ANNOTATION: "off",
                            heldout_launch.CREATE_ONCE_ANNOTATION: "true"},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1800,
            "template": {
                "metadata": {"labels": {"cyber-post-train.fleet.ai/experiment": NAME,
                                        "cyber-post-train.fleet.ai/owner": "chris",
                                        heldout_launch.POSTGRES_CLIENT_LABEL: "true"}},
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "nodeSelector": {"kubernetes.io/arch": "amd64",
                                     "workload": "fleetai-training-ng-cpu"},
                    "tolerations": [{"key": "workload", "operator": "Equal",
                                     "value": "fleetai-training-ng-cpu",
                                     "effect": "NoSchedule"}],
                    "containers": [{
                        "name": "gate",
                        "image": IMAGE,
                        "command": ["/bin/bash", "-ceu", "--"],
                        "args": [
                            "apt-get update\n"
                            "apt-get install --yes --no-install-recommends kubernetes-client\n"
                            "mkdir -p /workspace/source\n"
                            "python - <<'PY'\n"
                            "import gzip,json,pathlib\n"
                            "root=pathlib.Path('/workspace/source')\n"
                            "data=json.loads(gzip.decompress(pathlib.Path('/gate-input/runtime.json.gz').read_bytes()))\n"
                            "for name,text in data.items():\n"
                            " p=root/name; p.parent.mkdir(parents=True,exist_ok=True)\n"
                            " p.write_text(text)\n"
                            "PY\n"
                            "cd /workspace/source\n"
                            "token=/var/run/secrets/kubernetes.io/serviceaccount\n"
                            "printf 'apiVersion: v1\\nkind: Config\\nclusters:\\n"
                            "- name: incluster\\n  cluster:\\n"
                            "    server: https://kubernetes.default.svc\\n"
                            "    certificate-authority: %s/ca.crt\\nusers:\\n"
                            "- name: gate\\n  user:\\n    tokenFile: %s/token\\n"
                            "contexts:\\n- name: incluster\\n  context:\\n"
                            "    cluster: incluster\\n    user: gate\\n"
                            "    namespace: fleet-train-jobs\\n"
                            "current-context: incluster\\n' \"$token\" \"$token\" "
                            "> kubeconfig.yaml\n"
                            "chmod 600 kubeconfig.yaml\n"
                            "export KUBECONFIG=$PWD/kubeconfig.yaml\n"
                            "exec uv run --no-project --with httpx==0.28.1 --with pyyaml==6.0.3 "
                            "--with 'psycopg[binary]==3.3.5' python gate.py\n"
                        ],
                        "env": [{"name": "ROLLOUT_DATABASE_URL",
                                 "valueFrom": {"secretKeyRef": {
                                     "name": heldout_launch.ROLLOUT_DATABASE_SECRET,
                                     "key": heldout_launch.ROLLOUT_DATABASE_ENV}}}],
                        "resources": {"requests": {"cpu": "1", "memory": "2Gi"},
                                      "limits": {"cpu": "2", "memory": "4Gi"}},
                        "volumeMounts": [
                            {"name": "input", "mountPath": "/gate-input", "readOnly": True},
                            {"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True},
                        ],
                    }],
                    "volumes": [
                        {"name": "input", "projected": {"sources": projections}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared",
                                                                     "readOnly": True}},
                    ],
                },
            },
        },
    }


def _normalized_preview(response: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    objects = response.get("items") if response.get("kind") == "List" else [response]
    if not isinstance(objects, list):
        raise ValueError("server preview did not return objects")
    indexed = {(item.get("kind"), item.get("metadata", {}).get("name")): item for item in objects}
    expected = {(item["kind"], item["metadata"]["name"]): item for item in source["items"]}
    if set(indexed) != set(expected):
        raise ValueError("server preview object set changed")
    normalized = []
    for key in sorted(indexed):
        actual, wanted = indexed[key], expected[key]
        if key[0] == "Job":
            annotations = actual.get("metadata", {}).get("annotations", {})
            pod = actual.get("spec", {}).get("template", {}).get("spec", {})
            if annotations.get(heldout_launch.FAILURE_ALERT_ANNOTATION) != "off":
                raise ValueError("server-rendered root Job lacks failure-alerts off")
            if pod.get("priorityClassName") != "c1":
                raise ValueError("server-rendered Job priority changed")
            heldout_launch._assert_cpu_only(pod)  # noqa: SLF001
            stable = stable_job_preview(actual)
            if not heldout_launch._contains(stable, stable_job_preview(wanted)):  # noqa: SLF001
                raise ValueError("server-rendered Job changed sealed fields")
            normalized.append(stable)
        else:
            stable = {"apiVersion": actual.get("apiVersion"), "kind": actual.get("kind"),
                      "metadata": {"name": key[1], "namespace": NAMESPACE},
                      "immutable": actual.get("immutable"),
                      "binaryData": actual.get("binaryData")}
            wanted_stable = {"apiVersion": wanted.get("apiVersion"), "kind": wanted.get("kind"),
                             "metadata": {"name": key[1], "namespace": NAMESPACE},
                             "immutable": wanted.get("immutable"),
                             "binaryData": wanted.get("binaryData")}
            if stable != wanted_stable:
                raise ValueError("server-rendered ConfigMap changed")
            normalized.append(stable)
    return {"objects": normalized}


def _preview(bundle: dict[str, Any], context: str) -> str:
    cluster = heldout_launch.KubectlCluster(context)
    first = _normalized_preview(cluster.server_dry_run(NAMESPACE, bundle), bundle)
    second = _normalized_preview(cluster.server_dry_run(NAMESPACE, bundle), bundle)
    if first != second:
        raise ValueError("server preview changed across two calls")
    return _digest(first)


def render(*, packets_root: Path, output: Path, context: str) -> dict[str, Any]:
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new path below an existing directory")
    _assert_commit()
    plan, packet_bundles = _packet_plan(packets_root)
    runtime = _runtime(plan)
    config_maps = [_config_map(f"{NAME}-runtime", "runtime.json.gz", runtime)]
    for item_id, bundle in packet_bundles:
        config_maps.append(_config_map(f"{NAME}-{item_id}", "packet.json.gz", bundle))
    job = _job([item["metadata"]["name"] for item in config_maps])
    bundle = {"apiVersion": "v1", "kind": "List", "items": [*config_maps, job]}
    preview_sha256 = _preview(bundle, context)
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-h20-gate-", dir=output.parent))
    try:
        manifest = temporary / "gate.yaml"
        manifest.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        (temporary / "gate-plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt = {
            "schema": "cyber_fleet_heldout_budget_duplicate_gate_render_v1",
            "repo_commit": EXPECTED_COMMIT,
            "manifest": {"path": "gate.yaml", "sha256": _file_digest(manifest)},
            "plan": {"path": "gate-plan.json", "sha256": _file_digest(temporary / "gate-plan.json"),
                     "self_sha256": plan["sha256"]},
            "server_preview_sha256": preview_sha256,
            "server_preview_count": 2,
            "objects": {"jobs": 1, "config_maps": len(config_maps)},
            "job": {"name": NAME, "namespace": NAMESPACE,
                    "failure_alerts": "off", "priority_class": "c1", "gpus": 0,
                    "backoff_limit": 0, "active_deadline_seconds": 1800,
                    "sfs_read_only": True, "postgres_read_only": True},
            "cleanup": {
                "require_exact_job_uid": True,
                "exact_config_map_names": [item["metadata"]["name"] for item in config_maps],
            },
            "external_mutations": 0,
            "launch_performed": False,
        }
        receipt["sha256"] = _digest(receipt)
        (temporary / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    print(json.dumps(render(packets_root=args.packets_root, output=args.output,
                            context=args.context), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
