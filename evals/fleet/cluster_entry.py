"""Create-once cluster entrypoint for one matched Fleet evaluation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from cyber_post_train.jobs import digest
from evals.fleet import evaluate, rollout_postgres, rollout_worker


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def dedicated_dsn(admin_dsn: str, database: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,62}", database) is None:
        raise ValueError("dedicated database name is invalid")
    parsed = urlsplit(admin_dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("PostgreSQL administrator DSN is invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + database, parsed.query, ""))


def create_database_once(admin_dsn: str, database: str) -> str:
    dsn = dedicated_dsn(admin_dsn, database)
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=30) as connection:
        exists = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = %s)", (database,)
        ).fetchone()[0]
        if exists:
            raise FileExistsError("dedicated evaluation database already exists")
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    return dsn


def stage_images(
    *,
    config: dict,
    harness_tar: Path,
    harness_receipt: Path,
    receipt_sha256: str,
) -> None:
    if _sha256(harness_receipt) != receipt_sha256:
        raise ValueError("harness build receipt digest differs")
    receipt = evaluate.read_mapping(harness_receipt)
    if (
        receipt.get("schema") != "fleet_harness_build_v1"
        or receipt.get("tar_sha256") != _sha256(harness_tar)
        or receipt.get("image_id") != config["images"]["agent"]
        or receipt.get("platform") != "linux/amd64"
        or receipt.get("release_asset_sha256") != config["harness"]["release_asset_sha256"]
    ):
        raise ValueError("harness image build evidence differs")
    subprocess.run(["docker", "load", "--input", str(harness_tar)], check=True, timeout=600)
    subprocess.run(
        ["docker", "image", "inspect", config["images"]["agent"]],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=30,
    )
    subprocess.run(["docker", "pull", config["images"]["proxy"]], check=True, timeout=600)


def execute(args: argparse.Namespace) -> dict:
    config_path, output = Path(args.config), Path(args.output)
    if output.exists():
        raise FileExistsError("create-once evaluation output already exists")
    config = evaluate.read_mapping(config_path)
    stage_images(
        config=config,
        harness_tar=Path(args.harness_tar),
        harness_receipt=Path(args.harness_receipt),
        receipt_sha256=args.harness_receipt_sha256,
    )
    prepared = evaluate.prepare(config, output, relative_to=config_path.parent)
    preflight = evaluate.preflight(output)
    admin_dsn = os.environ.get("ROLLOUT_DATABASE_URL")
    if not admin_dsn:
        raise ValueError("ROLLOUT_DATABASE_URL is required")
    dsn = create_database_once(admin_dsn, args.database)
    initialized = rollout_postgres.initialize(dsn, output / "plan.csv")
    plan = evaluate.load(output)
    rollout_worker._safe_write_once(  # noqa: SLF001
        output / "DATABASE.json",
        {
            "schema": "fleet_eval_database_v1",
            "database": args.database,
            "plan_sha256": plan["sha256"],
            "cell_count": initialized["cells"],
        },
    )

    def one(route: str) -> dict:
        limit = len(plan["routes"][route]["task_versions"]) * plan["pass_k"]
        return evaluate.run(
            output,
            dsn=dsn,
            route=route,
            worker_id=route + "-v1",
            limit=limit,
        )

    with ThreadPoolExecutor(max_workers=len(plan["routes"])) as pool:
        routes = sorted(plan["routes"])
        terminals = dict(zip(routes, pool.map(one, routes), strict=True))
    summary = rollout_postgres.summary(dsn)
    terminal = {
        "schema": "fleet_eval_campaign_terminal_v1",
        "plan_sha256": plan["sha256"],
        "prepared": prepared,
        "preflight": preflight,
        "routes": terminals,
        "summary": summary,
    }
    terminal["sha256"] = digest(terminal)
    rollout_worker._safe_write_once(output / "EVAL_TERMINAL.json", terminal)  # noqa: SLF001
    if summary["by_state"].get("accepted") != summary["total"]:
        raise RuntimeError("matched evaluation ended without one accepted result per cell")
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--harness-tar", required=True)
    parser.add_argument("--harness-receipt", required=True)
    parser.add_argument("--harness-receipt-sha256", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(execute(parse_args()), sort_keys=True))
