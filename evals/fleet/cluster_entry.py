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
from evals.fleet import (
    evaluate,
    model_artifact,
    rollout_postgres,
    rollout_worker,
)

MODEL_ARTIFACT_V2_PACKET_SCHEMA = "cyber_fleet_eval_model_artifact_packet_v2"
MODEL_ARTIFACT_V3_PACKET_SCHEMA = "cyber_fleet_eval_model_artifact_packet_v3"


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


def model_artifact_plan(config: dict, proof: dict) -> dict | None:
    binding = config.get("model_artifact_binding")
    if binding is None:
        if proof:
            raise ValueError("model artifact proof is not allowed without a config binding")
        return None
    plan = {
        "schema": "fleet_eval_model_artifact_plan_v1",
        "campaign_name": config.get("name"),
        "binding": binding,
        "proof": proof,
    }
    return {**plan, "sha256": digest(plan)}


def execute(args: argparse.Namespace) -> dict:
    config_path, output = Path(args.config), Path(args.output)
    if output.exists():
        raise FileExistsError("create-once evaluation output already exists")
    config = evaluate.read_mapping(config_path)
    artifact_packet = (
        evaluate.read_mapping(Path(args.model_artifact_binding))
        if getattr(args, "model_artifact_binding", None)
        else None
    )
    artifact_acceptance = (
        Path(args.model_artifact_acceptance)
        if getattr(args, "model_artifact_acceptance", None)
        else None
    )
    if artifact_packet is not None and artifact_packet.get("campaign_name") != config.get("name"):
        raise ValueError("model artifact packet belongs to a different evaluation campaign")
    artifact_validator = model_artifact
    if artifact_packet is not None and artifact_packet.get("schema") == (
        MODEL_ARTIFACT_V2_PACKET_SCHEMA
    ):
        # Imported only for a v2 packet so historical v1 bootstrap bundles do
        # not need to stage a module they can never execute.
        from evals.fleet import model_artifact_v2  # noqa: PLC0415

        artifact_validator = model_artifact_v2
    if artifact_packet is not None and artifact_packet.get("schema") == (
        MODEL_ARTIFACT_V3_PACKET_SCHEMA
    ):
        # v3 keeps the same strict receipt chain while preserving whether the
        # reviewed reload was owned by a bounded Pod or by a RayJob.
        from evals.fleet import model_artifact_v3  # noqa: PLC0415

        artifact_validator = model_artifact_v3
    # Reopen the complete local checkpoint/export/reload receipt chain before
    # pulling images, writing output, contacting Fleet, or creating a database.
    artifact_proof = artifact_validator.validate_live_models(
        config.get("models", {}),
        artifact_packet,
        config_binding=config.get("model_artifact_binding"),
        packet_path=Path(args.model_artifact_binding)
        if getattr(args, "model_artifact_binding", None)
        else None,
        acceptance_path=artifact_acceptance,
    )
    stage_images(
        config=config,
        harness_tar=Path(args.harness_tar),
        harness_receipt=Path(args.harness_receipt),
        receipt_sha256=args.harness_receipt_sha256,
    )
    scientific_config = dict(config)
    scientific_config.pop("model_artifact_binding", None)
    prepared = evaluate.prepare(scientific_config, output, relative_to=config_path.parent)
    artifact_plan = model_artifact_plan(config, artifact_proof)
    artifact_plan_path = output / "MODEL_ARTIFACT_PLAN.json"
    if artifact_plan is not None:
        rollout_worker._safe_write_once(artifact_plan_path, artifact_plan)  # noqa: SLF001
    preflight = evaluate.preflight(output)
    if artifact_proof != artifact_validator.validate_live_models(
        config["models"],
        artifact_packet,
        config_binding=config.get("model_artifact_binding"),
        packet_path=Path(args.model_artifact_binding)
        if getattr(args, "model_artifact_binding", None)
        else None,
        acceptance_path=artifact_acceptance,
    ):
        raise ValueError("model artifact receipts changed during preflight")
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
        "model_artifacts": artifact_proof,
        "routes": terminals,
        "summary": summary,
    }
    if config.get("model_artifact_binding") is not None:
        terminal["model_artifact_binding"] = config["model_artifact_binding"]
        terminal["model_artifact_plan"] = {
            "path": artifact_plan_path.name,
            "file_sha256": _sha256(artifact_plan_path),
            "sha256": artifact_plan["sha256"],
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
    parser.add_argument("--model-artifact-binding")
    parser.add_argument("--model-artifact-acceptance")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(execute(parse_args()), sort_keys=True))
