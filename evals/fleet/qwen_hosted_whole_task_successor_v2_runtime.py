"""Runtime boundary for the fresh-generation hosted-Qwen successors."""

from __future__ import annotations

import argparse
import contextlib
import os
import stat
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_whole_task_successor_v2 as successor
from evals.fleet import self_hosted


def _load_private_regular(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("fresh hosted whole-task private input path drifted")
    try:
        info = path.lstat()
        parent = path.parent.stat()
    except OSError as exc:
        raise RuntimeError("fresh hosted whole-task private input is absent") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise RuntimeError("fresh hosted whole-task private input mode drifted")
    return successor.load(path)


def _load_bound_inputs(
    plan: dict[str, Any], package_source_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    plans = successor.build_plans(Path(plan["repo_root"]))
    package_source = _load_private_regular(package_source_path)
    if package_source.get("receipt_sha256") != os.environ.get(
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
    ):
        raise RuntimeError("fresh hosted whole-task package digest drifted")
    if plan != plans.get(plan.get("controller")):
        raise RuntimeError("fresh hosted whole-task runtime plan drifted")
    return plans, package_source


def run(
    plan: dict[str, Any],
    *,
    out: Path,
    proxy: Path,
    diagnostic_root: Path,
    package_source_path: Path,
) -> dict[str, Any]:
    if out.is_symlink():
        raise RuntimeError("fresh hosted whole-task output root is unsafe")
    plans, package_source = _load_bound_inputs(plan, package_source_path)
    successor.load_runtime_release(plans, package_source)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    state: dict[str, Any] = {"stage": None, "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        state.update(stage=stage, item=item)
        if stage == "07-claim-written":
            if item is None:
                raise RuntimeError("fresh hosted whole-task model boundary item is missing")
            provider.mark_model_boundary(item)

    prior_bulk = engine.bulk
    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: successor.load_runtime_release(plans, package_source),
            claim_provider=provider,
            stage_observer=observe,
        )
    except Exception as exc:
        item = state["item"]
        body: dict[str, Any] = {
            "schema_version": "fleet-qwen38-hosted-whole-task-runtime-failure-v2",
            "status": "FAILED",
            "controller": plan["controller"],
            "plan_sha256": plan["plan_sha256"],
            "last_completed_stage": state["stage"],
            "reservation_committed": out.joinpath("RESERVATION.json").is_file(),
            "error_type": type(exc).__name__,
            "error_sha256": self_hosted.sha256(str(exc).encode()),
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
            "scores_included": False,
            "prompts_or_traces_included": False,
            "credentials_included": False,
        }
        if item is not None:
            body.update(
                cell_id=item["cell_id"],
                execution_id=item["execution_id"],
                selection_rank=item["selection_rank"],
                attempt=item["attempt"],
            )
        receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
        diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            self_hosted.write_json_once(diagnostic_root / "FAILED.json", receipt)
        raise
    finally:
        engine.bulk = prior_bulk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    args = parser.parse_args()
    run(
        successor.load(args.plan),
        out=args.out,
        proxy=args.proxy,
        diagnostic_root=args.diagnostic_root,
        package_source_path=args.package_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
