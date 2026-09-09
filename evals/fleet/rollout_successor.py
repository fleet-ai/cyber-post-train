"""Render one identity-safe successor from a completed rollout Job; never submit it."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.fleet.rollout_refiller import (
    REFILL_LABEL,
    WORKER_PREFIX_ANNOTATION,
)

POSTGRES_CLIENT_LABEL = "cyber-post-train.fleet.ai/postgres-client"
POSTGRES_MODULES = ("rollout_ledger.py", "rollout_postgres.py", "rollout_worker.py")
ALERT_SAFE_POLICY_ANNOTATION = "cyber-post-train.fleet.ai/alert-safe-job-policy"
ALERT_SAFE_POLICY_VERSION = "retry-without-terminal-failure-v1"
ALERT_SAFE_BACKOFF_LIMIT = 2_147_483_647
CONCURRENCY_STAGE_ANNOTATION = "cyber-post-train.fleet.ai/concurrency-stage"
WORKLOAD_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
WORKLOAD_PRIORITY_UID_ANNOTATION = "cyber-post-train.fleet.ai/workload-priority-class-uid"

RUNTIME_METADATA = {
    "creationTimestamp",
    "deletionGracePeriodSeconds",
    "deletionTimestamp",
    "generation",
    "managedFields",
    "ownerReferences",
    "resourceVersion",
    "selfLink",
    "uid",
}
CONTROLLER_KEYS = {
    "batch.kubernetes.io/controller-uid",
    "batch.kubernetes.io/job-name",
    "controller-uid",
    "job-name",
    "kueue.x-k8s.io/podset",
    "kueue.x-k8s.io/workload",
}


class SuccessorError(RuntimeError):
    """A fail-closed successor rendering or creation error."""


def _replace_strings(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _clean_metadata(metadata: dict[str, Any]) -> None:
    for key in RUNTIME_METADATA:
        metadata.pop(key, None)
    for field in ("annotations", "labels"):
        values = metadata.get(field)
        if not isinstance(values, dict):
            continue
        for key in CONTROLLER_KEYS:
            values.pop(key, None)
        if not values:
            metadata.pop(field, None)


def clone_config_map(
    source: dict[str, Any],
    *,
    source_name: str,
    new_name: str,
    old_worker_id: str,
    new_worker_id: str,
    refiller_id: str,
) -> dict[str, Any]:
    result = _replace_strings(
        copy.deepcopy(source), ((source_name, new_name), (old_worker_id, new_worker_id))
    )
    result.pop("status", None)
    _clean_metadata(result.setdefault("metadata", {}))
    result["metadata"]["name"] = new_name
    result["metadata"].setdefault("labels", {})[REFILL_LABEL] = refiller_id
    _validate_clone(
        result,
        source_name=source_name,
        new_name=new_name,
        old_worker_id=old_worker_id,
        new_worker_id=new_worker_id,
        require_worker=True,
    )
    return result


def clone_job(
    source: dict[str, Any],
    *,
    source_name: str,
    new_name: str,
    old_worker_id: str,
    new_worker_id: str,
    refiller_id: str,
) -> dict[str, Any]:
    result = _replace_strings(
        copy.deepcopy(source), ((source_name, new_name), (old_worker_id, new_worker_id))
    )
    result.pop("status", None)
    metadata = result.setdefault("metadata", {})
    _clean_metadata(metadata)
    metadata["name"] = new_name
    metadata.setdefault("labels", {})[REFILL_LABEL] = refiller_id
    # A previous priority exception is not authorization for its successor.
    metadata["labels"].pop(WORKLOAD_PRIORITY_LABEL, None)
    metadata.setdefault("annotations", {}).pop(WORKLOAD_PRIORITY_UID_ANNOTATION, None)
    spec = result.setdefault("spec", {})
    spec.pop("selector", None)
    spec.pop("manualSelector", None)
    # The Fleet status monitor pages every Job UID that reaches Failed=True.
    # Rollout controllers already turn handled rollout outcomes into a durable
    # receipt and exit zero.  Keep unexpected Pod failures observable and
    # retryable without letting a single infrastructure/process failure turn
    # the Job terminally failed before an operator can inspect and suspend it.
    spec["backoffLimit"] = ALERT_SAFE_BACKOFF_LIMIT
    spec.pop("activeDeadlineSeconds", None)
    spec.pop("backoffLimitPerIndex", None)
    spec.pop("maxFailedIndexes", None)
    spec.pop("podFailurePolicy", None)
    metadata.setdefault("annotations", {})[ALERT_SAFE_POLICY_ANNOTATION] = ALERT_SAFE_POLICY_VERSION
    template_metadata = spec.setdefault("template", {}).setdefault("metadata", {})
    _clean_metadata(template_metadata)
    template_metadata.setdefault("labels", {})[REFILL_LABEL] = refiller_id
    _validate_clone(
        result,
        source_name=source_name,
        new_name=new_name,
        old_worker_id=old_worker_id,
        new_worker_id=new_worker_id,
        require_worker=False,
    )
    if spec.get("backoffLimit") != ALERT_SAFE_BACKOFF_LIMIT:
        raise SuccessorError("successor lacks the alert-safe retry ceiling")
    if "activeDeadlineSeconds" in spec:
        raise SuccessorError("successor must not have a failure-producing active deadline")
    if "podFailurePolicy" in spec:
        raise SuccessorError("successor must not carry a source pod failure policy")
    command = json.dumps(spec.get("template", {}).get("spec", {}).get("containers", []))
    if "fleet-rollout-controller-clean-exit-v1" not in command:
        raise SuccessorError("successor is missing the clean-exit controller wrapper")
    return result


def _validate_clone(
    value: dict[str, Any],
    *,
    source_name: str,
    new_name: str,
    old_worker_id: str,
    new_worker_id: str,
    require_worker: bool,
) -> None:
    encoded = json.dumps(value, sort_keys=True)
    if source_name in encoded or old_worker_id in encoded:
        raise SuccessorError("source Kubernetes or worker identity survived clone rendering")
    if new_name not in encoded:
        raise SuccessorError("new Kubernetes identity is absent from rendered clone")
    if require_worker and new_worker_id not in encoded:
        raise SuccessorError("new worker identity is absent from rendered ConfigMap")
    if any(f'"{key}"' in encoded for key in CONTROLLER_KEYS):
        raise SuccessorError("controller or Kueue runtime metadata survived clone rendering")


def configure_postgres(
    config_map: dict[str, Any],
    job: dict[str, Any],
    *,
    repo_root: Path,
    postgres_secret: str,
    worker_prefix: str,
) -> None:
    data = config_map.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("run.sh"), str):
        raise SuccessorError("source ConfigMap lacks the rollout bootstrap")
    for module in POSTGRES_MODULES:
        path = repo_root / "evals" / "fleet" / module
        if not path.is_file():
            raise SuccessorError(f"PostgreSQL successor source is missing {module}")
        data[module] = path.read_text(encoding="utf-8")
    run_script = data["run.sh"]
    old_modules = "rollout_worker.py rollout_campaign.py rollout_ledger.py opencode_self_hosted.py"
    new_modules = (
        "rollout_worker.py rollout_campaign.py rollout_ledger.py rollout_postgres.py "
        "opencode_self_hosted.py"
    )
    if run_script.count(old_modules) != 1:
        raise SuccessorError("source bootstrap module list changed")
    run_script = run_script.replace(old_modules, new_modules)
    old_runtime = "uv run --no-project --with httpx==0.28.1 python -m evals.fleet.rollout_worker"
    new_runtime = (
        "uv run --no-project --with httpx==0.28.1 "
        "--with 'psycopg[binary]==3.3.5' python -m evals.fleet.rollout_worker"
    )
    if run_script.count(old_runtime) != 1:
        raise SuccessorError("source rollout runtime command changed")
    run_script = run_script.replace(old_runtime, new_runtime)
    old_database = '--database "$state/ledger.sqlite3"'
    if run_script.count(old_database) != 1:
        raise SuccessorError("source SQLite database argument changed")
    run_script = run_script.replace(old_database, "--postgres-dsn-env ROLLOUT_DATABASE_URL")
    data["run.sh"] = run_script

    metadata = job.setdefault("metadata", {})
    metadata.setdefault("annotations", {})[WORKER_PREFIX_ANNOTATION] = worker_prefix
    pod = job.setdefault("spec", {}).setdefault("template", {})
    pod.setdefault("metadata", {}).setdefault("labels", {})[POSTGRES_CLIENT_LABEL] = "true"
    containers = pod.setdefault("spec", {}).get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise SuccessorError("successor must contain exactly one evaluator container")
    environment = containers[0].setdefault("env", [])
    if any(item.get("name") == "ROLLOUT_DATABASE_URL" for item in environment):
        raise SuccessorError("source already defines ROLLOUT_DATABASE_URL")
    environment.append(
        {
            "name": "ROLLOUT_DATABASE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": postgres_secret,
                    "key": "ROLLOUT_DATABASE_URL",
                }
            },
        }
    )

    encoded = json.dumps({"config_map": config_map, "job": job}, sort_keys=True)
    if old_database in encoded or "sqlite3.connect" in data["rollout_worker.py"]:
        raise SuccessorError("PostgreSQL successor still selects the SQLite runtime")


def _kubectl_json(
    arguments: Sequence[str], *, expect_absent: bool = False
) -> dict[str, Any] | None:
    completed = subprocess.run(["kubectl", *arguments], check=False, capture_output=True, text=True)
    if expect_absent and completed.returncode != 0 and "NotFound" in completed.stderr:
        return None
    if completed.returncode != 0:
        raise SuccessorError("kubectl read failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SuccessorError("kubectl returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SuccessorError("kubectl did not return an object")
    return value


def render_live(
    *,
    namespace: str,
    source_name: str,
    expected_source_uid: str,
    new_name: str,
    old_worker_id: str,
    new_worker_id: str,
    refiller_id: str,
    repo_root: Path | None = None,
    postgres_secret: str | None = None,
    concurrency_stage: int | None = None,
    workload_priority_class: str | None = None,
    expected_priority_class_uid: str | None = None,
) -> dict[str, Any]:
    if bool(workload_priority_class) != bool(expected_priority_class_uid):
        raise SuccessorError("priority class and its expected UID must be supplied together")
    if _kubectl_json(["-n", namespace, "get", "job", new_name, "-o", "json"], expect_absent=True):
        raise SuccessorError("new Job name already exists")
    if _kubectl_json(
        ["-n", namespace, "get", "configmap", new_name, "-o", "json"], expect_absent=True
    ):
        raise SuccessorError("new ConfigMap name already exists")
    source_job = _kubectl_json(["-n", namespace, "get", "job", source_name, "-o", "json"])
    source_config = _kubectl_json(["-n", namespace, "get", "configmap", source_name, "-o", "json"])
    if source_job is None or source_config is None:
        raise SuccessorError("source objects are missing")
    if (source_job.get("metadata") or {}).get("uid") != expected_source_uid:
        raise SuccessorError("source Job UID changed")
    conditions = (source_job.get("status") or {}).get("conditions") or []
    if any(c.get("type") == "Failed" and c.get("status") == "True" for c in conditions):
        raise SuccessorError("source Job has a terminal failure condition")
    if not any(c.get("type") == "Complete" and c.get("status") == "True" for c in conditions):
        raise SuccessorError("source Job is not complete")
    config_map = clone_config_map(
        source_config,
        source_name=source_name,
        new_name=new_name,
        old_worker_id=old_worker_id,
        new_worker_id=new_worker_id,
        refiller_id=refiller_id,
    )
    job = clone_job(
        source_job,
        source_name=source_name,
        new_name=new_name,
        old_worker_id=old_worker_id,
        new_worker_id=new_worker_id,
        refiller_id=refiller_id,
    )
    if concurrency_stage is not None:
        if concurrency_stage < 1:
            raise SuccessorError("concurrency stage must be positive")
        job.setdefault("metadata", {}).setdefault("annotations", {})[
            CONCURRENCY_STAGE_ANNOTATION
        ] = str(concurrency_stage)
    if workload_priority_class is not None:
        priority = _kubectl_json(
            ["get", "workloadpriorityclass.kueue.x-k8s.io", workload_priority_class, "-o", "json"]
        )
        if (
            priority is None
            or priority.get("metadata", {}).get("uid") != expected_priority_class_uid
        ):
            raise SuccessorError("workload priority class identity changed")
        if type(priority.get("value")) is not int:
            raise SuccessorError("workload priority class has no integer priority")
        # Explicit, reviewed opt-in only. Do not change Pod priority or admission.
        job["metadata"].setdefault("labels", {})[WORKLOAD_PRIORITY_LABEL] = workload_priority_class
        job["metadata"].setdefault("annotations", {})[WORKLOAD_PRIORITY_UID_ANNOTATION] = (
            expected_priority_class_uid
        )
    if bool(repo_root) != bool(postgres_secret):
        raise SuccessorError("PostgreSQL repo root and secret must be supplied together")
    if repo_root is not None and postgres_secret is not None:
        configure_postgres(
            config_map,
            job,
            repo_root=repo_root,
            postgres_secret=postgres_secret,
            worker_prefix=new_worker_id,
        )
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            config_map,
            job,
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="fleet-train-jobs")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--expected-source-uid", required=True)
    parser.add_argument("--new-name", required=True)
    parser.add_argument("--old-worker-id", required=True)
    parser.add_argument("--new-worker-id", required=True)
    parser.add_argument("--refiller-id", default="q38-glm53-pass4-v1")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--postgres-secret")
    parser.add_argument("--concurrency-stage", type=int)
    parser.add_argument("--workload-priority-class", help="Explicitly authorized Kueue priority")
    parser.add_argument("--expected-priority-class-uid", help="Immutable priority class UID")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    value = render_live(
        namespace=arguments.namespace,
        source_name=arguments.source_name,
        expected_source_uid=arguments.expected_source_uid,
        new_name=arguments.new_name,
        old_worker_id=arguments.old_worker_id,
        new_worker_id=arguments.new_worker_id,
        refiller_id=arguments.refiller_id,
        repo_root=arguments.repo_root,
        postgres_secret=arguments.postgres_secret,
        concurrency_stage=arguments.concurrency_stage,
        workload_priority_class=arguments.workload_priority_class,
        expected_priority_class_uid=arguments.expected_priority_class_uid,
    )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
