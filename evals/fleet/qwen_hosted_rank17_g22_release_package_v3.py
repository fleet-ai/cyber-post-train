"""Render the held finite-path hosted-Qwen rank-17 release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen_hosted_generation18_package as job_base
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as observer_v1
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v3 as observer
from evals.fleet import qwen_hosted_rank17_g22_release_package_v1 as package_v1
from evals.fleet import self_hosted

JOB_NAME = "chris-q38-hosted-r017-release-gate-g22-v3"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
ENGINE_PATH = Path("evals/fleet/exact_pass4_bulk_runtime_v3.py")
ENGINE_SHA256 = "sha256:b80b39412906d3cb5163e55a882d0f5ae936fd03417fae61bc0a497db1f18daa"
HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-release-held-v3.json"
)


def _prove_engine_paths(root: Path) -> None:
    source = (root / ENGINE_PATH).read_text()
    if self_hosted.sha256(source.encode()) != ENGINE_SHA256 or any(
        marker not in source
        for marker in (
            'return execution_id.removeprefix("sha256:") + ".json"',
            'out / "accepted" / f"{item[\'run_id\']}.json"',
            'attempt_out = out / "attempts" / item["run_id"]',
        )
    ):
        raise ValueError("rank17 engine path contract drifted")


def build_binding(root: Path) -> dict[str, Any]:
    _prove_engine_paths(root)
    binding = package_v1.build_binding(root)
    observer_v1.validate_binding(binding)
    observer.finite_paths(binding)
    return binding


def render(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    names = (
        "qwen_hosted_rank17_g22_release_observer_v1.py",
        "qwen_hosted_rank17_g22_release_observer_v3.py",
    )
    data = {name: (root / "evals/fleet" / name).read_text() for name in names}
    data["binding.json"] = json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    package_body = {
        "schema_version": observer.PACKAGE_SCHEMA,
        "files": {name: observer_v1.sha256(value.encode()) for name, value in sorted(data.items())},
        "file_count": len(data),
    }
    data["package-source.json"] = (
        json.dumps(observer_v1.seal(package_body), sort_keys=True, separators=(",", ":")) + "\n"
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": observer_v1.NAMESPACE},
        "immutable": True,
        "data": data,
    }
    job = job_base._base_job(JOB_NAME, CONFIGMAP_NAME, scored=False)  # noqa: SLF001
    experiment = "q38-hosted-r017-release-gate-g22-v3"
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = experiment
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/create-once": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
            "cyber-post-train.fleet.ai/diagnostic-only": "true",
            "cyber-post-train.fleet.ai/score-free": "true",
            "cyber-post-train.fleet.ai/plan-commit": observer_v1.PLAN_COMMIT,
            "cyber-post-train.fleet.ai/supersedes-job-uid": (
                "4c5a5991-c00f-4c09-b3b2-c45c6a825a52"
            ),
        }
    )
    job["spec"]["activeDeadlineSeconds"] = 900
    job["spec"]["template"]["metadata"]["labels"][
        "cyber-post-train.fleet.ai/experiment"
    ] = experiment
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        "exec python /bootstrap/qwen_hosted_rank17_g22_release_observer_v3.py "
        "--binding /bootstrap/binding.json --package-source /bootstrap/package-source.json "
        f"--output {OUTPUT_ROOT}/OBSERVATION.json"
    ]
    container["resources"] = {
        "requests": {"cpu": "100m", "memory": "256Mi", "ephemeral-storage": "256Mi"},
        "limits": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "1Gi"},
    }
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def held(root: Path) -> dict[str, Any]:
    binding = build_binding(root)
    rendered = render(root)
    configmap, job = rendered["items"]
    paths = observer.finite_paths(binding)
    return observer_v1.seal(
        {
            "schema_version": "fleet-qwen38-hosted-rank17-g22-release-held-v3",
            "status": "HELD_PENDING_INDEPENDENT_REVIEW",
            "launch_authorized": False,
            "scoring_authorized": False,
            "plan_commit": observer_v1.PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "binding_sha256": binding["binding_sha256"],
            "authoritative_tally": observer_v1.EXPECTED_TALLY,
            "terminal_predecessor": {
                "job_uid": "4c5a5991-c00f-4c09-b3b2-c45c6a825a52",
                "pod_uid": "ecaab3e5-6bf0-4226-8eed-46af047d5b48",
                "status": "TERMINAL_FAILED_NO_RETRY",
                "failure_code": "scan_receipt_invalid",
                "effects": 0,
            },
            "finite_path_authority": {
                "engine_path": str(ENGINE_PATH),
                "engine_sha256": ENGINE_SHA256,
                "canonical_claim_path_count": len(paths["canonical_claim"]),
                "legacy_per_run_accepted_path_count": len(paths["legacy_per_run_accepted"]),
                "engine_accepted_registry_path_count": len(paths["engine_accepted_registry"]),
                "recursive_historical_jobs_scan": False,
                "authoritative_task_model_version_session_inventory": True,
            },
            "observer": {
                "job_name": JOB_NAME,
                "configmap_name": CONFIGMAP_NAME,
                "output_root": OUTPUT_ROOT,
                "manifest_sha256": self_hosted.sha256(
                    json.dumps(rendered, sort_keys=True, separators=(",", ":")).encode()
                ),
                "configmap_immutable": configmap["immutable"],
                "create_once": True,
                "score_free": True,
                "preemption_policy": job["spec"]["template"]["spec"]["preemptionPolicy"],
                "priority_class": job["spec"]["template"]["spec"]["priorityClassName"],
            },
            "required_clear_observation": {
                "canonical_claim_paths_absent": 4,
                "legacy_per_run_accepted_paths_absent": 4,
                "engine_accepted_registry_paths_absent": 4,
                "authoritative_session_collisions": 0,
                "fresh_job_configmap_pod_collisions": 0,
                "fresh_sfs_output_collisions": 0,
                "endpoint_lease_slots_simultaneously_free": 2,
                "observer_job_succeeded": True,
                "observer_restarts": 0,
                "independent_terminal_review_required": True,
            },
            "side_effects": {
                "claims_created": 0,
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scored_jobs_created": 0,
            },
            "privacy": {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = yaml.safe_dump(render(args.root.resolve(strict=True)), sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
