from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_generation18 as g18
from evals.fleet import qwen_hosted_generation18_package as package

ROOT = Path(__file__).resolve().parents[1]


def test_g18_canary_is_one_fresh_rank5_cell() -> None:
    plan = g18.load(ROOT / g18.PLAN_PATH)
    g18.validate_plan(plan)
    assert len(plan["attempts"]) == 1
    assert plan["attempts"][0]["selection_rank"] == 5
    assert plan["attempts"][0]["attempt"] == 1
    assert plan["attempts"][0]["execution_generation"] == 18
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]


def test_g18_package_preserves_queue_and_nonpreemption() -> None:
    rendered = package.render(ROOT)
    jobs = [row for row in rendered["items"] if row["kind"] == "Job"]
    assert {row["metadata"]["name"] for row in jobs} == {
        g18.PREFLIGHT_JOB_NAME,
        g18.JOB_NAME,
    }
    for job in jobs:
        assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
    scored = next(row for row in jobs if row["metadata"]["name"] == g18.JOB_NAME)
    refs = [
        env["valueFrom"]["secretKeyRef"]["name"]
        for env in scored["spec"]["template"]["spec"]["containers"][0]["env"]
        if "valueFrom" in env and "secretKeyRef" in env["valueFrom"]
    ]
    assert refs == [g18.FLEET_API_KEY_SECRET]


def test_scored_render_requires_digest_for_live_preflight() -> None:
    with pytest.raises(ValueError, match="preflight digest"):
        package.render(ROOT, "not-a-digest")
