"""The CPU Job launcher supports only the exact v3 canary/full cell counts."""

from __future__ import annotations

import json
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_job as job
from evals.fleet import visible_action_collection_v3 as runtime
from training import collection_campaign_v3 as campaign

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"
SOURCE_COMMIT = "c" * 40
SOURCE_TREE = "d" * 40


def _load(name: str) -> dict:
    return json.loads((V2 / name).read_text())


def test_v3_one_family_pass4_canary_renders_exact_cpu_job(monkeypatch, tmp_path: Path) -> None:
    selection = _load("task-selection.json")
    task = selection["tasks"][0]
    selection = {**selection, "tasks": [task]}
    selection.pop("sha256")
    selection["sha256"] = "sha256:" + digest(selection)

    config = campaign._runtime_config(_load("eval-config.json"))  # noqa: SLF001
    config["name"] = "q38-base-actions-budget-canary-p4-v1"
    config["routes"]["base"]["task_versions"] = [task["task_version_id"]]
    config["collection_runtime"]["maximum_planned_cells"] = 4
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name, value in (("eval-config.json", config), ("task-selection.json", selection)):
        (inputs / name).write_text(json.dumps(value))
    plan = runtime.compile_eval(config, relative_to=inputs)
    authorization = runtime.build_operation_authorization(plan)
    packet_input = _load("collection-packet.json")
    packet_input["task_selection_sha256"] = selection["sha256"]
    packet = campaign._packet(packet_input, config, plan, authorization)  # noqa: SLF001
    (inputs / "operation-authorization.json").write_text(json.dumps(authorization))
    (inputs / "collection-packet.json").write_text(json.dumps(packet))

    def fake_git(_repo: Path, *arguments: str) -> str:
        return {
            ("status", "--porcelain"): "",
            ("rev-parse", "HEAD"): SOURCE_COMMIT,
            ("rev-parse", "HEAD^{tree}"): SOURCE_TREE,
        }[arguments]

    monkeypatch.setattr(job, "_git", fake_git)
    result = job.prepare_packet(
        repo_root=ROOT,
        config_path=inputs / "eval-config.json",
        task_selection_path=inputs / "task-selection.json",
        authorization_path=inputs / "operation-authorization.json",
        collection_packet_path=inputs / "collection-packet.json",
        output=tmp_path / "packet",
        expected_source_commit=SOURCE_COMMIT,
    )
    package = job.build_package(Path(result["packet"]))

    assert result["planned_cells"] == 4
    assert package.packet.value["operation"]["worker_id"] == "base-v3"
    assert package.packet.value["operation"]["planned_cells"] == 4
    assert set(package.packet.value["source"]["files"]) == set(job.SOURCE_FILES_V3) | {"run.sh"}
    assert package.job["metadata"]["annotations"][job.FAILURE_ALERT_ANNOTATION] == "off"
    assert package.job["spec"]["activeDeadlineSeconds"] == job.ACTIVE_DEADLINE_SECONDS
    assert package.job["spec"]["backoffLimit"] == 0
    pod = package.job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert pod["restartPolicy"] == "Never"
    for container in pod["initContainers"] + pod["containers"]:
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
