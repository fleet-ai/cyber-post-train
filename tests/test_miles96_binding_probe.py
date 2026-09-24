from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training import miles96_binding_probe as probe
from training import miles96_mechanics_canary as mechanics


def test_packet_is_inert_zero_gpu_read_only_and_alert_off() -> None:
    packet = probe.build_packet()
    proof = probe.validate_packet(packet)
    assert proof["gpus"] == 0
    assert packet["precreate"] == {
        "exact_name_duplicate_census_required": True,
        "server_dry_run_count": 2,
        "stable_preview_digests_must_match": True,
        "create_request_count": 1,
        "automatic_create_retry": False,
    }
    config_map, job = packet["bundle"]["items"]
    assert config_map["immutable"] is True
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert all(
        "nvidia.com/gpu" not in values
        for container in pod.get("initContainers", []) + pod["containers"]
        for values in container.get("resources", {}).values()
    )
    assert next(v for v in pod["volumes"] if v["name"] == "sfs")[
        "persistentVolumeClaim"
    ]["readOnly"] is True
    assert next(
        m for m in pod["containers"][0]["volumeMounts"] if m["name"] == "sfs"
    )["readOnly"] is True


def test_packet_embeds_exact_current_probe_sources() -> None:
    packet = probe.build_packet()
    data = packet["bundle"]["items"][0]["data"]
    assert data["driver.py"] == probe.DRIVER
    assert data["probe_module.py"] == Path(probe.__file__).read_text()
    assert set(data) == {"training_init.py", "mechanics.py", "probe_module.py", "driver.py"}


def test_embedded_probe_imports_in_an_isolated_interpreter(tmp_path: Path) -> None:
    data = probe.build_packet()["bundle"]["items"][0]["data"]
    package = tmp_path / "training"
    package.mkdir()
    (package / "__init__.py").write_text(data["training_init.py"])
    (package / "miles96_mechanics_canary.py").write_text(data["mechanics.py"])
    (package / "miles96_binding_probe.py").write_text(data["probe_module.py"])
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "import training.miles96_binding_probe as p;print(p.JOB_NAME)",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == probe.JOB_NAME


def test_task_authority_candidate_is_self_digesting_and_private_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/qualification/qwen38-miles96-task-authority-candidate-20260924-v1.json"
    )
    value = json.loads(path.read_text())
    body = {key: item for key, item in value.items() if key != "sha256"}
    assert value["sha256"] == "sha256:" + mechanics.digest(body)
    authority = {key: item for key, item in value["authority"].items() if key != "authority_receipt_sha256"}
    assert value["authority"]["authority_receipt_sha256"] == (
        "sha256:" + mechanics.digest(authority)
    )
    assert value["prompts_responses_flags_rewards_or_traces_included"] is False
