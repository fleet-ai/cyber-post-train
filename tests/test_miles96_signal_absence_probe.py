from __future__ import annotations

from pathlib import Path

from training import miles96_signal_absence_probe as probe


def test_probe_observes_only_exact_path_absence(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(probe, "OUTPUT", output)
    receipt = probe.observe()
    assert receipt["status"] == "absent"
    assert receipt["gpus"] == 0
    assert receipt["read_only"] is True
    output.mkdir()
    assert probe.observe()["status"] == "present"


def test_packet_is_zero_gpu_read_only_create_once_and_alerts_off() -> None:
    packet = probe.build_packet()
    proof = probe.validate_packet(packet)
    assert proof["gpus"] == 0
    config_map, job = packet["bundle"]["items"]
    assert config_map["immutable"] is True
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert pod["securityContext"]["runAsUser"] == 1000
    assert pod["containers"][0]["volumeMounts"][1]["readOnly"] is True
    assert packet["execution_sequence"] == {
        "config_map_create_request_count": 1,
        "job_create_request_count": 1,
        "create_retries_allowed": False,
        "controller_managed_unsuspend": True,
        "operator_patch_request_count": 0,
        "exact_uid_monitor_and_cleanup_required": True,
    }
