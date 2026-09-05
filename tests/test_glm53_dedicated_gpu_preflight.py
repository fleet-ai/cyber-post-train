import json
import os
import subprocess
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_gpu_preflight as preflight


def _inventory(memories: list[int]) -> str:
    return "\n".join(
        f"{index}, NVIDIA B300 SXM6 PC, {memory}, 10.3"
        for index, memory in enumerate(memories)
    )


def test_exact_homogeneous_b300_inventory_passes() -> None:
    rows = preflight.validate_nvidia_smi_csv(
        _inventory([preflight.EXPECTED_MEMORY_MIB] * 8)
    )
    assert len(rows) == 8


def test_v8_observed_inconsistent_gpu_memory_fails_closed() -> None:
    observed = [275040, 275040, 275040, 275040, 183359, 275040, 275040, 275040]
    with pytest.raises(ValueError, match="memory inventory is inconsistent"):
        preflight.validate_nvidia_smi_csv(_inventory(observed))


def test_missing_gpu_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        preflight.validate_nvidia_smi_csv(_inventory([275040] * 7))


def test_future_lifecycle_gates_hardware_and_persists_pre_ready_exit() -> None:
    source = Path("evals/fleet/scripts/glm53_dedicated_lifecycle_v2.sh").read_text()
    assert "HARDWARE-PREFLIGHT.json" in source
    assert "{row[\"memory_mib\"] for row in rows} == {275040}" in source
    assert "FAILED_BEFORE_READINESS" in source
    assert "set +e\n  wait \"$server_pid\"\n  status=$?\n  set -e" in source
    assert "except (OSError, urllib.error.URLError)" in source


def test_future_lifecycle_records_pre_ready_process_failure(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nvidia_smi = binaries / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        "i=0\n"
        "while [ \"$i\" -lt 8 ]; do\n"
        "  printf '%s, NVIDIA B300 SXM6 PC, 275040, 10.3\\n' \"$i\"\n"
        "  i=$((i+1))\n"
        "done\n"
    )
    nvidia_smi.chmod(0o755)
    fake_sleep = binaries / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexit 0\n")
    fake_sleep.chmod(0o755)
    run_dir = tmp_path / "run"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "GLM53_RUN_DIR": str(run_dir),
        "GLM53_REPLICA": "A",
    }
    completed = subprocess.run(
        [
            "evals/fleet/scripts/glm53_dedicated_lifecycle_v2.sh",
            "sh",
            "-c",
            "exit 23",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 23
    preflight_receipt = json.loads(
        (run_dir / "lifecycle/HARDWARE-PREFLIGHT.json").read_text()
    )
    exit_receipt = json.loads((run_dir / "lifecycle/SERVER-EXIT.json").read_text())
    assert preflight_receipt["status"] == "PASSED"
    assert exit_receipt["status"] == "FAILED_BEFORE_READINESS"
    assert exit_receipt["server_exit_code"] == 23
