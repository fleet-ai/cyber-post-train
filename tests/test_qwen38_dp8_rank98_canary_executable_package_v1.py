from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_rank98_canary_controller_v1 as controller
from evals.fleet import qwen38_dp8_rank98_canary_executable_package_v1 as executable
from evals.fleet import qwen38_dp8_rank98_canary_package_v1 as held_package
from evals.fleet import self_hosted
from tests.test_qwen38_dp8_rank98_canary_controller_v1 import _root_release
from tests.test_qwen38_dp8_rank98_canary_package_v1 import _root_with_parity


def _released(tmp_path: Path, monkeypatch):
    root, binding, qualification, held_release = _root_with_parity(tmp_path, monkeypatch)
    held = held_package.render_bundle(root, binding, qualification, held_release)
    release = _root_release(
        binding,
        qualification,
        held_release["held_partition"],
        held["package_sha256"],
    )
    return root, binding, qualification, held_release, release


def test_executable_renderer_requires_root_release_and_exact_runtime_closure(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, held_release, release = _released(tmp_path, monkeypatch)
    value = executable.render_bundle(
        root, binding, qualification, held_release, release
    )
    configmap, job = value["objects"]
    assert configmap["immutable"] is True
    assert set(
        ("controller.py", "exact_pass4_crypto.py", "exact_pass4_universe.py", "root-release.json")
    ).issubset(configmap["data"])
    assert job["spec"]["suspend"] is False
    assert job["metadata"]["annotations"][
        "cyber-post-train.fleet.ai/root-release-sha256"
    ] == release["receipt_sha256"]
    assert value["submit_calls"] == 0
    assert value["kubernetes_mutation_calls"] == 0
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_executable_renderer_fails_closed_without_exact_release(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, held_release, release = _released(tmp_path, monkeypatch)
    stale = copy.deepcopy(release)
    stale["all_four_claims_absent"] = False
    stale["receipt_sha256"] = self_hosted.digest_without(stale, "receipt_sha256")
    with pytest.raises(controller.ControllerError, match="root release"):
        executable.render_bundle(root, binding, qualification, held_release, stale)


def test_executable_renderer_has_no_submit_or_apply_verb() -> None:
    assert not hasattr(executable, "submit")
    assert not hasattr(executable, "create")
    assert not hasattr(executable, "apply")
