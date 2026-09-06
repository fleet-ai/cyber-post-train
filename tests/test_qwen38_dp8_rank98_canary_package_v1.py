from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_rank98_canary_package_v1 as package
from evals.fleet import qwen38_dp8_rank98_canary_partition_v1 as partition
from evals.fleet import self_hosted
from tests.test_qwen38_dp8_rank98_canary_partition_v1 import (
    _qualification,
    _scan,
    _server,
)

ROOT = Path(__file__).resolve().parents[1]


def _inputs(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict]:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    binding = _server()
    parity = {
        "schema_version": "test-score-free-parity-v1",
        "status": "PASSED_NON_SCORED",
    }
    parity["receipt_sha256"] = self_hosted.digest_without(parity, "receipt_sha256")
    parity_path = (
        tmp_path / "docs/evidence/qwen38-study/test-future-dp8-parity.json"
    )
    parity_path.parent.mkdir(parents=True)
    parity_path.write_bytes(self_hosted.canonical_json(parity) + b"\n")
    binding["parity_path"] = str(parity_path.relative_to(tmp_path))
    binding["parity_receipt_sha256"] = parity["receipt_sha256"]
    binding["parity_file_sha256"] = self_hosted.sha256(parity_path.read_bytes())
    binding["receipt_sha256"] = self_hosted.digest_without(binding, "receipt_sha256")
    scan = _scan(binding)
    held = partition.build_held(_qualification(), binding, scan)
    release = {"live_scan": scan, "held_partition": held}
    return binding, _qualification(), release


def _root_with_parity(tmp_path: Path, monkeypatch) -> tuple[Path, dict, dict, dict]:
    binding, qualification, release = _inputs(tmp_path, monkeypatch)
    parity_source = tmp_path / binding["parity_path"]
    monkeypatch.setattr(
        package,
        "_validate_parity",
        lambda _root, _binding: parity_source.read_text(),
    )
    binding["receipt_sha256"] = self_hosted.digest_without(binding, "receipt_sha256")
    scan = _scan(binding)
    release = {
        "live_scan": scan,
        "held_partition": partition.build_held(qualification, binding, scan),
    }
    return ROOT, binding, qualification, release


def test_renderer_reuses_proven_cpu_dind_sfs_job_and_stays_suspended(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, release = _root_with_parity(tmp_path, monkeypatch)
    value = package.render_bundle(root, binding, qualification, release)
    configmap, job = value["objects"]
    assert configmap["kind"] == "ConfigMap"
    assert configmap["immutable"] is True
    assert job["kind"] == "Job"
    assert job["spec"]["suspend"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    dind = next(row for row in pod["initContainers"] if row["name"] == "dind")
    assert dind["securityContext"]["privileged"] is True
    assert any(row["name"] == "sfs" for row in pod["volumes"])
    assert any(row["name"] == "docker-socket" for row in pod["volumes"])
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["controller_create_permitted"] is False
    assert value["api_mutation_calls"] == 0
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_package_binds_only_a1_and_keeps_a2_to_a4_held(
    tmp_path: Path, monkeypatch
) -> None:
    root, binding, qualification, release = _root_with_parity(tmp_path, monkeypatch)
    value = package.render_bundle(root, binding, qualification, release)
    data = value["objects"][0]["data"]
    contract = json.loads(data["controller-contract.json"])
    assert contract["active_attempt"] == 1
    assert contract["held_attempts"] == [2, 3, 4]
    assert contract["claim_or_model_calls"] == 0
    job = value["objects"][1]
    env = {
        row["name"]: row.get("value")
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["QWEN_DP8_CANARY_ACTIVE_ATTEMPT"] == "1"
    assert env["QWEN_DP8_CANARY_HELD_ATTEMPTS"] == "2,3,4"
    assert env["QWEN_DP8_CANARY_LAUNCH_AUTHORIZED"] == "false"
    assert env["QWEN_DP8_CANARY_SCORING_AUTHORIZED"] == "false"


def test_package_binds_exact_future_authorities(tmp_path: Path, monkeypatch) -> None:
    root, binding, qualification, release = _root_with_parity(tmp_path, monkeypatch)
    value = package.render_bundle(root, binding, qualification, release)
    annotations = value["objects"][0]["metadata"]["annotations"]
    assert annotations["cyber-post-train.fleet.ai/server-binding-sha256"] == binding[
        "receipt_sha256"
    ]
    assert annotations["cyber-post-train.fleet.ai/qualification-result-sha256"] == (
        qualification["receipt_sha256"]
    )
    assert annotations["cyber-post-train.fleet.ai/release-sha256"] == release[
        "held_partition"
    ]["receipt_sha256"]
    assert annotations["cyber-post-train.fleet.ai/package-sha256"] == value[
        "package_sha256"
    ]


def test_renderer_rejects_stale_release_or_server_binding(tmp_path: Path, monkeypatch) -> None:
    root, binding, qualification, release = _root_with_parity(tmp_path, monkeypatch)
    stale = copy.deepcopy(release)
    stale["held_partition"]["canary"]["only_authorized_attempt"] = 2
    with pytest.raises(package.PackageError, match="release drifted"):
        package.render_bundle(root, binding, qualification, stale)

    stale_binding = copy.deepcopy(binding)
    stale_binding["service_uid"] = "55555555-5555-4555-8555-555555555555"
    stale_binding["receipt_sha256"] = self_hosted.digest_without(
        stale_binding, "receipt_sha256"
    )
    with pytest.raises(partition.PartitionError):
        package.render_bundle(root, stale_binding, qualification, release)


def test_renderer_has_no_create_or_submit_verb() -> None:
    assert not hasattr(package, "submit")
    assert not hasattr(package, "create")
    assert not hasattr(package, "apply")
