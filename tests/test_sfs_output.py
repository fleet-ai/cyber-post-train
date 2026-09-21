from copy import deepcopy

import pytest

from cyber_post_train.jobs import digest
from cyber_post_train.sfs_output import (
    OUTPUT_ABSENCE_MAX_AGE_SECONDS,
    build_output_absence_receipt,
    prove_output_absent,
    validate_output_absence_receipt,
)


def plan():
    return {"schema": "cyber_sft_runtime_dense_v1", "source": {"commit": "a" * 40}}


def request():
    return {
        "name": "researcher-sft",
        "run_dir": "/mnt/sfs/jobs/researcher-sft-v1",
    }


def test_live_sfs_check_builds_exact_source_bound_receipt(tmp_path):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    receipt = build_output_absence_receipt(plan(), request(), jobs_root=jobs_root, now=1000)
    assert receipt == validate_output_absence_receipt(receipt, plan(), request(), now=1001)
    assert receipt["output_absent"] is True
    assert receipt["checked_at_epoch"] == 1000


def test_live_sfs_check_rejects_existing_output(tmp_path):
    jobs_root = tmp_path / "jobs"
    (jobs_root / "researcher-sft-v1").mkdir(parents=True)
    with pytest.raises(ValueError, match="output already exists"):
        build_output_absence_receipt(plan(), request(), jobs_root=jobs_root, now=1000)


def test_live_sfs_check_rejects_symlink_output(tmp_path):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    (jobs_root / "researcher-sft-v1").symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="output already exists"):
        build_output_absence_receipt(plan(), request(), jobs_root=jobs_root, now=1000)


@pytest.mark.parametrize(
    "fault",
    [
        "plan",
        "request",
        "receipt-name",
        "receipt-path",
        "digest",
        "extra",
        "stale",
        "future",
    ],
)
def test_remote_receipt_fails_closed_on_binding_digest_or_freshness_drift(tmp_path, fault):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    receipt = build_output_absence_receipt(plan(), request(), jobs_root=jobs_root, now=1000)
    plan_value = plan()
    request_value = request()
    now = 1001
    if fault == "plan":
        plan_value["source"]["commit"] = "b" * 40
    elif fault == "request":
        request_value["run_dir"] += "-other"
    elif fault == "receipt-name":
        receipt["run_name"] = "other"
        receipt["sha256"] = digest(
            {key: value for key, value in receipt.items() if key != "sha256"}
        )
    elif fault == "receipt-path":
        receipt["run_dir"] += "-other"
        receipt["sha256"] = digest(
            {key: value for key, value in receipt.items() if key != "sha256"}
        )
    elif fault == "digest":
        receipt["sha256"] = "0" * 64
    elif fault == "extra":
        receipt["unreviewed"] = True
    elif fault == "stale":
        now = 1000 + OUTPUT_ABSENCE_MAX_AGE_SECONDS + 1
    else:
        now = 999
    with pytest.raises(ValueError):
        validate_output_absence_receipt(receipt, plan_value, request_value, now=now)


def test_unmounted_submitter_requires_fresh_receipt_and_revalidates_age(tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    receipt = build_output_absence_receipt(plan(), request(), jobs_root=mounted, now=1000)
    unavailable = tmp_path / "unavailable"
    with pytest.raises(ValueError, match="provide a fresh"):
        prove_output_absent(plan(), request(), jobs_root=unavailable, now=1001)
    assert prove_output_absent(
        plan(), request(), jobs_root=unavailable, receipt=receipt, now=1001
    ) == deepcopy(receipt)
    with pytest.raises(ValueError, match="stale"):
        prove_output_absent(
            plan(),
            request(),
            jobs_root=unavailable,
            receipt=receipt,
            now=1000 + OUTPUT_ABSENCE_MAX_AGE_SECONDS + 1,
        )
