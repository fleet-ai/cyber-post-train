import copy

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v32_live_authorization_v1 as live
from evals.fleet import glm53_dedicated_v32_stale_run_reconciliation_v1 as stale


def _row(
    name: str = "ft-run-deadbeef",
    *,
    status: object = "submitted",
    run_dir: str = "/mnt/sfs/jobs/chris-cyber-evalserve-old-v1",
) -> dict[str, object]:
    return {
        "name": name,
        "title": "chris-cyber-evalserve-old-v1",
        "run_dir": run_dir,
        "status": status,
    }


class FakeBackend:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = [_row()]
        self.pages = 2
        self.exact: dict[str, dict[str, object]] = {}
        self.items: list[dict[str, object]] = []
        self.root_exists = True
        self.unsafe_symlink = False
        self.receipts: list[dict[str, object]] = [
            {
                "relative_path": "release/RELEASE.json",
                "size": 123,
                "sha256": "sha256:" + "1" * 64,
            }
        ]

    def list_runs(self) -> tuple[list[dict[str, object]], int]:
        return copy.deepcopy(self.rows), self.pages

    def get_run(self, api_run_id: str) -> dict[str, object] | None:
        return copy.deepcopy(self.exact.get(api_run_id))

    def kubernetes_inventory(self) -> dict[str, object]:
        return {"items": copy.deepcopy(self.items)}

    def sfs_evidence(self, run_dirs: tuple[str, ...]) -> dict[str, object]:
        return {
            "observer_pod_name": live.SFS_OBSERVER_POD_NAME,
            "observer_pod_uid": live.SFS_OBSERVER_POD_UID,
            "observer_sfs_mount_path": live.SFS_OBSERVER_MOUNT_PATH,
            "roots": {
                run_dir: {
                    "root_exists": self.root_exists,
                    "unsafe_symlink": self.unsafe_symlink,
                    "terminal_evidence": copy.deepcopy(self.receipts),
                }
                for run_dir in run_dirs
            },
        }


def _build(backend: FakeBackend | None = None) -> dict[str, object]:
    selected = backend or FakeBackend()
    rows, pages = selected.list_runs()
    return stale.build_reconciliation(
        backend=selected,
        now=1_000.0,
        rows_snapshot=(rows, pages),
    )


def test_reconciles_every_submitted_exact_404_row_from_one_snapshot() -> None:
    backend = FakeBackend()
    backend.rows.append(
        _row(
            "ft-run-00000001",
            run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-other-v1",
        )
    )
    value = _build(backend)
    stale.validate_reconciliation(value, rows=backend.rows, now=1_001.0)

    assert value["jobs_api_pages"] == 2
    assert [row["api_run_id"] for row in value["reconciled_rows"]] == [
        "ft-run-00000001",
        "ft-run-deadbeef",
    ]
    assert all(
        row["terminal_basis"]
        == "EXACT_GET_404_AND_ZERO_KUBERNETES_REFERENCES"
        for row in value["reconciled_rows"]
    )
    assert value["api_mutation_calls"] == value["scoring_calls"] == 0
    assert value["protected_content_included"] is False


@pytest.mark.parametrize("status", ["PENDING", "CREATED", "MYSTERY", None, 7])
def test_unknown_or_malformed_status_blocks(status: object) -> None:
    backend = FakeBackend()
    backend.rows[0]["status"] = status
    with pytest.raises(stale.ReconciliationError, match="status_invalid"):
        _build(backend)


def test_active_exact_get_blocks_and_terminal_exact_get_is_bound() -> None:
    backend = FakeBackend()
    backend.exact["ft-run-deadbeef"] = _row(status="RUNNING")
    with pytest.raises(stale.ReconciliationError, match="current_server_present"):
        _build(backend)

    backend.exact["ft-run-deadbeef"] = _row(status="FAILED")
    value = _build(backend)
    assert value["reconciled_rows"][0]["exact_get_http_status"] == 200
    assert value["reconciled_rows"][0]["exact_get_status"] == "FAILED"


def test_terminal_list_row_with_exact_active_get_blocks() -> None:
    backend = FakeBackend()
    backend.rows[0]["status"] = "FAILED"
    backend.exact["ft-run-deadbeef"] = _row(status="RUNNING")
    with pytest.raises(stale.ReconciliationError, match="current_server_present"):
        _build(backend)


@pytest.mark.parametrize("kind", ["RayJob", "RayCluster", "Workload", "Service"])
def test_any_project_kubernetes_remnant_blocks(kind: str) -> None:
    backend = FakeBackend()
    backend.items = [
        {
            "kind": kind,
            "metadata": {
                "name": "orphan",
                "uid": "11111111-1111-4111-8111-111111111111",
                "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
            },
        }
    ]
    with pytest.raises(stale.ReconciliationError, match="kubernetes_or_gpu_remnant"):
        _build(backend)


def test_any_project_gpu_pod_blocks() -> None:
    backend = FakeBackend()
    backend.items = [
        {
            "kind": "Pod",
            "metadata": {
                "name": "gpu",
                "uid": "11111111-1111-4111-8111-111111111111",
                "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
            },
            "status": {"phase": "Running"},
            "spec": {
                "containers": [
                    {"resources": {"requests": {"nvidia.com/gpu": "8"}}}
                ]
            },
        }
    ]
    with pytest.raises(stale.ReconciliationError, match="kubernetes_or_gpu_remnant"):
        _build(backend)


def test_terminal_rayjob_and_finished_workload_are_bound_not_ignored() -> None:
    backend = FakeBackend()
    backend.items = [
        {
            "kind": "RayJob",
            "metadata": {
                "name": "ft-run-deadbeef",
                "uid": "11111111-1111-4111-8111-111111111111",
                "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
            },
            "status": {"jobStatus": "FAILED"},
        },
        {
            "kind": "Workload",
            "metadata": {
                "name": "rayjob-ft-run-deadbeef-12345",
                "uid": "22222222-2222-4222-8222-222222222222",
                "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
                "ownerReferences": [
                    {
                        "kind": "RayJob",
                        "name": "ft-run-deadbeef",
                        "uid": "11111111-1111-4111-8111-111111111111",
                    }
                ],
            },
            "status": {
                "conditions": [
                    {"type": "Finished", "status": "True", "reason": "Failed"}
                ]
            },
        },
    ]
    value = _build(backend)
    stale.validate_reconciliation(value, rows=backend.rows, now=1_001.0)
    row = value["reconciled_rows"][0]
    assert row["terminal_basis"] == (
        "EXACT_GET_404_AND_TERMINAL_KUBERNETES_UID_CHAIN"
    )
    assert row["kubernetes_reference_count"] == 2
    assert value["terminal_project_object_count"] == 2


@pytest.mark.parametrize("target", ["global", "per_row"])
@pytest.mark.parametrize(
    "owner_mutation",
    [
        {"kind": "Pod", "name": "unsafe", "uid": "3" * 36},
        {"kind": "RayJob", "name": "../unsafe", "uid": "3" * 36},
        {"kind": "RayJob", "name": "safe", "uid": "not-a-uuid"},
        {
            "kind": "RayJob",
            "name": "safe",
            "uid": "33333333-3333-4333-8333-333333333333",
            "protected": "forbidden",
        },
    ],
)
def test_terminal_owner_identity_mutations_reject(
    target: str, owner_mutation: dict[str, object]
) -> None:
    backend = FakeBackend()
    backend.items = [
        {
            "kind": "Workload",
            "metadata": {
                "name": "rayjob-ft-run-deadbeef-12345",
                "uid": "22222222-2222-4222-8222-222222222222",
                "labels": {"cyber-post-train.fleet.ai/owner": "chris"},
                "ownerReferences": [
                    {
                        "kind": "RayJob",
                        "name": "ft-run-deadbeef",
                        "uid": "11111111-1111-4111-8111-111111111111",
                    }
                ],
            },
            "status": {
                "conditions": [
                    {"type": "Finished", "status": "True", "reason": "Failed"}
                ]
            },
        }
    ]
    value = _build(backend)
    if target == "global":
        value["terminal_project_objects"][0]["owners"][0] = owner_mutation
        value["terminal_project_object_snapshot_sha256"] = crypto.sha256(
            crypto.canonical_json(value["terminal_project_objects"])
        )
    else:
        value["reconciled_rows"][0]["terminal_kubernetes_evidence"][0]["owners"][
            0
        ] = owner_mutation
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    with pytest.raises(stale.ReconciliationError, match="reconciliation_invalid"):
        stale.validate_reconciliation(value, rows=backend.rows, now=1_001.0)


def test_sfs_symlink_or_malformed_evidence_blocks() -> None:
    backend = FakeBackend()
    backend.unsafe_symlink = True
    with pytest.raises(stale.ReconciliationError, match="sfs_evidence_invalid"):
        _build(backend)

    backend.unsafe_symlink = False
    backend.receipts[0]["relative_path"] = "../RELEASE.json"
    value = _build(backend)
    with pytest.raises(stale.ReconciliationError, match="reconciliation_invalid"):
        stale.validate_reconciliation(value, rows=backend.rows, now=1_001.0)


@pytest.mark.parametrize("mutation", ["omit", "extra", "identity", "observer"])
def test_rehashed_posthoc_mutation_rejects(mutation: str) -> None:
    backend = FakeBackend()
    value = _build(backend)
    if mutation == "omit":
        value["reconciled_rows"] = []
    elif mutation == "extra":
        value["reconciled_rows"].append(copy.deepcopy(value["reconciled_rows"][0]))
    elif mutation == "identity":
        value["reconciled_rows"][0]["api_run_id"] = "ft-run-forged"
    elif mutation == "observer":
        value["sfs_observation"]["observer_pod_name"] = "arbitrary-observer"
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    with pytest.raises(stale.ReconciliationError, match="reconciliation_invalid"):
        stale.validate_reconciliation(value, rows=backend.rows, now=1_001.0)


def test_reconciliation_expires_and_held_receipt_never_authorizes_launch() -> None:
    backend = FakeBackend()
    value = _build(backend)
    with pytest.raises(stale.ReconciliationError, match="reconciliation_invalid"):
        stale.validate_reconciliation(
            value,
            rows=backend.rows,
            now=1_000.0 + stale.MAX_AGE_SECONDS + 1,
        )
    held = stale.build_held("a" * 40)
    assert held["server_launch_authorized"] is False
    assert held["api_mutation_calls"] == held["scoring_calls"] == 0
    assert held["receipt_sha256"] == crypto.digest_without(
        held, "receipt_sha256"
    )
