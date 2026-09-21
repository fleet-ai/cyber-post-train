import pytest

from cyber_post_train.jobs import JobsError
from cyber_post_train.kubernetes_identity import (
    exact_name_query,
    identity_queries,
    require_canonical_output_root,
    require_empty_identity_list,
)

RUN_NAME = "researcher-sft"
RUN_ID = "12345678-1234-4234-9234-123456789abc"
OBJECT_NAME = "researcher-sft-12345678"


def test_identity_queries_are_complete_server_side_exact_selectors():
    queries = identity_queries(RUN_NAME, RUN_ID, OBJECT_NAME)

    assert [(query.flag, query.value, query.purpose) for query in queries] == [
        ("--selector", "fleet.ai/run-name=researcher-sft", "run-name/output"),
        ("--selector", f"fleet.ai/run-id={RUN_ID}", "run-id"),
        ("--field-selector", "metadata.name=researcher-sft-12345678", "exact-name"),
    ]
    assert exact_name_query(OBJECT_NAME).kubectl_args() == [
        "--field-selector",
        "metadata.name=researcher-sft-12345678",
    ]


def test_canonical_sft_output_is_bound_to_the_run_name():
    require_canonical_output_root(RUN_NAME, "/mnt/sfs/jobs/researcher-sft")

    with pytest.raises(JobsError, match="canonical run-name path"):
        require_canonical_output_root(RUN_NAME, "/mnt/sfs/jobs/some-other-output")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"kind": "RayJob", "items": []},
        {"kind": "RayJobList", "items": {}},
        {"kind": "RayJobList", "items": [{}]},
    ],
)
def test_identity_query_rejects_malformed_or_partial_absence(payload):
    query = identity_queries(RUN_NAME, RUN_ID, OBJECT_NAME)[0]

    with pytest.raises(JobsError):
        require_empty_identity_list(payload, query)


def test_identity_query_rejects_a_live_owner():
    query = identity_queries(RUN_NAME, RUN_ID, OBJECT_NAME)[0]
    payload = {"kind": "RayJobList", "items": [{"metadata": {"name": "existing"}}]}

    with pytest.raises(JobsError, match="already owns"):
        require_empty_identity_list(payload, query)
