from pathlib import Path

from evals.fleet import qwen38_dedicated_acceptance_reconcile_v1 as reconcile


def test_reconciler_is_exactly_read_only_against_fleet() -> None:
    source = Path(reconcile.__file__).read_text()
    assert "canary._classify" in source
    assert '"fleet_api_mutations": 0' in source
    assert '"ACCEPTED.json"' in source and '"TERMINAL.json"' in source
    assert "legacy_list_fields_may_be_null_but_never_mismatched_v1" in source
    assert "self_hosted._request" not in source
    assert "POST" not in source and "DELETE" not in source and "PATCH" not in source
