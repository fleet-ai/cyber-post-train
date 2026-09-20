import pytest

from evals.fleet import cluster_entry


def test_dedicated_dsn_preserves_credentials_and_options():
    actual = cluster_entry.dedicated_dsn(
        "postgresql://user:password@database.example:5432/admin?sslmode=require",
        "q38_f75_dev17_p1_v1",
    )
    assert actual == (
        "postgresql://user:password@database.example:5432/q38_f75_dev17_p1_v1?sslmode=require"
    )


@pytest.mark.parametrize("name", ["", "Upper", "hyphen-name", "1starts_with_number"])
def test_dedicated_dsn_rejects_unsafe_name(name):
    with pytest.raises(ValueError):
        cluster_entry.dedicated_dsn("postgresql://u:p@host/admin", name)
