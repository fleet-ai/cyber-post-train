from __future__ import annotations

import pytest

from evals.external_ctf.external_proxy import null_seed_overrides, proxy_namespace
from evals.fleet.fixed_proxy import completion_overrides


def test_null_seed_reuses_fixed_policy_without_inventing_a_seed() -> None:
    policy = {
        "model": "served-model",
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": None,
        "max_tokens": 32768,
    }

    assert null_seed_overrides(completion_overrides, policy) == policy
    assert policy["seed"] is None


@pytest.mark.parametrize("seed", [0, 1, "0", False])
def test_null_seed_rejects_any_provider_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="null seed"):
        null_seed_overrides(
            completion_overrides,
            {
                "model": "served-model",
                "temperature": 1.0,
                "top_p": 0.95,
                "seed": seed,
                "max_tokens": 32768,
            },
        )


def test_null_seed_still_enforces_the_existing_fixed_policy() -> None:
    with pytest.raises(ValueError, match="top_p"):
        null_seed_overrides(
            completion_overrides,
            {
                "model": "served-model",
                "temperature": 1.0,
                "top_p": 0,
                "seed": None,
                "max_tokens": 32768,
            },
        )


def test_proxy_handler_actually_uses_the_null_seed_validator() -> None:
    namespace = proxy_namespace()
    validate = namespace["Handler"]._forward.__globals__["completion_overrides"]  # noqa: SLF001

    assert (
        validate(
            {
                "model": "served-model",
                "temperature": 1.0,
                "top_p": 0.95,
                "seed": None,
                "max_tokens": 32768,
            }
        )["seed"]
        is None
    )
