"""Secret hygiene without corrupting task-local synthetic credentials.

We redact exact control-plane secret values supplied by the operator.  We do
not broadly erase strings such as every bearer token: credentials discovered
inside an isolated challenge are part of the behavior being learned.  Raw
grader output is excluded elsewhere at the schema boundary.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

DEFAULT_SECRET_ENV = (
    "FLEET_API_KEY",
    "SUPABASE_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "NEBIUS_IAM_TOKEN",
)
REDACTED = "<CONTROL_PLANE_SECRET_REDACTED>"


def secret_values(env_names: Iterable[str] = DEFAULT_SECRET_ENV) -> tuple[str, ...]:
    values = {os.environ.get(name, "") for name in env_names}
    return tuple(sorted((value for value in values if len(value) >= 8), key=len, reverse=True))


def redact_exact(value: Any, secrets: Iterable[str]) -> Any:
    secret_tuple = tuple(secret for secret in secrets if secret)
    if isinstance(value, str):
        for secret in secret_tuple:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, list):
        return [redact_exact(item, secret_tuple) for item in value]
    if isinstance(value, dict):
        return {key: redact_exact(item, secret_tuple) for key, item in value.items()}
    return value


def contains_secret(value: Any, secrets: Iterable[str]) -> bool:
    serialized = str(value)
    return any(secret and secret in serialized for secret in secrets)
