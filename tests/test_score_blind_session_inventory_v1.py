from __future__ import annotations

import json

import pytest

from evals.fleet import score_blind_session_inventory_v1 as inventory


def _chunks(value: bytes, width: int = 7) -> list[bytes]:
    return [value[index : index + width] for index in range(0, len(value), width)]


def test_retains_only_identity_and_pagination_fields() -> None:
    marker = "sk_live_secret_score_0.9375"
    raw = json.dumps(
        {
            "sessions": [
                {
                    "session_id": "session-1",
                    "eval_task_id": "task-1",
                    "task_key": "task-key",
                    "model": "model-1",
                    "status": "completed",
                    "created_at": "ignored",
                    "verifier_execution": {
                        "score": 0.9375,
                        "success": True,
                        marker: [marker, {"nested": marker}],
                    },
                }
            ],
            "limit": 500,
            "offset": 0,
            "has_more": False,
            "ignored": {"private": marker},
        },
        separators=(",", ":"),
    ).encode()
    page = inventory.parse_session_page(_chunks(raw))
    assert page.sessions == (
        {
            "session_id": "session-1",
            "eval_task_id": "task-1",
            "task_key": "task-key",
            "model": "model-1",
            "status": "completed",
        },
    )
    assert (page.limit, page.offset, page.has_more) == (500, 0, False)
    assert marker not in repr(page)


def test_skipped_secret_never_appears_in_fixed_error() -> None:
    marker = "sk_live_secret_score_0.9375"
    raw = (
        '{"sessions":[{"session_id":"s","eval_task_id":"t",'
        '"task_key":"k","model":"m","status":"completed",'
        f'"verifier_execution":{{"score":"{marker}"}}}}],'
        '"limit":500,"offset":0,"has_more":false} trailing'
    ).encode()
    with pytest.raises(inventory.InventoryError) as caught:
        inventory.parse_session_page(_chunks(raw, 3))
    assert str(caught.value) == "session_inventory_invalid"
    assert marker not in str(caught.value)
    assert marker not in repr(caught.value)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"sessions":[],"limit":500,"offset":0,"has_more":null}',
        b'{"sessions":{},"limit":500,"offset":0,"has_more":false}',
        b'{"sessions":[],"limit":500.0,"offset":0,"has_more":false}',
        b'{"sessions":[],"limit":500,"offset":0,"has_more":false}x',
        b'\xff',
    ],
)
def test_rejects_invalid_envelope_with_fixed_code(raw: bytes) -> None:
    with pytest.raises(inventory.InventoryError, match="^session_inventory_invalid$"):
        inventory.parse_session_page(_chunks(raw))


def test_rejects_byte_ceiling_without_echoing_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inventory, "MAX_BYTES", 8)
    with pytest.raises(inventory.InventoryError, match="^session_inventory_too_large$"):
        inventory.parse_session_page([b'{"secret":', b'"marker"}'])


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sessions":[],"sessions":[],"limit":500,"offset":0,"has_more":false}',
        b'{"sessions":[{"session_id":"a","session_id":"b"}],"limit":500,"offset":0,"has_more":false}',
    ],
)
def test_duplicate_retained_identity_keys_fail_closed(raw: bytes) -> None:
    with pytest.raises(inventory.InventoryError, match="^session_inventory_invalid$"):
        inventory.parse_session_page(_chunks(raw))
