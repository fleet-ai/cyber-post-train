"""Bounded streaming parser for Fleet session-list identity metadata.

The Fleet session-list response also carries verifier summaries.  Collision
gates must therefore consume the response as a token stream: retain only the
small identity allowlist and structurally skip every other value without
materializing, hashing, logging, or branching on it.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

MAX_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 32
MAX_ITEMS = 4096
SESSION_FIELDS = {"session_id", "eval_task_id", "task_key", "model", "status"}
PAGE_FIELDS = {"sessions", "limit", "offset", "has_more"}


class InventoryError(RuntimeError):
    """Fixed-text parser failure safe for receipts and logs."""


@dataclass(frozen=True)
class SessionPage:
    sessions: tuple[dict[str, Any], ...]
    limit: int
    offset: int
    has_more: bool


def _characters(chunks: Iterable[bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    total = 0
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise InventoryError("session_inventory_invalid")
            total += len(chunk)
            if total > MAX_BYTES:
                raise InventoryError("session_inventory_too_large")
            yield from decoder.decode(chunk, final=False)
        yield from decoder.decode(b"", final=True)
    except UnicodeError:
        raise InventoryError("session_inventory_invalid") from None


class _Cursor:
    def __init__(self, chars: Iterable[str]) -> None:
        self._chars = iter(chars)
        self._next: str | None = None

    def peek(self) -> str:
        if self._next is None:
            try:
                self._next = next(self._chars)
            except StopIteration:
                return ""
        return self._next

    def take(self) -> str:
        value = self.peek()
        self._next = None
        return value

    def ws(self) -> None:
        while self.peek() and self.peek() in " \t\r\n":
            self.take()

    def expect(self, expected: str) -> None:
        self.ws()
        if self.take() != expected:
            raise InventoryError("session_inventory_invalid")

    def string(self, *, retain: bool) -> str | None:
        self.ws()
        if self.take() != '"':
            raise InventoryError("session_inventory_invalid")
        value: list[str] | None = [] if retain else None
        while True:
            char = self.take()
            if not char:
                raise InventoryError("session_inventory_invalid")
            if char == '"':
                return "".join(value) if value is not None else None
            if char == "\\":
                escaped = self.take()
                if escaped in '"\\/':
                    decoded = escaped
                elif escaped in "bfnrt":
                    decoded = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}[escaped]
                elif escaped == "u":
                    digits = "".join(self.take() for _ in range(4))
                    if len(digits) != 4 or any(c not in "0123456789abcdefABCDEF" for c in digits):
                        raise InventoryError("session_inventory_invalid")
                    decoded = chr(int(digits, 16))
                else:
                    raise InventoryError("session_inventory_invalid")
                if value is not None:
                    value.append(decoded)
            else:
                if ord(char) < 0x20:
                    raise InventoryError("session_inventory_invalid")
                if value is not None:
                    value.append(char)

    def literal(self, word: str, value: Any) -> Any:
        self.ws()
        if "".join(self.take() for _ in word) != word:
            raise InventoryError("session_inventory_invalid")
        return value

    def integer(self) -> int:
        self.ws()
        digits: list[str] = []
        if self.peek() == "-":
            digits.append(self.take())
        while self.peek().isdigit():
            digits.append(self.take())
        if not digits or digits == ["-"] or self.peek() in ".eE":
            raise InventoryError("session_inventory_invalid")
        return int("".join(digits))

    def skip_number(self) -> None:
        """Validate one JSON number without retaining its digits or value."""
        self.ws()
        if self.peek() == "-":
            self.take()
        if self.peek() == "0":
            self.take()
            if self.peek().isdigit():
                raise InventoryError("session_inventory_invalid")
        elif self.peek() in "123456789":
            self.take()
            while self.peek().isdigit():
                self.take()
        else:
            raise InventoryError("session_inventory_invalid")
        if self.peek() == ".":
            self.take()
            if not self.peek().isdigit():
                raise InventoryError("session_inventory_invalid")
            while self.peek().isdigit():
                self.take()
        if self.peek() in "eE":
            self.take()
            if self.peek() in "+-":
                self.take()
            if not self.peek().isdigit():
                raise InventoryError("session_inventory_invalid")
            while self.peek().isdigit():
                self.take()

    def scalar(self) -> Any:
        self.ws()
        char = self.peek()
        if char == '"':
            return self.string(retain=True)
        if char == "t":
            return self.literal("true", True)
        if char == "f":
            return self.literal("false", False)
        if char == "n":
            return self.literal("null", None)
        return self.integer()

    def skip(self, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            raise InventoryError("session_inventory_invalid")
        self.ws()
        char = self.peek()
        if char == '"':
            self.string(retain=False)
            return
        if char == "{":
            self.take()
            self.ws()
            if self.peek() == "}":
                self.take()
                return
            for index in range(MAX_ITEMS + 1):
                if index == MAX_ITEMS:
                    raise InventoryError("session_inventory_invalid")
                self.string(retain=False)
                self.expect(":")
                self.skip(depth + 1)
                self.ws()
                delimiter = self.take()
                if delimiter == "}":
                    return
                if delimiter != ",":
                    raise InventoryError("session_inventory_invalid")
        elif char == "[":
            self.take()
            self.ws()
            if self.peek() == "]":
                self.take()
                return
            for index in range(MAX_ITEMS + 1):
                if index == MAX_ITEMS:
                    raise InventoryError("session_inventory_invalid")
                self.skip(depth + 1)
                self.ws()
                delimiter = self.take()
                if delimiter == "]":
                    return
                if delimiter != ",":
                    raise InventoryError("session_inventory_invalid")
        elif char == "t":
            self.literal("true", True)
        elif char == "f":
            self.literal("false", False)
        elif char == "n":
            self.literal("null", None)
        else:
            self.skip_number()


def _session(cursor: _Cursor) -> dict[str, Any]:
    cursor.expect("{")
    retained: dict[str, Any] = {}
    cursor.ws()
    if cursor.peek() == "}":
        cursor.take()
        return retained
    for index in range(MAX_ITEMS + 1):
        if index == MAX_ITEMS:
            raise InventoryError("session_inventory_invalid")
        key = cursor.string(retain=True)
        cursor.expect(":")
        if key in SESSION_FIELDS:
            if key in retained:
                raise InventoryError("session_inventory_invalid")
            retained[str(key)] = cursor.scalar()
        else:
            cursor.skip(1)
        cursor.ws()
        delimiter = cursor.take()
        if delimiter == "}":
            return retained
        if delimiter != ",":
            raise InventoryError("session_inventory_invalid")
    raise InventoryError("session_inventory_invalid")


def _sessions(cursor: _Cursor) -> tuple[dict[str, Any], ...]:
    cursor.expect("[")
    rows: list[dict[str, Any]] = []
    cursor.ws()
    if cursor.peek() == "]":
        cursor.take()
        return tuple(rows)
    for index in range(MAX_ITEMS + 1):
        if index == MAX_ITEMS:
            raise InventoryError("session_inventory_invalid")
        rows.append(_session(cursor))
        cursor.ws()
        delimiter = cursor.take()
        if delimiter == "]":
            return tuple(rows)
        if delimiter != ",":
            raise InventoryError("session_inventory_invalid")
    raise InventoryError("session_inventory_invalid")


def parse_session_page(chunks: Iterable[bytes]) -> SessionPage:
    """Return only allowlisted identity fields from one bounded JSON page."""
    cursor = _Cursor(_characters(chunks))
    cursor.expect("{")
    page: dict[str, Any] = {}
    cursor.ws()
    if cursor.peek() == "}":
        raise InventoryError("session_inventory_invalid")
    for index in range(MAX_ITEMS + 1):
        if index == MAX_ITEMS:
            raise InventoryError("session_inventory_invalid")
        key = cursor.string(retain=True)
        cursor.expect(":")
        if key == "sessions":
            if key in page:
                raise InventoryError("session_inventory_invalid")
            page[key] = _sessions(cursor)
        elif key in PAGE_FIELDS:
            if key in page:
                raise InventoryError("session_inventory_invalid")
            page[str(key)] = cursor.scalar()
        else:
            cursor.skip(1)
        cursor.ws()
        delimiter = cursor.take()
        if delimiter == "}":
            break
        if delimiter != ",":
            raise InventoryError("session_inventory_invalid")
    cursor.ws()
    if cursor.take():
        raise InventoryError("session_inventory_invalid")
    if set(page) != PAGE_FIELDS:
        raise InventoryError("session_inventory_invalid")
    if (
        not isinstance(page["sessions"], tuple)
        or isinstance(page["limit"], bool)
        or not isinstance(page["limit"], int)
        or isinstance(page["offset"], bool)
        or not isinstance(page["offset"], int)
        or not isinstance(page["has_more"], bool)
    ):
        raise InventoryError("session_inventory_invalid")
    return SessionPage(**page)
