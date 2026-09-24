"""Visible-assistant SFT segmentation, consolidated from the qualified dense run.

Task splits and teacher/student selection are caller inputs, not hardcoded here.
Complete immediate tool rounds are retained; copied context has zero loss.
No source text or tokens should be printed by callers.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import digest_json, file_sha256

FORMAT = "pretokenized_assistant_segments_v1"
MAX_TOKENS = 16384
CONTEXT_BUDGET = 4096
NATIVE_HELPER_SHA = "sha256:55c15b660067749febda00d4fb1c2110ff436717bbd4b73bf66055a73d0b87d5"
HELPER_NAMES = (
    "get_generation_prompt_ids",
    "encode_messages_subset",
    "_find_generation_prompt_boundary",
    "get_response_ids_and_loss_mask_from_messages",
)


def _normalized_for_template(message: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the qualified Qwen-facing normalization for exact length checks."""
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    normalized = {"role": str(message.get("role") or ""), "content": content or ""}
    if message.get("tool_calls"):
        calls = []
        for raw_call in message["tool_calls"]:
            call = dict(raw_call)
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                with contextlib.suppress(ValueError):
                    arguments = json.loads(arguments)
            function["arguments"] = arguments
            call["function"] = function
            calls.append(call)
        normalized["tool_calls"] = calls
    return normalized


def _chat_token_count(tokenizer, messages: list[dict[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        [_normalized_for_template(message) for message in messages],
        tokenize=True,
        add_generation_prompt=False,
    )
    # Transformers 5 returns BatchEncoding; older versions return token IDs.
    return len(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)


class Excluded(ValueError):
    """Only fixed, non-payload reason codes may leave the private builder."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def native_helper(path: Path):
    """Execute only four functions from an independently locked SkyRL source."""
    if file_sha256(path) != NATIVE_HELPER_SHA:
        raise ValueError("native helper digest mismatch")
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPER_NAMES]
    if {n.name for n in nodes} != set(HELPER_NAMES):
        raise ValueError("native helper functions missing")
    source = "from __future__ import annotations\n" + "\n\n".join(ast.unparse(n) for n in nodes)
    scope: dict[str, Any] = {}
    exec(compile(source, "<digest-bound-native-mask-helpers>", "exec"), scope)
    return scope["get_response_ids_and_loss_mask_from_messages"]


def compatible_messages(record: dict) -> tuple[list[dict], dict[str, int]]:
    """Keep the original visible bash/report interface; no tool-surface expansion.

    The old normalization is reused verbatim. use_tool discovery/wrappers are
    NOT rewritten: their full trajectory is held out because replacing the call
    without changing its schema/context has no proven deployment equivalence.
    """
    raw = record.get("messages") or []
    names = {
        (call.get("function") or {}).get("name") for m in raw for call in m.get("tool_calls") or []
    }
    if "context_compaction" in names:
        raise Excluded("opaque_compaction")
    if names - {"bash", "submit_report"}:
        raise Excluded("unsupported_or_unproven_tool_interface")
    if not raw or [m.get("role") for m in raw[:2]] != ["system", "user"]:
        raise Excluded("missing_original_system_and_task_anchor")
    if any(m.get("role") not in {"assistant", "tool"} for m in raw[2:]):
        raise Excluded("unproven_additional_user_or_system_transition")
    pending: set[str] = set()
    seen: set[str] = set()
    visible = []
    counts: collections.Counter[str] = collections.Counter()
    for message in raw:
        if message.get("role") == "assistant" and pending:
            raise Excluded("missing_tool_result_before_next_assistant")
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise Excluded("missing_or_duplicate_tool_call_id")
            seen.add(call_id)
            pending.add(call_id)
            fn = call.get("function") or {}
            args = fn.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except (ValueError, TypeError):
                raise Excluded("invalid_tool_arguments") from None
            if not isinstance(args, dict):
                raise Excluded("invalid_tool_arguments")
            if fn["name"] == "bash":
                if (
                    set(args) - {"script", "timeoutMs"}
                    or not isinstance(args.get("script"), str)
                    or (
                        "timeoutMs" in args
                        and (type(args["timeoutMs"]) is not int or args["timeoutMs"] <= 0)
                    )
                ):
                    raise Excluded("unproven_bash_argument_schema")
            elif (
                set(args) - {"flag", "flags", "explanation"}
                or not isinstance(args.get("explanation"), str)
                or not (
                    (isinstance(args.get("flag"), str) and "flags" not in args)
                    or (
                        isinstance(args.get("flags"), list)
                        and "flag" not in args
                        and all(isinstance(x, str) for x in args["flags"])
                    )
                )
            ):
                raise Excluded("unproven_report_argument_schema")
            counts[fn["name"]] += 1
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise Excluded("orphan_or_duplicate_tool_result")
            pending.remove(call_id)
        visible.append(_normalized_for_template(message))
    if pending:
        raise Excluded("missing_terminal_tool_result")
    if not counts["submit_report"]:
        raise Excluded("missing_report_submission")
    if not any(m["role"] == "assistant" for m in visible):
        raise Excluded("no_assistant_targets")
    return visible, dict(counts)


def encode_record(messages: list[dict], tokenizer: Any, helper) -> tuple[list[int], list[dict]]:
    """Use the pinned ALL_ASSISTANT_MESSAGES policy, retaining complete turns."""
    try:
        anchor = tokenizer.apply_chat_template(
            messages[:2], tokenize=True, add_generation_prompt=False, return_dict=False, tools=[]
        )
        chunks = []
        ordinal = 0
        for index, message in enumerate(messages[2:], 2):
            ids, mask, _ = helper([message], tokenizer, tokenizer_kwargs={"tools": []})
            if len(ids) != len(mask):
                raise ValueError("native lengths")
            assistant = message["role"] == "assistant"
            if assistant:
                positions = [i for i, v in enumerate(mask) if v]
                if not positions or positions != list(range(positions[0], positions[-1] + 1)):
                    raise ValueError("native assistant mask not contiguous")
                target = (positions[0], positions[-1] + 1)
            else:
                if any(mask):
                    raise ValueError("native tool masked incorrectly")
                target = None
            chunks.append(
                {
                    "ids": ids,
                    "mask": mask,
                    "message_index": index,
                    "assistant_index": ordinal if assistant else None,
                    "target": target,
                }
            )
            ordinal += assistant
        return anchor, chunks
    except Exception:
        # Native assertion messages may contain token IDs. Never forward them.
        raise Excluded("native_template_or_mask_contract") from None


def segment_record(
    record: dict,
    anchor: list[int],
    chunks: list[dict],
    *,
    max_tokens: int = MAX_TOKENS,
    context_budget: int = CONTEXT_BUDGET,
) -> list[dict]:
    """Train every fitting original response once, not every trajectory or window.

    A response is eligible iff task anchor + its immediately prior complete
    assistant/tool round + response fit. Excluded assistants can still appear as
    zero-loss context for later fitting responses. No token truncation occurs.
    """
    if max_tokens <= 1 or context_budget < 0:
        raise ValueError("invalid segment bounds")
    positions = [i for i, c in enumerate(chunks) if c["assistant_index"] is not None]
    if not positions or positions[0] != 0:
        raise Excluded("missing_first_assistant")
    if len(anchor) >= max_tokens:
        raise Excluded("overlength_original_task_anchor")
    count = len(positions)
    prefix = [0]
    for chunk in chunks:
        prefix.append(prefix[-1] + len(chunk["ids"]))
    eligible, excluded = [], []
    for ordinal, pos in enumerate(positions):
        previous = positions[max(0, ordinal - 1)]
        if len(anchor) + prefix[pos + 1] - prefix[previous] <= max_tokens:
            eligible.append(ordinal)
        else:
            reason = (
                "overlength_assistant_target"
                if len(anchor) + len(chunks[pos]["ids"]) > max_tokens
                else "overlength_required_previous_round"
            )
            excluded.append(
                {
                    "assistant_index": ordinal,
                    "source_message_index": chunks[pos]["message_index"],
                    "reason": reason,
                }
            )
    if not eligible:
        raise Excluded("no_context_fitting_assistant_targets")
    eligible_set = set(eligible)
    first, rows = eligible[0], []
    while first < count:
        if first not in eligible_set:
            first += 1
            continue
        context_first = max(0, first - 1)
        first_pos = positions[first]
        while context_first > 0:
            trial = positions[context_first - 1]
            if (
                prefix[first_pos] - prefix[trial] > context_budget
                or len(anchor) + prefix[first_pos + 1] - prefix[trial] > max_tokens
            ):
                break
            context_first -= 1
        start_pos = positions[context_first]
        last = first
        while (
            last + 1 in eligible_set
            and len(anchor) + prefix[positions[last + 1] + 1] - prefix[start_pos] <= max_tokens
        ):
            last += 1
        ids, mask, spans = list(anchor), [0] * len(anchor), []
        for chunk in chunks[start_pos : positions[last] + 1]:
            offset = len(ids)
            ids.extend(chunk["ids"])
            ordinal = chunk["assistant_index"]
            target = ordinal is not None and first <= ordinal <= last
            mask.extend(chunk["mask"] if target else [0] * len(chunk["ids"]))
            if target:
                a, b = chunk["target"]
                spans.append(
                    {
                        "assistant_index": ordinal,
                        "source_message_index": chunk["message_index"],
                        "token_start": offset + a,
                        "token_end": offset + b,
                        "source_target_sha256": digest_json(chunk["ids"][a:b]),
                    }
                )
        sid = record["record_id"]
        wid = f"{sid}::dense::{first}-{last}"
        rows.append(
            {
                "input_ids": ids,
                "loss_mask": mask,
                "token_count": len(ids),
                "target_token_count": sum(mask),
                "task_key": record["lineage"]["task_key"],
                "window_id": wid,
                "segment_id": wid,
                "source_session_id": sid,
                "source_model": record["source"]["model"],
                "source_assistant_count": count,
                "eligible_assistant_indices": eligible,
                "excluded_assistant_targets": excluded,
                "target_spans": spans,
                "copied_context_assistant_indices": list(range(context_first, first)),
                "context_start_message_index": chunks[start_pos]["message_index"],
                "split": "train",
            }
        )
        first = last + 1
    actual = [
        (s["assistant_index"], s["source_target_sha256"]) for r in rows for s in r["target_spans"]
    ]
    expected = {
        i: digest_json(
            chunks[positions[i]]["ids"][
                chunks[positions[i]]["target"][0] : chunks[positions[i]]["target"][1]
            ]
        )
        for i in eligible
    }
    if len(actual) != len(expected) or dict(actual) != expected:
        raise ValueError("source eligible coverage differs")
    return rows


def clean_dev_windows(
    record: dict, tokenizer: Any, *, max_tokens: int = MAX_TOKENS, targets: int = 5
) -> list[dict]:
    """A bounded spread of contiguous last-response targets with complete context.

    Eligibility is measured without computing model losses. Never retain a
    response without the complete immediately prior tool round that produced
    its observation. Optional older context is copied as complete rounds.
    """
    if type(targets) is not int or targets <= 0 or max_tokens <= 1:
        raise ValueError("invalid dev window bounds")
    messages, _ = compatible_messages(record)
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    fitting = []
    for ordinal, target in enumerate(positions):
        start = positions[max(0, ordinal - 1)]
        current = messages[:2] + messages[start : target + 1]
        n = _chat_token_count(tokenizer, current)
        if n <= max_tokens:
            fitting.append((ordinal, start, target, n))
    if not fitting:
        raise Excluded("no_context_fitting_dev_target")
    # Deterministic farthest-point coverage, without outcomes/loss-based ranking.
    chosen = {len(fitting) - 1} if targets == 1 else {0, len(fitting) - 1}
    while len(chosen) < min(targets, len(fitting)):
        chosen.add(
            max(
                (i for i in range(len(fitting)) if i not in chosen),
                key=lambda i: (min(abs(i - j) for j in chosen), -i),
            )
        )
    windows = []
    for idx in sorted(chosen):
        ordinal, start, target, _ = fitting[idx]
        # Earlier full rounds are optional. Binary search the earliest fitting prefix.
        boundaries = [p for p in positions if p <= start]
        low, high, best = 0, len(boundaries) - 1, start
        while low <= high:
            middle = (low + high) // 2
            n = _chat_token_count(
                tokenizer, messages[:2] + messages[boundaries[middle] : target + 1]
            )
            if n <= max_tokens:
                best = boundaries[middle]
                high = middle - 1
            else:
                low = middle + 1
        current = messages[:2] + messages[best : target + 1]
        n = _chat_token_count(tokenizer, current)
        for message in current:
            for call in message.get("tool_calls") or []:
                args = call["function"]["arguments"]
                if not isinstance(args, str):
                    call["function"]["arguments"] = json.dumps(args, sort_keys=True)
        # Canonical nested structs make byte-content comparisons independent of
        # Arrow's null fill for missing fields on system/user/tool messages.
        current = [
            {
                "role": m["role"],
                "content": m["content"],
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "id": c.get("id"),
                        "type": c.get("type"),
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": c["function"]["arguments"],
                        },
                    }
                    for c in m.get("tool_calls") or []
                ]
                or None,
            }
            for m in current
        ]
        windows.append(
            {
                "messages": current,
                "tools": "[]",
                "task_key": record["lineage"]["task_key"],
                "window_id": f"{record['record_id']}::dev::{target}",
                "source_session_id": record["record_id"],
                "source_model": record["source"]["model"],
                "token_count": n,
                "source_assistant_index": ordinal,
                "source_message_index": target,
                "context_start_message_index": best,
                "split": "dev",
            }
        )
    return windows
