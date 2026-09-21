"""Bind an aggregate reasoning-field census to a dense SFT corpus.

This module intentionally accepts only sealed aggregate metadata.  It never
opens a transcript, session export, training parquet, or model prompt.  The
current training path is authorized only for visible assistant actions.  A
future student-visible reasoning objective needs a separately reviewed schema,
renderer, mask, and an update to ``AUTHORIZED_REASONING_SCHEMAS`` in the same
change; unknown schemas fail closed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import digest_json

CENSUS_SCHEMA = "cyber_reasoning_field_census_v1"
SELECTION_SCHEMA = "cyber_reasoning_selection_v1"
AGGREGATE_METHOD = "aggregate_index_only_v1"
VISIBLE_ACTIONS_ONLY = "visible_actions_only"
STUDENT_VISIBLE_REASONING = "student_visible_reasoning"
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")

# Deliberately empty.  The existing visible-action renderer drops these fields,
# and the runtime rejects them before tokenization.  Adding a schema here alone
# is insufficient: a future reviewed change must also implement and test its
# student-visible rendering and loss-mask contract.
AUTHORIZED_REASONING_SCHEMAS: frozenset[str] = frozenset()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected:
        raise ValueError(f"{label} has unknown or missing fields")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in value.removeprefix("sha256:")):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if result.get("schema") != schema or result.get("sha256") != digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {label} digest or schema")
    return result


def validate_census(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed metadata-only field census without opening its source.

    ``available_sessions`` counts availability in the source index; the two
    ``selected_*`` values count material selected for this exact corpus.  This
    lets a visible-action corpus record that a private field existed upstream
    while proving that none of it reached a supervised target.
    """

    census = _sealed(value, CENSUS_SCHEMA, "reasoning census")
    _exact(
        census,
        {
            "schema",
            "source",
            "selected",
            "fields",
            "reasoning_schema",
            "method",
            "sha256",
        },
        "reasoning census",
    )
    if census["method"] != AGGREGATE_METHOD:
        raise ValueError("reasoning census must use the aggregate-only method")
    source = _exact(
        census["source"],
        {
            "normalized_records_sha256",
            "success_evidence_sha256",
            "candidate_sessions",
            "selected_sessions",
        },
        "reasoning census source",
    )
    _sha256(source["normalized_records_sha256"], "normalized records")
    _sha256(source["success_evidence_sha256"], "success evidence")
    candidates = _count(source["candidate_sessions"], "candidate sessions")
    selected_sessions = _count(source["selected_sessions"], "selected sessions")
    if selected_sessions > candidates:
        raise ValueError("selected sessions exceed candidate sessions")

    selected = _exact(
        census["selected"],
        {"supervised_tokens", "assistant_targets"},
        "reasoning census selected totals",
    )
    tokens = _count(selected["supervised_tokens"], "selected supervised tokens")
    targets = _count(selected["assistant_targets"], "selected assistant targets")
    if bool(selected_sessions) != bool(tokens) or bool(selected_sessions) != bool(targets):
        raise ValueError("selected aggregate totals must be jointly empty or positive")

    fields = _exact(census["fields"], set(REASONING_FIELDS), "reasoning census fields")
    for name in REASONING_FIELDS:
        counts = _exact(
            fields[name],
            {"available_sessions", "selected_sessions", "selected_target_tokens"},
            f"reasoning field {name}",
        )
        available = _count(counts["available_sessions"], f"{name} available sessions")
        selected_field_sessions = _count(counts["selected_sessions"], f"{name} selected sessions")
        selected_field_tokens = _count(
            counts["selected_target_tokens"], f"{name} selected target tokens"
        )
        if available > candidates or selected_field_sessions > available:
            raise ValueError("reasoning field session counts are inconsistent")
        if selected_field_sessions > selected_sessions or selected_field_tokens > tokens:
            raise ValueError("reasoning field selected totals are inconsistent")
        if bool(selected_field_sessions) != bool(selected_field_tokens):
            raise ValueError("reasoning field selected totals must agree")

    schema = census["reasoning_schema"]
    if schema is not None and (not isinstance(schema, str) or not schema):
        raise ValueError("reasoning schema must be a nonempty string or null")
    return copy.deepcopy(census)


def _manifest_totals(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _sealed(value, "cyber_dense_sft_corpus_v1", "dense corpus manifest")
    provenance = _mapping(manifest.get("catalog_provenance"), "dense corpus provenance")
    train = _mapping(_mapping(manifest.get("files"), "dense corpus files").get("train"), "train")
    totals = {
        "corpus_manifest_sha256": _sha256(manifest["sha256"], "corpus manifest"),
        "normalized_records_sha256": _sha256(
            provenance.get("normalized_records_file_sha256"), "manifest normalized records"
        ),
        "success_evidence_sha256": _sha256(
            provenance.get("success_evidence_file_sha256"), "manifest success evidence"
        ),
        "source_sessions": _count(train.get("source_sessions"), "manifest source sessions"),
        "supervised_tokens": _count(train.get("supervised_tokens"), "manifest supervised tokens"),
        "assistant_targets": _count(train.get("assistant_responses"), "manifest assistant targets"),
    }
    if not all(
        totals[name] > 0 for name in ("source_sessions", "supervised_tokens", "assistant_targets")
    ):
        raise ValueError("dense corpus aggregate totals must be positive")
    return totals


def _authorize(census: dict[str, Any], objective: str) -> str | None:
    if objective == VISIBLE_ACTIONS_ONLY:
        if census["reasoning_schema"] is not None:
            raise ValueError("visible-action selection must not name a reasoning schema")
        for name in REASONING_FIELDS:
            counts = census["fields"][name]
            if counts["selected_sessions"] or counts["selected_target_tokens"]:
                raise ValueError("visible-action selection contains reasoning targets")
        return None
    if objective != STUDENT_VISIBLE_REASONING:
        raise ValueError("unsupported supervision objective")
    schema = census["reasoning_schema"]
    if schema not in AUTHORIZED_REASONING_SCHEMAS:
        raise ValueError("reasoning schema is not authorized for training")
    raise ValueError("student-visible reasoning renderer and mask are not qualified")


def select(
    census: Mapping[str, Any], corpus_manifest: Mapping[str, Any], *, objective: str
) -> dict[str, Any]:
    """Create one digest-bound selection from aggregate census and manifest data."""

    checked_census = validate_census(census)
    totals = _manifest_totals(corpus_manifest)
    source = checked_census["source"]
    selected = checked_census["selected"]
    if (
        source["normalized_records_sha256"] != totals["normalized_records_sha256"]
        or source["success_evidence_sha256"] != totals["success_evidence_sha256"]
        or source["selected_sessions"] != totals["source_sessions"]
        or selected["supervised_tokens"] != totals["supervised_tokens"]
        or selected["assistant_targets"] != totals["assistant_targets"]
    ):
        raise ValueError("reasoning census does not bind the exact dense corpus aggregates")
    schema = _authorize(checked_census, objective)
    selection = {
        "schema": SELECTION_SCHEMA,
        "census_sha256": checked_census["sha256"],
        "corpus_manifest_sha256": totals["corpus_manifest_sha256"],
        "objective": objective,
        "authorized_reasoning_schema": schema,
        "source": {
            "normalized_records_sha256": source["normalized_records_sha256"],
            "success_evidence_sha256": source["success_evidence_sha256"],
        },
        "selected": {
            "source_sessions": source["selected_sessions"],
            "supervised_tokens": selected["supervised_tokens"],
            "assistant_targets": selected["assistant_targets"],
        },
        "status": "approved",
    }
    selection["sha256"] = digest_json(selection)
    return selection


def validate_selection(
    value: Mapping[str, Any], *, census: Mapping[str, Any], corpus_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject a hand-authored or stale selection by rebuilding it from aggregates."""

    selection = _sealed(value, SELECTION_SCHEMA, "reasoning selection")
    _exact(
        selection,
        {
            "schema",
            "census_sha256",
            "corpus_manifest_sha256",
            "objective",
            "authorized_reasoning_schema",
            "source",
            "selected",
            "status",
            "sha256",
        },
        "reasoning selection",
    )
    if selection["status"] != "approved" or not isinstance(selection["objective"], str):
        raise ValueError("reasoning selection is not approved")
    expected = select(census, corpus_manifest, objective=selection["objective"])
    if selection != expected:
        raise ValueError("reasoning selection differs from its aggregate bindings")
    return copy.deepcopy(selection)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("aggregate metadata file is unreadable") from exc
    return _mapping(value, "aggregate metadata file")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--objective", default=VISIBLE_ACTIONS_ONLY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("reasoning selection destination already exists")
        selection = select(
            _read(args.census), _read(args.corpus_manifest), objective=args.objective
        )
        _write_once(args.output, selection)
    except (OSError, ValueError) as exc:
        # Do not forward potentially private metadata into stdout/stderr.
        print(f"error: {type(exc).__name__}", file=os.sys.stderr)
        return 2
    print(json.dumps({"schema": selection["schema"], "sha256": selection["sha256"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
