"""Authorize one distinct Qwen/OpenCode visible-reasoning corpus by aggregates.

This is deliberately separate from :mod:`training.reasoning_census`.  That
module remains the action-only gate and must continue to reject every written
reasoning objective.  This module accepts only the narrow Qwen-self/OpenCode
corpus identity produced by ``fleet_visible_reasoning_corpus``; it never opens
records, prompts, token rows, or transcripts.
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
from .task_family_split import TRUSTED_FLEET_COLLECTION_ROOT_ID

SOURCE_CENSUS_SCHEMA = "cyber_qwen_opencode_student_visible_reasoning_census_v1"
SELECTION_SCHEMA = "cyber_qwen_opencode_visible_reasoning_training_selection_v1"
OBJECTIVE = "qwen_opencode_student_visible_reasoning_plus_actions"
VISIBILITY = ("student_visible", "private_or_unknown", "absent")
COMPACTION = ("none", "exact_student_generated", "opaque_rejected")
MINIMUM_SUPERVISED_TOKENS = 20_000_000
MAXIMUM_FAMILY_TARGET_FRACTION = 0.25


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected:
        raise ValueError(f"{label} has unknown or missing fields")
    return result


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in value.removeprefix("sha256:")):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _count(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if result.get("schema") != schema or result.get("sha256") != digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {label} digest or schema")
    return result


def validate_source_census(
    value: Mapping[str, Any],
    *,
    source_profile_sha256: str,
    source_authorization_sha256: str,
    collection_packet_sha256: str,
    private_selection_sha256: str,
    success_evidence_sha256: str,
    selected_sessions: int,
) -> dict[str, Any]:
    """Validate source-only aggregate evidence before private records are opened."""

    census = _sealed(value, SOURCE_CENSUS_SCHEMA, "visible-reasoning source census")
    _exact(
        census,
        {
            "schema",
            "source_profile_sha256",
            "source_authorization_sha256",
            "collection_packet_sha256",
            "private_selection_sha256",
            "success_evidence_sha256",
            "sessions",
            "visibility",
            "target_tokens",
            "compaction",
            "sha256",
        },
        "visible-reasoning source census",
    )
    expected = {
        "source_profile_sha256": source_profile_sha256,
        "source_authorization_sha256": source_authorization_sha256,
        "collection_packet_sha256": collection_packet_sha256,
        "private_selection_sha256": private_selection_sha256,
        "success_evidence_sha256": success_evidence_sha256,
    }
    for name, target in expected.items():
        if census.get(name) != _sha(target, name):
            raise ValueError("visible-reasoning source census changes an immutable binding")
    sessions = _exact(
        census["sessions"],
        {"candidates", "verified_successes", "selected"},
        "visible-reasoning census sessions",
    )
    candidates = _count(sessions["candidates"], "candidate sessions")
    verified = _count(sessions["verified_successes"], "verified success sessions")
    selected = _count(sessions["selected"], "selected sessions")
    if (
        not candidates
        or not verified
        or selected != selected_sessions
        or selected > verified
        or verified > candidates
    ):
        raise ValueError("visible-reasoning census session totals are inconsistent")
    visibility = _exact(
        census["visibility"], set(VISIBILITY), "visible-reasoning census visibility"
    )
    values = {name: _count(visibility[name], f"visibility {name}") for name in VISIBILITY}
    if sum(values.values()) != candidates or values["student_visible"] < selected:
        raise ValueError("visible-reasoning census visibility totals are inconsistent")
    targets = _exact(
        census["target_tokens"], {"student_visible_reasoning", "visible_action"}, "target totals"
    )
    if not all(_count(targets[name], f"target tokens {name}", positive=True) for name in targets):
        raise ValueError("visible-reasoning source census requires paired positive target totals")
    compaction = _exact(
        census["compaction"], set(COMPACTION), "visible-reasoning census compaction"
    )
    compaction_values = {
        name: _count(compaction[name], f"compaction {name}") for name in COMPACTION
    }
    if sum(compaction_values.values()) != candidates:
        raise ValueError("visible-reasoning census compaction totals are inconsistent")
    return copy.deepcopy(census)


def _manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _sealed(
        value,
        "cyber_qwen_opencode_visible_reasoning_sft_corpus_v1",
        "visible-reasoning corpus manifest",
    )
    _exact(
        manifest,
        {
            "schema",
            "source_profile_sha256",
            "source_authorization_sha256",
            "collection_packet_sha256",
            "selection_sha256",
            "success_evidence_sha256",
            "source_census_sha256",
            "catalog_inventory_sha256",
            "family_split_sha256",
            "root_role_anchor_id",
            "family_role_anchor_sha256",
            "protected_family_lock_sha256",
            "runtime_bindings_sha256",
            "files",
            "counts",
            "validation_mode",
            "coverage_sha256",
            "limitations",
            "builder_sha256",
            "sha256",
        },
        "visible-reasoning corpus manifest",
    )
    if manifest["validation_mode"] != "pending_reasoning_selection":
        raise ValueError("visible-reasoning corpus has already bypassed final selection")
    if manifest["root_role_anchor_id"] != TRUSTED_FLEET_COLLECTION_ROOT_ID:
        raise ValueError("visible-reasoning corpus is not rooted in the trusted family split")
    for name in (
        "source_profile_sha256",
        "source_authorization_sha256",
        "collection_packet_sha256",
        "selection_sha256",
        "success_evidence_sha256",
        "source_census_sha256",
        "catalog_inventory_sha256",
        "family_split_sha256",
        "family_role_anchor_sha256",
        "protected_family_lock_sha256",
        "runtime_bindings_sha256",
        "coverage_sha256",
    ):
        _sha(manifest[name], f"visible-reasoning manifest {name}")
    files = _exact(manifest["files"], {"train"}, "visible-reasoning corpus files")
    train = _exact(files["train"], {"path", "sha256", "rows"}, "visible-reasoning train file")
    if train["path"] != "train.parquet" or _count(train["rows"], "train rows", positive=True) < 1:
        raise ValueError("visible-reasoning corpus train file is malformed")
    _sha(train["sha256"], "visible-reasoning train file")
    counts = _exact(
        manifest["counts"],
        {
            "source_records",
            "windows",
            "student_visible_reasoning_target_tokens",
            "visible_action_target_tokens",
            "supervised_tokens",
        },
        "visible-reasoning manifest counts",
    )
    for name in counts:
        _count(counts[name], f"manifest {name}", positive=True)
    if counts["supervised_tokens"] != (
        counts["student_visible_reasoning_target_tokens"] + counts["visible_action_target_tokens"]
    ):
        raise ValueError("visible-reasoning manifest target totals are inconsistent")
    if train["rows"] != counts["windows"]:
        raise ValueError("visible-reasoning train rows differ from manifest windows")
    return manifest


def _coverage(value: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _sealed(
        value,
        "cyber_qwen_opencode_visible_reasoning_coverage_v1",
        "visible-reasoning coverage",
    )
    _exact(
        coverage,
        {
            "schema",
            "selection_sha256",
            "success_evidence_sha256",
            "source_census_sha256",
            "collection_packet_sha256",
            "source_profile_sha256",
            "source_authorization_sha256",
            "selected_source_records",
            "visible_reasoning_windows",
            "student_visible_reasoning_target_tokens",
            "visible_action_target_tokens",
            "unique_supervised_tokens",
            "minimum_unique_supervised_tokens",
            "target_goal_reached",
            "family_token_concentration",
            "compaction",
            "heldout_families_materialized",
            "raw_text_written",
            "sha256",
        },
        "visible-reasoning coverage",
    )
    if coverage["heldout_families_materialized"] != 0 or coverage["raw_text_written"] is not False:
        raise ValueError("visible-reasoning coverage violates the private family boundary")
    for name in (
        "selection_sha256",
        "success_evidence_sha256",
        "source_census_sha256",
        "collection_packet_sha256",
        "source_profile_sha256",
        "source_authorization_sha256",
    ):
        _sha(coverage[name], f"visible-reasoning coverage {name}")
    for name in (
        "selected_source_records",
        "visible_reasoning_windows",
        "student_visible_reasoning_target_tokens",
        "visible_action_target_tokens",
        "unique_supervised_tokens",
        "minimum_unique_supervised_tokens",
    ):
        _count(coverage[name], f"visible-reasoning coverage {name}", positive=True)
    if type(coverage["target_goal_reached"]) is not bool:
        raise ValueError("visible-reasoning coverage target status is invalid")
    minimum = _count(
        coverage["minimum_unique_supervised_tokens"],
        "visible-reasoning coverage minimum supervised tokens",
        positive=True,
    )
    unique = _count(
        coverage["unique_supervised_tokens"],
        "visible-reasoning coverage unique supervised tokens",
        positive=True,
    )
    if minimum < MINIMUM_SUPERVISED_TOKENS or coverage["target_goal_reached"] is not (
        unique >= minimum
    ):
        raise ValueError("visible-reasoning coverage target gate is not mathematically bound")
    family = _exact(
        coverage["family_token_concentration"],
        {
            "families_with_targets",
            "largest_family_target_token_fraction",
            "maximum_allowed_fraction",
            "within_limit",
        },
        "family concentration",
    )
    if (
        _count(family["families_with_targets"], "families with targets", positive=True) < 1
        or type(family["largest_family_target_token_fraction"]) not in {int, float}
        or type(family["maximum_allowed_fraction"]) not in {int, float}
        or family["maximum_allowed_fraction"] != MAXIMUM_FAMILY_TARGET_FRACTION
        or not 0 < family["largest_family_target_token_fraction"] <= MAXIMUM_FAMILY_TARGET_FRACTION
        or family["within_limit"] is not True
    ):
        raise ValueError("visible-reasoning coverage exceeds the family concentration limit")
    compaction = _exact(
        coverage["compaction"],
        {
            "uncompacted_windows",
            "exact_student_generated_continuation_windows",
            "opaque_compaction_windows",
        },
        "visible-reasoning corpus compaction",
    )
    values = {
        name: _count(compaction[name], f"visible-reasoning compaction {name}")
        for name in compaction
    }
    if (
        values["opaque_compaction_windows"]
        or sum(values.values()) != coverage["visible_reasoning_windows"]
    ):
        raise ValueError("visible-reasoning coverage includes opaque or unbound compaction")
    return coverage


def select(
    census: Mapping[str, Any], corpus_manifest: Mapping[str, Any], coverage: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a source-only qualification, never collection or training authority."""

    manifest = _manifest(corpus_manifest)
    checked_coverage = _coverage(coverage)
    checked_census = validate_source_census(
        census,
        source_profile_sha256=manifest["source_profile_sha256"],
        source_authorization_sha256=manifest["source_authorization_sha256"],
        collection_packet_sha256=manifest["collection_packet_sha256"],
        private_selection_sha256=manifest["selection_sha256"],
        success_evidence_sha256=manifest["success_evidence_sha256"],
        selected_sessions=manifest["counts"]["source_records"],
    )
    for name in (
        "selection_sha256",
        "success_evidence_sha256",
        "source_census_sha256",
        "collection_packet_sha256",
        "source_profile_sha256",
        "source_authorization_sha256",
    ):
        manifest_name = name
        if name == "source_census_sha256":
            expected = checked_census["sha256"]
        else:
            expected = manifest[manifest_name]
        if checked_coverage.get(name) != expected:
            raise ValueError("visible-reasoning coverage does not bind the exact manifest")
    counts = manifest["counts"]
    if (
        checked_coverage["selected_source_records"] != counts["source_records"]
        or checked_coverage["visible_reasoning_windows"] != counts["windows"]
        or checked_coverage["student_visible_reasoning_target_tokens"]
        != counts["student_visible_reasoning_target_tokens"]
        or checked_coverage["visible_action_target_tokens"]
        != counts["visible_action_target_tokens"]
        or checked_coverage["unique_supervised_tokens"] != counts["supervised_tokens"]
    ):
        raise ValueError("visible-reasoning coverage totals differ from the corpus manifest")
    source_targets = checked_census["target_tokens"]
    if (
        source_targets["student_visible_reasoning"]
        != counts["student_visible_reasoning_target_tokens"]
        or source_targets["visible_action"] != counts["visible_action_target_tokens"]
    ):
        raise ValueError("visible-reasoning source census target totals differ from the corpus")
    if not checked_coverage["target_goal_reached"] or not checked_coverage[
        "family_token_concentration"
    ].get("within_limit"):
        raise ValueError("visible-reasoning corpus has not reached its declared collection gates")
    selection = {
        "schema": SELECTION_SCHEMA,
        "objective": OBJECTIVE,
        "source_census_sha256": checked_census["sha256"],
        "corpus_manifest_sha256": manifest["sha256"],
        "coverage_sha256": checked_coverage["sha256"],
        "source_profile_sha256": manifest["source_profile_sha256"],
        "source_authorization_sha256": manifest["source_authorization_sha256"],
        "collection_packet_sha256": manifest["collection_packet_sha256"],
        "success_evidence_sha256": manifest["success_evidence_sha256"],
        "selected": {
            "source_records": counts["source_records"],
            "windows": counts["windows"],
            "student_visible_reasoning_target_tokens": counts[
                "student_visible_reasoning_target_tokens"
            ],
            "visible_action_target_tokens": counts["visible_action_target_tokens"],
            "supervised_tokens": counts["supervised_tokens"],
        },
        "status": "source_only_qualified",
    }
    selection["sha256"] = digest_json(selection)
    return selection


def validate_selection(
    value: Mapping[str, Any],
    *,
    census: Mapping[str, Any],
    corpus_manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    selection = _sealed(value, SELECTION_SCHEMA, "visible-reasoning training selection")
    expected = select(census, corpus_manifest, coverage)
    if selection != expected:
        raise ValueError("visible-reasoning training selection differs from its aggregate bindings")
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


def authorize_paths(
    census_path: Path, corpus_manifest_path: Path, coverage_path: Path, output_path: Path
) -> dict[str, Any]:
    """Write an aggregate-only qualification once; never open corpus rows.

    This is deliberately separate from materialization.  A caller receives a
    digest-bound qualification only after checking the aggregate source census
    against the token-only corpus manifest and coverage receipt. It does not
    authenticate a live Registry artifact, authorize collection, or make an
    SFT launcher eligible to consume the corpus.
    """

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("visible-reasoning selection destination already exists")
    selection = select(_read(census_path), _read(corpus_manifest_path), _read(coverage_path))
    _write_once(output_path, selection)
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        selection = authorize_paths(args.census, args.corpus_manifest, args.coverage, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}", file=os.sys.stderr)
        return 2
    print(json.dumps({"schema": selection["schema"], "sha256": selection["sha256"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
