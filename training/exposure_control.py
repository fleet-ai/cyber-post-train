"""Build a whole-episode, exact-dose control for teacher exposure balancing.

Only sealed source metadata and aggregate dense-coverage receipts are read.  The
private records, prompts, token arrays, scores and held-out outcomes are not
inputs.  This module selects from an existing availability-weighted source set;
it does not prepare or launch training.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .io import digest_json, file_sha256, iter_jsonl
from .source_coverage import COUNTS, COVERAGE_SCHEMA
from .study_data import EPISODE_SCHEMA, _check_split, _sha, check_seal, seal, select_sources

POLICY_SCHEMA = "cyber_teacher_exposure_match_policy_v1"
RECEIPT_SCHEMA = "cyber_teacher_exposure_matched_control_v1"
COVERAGE_MANIFEST_SCHEMA = "cyber_sft_coverage_extraction_v1"
CORPUS_SCHEMA = "cyber_dense_sft_corpus_v1"
MAX_REACHABILITY_BITS = 2_000_000


class SearchExhausted(ValueError):
    """An exact answer was not established within the explicit state bound."""


@dataclass(frozen=True)
class Option:
    ids: tuple[str, ...]
    episodes: int
    segments: int
    supervised_tokens: int
    rank: tuple[int, int, int, str]


def _private_write_once(path: Path, value: dict) -> None:
    """Create a mode-0600 JSON file without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _write_outputs(
    selection_path: Path, receipt_path: Path, selection: dict, receipt: dict
) -> None:
    """Publish the create-once pair, rolling back a lone selection on failure."""
    if selection_path.resolve() == receipt_path.resolve():
        raise ValueError("selection and receipt outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in (selection_path, receipt_path)):
        raise FileExistsError("create-once exposure-control output exists")
    _private_write_once(selection_path, selection)
    try:
        _private_write_once(receipt_path, receipt)
    except BaseException:
        selection_path.unlink(missing_ok=True)
        raise


def _unique(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in result:
            raise ValueError(f"{label} requires unique episode identities")
        result[episode_id] = row
    return result


def _coverage_inputs(directory: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    check_seal(manifest, COVERAGE_MANIFEST_SCHEMA)
    expected_files = {"episodes.jsonl", "coverage-receipts.jsonl", "target-policy.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("coverage manifest file inventory differs")
    for name, expected in manifest["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError("coverage input file digest differs")
    episodes = _unique(list(iter_jsonl(directory / "episodes.jsonl")), "episode metadata")
    receipts = _unique(list(iter_jsonl(directory / "coverage-receipts.jsonl")), "coverage receipts")
    return manifest, episodes, receipts


def _selection(path: Path) -> dict:
    value = json.loads(path.read_text())
    check_seal(value, "cyber_sft_source_selection_v1")
    if value.get("status") != "ready" or value.get("blockers") != []:
        raise ValueError("source selection is not ready")
    return value


def _corpus(path: Path, selection: dict) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("corpus manifest must be a regular file")
    value = json.loads(path.read_text())
    check_seal(value, CORPUS_SCHEMA)
    files = value.get("files", {})
    train = files.get("train")
    if (
        set(files) != {"train"}
        or value.get("validation_mode") != "task_outcomes_only"
        or value.get("dev_windows") != 0
        or not isinstance(train, dict)
        or train.get("path") != "train.parquet"
        or not _sha(train.get("sha256"))
        or value.get("source_selection", {}).get("source_selection_sha256") != selection["sha256"]
        or train.get("source_sessions") != selection["totals"]["episodes"]
        or train.get("supervised_tokens") != selection["totals"]["supervised_tokens"]
    ):
        raise ValueError("train-only corpus does not reconcile to source selection")
    return value


def _episode_receipt(episode: dict, receipt: dict, target_policy_sha256: str) -> None:
    check_seal(episode, EPISODE_SCHEMA)
    check_seal(receipt, COVERAGE_SCHEMA)
    episode_coverage = episode["coverage"]
    receipt_coverage = receipt["coverage"]
    if (
        receipt["episode_id"] != episode["episode_id"]
        or episode_coverage["receipt_sha256"] != receipt["sha256"]
        or receipt["target_policy_sha256"] != target_policy_sha256
        or episode_coverage["target_policy_sha256"] != target_policy_sha256
        or receipt["trace_sha256"] != episode["trace_sha256"]
        or receipt["acceptance_sha256"] != episode["acceptance_sha256"]
        or receipt["normalized_record_sha256"] != episode["normalized_record_sha256"]
        or {key: receipt_coverage[key] for key in COUNTS}
        != {key: episode_coverage[key] for key in COUNTS}
        or type(receipt.get("segments")) is not int
        or receipt["segments"] <= 0
    ):
        raise ValueError("episode metadata and dense-coverage receipt differ")


def _possible(options: list[list[Option]], target_episodes: int, target_value: int, field: str):
    """Suffix reachability bitsets used only to prune the joint exact search."""
    mask = (1 << (target_value + 1)) - 1
    suffix: list[dict[int, int]] = [{} for _ in range(len(options) + 1)]
    suffix[-1] = {0: 1}
    for index in range(len(options) - 1, -1, -1):
        current: dict[int, int] = {}
        for option in options[index]:
            delta = getattr(option, field)
            for episodes, bits in suffix[index + 1].items():
                new_episodes = episodes + option.episodes
                if new_episodes <= target_episodes and delta <= target_value:
                    current[new_episodes] = current.get(new_episodes, 0) | ((bits << delta) & mask)
        suffix[index] = current
    return suffix


def _reachable(table: dict[int, int], episodes: int, value: int) -> bool:
    return episodes >= 0 and value >= 0 and bool((table.get(episodes, 0) >> value) & 1)


def choose_exact(
    families: dict[str, list[tuple[str, int, int]]],
    *,
    target_episodes: int,
    target_segments: int,
    target_supervised_tokens: int,
    seed: str,
    max_search_states: int = 200_000,
) -> tuple[list[str], int]:
    """Return the first family-locally preferred exact feasible whole-episode set.

    Each tuple is ``(episode_id, segments, supervised_tokens)``.  Every family
    contributes a nonempty subset.  Family and option ordering are canonical,
    so input order cannot change the answer.
    """
    if (
        not families
        or not isinstance(seed, str)
        or not seed
        or any(
            type(value) is not int or value <= 0
            for value in (target_episodes, target_segments, target_supervised_tokens)
        )
        or type(max_search_states) is not int
        or max_search_states <= 0
    ):
        raise ValueError("exact positive exposure targets and search bound required")
    if any(not isinstance(family_id, str) or not family_id for family_id in families):
        raise ValueError("family identities must be nonempty strings")
    all_ids = [episode_id for rows in families.values() for episode_id, _, _ in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("candidate episodes must be globally unique")
    parent_episodes = len(all_ids)
    parent_segments = sum(segments for rows in families.values() for _, segments, _ in rows)
    parent_tokens = sum(tokens for rows in families.values() for _, _, tokens in rows)
    if (
        target_episodes > parent_episodes
        or target_segments > parent_segments
        or target_supervised_tokens > parent_tokens
    ):
        raise ValueError("exposure target exceeds the candidate universe")
    if max(target_segments, target_supervised_tokens) > MAX_REACHABILITY_BITS:
        raise ValueError("exposure target exceeds the explicit reachability-bitset bound")
    family_order = sorted(families)
    ranked: list[list[Option]] = []
    for family_id in family_order:
        rows = sorted(families[family_id])
        if (
            not rows
            or len(rows) > 3
            or any(
                not isinstance(episode_id, str)
                or not episode_id
                or type(segments) is not int
                or segments <= 0
                or type(tokens) is not int
                or tokens <= 0
                for episode_id, segments, tokens in rows
            )
        ):
            raise ValueError("each family needs positive, identified episode exposures")
        family_segments = sum(row[1] for row in rows)
        family_tokens = sum(row[2] for row in rows)
        options = []
        for count in range(1, len(rows) + 1):
            for subset in itertools.combinations(rows, count):
                ids = tuple(row[0] for row in subset)
                segments = sum(row[1] for row in subset)
                tokens = sum(row[2] for row in subset)
                options.append(
                    Option(
                        ids=ids,
                        episodes=count,
                        segments=segments,
                        supervised_tokens=tokens,
                        rank=(
                            abs(parent_tokens * tokens - target_supervised_tokens * family_tokens),
                            abs(parent_segments * segments - target_segments * family_segments),
                            abs(parent_episodes * count - target_episodes * len(rows)),
                            digest_json(
                                {
                                    "seed": seed,
                                    "family_id": family_id,
                                    "selected_episode_ids": list(ids),
                                }
                            ),
                        ),
                    )
                )
        ranked.append(sorted(options, key=lambda option: option.rank))

    token_possible = _possible(
        ranked, target_episodes, target_supervised_tokens, "supervised_tokens"
    )
    segment_possible = _possible(ranked, target_episodes, target_segments, "segments")
    states = 0

    @functools.cache
    def search(index: int, episodes: int, segments: int, tokens: int):
        nonlocal states
        states += 1
        if states > max_search_states:
            raise SearchExhausted("exact exposure search exceeded its explicit state bound")
        if not _reachable(token_possible[index], episodes, tokens) or not _reachable(
            segment_possible[index], episodes, segments
        ):
            return None
        if index == len(ranked):
            return () if episodes == segments == tokens == 0 else None
        for option in ranked[index]:
            suffix = search(
                index + 1,
                episodes - option.episodes,
                segments - option.segments,
                tokens - option.supervised_tokens,
            )
            if suffix is not None:
                return (option, *suffix)
        return None

    chosen = search(0, target_episodes, target_segments, target_supervised_tokens)
    if chosen is None:
        raise ValueError("no exact whole-episode exposure match exists")
    return sorted(episode_id for option in chosen for episode_id in option.ids), states


def _distribution(
    ids: set[str],
    family_by_id: dict[str, str],
    episodes: dict[str, dict],
    receipts: dict[str, dict],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for episode_id in ids:
        stats = result.setdefault(
            family_by_id[episode_id], {"episodes": 0, "segments": 0, "supervised_tokens": 0}
        )
        stats["episodes"] += 1
        stats["segments"] += receipts[episode_id]["segments"]
        stats["supervised_tokens"] += episodes[episode_id]["coverage"]["supervised_tokens"]
    return result


def _tv(selected: dict[str, dict[str, int]], parent: dict[str, dict[str, int]], field: str):
    selected_total = sum(row[field] for row in selected.values())
    parent_total = sum(row[field] for row in parent.values())
    return (
        sum(
            abs(selected[family][field] / selected_total - row[field] / parent_total)
            for family, row in parent.items()
        )
        / 2
    )


def _gini(distribution: dict[str, dict[str, int]], field: str) -> float:
    values = [row[field] for row in distribution.values()]
    return sum(abs(left - right) for left in values for right in values) / (
        2 * len(values) * sum(values)
    )


def build(
    *,
    coverage_dir: Path,
    study_split_path: Path,
    available_selection_path: Path,
    balanced_selection_path: Path,
    available_corpus_manifest_path: Path,
    balanced_corpus_manifest_path: Path,
    seed: str,
    max_search_states: int = 200_000,
) -> tuple[dict, dict]:
    """Build a private source selection and a sanitized aggregate receipt."""
    coverage_manifest, episodes, receipts = _coverage_inputs(coverage_dir)
    split = json.loads(study_split_path.read_text())
    _check_split(split)
    available = _selection(available_selection_path)
    balanced = _selection(balanced_selection_path)
    available_corpus = _corpus(available_corpus_manifest_path, available)
    balanced_corpus = _corpus(balanced_corpus_manifest_path, balanced)
    if any(
        available[key] != balanced[key]
        for key in ("study_split_sha256", "training_split_sha256", "target_policy_sha256", "models")
    ) or any(
        available[key] != expected
        for key, expected in (
            ("study_split_sha256", split["sha256"]),
            ("training_split_sha256", split["training_split"]["sha256"]),
            ("target_policy_sha256", coverage_manifest["target_policy_sha256"]),
        )
    ):
        raise ValueError("available, balanced, coverage and split bindings differ")
    parity_fields = (
        "source_sha256",
        "split_sha256",
        "tokenizer",
        "train_models",
        "max_length",
        "context_tokens",
        "dev_windows",
        "validation_mode",
        "fleet_dev_protocol_sha256",
        "builder_sha256",
    )
    if any(available_corpus[key] != balanced_corpus[key] for key in parity_fields):
        raise ValueError("available and balanced corpora do not share one dense interface")

    assignments = {
        (row["task_key"], row["task_version_id"]): row["group_id"] for row in split["tasks"]
    }
    relevant = set(available["selected_episode_ids"]) | set(balanced["selected_episode_ids"])
    family_by_id = {}
    for selection in (available, balanced):
        evidence = {row["episode_id"]: row for row in selection["selected_evidence"]}
        if set(evidence) != set(selection["selected_episode_ids"]):
            raise ValueError("source selection evidence is incomplete")
        for episode_id in selection["selected_episode_ids"]:
            if episode_id not in episodes or episode_id not in receipts:
                raise ValueError("selected episode lacks sealed coverage metadata")
            episode = episodes[episode_id]
            _episode_receipt(episode, receipts[episode_id], available["target_policy_sha256"])
            if any(
                evidence[episode_id][evidence_key] != episode[episode_key]
                for evidence_key, episode_key in (
                    ("metadata_sha256", "sha256"),
                    ("trace_sha256", "trace_sha256"),
                    ("acceptance_sha256", "acceptance_sha256"),
                    ("normalized_record_sha256", "normalized_record_sha256"),
                )
            ):
                raise ValueError("source selection evidence differs from episode metadata")
            key = (episode["task_key"], episode["task_version_id"])
            if key not in assignments:
                raise ValueError("selected episode is outside the exact study split")
            family_by_id[episode_id] = assignments[key]
    if set(family_by_id) != relevant:
        raise ValueError("selected episode join is incomplete")

    available_ids = set(available["selected_episode_ids"])
    balanced_ids = set(balanced["selected_episode_ids"])
    available_distribution = _distribution(available_ids, family_by_id, episodes, receipts)
    balanced_distribution = _distribution(balanced_ids, family_by_id, episodes, receipts)
    if set(available_distribution) != set(balanced_distribution):
        raise ValueError("balanced and available selections cover different families")
    target = {
        "episodes": balanced_corpus["files"]["train"]["source_sessions"],
        "segments": balanced_corpus["files"]["train"]["rows"],
        "supervised_tokens": balanced_corpus["files"]["train"]["supervised_tokens"],
    }
    if (
        target["episodes"] != len(balanced_ids)
        or target["segments"] != sum(receipts[row]["segments"] for row in balanced_ids)
        or target["supervised_tokens"]
        != sum(episodes[row]["coverage"]["supervised_tokens"] for row in balanced_ids)
    ):
        raise ValueError("balanced selection and corpus target counts differ")
    families: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for episode_id in available_ids:
        families[family_by_id[episode_id]].append(
            (
                episode_id,
                receipts[episode_id]["segments"],
                episodes[episode_id]["coverage"]["supervised_tokens"],
            )
        )
    chosen, search_states = choose_exact(
        dict(families),
        target_episodes=target["episodes"],
        target_segments=target["segments"],
        target_supervised_tokens=target["supervised_tokens"],
        seed=seed,
        max_search_states=max_search_states,
    )
    chosen_ids = set(chosen)
    chosen_distribution = _distribution(chosen_ids, family_by_id, episodes, receipts)
    if (
        not chosen_ids <= available_ids
        or set(chosen_distribution) != set(available_distribution)
        or {
            field: sum(row[field] for row in chosen_distribution.values())
            for field in ("episodes", "segments", "supervised_tokens")
        }
        != target
    ):
        raise ValueError("solver result violates the exact exposure contract")

    policy = seal(
        {
            "schema": POLICY_SCHEMA,
            "role": "availability_shaped_exposure_matched_control",
            "candidate_universe": "exact_available_source_selection",
            "unit": "whole_certified_native_episode",
            "family_coverage": "every_parent_covered_family_once_or_more",
            "hard_match": ["episodes", "dense_segments", "supervised_tokens"],
            "family_order": "ascending_group_id",
            "option_order": [
                "family_supervised_token_share_cross_product_error",
                "family_dense_segment_share_cross_product_error",
                "family_episode_share_cross_product_error",
                "seeded_subset_digest",
            ],
            "search": "lexicographically_first_family-local_option_vector_with_exact_completion",
            "seed": seed,
            "max_search_states": max_search_states,
            "prohibited": [
                "partial episodes",
                "duplicated target tokens",
                "epoch-count dose correction",
                "payload, score or held-out outcome selection",
            ],
        }
    )
    selected_episodes = [episodes[episode_id] for episode_id in chosen]
    result = select_sources(
        selected_episodes,
        split,
        models=available["models"],
        target_policy_sha256=available["target_policy_sha256"],
        max_episodes_per_family=available["policy"]["max_episodes_per_family"],
        max_supervised_tokens_per_family=available["policy"]["max_supervised_tokens_per_family"],
        source_seed=seed,
        max_submit_token_share=available["policy"]["max_submit_token_share"],
        max_submit_response_share=available["policy"]["max_submit_response_share"],
    )
    if set(result["selected_episode_ids"]) != chosen_ids or result["status"] != "ready":
        raise ValueError("standard source-selection gates changed the exact solver result")
    payload = {key: value for key, value in result.items() if key != "sha256"}
    payload["policy"] = {
        **payload["policy"],
        "family_selection": "exact availability-shaped exposure match; whole episodes",
        "exposure_match_policy_sha256": policy["sha256"],
        "available_source_selection_sha256": available["sha256"],
        "balanced_source_selection_sha256": balanced["sha256"],
        "exact_target": target,
    }
    payload["limitations"] = [
        *payload["limitations"],
        "Aggregate input/context-token exposure is unavailable and therefore not matched.",
        "Family-share preservation is locally preferred subject to exact joint feasibility; "
        "it is not a global minimum-distance solution or an unbiased sample.",
    ]
    result = seal(payload)
    achieved = {
        "families": len(chosen_distribution),
        **target,
        "nominal_batch8_batches": (target["segments"] + 7) // 8,
    }
    receipt = seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "exact_feasible",
            "policy": policy,
            "inputs": {
                "study_split_sha256": split["sha256"],
                "coverage_manifest_sha256": coverage_manifest["sha256"],
                "episode_metadata_sha256": coverage_manifest["files"]["episodes.jsonl"],
                "coverage_receipts_sha256": coverage_manifest["files"]["coverage-receipts.jsonl"],
                "target_policy_sha256": available["target_policy_sha256"],
                "available_source_selection_sha256": available["sha256"],
                "balanced_source_selection_sha256": balanced["sha256"],
                "available_corpus_sha256": available_corpus["sha256"],
                "balanced_corpus_sha256": balanced_corpus["sha256"],
                "algorithm_sha256": file_sha256(Path(__file__)),
            },
            "candidate": {
                "families": len(available_distribution),
                "episodes": len(available_ids),
                "segments": sum(row["segments"] for row in available_distribution.values()),
                "supervised_tokens": sum(
                    row["supervised_tokens"] for row in available_distribution.values()
                ),
            },
            "target": target,
            "achieved": achieved,
            "selected_episode_ids_sha256": digest_json(chosen),
            "source_selection_sha256": result["sha256"],
            "search_states": search_states,
            "family_distribution": {
                label: {
                    "supervised_token_gini": round(_gini(distribution, "supervised_tokens"), 9),
                    "tv_to_available": {
                        field: (
                            round(_tv(distribution, available_distribution, field), 9)
                            if label != "available"
                            else 0.0
                        )
                        for field in ("episodes", "segments", "supervised_tokens")
                    },
                }
                for label, distribution in (
                    ("available", available_distribution),
                    ("matched_control", chosen_distribution),
                    ("balanced", balanced_distribution),
                )
            },
            "interpretation": (
                "This controls episode, dense-window and supervised-token dose while retaining "
                "the available-teacher family weighting as closely as the frozen local policy "
                "permits. It is a teacher-availability treatment, not an optimality claim."
            ),
            "limitations": [
                "Whole episodes are indivisible, so exact dose and exact parent family shares "
                "cannot generally both hold.",
                "Aggregate input/context-token exposure is unavailable and is not matched.",
                "Outcome comparison still requires exact dev qualification and matched evaluation.",
            ],
        }
    )
    return result, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", required=True, type=Path)
    parser.add_argument("--study-split", required=True, type=Path)
    parser.add_argument("--available-selection", required=True, type=Path)
    parser.add_argument("--balanced-selection", required=True, type=Path)
    parser.add_argument("--available-corpus-manifest", required=True, type=Path)
    parser.add_argument("--balanced-corpus-manifest", required=True, type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-selection", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--max-search-states", type=int, default=200_000)
    args = parser.parse_args()
    try:
        selection, receipt = build(
            coverage_dir=args.coverage_dir,
            study_split_path=args.study_split,
            available_selection_path=args.available_selection,
            balanced_selection_path=args.balanced_selection,
            available_corpus_manifest_path=args.available_corpus_manifest,
            balanced_corpus_manifest_path=args.balanced_corpus_manifest,
            seed=args.seed,
            max_search_states=args.max_search_states,
        )
        _write_outputs(args.output_selection, args.output_receipt, selection, receipt)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"exposure control not written ({type(exc).__name__})\n")
    print(
        json.dumps(
            {
                "schema": RECEIPT_SCHEMA,
                "receipt_sha256": receipt["sha256"],
                "source_selection_sha256": selection["sha256"],
                "achieved": receipt["achieved"],
                "submitted_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
