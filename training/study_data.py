"""Metadata-only representative splits and SFT source coverage gates.

No source payloads, scores, model losses, or external benchmark metadata are
inputs. Certification/taxonomy/exposure claims need immutable evidence digests;
this module validates bindings, not the truth of an unavailable upstream receipt.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .io import digest_json

AXES = ("application", "environment", "vulnerability_family", "difficulty")
INVENTORY_SCHEMA = "cyber_study_inventory_v1"
LOCK_SCHEMA = "cyber_final_test_lock_v1"
SPLIT_SCHEMA = "cyber_study_split_v1"
EPISODE_SCHEMA = "cyber_sft_episode_metadata_v1"


def seal(value: dict) -> dict:
    return {**value, "sha256": digest_json(value)}


def check_seal(value: Mapping, schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != digest_json(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("schema or immutable metadata digest mismatch")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _label(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value.strip()) <= 256 and "\n" not in value


def _rank(seed: str, value: str) -> str:
    if not _label(seed):
        raise ValueError("explicit nonempty split seed required")
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _taxonomy(field: Mapping | None, *, grouping: bool = False) -> tuple[str, ...]:
    if field is None:
        return ("__missing__",)
    if not isinstance(field, Mapping) or field.get("status") not in {
        "missing",
        "unverified",
        "verified",
    }:
        raise ValueError("taxonomy status must explicitly distinguish missing and unverified")
    if field["status"] == "missing":
        return ("__missing__",)
    if field["status"] == "unverified":
        return ("__unverified__",)
    value = field.get("value")
    labels = [value] if isinstance(value, str) else value
    if (
        not _sha(field.get("evidence_sha256"))
        or not isinstance(labels, list)
        or not labels
        or any(not _label(x) or x.startswith("__") for x in labels)
        or len(set(labels)) != len(labels)
        or (grouping and len(labels) != 1)
    ):
        raise ValueError("verified taxonomy requires exact evidence and unique bounded labels")
    return tuple(sorted(labels))


def _groups(inventory: dict) -> tuple[dict, dict]:
    check_seal(inventory, INVENTORY_SCHEMA)
    if inventory.get("origin") != "fleet_cyber":
        raise ValueError("only authorized Fleet metadata may enter training split design")
    rows = inventory.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty exact task inventory required")
    groups, identities, lineage_groups, task_groups = {}, set(), {}, {}
    quarantine, missing = collections.Counter(), collections.Counter()
    unresolved_aliases = set()
    for row in rows:
        if set(row) != {
            "task_key",
            "task_version_id",
            "lineage_id",
            "taxonomy",
            "certification",
            "exposure",
        }:
            raise ValueError("task inventory must contain only the metadata contract fields")
        identity = (row["task_key"], row["task_version_id"])
        if any(not _label(x) for x in (*identity, row["lineage_id"])) or identity in identities:
            raise ValueError("missing or duplicate exact task identity")
        identities.add(identity)
        cert = row["certification"]
        if not isinstance(cert, Mapping) or cert.get("status") not in {
            "accepted",
            "pending",
            "rejected",
            "unknown",
        }:
            raise ValueError("explicit task certification status required")
        accepted = cert["status"] == "accepted"
        if accepted and not _sha(cert.get("receipt_sha256")):
            raise ValueError("accepted task lacks immutable certification receipt")
        taxonomy = row["taxonomy"]
        if not isinstance(taxonomy, Mapping) or set(taxonomy) - {*AXES, "task_family"}:
            raise ValueError("unknown taxonomy axis; do not infer labels from payloads")
        family = _taxonomy(taxonomy.get("task_family"), grouping=True)
        if family[0].startswith("__"):
            quarantine["missing_verified_task_family"] += 1
            unresolved_aliases.update({("lineage", row["lineage_id"]), ("key", row["task_key"])})
            continue
        group_id = digest_json({"task_family": family[0]})
        for mapping, key in ((lineage_groups, row["lineage_id"]), (task_groups, row["task_key"])):
            if mapping.setdefault(key, group_id) != group_id:
                raise ValueError("one task lineage or key changes reviewed family across versions")
        group = groups.setdefault(
            group_id,
            {
                "id": group_id,
                "task_family": family[0],
                "rows": [],
                "features": set(),
                "unexposed": True,
            },
        )
        exposure = row["exposure"]
        if not isinstance(exposure, Mapping) or exposure.get("status") not in {
            "unexposed",
            "exposed",
            "unknown",
        }:
            raise ValueError("unknown training-exposure status")
        unexposed = exposure["status"] == "unexposed" and _sha(exposure.get("receipt_sha256"))
        if exposure["status"] == "unexposed" and not unexposed:
            raise ValueError("untouched-test eligibility needs exposure evidence")
        group["unexposed"] &= unexposed
        # A rejected/pending version is not runnable, but its known exposure must
        # still disqualify the whole family from an untouched final test.
        if not accepted:
            quarantine["not_certified_accepted"] += 1
            continue
        group["rows"].append(row)
        for axis in AXES:
            values = _taxonomy(taxonomy.get(axis))
            group["features"].update((axis, value) for value in values)
            if values[0].startswith("__"):
                missing[axis] += 1
    for key, group in list(groups.items()):
        if any(
            ("lineage", r["lineage_id"]) in unresolved_aliases
            or ("key", r["task_key"]) in unresolved_aliases
            for r in group["rows"]
        ):
            quarantine["family_has_unresolved_alias"] += len(group["rows"])
            del groups[key]
        elif not group["rows"]:
            del groups[key]
    for group in groups.values():
        group["rows"].sort(key=lambda r: (r["task_key"], r["task_version_id"]))
    audit = {
        "inventory_task_versions": len(rows),
        "eligible_task_versions": sum(len(g["rows"]) for g in groups.values()),
        "reviewed_task_families": len(groups),
        "quarantined_task_versions": dict(quarantine),
        "missing_or_unverified_taxonomy_task_versions": {k: missing[k] for k in AXES},
        "full_taxonomy_available": not missing and not quarantine,
        "evidence_scope": (
            "metadata claims bound by digests; unavailable upstream receipts are not re-certified"
        ),
    }
    return groups, audit


def audit_inventory(inventory: dict) -> dict:
    _, audit = _groups(inventory)
    return audit


def _representative(
    groups: dict, count: int, seed: str, *, initial: Iterable[str] = ()
) -> list[str]:
    """Deterministic greedy balancing of group-weighted taxonomy marginals."""
    if type(count) is not int or not 0 <= count <= len(groups):
        raise ValueError("requested grouped split size exceeds eligible inventory")
    picked = sorted(set(initial))
    if not set(picked) <= groups.keys() or len(picked) > count:
        raise ValueError("fixed split families exceed candidate pool or target size")
    population = collections.Counter(f for g in groups.values() for f in g["features"])
    axis_sizes = collections.Counter(axis for axis, _ in population)
    observed = collections.Counter(f for key in picked for f in groups[key]["features"])
    for step in range(len(picked) + 1, count + 1):
        # Each axis contributes equally; rare labels do not create huge joint strata.
        def cost(key: str, step: int = step) -> tuple[float, str]:
            features = groups[key]["features"]
            error = sum(
                (observed[f] + (f in features) - step * n / len(groups)) ** 2 / axis_sizes[f[0]]
                for f, n in sorted(population.items())
            )
            return error, _rank(seed, key)

        chosen = min((k for k in groups if k not in picked), key=cost)
        picked.append(chosen)
        observed.update(groups[chosen]["features"])
    return sorted(picked)


def _distribution(groups: dict, selected: set[str]) -> dict:
    result = {}
    for axis in AXES:
        available = collections.Counter(
            value for g in groups.values() for a, value in g["features"] if a == axis
        )
        actual = collections.Counter(
            value for k in selected for a, value in groups[k]["features"] if a == axis
        )
        result[axis] = {
            "population_group_counts": dict(sorted(available.items())),
            "split_group_counts": dict(sorted(actual.items())),
            "max_abs_group_share_gap": max(
                (abs(actual[k] / len(selected) - n / len(groups)) for k, n in available.items()),
                default=0,
            )
            if selected
            else None,
        }
    return result


def freeze_final_test(
    inventory: dict,
    *,
    groups_count: int,
    seed: str,
    protected_task_versions: Iterable[tuple[str, str]] = (),
) -> dict:
    """Freeze once before ablations; only evidenced-unexposed groups are eligible.

    Existing sealed test identities may be supplied as protected exact tuples.
    Their complete reviewed family is included, never only one task version.
    """
    groups, audit = _groups(inventory)
    _rank(seed, "validate_seed")
    eligible = {k: g for k, g in groups.items() if g["unexposed"]}
    protected = set(protected_task_versions)
    resolved = {
        k
        for k, g in groups.items()
        if any((r["task_key"], r["task_version_id"]) in protected for r in g["rows"])
    }
    known = {(r["task_key"], r["task_version_id"]) for k in resolved for r in groups[k]["rows"]}
    if not protected <= known or not resolved <= eligible.keys():
        raise ValueError(
            "protected final test is absent, uncertified, exposed or lacks exposure evidence"
        )
    if (
        type(groups_count) is not int
        or not 0 < groups_count < len(groups)
        or len(resolved) > groups_count
    ):
        raise ValueError("invalid final-test family count")
    chosen = set(_representative(eligible, groups_count, seed, initial=resolved))
    return seal(
        {
            "schema": LOCK_SCHEMA,
            "inventory_sha256": inventory["sha256"],
            "seed": seed,
            "group_ids": sorted(chosen),
            "tasks": sorted(
                [
                    {
                        "task_key": r["task_key"],
                        "task_version_id": r["task_version_id"],
                        "group_id": k,
                    }
                    for k in chosen
                    for r in groups[k]["rows"]
                ],
                key=lambda r: (r["task_key"], r["task_version_id"]),
            ),
            "selection_basis": (
                "certified inventory taxonomy and unexposed-family evidence only; "
                "no outcomes or losses"
            ),
            "protected_task_versions": sorted(protected),
            "inventory_audit": audit,
            "representation": _distribution(groups, chosen),
        }
    )


def split_variant(inventory: dict, final_test: dict, *, dev_groups: int, seed: str) -> dict:
    groups, audit = _groups(inventory)
    _rank(seed, "validate_seed")
    check_seal(final_test, LOCK_SCHEMA)
    if final_test["inventory_sha256"] != inventory["sha256"]:
        raise ValueError("ablation inventory changed; do not silently refreeze the final test")
    test_ids = set(final_test["group_ids"])
    if (
        not test_ids
        or len(test_ids) != len(final_test["group_ids"])
        or not test_ids <= groups.keys()
        or any(not groups[k]["unexposed"] for k in test_ids)
    ):
        raise ValueError("final test lock is not exactly eligible in this inventory")
    expected = {
        (r["task_key"], r["task_version_id"], k) for k in test_ids for r in groups[k]["rows"]
    }
    if len(expected) != len(final_test["tasks"]) or expected != {
        (r["task_key"], r["task_version_id"], r["group_id"]) for r in final_test["tasks"]
    }:
        raise ValueError("final test lock drops a family version or changes identity")
    available = {k: g for k, g in groups.items() if k not in test_ids}
    if type(dev_groups) is not int or not 0 < dev_groups < len(available):
        raise ValueError("positive train and development family counts required")
    dev_ids = set(_representative(available, dev_groups, seed))
    sets = {"train": set(available) - dev_ids, "dev": dev_ids, "final_test": test_ids}
    tasks = sorted(
        [
            {
                "task_key": r["task_key"],
                "task_version_id": r["task_version_id"],
                "group_id": k,
                "split": assigned,
            }
            for assigned, keys in sets.items()
            for k in keys
            for r in groups[k]["rows"]
        ],
        key=lambda r: (r["task_key"], r["task_version_id"]),
    )
    evals = {
        name: seal(
            {
                "schema": "cyber_eval_task_selection_v1",
                "selection_role": name,
                "final_test_lock_sha256": final_test["sha256"],
                "tasks": [r for r in tasks if r["split"] == name],
            }
        )
        for name in ("dev", "final_test")
    }
    corpus = seal(
        {
            "schema": "cyber_task_split_v2",
            "tasks": [
                {k: r[k] for k in ("task_key", "task_version_id", "split")}
                for r in tasks
                if r["split"] == "train"
            ],
        }
    )
    return seal(
        {
            "schema": SPLIT_SCHEMA,
            "inventory_sha256": inventory["sha256"],
            "final_test_lock_sha256": final_test["sha256"],
            "variant_seed": seed,
            "split_unit": "reviewed task family; all lineage aliases and versions grouped",
            "generalization_scope": "task-family-held-out; applications may be shared",
            "tasks": tasks,
            "training_split": corpus,
            "evaluation": evals,
            "counts": {
                name: {
                    "groups": len(keys),
                    "task_versions": sum(len(groups[k]["rows"]) for k in keys),
                }
                for name, keys in sets.items()
            },
            "inventory_audit": audit,
            "representation": {name: _distribution(groups, keys) for name, keys in sets.items()},
            "limitations": [
                "Missing taxonomy remains explicit; representative marginals do not ensure "
                "equal difficulty.",
                "Greedy grouped balancing is deterministic, not globally optimal.",
            ],
        }
    )


def _check_split(split: dict) -> None:
    check_seal(split, SPLIT_SCHEMA)
    seen, groups = set(), {}
    for row in split["tasks"]:
        key = (row["task_key"], row["task_version_id"])
        if (
            key in seen
            or row["split"] not in {"train", "dev", "final_test"}
            or groups.setdefault(row["group_id"], row["split"]) != row["split"]
        ):
            raise ValueError("study split duplicates an identity or leaks a family")
        seen.add(key)
    check_seal(split["training_split"], "cyber_task_split_v2")
    expected_train = [
        {k: r[k] for k in ("task_key", "task_version_id", "split")}
        for r in split["tasks"]
        if r["split"] == "train"
    ]
    if not expected_train or split["training_split"]["tasks"] != expected_train:
        raise ValueError("corpus-facing split must be exactly train-only and reference-free")
    for name in ("dev", "final_test"):
        evaluation = split["evaluation"][name]
        check_seal(evaluation, "cyber_eval_task_selection_v1")
        if (
            evaluation["selection_role"] != name
            or evaluation["final_test_lock_sha256"] != split["final_test_lock_sha256"]
            or evaluation["tasks"] != [r for r in split["tasks"] if r["split"] == name]
        ):
            raise ValueError("outcome evaluation manifest disagrees with the study split")


def _coverage(summary: Mapping, target_policy_sha256: str) -> None:
    fields = (
        "assistant_responses",
        "supervised_tokens",
        "submit_report_responses",
        "submit_report_tokens",
        "non_submit_tool_responses",
        "decision_responses",
        "other_responses",
        "completed_non_submit_tool_rounds",
    )
    if (
        not isinstance(summary, Mapping)
        or summary.get("status") != "certified"
        or not _sha(summary.get("receipt_sha256"))
        or summary.get("scope") != "eligible_dense_targets"
        or summary.get("target_policy_sha256") != target_policy_sha256
        or set(summary) != {*fields, "status", "receipt_sha256", "scope", "target_policy_sha256"}
    ):
        raise ValueError("missing certified source-target coverage metadata")
    if any(type(summary.get(k)) is not int or summary[k] < 0 for k in fields):
        raise ValueError("source-target coverage needs exact nonnegative counts")
    if (
        summary["assistant_responses"]
        != sum(
            summary[k]
            for k in (
                "submit_report_responses",
                "non_submit_tool_responses",
                "decision_responses",
                "other_responses",
            )
        )
        or summary["submit_report_tokens"] > summary["supervised_tokens"]
        or summary["completed_non_submit_tool_rounds"] > summary["non_submit_tool_responses"]
        or summary["supervised_tokens"] < summary["assistant_responses"]
        or summary["submit_report_tokens"] < summary["submit_report_responses"]
        or (summary["submit_report_responses"] == 0 and summary["submit_report_tokens"] != 0)
        or summary["supervised_tokens"] - summary["submit_report_tokens"]
        < summary["assistant_responses"] - summary["submit_report_responses"]
    ):
        raise ValueError("source-target category counts contradict each other")


def select_sources(
    episodes: list[dict],
    split: dict,
    *,
    models: list[str],
    target_policy_sha256: str,
    max_episodes_per_family: int,
    max_supervised_tokens_per_family: int,
    source_seed: str,
    balance_target_supervised_tokens_per_family: int | None = None,
    max_submit_token_share: float = 0.5,
    max_submit_response_share: float = 0.5,
) -> dict:
    """Train-only source selection; held-out outcomes are never inspected.

    Keep all dense targets from useful certified sources, including submission.
    Do not fix report dominance by unmasking overlap or duplicating action rows.
    Aggregate dominance blocks materialization and requests an explicit review.
    Counts follow the bound tokenizer/normalization/context eligibility policy,
    not actions that the dense compiler would later exclude.
    Tool/decision counts are evidence-backed coverage proxies, not semantic proof
    that the assistant made meaningful progress or that the model will improve.
    """
    _check_split(split)
    _rank(source_seed, "validate_source_seed")
    if (
        not _sha(target_policy_sha256)
        or not models
        or len(set(models)) != len(models)
        or any(not _label(m) for m in models)
        or any(
            type(v) is not int or v <= 0
            for v in (max_episodes_per_family, max_supervised_tokens_per_family)
        )
        or (
            balance_target_supervised_tokens_per_family is not None
            and (
                type(balance_target_supervised_tokens_per_family) is not int
                or not 0
                < balance_target_supervised_tokens_per_family
                <= max_supervised_tokens_per_family
            )
        )
        or any(
            type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
            for v in (max_submit_token_share, max_submit_response_share)
        )
    ):
        raise ValueError("explicit source models and bounded dominance thresholds required")
    assignments = {(r["task_key"], r["task_version_id"]): r for r in split["tasks"]}
    train_tasks = {key for key, r in assignments.items() if r["split"] == "train"}
    candidates, excluded, by_source, per_task, seen = [], collections.Counter(), {}, {}, set()
    for episode in episodes:
        # Identity alone is read before this boundary. No held-out outcome, model,
        # coverage or receipt may influence source selection or source-quality tuning.
        key = (episode["task_key"], episode["task_version_id"])
        if key not in train_tasks:
            excluded["outside_training_split"] += 1
            continue
        if set(episode) != {
            "schema",
            "sha256",
            "task_key",
            "task_version_id",
            "episode_id",
            "model_id",
            "source_kind",
            "validity",
            "verified_success",
            "acceptance_sha256",
            "trace_sha256",
            "normalized_record_sha256",
            "coverage",
        }:
            raise ValueError(
                "episode input must contain only certified metadata fields, "
                "never payloads or scores"
            )
        check_seal(episode, EPISODE_SCHEMA)
        sid = episode["episode_id"]
        if not _label(sid) or sid in seen:
            raise ValueError("missing or duplicate training episode identity")
        seen.add(sid)
        if episode["model_id"] not in models:
            excluded["other_model"] += 1
            continue
        if episode.get("source_kind") not in {"teacher", "self"}:
            raise ValueError("source kind must distinguish teacher and self")
        if (
            episode.get("validity") != "valid"
            or episode.get("verified_success") is not True
            or not _sha(episode.get("acceptance_sha256"))
            or not _sha(episode.get("trace_sha256"))
        ):
            excluded["not_certified_verified_success"] += 1
            continue
        if not _sha(episode.get("normalized_record_sha256")):
            excluded["missing_private_record_digest_binding"] += 1
            continue
        try:
            _coverage(episode.get("coverage", {}), target_policy_sha256)
        except ValueError:
            excluded["missing_or_invalid_target_coverage"] += 1
            continue
        c = episode["coverage"]
        if not c["supervised_tokens"] or not c["assistant_responses"]:
            excluded["no_supervised_targets"] += 1
            continue
        if not c["non_submit_tool_responses"] + c["decision_responses"]:
            excluded["no_non_submission_decisions"] += 1
            continue
        if not c["completed_non_submit_tool_rounds"]:
            excluded["no_completed_non_submission_tool_round"] += 1
            continue
        candidates.append(episode)
    kept, cap_excluded = _cap_sources(
        candidates,
        assignments,
        max_episodes_per_family,
        max_supervised_tokens_per_family,
        source_seed,
        balance_target_supervised_tokens_per_family,
    )
    excluded.update(cap_excluded)
    per_family = {}
    for episode in kept:
        key = (episode["task_key"], episode["task_version_id"])
        c = episode["coverage"]
        source = episode["source_kind"] + ":" + episode["model_id"]
        for table, bucket in (
            (by_source, source),
            (per_task, json.dumps(key, separators=(",", ":"))),
            (per_family, assignments[key]["group_id"]),
        ):
            stats = table.setdefault(
                bucket,
                {
                    "episodes": 0,
                    "assistant_responses": 0,
                    "supervised_tokens": 0,
                    "submit_report_responses": 0,
                    "submit_report_tokens": 0,
                    "non_submit_tool_responses": 0,
                    "decision_responses": 0,
                    "completed_non_submit_tool_rounds": 0,
                },
            )
            stats["episodes"] += 1
            for field in stats.keys() - {"episodes"}:
                stats[field] += c[field]
    totals = {
        field: sum(stats[field] for stats in by_source.values())
        for field in (
            "episodes",
            "assistant_responses",
            "supervised_tokens",
            "submit_report_responses",
            "submit_report_tokens",
            "non_submit_tool_responses",
            "decision_responses",
            "completed_non_submit_tool_rounds",
        )
    }
    token_share = (
        totals["submit_report_tokens"] / totals["supervised_tokens"]
        if totals["supervised_tokens"]
        else None
    )
    response_share = (
        totals["submit_report_responses"] / totals["assistant_responses"]
        if totals["assistant_responses"]
        else None
    )
    blockers = []
    if not kept:
        blockers.append("no_qualified_progress_bearing_sources")
    if token_share is not None and token_share > max_submit_token_share:
        blockers.append("final_submission_token_dominance")
    if response_share is not None and response_share > max_submit_response_share:
        blockers.append("final_submission_response_dominance")
    covered = {(r["task_key"], r["task_version_id"]) for r in kept}
    for source, stats in by_source.items():
        stats["covered_task_versions"] = len(
            {
                (r["task_key"], r["task_version_id"])
                for r in kept
                if r["source_kind"] + ":" + r["model_id"] == source
            }
        )
        stats["submission_token_share"] = stats["submit_report_tokens"] / stats["supervised_tokens"]
        stats["submission_response_share"] = (
            stats["submit_report_responses"] / stats["assistant_responses"]
        )
    family_selection = "model-diverse round robin; seeded identity order; whole episodes"
    policy = {
        "min_non_submit_decisions": 1,
        "min_completed_tool_rounds": 1,
        "max_submit_token_share": max_submit_token_share,
        "max_submit_response_share": max_submit_response_share,
        "max_episodes_per_family": max_episodes_per_family,
        "max_supervised_tokens_per_family": max_supervised_tokens_per_family,
        "source_seed": source_seed,
        "family_selection": family_selection,
    }
    if balance_target_supervised_tokens_per_family is not None:
        policy.update(
            balance_target_supervised_tokens_per_family=(
                balance_target_supervised_tokens_per_family
            ),
            family_selection=(
                "greedy nearest supervised-token target; model-diverse seeded ties; whole episodes"
            ),
        )
    return seal(
        {
            "schema": "cyber_sft_source_selection_v1",
            "study_split_sha256": split["sha256"],
            "training_split_sha256": split["training_split"]["sha256"],
            "target_policy_sha256": target_policy_sha256,
            "status": "ready" if not blockers else "blocked",
            "blockers": blockers,
            "models": sorted(models),
            "policy": policy,
            "selected_episode_ids": sorted(r["episode_id"] for r in kept),
            "selected_evidence": sorted(
                [
                    {
                        "episode_id": r["episode_id"],
                        "metadata_sha256": r["sha256"],
                        "trace_sha256": r["trace_sha256"],
                        "acceptance_sha256": r["acceptance_sha256"],
                        "normalized_record_sha256": r["normalized_record_sha256"],
                    }
                    for r in kept
                ],
                key=lambda r: r["episode_id"],
            ),
            "totals": totals,
            "excluded": dict(excluded),
            "by_source": dict(sorted(by_source.items())),
            "per_task": dict(sorted(per_task.items())),
            "per_family": dict(sorted(per_family.items())),
            "coverage": {
                "train_task_versions": len(train_tasks),
                "covered_train_task_versions": len(covered),
                "missing_train_task_versions": len(train_tasks - covered),
                "qualified_candidate_episodes_before_caps": len(candidates),
                "covered_train_families": len(per_family),
                "train_families": len({assignments[key]["group_id"] for key in train_tasks}),
            },
            "dominance": {
                "submission_token_share": token_share,
                "submission_response_share": response_share,
                "max_family_supervised_token_share": max(
                    (
                        r["supervised_tokens"] / totals["supervised_tokens"]
                        for r in per_family.values()
                    ),
                    default=None,
                ),
                "max_family_episode_share": max(
                    (r["episodes"] / totals["episodes"] for r in per_family.values()), default=None
                ),
            },
            "limitations": [
                "Certified action-category counts are coverage proxies, "
                "not semantic progress judgments.",
                "Source coverage is not a matched teacher/self comparison "
                "unless tasks and target budgets are matched.",
            ],
        }
    )


def _cap_sources(candidates, assignments, episode_cap, token_cap, seed, balance_target=None):
    """Whole-source, per-family ceilings with deterministic model diversity."""
    families = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in candidates:
        family = assignments[(row["task_key"], row["task_version_id"])]["group_id"]
        families[family][row["source_kind"] + ":" + row["model_id"]].append(row)
    selected, excluded = [], collections.Counter()
    for family, pools in sorted(families.items()):
        models = sorted(pools, key=lambda model: _rank(seed, family + ":" + model))
        for pool in pools.values():
            pool.sort(key=lambda row: _rank(seed, family + ":" + row["episode_id"]))
        if balance_target is not None:
            pool = [row for model in models for row in pools[model]]
            used_models, episodes_used, tokens_used = set(), 0, 0
            while episodes_used < episode_cap and pool:
                fitting = [
                    row
                    for row in pool
                    if tokens_used + row["coverage"]["supervised_tokens"] <= token_cap
                ]
                if not fitting:
                    break
                candidate = min(
                    fitting,
                    key=lambda row: (
                        abs(tokens_used + row["coverage"]["supervised_tokens"] - balance_target),
                        row["source_kind"] + ":" + row["model_id"] in used_models,
                        _rank(seed, family + ":" + row["episode_id"]),
                    ),
                )
                new_total = tokens_used + candidate["coverage"]["supervised_tokens"]
                if tokens_used and abs(new_total - balance_target) >= abs(
                    tokens_used - balance_target
                ):
                    break
                selected.append(candidate)
                pool.remove(candidate)
                episodes_used += 1
                tokens_used = new_total
                used_models.add(candidate["source_kind"] + ":" + candidate["model_id"])
            for row in pool:
                reason = (
                    "family_episode_cap"
                    if episodes_used == episode_cap
                    else (
                        "family_supervised_token_cap"
                        if tokens_used + row["coverage"]["supervised_tokens"] > token_cap
                        else "family_balance_target"
                    )
                )
                excluded[reason] += 1
            continue
        episodes_used, tokens_used = 0, 0
        while episodes_used < episode_cap and any(pools.values()):
            for model in models:
                if episodes_used == episode_cap:
                    break
                while pools[model]:
                    candidate = pools[model].pop(0)
                    tokens = candidate["coverage"]["supervised_tokens"]
                    if tokens_used + tokens > token_cap:
                        excluded["family_supervised_token_cap"] += 1
                        continue
                    selected.append(candidate)
                    episodes_used += 1
                    tokens_used += tokens
                    break
        excluded["family_episode_cap"] += sum(len(pool) for pool in pools.values())
    return selected, +excluded


def filter_records(records: Iterable[dict], selection: dict, split: dict) -> list[dict]:
    """Adapter for existing corpus mechanisms; preserve private records unchanged."""
    check_seal(selection, "cyber_sft_source_selection_v1")
    _check_split(split)
    if (
        selection["status"] != "ready"
        or selection["study_split_sha256"] != split["sha256"]
        or selection["training_split_sha256"] != split["training_split"]["sha256"]
    ):
        raise ValueError("source selection is blocked or belongs to another split")
    allowed = set(selection["selected_episode_ids"])
    evidence = {r["episode_id"]: r for r in selection["selected_evidence"]}
    if (
        not allowed
        or len(allowed) != len(selection["selected_episode_ids"])
        or set(evidence) != allowed
        or len(evidence) != len(selection["selected_evidence"])
    ):
        raise ValueError("selected source identities or digest bindings are incomplete")
    train = {(r["task_key"], r["task_version_id"]) for r in split["training_split"]["tasks"]}
    found, selected = set(), []
    for record in records:
        sid = record.get("record_id")
        if sid not in allowed:
            continue
        lineage = record["lineage"]
        if sid in found or (lineage["task_key"], lineage["eval_task_version_id"]) not in train:
            raise ValueError("selected private source is duplicated or outside train")
        if digest_json(record) != evidence[sid]["normalized_record_sha256"]:
            raise ValueError("selected private record differs from certified metadata")
        found.add(sid)
        selected.append(record)
    if found != allowed:
        raise ValueError("selected private sources are missing; do not publish a partial corpus")
    return sorted(selected, key=lambda r: r["record_id"])
