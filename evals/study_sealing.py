"""Fail-closed launch and result sealing for the Qwen outcome study.

This module reads only repository protocols and sanitized completion receipts.
It never opens raw evaluation results and has no network or launch operation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from evals.fleet import dev_outcome_protocol as fleet
from evals.webexploitbench import study_parent_protocol as web
from training.io import canonical_json, digest_json, file_sha256

CHILD_SCHEMA = "cyber_study_evaluation_child_v1"
COMPLETION_SCHEMA = "cyber_private_evaluation_completion_v1"
BLINDED_SCHEMA = "cyber_blinded_evaluation_completion_v1"
FREEZE_SCHEMA = "cyber_study_outcome_access_freeze_v1"
ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
TAGGED_IMAGE = re.compile(r"^[A-Za-z0-9./_-]+:[A-Za-z0-9._-]+$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")

RESULT_POLICY = {
    "raw_outcomes": "private_only",
    "wandb_export_allowed": False,
    "blinded_receipt_only_before_outcome_access_gate": True,
    "training_data_eligible": False,
}
MATCHED_RUNTIME_FIELDS = (
    "endpoint_origin",
    "serving_block_kind",
    "engine",
    "precision",
    "quantization",
    "tensor_parallel_size",
    "data_parallel_size",
    "context_length",
    "kv_cache_dtype",
    "reasoning_parser",
    "tool_call_parser",
    "serving_image_digest",
    "normalized_server_arguments_sha256",
    "tokenizer_manifest_sha256",
    "chat_template_sha256",
    "agent_image_digest",
    "proxy_image_digest",
    "live_pair_parity_receipt_sha256",
)
COMPLETION_FIELDS = {
    "schema",
    "child_sha256",
    "blinded_evaluation_id",
    "terminal_classification",
    "terminal_evidence_sha256",
    "raw_result_manifest_sha256",
    "private_results_root",
    "resources_released",
    "wandb_exported",
    "sha256",
}
BLINDED_FIELDS = {
    "schema",
    "blinded_evaluation_id",
    "surface",
    "parent_protocol_sha256",
    "sealed_terminal",
    "private_completion_receipt_sha256",
    "raw_result_manifest_sha256",
    "outcome_or_validity_exposed",
    "wandb_exported",
    "outcome_access_gate",
    "sha256",
}
FREEZE_FIELDS = {
    "schema",
    "study_id",
    "surface",
    "parent_protocol_sha256",
    "expected_blinded_evaluation_ids",
    "gate",
    "selection_rule_sha256",
    "selected_checkpoint_seal_sha256",
    "frozen_before_outcome_access",
    "sha256",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def _check_seal(value: dict[str, Any], schema: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != digest_json(unsigned):
        raise ValueError(f"invalid {schema} seal")


def _repo_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("protocol references must be repository-relative")
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("protocol reference escapes the repository")
    return path


def _reference(value: Any, root: Path) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "file_sha256"}:
        raise ValueError("an exact path, logical digest and file digest are required")
    path = _repo_path(value["path"], root)
    loaded = _read(path)
    if value["sha256"] != loaded.get("sha256") or value["file_sha256"] != file_sha256(path):
        raise ValueError("referenced protocol identity changed")
    return loaded, path


def _surface_parent(child: dict[str, Any], root: Path) -> tuple[str, dict[str, Any], Path]:
    parent, parent_path = _reference(child.get("parent_protocol"), root)
    surface = child.get("surface")
    if parent.get("schema") in fleet.PROTOCOL_SCHEMAS and surface == "fleet_dev":
        task_set, _ = _reference(
            {
                "path": parent["task_set"]["path"],
                "sha256": parent["task_set"]["sha256"],
                "file_sha256": parent["task_set"]["file_sha256"],
            },
            root,
        )
        fleet.validate_protocol(parent, task_set)
    elif parent.get("schema") == fleet.FINAL_PROTOCOL_SCHEMA and surface == "fleet_final":
        task_set, _ = _reference(
            {
                "path": parent["task_set"]["path"],
                "sha256": parent["task_set"]["sha256"],
                "file_sha256": parent["task_set"]["file_sha256"],
            },
            root,
        )
        fleet.validate_final_protocol(parent, task_set)
    elif (
        parent.get("schema") in {web.SCHEMA, web.POST_TRAINING_SCHEMA}
        and surface == "webexploitbench_tensorlake"
    ):
        web.validate_parent(parent)
    else:
        raise ValueError("child surface and parent protocol differ")
    return surface, parent, parent_path


def _exact_fields(
    value: Any,
    fields: set[str] | tuple[str, ...],
    *,
    allow_null: bool,
    allow_tagged_images: bool = False,
) -> bool:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("binding fields differ from the frozen parent")
    complete = True
    for key, item in value.items():
        if item is None:
            complete = False
            if not allow_null:
                raise ValueError(f"exact binding {key} is absent")
        elif key.endswith("sha256"):
            if not isinstance(item, str) or SHA256.fullmatch(item) is None:
                raise ValueError(f"exact binding {key} is not a SHA-256")
        elif key == "checkpoint_step":
            if type(item) is not int or item < 1:
                raise ValueError("checkpoint step must be a positive integer")
        elif key == "sandbox_resources":
            if (
                not isinstance(item, dict)
                or set(item) != web.SANDBOX_RESOURCE_KEYS
                or any(type(number) is not int or number <= 0 for number in item.values())
            ):
                raise ValueError("sandbox resources must be exact positive integers")
        elif key == "sandbox_timeout_secs":
            if type(item) is not int or item < 60:
                raise ValueError("sandbox timeout must be at least 60 seconds")
        elif key in {"tensor_parallel_size", "data_parallel_size", "context_length"}:
            if type(item) is not int or item < 1:
                raise ValueError(f"matched runtime {key} must be a positive integer")
        elif key == "evaluator_image":
            if not isinstance(item, str) or IMAGE.fullmatch(item) is None:
                raise ValueError(f"{key} must be an immutable image digest")
        elif key in {"agent_image", "netproxy_image"}:
            if not isinstance(item, str) or (
                IMAGE.fullmatch(item) is None
                and (not allow_tagged_images or TAGGED_IMAGE.fullmatch(item) is None)
            ):
                raise ValueError(f"{key} must be an exact image reference")
        elif not isinstance(item, str) or not item:
            raise ValueError(f"exact binding {key} must be a nonempty immutable identity")
    return complete


def _runtime(value: Any, *, allow_null: bool) -> bool:
    complete = _exact_fields(value, MATCHED_RUNTIME_FIELDS, allow_null=allow_null)
    if not complete:
        return False
    if value["endpoint_origin"] != "https://inference.flt.build":
        raise ValueError("model traffic must use the frozen Fleet inference gateway")
    for key in ("tensor_parallel_size", "data_parallel_size", "context_length"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"matched runtime {key} must be a positive integer")
    return True


def _private_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("raw evaluation paths must be absolute and private")
    path = Path(value).resolve()
    repository = root.resolve()
    private = (repository / "data/private").resolve()
    if (
        (path == repository or repository in path.parents)
        and path != private
        and private not in path.parents
    ):
        raise ValueError("repository-local outcomes belong only under data/private")
    return path


def _post_training_child_reference(
    value: Any,
    parent: dict[str, Any],
    parent_path: Path,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "child_sha256",
        "file_sha256",
    }:
        raise ValueError("an exact private post-training child reference is required")
    if Path(value["path"]).is_symlink():
        raise ValueError("post-training child must be a regular nonsymlink file")
    path = _private_path(value["path"], root)
    if path.is_symlink() or not path.is_file():
        raise ValueError("post-training child must be a regular nonsymlink file")
    info = path.stat()
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("post-training child must be a private file")
    loaded = _read(path)
    if value["child_sha256"] != loaded.get("child_sha256") or value["file_sha256"] != file_sha256(
        path
    ):
        raise ValueError("referenced post-training child identity changed")
    web.validate_child(loaded, parent, parent_path)
    return loaded


def validate_child(child: dict[str, Any], *, root: Path = ROOT) -> bool:
    """Validate one child and return whether every launch binding is complete."""
    _check_seal(child, CHILD_SCHEMA)
    surface, parent, parent_path = _surface_parent(child, root)
    expected_keys = {
        "schema",
        "study_id",
        "arm_id",
        "blinded_evaluation_id",
        "surface",
        "parent_protocol",
        "base_control",
        "bindings",
        "execution",
        "result_policy",
        "launchable",
        "sha256",
    }
    if parent.get("schema") == web.POST_TRAINING_SCHEMA:
        expected_keys.remove("base_control")
        expected_keys.add("post_training_child")
    if set(child) != expected_keys:
        raise ValueError("study child has unknown or missing fields")
    for key in ("study_id", "arm_id", "blinded_evaluation_id"):
        if not isinstance(child[key], str) or NAME.fullmatch(child[key]) is None:
            raise ValueError(f"invalid {key}")
    bindings = child.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("child bindings must be an object")
    allow_null = child.get("launchable") is False
    post_training_child = None

    if surface == "fleet_dev":
        base, _ = _reference(child.get("base_control"), root)
        task_set = _read(_repo_path(parent["task_set"]["path"], root))
        fleet.validate_base_control(base, parent, task_set)
        if set(bindings) != {"post_sft", "base", "matched_runtime"}:
            raise ValueError("Fleet-dev child requires base, post-SFT and matched runtime")
        complete = _exact_fields(
            bindings["post_sft"], fleet.UNBOUND_CHECKPOINT_FIELDS, allow_null=allow_null
        )
        complete &= _exact_fields(
            bindings["base"], fleet.UNBOUND_BASE_SERVING_FIELDS, allow_null=allow_null
        )
        if complete:
            for key in ("weights_manifest_sha256",):
                if bindings["base"][key] == bindings["post_sft"][key]:
                    raise ValueError("base and post-SFT weights must be distinct")
            for key in (
                "tokenizer_manifest_sha256",
                "chat_template_sha256",
                "agent_image_digest",
                "proxy_image_digest",
            ):
                if bindings["base"][key] != bindings["post_sft"][key]:
                    raise ValueError(f"base/post {key} differs")
    elif surface == "fleet_final":
        if child.get("base_control") is not None:
            raise ValueError("Fleet-final parent already owns both arm templates")
        if set(bindings) != {"arms", "matched_runtime"} or set(bindings["arms"]) != {
            "base",
            "post_sft",
        }:
            raise ValueError("Fleet-final child requires exactly two arms")
        arm_completeness = [
            _exact_fields(arm, fleet.UNBOUND_CHECKPOINT_FIELDS, allow_null=allow_null)
            for arm in bindings["arms"].values()
        ]
        complete = all(arm_completeness)
        if complete:
            base, post = bindings["arms"]["base"], bindings["arms"]["post_sft"]
            if base["weights_manifest_sha256"] == post["weights_manifest_sha256"]:
                raise ValueError("base and post-SFT weights must be distinct")
            for key in (
                "tokenizer_manifest_sha256",
                "chat_template_sha256",
                "agent_image_digest",
                "proxy_image_digest",
                "live_parity_receipt_sha256",
            ):
                if base[key] != post[key]:
                    raise ValueError(f"final base/post {key} differs")
    elif parent.get("schema") == web.SCHEMA:
        if child.get("base_control") is not None:
            raise ValueError("WebExploitBench child does not use a Fleet base-control file")
        if set(bindings) != {
            "checkpoint_and_serving",
            "tensorlake_qualification",
            "matched_runtime",
        }:
            raise ValueError("WebExploitBench child bindings differ from its parent")
        complete = _exact_fields(
            bindings["checkpoint_and_serving"], web.CHECKPOINT_FIELDS, allow_null=allow_null
        )
        complete &= _exact_fields(
            bindings["tensorlake_qualification"], web.TENSORLAKE_FIELDS, allow_null=allow_null
        )
    else:
        post_training_child = _post_training_child_reference(
            child.get("post_training_child"), parent, parent_path, root
        )
        if set(bindings) != {
            "checkpoint_and_serving",
            "tensorlake_qualification",
            "matched_runtime",
        }:
            raise ValueError("WebExploitBench child bindings differ from its parent")
        complete = _exact_fields(
            bindings["checkpoint_and_serving"], web.CHECKPOINT_FIELDS, allow_null=allow_null
        )
        complete &= _exact_fields(
            bindings["tensorlake_qualification"],
            web.TENSORLAKE_FIELDS,
            allow_null=allow_null,
            allow_tagged_images=True,
        )
        if (
            bindings["checkpoint_and_serving"] != post_training_child["checkpoint_and_serving"]
            or bindings["tensorlake_qualification"]
            != post_training_child["tensorlake_qualification"]
        ):
            raise ValueError("study and post-training child bindings differ")
        if child["arm_id"] != post_training_child["checkpoint_and_serving"]["training_arm_id"]:
            raise ValueError("study and post-training child arm identities differ")

    complete &= _runtime(bindings["matched_runtime"], allow_null=allow_null)
    execution = child.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "campaign_id",
        "private_results_root",
        "claim_journal",
    }:
        raise ValueError("child execution identity is incomplete")
    if (
        not isinstance(execution["campaign_id"], str)
        or NAME.fullmatch(execution["campaign_id"]) is None
    ):
        raise ValueError("invalid campaign id")
    if (
        post_training_child is not None
        and execution["campaign_id"] != post_training_child["campaign"]["id"]
    ):
        raise ValueError("study and post-training child campaign identities differ")
    if post_training_child is not None:
        if child.get("launchable") is not False:
            raise ValueError(
                "generic post-training study child must remain non-launchable until exact "
                "paired-plan, preview, and schedule evidence is bound"
            )
        # The repository has no authoritative producers yet for two arm-specific
        # snapshot qualifications, a fresh pair-level duplicate claim, a runtime
        # projection derived from live serving evidence, or an executable paired
        # Tensorlake scheduler.  Caller-authored digests or detached receipts may
        # not stand in for those gates.
        complete = False
    result_root = _private_path(execution["private_results_root"], root)
    journal = _private_path(execution["claim_journal"], root)
    if result_root != journal and result_root not in journal.parents:
        raise ValueError("claim journal must live inside the private result root")
    if child.get("result_policy") != RESULT_POLICY:
        raise ValueError("raw outcomes must remain private and outside W&B")
    if child.get("launchable") is not complete:
        raise ValueError("launchable must exactly reflect binding completeness")
    if complete:
        _private_directory(result_root)
    return complete


def require_launchable_child(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    child = _read(path)
    if not validate_child(child, root=root):
        raise ValueError("study child remains non-launchable")
    return child


def _private_directory(path: Path) -> None:
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("raw result root must be an existing private directory")


def blinded_completion(
    child: dict[str, Any], completion: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    """Project a sanitized completion receipt without opening raw outcomes."""
    if not validate_child(child, root=root):
        raise ValueError("cannot seal results for a non-launchable child")
    _check_seal(completion, COMPLETION_SCHEMA)
    if set(completion) != COMPLETION_FIELDS:
        raise ValueError("private completion contains outcome-bearing or unknown fields")
    if (
        completion["child_sha256"] != child["sha256"]
        or completion["blinded_evaluation_id"] != child["blinded_evaluation_id"]
        or completion["private_results_root"] != child["execution"]["private_results_root"]
        or completion["terminal_classification"]
        not in {"valid_outcome_set", "infrastructure_invalid"}
        or completion["resources_released"] is not True
        or completion["wandb_exported"] is not False
    ):
        raise ValueError("private completion does not satisfy the child result contract")
    for key in ("terminal_evidence_sha256", "raw_result_manifest_sha256"):
        if not isinstance(completion[key], str) or SHA256.fullmatch(completion[key]) is None:
            raise ValueError("private completion lacks an exact artifact digest")
    _private_directory(_private_path(completion["private_results_root"], root))
    gate = (
        "arm_set_and_selection_rule_frozen"
        if child["surface"] == "fleet_dev"
        else "selected_checkpoint_frozen"
    )
    return {
        "schema": BLINDED_SCHEMA,
        "blinded_evaluation_id": child["blinded_evaluation_id"],
        "surface": child["surface"],
        "parent_protocol_sha256": child["parent_protocol"]["sha256"],
        "sealed_terminal": True,
        "private_completion_receipt_sha256": completion["sha256"],
        "raw_result_manifest_sha256": completion["raw_result_manifest_sha256"],
        "outcome_or_validity_exposed": False,
        "wandb_exported": False,
        "outcome_access_gate": gate,
    }


def write_blinded_completion(
    child_path: Path, completion_path: Path, output_path: Path, *, root: Path = ROOT
) -> dict[str, Any]:
    child = require_launchable_child(child_path, root=root)
    value = blinded_completion(child, _read(completion_path), root=root)
    sealed = {**value, "sha256": digest_json(value)}
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(sealed, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sealed


def validate_outcome_access(blinded: dict[str, Any], freeze: dict[str, Any]) -> None:
    """Validate the predeclared gate before any private outcome is opened."""
    _check_seal(blinded, BLINDED_SCHEMA)
    _check_seal(freeze, FREEZE_SCHEMA)
    if set(blinded) != BLINDED_FIELDS or set(freeze) != FREEZE_FIELDS:
        raise ValueError("outcome-access receipts contain unknown or outcome-bearing fields")
    if blinded.get("surface") not in {
        "fleet_dev",
        "fleet_final",
        "webexploitbench_tensorlake",
    }:
        raise ValueError("unknown blinded evaluation surface")
    expected_gate = (
        "arm_set_and_selection_rule_frozen"
        if blinded["surface"] == "fleet_dev"
        else "selected_checkpoint_frozen"
    )
    for key in (
        "parent_protocol_sha256",
        "private_completion_receipt_sha256",
        "raw_result_manifest_sha256",
    ):
        if not isinstance(blinded.get(key), str) or SHA256.fullmatch(blinded[key]) is None:
            raise ValueError("blinded completion lacks an exact artifact digest")
    expected_ids = freeze.get("expected_blinded_evaluation_ids")
    if (
        blinded.get("sealed_terminal") is not True
        or blinded.get("outcome_or_validity_exposed") is not False
        or blinded.get("wandb_exported") is not False
        or blinded.get("outcome_access_gate") != expected_gate
        or not isinstance(blinded.get("blinded_evaluation_id"), str)
        or NAME.fullmatch(blinded["blinded_evaluation_id"]) is None
        or not isinstance(freeze.get("study_id"), str)
        or NAME.fullmatch(freeze["study_id"]) is None
        or freeze.get("gate") != expected_gate
        or freeze.get("surface") != blinded["surface"]
        or freeze.get("parent_protocol_sha256") != blinded["parent_protocol_sha256"]
        or freeze.get("frozen_before_outcome_access") is not True
        or blinded["blinded_evaluation_id"]
        not in (expected_ids if isinstance(expected_ids, list) else [])
        or not isinstance(expected_ids, list)
        or not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or any(not isinstance(item, str) or NAME.fullmatch(item) is None for item in expected_ids)
        or not isinstance(freeze.get("selection_rule_sha256"), str)
        or SHA256.fullmatch(freeze["selection_rule_sha256"]) is None
    ):
        raise ValueError("study selection was not frozen before outcome access")
    checkpoint = freeze.get("selected_checkpoint_seal_sha256")
    if expected_gate == "selected_checkpoint_frozen":
        if not isinstance(checkpoint, str) or SHA256.fullmatch(checkpoint) is None:
            raise ValueError("selected checkpoint was not frozen")
    elif checkpoint is not None:
        raise ValueError("Fleet-dev access freeze must precede checkpoint selection")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-child")
    check.add_argument("child", type=Path)
    seal = commands.add_parser("seal-completion")
    seal.add_argument("--child", type=Path, required=True)
    seal.add_argument("--completion", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    access = commands.add_parser("check-outcome-access")
    access.add_argument("--receipt", type=Path, required=True)
    access.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check-child":
        child = _read(args.child)
        print(canonical_json({"launchable": validate_child(child), "sha256": child["sha256"]}))
    elif args.command == "seal-completion":
        value = write_blinded_completion(args.child, args.completion, args.output)
        print(canonical_json({"blinded": True, "sha256": value["sha256"]}))
    else:
        validate_outcome_access(_read(args.receipt), _read(args.freeze))
        print(canonical_json({"outcome_access": "authorized_by_frozen_gate"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
