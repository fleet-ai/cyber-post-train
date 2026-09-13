"""Current-source, offline Qwen3.8 direct self-trace collection gate.

V3 preserves the immutable V2 scientific treatment and task roster, rebinds
the collector source closure after the recorder changed, and exposes strict
validators for the two still-missing external receipts.  It is not a launcher
and performs no cluster, API, registry, task, model, or scoring operation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import self_trace_collection as v1
from . import self_trace_collection_v2 as v2
from .io import digest_json, file_sha256

REQUEST_SCHEMA = "cyber_qwen_direct_self_trace_collection_request_v3"
PARITY_SCHEMA = "cyber_qwen_direct_recorder_dense_parity_v3"
FIXTURE_SCHEMA = "cyber_qwen_direct_recorder_dense_fixture_v3"
PREDECESSOR_SHA256 = "sha256:2c3040c28f6fc474908841e3582fb47cb9a69a5ce6993f2f9bc33ad702c96c22"
PREDECESSOR_FILE_SHA256 = "sha256:7e9f96bd8c76df25bf78545cd941233282e8e5dfe2da2e386df69ec4b4bc5a1b"

SOURCE_CLOSURE_PATHS = {
    **v2.SOURCE_CLOSURE_PATHS,
    "self_trace_collection_v3.py": Path(__file__),
}


class CollectionError(v2.CollectionError):
    """A fixed, payload-free V3 collection qualification rejection."""


def _sealed(value: dict, schema: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256")
        != digest_json({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise CollectionError("sealed v3 collection metadata differs")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise CollectionError("bound v3 collection JSON is unreadable") from None


def _bound(reference: object, relative_to: Path, *, schema: str) -> tuple[Path, dict]:
    try:
        path, value = v1._bound(reference, relative_to, sealed=True)
    except Exception:
        raise CollectionError("immutable v3 artifact binding differs") from None
    if not isinstance(value, dict):
        raise CollectionError("bound v3 artifact is not an object")
    _sealed(value, schema)
    return path, value


def _source_closure(value: object, relative_to: Path) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SOURCE_CLOSURE_PATHS):
        raise CollectionError("v3 collector source closure fields differ")
    result = {}
    for name, reference in value.items():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise CollectionError("v3 collector source binding fields differ")
        raw = reference.get("path")
        path = Path(raw) if isinstance(raw, str) else Path()
        path = path if path.is_absolute() else relative_to / path
        if (
            not isinstance(raw, str)
            or not raw
            or path.is_symlink()
            or not path.is_file()
            or path.name != name
            or not v2._sha(reference.get("sha256"))
            or file_sha256(path) != reference["sha256"]
        ):
            raise CollectionError("v3 collector source file digest differs")
        result[name] = reference["sha256"]
    return result


def _frozen_inputs(predecessor: dict, relative_to: Path) -> dict[str, Any]:
    """Read the frozen roster without pretending its old source closure is current."""
    if (
        predecessor.get("sha256") != PREDECESSOR_SHA256
        or predecessor.get("status") != "blocked_external_bindings"
        or predecessor.get("runtime", {}).get("collector_qualification") is not None
        or predecessor.get("runtime", {}).get("direct_route_certificate") is not None
    ):
        raise CollectionError("immutable v2 predecessor differs")
    base_path, base = _bound(predecessor["base_request"], relative_to, schema=v1.REQUEST_SCHEMA)
    _, study = _bound(predecessor["study"], relative_to, schema=v2.STUDY_SCHEMA)
    try:
        v2._validate_study(
            study,
            base_plan_sha256=base["producer_plan"]["document_sha256"],
            base_request_sha256=base["sha256"],
        )
        _, plan = v1._bound(base["producer_plan"], base_path.parent, sealed=True)
        _, eligible = v1._bound(base["selection"]["eligible_inventory"], base_path.parent)
        splits = {
            name: v1._bound(reference, base_path.parent, sealed=True)[1]
            for name, reference in base["selection"]["representative_splits"].items()
        }
        tasks = v1._train_rows(splits["a"]) | v1._train_rows(splits["b"])
        rows = eligible["task_versions"]
        inventory = {(row["task_key"], row["task_version_id"]): row for row in rows}
        selection = base["selection"]
        roster = v1.derive_roster(
            tasks,
            attempts_per_task_version=selection["attempts_per_task_version"],
            seed=selection["attempt_seed"],
        )
        _, catalog = v1._tool_catalog(base["interface"]["tool_catalog"], base_path.parent)
    except Exception:
        raise CollectionError("frozen v2 roster bindings differ") from None
    if (
        len(inventory) != len(rows)
        or not tasks.issubset(inventory)
        or any(not v1._valid_runtime_inventory(inventory[key]) for key in tasks)
        or selection.get("task_versions") != len(tasks)
        or selection.get("maximum_sessions") != len(roster)
        or selection.get("attempt_roster_sha256") != digest_json(roster)
        or predecessor.get("model") != base.get("model")
        or predecessor.get("interface", {}).get("tool_catalog")
        != base.get("interface", {}).get("tool_catalog")
    ):
        raise CollectionError("frozen v2 task roster differs")
    return {
        "base_request": base,
        "plan": plan,
        "study": study,
        "tasks": tasks,
        "inventory": inventory,
        "roster": roster,
        "tool_catalog": catalog,
    }


def _fixture_sha256(parity: dict, schema: str) -> str:
    return digest_json(
        {
            "schema": schema,
            "prompt_policy_sha256": v2.PROMPT_POLICY_SHA256,
            "native": parity["native"],
            "dense": parity["dense"],
            "dense_policy": parity["dense_policy"],
            "first_observation": parity["first_observation"],
        }
    )


def _validate_parity(
    parity: dict,
    *,
    model: dict,
    source_closure: dict[str, str],
    qualification: dict,
) -> None:
    _sealed(parity, PARITY_SCHEMA)
    if parity.get("source_closure_sha256") != digest_json(source_closure) or parity.get(
        "fixture_sha256"
    ) != _fixture_sha256(parity, FIXTURE_SCHEMA):
        raise CollectionError("v3 synthetic parity binding differs")
    legacy_closure = {name: source_closure[name] for name in v2.SOURCE_CLOSURE_PATHS}
    legacy = json.loads(json.dumps(parity))
    legacy["schema"] = v2.PARITY_SCHEMA
    legacy["source_closure_sha256"] = digest_json(legacy_closure)
    legacy["fixture_sha256"] = _fixture_sha256(
        legacy, "cyber_qwen_direct_recorder_dense_fixture_v2"
    )
    legacy["sha256"] = digest_json({key: item for key, item in legacy.items() if key != "sha256"})
    try:
        v2._validate_parity_receipt(
            legacy,
            model=model,
            source_closure=legacy_closure,
            qualification=qualification,
        )
    except Exception:
        raise CollectionError("v3 synthetic parity receipt fields differ") from None


def validate_collector_qualification(
    qualification: dict, *, model: dict, source_closure: dict[str, str]
) -> str:
    """Validate a future pull-qualified zero-GPU collector receipt."""
    try:
        return v2._validate_collector_qualification(
            qualification, model=model, source_closure=source_closure
        )
    except Exception:
        raise CollectionError("collector qualification artifact differs") from None


def validate_route_certificate(
    certificate: dict,
    *,
    model: dict,
    collector_qualification_sha256: str,
    collector_image: str,
) -> None:
    """Validate a future fresh direct-base-route receipt cross-bound to the collector."""
    try:
        v2._validate_route_certificate(
            certificate,
            model=model,
            collector_qualification_sha256=collector_qualification_sha256,
            collector_image=collector_image,
        )
    except Exception:
        raise CollectionError("direct base-route certificate artifact differs") from None


def validate_request(
    request: dict, *, relative_to: Path, require_ready: bool = False
) -> dict[str, Any]:
    """Validate V3 and return content-free metadata; never launch collection."""
    _sealed(request, REQUEST_SCHEMA)
    if set(request) != {
        "schema",
        "status",
        "predecessor",
        "runtime",
        "source_closure",
        "execution",
        "sha256",
    }:
        raise CollectionError("v3 collection request fields differ")
    predecessor_ref = request["predecessor"]
    if (
        not isinstance(predecessor_ref, dict)
        or predecessor_ref.get("file_sha256") != PREDECESSOR_FILE_SHA256
        or predecessor_ref.get("document_sha256") != PREDECESSOR_SHA256
    ):
        raise CollectionError("v3 predecessor binding differs")
    predecessor_path, predecessor = _bound(predecessor_ref, relative_to, schema=v2.REQUEST_SCHEMA)
    frozen = _frozen_inputs(predecessor, predecessor_path.parent)
    model = predecessor["model"]
    source_closure = _source_closure(request["source_closure"], relative_to)
    runtime = request.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "collector_qualification",
        "direct_route_certificate",
        "synthetic_parity_receipt",
    }:
        raise CollectionError("v3 runtime artifact fields differ")
    _, parity = _bound(runtime["synthetic_parity_receipt"], relative_to, schema=PARITY_SCHEMA)
    _validate_parity(
        parity,
        model=model,
        source_closure=source_closure,
        qualification=frozen["base_request"]["qualification"],
    )

    blockers = []
    collector = collector_image = None
    if runtime["collector_qualification"] is None:
        blockers.append("missing_immutable_collector_qualification")
    else:
        _, collector = _bound(
            runtime["collector_qualification"],
            relative_to,
            schema=v2.COLLECTOR_QUALIFICATION_SCHEMA,
        )
        collector_image = validate_collector_qualification(
            collector, model=model, source_closure=source_closure
        )

    route = None
    if runtime["direct_route_certificate"] is None:
        blockers.append("missing_exact_direct_route_certificate")
    elif collector is None or collector_image is None:
        raise CollectionError("route certificate requires collector qualification")
    else:
        _, route = _bound(
            runtime["direct_route_certificate"],
            relative_to,
            schema=v2.ROUTE_CERTIFICATE_SCHEMA,
        )
        validate_route_certificate(
            route,
            model=model,
            collector_qualification_sha256=collector["sha256"],
            collector_image=collector_image,
        )

    if request.get("execution") != {
        "kind": "offline_qualification_not_a_launcher",
        "launchable": False,
        "cluster_or_api_mutations_performed": False,
        "job_submission_performed": False,
        "credentials_required": False,
    }:
        raise CollectionError("v3 collection execution boundary differs")
    expected_status = "qualified" if not blockers else "blocked_external_bindings"
    if request.get("status") != expected_status:
        raise CollectionError("v3 collection status differs from its blockers")
    if require_ready and blockers:
        raise CollectionError("v3 collection request is not ready")
    return {
        **frozen,
        "request": request,
        "predecessor": predecessor,
        "source_closure": source_closure,
        "parity": parity,
        "collector_qualification": collector,
        "collector_image": collector_image,
        "route_certificate": route,
        "blockers": blockers,
    }


def recorder_dense_parity(
    sample: object,
    conversation: dict,
    *,
    dense_reference_tokens: list[int],
    dense_reference_loss_mask: list[int],
    model: dict,
    interface: dict,
    source_closure: dict[str, str],
    max_length: int,
    context_tokens: int,
) -> dict:
    """Run the proven V2 fixture logic and bind it to the complete V3 closure."""
    if set(source_closure) != set(SOURCE_CLOSURE_PATHS):
        raise CollectionError("v3 synthetic parity source closure differs")
    legacy_closure = {name: source_closure[name] for name in v2.SOURCE_CLOSURE_PATHS}
    try:
        receipt = v2.recorder_dense_parity(
            sample,
            conversation,
            dense_reference_tokens=dense_reference_tokens,
            dense_reference_loss_mask=dense_reference_loss_mask,
            model=model,
            interface=interface,
            source_closure=legacy_closure,
            max_length=max_length,
            context_tokens=context_tokens,
        )
    except Exception:
        raise CollectionError("v3 synthetic recorder and dense adapter differ") from None
    receipt["schema"] = PARITY_SCHEMA
    receipt["source_closure_sha256"] = digest_json(source_closure)
    receipt["fixture_sha256"] = _fixture_sha256(receipt, FIXTURE_SCHEMA)
    receipt["sha256"] = digest_json({key: item for key, item in receipt.items() if key != "sha256"})
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Validate one V3 request without loading private task or trace content."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = _read_json(args.request)
        if not isinstance(request, dict):
            raise CollectionError("v3 collection request is not an object")
        validated = validate_request(
            request, relative_to=args.request.parent, require_ready=args.require_ready
        )
    except Exception:
        print(json.dumps({"status": "rejected", "reason": "collection_request_not_qualified"}))
        return 2
    print(
        json.dumps(
            {
                "status": request["status"],
                "request_sha256": request["sha256"],
                "task_versions": len(validated["tasks"]),
                "maximum_sessions": len(validated["roster"]),
                "blockers": validated["blockers"],
                "launchable": False,
                "cluster_or_api_mutations_performed": False,
                "job_submission_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
