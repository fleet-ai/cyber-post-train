"""Create-only metadata successor for an already materialized outcome-only corpus.

This never opens Parquet, source records, held-out messages, or a remote API.
Changing an outcome protocol needs separate review; this utility is not authority.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cyber_post_train.jobs import digest

from .io import atomic_write_json, file_sha256


def _sealed(value: dict, schema: str, expected: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256") != expected
        or expected != "sha256:" + digest({k: v for k, v in value.items() if k != "sha256"})
    ):
        raise ValueError("source schema or exact digest mismatch")


def successor(manifest: dict, protocol: dict, *, source_sha256: str, protocol_sha256: str) -> dict:
    """Change only the protocol pointer and seal; preserve all data provenance."""
    _sealed(manifest, "cyber_dense_sft_corpus_v1", source_sha256)
    _sealed(protocol, "cyber_fleet_dev_outcome_protocol_v2", protocol_sha256)
    if manifest.get("validation_mode") != "task_outcomes_only" or set(
        manifest.get("files", {})
    ) != {"train"}:
        raise ValueError("only an existing train-only outcome corpus may be rebound")
    if manifest.get("fleet_dev_protocol_sha256") == protocol_sha256:
        raise ValueError("protocol is already bound; no new artifact is needed")
    if not isinstance(manifest.get("source_selection"), dict) or not isinstance(
        manifest.get("builder_sha256"), dict
    ):
        raise ValueError("explicit source-selection and builder provenance required")
    result = copy.deepcopy(manifest)
    result.pop("sha256")
    result["fleet_dev_protocol_sha256"] = protocol_sha256
    return {**result, "sha256": "sha256:" + digest(result)}


def publish(
    source: Path,
    protocol_path: Path,
    output: Path,
    *,
    source_sha256: str,
    protocol_sha256: str,
) -> dict:
    """Publish into a new private directory; existing/ambiguous output is a hold."""
    if source.is_symlink() or protocol_path.is_symlink():
        raise ValueError("metadata inputs must not be symlinks")
    source_bytes, protocol_bytes = source.read_bytes(), protocol_path.read_bytes()
    manifest, protocol = json.loads(source_bytes), json.loads(protocol_bytes)
    result = successor(
        manifest, protocol, source_sha256=source_sha256, protocol_sha256=protocol_sha256
    )
    # Reserve the complete publication once, before writing any of its files.
    # An interrupted publication is not an invitation to replay into this path.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_write_json(output / "manifest.json", result, private=True)
    if source.read_bytes() != source_bytes or protocol_path.read_bytes() != protocol_bytes:
        raise ValueError("metadata source changed during publication; output remains unaccepted")
    proof = {
        "schema": "cyber_sft_protocol_manifest_successor_receipt_v1",
        "classification": "local_metadata_only_unstaged_no_launch",
        "source_manifest_sha256": source_sha256,
        "source_manifest_file_sha256": file_sha256(source),
        "successor_manifest_sha256": result["sha256"],
        "successor_manifest_file_sha256": file_sha256(output / "manifest.json"),
        "previous_protocol_sha256": manifest["fleet_dev_protocol_sha256"],
        "protocol_sha256": protocol_sha256,
        "protocol_file_sha256": file_sha256(protocol_path),
        "changed_manifest_fields": ["fleet_dev_protocol_sha256", "sha256"],
        "preserved": {
            "source_sha256": result["source_sha256"],
            "split_sha256": result["split_sha256"],
            "tokenizer_identity_sha256": "sha256:" + digest(result["tokenizer"]),
            "train_file_sha256": result["files"]["train"]["sha256"],
            "train_rows": result["files"]["train"]["rows"],
            "supervised_tokens": result["files"]["train"]["supervised_tokens"],
            "builder_sha256": result["builder_sha256"],
            "source_selection_sha256": result["source_selection"]["source_selection_sha256"],
            "target_policy_sha256": result["source_selection"]["target_policy_sha256"],
        },
        "parquet_or_source_records_read": False,
        "remote_operations_performed": False,
        "launch_authorized": False,
    }
    proof["sha256"] = "sha256:" + digest(proof)
    atomic_write_json(output / "SUCCESSOR.json", proof, private=True)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        proof = publish(
            args.source,
            args.protocol,
            args.output,
            source_sha256=args.source_sha256,
            protocol_sha256=args.protocol_sha256,
        )
    except Exception as exc:
        print(json.dumps({"status": "stopped_no_retry", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
