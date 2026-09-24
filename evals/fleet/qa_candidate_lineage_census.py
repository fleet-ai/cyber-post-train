"""Capture content-free atom lineage for an exact Fleet task-version roster."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import task_quality_qualification as qualification
from training import task_family_split
from training.shared_atom_lineage import atom_artifact_key

SCHEMA = "fleet_blackbox_qa_candidate_live_lineage_census_v1"
CANDIDATE_SCHEMA = "fleet_current_blackbox_qa_candidates_v1"
DEFAULT_CANDIDATES = "configs/data/fleet-blackbox-qa-candidates-20260924-v1.json"
EXPECTED_CANDIDATE_SHA256 = (
    "sha256:4f61fd78b4d92c933030f026e0137abe199e0157fd72b2f959df0354df489be8"
)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_candidates(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("schema") != CANDIDATE_SCHEMA
        or value.get("sha256") != EXPECTED_CANDIDATE_SHA256
        or task_family_split.canonical_digest(unsigned) != EXPECTED_CANDIDATE_SHA256
    ):
        raise ValueError("candidate roster is not the exact reviewed Sep24 authority")
    rows = value.get("tasks")
    if not isinstance(rows, list) or len(rows) != 33:
        raise ValueError("candidate roster must contain exactly 33 task versions")
    return value


def _source_identity(value: object, *, suffix: str) -> tuple[str, int]:
    if not isinstance(value, str) or not value.endswith(suffix):
        raise ValueError("source locator is not an exact reviewed Registry locator")
    locator = value.removesuffix(suffix)
    try:
        key, version = locator.rsplit("@", 1)
    except ValueError as error:
        raise ValueError("source locator omits an exact version") from error
    if not key.startswith("cyber/") or not version.isdigit():
        raise ValueError("source locator is not an exact reviewed Registry locator")
    version_index = int(version)
    if value != f"{key}@{version_index}{suffix}":
        raise ValueError("source locator is not canonical")
    return key, version_index


def project_lineage(task: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    """Project only exact identity and reviewed source lineage from a task GET."""
    if task.get("key") != selected.get("task_key"):
        raise ValueError("live task key differs from the candidate roster")
    if task.get("eval_task_version_id") != selected.get("task_version_id"):
        raise ValueError("live task version differs from the candidate roster")
    metadata = task.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("projection_id") != "blackbox_ctf_v1":
        raise ValueError("live task is not an exact blackbox projection")
    subject = metadata.get("cyber_subject")
    if not isinstance(subject, dict):
        raise ValueError("live task omits reviewed cyber subject metadata")
    sources = subject.get("atom_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("live task omits reviewed atom sources")
    atoms = []
    atom_locators = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("live atom source is malformed")
        key = atom_artifact_key(source.get("artifact_key"))
        locator_key, locator_version = _source_identity(
            source.get("locator"), suffix=":atom_source"
        )
        version = source.get("version_index")
        if (
            type(version) is not int
            or version < 0
            or locator_key != key
            or locator_version != version
        ):
            raise ValueError("live atom source identity is inconsistent")
        atoms.append(key)
        atom_locators.append(source["locator"])
    if len(atoms) != len(set(atoms)):
        raise ValueError("live task repeats an atom artifact key")
    task_graph_id = subject.get("task_graph_id")
    graph_locator = subject.get("source_locator")
    graph_key, _ = _source_identity(graph_locator, suffix=":task_graph_source")
    if (
        not isinstance(task_graph_id, str)
        or not task_graph_id
        or graph_key != f"cyber/task-graphs/{task_graph_id}"
    ):
        raise ValueError("live task graph identity is inconsistent")
    difficulty = metadata.get("task_graph_band") or metadata.get("expected_difficulty")
    if not isinstance(difficulty, str) or not difficulty:
        raise ValueError("live task difficulty metadata is missing")
    applications = sorted({key.split("/", 3)[2] for key in atoms})
    row = {
        "task_id": selected.get("task_id"),
        "task_key": selected["task_key"],
        "task_version_id": selected["task_version_id"],
        "qa_status": selected.get("qa_status"),
        "lineage": {
            "application": "+".join(applications),
            "difficulty": difficulty,
            "task_family": atoms[0] if len(atoms) == 1 else graph_key,
            "vulnerability_family": [f"atom:{key}" for key in sorted(atoms)],
        },
        "atom_artifact_keys": sorted(atoms),
        "atom_source_locators": sorted(atom_locators),
        "task_graph_source_locator": graph_locator,
    }
    row["lineage_binding_sha256"] = task_family_split.canonical_digest(row)
    return row


def capture(
    *,
    client: Any,
    candidates: dict[str, Any],
    candidate_path: str,
    candidate_file_sha256: str,
    observed_at: str,
) -> dict[str, Any]:
    qualification._account(client)  # noqa: SLF001
    rows = []
    for selected in candidates["tasks"]:
        task = self_hosted._request(  # noqa: SLF001
            client,
            "GET",
            f"/v1/tasks/{selected['task_key']}",
            params={"version_id": selected["task_version_id"]},
        )
        rows.append(project_lineage(task, selected))
    rows.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    body = {
        "schema": SCHEMA,
        "observed_at": observed_at,
        "source": {
            "authority": "read-only exact Fleet task-version GET",
            "candidate_path": candidate_path,
            "candidate_sha256": candidates["sha256"],
            "candidate_file_sha256": candidate_file_sha256,
            "controller_file_sha256": _file_digest(Path(__file__).resolve()),
        },
        "counts": {
            "candidate_task_versions": len(candidates["tasks"]),
            "exact_lineage_bound": len(rows),
            "runtime_receipt_qualified": 0,
        },
        "task_versions": rows,
        "evidence_boundary": {
            "proves": "exact task-version to reviewed atom-source lineage",
            "does_not_prove": [
                "environment startup",
                "tool reachability",
                "verifier execution",
                "finite authoritative outcome",
                "environment cleanup",
            ],
            "training_or_evaluation_admission": False,
        },
        "privacy": {
            "task_content_persisted": False,
            "session_content_persisted": False,
            "credentials_persisted": False,
        },
        "safety": {"external_mutations": 0, "launch_authorized": False},
    }
    body["sha256"] = task_family_split.canonical_digest(body)
    return body


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"refusing to replace immutable lineage census {path}") from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(DEFAULT_CANDIDATES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = _read_candidates(args.candidates)
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise SystemExit("FLEET_API_KEY is required")
    client = qualification._client(api_key)  # noqa: SLF001
    try:
        value = capture(
            client=client,
            candidates=candidates,
            candidate_path=str(args.candidates),
            candidate_file_sha256=_file_digest(args.candidates),
            observed_at=dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        )
    finally:
        client.close()
    _write_once(args.output, value)


if __name__ == "__main__":
    main()
