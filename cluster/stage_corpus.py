"""Fail-closed promotion of a locally sealed SFT corpus onto cluster SFS."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

UPLOAD_ROOT = Path("/upload")
SFS_ROOT = Path("/mnt/sfs")
ALLOWED_PREFIXES = ("session-corpus/", "session-scores/")


def _sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _safe_relative(value: str) -> Path:
    if not value.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"refusing corpus destination outside allowed prefixes: {value}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative corpus path: {value}")
    return relative


def main() -> None:
    archive = UPLOAD_ROOT / "stage.tar.gz"
    manifest_expected = os.environ["EXPECTED_MANIFEST_SHA256"]
    corpus_job_id = os.environ["EXPECTED_CORPUS_JOB_ID"]
    emitted_sessions = int(os.environ["EXPECTED_EMITTED_SESSIONS"])
    receipt_name = os.environ["RECEIPT_NAME"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", receipt_name):
        raise ValueError("unsafe receipt name")

    extracted = Path("/tmp/stage")
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (extracted / member.name).resolve()
            if extracted.resolve() not in target.parents and target != extracted.resolve():
                raise ValueError(f"unsafe archive member: {member.name}")
        bundle.extractall(extracted, filter="data")

    manifest_path = extracted / "stage-manifest.json"
    if _sha256(manifest_path) != manifest_expected:
        raise ValueError("stage manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "fleet_sft_corpus_stage_v1"
    assert manifest["corpus_job_id"] == corpus_job_id
    assert manifest["emitted_sessions"] == emitted_sessions

    for item in [*manifest["corpus"], manifest["scores"]]:
        relative = _safe_relative(item["relative_path"])
        source = extracted / relative
        expected = item["sha256"].removeprefix("sha256:")
        if _sha256(source) != expected:
            raise ValueError(f"source digest mismatch: {relative}")
        target = SFS_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) != expected:
                raise ValueError(f"refusing to overwrite conflicting artifact: {target}")
        else:
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(source, temporary)
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        if target.stat().st_mode & 0o777 != 0o644:
            raise ValueError(f"artifact is not trainer-readable: {target}")
        print("STAGED_VERIFIED", target, expected, flush=True)

    receipt = SFS_ROOT / "cyber-post-train" / "manifests" / f"{receipt_name}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(".json.partial")
    shutil.copyfile(manifest_path, temporary)
    os.chmod(temporary, 0o600)
    os.replace(temporary, receipt)
    print("CORPUS_STAGE_COMPLETE", receipt, flush=True)


if __name__ == "__main__":
    main()
