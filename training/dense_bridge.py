"""Run the frozen, multi-target Teacher3K packer without vendoring it.

This is a CPU-side preparation boundary, not a cluster launcher. Private source
rows and failures stay local; only digest/count receipts may be printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "34f41b8f172e30e4e12c22b71bafa365bcf9a31f"
SOURCES = {
    "training/__init__.py": "ecf358039bbb9b6cbab6546c9b1e61b9bc06c5b2d5b19907ff303a9277d77e27",
    "training/message_aligned_teacher_corpus.py": "02eee25e57bdf9c2084ce6790672102e07668168ebf3e4b82038aae4981bad50",
    "training/corpus.py": "28669e75e71323252765da81b8bd605e5dafaeecda55782a6316f3a0eb39f809",
    "training/dense.py": "712038f1abf4f461cf471afc50f7fbbfe90c9d2d7b9e0be1a81583378c933012",
    "training/io.py": "7a0b734a4ab7fb8b702430094c58c72f19ac8fc7cc5e056eb8410267e6bfdfe3",
    "training/post_sft_staging.py": "4467bd3fd47661b3cef01e47f7c40c32b0520d30e285f98a2f38651745948840",
    "training/post_sft_artifacts.py": "c09555c61cb25ad5867cce80ed8bef0684734231b5708b1be892e75b9dd980eb",
    "training/post_sft_base_surface.py": "f2e82927348fe844ff685f6506e1799a53cb9ecfbe58d50c654f2b1ba70c4f4f",
    "training/sft.py": "447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d",
    "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
    "training/task_family_split.py": "718145ae63c551aa67e91c806da78638708fad9a99f48f8a62a5f45b97716e55",
    "training/splits.py": "b3b5dd76e9307edb7572b5d71e192a4ae7846ac78e34f61cd19b57289bfda1c5",
    "training/models.py": "712adce5f073de13d02168cd17c2f7c396ba094a1ebc9a34decafb8639b8375e",
    "cyber_post_train/__init__.py": "3e70d0bf91f68f0190f2eb6d08e4e83d1ba58c1cbff29eb78e600c24e89c96a6",
    "cyber_post_train/jobs.py": "bc0d08a7a27b48ff5a21a5d824356d6e92e4856b17cc7296a976aa18a2baa40d",
}
INPUTS = ("normalized", "evidence", "family_roster", "tool_catalog", "tool_contract",
          "model_request_capture", "model_lock", "native_helper")
FIELDS = {"schema", *INPUTS, "tokenizer_root", "teacher_models", "max_length",
          "context_tokens", "output", "sha256"}
REQUEST_SCHEMA = "cyber_message_aligned_teacher_corpus_request_v1"
MANIFEST_SCHEMA = "cyber_dense_sft_corpus_v1"
RECEIPT_SCHEMA = "cyber_message_aligned_teacher_corpus_receipt_v1"
MODEL_LOCK_SHA = "sha256:f3926fe675263b25dc79c2b3881a9c463d6b7931e9d61aeb777d15efc61e35ac"
NATIVE_HELPER_SHA = "sha256:55c15b660067749febda00d4fb1c2110ff436717bbd4b73bf66055a73d0b87d5"
ALGORITHM = "anchored_complete_message_rounds_with_exact_tool_contract_v1"


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _legacy_digest(value: object) -> str:
    # The historical builder seals its manifest/receipt through jobs.digest,
    # whose JSON encoder uses Python's default ensure_ascii=True.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            and all(c in "0123456789abcdef" for c in value[7:]))


def validate_request(path: Path) -> dict:
    """Check request bytes and every bound input before running private code."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("request is absent or linked")
    request = json.loads(path.read_text())
    if (not isinstance(request, dict) or set(request) != FIELDS
            or request["schema"] != REQUEST_SCHEMA
            or request["sha256"] != _digest({k: v for k, v in request.items() if k != "sha256"})):
        raise ValueError("request schema or seal differs")
    for key in INPUTS:
        binding = request[key]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("input binding is malformed")
        relative = binding["path"]
        if not isinstance(relative, str) or not relative or not _sha(binding["sha256"]):
            raise ValueError("input identity is malformed")
        target = Path(relative)
        target = target if target.is_absolute() else path.parent / target
        if target.is_symlink() or not target.is_file() or _file_sha(target) != binding["sha256"]:
            raise ValueError("bound input differs")
    if (not isinstance(request["tokenizer_root"], str) or not request["tokenizer_root"]
            or not isinstance(request["output"], str) or not request["output"]
            or not isinstance(request["teacher_models"], list)
            or not request["teacher_models"]
            or any(not isinstance(v, str) or not v for v in request["teacher_models"])
            or len(set(request["teacher_models"])) != len(request["teacher_models"])
            or type(request["max_length"]) is not int
            or type(request["context_tokens"]) is not int
            or not 2 <= request["max_length"]
            or not 0 <= request["context_tokens"] <= request["max_length"]):
        raise ValueError("request parameters are malformed")
    output = Path(request["output"])
    output = output if output.is_absolute() else path.parent / output
    if output.exists() or output.is_symlink():
        raise FileExistsError("immutable output already exists")
    return request


def stage_historical(root: Path, *, repository: Path | None = None) -> None:
    """Copy only the exact historical source closure from a Git object."""
    repository = repository or Path(__file__).resolve().parents[1]
    for relative, expected in SOURCES.items():
        result = subprocess.run(["git", "-C", str(repository), "show", f"{COMMIT}:{relative}"],
                                capture_output=True, check=False)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise ValueError("frozen source object is unavailable or differs")
        target = Path(root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(result.stdout)


def build_dense(request_path: Path) -> dict:
    """Run the frozen packer, then independently verify its sealed output."""
    request_path = Path(request_path).absolute()
    request = validate_request(request_path)
    if (request["model_lock"]["sha256"] != MODEL_LOCK_SHA
            or request["native_helper"]["sha256"] != NATIVE_HELPER_SHA):
        raise ValueError("frozen model or native helper differs")
    output = Path(request["output"])
    output = output if output.is_absolute() else request_path.parent / output
    with tempfile.TemporaryDirectory(prefix="q38-frozen-dense-") as temporary:
        stage_historical(Path(temporary))
        result = subprocess.run(
            [sys.executable, "-m", "training.message_aligned_teacher_corpus",
             "--config", str(request_path)], cwd=temporary, capture_output=True,
            text=True, check=False, env={**os.environ, "PYTHONPATH": temporary},
        )
    if result.returncode:
        # The historical builder may quote private source text in an exception.
        raise RuntimeError("frozen dense builder rejected its inputs")
    manifest_path, receipt_path = output / "manifest.json", output / "RECEIPT.json"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, receipt_path)):
        raise ValueError("frozen builder output is incomplete")
    manifest, receipt = json.loads(manifest_path.read_text()), json.loads(receipt_path.read_text())
    train = manifest.get("files", {}).get("train", {})
    parquet = output / "train.parquet"
    expected_builders = {
        "message_aligned_teacher_corpus.py": "sha256:" + SOURCES["training/message_aligned_teacher_corpus.py"],
        "dense.py": "sha256:" + SOURCES["training/dense.py"],
        "corpus.py": "sha256:" + SOURCES["training/corpus.py"],
        "native_helper": NATIVE_HELPER_SHA,
    }
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("sha256") != _legacy_digest(
            {k: v for k, v in manifest.items() if k != "sha256"})
            or manifest.get("algorithm") != ALGORITHM
            or manifest.get("builder_sha256") != expected_builders
            or manifest.get("validation_mode") != "task_outcomes_only"
            or manifest.get("materialization", {}).get("request_sha256") != request["sha256"]
            or train.get("path") != "train.parquet" or train.get("format") != "pretokenized_assistant_segments_v1"
            or not isinstance(train.get("rows"), int) or train["rows"] < 1
            or parquet.is_symlink() or not parquet.is_file() or _file_sha(parquet) != train.get("sha256")
            or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("sha256") != _legacy_digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt.get("manifest_file_sha256") != _file_sha(manifest_path)
            or receipt.get("manifest_sha256") != manifest["sha256"]
            or receipt.get("train_parquet_sha256") != train["sha256"]
            or receipt.get("rows") != train["rows"]
            or receipt.get("supervised_tokens") != train.get("supervised_tokens")):
        raise ValueError("frozen dense output failed independent receipt checks")
    return {"manifest_sha256": manifest["sha256"], "receipt_sha256": receipt["sha256"],
            "train_sha256": train["sha256"], "rows": train["rows"],
            "supervised_tokens": train["supervised_tokens"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = build_dense(args.request)
    except Exception as error:
        print(f"frozen dense build rejected: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
