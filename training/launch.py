"""Bind the qualified 96k SkyRL SFT runtime without vendoring its source.

Prepare/preview allocate no GPUs. A zero-GPU Job proves native shared-SFS
loading before a separately reviewed, create-once GPU submission.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

COMMIT = "c908d3a828d070c6b27611fc388b1e7e3b4049dd"
ROOT = Path(__file__).resolve().parents[1]
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
NAMESPACE = "fleet-train-jobs"
LEASE = "chris-cpt-gpu-submit"
FULL_NAME = "chris-q38-corr96-full-v1"
FULL_OUTPUT = f"/mnt/sfs/jobs/{FULL_NAME}"
FULL_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-corrected-corpus-v1/full96-data"
MECHANICS_NAME = "chris-q38-prov96-step1-v1"
MECHANICS_OUTPUT = f"/mnt/sfs/jobs/{MECHANICS_NAME}"
MECHANICS_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-provisional96-corpus-v1/step1-data"
MECHANICS_BUILDER = "sha256:178d2c4f2ed3d6ad98bd1915b434b61cc714fb157cc30314e9aaf076ab4ae02c"
PROVISIONAL_SOURCE = "sha256:ffb3a687c4f1a325d2000dec30855f90cf6166acf7e03c7c6e13acbc5cf9a54a"
PROJECTION_FILE_SHA = "574adda7f9ed2bda5edb1c6e1453f69eb122fff4bfbf6806f9258f021d3d08bd"
SUBSET_FILE_SHA = "aa491003e49f823feffefcc9680b312cab82e533a8def1e992e6aa2d8d00234e"
SOURCES = {
    "training/__init__.py": "ecf358039bbb9b6cbab6546c9b1e61b9bc06c5b2d5b19907ff303a9277d77e27",
    "training/io.py": "7a0b734a4ab7fb8b702430094c58c72f19ac8fc7cc5e056eb8410267e6bfdfe3",
    "training/models.py": "712adce5f073de13d02168cd17c2f7c396ba094a1ebc9a34decafb8639b8375e",
    "training/sft.py": "447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d",
    "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
    "cyber_post_train/__init__.py": "3e70d0bf91f68f0190f2eb6d08e4e83d1ba58c1cbff29eb78e600c24e89c96a6",
    "cyber_post_train/jobs.py": "bc0d08a7a27b48ff5a21a5d824356d6e92e4856b17cc7296a976aa18a2baa40d",
    "configs/models/qwen38-27b-1d4bf0f2.lock.json": "f3926fe675263b25dc79c2b3881a9c463d6b7931e9d61aeb777d15efc61e35ac",
    "configs/models/qwen38-27b-1d4bf0f2.weights.json": "80a5e9de066e068abb012a0e9bd144676813a31042bb578226472803f8e0e84f",
}
COMMON_RECIPE = {"nodes": 1, "gpus_per_node": 8, "max_length": 98304,
                 "batch_size": 8, "microbatch_per_gpu": 1}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def historical_source(name: str) -> bytes:
    if name not in SOURCES:
        raise ValueError("unreviewed historical source")
    result = subprocess.run(
        ["git", "show", f"{COMMIT}:{name}"], cwd=ROOT, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError("qualified historical source is unavailable")
    if sha(result.stdout) != SOURCES[name]:
        raise ValueError("qualified historical source digest changed")
    return result.stdout


def _legacy(mode: str, value: dict, *, manifest: bytes | None = None) -> dict:
    """Run the historical compiler or Jobs client in its own import path."""
    script = """
import json,os,sys
from pathlib import Path
from training.sft import compile_sft,job_request
from cyber_post_train.jobs import Jobs,validate_preview
v=json.load(sys.stdin)
try:
    if os.environ['SFT_BRIDGE_MODE']=='compile':
        p=compile_sft(v['config'],relative_to=Path('configs/runs'))
        result={'plan':p,'request':job_request(p)}
    elif os.environ['SFT_BRIDGE_MODE']=='request':
        result={'request':job_request(v['plan'])}
    elif os.environ['SFT_BRIDGE_MODE']=='validate_preview':
        result=validate_preview(v['request'],v['preview'])
    elif os.environ['SFT_BRIDGE_MODE']=='preview':
        token=os.environ.get('FLEET_API_KEY')
        if not token: raise ValueError('missing API credential')
        r=v['request']
        with Jobs(token) as jobs:
            for existing in jobs.all_runs():
                if (existing.get('run_dir')==r['run_dir'] or
                    existing.get('name')==r['name'] or
                    existing.get('name','').startswith(r['name']+'-') or
                    existing.get('title')==r['title']):
                    raise ValueError('duplicate API run identity')
            result=validate_preview(r,jobs.preview(r))
    elif os.environ['SFT_BRIDGE_MODE']=='submit':
        token=os.environ.get('FLEET_API_KEY')
        if not token: raise ValueError('missing API credential')
        class ReviewedJobs(Jobs):
            def preview(self, request):
                rendered=super().preview(request)
                if validate_preview(request,rendered)['manifest_sha256']!=v['reviewed_manifest_sha256']:
                    raise ValueError('server render differs from reviewed preview')
                return rendered
        with ReviewedJobs(token) as jobs:
            result=jobs.submit_once(v['request'],Path(v['journal']))
    else: raise ValueError('unsupported bridge stage')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
except BaseException as exc:
    # Historical errors can contain private source paths or trainer details.
    print(type(exc).__name__,file=sys.stderr)
    sys.exit(2)
"""
    with tempfile.TemporaryDirectory(prefix="q38-sft-bridge-") as temporary:
        root = Path(temporary)
        (root / "configs/runs").mkdir(parents=True)
        for name in SOURCES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(historical_source(name))
        if manifest is not None:
            path = root / "configs/data/corpus.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(manifest)
        env = {**os.environ, "PYTHONPATH": str(root), "SFT_BRIDGE_MODE": mode}
        result = subprocess.run(
            [sys.executable, "-c", script], input=canonical(value), cwd=root,
            env=env, capture_output=True, check=False,
        )
    if result.returncode:
        # Captured stderr may include private trainer data; expose only a type.
        reason = result.stderr.decode(errors="replace").splitlines()[-1:] or [""]
        reason = reason[0].split(":", 1)[0].strip()
        if reason not in {"ValueError", "KeyError", "FileNotFoundError", "JobsError",
                          "ImportError", "ModuleNotFoundError", "RuntimeError", "TypeError"}:
            reason = "runtime_error"
        raise ValueError(f"historical {mode} gate rejected input ({reason})")
    lines = result.stdout.splitlines()
    if not lines:
        raise ValueError(f"historical {mode} gate returned no receipt")
    return json.loads(lines[-1])


def _require_profile_config(config: dict, full: bool) -> None:
    name, output, data_root, group, manifest = (
        (FULL_NAME, FULL_OUTPUT, FULL_DATA_ROOT, "qwen38-corrected-teacher96-full-v1", "qwen38-96k-full-v1.manifest.json")
        if full else (MECHANICS_NAME, MECHANICS_OUTPUT, MECHANICS_DATA_ROOT,
                      "qwen38-provisional96-mechanics-v1", "manifest.json"))
    recipe, data, wandb = (config.get(key, {}) for key in ("recipe", "data", "wandb"))
    expected = {**COMMON_RECIPE, "epochs": 1, "lr": 3e-6, "seed": 20260925,
                "eval_interval": 50 if full else 0, "checkpoint_interval": 50 if full else 1,
                "keep_checkpoints": "all" if full else 2}
    if (config.get("name") != name or config.get("output_root") != output
        or config.get("backend") != "skyrl" or config.get("cluster", {}).get("priority") != "c1"
        or config.get("model", {}).get("root") != "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
        or data.get("root") != data_root or not str(data.get("manifest", "")).endswith(manifest)
        or any(recipe.get(key) != value for key, value in expected.items())
        or wandb.get("entity") != "thefleet" or wandb.get("project") != "cyber-post-train"
        or wandb.get("group") != group or wandb.get("run_id") != name
        or wandb.get("name") != name or (not full and config.get("pause_after_step") != 1)):
        raise ValueError("96k profile identity, science, or c1 recipe differs")


def _require_mechanics_manifest(manifest: dict, manifest_bytes: bytes, receipt_bytes: bytes,
                                proof: dict, parquet: Path | None = None) -> None:
    from . import dense_bridge as dense
    from .projection import _digest
    from .runtime import MODEL, TOKENIZER_FILES
    train, material = manifest.get("files", {}).get("train", {}), manifest.get("materialization", {})
    receipt = json.loads(receipt_bytes)
    if (manifest.get("schema") != "cyber_dense_sft_corpus_v1"
        or manifest.get("sha256") != dense._legacy_digest({k: v for k, v in manifest.items() if k != "sha256"})
        or manifest.get("algorithm") != dense.ALGORITHM or "trainer_ready" in manifest
        or manifest.get("validation_mode") != "task_outcomes_only"
        or set(manifest.get("files", {})) != {"train"}
        or manifest.get("builder_sha256") != {
            "message_aligned_teacher_corpus.py": MECHANICS_BUILDER,
            "dense.py": "sha256:" + dense.SOURCES["training/dense.py"],
            "corpus.py": "sha256:" + dense.SOURCES["training/corpus.py"],
            "native_helper": dense.NATIVE_HELPER_SHA}
        or any(manifest.get("tokenizer", {}).get(key) != value for key, value in {
            "repo": MODEL[0], "revision": MODEL[1],
            "files": [{"path": p, "sha256": d} for p, d in TOKENIZER_FILES.items()]}.items())
        or manifest.get("split_sha256") != dense.TARGET_ANCHOR_SHA
        or proof.get("schema") != "qwen38_diagnostic_whole_train_subset_v1"
        or proof.get("sha256") != _digest({k: v for k, v in proof.items() if k != "sha256"})
        or proof.get("source_receipt_file_sha256") != PROVISIONAL_SOURCE
        or proof.get("selected_sessions") != 8 or proof.get("whole_sessions") is not True
        or proof.get("diagnostic_only") is not True or proof.get("training_ready") is not False
        or material.get("normalized_sha256") != proof.get("child_files_sha256", {}).get("normalized")
        or material.get("success_evidence_sha256") != proof.get("child_files_sha256", {}).get("evidence")
        or material.get("request_sha256") != proof.get("child_request_sha256")
        or material.get("family_role_anchor_sha256") != dense.TARGET_ANCHOR_SHA
        or manifest.get("max_length") != 98304 or manifest.get("context_tokens") != 98304
        or train.get("format") != "pretokenized_assistant_segments_v1" or train.get("path") != "train.parquet"
        or type(train.get("rows")) is not int or train["rows"] <= 8
        or train.get("source_sessions") != proof["selected_sessions"]
        or receipt.get("schema") != dense.RECEIPT_SCHEMA
        or receipt.get("sha256") != dense._legacy_digest({k: v for k, v in receipt.items() if k != "sha256"})
        or receipt.get("manifest_file_sha256") != "sha256:" + sha(manifest_bytes)
        or receipt.get("manifest_sha256") != manifest["sha256"]
        or receipt.get("train_parquet_sha256") != train.get("sha256")
        or receipt.get("source_selection_sha256") != manifest.get("source_sha256")
        or (receipt.get("rows"), receipt.get("source_sessions"), receipt.get("supervised_tokens"))
           != (train["rows"], train["source_sessions"], train.get("supervised_tokens"))):
        raise ValueError("sealed complete-session v2 parent corpus is not qualified")
    if parquet is not None and (parquet.is_symlink() or not parquet.is_file()
                                or dense._file_sha(parquet) != train["sha256"]):
        raise ValueError("parent v2 TRAIN Parquet differs from sealed receipt")


def _require_goal_anchor(_manifest: dict) -> None:
    # A relabeled manifest cannot replace reviewed per-session target-anchor proof.
    raise ValueError("corrected target-anchor producer/attestation is not yet qualified")


def _require_full_manifest(manifest: dict, data_root: str) -> None:
    train, dev = (manifest.get("files", {}).get(key, {}) for key in ("train", "dev"))
    composition = manifest.get("composition", {})
    from .runtime import MODEL, TOKENIZER_FILES

    tokenizer = manifest.get("tokenizer", {})
    if (manifest.get("schema") != "cyber_dense_sft_corpus_v1"
        or manifest.get("sha256") != "sha256:" + sha(canonical(
            {key: value for key, value in manifest.items() if key != "sha256"}))
        or manifest.get("validation_mode") != "teacher_cross_entropy"
        or set(manifest.get("files", {})) != {"train", "dev"}
        or train.get("format") != "pretokenized_assistant_segments_v1"
        or dev.get("format") != "chat_messages_last_assistant_v2"
        or type(train.get("supervised_tokens")) is not int
        or train["supervised_tokens"] < 20_000_000
        or type(train.get("rows")) is not int or train["rows"] < 1
        or type(dev.get("rows")) is not int or dev["rows"] < 1
        or type(train.get("source_sessions")) is not int or train["source_sessions"] < 1
        or tokenizer.get("repo") != MODEL[0] or tokenizer.get("revision") != MODEL[1]
        or tokenizer.get("files") != [{"path": p, "sha256": d}
                                       for p, d in TOKENIZER_FILES.items()]
        or composition.get("schema") != "qwen38_dense_train_contiguous_dev_v1"
        or composition.get("corpus_root") != data_root
        or any(not re.fullmatch(r"sha256:[a-f0-9]{64}", composition.get(key, ""))
               for key in ("dense_manifest_sha256", "dense_receipt_sha256",
                           "dev_manifest_sha256", "dev_source_receipt_sha256",
                           "family_roster_sha256"))
        or composition.get("family_roster_sha256") != manifest.get("materialization", {}).get(
            "family_roster_sha256")
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest.get("split_sha256", ""))
        or not re.fullmatch(r"sha256:[a-f0-9]{64}",
                            manifest.get("materialization", {}).get("request_sha256", ""))
        or set(train.get("task_keys", [])) & set(dev.get("task_keys", []))):
        raise ValueError("full corpus lacks sealed dense train and disjoint teacher-CE dev")
    _require_goal_anchor(manifest)


def prepare(config_path: Path, destination: Path,
            mechanics_proof: tuple[Path, ...] | None = None) -> dict:
    if destination.exists() or destination.is_symlink():
        raise ValueError("prepared destination already exists")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    full = config.get("name") == FULL_NAME
    _require_profile_config(config, full)
    source = Path(config["data"]["manifest"])
    manifest_path = source if source.is_absolute() else config_path.parent / source
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    mechanics_receipt = b"" if full else (manifest_path.parent / "RECEIPT.json").read_bytes()
    subset_bytes = projection_bytes = b""
    if full:
        if mechanics_proof is not None:
            raise ValueError("full run cannot use diagnostic proof")
        _require_full_manifest(manifest, config["data"]["root"])
    else:
        if mechanics_proof is None or len(mechanics_proof) != 6:
            raise ValueError("independent whole-TRAIN-session proof paths required")
        from .projection import verify_subset
        proof = verify_subset(*mechanics_proof)
        subset_bytes = mechanics_proof[-1].read_bytes()
        projection_bytes = (mechanics_proof[1] / "PROJECTION.json").read_bytes()
        if (sha(subset_bytes) != SUBSET_FILE_SHA or sha(projection_bytes) != PROJECTION_FILE_SHA
            or json.loads(subset_bytes) != proof
            or "sha256:" + sha(projection_bytes) != proof["full_projection_file_sha256"]):
            raise ValueError("whole-session proof changed after replay")
        _require_mechanics_manifest(manifest, manifest_bytes, mechanics_receipt,
                                    proof, manifest_path.parent / "train.parquet")
    staged = json.loads(json.dumps(config))
    rows, batch = manifest["files"]["train"]["rows"], config["recipe"]["batch_size"]
    staged["recipe"]["keep_checkpoints"] = (rows + batch - 1) // batch
    staged["model"]["lock"] = "../models/qwen38-27b-1d4bf0f2.lock.json"
    staged["model"]["weights"] = "../models/qwen38-27b-1d4bf0f2.weights.json"
    staged["data"]["manifest"] = "../data/corpus.json"
    compiled = _legacy("compile", {"config": staged}, manifest=manifest_bytes)
    plan, request = compiled["plan"], compiled["request"]
    if (plan["model"]["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        or plan["validation_mode"] != ("teacher_cross_entropy" if full else "task_outcomes_only")
        or plan["datasets"]["train"].get("format") != "pretokenized_assistant_segments_v1"
        or set(plan["datasets"]) != ({"train", "dev"} if full else {"train"})
        or (full and plan["datasets"]["dev"].get("format") != "chat_messages_last_assistant_v2")
        or plan["schema"] != "cyber_sft_runtime_dense_v1"
        or (not full and (plan.get("pause_after_step") != 1 or plan["recipe"]["max_steps"] <= 1
                          or plan["recipe"]["checkpoint_interval"] != 1
                          or plan["recipe"]["eval_interval"] != 0))
        or plan["recipe"]["keep_checkpoints"] <
            (plan["recipe"]["max_steps"] + plan["recipe"]["checkpoint_interval"] - 1)
            // plan["recipe"]["checkpoint_interval"]
        or request["failureAlerts"] is not False
        or request["priority_class"] != "c1"
        or request["workers"] != 1
        or request["gpus_per_worker"] != 8
        or Path(request["run_dir"]).name != request["name"]
        or (not full and (request["name"] != MECHANICS_NAME
                          or request["run_dir"] != MECHANICS_OUTPUT
                          or plan["corpus_manifest_sha256"] != manifest["sha256"]))
        or (full and (request["name"] != FULL_NAME or request["run_dir"] != FULL_OUTPUT
                      or plan["datasets"]["train"]["supervised_tokens"] < 20_000_000
                      or plan["corpus_manifest_sha256"] != manifest["sha256"]
                      or plan["recipe"]["keep_checkpoints"] != plan["recipe"]["max_steps"]))):
        raise ValueError("compiled 96k request failed an immutable safety/science gate")
    receipt = {
        "schema": "qwen38_96k_full_prepared_v1" if full else "qwen38_96k_mechanics_prepared_v1",
        "historical_commit": COMMIT,
        "config_sha256": sha(config_bytes), "manifest_file_sha256": sha(manifest_bytes),
        "plan_sha256": sha(canonical(plan)), "request_sha256": sha(canonical(request)),
        "status": "prepared_not_submitted", "purpose": "full_sft" if full else "one_step_mechanics_only",
    }
    if not full:
        receipt["corpus_receipt_file_sha256"] = sha(mechanics_receipt)
        receipt["provisional_source_receipt_file_sha256"] = PROVISIONAL_SOURCE
        receipt["train_parquet_sha256"] = manifest["files"]["train"]["sha256"]
        receipt["subset_receipt_file_sha256"] = sha(subset_bytes)
        receipt["projection_receipt_file_sha256"] = sha(projection_bytes)
    if full:
        receipt["supervised_tokens"] = plan["datasets"]["train"]["supervised_tokens"]
        interval = plan["recipe"]["checkpoint_interval"]
        receipt["planned_native_checkpoints"] = (plan["recipe"]["max_steps"] + interval - 1) // interval
        receipt["checkpoint_retention_capacity"] = plan["recipe"]["keep_checkpoints"]
    destination.mkdir(parents=True, mode=0o700)
    _create_only(destination / "corpus.manifest.json", manifest_bytes)
    if not full:
        _create_only(destination / "corpus.RECEIPT.json", mechanics_receipt)
        _create_only(destination / "SUBSET.json", subset_bytes)
        _create_only(destination / "PROJECTION.json", projection_bytes)
    for name, value in (("plan.json", plan), ("request.json", request), ("PREPARED.json", receipt)):
        _create_only(destination / name, value)
    return receipt


def prepared(directory: Path) -> tuple[dict, dict, dict]:
    receipt = json.loads((directory / "PREPARED.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if (receipt.get("historical_commit") != COMMIT
        or receipt.get("plan_sha256") != sha(canonical(plan))
        or receipt.get("request_sha256") != sha(canonical(request))
        or _legacy("request", {"plan": plan})["request"] != request):
        raise ValueError("prepared binding changed")
    if (plan.get("execution", {}).get("priority") != "c1" or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8 or any(
            plan["recipe"].get(key) != value for key, value in COMMON_RECIPE.items())):
        raise ValueError("prepared c1, root alerts, or one-node 96k recipe changed")
    manifest_bytes = (directory / "corpus.manifest.json").read_bytes()
    if receipt.get("manifest_file_sha256") != sha(manifest_bytes):
        raise ValueError("corpus manifest bytes changed")
    manifest = json.loads(manifest_bytes)
    if receipt.get("schema") == "qwen38_96k_full_prepared_v1":
        _require_full_manifest(manifest, FULL_DATA_ROOT)
        interval = plan["recipe"]["checkpoint_interval"]
        if (plan.get("schema") != "cyber_sft_runtime_dense_v1"
            or plan.get("corpus_manifest_sha256") != manifest["sha256"]
            or plan["recipe"]["eval_interval"] != interval
            or plan["recipe"]["keep_checkpoints"] != plan["recipe"]["max_steps"]
            or receipt.get("planned_native_checkpoints") != (plan["recipe"]["max_steps"] + interval - 1) // interval
            or receipt.get("checkpoint_retention_capacity") != plan["recipe"]["keep_checkpoints"]
            or receipt.get("supervised_tokens") != plan["datasets"]["train"]["supervised_tokens"]):
            raise ValueError("full run evidence/retention binding changed")
    elif receipt.get("schema") == "qwen38_96k_mechanics_prepared_v1":
        mechanics_receipt = (directory / "corpus.RECEIPT.json").read_bytes()
        subset_bytes = (directory / "SUBSET.json").read_bytes()
        projection_bytes = (directory / "PROJECTION.json").read_bytes()
        proof = json.loads(subset_bytes)
        _require_mechanics_manifest(manifest, manifest_bytes, mechanics_receipt, proof)
        if (receipt.get("purpose") != "one_step_mechanics_only"
            or receipt.get("corpus_receipt_file_sha256") != sha(mechanics_receipt)
            or receipt.get("provisional_source_receipt_file_sha256") != PROVISIONAL_SOURCE
            or receipt.get("train_parquet_sha256") != manifest["files"]["train"]["sha256"]
            or receipt.get("subset_receipt_file_sha256") != sha(subset_bytes)
            or receipt.get("projection_receipt_file_sha256") != sha(projection_bytes)
            or sha(subset_bytes) != SUBSET_FILE_SHA or sha(projection_bytes) != PROJECTION_FILE_SHA
            or proof.get("full_projection_file_sha256") != "sha256:" + sha(projection_bytes)
            or plan.get("schema") != "cyber_sft_runtime_dense_v1"
            or plan.get("validation_mode") != "task_outcomes_only"
            or set(plan.get("datasets", {})) != {"train"}
            or plan.get("corpus_manifest_sha256") != manifest["sha256"]
            or plan.get("pause_after_step") != 1 or plan["recipe"]["max_steps"] <= 1
            or plan["recipe"]["eval_interval"] != 0
            or plan["recipe"]["checkpoint_interval"] != 1
            or plan["recipe"]["keep_checkpoints"] != plan["recipe"]["max_steps"]
            or request["name"] != MECHANICS_NAME or request["run_dir"] != MECHANICS_OUTPUT):
            raise ValueError("one-step mechanics binding changed")
    else:
        raise ValueError("unsupported prepared run profile")
    return plan, request, receipt


def _preflight_matches(result: dict, receipt: dict) -> bool:
    required = {
        "native_sources", "model_files", "dataset_files", "native_config",
        "native_forward_backward_signature", "native_train_only_loader",
        "tokenization", "target_accounting",
    }
    counts = result.get("counts", {})
    return (
        result.get("status") == "passed" and result.get("gpus") == 0
        and result.get("request_sha256") == receipt["request_sha256"]
        and result.get("plan_sha256") == receipt["plan_sha256"]
        and required <= set(result.get("checked", []))
        and set(counts) == ({"train", "dev"} if receipt["schema"] ==
                            "qwen38_96k_full_prepared_v1" else {"train"})
        and all(counts[split].get("rows", 0) > 0 and
                counts[split].get("supervised_tokens", 0) > 0
                for split in counts)
        and (receipt["schema"] == "qwen38_96k_full_prepared_v1" or counts["train"]["rows"] > 8)
    )


def _kubectl(context: str, args: list[str], payload: dict | None = None,
             *, missing_ok: bool = False) -> dict | None:
    if not context or not re.fullmatch(r"[A-Za-z0-9_.:@/-]+", context):
        raise ValueError("explicit Kubernetes context required")
    result = subprocess.run(
        ["kubectl", "--context", context, "--request-timeout=60s", *args, "-o", "json"],
        input=None if payload is None else canonical(payload), capture_output=True,
        timeout=75, check=False,
    )
    if result.returncode or (not result.stdout and not missing_ok):
        raise ValueError("Kubernetes operation failed; reconcile before retry")
    if not result.stdout and missing_ok:
        return None
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Kubernetes returned a non-object")
    return value


def cpu_job(directory: Path, attempt: int = 1) -> dict:
    plan, request, receipt = prepared(directory)
    if type(attempt) is not int or not 1 <= attempt <= 99:
        raise ValueError("CPU attempt must be 1..99")
    name = request["name"] + f"-pre-a{attempt:02d}"
    if len(name) > 63:
        raise ValueError("CPU Job name exceeds Kubernetes limit")
    files = {"src/" + name: historical_source(name).decode()
             for name in SOURCES if name.endswith(".py")}
    worker = (ROOT / "training/preflight_worker.py").read_bytes()
    manifest = {
        "schema": "qwen38_cpu_preflight_bundle_v1", "run_name": request["name"],
        "output_root": request["run_dir"], "plan_sha256": receipt["plan_sha256"],
        "request_sha256": receipt["request_sha256"],
        "files": {name: sha(text.encode()) for name, text in files.items()},
    }
    manifest["sha256"] = sha(canonical(manifest))
    blob = gzip.compress(canonical({"manifest": manifest, "files": files, "plan": plan}), mtime=0)
    encoded = base64.b64encode(blob).decode()
    chunks = [encoded[i:i + 48_000] for i in range(0, len(encoded), 48_000)]
    if not 1 <= len(chunks) <= 32:
        raise ValueError("CPU preflight bundle exceeds environment bound")
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/plan-sha256": receipt["plan_sha256"],
        "cyber-post-train.fleet.ai/request-sha256": receipt["request_sha256"],
        "cyber-post-train.fleet.ai/output-root": request["run_dir"],
        "cyber-post-train.fleet.ai/bundle-sha256": sha(blob),
        "cyber-post-train.fleet.ai/worker-sha256": sha(worker),
    }
    env = {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
           "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "Q38_BUNDLE_SHA256": sha(blob),
           **{f"Q38_BUNDLE_{i}": text for i, text in enumerate(chunks)}}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": "fleet-train-jobs",
                     "annotations": annotations,
                     "labels": {"kueue.x-k8s.io/queue-name": "training-lq",
                                "kueue.x-k8s.io/priority-class": "q1",
                                "cyber-post-train.fleet.ai/owner": "chris"}},
        "spec": {"activeDeadlineSeconds": 1800, "backoffLimit": 0,
                 "suspend": True, "ttlSecondsAfterFinished": 7200,
                 "template": {"spec": {
                     "automountServiceAccountToken": False, "restartPolicy": "Never",
                     "nodeSelector": {"kubernetes.io/arch": "amd64",
                                      "workload": "fleetai-training-ng-cpu"},
                     "priorityClassName": "c1", "priority": 10000,
                     "tolerations": [{"key": "workload", "operator": "Equal",
                                      "value": "fleetai-training-ng-cpu", "effect": "NoSchedule"}],
                     "containers": [{
                         "name": "preflight", "image": request["image"],
                         "command": ["python", "-u", "-c", worker.decode()],
                         "env": [{"name": k, "value": v} for k, v in sorted(env.items())],
                         "terminationMessagePolicy": "File",
                         "resources": {"requests": {"cpu": "4", "memory": "16Gi"},
                                       "limits": {"cpu": "8", "memory": "24Gi"}},
                         "securityContext": {"allowPrivilegeEscalation": False,
                                             "privileged": False},
                         "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs",
                                           "readOnly": True}],
                     }],
                     "volumes": [{"name": "sfs", "persistentVolumeClaim": {
                         "claimName": "sfs-shared", "readOnly": True}}],
                 }}},
    }


def _check_cpu_render(expected: dict, actual: dict, *, allow_unsuspended: bool = False) -> None:
    meta, spec = actual.get("metadata", {}), actual.get("spec", {})
    template = spec.get("template", {}).get("spec", {})
    original = expected["spec"]["template"]["spec"]
    containers = template.get("containers", [])
    actual_env = containers[0].get("env", []) if len(containers) == 1 else []
    expected_env = original["containers"][0]["env"]
    env_equal = (len(actual_env) == len(expected_env)
                 and len({v.get("name") for v in actual_env}) == len(actual_env)
                 and {v.get("name"): v.get("value", "") for v in actual_env}
                     == {v["name"]: v["value"] for v in expected_env})
    if (actual.get("kind") != "Job" or meta.get("namespace") != "fleet-train-jobs"
        or meta.get("name") != expected["metadata"]["name"]
        or meta.get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
        or any(meta.get("annotations", {}).get(k) != v
               for k, v in expected["metadata"]["annotations"].items())
        or meta.get("labels", {}).get("kueue.x-k8s.io/queue-name") != "training-lq"
        or meta.get("labels", {}).get("kueue.x-k8s.io/priority-class") != "q1"
        or (spec.get("suspend") is not True
            and not (allow_unsuspended and spec.get("suspend") is False))
        or spec.get("backoffLimit") != 0
        or template.get("priorityClassName") != "c1" or template.get("priority") != 10000
        or template.get("nodeSelector") != original["nodeSelector"]
        or len(containers) != 1 or containers[0].get("image") != original["containers"][0]["image"]
        or containers[0].get("command") != original["containers"][0]["command"]
        or not env_equal
        or containers[0].get("resources") != original["containers"][0]["resources"]
        or containers[0].get("securityContext") != original["containers"][0]["securityContext"]
        or containers[0].get("envFrom")
        or template.get("automountServiceAccountToken") is not False
        or any("nvidia.com/gpu" in str(c.get("resources", {})) for c in containers)
        or template.get("volumes") != original["volumes"]
        or containers[0].get("volumeMounts") != original["containers"][0]["volumeMounts"]):
        raise ValueError("server-rendered CPU Job failed root alert, c1/q1, zero-GPU or SFS gate")


def cpu_preview(directory: Path, context: str, attempt: int = 1) -> dict:
    if context != PROD_CONTEXT:
        raise ValueError("96k shared-SFS CPU proof requires the production context")
    pvc = _kubectl(context, ["-n", NAMESPACE, "get", "pvc", "sfs-shared"])
    if (pvc.get("metadata", {}).get("uid") != "34cb6b11-8766-4294-9f9e-332064ea17d5"
        or pvc.get("status", {}).get("phase") != "Bound"):
        raise ValueError("qualified SFS PVC identity drifted")
    job = cpu_job(directory, attempt)
    actual = _kubectl(context, ["create", "--dry-run=server", "-f", "-"], job)
    _check_cpu_render(job, actual)
    return {"status": "previewed_not_created", "name": job["metadata"]["name"],
            "job_sha256": sha(canonical(job)), "gpus": 0,
            "root_failure_alerts": "off", "priority": "c1/q1"}


def _create_only(path: Path, value: dict | bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value if isinstance(value, bytes) else canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def cpu_create(directory: Path, context: str, reviewed_sha: str, attempt: int = 1) -> dict:
    proof = cpu_preview(directory, context, attempt)
    if proof["job_sha256"] != reviewed_sha:
        raise ValueError("CPU server preview differs from reviewed Job")
    job = cpu_job(directory, attempt)
    name = job["metadata"]["name"]
    if _kubectl(context, ["-n", "fleet-train-jobs", "get", "job", name,
                          "--ignore-not-found"], missing_ok=True) is not None:
        raise ValueError("CPU Job already exists")
    journal = directory / f"CPU-{attempt:02d}-INTENT.json"
    _create_only(journal, {"state": "CREATE_INTENT_DO_NOT_RETRY", **proof})
    actual = _kubectl(context, ["-n", "fleet-train-jobs", "create", "-f", "-"], job)
    _check_cpu_render(job, actual)
    uid = actual.get("metadata", {}).get("uid")
    if not uid:
        raise ValueError("ambiguous CPU create response; reconcile before retry")
    _create_only(directory / f"CPU-{attempt:02d}-CREATED.json",
                 {"name": name, "uid": uid, "job_sha256": reviewed_sha})
    return {"status": "created", "name": name, "uid": uid, "gpus": 0}


def _cpu_observation(directory: Path, context: str, attempt: int = 1) -> dict:
    job = cpu_job(directory, attempt)
    created = json.loads((directory / f"CPU-{attempt:02d}-CREATED.json").read_text())
    if created["job_sha256"] != sha(canonical(job)):
        raise ValueError("CPU created binding drifted")
    name, uid = created["name"], created["uid"]
    actual = _kubectl(context, ["-n", "fleet-train-jobs", "get", "job", name])
    _check_cpu_render(job, actual, allow_unsuspended=True)
    if (actual["metadata"].get("uid") != uid or actual.get("status", {}).get("succeeded") != 1
        or not any(c.get("type") == "Complete" and c.get("status") == "True"
                   for c in actual.get("status", {}).get("conditions", []))):
        raise ValueError("CPU Job has not completed successfully")
    pods = _kubectl(context, ["-n", "fleet-train-jobs", "get", "pods", "-l",
                              f"batch.kubernetes.io/job-name={name}"])["items"]
    owned = [p for p in pods if any(o.get("uid") == uid and o.get("kind") == "Job"
                                      for o in p.get("metadata", {}).get("ownerReferences", []))]
    if len(owned) != 1 or owned[0].get("status", {}).get("phase") != "Succeeded":
        raise ValueError("CPU Pod identity or completion drifted")
    pod = owned[0]
    statuses = pod["status"].get("containerStatuses", [])
    pod_spec = pod.get("spec", {})
    pod_containers = pod_spec.get("containers", [])
    mounts = pod_containers[0].get("volumeMounts", []) if len(pod_containers) == 1 else []
    if (pod_spec.get("priorityClassName") != "c1" or
        len(pod_containers) != 1 or not mounts or
        "nvidia.com/gpu" in str(pod_containers[0].get("resources", {})) or
        mounts[0].get("readOnly") is not True):
        raise ValueError("CPU Pod was mutated to a GPU, non-c1, or writable mount")
    digest = job["spec"]["template"]["spec"]["containers"][0]["image"].split("@")[-1]
    if (len(statuses) != 1 or statuses[0].get("restartCount") != 0
        or not statuses[0].get("imageID", "").endswith("@" + digest)
        or statuses[0].get("state", {}).get("terminated", {}).get("exitCode") != 0):
        raise ValueError("CPU Pod restart, image, or exit drifted")
    envelope = json.loads(statuses[0]["state"]["terminated"].get("message", ""))
    plan, _request, receipt = prepared(directory)
    result = envelope.get("native", {})
    stamp = envelope.get("observed_at_unix")
    if (envelope.get("schema") != "qwen38_cpu_preflight_observation_v1"
        or envelope.get("status") != "passed" or envelope.get("output_absent") is not True
        or type(stamp) not in (int, float) or not 0 <= time.time() - stamp <= 1800
        or envelope.get("bundle_sha256") != job["metadata"]["annotations"][
            "cyber-post-train.fleet.ai/bundle-sha256"]
        or not _preflight_matches(result, receipt)
        or plan["output_root"] != job["metadata"]["annotations"].get(
            "cyber-post-train.fleet.ai/output-root")):
        raise ValueError("CPU native receipt or output-absence proof drifted")
    result = {**result, "cpu_job": {"name": name, "uid": uid,
              "pod_uid": pod["metadata"]["uid"], "context": context,
              "observed_at_unix": envelope["observed_at_unix"], "output_absent": True}}
    return result


def cpu_collect(directory: Path, context: str, attempt: int = 1) -> dict:
    result = _cpu_observation(directory, context, attempt)
    _create_only(directory / "PREFLIGHT.json", result)
    return {"status": "passed", "job_uid": result["cpu_job"]["uid"],
            "pod_uid": result["cpu_job"]["pod_uid"],
            "plan_sha256": result["plan_sha256"]}


def preview(directory: Path) -> dict:
    _plan, request, receipt = prepared(directory)
    gate = json.loads((directory / "PREFLIGHT.json").read_text())
    if not _preflight_matches(gate, receipt) or gate.get("cpu_job", {}).get("output_absent") is not True:
        raise ValueError("exact CPU preflight is absent")
    proof = _legacy("preview", {"request": request})
    return {**proof, "status": "previewed_not_submitted", "request_sha256": receipt["request_sha256"]}


def _owned(obj: dict) -> bool:
    meta = obj.get("metadata", {})
    labels = meta.get("labels", {}) or {}
    return any(str(value).startswith("chris-") for value in (
        meta.get("name", ""), labels.get("fleet.ai/run-name", ""),
        labels.get("inference.fleet.ai/model", ""),
        labels.get("cyber-post-train.fleet.ai/owner", ""))) or labels.get(
            "cyber-post-train.fleet.ai/owner") == "chris"


def capacity() -> dict:
    """Count live owned GPU Pods and unadmitted RayJobs on both clusters."""
    active_nodes, gpus, queued, pod_count = set(), 0, set(), 0
    for context in (PROD_CONTEXT, DEV_CONTEXT):
        pods = _kubectl(context, ["get", "pods", "--all-namespaces"])["items"]
        rayjobs = _kubectl(context, ["get", "rayjobs.ray.io", "--all-namespaces"])["items"]
        batch_jobs = _kubectl(context, ["get", "jobs.batch", "--all-namespaces"])["items"]
        workloads = _kubectl(context, ["get", "workloads.kueue.x-k8s.io",
                                    "--all-namespaces"])["items"]
        allocated_names, allocated_clusters = set(), set()
        for pod in pods:
            if not _owned(pod) or pod.get("status", {}).get("phase") not in {
                "Pending", "Running", "Unknown"}:
                continue
            spec = pod.get("spec", {})
            containers = spec.get("containers", []) + spec.get("initContainers", [])
            quantities = [int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)
                              or c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
                          for c in containers]
            count = max(sum(quantities[:len(spec.get("containers", []))]),
                        max(quantities[len(spec.get("containers", [])):] or [0]))
            if not count:
                continue
            pod_count += 1
            identity = (pod.get("metadata", {}).get("labels") or {}).get("fleet.ai/run-name")
            if identity:
                allocated_names.add(identity)
            cluster = (pod.get("metadata", {}).get("labels") or {}).get("ray.io/cluster")
            if cluster:
                allocated_clusters.add(cluster)
            if spec.get("nodeName"):
                active_nodes.add((context, spec["nodeName"]))
                gpus += count
            else:
                queued.add((context, identity or pod["metadata"]["name"]))
        for job in rayjobs:
            if not _owned(job) or job.get("status", {}).get("jobStatus") in {
                "SUCCEEDED", "FAILED", "STOPPED"}:
                continue
            name = job["metadata"]["name"]
            cluster = job.get("status", {}).get("rayClusterName")
            if name not in allocated_names and cluster not in allocated_clusters:
                queued.add((context, name))
        for job in batch_jobs:
            if not _owned(job) or job.get("status", {}).get("active", 0):
                continue
            conditions = job.get("status", {}).get("conditions", [])
            if any(c.get("type") in {"Complete", "Failed"} and c.get("status") == "True"
                   for c in conditions):
                continue
            queued.add((context, job["metadata"]["name"]))
        for workload in workloads:
            owners = [o for o in workload.get("metadata", {}).get("ownerReferences", [])
                      if o.get("kind") == "RayJob" and str(o.get("name", "")).startswith("chris-")]
            if not owners or any(c.get("type") == "Finished" and c.get("status") == "True"
                                     for c in workload.get("status", {}).get("conditions", [])):
                continue
            if not any(c.get("type") == "Admitted" and c.get("status") == "True"
                       for c in workload.get("status", {}).get("conditions", [])):
                queued.add((context, owners[0]["name"]))
    result = {"active_nodes": len(active_nodes), "active_gpus": gpus,
              "queued_jobs": len(queued), "owned_active_gpu_pods": pod_count}
    if result["active_nodes"] + 1 > 10 or gpus + 8 > 80 or result["queued_jobs"] + 1 > 10:
        raise ValueError("project-owned active GPU or queue capacity is exhausted")
    return result


def _lease(holder: str | None, *, expected_uid: str | None = None,
           expected_holder: str | None = None) -> dict:
    """Use Kubernetes resourceVersion as a cross-process submit compare-and-swap."""
    current = _kubectl(PROD_CONTEXT, ["-n", NAMESPACE, "get", "lease", LEASE,
                                      "--ignore-not-found"], missing_ok=True)
    now = datetime.now(timezone.utc)
    if holder is not None:
        if current:
            spec = current.get("spec", {})
            renew = spec.get("renewTime") or spec.get("acquireTime")
            stamp = datetime.fromisoformat(renew.replace("Z", "+00:00")) if renew else now
            if spec.get("holderIdentity") and (now - stamp).total_seconds() < min(
                int(spec.get("leaseDurationSeconds", 900)), 900):
                raise ValueError("another project submitter holds the capacity lease")
            verb = "replace"
        else:
            verb = "create"
        spec = {"holderIdentity": holder, "leaseDurationSeconds": 900,
                "acquireTime": now.isoformat(), "renewTime": now.isoformat()}
    else:
        if (not current or current["metadata"].get("uid") != expected_uid
            or current.get("spec", {}).get("holderIdentity") != expected_holder):
            raise ValueError("capacity lease identity changed before release")
        verb = "replace"
        spec = {"holderIdentity": "", "leaseDurationSeconds": 0,
                "renewTime": now.isoformat()}
    meta = {"name": LEASE, "namespace": NAMESPACE}
    if current:
        meta["resourceVersion"] = current["metadata"]["resourceVersion"]
    lease = {"apiVersion": "coordination.k8s.io/v1", "kind": "Lease",
             "metadata": meta, "spec": spec}
    result = _kubectl(PROD_CONTEXT, ["-n", NAMESPACE, verb, "-f", "-"], lease)
    if result.get("spec", {}).get("holderIdentity") != (holder or ""):
        raise ValueError("capacity lease compare-and-swap drifted")
    return result


def submit(directory: Path, reviewed_manifest_sha256: str) -> dict:
    """Create one paid run only after the independent CPU and server-review gates."""
    _plan, request, receipt = prepared(directory)
    gate = json.loads((directory / "PREFLIGHT.json").read_text())
    cpu = gate.get("cpu_job", {})
    stamp = cpu.get("observed_at_unix")
    age = time.time() - stamp if type(stamp) in (int, float) else float("inf")
    if (not _preflight_matches(gate, receipt) or cpu.get("output_absent") is not True
        or cpu.get("context") != PROD_CONTEXT or not cpu.get("uid") or not cpu.get("pod_uid")
        or not 0 <= age <= 1800):
        raise ValueError("fresh UID-bound shared-SFS CPU proof is absent")
    if gate != _cpu_observation(directory, PROD_CONTEXT):
        raise ValueError("saved CPU receipt differs from live exact Job and Pod")
    if not re.fullmatch(r"[a-f0-9]{64}", reviewed_manifest_sha256):
        raise ValueError("reviewed server manifest SHA-256 required")
    journal = directory / "GPU-POST-INTENT.jsonl"
    if journal.exists() or journal.is_symlink():
        raise ValueError("GPU POST intent already exists; reconcile, never retry")
    holder = f"{request['name']}-{uuid.uuid4()}"
    acquired = _lease(holder)
    lease_uid = acquired["metadata"]["uid"]
    started = time.time()
    try:
        counts = capacity()
        proof = _legacy("preview", {"request": request})
        if proof["manifest_sha256"] != reviewed_manifest_sha256:
            raise ValueError("reviewed GPU server render changed")
        if time.time() - started > 600 or time.time() - cpu["observed_at_unix"] > 1800:
            raise ValueError("capacity lease is too old to submit")
        result = _legacy("submit", {"request": request, "journal": str(journal),
                                    "reviewed_manifest_sha256": reviewed_manifest_sha256})
    finally:
        if not journal.exists() or "result" in locals():
            _lease(None, expected_uid=lease_uid, expected_holder=holder)
    return {"status": "submitted_once", "run": result, "capacity_before": counts,
            "root_failure_alerts": "off", "priority": "c1/q1"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare", help="compile a create-once exact request off-GPU")
    p.add_argument("config", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--mechanics-proof", nargs=6, type=Path,
                   metavar=("SOURCE", "FULL", "LEGACY", "CHILD", "SELECTION", "SUBSET"))
    for action in ("cpu-preview", "cpu-collect", "preview"):
        p = commands.add_parser(action)
        p.add_argument("prepared_directory", type=Path)
    for action in ("cpu-create", "submit"):
        p = commands.add_parser(action)
        p.add_argument("prepared_directory", type=Path)
        p.add_argument("reviewed_sha256")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.config, args.destination, args.mechanics_proof)
        elif args.action == "cpu-preview":
            result = cpu_preview(args.prepared_directory, PROD_CONTEXT)
        elif args.action == "cpu-create":
            result = cpu_create(args.prepared_directory, PROD_CONTEXT, args.reviewed_sha256)
        elif args.action == "cpu-collect":
            result = cpu_collect(args.prepared_directory, PROD_CONTEXT)
        elif args.action == "submit":
            result = submit(args.prepared_directory, args.reviewed_sha256)
        else:
            result = preview(args.prepared_directory)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"{args.action} rejected: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
