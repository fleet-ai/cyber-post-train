"""Fail-closed probes for the exact GLM-5.2 training environment.

This is deliberately split from the expensive model test.  It proves checkpoint
identity, tokenizer/template identity, framework imports, GPU identity, and NCCL
collectives before a job is allowed to spend hours loading 1.5 TB of weights.
It never promotes the full compatibility receipt on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json

MODEL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
WEIGHTS_MANIFEST_SHA256 = "sha256:72b917aa2a1664751f99c4f19b1f1b23acb5eff5cffef4ff5522caef767324d4"
CHAT_TEMPLATE_SHA256 = "sha256:172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"
VERIFIED_SHARDS = 282


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_checkpoint_lock(model_root: Path) -> dict[str, Any]:
    lock_path = model_root / ".cyber-post-train-lock.json"
    lock = json.loads(lock_path.read_text())
    expected = {
        "schema": "cyber_post_train_checkpoint_lock_v1",
        "repo": "zai-org/GLM-5.2",
        "revision": MODEL_REVISION,
        "weights_manifest_sha256": WEIGHTS_MANIFEST_SHA256,
        "verified_shards": VERIFIED_SHARDS,
    }
    mismatches = {
        key: {"expected": value, "actual": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint lock mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if sha256_file(model_root / "chat_template.jinja") != CHAT_TEMPLATE_SHA256:
        raise ValueError("chat_template.jinja digest mismatch")
    return lock


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()


def static_probe(model_root: Path) -> dict[str, Any]:
    lock = validate_checkpoint_lock(model_root)
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    importlib.import_module("nemo_rl")
    importlib.import_module("megatron.core")
    importlib.import_module("vllm")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_root), local_files_only=True, trust_remote_code=True
    )
    conversation = [
        {"role": "system", "content": "You are a compatibility probe."},
        {"role": "user", "content": "Reply with the word ready."},
    ]
    rendered = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )
    if not rendered or not token_ids:
        raise RuntimeError("tokenizer chat-template roundtrip produced empty output")

    cuda_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or cuda_count != 8:
        raise RuntimeError(f"expected exactly 8 visible GPUs, found {cuda_count}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(cuda_count)]
    capabilities = [list(torch.cuda.get_device_capability(index)) for index in range(cuda_count)]
    if any(capability != [10, 3] for capability in capabilities):
        raise RuntimeError(f"expected B300 compute capability [10, 3], found {capabilities}")

    return {
        "schema": "glm52_environment_preflight_v1",
        "model": {
            "root": str(model_root),
            "revision": lock["revision"],
            "weights_manifest_sha256": lock["weights_manifest_sha256"],
            "verified_shards": lock["verified_shards"],
            "verified_bytes": lock["verified_bytes"],
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "rendered_template_sha256": "sha256:" + hashlib.sha256(rendered.encode()).hexdigest(),
            "roundtrip_token_count": len(token_ids),
        },
        "runtime": {
            "image_digest": os.environ.get("TRAINING_IMAGE_DIGEST"),
            "nemo_rl_revision": os.environ.get("NEMO_RL_COMMIT"),
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in (
                    "nemo-rl",
                    "megatron-core",
                    "torch",
                    "transformers",
                    "vllm",
                )
            },
        },
        "gpu": {
            "count": cuda_count,
            "names": gpu_names,
            "compute_capabilities": capabilities,
            "nvidia_smi_topology": _run(["nvidia-smi", "topo", "-m"]),
        },
        "checks": {
            "checkpoint_lock": True,
            "framework_imports": True,
            "tokenizer_chat_template_roundtrip": True,
            "b300_identity": True,
        },
    }


def collective_probe() -> dict[str, Any] | None:
    torch = importlib.import_module("torch")
    distributed = torch.distributed
    distributed.init_process_group(backend="nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    reduced = torch.tensor([float(rank)], device=device)
    distributed.all_reduce(reduced)
    expected_sum = world_size * (world_size - 1) / 2
    if reduced.item() != expected_sum:
        raise RuntimeError(f"NCCL all-reduce mismatch: {reduced.item()} != {expected_sum}")

    sent = torch.tensor(
        [rank * world_size + destination for destination in range(world_size)],
        dtype=torch.int64,
        device=device,
    )
    received = torch.empty_like(sent)
    distributed.all_to_all_single(received, sent)
    expected = torch.tensor(
        [source * world_size + rank for source in range(world_size)],
        dtype=torch.int64,
        device=device,
    )
    if not torch.equal(received, expected):
        raise RuntimeError("NCCL all-to-all mismatch")
    distributed.barrier()
    result = None
    if rank == 0:
        result = {
            "schema": "glm52_nccl_preflight_v1",
            "world_size": world_size,
            "checks": {"nccl_all_reduce": True, "nccl_all_to_all": True},
        }
    distributed.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("static", "collective"))
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "static":
        if args.model_root is None:
            parser.error("--model-root is required in static mode")
        result = static_probe(args.model_root)
    else:
        result = collective_probe()
        if result is None:
            return
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
