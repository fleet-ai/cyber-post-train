"""Freeze public HF metadata, not weights or model compatibility claims."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx

from .io import digest_json

TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
}
CONFIG_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
}
MAX_METADATA_BYTES = 64 * 1024 * 1024


def _path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("noncanonical repository path")
    return value


def _lfs(row: dict) -> str:
    lfs = row.get("lfs", {})
    if (
        type(row.get("size")) is not int
        or row["size"] <= 0
        or lfs.get("size") != row["size"]
        or not re.fullmatch(r"[a-f0-9]{64}", str(lfs.get("oid", "")))
    ):
        raise ValueError("missing or invalid LFS payload identity")
    return lfs["oid"]


def _download(client: httpx.Client, url: str, row: dict) -> bytes:
    size = row.get("size")
    if type(size) is not int or not 0 <= size <= MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds download bound")
    content = bytearray()
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > size:
                raise ValueError("metadata response exceeds declared size")
    if len(content) != size:
        raise ValueError("metadata response is truncated")
    if "lfs" in row:
        valid = hashlib.sha256(content).hexdigest() == _lfs(row)
    else:
        # HF's non-LFS oid is a Git blob SHA-1, not a file SHA-256.
        header = f"blob {size}\0".encode()
        valid = hashlib.sha1(header + content).hexdigest() == row.get("oid")
    if not valid:
        raise ValueError("download differs from exact-revision inventory")
    return bytes(content)


def freeze(repo: str, revision: str, client: httpx.Client) -> tuple[dict, dict]:
    """Pin indexed safetensors and standard runtime sidecars at a full Git SHA.

    Only public, native-Transformers repositories are supported here. This does
    not stage weights, execute remote code, qualify a loader, or submit a job.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("expected an owner/model repository")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("model revision must be an exact 40-character Git SHA")
    tree = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    source = tree + "?recursive=true&limit=1000"
    url, seen, inventory = source, set(), {}
    while url:
        parts = urlsplit(url)
        if (
            (parts.scheme, parts.netloc, parts.path)
            != ("https", "huggingface.co", urlsplit(tree).path)
            or url in seen
            or len(seen) >= 100
        ):
            raise ValueError("invalid or repeated HF pagination link")
        seen.add(url)
        response = client.get(url)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("HF tree response is not an inventory")
        for row in rows:
            if row.get("type") == "directory":
                continue
            if row.get("type") != "file":
                raise ValueError("unsupported repository entry")
            path = _path(row["path"])
            if path in inventory:
                raise ValueError("duplicate repository path across pages")
            inventory[path] = row
        url = response.links.get("next", {}).get("url")

    resolve = f"https://huggingface.co/{repo}/resolve/{revision}/"
    index_path = "model.safetensors.index.json"
    selected = (TOKENIZER_FILES | CONFIG_FILES | {index_path}) & inventory.keys()
    if not {index_path, "config.json", "tokenizer_config.json"} <= selected:
        raise ValueError("missing indexed weights/configuration/tokenizer metadata")
    if (
        not {
            "tokenizer.json",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "vocab.txt",
            "vocab.json",
        }
        & selected
    ):
        raise ValueError("no supported tokenizer payload")
    payloads = {p: _download(client, resolve + p, inventory[p]) for p in sorted(selected)}
    index = json.loads(payloads[index_path])
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("empty or invalid safetensors index")
    files = []
    for path in sorted(set(weight_map.values())):
        if not _path(path).endswith(".safetensors") or path not in inventory:
            raise ValueError("index points outside the weight inventory")
        row = inventory[path]
        files.append({"path": path, "size": row["size"], "sha256": _lfs(row)})
    tokenizer_config = json.loads(payloads["tokenizer_config.json"])
    if "chat_template.jinja" not in selected and not tokenizer_config.get("chat_template"):
        raise ValueError("missing chat template")
    tokenizer = [
        {"path": p, "sha256": hashlib.sha256(payloads[p]).hexdigest()}
        for p in sorted(TOKENIZER_FILES & selected)
    ]
    weights = {
        "schema": "huggingface_weight_manifest_v1",
        "repo": repo,
        "revision": revision,
        "source": source,
        "files": files,
    }
    lock = {
        "schema": "huggingface_model_lock_v1",
        "repo": repo,
        "revision": revision,
        "source_api": source,
        "weights": {
            "format": "safetensors",
            "shards": len(files),
            "bytes": sum(f["size"] for f in files),
            "manifest_sha256": digest_json(files),
            "index_sha256": "sha256:" + hashlib.sha256(payloads[index_path]).hexdigest(),
        },
        "tokenizer": {"manifest_sha256": digest_json(tokenizer), "files": tokenizer},
        "configuration": {
            p.removesuffix(".json") + "_sha256": "sha256:" + hashlib.sha256(payloads[p]).hexdigest()
            for p in sorted(CONFIG_FILES & selected)
        },
    }
    return lock, weights
