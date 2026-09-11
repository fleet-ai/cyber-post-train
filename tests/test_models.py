"""Public immutable metadata, with real HTTP parsing and no weight downloads."""

import copy
import hashlib
import json

import httpx
import pytest

from training import models
from training.io import digest_json

REPO, REVISION = "Example/Model", "a" * 40
TREE = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}"


@pytest.fixture
def remote():
    payloads = {
        "config.json": b'{"model_type":"synthetic"}',
        "generation_config.json": b"{}",
        "tokenizer_config.json": b'{"chat_template":"synthetic"}',
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": b'{"weight_map":{"w":"model-1.safetensors"}}',
    }
    rows = [
        {
            "type": "file",
            "path": p,
            "size": len(b),
            "oid": hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest(),
        }
        for p, b in payloads.items()
    ]
    weight = {
        "type": "file",
        "path": "model-1.safetensors",
        "size": 512,
        "lfs": {"oid": "b" * 64, "size": 512},
    }
    rows.extend([weight, {"type": "directory", "path": "unused"}])
    state = {"rows": rows, "payloads": payloads, "requests": [], "pages": None}

    def handle(request):
        state["requests"].append(str(request.url))
        assert request.method == "GET"
        assert "authorization" not in request.headers
        assert not request.url.path.endswith(".safetensors"), "weights must not download"
        if "/tree/" in request.url.path:
            if state["pages"]:
                page = 1 if request.url.params.get("cursor") else 0
                data, next_link = state["pages"][page]
                headers = {"link": f'<{next_link}>; rel="next"'} if next_link else {}
                return httpx.Response(200, json=data, headers=headers)
            return httpx.Response(200, json=state["rows"])
        return httpx.Response(200, content=state["payloads"][request.url.path.rsplit("/", 1)[1]])

    state["client"] = httpx.Client(transport=httpx.MockTransport(handle))
    yield state
    state["client"].close()


def freeze(remote):
    return models.freeze(REPO, REVISION, remote["client"])


def test_pins_payload_hashes_not_git_blob_ids_and_never_downloads_weights(remote):
    lock, weights = freeze(remote)
    assert lock["weights"]["bytes"] == 512
    assert lock["weights"]["shards"] == 1
    assert lock["weights"]["manifest_sha256"] == digest_json(weights["files"])
    assert weights["files"] == [{"path": "model-1.safetensors", "size": 512, "sha256": "b" * 64}]
    assert lock["configuration"]["config_sha256"] == (
        "sha256:" + hashlib.sha256(remote["payloads"]["config.json"]).hexdigest()
    )
    assert lock["tokenizer"]["manifest_sha256"] == digest_json(lock["tokenizer"]["files"])
    assert (weights["repo"], weights["revision"]) == (REPO, REVISION)
    assert freeze(remote) == (lock, weights)


def test_pages_complete_inventory_and_accepts_lfs_metadata(remote):
    row = next(r for r in remote["rows"] if r["path"] == "tokenizer.json")
    row["lfs"] = {"size": row["size"], "oid": hashlib.sha256(b"{}").hexdigest()}
    remote["pages"] = [(remote["rows"][:2], TREE + "?cursor=second"), (remote["rows"][2:], None)]
    assert freeze(remote)[0]["weights"]["shards"] == 1
    assert TREE + "?cursor=second" in remote["requests"]


@pytest.mark.parametrize(
    "repo,revision",
    [("../escape", REVISION), ("a/b/c", REVISION), (REPO, "main"), (REPO, "a" * 39)],
)
def test_rejects_mutable_or_unsafe_identity_before_network(remote, repo, revision):
    with pytest.raises(ValueError):
        models.freeze(repo, revision, remote["client"])
    assert not remote["requests"]


@pytest.mark.parametrize(
    "link",
    [
        "https://other.invalid/",
        TREE.replace(REVISION, "main"),
        TREE + "?recursive=true&limit=1000",
        "http://huggingface.co/x",
    ],
)
def test_pagination_cannot_escape_identity_or_cycle(remote, link):
    remote["pages"] = [([], link)]
    with pytest.raises(ValueError, match="pagination"):
        freeze(remote)
    assert len(remote["requests"]) == 1


def test_pagination_is_bounded():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json=[], headers={"link": f'<{TREE}?cursor={len(calls)}>; rel="next"'}
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client, pytest.raises(ValueError):
        models.freeze(REPO, REVISION, client)
    assert len(calls) == 100


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "symlink", "path": "model"},
        {"type": "file", "path": "../evil"},
        {"type": "file", "path": "/evil"},
        {"type": "file", "path": "a//b"},
        {"type": "file", "path": ""},
        {"type": "file", "path": "config.json"},
    ],
)
def test_invalid_or_duplicate_entries_rejected(remote, entry):
    remote["rows"].append(entry)
    with pytest.raises(ValueError):
        freeze(remote)


@pytest.mark.parametrize(
    "path",
    ["config.json", "tokenizer_config.json", "model.safetensors.index.json", "tokenizer.json"],
)
def test_missing_required_files_rejected(remote, path):
    remote["rows"] = [r for r in remote["rows"] if r["path"] != path]
    with pytest.raises(ValueError):
        freeze(remote)


@pytest.mark.parametrize(
    "change",
    [
        {"size": True},
        {"size": 0},
        {"size": -1},
        {"lfs": {"oid": "b" * 64, "size": 511}},
        {"lfs": {"oid": "invalid", "size": 512}},
        {"lfs": {}},
    ],
)
def test_weight_manifest_requires_real_lfs_identity(remote, change):
    next(r for r in remote["rows"] if r["path"] == "model-1.safetensors").update(change)
    with pytest.raises(ValueError, match="LFS"):
        freeze(remote)


@pytest.mark.parametrize("size", [True, -1, models.MAX_METADATA_BYTES + 1])
def test_metadata_download_bound_checked_before_transfer(remote, size):
    remote["rows"][0]["size"] = size
    with pytest.raises(ValueError, match="bound"):
        freeze(remote)


@pytest.mark.parametrize("data", [b"", b"x", b"x" * 80])
def test_truncated_tampered_or_oversized_downloads_rejected(remote, data):
    remote["payloads"]["config.json"] = data
    with pytest.raises(ValueError):
        freeze(remote)


def replace_payload(remote, name, payload):
    remote["payloads"][name] = payload
    remote["rows"] = [r for r in remote["rows"] if r["path"] != name]
    remote["rows"].append(
        {
            "type": "file",
            "path": name,
            "size": len(payload),
            "oid": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
        }
    )


@pytest.mark.parametrize(
    "index",
    [
        {},
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"w": "missing.safetensors"}},
        {"weight_map": {"w": "config.json"}},
    ],
)
def test_index_must_select_existing_safetensors(remote, index):
    replace_payload(remote, "model.safetensors.index.json", json.dumps(index).encode())
    with pytest.raises(ValueError):
        freeze(remote)


def test_chat_template_must_exist_in_config_or_separate_file(remote):
    replace_payload(remote, "tokenizer_config.json", b"{}")
    with pytest.raises(ValueError, match="chat template"):
        freeze(remote)
    replace_payload(remote, "chat_template.jinja", b"synthetic template")
    assert "chat_template.jinja" in {r["path"] for r in freeze(remote)[0]["tokenizer"]["files"]}


def test_bad_git_blob_or_lfs_metadata_digest_rejected(remote):
    row = remote["rows"][0]
    original = copy.deepcopy(row)
    row["oid"] = "c" * 40
    with pytest.raises(ValueError, match="differs"):
        freeze(remote)
    row.update(original)
    row["lfs"] = {"size": row["size"], "oid": "c" * 64}
    with pytest.raises(ValueError, match="differs"):
        freeze(remote)


@pytest.mark.parametrize("status,payload", [(403, {}), (429, {}), (500, {}), (200, {})])
def test_http_and_malformed_inventory_fail_without_retry(status, payload):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json=payload)

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as client,
        pytest.raises((httpx.HTTPStatusError, ValueError)),
    ):
        models.freeze(REPO, REVISION, client)
    assert len(requests) == 1
