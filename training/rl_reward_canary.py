"""Fail-closed source and submission gate for the Qwen3.8 SkyRL reward canary.

The source branch's preflight is retained as provenance, not as certification
of these rebased bytes.  This module reopens the immutable source package
before data acquisition or launch preparation and carries an explicit blocked
submission gate into the prepared plan.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "qwen38_skyrl_reward_canary_v3"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
TASK_SET_PATH = "configs/data/qwen38-rl-reward-canary-task-set-v3.json"
SPLIT_PATH = "configs/data/qwen38-rl-reward-canary-split-v1.json"
EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v3.json"
HORIZON_PATH = "configs/data/qwen38-rl-reward-canary-horizon-v1.json"
TOOL_CATALOG_PATH = "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
QUALIFICATION_PATH = "configs/qualification/qwen38-rl-reward-canary-port-v3.json"

TASK_SET_FILE_SHA256 = "sha256:c3ee0a239dcb836b1951e65eda5a9813360a0378747716df8d8ca3753e741517"
TASK_SET_SELF_SHA256 = "sha256:9b88bac1510720926ea041d942044804d6c54c083a0ea56d07cae5f448685909"
SPLIT_FILE_SHA256 = "sha256:c3a34993a932c2b29e2c70f45682ce03ee02d03a9c4493edbab45eb349cb64fa"
SPLIT_SELF_SHA256 = "sha256:8279ea19808ad1accb00d3f3145c3ec087030677786e0188251cfc197f306adf"
EVIDENCE_FILE_SHA256 = "sha256:73be894762968970c8cf0454281fe13d46615e40039b61b8f8a97c8cc883f365"
EVIDENCE_SELF_SHA256 = "sha256:50491796b829388164faefc5fb9a9ad09fac89a8d94648a1791240f91826a285"
HORIZON_FILE_SHA256 = "sha256:435866421cc4cdcc4a5598c92112fadfc5c01634c076334df57d6d46b2fe7b85"
HORIZON_SELF_SHA256 = "sha256:a4e861dedcf0538633e15e1762bca9f37156dfc392dda70a8fc50df06020076e"
TOOL_CATALOG_FILE_SHA256 = "sha256:e4a3c4fb5b5c34cdaf64ec568eb31fcc0d55a63cc0a134808db348c65d7b6858"
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
QUALIFICATION_FILE_SHA256 = (
    "sha256:b44deb720d961f1acb43d83834f97bd796d61f79d55bb8aa0ebb9b1abd22e766"
)
QUALIFICATION_SELF_SHA256 = (
    "sha256:cbd2d9b29de72bf61654b8e1ba7d35d9c66a05642dd970bb86ee7384dbec9827"
)

LIMITS = {
    "context_tokens": 98304,
    "response_tokens": 81920,
    "max_tokens_per_turn": 4096,
    "max_turns": 600,
    "episode_seconds": 2400,
    "tool_seconds": 330,
    "tool_result_chars": 50000,
}
RECIPE = {
    "nodes": 1,
    "steps": 1,
    "groups": 1,
    "samples_per_prompt": 8,
    "lr": 1e-6,
    "eval_interval": 1,
    "checkpoint_interval": 1,
    "keep_checkpoints": 2,
    "seed": 42,
}
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}
MODEL = {
    "repo": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    "weight_manifest_sha256": (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    ),
}
TASKS = (
    {
        "task_key": ("cysec1-2-current-gen_blackbox-0708aea62e0d8ccf09331568__blackbox_ctf_v1"),
        "task_version_id": "54425601-6fd2-43d8-8cb9-e565b767676a",
        "env_key": "cysec1-2-current-gen",
        "env_version": "v0.0.3",
        "environment_version_id": "d70c4fe9-70c5-4020-91b1-a23d886a1e22",
        "data_key": "commercial",
        "data_version": "v0.0.9",
        "verifier_version_id": "14c6b6dd-f189-416c-b34d-29bf55ff1b87",
        "split": "train",
    },
    {
        "task_key": ("cysec1-2-fakelook-gen_blackbox-7317189e8fefb9033c5be097__blackbox_ctf_v1"),
        "task_version_id": "2fdc9511-f5fe-4386-a4a5-a5a16a30f359",
        "env_key": "cysec1-2-fakelook-gen",
        "env_version": "v0.0.3",
        "environment_version_id": "a8f554e8-2d1b-4437-aa2c-b140edc781c4",
        "data_key": "fakelook-cyber-gen",
        "data_version": "v0.0.2",
        "verifier_version_id": "928e5c13-b15a-412d-a037-fbed7389572a",
        "split": "dev",
    },
)
RUNTIME_SOURCES = {
    "training/rl_data.py": (
        "sha256:f0c327d2ecc464610a5fd86e11b112870d28feb43d9155d7507d8fedbd87632b"
    ),
    "training/rl_episode.py": (
        "sha256:8bb64464bad854ab477bec1149bc0d53e4e656f1292b8a47ad31687446ba1dfe"
    ),
    "evals/fleet/opencode_self_hosted.py": (
        "sha256:428e9f2e4d4c758c4f682cbed314cf96b866d051b34fa0eea8757a3aad97aa1d"
    ),
}
PORT_COMMITS = {
    "runtime_commit": "8b5e8a00521b4fe412e3e7d0bce9b8ea6855b005",
    "preflight_commit": "6c0bc541e136fc114a3f7a17ce6ba07e9178bb78",
    "ported_onto_commit": "607b7b0bb30d798a3af1ba5f17e1ebc9f2366a77",
}
SUBMISSION_BLOCKERS = [
    "fresh_main_based_cpu_preflight_and_preview_not_yet_recorded",
    "fresh_jobs_sfs_wandb_identity_absence_not_yet_proven",
    "dev_api_target_not_bound_by_current_main_jobs_client",
    "preserved_2400_second_episode_conflicts_with_30_minute_dev_deadline_policy",
    "one_by_eight_training_topology_not_qualified_by_two_by_four_engine_receipt",
]


def _sealed(value: dict, schema: str) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(body):
        raise ValueError("reward canary evidence digest mismatch")


def _path(path: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("reward canary evidence path escapes or is unavailable")
    return resolved


def _bound_bytes(path: Path, expected_sha256: str) -> bytes:
    source = _path(path)
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("reward canary evidence changed while being read")
    if fleet.sha256(payload) != expected_sha256:
        raise ValueError("reward canary evidence file digest mismatch")
    return payload


def _bound_json(path: Path, expected_sha256: str) -> dict:
    try:
        value = json.loads(_bound_bytes(path, expected_sha256))
    except (TypeError, ValueError) as error:
        raise ValueError("reward canary evidence is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("reward canary evidence must be an object")
    return value


def _binding(base: Path, value: object, *, schema: str) -> dict:
    required = {"path", "file_sha256", "self_sha256"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("reward canary evidence binding is incomplete")
    result = _bound_json(base / value["path"], value["file_sha256"])
    _sealed(result, schema)
    if result["sha256"] != value["self_sha256"]:
        raise ValueError("reward canary evidence self digest mismatch")
    return result


def _validate_receipt_package(evidence: dict, base: Path) -> None:
    expected = (
        (
            "eligible_source",
            evidence["eligible_source"],
            TASKS[0]["task_version_id"],
            None,
        ),
        (
            "metadata_only_successor",
            evidence["rejected_metadata_only_successor"],
            "c99340e2-3801-5c3c-a50c-3b96cee4572f",
            ["bash", "submit_report"],
        ),
    )
    for role, parent, version_id, metadata_tools in expected:
        binding = parent["version_receipt"]
        receipt = _binding(
            base,
            binding,
            schema="cyber_rl_reward_canary_task_version_receipt_v2",
        )
        observation = _binding(
            base,
            binding["authority_observation"],
            schema="cyber_rl_reward_canary_task_version_get_observation_v1",
        )
        response = _binding(
            base,
            observation["sanitized_response"],
            schema="cyber_rl_reward_canary_sanitized_task_version_response_v1",
        )
        journal = _binding(
            base,
            observation["acquisition_journal"],
            schema="cyber_rl_reward_canary_task_version_get_journal_v1",
        )
        request = {
            "method": "GET",
            "url": f"https://orchestrator.fleetai.com/v1/tasks/{TASKS[0]['task_key']}",
            "query": {"version_id": version_id},
        }
        if (
            receipt["role"] != role
            or receipt["task_key"] != TASKS[0]["task_key"]
            or receipt["task_version_id"] != version_id
            or receipt["environment_version_id"] != TASKS[0]["environment_version_id"]
            or receipt["verifier_version_id"] != TASKS[0]["verifier_version_id"]
            or receipt["metadata_tools"] != metadata_tools
            or observation["request"] != request
            or observation["status_code"] != 200
            or response["task_version_id"] != version_id
            or response["environment_version_id"] != receipt["environment_version_id"]
            or response["verifier_version_id"] != receipt["verifier_version_id"]
            or journal["request"] != request
            or journal["status_code"] != 200
            or journal["response_body_sha256"] != observation["response_body_sha256"]
        ):
            raise ValueError("reward canary exact-version package changed")


def validate_source_package(
    task_set_path: Path,
    split_path: Path,
    tool_catalog_path: Path,
    limits: object,
) -> dict:
    """Reopen the exact task, version, tool and horizon package."""
    if (
        task_set_path.resolve() != (ROOT / TASK_SET_PATH).resolve()
        or split_path.resolve() != (ROOT / SPLIT_PATH).resolve()
        or tool_catalog_path.resolve() != (ROOT / TOOL_CATALOG_PATH).resolve()
        or limits != LIMITS
    ):
        raise ValueError("reward canary source selection or horizon changed")
    task_set = _bound_json(task_set_path, TASK_SET_FILE_SHA256)
    split = _bound_json(split_path, SPLIT_FILE_SHA256)
    evidence = _bound_json(ROOT / EVIDENCE_PATH, EVIDENCE_FILE_SHA256)
    horizon = _bound_json(ROOT / HORIZON_PATH, HORIZON_FILE_SHA256)
    _sealed(task_set, "cyber_rl_task_set_v1")
    _sealed(split, "cyber_task_split_v1")
    _sealed(evidence, "cyber_rl_exact_version_evidence_v1")
    _sealed(horizon, "cyber_rl_reward_canary_horizon_v1")
    if (
        task_set["sha256"] != TASK_SET_SELF_SHA256
        or split["sha256"] != SPLIT_SELF_SHA256
        or evidence["sha256"] != EVIDENCE_SELF_SHA256
        or horizon["sha256"] != HORIZON_SELF_SHA256
    ):
        raise ValueError("reward canary source self identity changed")

    from .rl_data import selection

    selected = selection(task_set, split)
    expected_selected = [
        {
            **{
                key: value
                for key, value in task.items()
                if key not in {"split", "verifier_version_id"}
            },
            "lineage": task_set["tasks"][i]["lineage"],
            "split": task["split"],
        }
        for i, task in reversed(tuple(enumerate(TASKS)))
    ]
    if selected != expected_selected:
        raise ValueError("reward canary exact train/dev selection changed")

    catalog = json.loads(_bound_bytes(tool_catalog_path, TOOL_CATALOG_FILE_SHA256))
    if (
        [item.get("name") for item in catalog] != ["bash", "submit_report"]
        or fleet.sha256(fleet.canonical_json(catalog)) != TOOL_CATALOG_SHA256
        or task_set["tool_catalog_sha256"] != TOOL_CATALOG_SHA256
    ):
        raise ValueError("reward canary ordered tool catalog changed")

    evidence_binding = task_set["reward_signal_provenance"]["exact_version_evidence"]
    if evidence_binding != {
        "path": "../data/qwen38-rl-reward-canary-exact-version-evidence-v3.json",
        "file_sha256": EVIDENCE_FILE_SHA256,
        "self_sha256": EVIDENCE_SELF_SHA256,
    }:
        raise ValueError("reward canary exact-version evidence binding changed")
    if (
        evidence["selected_version"]["task_key"] != TASKS[0]["task_key"]
        or evidence["selected_version"]["task_version_id"] != TASKS[0]["task_version_id"]
        or evidence["rejected_metadata_only_successor"]["successor_task_version_id"]
        != "c99340e2-3801-5c3c-a50c-3b96cee4572f"
        or evidence["port_provenance"]
        != {
            "runtime_commit": PORT_COMMITS["runtime_commit"],
            "preflight_commit": PORT_COMMITS["preflight_commit"],
            "historical_preflight_is_gating": False,
        }
    ):
        raise ValueError("reward canary exact-version decision changed")
    _validate_receipt_package(evidence, (ROOT / EVIDENCE_PATH).parent)

    manifest_binding = evidence["eligible_source"]
    eligible = _bound_json(
        (ROOT / EVIDENCE_PATH).parent / manifest_binding["manifest_path"],
        manifest_binding["manifest_file_sha256"],
    )
    if "sha256:" + eligible["sha256"] != manifest_binding["manifest_self_sha256"]:
        raise ValueError("reward canary eligible inventory identity changed")
    inventory = {
        (row["task_key"], row["task_version_id"]): row
        for row in eligible["task_versions"]
        if (row["task_key"], row["task_version_id"])
        in {(task["task_key"], task["task_version_id"]) for task in TASKS}
    }
    for task in TASKS:
        row = inventory.get((task["task_key"], task["task_version_id"]), {})
        environment = row.get("environment", {})
        if (
            environment.get("id") != task["env_key"]
            or environment.get("version") != task["env_version"]
            or environment.get("version_id") != task["environment_version_id"]
            or environment.get("data_id") != task["data_key"]
            or environment.get("data_version") != task["data_version"]
            or row.get("verifier", {}).get("version_id") != task["verifier_version_id"]
        ):
            raise ValueError("reward canary task runtime or verifier identity changed")

    authority = evidence["tool_surface_authority"]
    local_code = authority["local_code"]
    for name, path in (
        ("data_preparation", "training/rl_data.py"),
        ("episode_runtime", "training/rl_episode.py"),
        ("fleet_binding", "evals/fleet/opencode_self_hosted.py"),
    ):
        if (
            local_code[name]["file_sha256"] != RUNTIME_SOURCES[path]
            or fleet.sha256((ROOT / path).read_bytes()) != RUNTIME_SOURCES[path]
        ):
            raise ValueError("reward canary tool enforcement source changed")
    if (
        authority["ordered_tools"] != ["bash", "submit_report"]
        or authority["tool_catalog_sha256"] != TOOL_CATALOG_SHA256
        or authority["task_metadata_tools_required"] is not False
        or authority["reviewed_server_route"]["metadata_tools_read"] is not False
    ):
        raise ValueError("reward canary tool authority changed")

    horizon_binding = evidence["episode_horizon"]
    if horizon_binding != {
        "max_turns": 600,
        "contract_path": "qwen38-rl-reward-canary-horizon-v1.json",
        "contract_file_sha256": HORIZON_FILE_SHA256,
        "contract_self_sha256": HORIZON_SELF_SHA256,
        "aggregate_evidence_path": "../../docs/FLEET_NATIVE_HARNESS_PARITY.md",
        "aggregate_evidence_file_sha256": (
            "sha256:d6846aa328d4a915464af062b21032d5ea2a0e09bf0dd98d0050a2d498ffa632"
        ),
        "rationale": horizon_binding["rationale"],
        "fixed_limits": {key: value for key, value in LIMITS.items() if key != "max_turns"},
    }:
        raise ValueError("reward canary horizon binding changed")
    if (
        horizon["task_key"] != TASKS[0]["task_key"]
        or horizon["task_version_id"] != TASKS[0]["task_version_id"]
        or horizon["required_task_tools"] != ["bash", "submit_report"]
        or horizon["limits"] != LIMITS
    ):
        raise ValueError("reward canary dedicated horizon changed")
    aggregate = _bound_bytes(
        ROOT / "docs/FLEET_NATIVE_HARNESS_PARITY.md",
        horizon_binding["aggregate_evidence_file_sha256"],
    )
    if b"source Fleet job allowed 600 agent steps" not in aggregate or b"p90 154" not in aggregate:
        raise ValueError("reward canary aggregate horizon evidence changed")
    return source_proof()


def source_proof() -> dict:
    proof = {
        "schema": "cyber_rl_reward_canary_source_closure_v3",
        "profile": PROFILE,
        "task_set_file_sha256": TASK_SET_FILE_SHA256,
        "task_set_self_sha256": TASK_SET_SELF_SHA256,
        "split_file_sha256": SPLIT_FILE_SHA256,
        "split_self_sha256": SPLIT_SELF_SHA256,
        "evidence_file_sha256": EVIDENCE_FILE_SHA256,
        "evidence_self_sha256": EVIDENCE_SELF_SHA256,
        "horizon_file_sha256": HORIZON_FILE_SHA256,
        "horizon_self_sha256": HORIZON_SELF_SHA256,
        "tool_catalog_sha256": TOOL_CATALOG_SHA256,
        "limits": dict(LIMITS),
        "required_task_tools": ["bash", "submit_report"],
    }
    proof["sha256"] = "sha256:" + digest(proof)
    return proof


def validate_data_config(config: dict, *, relative_to: Path) -> dict | None:
    task_set_path = (relative_to / config["task_set"]).resolve()
    if task_set_path != (ROOT / TASK_SET_PATH).resolve():
        value = json.loads(task_set_path.read_bytes())
        if not isinstance(value, dict) or value.get("reward_signal_provenance") is None:
            return None
        raise ValueError("unknown reward-bearing task set; add a reviewed source closure")
    return validate_source_package(
        task_set_path,
        relative_to / config["split"],
        relative_to / config["tool_catalog"],
        config.get("limits"),
    )


def build(config: dict, *, relative_to: Path, client) -> dict:
    """Validate reward inputs before the generic builder makes authenticated GETs."""
    if "task_set" in config:
        validate_data_config(config, relative_to=relative_to)
    from .rl_data import build as generic_build

    return generic_build(config, relative_to=relative_to, client=client)


def _validate_qualification(path: Path) -> dict:
    if path.resolve() != (ROOT / QUALIFICATION_PATH).resolve():
        raise ValueError("unknown reward-canary qualification closure")
    value = _bound_json(path, QUALIFICATION_FILE_SHA256)
    _sealed(value, "cyber_qwen38_skyrl_reward_canary_port_v3")
    if (
        value["sha256"] != QUALIFICATION_SELF_SHA256
        or value["source"] != PORT_COMMITS
        or value["execution"]
        != {
            "cluster_target": "dev",
            "image": IMAGE,
            "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
        or value["source_closure"]
        != {
            "task_set": "../data/qwen38-rl-reward-canary-task-set-v3.json",
            "split": "../data/qwen38-rl-reward-canary-split-v1.json",
            "horizon": "../data/qwen38-rl-reward-canary-horizon-v1.json",
            "tool_catalog": "../data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
        }
        or value["submission_gate"]
        != {
            "preview_authorized": False,
            "submission_authorized": False,
            "blockers": SUBMISSION_BLOCKERS,
        }
    ):
        raise ValueError("reward-canary qualification closure changed")

    for binding in value["historical_evidence"].values():
        _bound_bytes(path.parent / binding["path"], binding["file_sha256"])
    historical = value["historical_evidence"]
    preflight = _bound_json(
        path.parent / historical["source_preflight"]["path"],
        historical["source_preflight"]["file_sha256"],
    )
    image = _bound_json(
        path.parent / historical["image_cpu_qualification"]["path"],
        historical["image_cpu_qualification"]["file_sha256"],
    )
    engine = _bound_json(
        path.parent / historical["engine_qualification"]["path"],
        historical["engine_qualification"]["file_sha256"],
    )
    if (
        historical["source_preflight"]["classification"] != "historical_preflight_provenance"
        or historical["source_preflight"]["gating"] is not False
        or preflight["source"]["git_commit"] != PORT_COMMITS["runtime_commit"]
        or preflight["submission"]["submitted"] is not False
        or preflight["submission"]["gpu_allocation"] != 0
        or image["requested_image"] != IMAGE
        or image["status"] != "qualified"
        or engine["status"] != "accepted"
        or engine["inputs"]["image"] != IMAGE
        or (engine["inputs"]["workers"], engine["inputs"]["gpus_per_worker"]) != (2, 4)
        or engine["receipt"]["rollouts"] != 0
        or engine["receipt"]["optimizer_steps"] != 0
        or engine["release"]["active_gpus"] != 0
        or engine["release"]["gpu_release_proven"] is not True
    ):
        raise ValueError("reward-canary historical evidence scope changed")
    return value


def validate_run_config(
    config: dict,
    metadata: dict,
    bound_model: dict,
    *,
    relative_to: Path,
) -> dict:
    """Bind the exact v3 profile while keeping fresh identities in JSON only."""
    qualification = _validate_qualification(relative_to / config["qualification"])
    proof = validate_source_package(
        ROOT / TASK_SET_PATH,
        ROOT / SPLIT_PATH,
        ROOT / TOOL_CATALOG_PATH,
        metadata.get("limits"),
    )
    data = config["data"]
    output = PurePosixPath(config["output_root"])
    data_root = PurePosixPath(data["root"])
    if (
        metadata.get("selection_sha256") != TASK_SET_SELF_SHA256
        or metadata.get("split_sha256") != SPLIT_SELF_SHA256
        or metadata.get("tool_catalog_sha256") != TOOL_CATALOG_SHA256
        or {key: item["rows"] for key, item in metadata.get("files", {}).items()}
        != {"train": 1, "dev": 1}
        or config["recipe"] != RECIPE
        or config["cluster"] != {"priority": "c1", "resources": RESOURCES}
        or any(bound_model.get(key) != value for key, value in MODEL.items())
        or config["name"] != metadata.get("name")
        or config["name"] != config["wandb"]["run_id"]
        or config["wandb"]["entity"] != "thefleet"
        or config["wandb"]["project"] != "cyber-post-train"
        or output.name != config["name"]
        or data["manifest"] != str(data_root / "manifest.json")
        or output == data_root
        or output in data_root.parents
        or data_root in output.parents
    ):
        raise ValueError("reward-canary model, data, recipe, resource, or identity drift")
    result = {
        "schema": "cyber_qwen38_skyrl_reward_canary_plan_binding_v3",
        "profile": PROFILE,
        "source_proof": proof,
        "qualification_file_sha256": QUALIFICATION_FILE_SHA256,
        "qualification_self_sha256": QUALIFICATION_SELF_SHA256,
        "image": qualification["execution"]["image"],
        "environment": qualification["execution"]["environment"],
        "cluster_target": qualification["execution"]["cluster_target"],
        "submission_gate": qualification["submission_gate"],
    }
    result["sha256"] = "sha256:" + digest(result)
    return result


def validate_plan_binding(binding: object, metadata: dict, arguments: dict) -> dict:
    """Validate a compiler-produced binding without mutable source-tree reads."""
    if not isinstance(binding, dict):
        raise ValueError("reward-canary plan lacks its source binding")
    body = {key: item for key, item in binding.items() if key != "sha256"}
    if (
        binding.get("schema") != "cyber_qwen38_skyrl_reward_canary_plan_binding_v3"
        or binding.get("sha256") != "sha256:" + digest(body)
        or binding.get("profile") != PROFILE
        or binding.get("source_proof") != source_proof()
        or binding.get("qualification_file_sha256") != QUALIFICATION_FILE_SHA256
        or binding.get("qualification_self_sha256") != QUALIFICATION_SELF_SHA256
        or binding.get("image") != IMAGE
        or binding.get("environment") != {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
        or binding.get("cluster_target") != "dev"
        or binding.get("submission_gate")
        != {
            "preview_authorized": False,
            "submission_authorized": False,
            "blockers": SUBMISSION_BLOCKERS,
        }
        or metadata.get("selection_sha256") != TASK_SET_SELF_SHA256
        or metadata.get("split_sha256") != SPLIT_SELF_SHA256
        or metadata.get("limits") != LIMITS
        or {key: item["rows"] for key, item in metadata.get("files", {}).items()}
        != {"train": 1, "dev": 1}
        or {key: arguments[key] for key in RECIPE} != RECIPE
        or arguments.get("context_tokens") != LIMITS["context_tokens"]
        or arguments.get("response_tokens") != LIMITS["response_tokens"]
        or arguments.get("tokens_per_turn") != LIMITS["max_tokens_per_turn"]
        or arguments.get("max_turns") != LIMITS["max_turns"]
    ):
        raise ValueError("reward-canary plan binding changed")
    return binding
