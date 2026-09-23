"""Fail-closed source and submission gate for the Qwen3.8 SkyRL reward canary.

The source branch's preflight is retained as provenance, not as certification
of these rebased bytes.  This module reopens the immutable source package
before data acquisition or launch preparation and carries an explicit blocked
submission gate into the prepared plan.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "qwen38_skyrl_reward_canary_v8"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
TASK_SET_PATH = "configs/data/qwen38-rl-reward-canary-task-set-v3.json"
SPLIT_PATH = "configs/data/qwen38-rl-reward-canary-split-v1.json"
EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v3.json"
RUNTIME_EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v9.json"
HORIZON_PATH = "configs/data/qwen38-rl-reward-canary-horizon-v2.json"
TOOL_CATALOG_PATH = "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
QUALIFICATION_PATH = "configs/qualification/qwen38-rl-reward-canary-port-v8.json"
FAST_UPDATE_3_QUALIFICATION_PATH = "configs/qualification/qwen38-rl-reward-canary-port-v9.json"
FAST_UPDATE_IDENTITY_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast1-identity-v1.json"
)
FAST_UPDATE_2_IDENTITY_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast2-identity-v1.json"
)
FAST_UPDATE_3_IDENTITY_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-identity-v1.json"
)
FAST_UPDATE_3_DIAGNOSTIC_PATH = (
    "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-generation-failure-diagnostic-v1.json"
)
FAST_UPDATE_3_RETRY_POLICY_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-generation-retry-v1.json"
)

TASK_SET_FILE_SHA256 = "sha256:c3ee0a239dcb836b1951e65eda5a9813360a0378747716df8d8ca3753e741517"
TASK_SET_SELF_SHA256 = "sha256:9b88bac1510720926ea041d942044804d6c54c083a0ea56d07cae5f448685909"
SPLIT_FILE_SHA256 = "sha256:c3a34993a932c2b29e2c70f45682ce03ee02d03a9c4493edbab45eb349cb64fa"
SPLIT_SELF_SHA256 = "sha256:8279ea19808ad1accb00d3f3145c3ec087030677786e0188251cfc197f306adf"
EVIDENCE_FILE_SHA256 = "sha256:73be894762968970c8cf0454281fe13d46615e40039b61b8f8a97c8cc883f365"
EVIDENCE_SELF_SHA256 = "sha256:50491796b829388164faefc5fb9a9ad09fac89a8d94648a1791240f91826a285"
RUNTIME_EVIDENCE_FILE_SHA256 = (
    "sha256:8885e2d7cf4502db8ec77dcb07f690bdaaa1fa970946869e89e2e63d9d5d32cc"
)
RUNTIME_EVIDENCE_SELF_SHA256 = (
    "sha256:809f0d8a18425ad539a6ff94caa83f83b72af78ddda51cc1ad2aa1e02752629c"
)
HORIZON_FILE_SHA256 = "sha256:d80cd804406ed1f181ba9cc6891cd11785d7fc3a63b84aa2f97a7238c3d435b3"
HORIZON_SELF_SHA256 = "sha256:a2a13e123b51041b314c81772134dd63e89dd99c4f8a68822fa817968ca69182"
TOOL_CATALOG_FILE_SHA256 = "sha256:e4a3c4fb5b5c34cdaf64ec568eb31fcc0d55a63cc0a134808db348c65d7b6858"
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
QUALIFICATION_FILE_SHA256 = (
    "sha256:1f6b5ccbf9632f8a2788c57ffc9fd250a2863af76de4432aa625d3d18a1272b2"
)
QUALIFICATION_SELF_SHA256 = (
    "sha256:1754606c1e7e720e905560b92826839b1a41fb3943ef427f515c8c1e824897d5"
)
FAST_UPDATE_3_QUALIFICATION_FILE_SHA256 = (
    "sha256:c6ed4d87eb91121158980c2ec1a3065304a8b9b4119aee5305dc4a5ca081464c"
)
FAST_UPDATE_3_QUALIFICATION_SELF_SHA256 = (
    "sha256:79f5eec958521f2f7e1a2aa99ba11edda4f4ce5f9f4a571618c77ae35c2562c0"
)
FAST_UPDATE_IDENTITY_FILE_SHA256 = (
    "sha256:8d5cbe08c630dc8812473fdde35b041b59b02e4deef073aa134338437374a884"
)
FAST_UPDATE_IDENTITY_SELF_SHA256 = (
    "sha256:2671fb67b61b0c94616850e3b9c3ee06989838ced0c13ae3ccecd0181db4df16"
)
FAST_UPDATE_2_IDENTITY_FILE_SHA256 = (
    "sha256:d1ae811bae6e0b9ce94f8c108c7264dcea4e2c9c50bda2d6bde8a872f2d18a5f"
)
FAST_UPDATE_2_IDENTITY_SELF_SHA256 = (
    "sha256:702e67b86eff25efdcb67d93796278765ef6d665356b58c45825ea221a3d7948"
)
FAST_UPDATE_3_IDENTITY_FILE_SHA256 = (
    "sha256:d77c51b8d54ce537321ff3ed366390bdb13418eb24a17ae02e4ff83d8c32a67a"
)
FAST_UPDATE_3_IDENTITY_SELF_SHA256 = (
    "sha256:101bf3babf6634978bcf14f9ce2f5de277acbc505ba671757183e71089ba6651"
)
FAST_UPDATE_3_DIAGNOSTIC_FILE_SHA256 = (
    "sha256:69ced0087abecd437308c2ffb66e8a0167bf56f7a3ba3ad22f3b1c18a481c2f0"
)
FAST_UPDATE_3_DIAGNOSTIC_SELF_SHA256 = (
    "200d51089c155cac2fa777db4f84287c50d14bdb23c86e60e8e47c2d3d2fd2fe"
)
FAST_UPDATE_3_RETRY_POLICY_FILE_SHA256 = (
    "sha256:895a4941c782f1c6936426b386d36c67055d9ca9bfd6db2a9efaff8f2159e971"
)
FAST_UPDATE_3_RETRY_POLICY_SELF_SHA256 = (
    "sha256:be1bc9e95ab7061720bf7515c725f79ad7f1db373360ce184ff438b8f299bab4"
)

LIMITS = {
    "context_tokens": 262144,
    "response_tokens": 4194304,
    "max_tokens_per_turn": 32768,
    "generation_chunk_tokens": 4096,
    "compaction_trigger_tokens": 163840,
    "compaction_summary_tokens": 8192,
    "max_turns": 1200,
    "episode_seconds": 14400,
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
FAST_UPDATE_RECIPE = {**RECIPE, "eval_before_train": False}
FAST_UPDATE_IDENTITY = {
    "run_name": "chris-q38-rlreward-prod11-fast1",
    "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast1",
    "data_root": ("/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast1-v1/data"),
    "wandb_run_id": "chris-q38-rlreward-prod11-fast1",
}
FAST_UPDATE_2_IDENTITY = {
    "run_name": "chris-q38-rlreward-prod11-fast2",
    "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast2",
    "data_root": ("/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast2-v1/data"),
    "wandb_run_id": "chris-q38-rlreward-prod11-fast2",
}
FAST_UPDATE_3_IDENTITY = {
    "run_name": "chris-q38-rlreward-prod11-fast3",
    "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast3",
    "data_root": ("/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast3-v1/data"),
    "wandb_run_id": "chris-q38-rlreward-prod11-fast3",
}
FAST_UPDATE_IDENTITIES = (
    {
        "identity": FAST_UPDATE_IDENTITY,
        "path": FAST_UPDATE_IDENTITY_PATH,
        "file_sha256": FAST_UPDATE_IDENTITY_FILE_SHA256,
        "self_sha256": FAST_UPDATE_IDENTITY_SELF_SHA256,
    },
    {
        "identity": FAST_UPDATE_2_IDENTITY,
        "path": FAST_UPDATE_2_IDENTITY_PATH,
        "file_sha256": FAST_UPDATE_2_IDENTITY_FILE_SHA256,
        "self_sha256": FAST_UPDATE_2_IDENTITY_SELF_SHA256,
    },
    {
        "identity": FAST_UPDATE_3_IDENTITY,
        "path": FAST_UPDATE_3_IDENTITY_PATH,
        "file_sha256": FAST_UPDATE_3_IDENTITY_FILE_SHA256,
        "self_sha256": FAST_UPDATE_3_IDENTITY_SELF_SHA256,
        "diagnostic": {
            "path": FAST_UPDATE_3_DIAGNOSTIC_PATH,
            "file_sha256": FAST_UPDATE_3_DIAGNOSTIC_FILE_SHA256,
            "self_sha256": FAST_UPDATE_3_DIAGNOSTIC_SELF_SHA256,
        },
        "retry_policy": {
            "path": FAST_UPDATE_3_RETRY_POLICY_PATH,
            "file_sha256": FAST_UPDATE_3_RETRY_POLICY_FILE_SHA256,
            "self_sha256": FAST_UPDATE_3_RETRY_POLICY_SELF_SHA256,
        },
    },
)
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
        "sha256:49a5032de0e18f1fdde78c0e475488bd035b107a0403633bbfcb43f1c7569b8e"
    ),
    "training/rl_episode.py": (
        "sha256:09517093e141fb82a38b8ba62dafa33ae3705c2dc236e16c6d976586d766c7de"
    ),
    "training/skyrl_episode.py": (
        "sha256:d630b5e6e714a2899726789e8bc36c1dbc5ab325d4573bbe917fc2053a195d2f"
    ),
    "training/skyrl_rollout.py": (
        "sha256:a9ba203eb93d0622fda161636fb8499b46b30c1bbb349df4eda484f5fd407c52"
    ),
    "evals/fleet/opencode_self_hosted.py": (
        "sha256:428e9f2e4d4c758c4f682cbed314cf96b866d051b34fa0eea8757a3aad97aa1d"
    ),
}
PORT_COMMITS = {
    "runtime_commit": "480d699a98ccabbf1f44cd76447b8ab700841408",
    "preflight_commit": "480d699a98ccabbf1f44cd76447b8ab700841408",
    "ported_onto_commit": "91ab058ebbf744c0931dbb9692831bdbd7ecefaf",
}
HISTORICAL_PORT_COMMITS = {
    "runtime_commit": "8b5e8a00521b4fe412e3e7d0bce9b8ea6855b005",
    "preflight_commit": "6c0bc541e136fc114a3f7a17ce6ba07e9178bb78",
}
SUBMISSION_BLOCKERS = [
    "prod8_data_not_yet_staged_and_digest_verified",
    "prod8_output_limit_cpu_preflight_not_yet_recorded",
    "prod8_jobs_kubernetes_sfs_wandb_absence_not_yet_rechecked",
]
TOPOLOGY_SUCCESSOR = {
    "config_path": "qwen38-skyrl-topology-probe-dev-v2.json",
    "config_file_sha256": (
        "sha256:ceb413e27765dbbd88865c9f22ad186ee4f39387de93874111714f7861f9d529"
    ),
    "config_self_sha256": (
        "sha256:f7d53ee426ee4f23e1c608bc17583f5f105ac30b6a8c19b04f966b908cc47d52"
    ),
    "name": "chris-q38-skyrl-probe-v17",
    "plan_sha256": "sha256:c7dba701b3b8805a8efbf5cda67f346a5aec0fddd2c8356756131830beca2c40",
    "request_sha256": ("sha256:6e6826a35a45028b8ef0aad05ef476a9c941f22fcb58dbbde830712891f95d95"),
    "fleetjob_manifest_sha256": (
        "sha256:0a8ba4e08b8a1e3149b9a19f1bba339834f0369894e9d8f42c854099fd0dae05"
    ),
    "preflight_manifest_sha256": (
        "sha256:4f97ad8c26a6a3529813218424249e5d2894f48657ac5306ae6cccce011646b4"
    ),
    "receipt_verifier_manifest_sha256": (
        "sha256:4eb41b7109baae0b6d95b993dcb293a55fdaf6651ca9f517dd74f9ef87b8aa27"
    ),
    "required_terminal_receipt": "TOPOLOGY_PROBE.json",
    "terminal_receipt_grace_seconds": 30,
    "accepted": False,
}
FAST_UPDATE_3_TOPOLOGY_SUCCESSOR = {
    **TOPOLOGY_SUCCESSOR,
    "plan_sha256": "sha256:1e3e3104e2d0e8632fcf7ea2ff5d50450f9e3f596c44b790b5ea0cffc20a9db5",
    "request_sha256": "sha256:636caaa767dedd5113ac6b1768dfae76cbed9e54042bd882b2972f157cb537ae",
    "fleetjob_manifest_sha256": (
        "sha256:0a3ad4da6ee8f56c357b4b112d0e2f257cc8be4377d803cb4f6076390f72c6a4"
    ),
    "preflight_manifest_sha256": (
        "sha256:29fe86dc5ec69cabdeb598abfd9ff0e50eabf43e8106dd2425ff354d4f02be50"
    ),
    "receipt_verifier_manifest_sha256": (
        "sha256:ae72d57f4b989e8b7e96868e65ef7ace254bcc36577b3bd8610d81554ef8f9c4"
    ),
}


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
    runtime_evidence = _bound_json(ROOT / RUNTIME_EVIDENCE_PATH, RUNTIME_EVIDENCE_FILE_SHA256)
    horizon = _bound_json(ROOT / HORIZON_PATH, HORIZON_FILE_SHA256)
    _sealed(task_set, "cyber_rl_task_set_v1")
    _sealed(split, "cyber_task_split_v1")
    _sealed(evidence, "cyber_rl_exact_version_evidence_v1")
    _sealed(runtime_evidence, "cyber_rl_exact_version_evidence_v1")
    _sealed(horizon, "cyber_rl_reward_canary_horizon_v1")
    if (
        task_set["sha256"] != TASK_SET_SELF_SHA256
        or split["sha256"] != SPLIT_SELF_SHA256
        or evidence["sha256"] != EVIDENCE_SELF_SHA256
        or runtime_evidence["sha256"] != RUNTIME_EVIDENCE_SELF_SHA256
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
            "runtime_commit": HISTORICAL_PORT_COMMITS["runtime_commit"],
            "preflight_commit": HISTORICAL_PORT_COMMITS["preflight_commit"],
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

    immutable_runtime_keys = set(evidence) - {
        "purpose",
        "tool_surface_authority",
        "episode_horizon",
        "sha256",
    }
    if any(runtime_evidence[key] != evidence[key] for key in immutable_runtime_keys):
        raise ValueError("reward canary runtime evidence changed task or horizon authority")
    authority = runtime_evidence["tool_surface_authority"]
    local_code = authority["local_code"]
    for name, path, symbols in (
        ("data_preparation", "training/rl_data.py", ["build"]),
        ("episode_runtime", "training/rl_episode.py", ["collect", "_agent"]),
        (
            "skyrl_episode_runtime",
            "training/skyrl_episode.py",
            ["Recorder", "offline_long_horizon_probe"],
        ),
        ("skyrl_rollout_runtime", "training/skyrl_rollout.py", ["Generator"]),
        (
            "fleet_binding",
            "evals/fleet/opencode_self_hosted.py",
            ["bind_task", "verify_task", "assert_required_task_tools"],
        ),
    ):
        source_path = ROOT / path
        declared = local_code.get(name, {})
        try:
            parsed = ast.parse(source_path.read_bytes(), filename=path)
        except (OSError, SyntaxError) as error:
            raise ValueError("reward canary tool enforcement source is unreadable") from error
        top_level_symbols = {
            node.name
            for node in parsed.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if (
            declared.get("path") != "../../" + path
            or declared.get("symbols") != symbols
            or declared.get("file_sha256") != RUNTIME_SOURCES[path]
            or fleet.sha256(source_path.read_bytes()) != RUNTIME_SOURCES[path]
            or not set(symbols) <= top_level_symbols
        ):
            raise ValueError("reward canary tool enforcement source changed")
    if (
        authority["ordered_tools"] != ["bash", "submit_report"]
        or authority["tool_catalog_sha256"] != TOOL_CATALOG_SHA256
        or authority["task_metadata_tools_required"] is not False
        or authority["reviewed_server_route"]["metadata_tools_read"] is not False
    ):
        raise ValueError("reward canary tool authority changed")

    horizon_binding = runtime_evidence["episode_horizon"]
    if horizon_binding != {
        "max_turns": 1200,
        "contract_path": "qwen38-rl-reward-canary-horizon-v2.json",
        "contract_file_sha256": HORIZON_FILE_SHA256,
        "contract_self_sha256": HORIZON_SELF_SHA256,
        "aggregate_evidence_path": "../../docs/FLEET_NATIVE_HARNESS_PARITY.md",
        "aggregate_evidence_file_sha256": (
            "sha256:d6846aa328d4a915464af062b21032d5ea2a0e09bf0dd98d0050a2d498ffa632"
        ),
        "rationale": horizon_binding["rationale"],
        "fixed_limits": {key: value for key, value in LIMITS.items() if key != "max_turns"},
        "output_limit_contract": {
            "gradeable_reason": "turn_response_budget_exhausted",
            "native_stop_reason": "length",
            "authoritative_reward_required": True,
            "partial_tool_execution_forbidden": True,
            "other_budget_stops_fail_closed": True,
            "incident_evidence_path": (
                "../../docs/evidence/qwen38-study/"
                "2026-09-21-skyrl-prod7-output-limit-rejection-v1.json"
            ),
            "incident_evidence_self_sha256": (
                "sha256:c45f12a8fa799900053a1bd7b1e33ff26e89d517ae552188bda0e66cf675ca2d"
            ),
        },
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


def _qualification_spec(path: Path) -> dict:
    choices = {
        (ROOT / QUALIFICATION_PATH).resolve(): {
            "schema": "cyber_qwen38_skyrl_reward_canary_port_v8",
            "path": QUALIFICATION_PATH,
            "file_sha256": QUALIFICATION_FILE_SHA256,
            "self_sha256": QUALIFICATION_SELF_SHA256,
            "topology_successor": TOPOLOGY_SUCCESSOR,
        },
        (ROOT / FAST_UPDATE_3_QUALIFICATION_PATH).resolve(): {
            "schema": "cyber_qwen38_skyrl_reward_canary_port_v9",
            "path": FAST_UPDATE_3_QUALIFICATION_PATH,
            "file_sha256": FAST_UPDATE_3_QUALIFICATION_FILE_SHA256,
            "self_sha256": FAST_UPDATE_3_QUALIFICATION_SELF_SHA256,
            "topology_successor": FAST_UPDATE_3_TOPOLOGY_SUCCESSOR,
        },
    }
    spec = choices.get(path.resolve())
    if spec is None:
        raise ValueError("unknown reward-canary qualification closure")
    return spec


def _validate_qualification(path: Path) -> dict:
    spec = _qualification_spec(path)
    value = _bound_json(path, spec["file_sha256"])
    _sealed(value, spec["schema"])
    if (
        value["sha256"] != spec["self_sha256"]
        or value["source"] != PORT_COMMITS
        or value["execution"]
        != {
            "cluster_target": "prod",
            "jobs_api_base_url": "https://api.ft.flt.build",
            "image": IMAGE,
            "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
        or value.get("topology_successor") != spec["topology_successor"]
        or value["source_closure"]
        != {
            "task_set": "../data/qwen38-rl-reward-canary-task-set-v3.json",
            "split": "../data/qwen38-rl-reward-canary-split-v1.json",
            "horizon": "../data/qwen38-rl-reward-canary-horizon-v2.json",
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

    topology_config = _bound_json(
        path.parent / TOPOLOGY_SUCCESSOR["config_path"],
        TOPOLOGY_SUCCESSOR["config_file_sha256"],
    )
    _sealed(topology_config, "cyber_skyrl_topology_probe_config_v1")
    if (
        topology_config["sha256"] != TOPOLOGY_SUCCESSOR["config_self_sha256"]
        or topology_config["name"] != TOPOLOGY_SUCCESSOR["name"]
        or topology_config["execution"].get("cluster_target") != "dev"
        or topology_config["submission_gate"].get("submission_authorized") is not False
    ):
        raise ValueError("reward-canary topology successor changed")

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
        or preflight["source"]["git_commit"] != HISTORICAL_PORT_COMMITS["runtime_commit"]
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


def _fast_update_spec(selected: dict) -> tuple[dict | None, bool]:
    matches = [spec for spec in FAST_UPDATE_IDENTITIES if selected == spec["identity"]]
    if len(matches) > 1:
        raise ValueError("reward-canary fast-update identity is ambiguous")
    partial = any(
        selected[key] == value
        for spec in FAST_UPDATE_IDENTITIES
        for key, value in spec["identity"].items()
    )
    return (matches[0] if matches else None), partial


def _fast_update_expected_binding(spec: dict) -> dict:
    result = {
        "schema": "cyber_rl_reward_canary_fast_update_binding_v1",
        "identity_path": spec["path"],
        "identity_file_sha256": spec["file_sha256"],
        "identity_self_sha256": spec["self_sha256"],
        "eval_before_train": False,
    }
    diagnostic, policy = spec.get("diagnostic"), spec.get("retry_policy")
    if diagnostic is not None or policy is not None:
        if diagnostic is None or policy is None:
            raise ValueError("reward-canary generation retry binding is incomplete")
        failure_source = {
            "schema": "cyber_rl_reward_canary_generation_failure_source_v1",
            "path": diagnostic["path"],
            "file_sha256": diagnostic["file_sha256"],
            "self_sha256": diagnostic["self_sha256"],
            "run_name": "chris-q38-rlreward-prod11",
            "episode_index": 6,
            "http_status_retained": False,
            "retryability_proven": False,
        }
        failure_source["sha256"] = "sha256:" + digest(failure_source)
        retry = {
            "schema": "cyber_rl_reward_canary_generation_retry_binding_v1",
            "policy_path": policy["path"],
            "policy_file_sha256": policy["file_sha256"],
            "policy_self_sha256": policy["self_sha256"],
            "max_http_attempts": 3,
            "retryable_http_statuses": {
                "exact": [429],
                "inclusive_ranges": [[500, 599]],
            },
            "backoff_seconds": [1, 2],
            "transport_retry": False,
            "follow_redirects": False,
            "whole_episode_retry": False,
            "failed_response_body_admitted": False,
            "failed_response_tokens_admitted": False,
            "admitted_response": "first_2xx_json_only",
            "extra_generation_compute_possible": True,
            "failure_source_diagnostic": failure_source,
        }
        retry["sha256"] = "sha256:" + digest(retry)
        result["generation_retry_policy"] = retry
    result["sha256"] = "sha256:" + digest(result)
    return result


def _validate_fast_update_diagnostic(spec: dict) -> None:
    from . import skyrl_episode

    binding, policy_binding = spec.get("diagnostic"), spec.get("retry_policy")
    if binding is None and policy_binding is None:
        return
    if binding is None or policy_binding is None:
        raise ValueError("reward-canary generation retry binding is incomplete")
    policy = _bound_json(ROOT / policy_binding["path"], policy_binding["file_sha256"])
    policy_body = {key: item for key, item in policy.items() if key != "sha256"}
    expected_policy = {
        "schema": "cyber_skyrl_generation_http_retry_policy_v1",
        "max_http_attempts": 3,
        "retryable_http_statuses": {
            "exact": [429],
            "inclusive_ranges": [[500, 599]],
        },
        "backoff_seconds": [1, 2],
        "transport_retry": False,
        "follow_redirects": False,
        "whole_episode_retry": False,
        "failed_response_body_admitted": False,
        "failed_response_tokens_admitted": False,
        "admitted_response": "first_2xx_json_only",
        "extra_generation_compute_possible": True,
        "historical_failure_retryability_proven": False,
        "failure_source_diagnostic": {
            "path": "../../" + binding["path"],
            "file_sha256": binding["file_sha256"],
            "self_sha256": binding["self_sha256"],
            "run_name": "chris-q38-rlreward-prod11",
        },
    }
    if (
        policy_body != expected_policy
        or policy.get("sha256") != policy_binding["self_sha256"]
        or policy.get("sha256") != "sha256:" + digest(policy_body)
        or expected_policy["max_http_attempts"] != skyrl_episode.GENERATION_MAX_ATTEMPTS
        or list(skyrl_episode.GENERATION_RETRY_BACKOFF_SECONDS)
        != expected_policy["backoff_seconds"]
    ):
        raise ValueError("reward-canary generation retry policy changed")
    value = _bound_json(ROOT / binding["path"], binding["file_sha256"])
    body = {key: item for key, item in value.items() if key != "sha256"}
    causes = value.get("episode_failure", {}).get("causes", [])
    source = value.get("source", {})
    last_frames = causes[-1].get("frames", []) if len(causes) == 3 else []
    if (
        set(value)
        != {
            "schema",
            "status",
            "source",
            "files",
            "episode_failure",
            "native_failure",
            "failed",
            "sfs_read_only",
            "sha256",
        }
        or value.get("schema") != "cyber_rl_prod11_failure_diagnostic_v1"
        or value.get("status") != "completed"
        or value.get("sfs_read_only") is not True
        or value.get("sha256") != binding["self_sha256"]
        or value.get("sha256") != digest(body)
        or source.get("plan_sha256")
        != "f86ca0c93754def88d2f9053fb7fa012af3b6d29046846fa5ad229972ce9910f"
        or source.get("episode_index") != 6
        or len(causes) != 3
        or causes[-1].get("error_type") != "InvalidEpisode"
        or (last_frames[-1] if last_frames else None)
        != {"file": "skyrl_episode.py", "function": "_post", "line": 108}
        or any("http_status" in item for item in causes)
    ):
        raise ValueError("reward-canary fast-update predecessor diagnostic changed")


def _fast_update_binding(config: dict) -> dict | None:
    selected = {
        "run_name": config.get("name"),
        "output_root": config.get("output_root"),
        "data_root": config.get("data", {}).get("root"),
        "wandb_run_id": config.get("wandb", {}).get("run_id"),
    }
    spec, selects_fast_identity = _fast_update_spec(selected)
    if config.get("recipe") == RECIPE and not selects_fast_identity:
        return None
    if config.get("recipe") != FAST_UPDATE_RECIPE or spec is None:
        raise ValueError("reward-canary recipe changed")
    identity = _bound_json(
        ROOT / spec["path"],
        spec["file_sha256"],
    )
    _sealed(identity, "cyber_skyrl_reward_direct_identity_v1")
    if (
        identity.get("sha256") != spec["self_sha256"]
        or {key: identity.get(key) for key in spec["identity"]} != spec["identity"]
        or selected != spec["identity"]
    ):
        raise ValueError("reward-canary fast-update identity changed")
    _validate_fast_update_diagnostic(spec)
    return _fast_update_expected_binding(spec)


def validate_run_config(
    config: dict,
    metadata: dict,
    bound_model: dict,
    *,
    relative_to: Path,
) -> dict:
    """Bind the exact v3 profile while keeping fresh identities in JSON only."""
    qualification_path = relative_to / config["qualification"]
    qualification_spec = _qualification_spec(qualification_path)
    qualification = _validate_qualification(qualification_path)
    proof = validate_source_package(
        ROOT / TASK_SET_PATH,
        ROOT / SPLIT_PATH,
        ROOT / TOOL_CATALOG_PATH,
        metadata.get("limits"),
    )
    data = config["data"]
    output = PurePosixPath(config["output_root"])
    data_root = PurePosixPath(data["root"])
    fast_update = _fast_update_binding(config)
    expected_qualification_path = (
        FAST_UPDATE_3_QUALIFICATION_PATH
        if fast_update is not None and "generation_retry_policy" in fast_update
        else QUALIFICATION_PATH
    )
    if (
        metadata.get("selection_sha256") != TASK_SET_SELF_SHA256
        or metadata.get("split_sha256") != SPLIT_SELF_SHA256
        or metadata.get("tool_catalog_sha256") != TOOL_CATALOG_SHA256
        or {key: item["rows"] for key, item in metadata.get("files", {}).items()}
        != {"train": 1, "dev": 1}
        or config["recipe"] not in (RECIPE, FAST_UPDATE_RECIPE)
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
        or qualification_spec["path"] != expected_qualification_path
    ):
        raise ValueError("reward-canary model, data, recipe, resource, or identity drift")
    result = {
        "schema": "cyber_qwen38_skyrl_reward_canary_plan_binding_v8",
        "profile": PROFILE,
        "source_proof": proof,
        "qualification_file_sha256": qualification_spec["file_sha256"],
        "qualification_self_sha256": qualification_spec["self_sha256"],
        "image": qualification["execution"]["image"],
        "environment": qualification["execution"]["environment"],
        "cluster_target": qualification["execution"]["cluster_target"],
        "jobs_api_base_url": qualification["execution"]["jobs_api_base_url"],
        "submission_gate": qualification["submission_gate"],
    }
    if fast_update is not None:
        result["fast_update"] = fast_update
    result["sha256"] = "sha256:" + digest(result)
    return result


def validate_plan_binding(binding: object, metadata: dict, arguments: dict) -> dict:
    """Validate a compiler-produced binding without mutable source-tree reads."""
    if not isinstance(binding, dict):
        raise ValueError("reward-canary plan lacks its source binding")
    body = {key: item for key, item in binding.items() if key != "sha256"}
    fast_update = binding.get("fast_update")
    fast_arguments = arguments.get("eval_before_train") is False
    selected = {
        "run_name": arguments.get("name"),
        "output_root": arguments.get("output_root"),
        "data_root": str(PurePosixPath(arguments.get("data_manifest", "")).parent),
        "wandb_run_id": arguments.get("wandb_run_id"),
    }
    fast_spec, selected_fast_identity = _fast_update_spec(selected)
    selected_fast_identity = selected_fast_identity or any(
        metadata.get("name") == spec["identity"]["run_name"] for spec in FAST_UPDATE_IDENTITIES
    )
    expected_fast_update = None
    if fast_spec is not None:
        expected_fast_update = _fast_update_expected_binding(fast_spec)
    expected_qualification = (
        {
            "file_sha256": FAST_UPDATE_3_QUALIFICATION_FILE_SHA256,
            "self_sha256": FAST_UPDATE_3_QUALIFICATION_SELF_SHA256,
        }
        if fast_spec is not None and fast_spec.get("retry_policy") is not None
        else {
            "file_sha256": QUALIFICATION_FILE_SHA256,
            "self_sha256": QUALIFICATION_SELF_SHA256,
        }
    )
    if (
        binding.get("schema") != "cyber_qwen38_skyrl_reward_canary_plan_binding_v8"
        or binding.get("sha256") != "sha256:" + digest(body)
        or binding.get("profile") != PROFILE
        or binding.get("source_proof") != source_proof()
        or binding.get("qualification_file_sha256") != expected_qualification["file_sha256"]
        or binding.get("qualification_self_sha256") != expected_qualification["self_sha256"]
        or binding.get("image") != IMAGE
        or binding.get("environment") != {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
        or binding.get("cluster_target") != "prod"
        or binding.get("jobs_api_base_url") != "https://api.ft.flt.build"
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
        or fast_arguments != selected_fast_identity
        or fast_arguments
        != (expected_fast_update is not None and fast_update == expected_fast_update)
        or (
            fast_arguments
            and (
                fast_spec is None
                or selected != fast_spec["identity"]
                or metadata.get("name") != fast_spec["identity"]["run_name"]
            )
        )
        or (not fast_arguments and ("eval_before_train" in arguments or fast_update is not None))
        or arguments.get("context_tokens") != LIMITS["context_tokens"]
        or arguments.get("response_tokens") != LIMITS["response_tokens"]
        or arguments.get("tokens_per_turn") != LIMITS["max_tokens_per_turn"]
        or arguments.get("max_turns") != LIMITS["max_turns"]
    ):
        raise ValueError("reward-canary plan binding changed")
    return binding
