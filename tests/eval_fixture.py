"""Synthetic paired cyber task; never submits or contains private task content."""

from evals.direct import digest
from evals.fleet import seal_protocol


def sha(char):
    return "sha256:" + char * 64


def fixture():
    task = {"task_key": "synthetic-blackbox-task", "task_version_id": "version-1",
            "application": "app", "family_id": "family-dev",
            "environment_version_id": "env-version-1", "data_version": "data-1",
            "verifier_sha256": sha("e")}
    protocol = seal_protocol({
        "schema": "fleet_paired_pass4_v2", "study_id": "synthetic",
        "role": "dev", "final_selection_sha256": None, "tasks": [task],
        "common": {
            "model_repository": "Qwen/synthetic", "tokenizer_sha256": sha("a"),
            "chat_template_sha256": sha("b"), "serving_image": "serve@" + sha("c"),
            "serving_config_sha256": sha("d"), "harness_image": "runner@" + sha("f"),
            "harness_version": "pinned-opencode", "system_prompt_sha256": sha("a"),
            "tools": ["fleet_bash", "fleet_submit_report"],
            "tool_schema_sha256": sha("b"), "context_policy": "native_compaction",
            "context_window_tokens": 262144, "max_output_tokens": 32768,
            "max_steps": 600, "max_duration_minutes": 480,
            "temperature": None, "top_p": None,
            "seed_policy": {"mode": "server_assigned_unobserved"},
            "scoring_mode": "partial", "pass_criterion": "cyber_ctf_full_solve_v1",
            "retry_limit": 0,
        },
        "arms": {
            "base": {"model_revision": "base-rev", "weights_sha256": sha("1")},
            "candidate": {"model_revision": "trained-rev", "weights_sha256": sha("2"),
                          "checkpoint_sha256": sha("3")},
        },
    })
    live = {
        "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
        "key": task["task_key"], "eval_task_version_id": task["task_version_id"],
        "environment_version_id": task["environment_version_id"],
        "data_version": task["data_version"],
        "verifier": {"sha256": "e" * 64, "verifier_version_id": "verifier-v1"},
        "metadata": {"projection_id": "blackbox_ctf_v1",
                     "runtime_seed_manifest": {"version": 1, "data_root": "synthetic-data",
                         "files": [{"sha256": "f" * 64, "target_path": "synthetic-seed",
                                    "size_bytes": 1}], "content_sha256": "f" * 64}},
        "seed_config": {"non_private_test_seed": True},
        "task_lifecycle_status": "production",
    }
    plan = {
        "schema": "fleet_native_paired_launch_v1", "protocol": protocol,
        "task_group_id": "11111111-1111-4111-8111-111111111111",
        "family_roles": {"train": [["app", "family-train"]],
                         "dev": [["app", "family-dev"]],
                         "final": [["app", "family-final"]]},
        "task_response_sha256": {task["task_version_id"]: digest(live)},
        "task_qualification_sha256": {task["task_version_id"]: sha("9")},
        "readiness_sha256": {
            "candidate_export": sha("1"), "candidate_reload": sha("2"),
            "base_registration": sha("3"), "candidate_registration": sha("4"),
            "live_parity": sha("5"),
        },
        "routes": {"base": "fleet-qwen/baseline-synthetic",
                   "candidate": "fleet/checkpoint-synthetic-step-16"},
    }
    group = {"id": plan["task_group_id"], "team_id": live["team_id"],
             "members": [{"eval_task_version_id": task["task_version_id"]}]}
    return plan, live, group
