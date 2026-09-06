"""Render a create-once CPU/DinD package for score-free DP6-m qualification."""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
import shlex
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import opencode_staged_image_v1 as staged_image
from evals.fleet import qwen38_dp6_m_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_m_scorefree_v1 as early
from evals.fleet import self_hosted

NAMESPACE = "fleet-train-jobs"
JOB_NAME = early.QUALIFIER_JOB
CONFIGMAP_NAME = early.QUALIFIER_JOB
OUTPUT_ROOT = early.QUALIFIER_OUTPUT_ROOT
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
DOCKER_CLI_SHA256 = "242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722"
DOCKER_BUILDX_SHA256 = "8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78"
DOCKER_CLI_TOTAL_BYTES = 105_594_160
DOCKER_CLI_VOLUME_SIZE = "256Mi"
RELEASE_SCHEMA = "fleet-qwen38-dp6-m-qualifier-release-v1"
PACKAGE_SCHEMA = "fleet-qwen38-dp6-m-qualifier-package-v1"
QUALIFIER_PRIORITY_CLASS = "fleet-serve-low"
QUALIFIER_PRIORITY_VALUE = 100
STATIC_PATHS = {
    Path("evals/__init__.py"),
    Path("evals/fleet/__init__.py"),
    Path("evals/fleet/models.py"),
    Path("evals/fleet/opencode_staged_image_v1.py"),
    Path("evals/fleet/configs/blackbox-ctf-tool-catalog-v1.json"),
    early.CONFIG_PATH,
    early.PLAN_PATH,
    early.PREVIEW_PATH,
    early.INVENTORY_PATH,
    early.RELEASE_PATH,
    early.SUCCESSOR_AUTHORITY_PATH,
    early.predecessor.PLAN_PATH,
    early.LIFECYCLE_PATH,
    early.OBSERVER_V1_PATH,
    early.OBSERVER_V2_PATH,
    early.OBSERVER_V3_PATH,
    early.OBSERVER_V4_PATH,
    early.OBSERVER_V5_PATH,
    early.QUALIFIER_PACKAGE_PATH,
    early.LIVE_GATE_PATH,
}


def _local_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "evals.fleet":
            imports.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("evals.fleet.")
        ):
            imports.add(node.module.rsplit(".", 1)[-1])
    return imports


def source_paths(root: Path) -> list[Path]:
    pending = ["qwen38_dp6_m_qualifier_runtime_v1"]
    modules: set[str] = set()
    while pending:
        name = pending.pop()
        if name in modules:
            continue
        path = root / "evals/fleet" / f"{name}.py"
        if not path.is_file():
            continue
        modules.add(name)
        pending.extend(_local_imports(path) - modules)
    paths = {Path("evals/fleet") / f"{name}.py" for name in modules} | STATIC_PATHS
    missing = [str(path) for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"qualifier package inputs missing: {missing}")
    return sorted(paths)


def archive_bytes(root: Path) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in source_paths(root):
            data = (root / path).read_bytes()
            info = tarfile.TarInfo(str(path))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def package_sha256(root: Path) -> str:
    return "sha256:" + hashlib.sha256(archive_bytes(root)).hexdigest()


def validate_release(
    value: Mapping[str, Any],
    submission: Mapping[str, Any],
    binding: Mapping[str, Any],
    root: Path,
) -> None:
    runtime.validate_binding(binding, submission, root)
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256") or (
        value.get("schema_version") != RELEASE_SCHEMA
        or value.get("status") != "RELEASED_FOR_ONE_NON_SCORED_QUALIFIER"
        or value.get("launch_authorized") is not True
        or value.get("scoring_authorized") is not False
        or value.get("job_name") != JOB_NAME
        or value.get("configmap_name") != CONFIGMAP_NAME
        or value.get("output_root") != OUTPUT_ROOT
        or value.get("serving_block") != early.SERVING_BLOCK
        or value.get("submission_receipt_sha256") != submission.get("receipt_sha256")
        or value.get("server_binding_receipt_sha256") != binding.get("receipt_sha256")
        or value.get("package_sha256") != package_sha256(root)
        or value.get("harness_runtime_image") != staged_image.identity()
        or value.get("fresh_job_matches") != 0
        or value.get("fresh_configmap_matches") != 0
        or value.get("fresh_output_root_exists") is not False
        or value.get("server_running_ready_restart0") is not True
        or value.get("api_mutations_before_create") != 0
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP6 qualifier release is not clear")


def _bootstrap() -> str:
    return """set -euo pipefail
mkdir -p /workspace/repo
mkdir -p /workspace/tmp
chmod 0700 /workspace/tmp
tar -xzf /bootstrap/package.tgz -C /workspace/repo
cd /workspace/repo
test "$(command -v docker)" = /docker-cli/bin/docker
test "$(sha256sum /docker-cli/bin/docker | awk '{print $1}')" = \
  242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722
test "$(sha256sum /docker-cli/plugins/docker-buildx | awk '{print $1}')" = \
  8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
test "$(( $(wc -c < /docker-cli/bin/docker) + \
  $(wc -c < /docker-cli/plugins/docker-buildx) ))" -eq 105594160
test ! -e "$DOCKER_CONFIG"
install -D -m 0755 /docker-cli/plugins/docker-buildx \
  "$DOCKER_CONFIG/cli-plugins/docker-buildx"
test "$(sha256sum "$DOCKER_CONFIG/cli-plugins/docker-buildx" | awk '{print $1}')" = \
  8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78
test "$(docker --version)" = 'Docker version 27.5.1, build 9f9e405'
"$DOCKER_CONFIG/cli-plugins/docker-buildx" version | grep -F 'v0.20.1' >/dev/null
until docker info >/dev/null 2>&1; do sleep 1; done
uv run --with httpx --with pyyaml python -m evals.fleet.opencode_staged_image_v1 \
  --receipt /mnt/sfs/jobs/chris-cyber-opencode11827-image-stage-v2/STAGED.json \
  --archive /mnt/sfs/jobs/chris-cyber-opencode11827-image-stage-v2/opencode-1.18.27-amd64.tar.gz
gzip -dc /mnt/sfs/jobs/chris-cyber-opencode11827-image-stage-v2/opencode-1.18.27-amd64.tar.gz \
  | docker load >/dev/null
test "$(docker image inspect chris/opencode:1.18.27-cyber-v1 --format '{{.Id}}')" = \
  sha256:4a46e71e98fbbc67f54dfd75fab15730af5ae575070d7b2ba1ad09ad4fa28b11
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
mkdir -p "$OUTPUT_ROOT"
exec uv run --with httpx --with pyyaml python -m evals.fleet.qwen38_dp6_m_qualifier_runtime_v1 \
  --submission /bootstrap/submission.json \
  --server-binding /bootstrap/server-binding.json \
  --release /bootstrap/release.json \
  --package /bootstrap/package.tgz \
  --output "$OUTPUT_ROOT/RESULT.json"
"""


def resource_sampler_script(
    *,
    sample_path: str,
    ready_path: str,
    cgroup_membership_path: str = "/proc/self/cgroup",
    cgroup_root: str = "/sys/fs/cgroup",
    one_shot: bool = False,
) -> str:
    """Render the producer against the container's exact cgroup-v2 leaf."""
    paths = (sample_path, ready_path, cgroup_membership_path, cgroup_root)
    if any(not value.startswith("/") or ".." in Path(value).parts for value in paths):
        raise ValueError("resource sampler paths must be safe absolute paths")
    loop = "write_sample" if one_shot else "observe &"
    return f"""set -eu
sample={shlex.quote(sample_path)}
ready={shlex.quote(ready_path)}
cgroup_membership={shlex.quote(cgroup_membership_path)}
cgroup_root={shlex.quote(cgroup_root)}
cgroup_relative=$(awk -F: '$1 == "0" && $2 == "" {{print $3}}' "$cgroup_membership")
case "$cgroup_relative" in
  /*) ;;
  *) exit 70 ;;
esac
case "$cgroup_relative/" in
  *"/../"*|*"/./"*|*//* ) exit 71 ;;
esac
cgroup_leaf="${{cgroup_root}}${{cgroup_relative}}"
test -d "$cgroup_leaf"
memory_events="$cgroup_leaf/memory.events"
cpu_stat="$cgroup_leaf/cpu.stat"
test -r "$memory_events"
test -r "$cpu_stat"
write_sample() {{
  now=$(date +%s)
  oom_kill=$(awk '$1 == "oom_kill" {{print $2; found=1}} END {{exit !found}}' "$memory_events")
  nr_throttled=$(awk '$1 == "nr_throttled" {{print $2; found=1}} END {{exit !found}}' "$cpu_stat")
  throttled_usec=$(
    awk '$1 == "throttled_usec" {{print $2; found=1}} END {{exit !found}}' "$cpu_stat"
  )
  case "$now:$oom_kill:$nr_throttled:$throttled_usec" in
    *[!0-9:]*) exit 72 ;;
  esac
  printf '%s %s %s %s\n' "$now" "$oom_kill" "$nr_throttled" \
    "$throttled_usec" >> "$sample"
}}
: > "$sample"
chmod 0644 "$sample"
write_sample
ready_tmp="${{ready}}.tmp.$$"
umask 022
printf '%s\n' 'fleet-qwen38-dp6-m-dind-resource-producer-v1' > "$ready_tmp"
ln "$ready_tmp" "$ready"
rm -f "$ready_tmp"
chmod 0644 "$ready"
observe() {{
  while true; do
    write_sample
    sleep 1
  done
}}
{loop}
"""


def _dind_bootstrap() -> str:
    sampler = resource_sampler_script(
        sample_path="/workspace/dind-resource-samples.tsv",
        ready_path="/workspace/dind-resource-producer.ready",
    )
    return sampler + 'exec /usr/local/bin/dockerd-entrypoint.sh "$@"\n'


def render(
    root: Path,
    submission: Mapping[str, Any],
    binding: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    validate_release(release, submission, binding, root)
    archive = archive_bytes(root)
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "binaryData": {"package.tgz": base64.b64encode(archive).decode()},
        "data": {
            "submission.json": json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n",
            "server-binding.json": json.dumps(binding, sort_keys=True, separators=(",", ":"))
            + "\n",
            "release.json": json.dumps(release, sort_keys=True, separators=(",", ":")) + "\n",
        },
    }
    if len(json.dumps(configmap, separators=(",", ":")).encode()) >= 900_000:
        raise ValueError("early DP6 qualifier ConfigMap exceeds safety budget")
    pod = {
        "restartPolicy": "Never",
        "priorityClassName": QUALIFIER_PRIORITY_CLASS,
        "preemptionPolicy": "Never",
        "nodeSelector": {"kubernetes.io/arch": "amd64", "workload": "fleetai-training-ng-cpu"},
        "tolerations": [
            {
                "key": "workload",
                "operator": "Equal",
                "value": "fleetai-training-ng-cpu",
                "effect": "NoSchedule",
            }
        ],
        "initContainers": [
            {
                "name": "docker-cli",
                "image": DIND_IMAGE,
                "command": ["/bin/sh", "-ec"],
                "args": [
                    " ".join(
                        (
                            "install -D -m 0755 /usr/local/bin/docker /cli/bin/docker;",
                            "install -D -m 0755",
                            "/usr/local/libexec/docker/cli-plugins/docker-buildx",
                            "/cli/plugins/docker-buildx;",
                            "test \"$(sha256sum /cli/bin/docker | awk '{print $1}')\" =",
                            f"{DOCKER_CLI_SHA256};",
                            'test "$(sha256sum /cli/plugins/docker-buildx | awk',
                            "'{print $1}')\" =",
                            f"{DOCKER_BUILDX_SHA256};",
                            'test "$(( $(wc -c < /cli/bin/docker) +',
                            '$(wc -c < /cli/plugins/docker-buildx) ))" -eq',
                            str(DOCKER_CLI_TOTAL_BYTES),
                        )
                    )
                ],
                "resources": {
                    "requests": {
                        "cpu": "10m",
                        "memory": "32Mi",
                        "ephemeral-storage": "32Mi",
                    },
                    "limits": {
                        "cpu": "100m",
                        "memory": "128Mi",
                        "ephemeral-storage": "128Mi",
                    },
                },
                "volumeMounts": [{"name": "docker-cli", "mountPath": "/cli"}],
            },
            {
                "name": "dind",
                "image": DIND_IMAGE,
                "restartPolicy": "Always",
                "command": ["/bin/sh", "-ceu", "--"],
                "args": [
                    _dind_bootstrap(),
                    "dind-observer",
                    "--host=unix:///var/run/docker.sock",
                    "--host=tcp://0.0.0.0:2375",
                    "--tls=false",
                ],
                "readinessProbe": {"tcpSocket": {"port": 2375}, "periodSeconds": 2},
                "securityContext": {"privileged": True},
                "resources": {
                    "requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
                    "limits": {"cpu": "10", "memory": "24Gi", "ephemeral-storage": "80Gi"},
                },
                "volumeMounts": [
                    {"name": "docker-socket", "mountPath": "/var/run"},
                    {"name": "docker-data", "mountPath": "/var/lib/docker"},
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "sfs", "mountPath": "/mnt/sfs"},
                ],
            },
        ],
        "containers": [
            {
                "name": "evaluator",
                "image": UV_IMAGE,
                "command": ["/bin/bash", "-ceu", "--"],
                "args": [_bootstrap()],
                "env": [
                    {
                        "name": "PATH",
                        "value": (
                            "/docker-cli/bin:/usr/local/sbin:/usr/local/bin:"
                            "/usr/sbin:/usr/bin:/sbin:/bin"
                        ),
                    },
                    {"name": "DOCKER_CONFIG", "value": "/workspace/docker-config"},
                    {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                    {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                    {"name": "OUTPUT_ROOT", "value": OUTPUT_ROOT},
                    {"name": "TMPDIR", "value": "/workspace/tmp"},
                ],
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi", "ephemeral-storage": "10Gi"},
                    "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
                },
                "volumeMounts": [
                    {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                    {
                        "name": "docker-cli",
                        "mountPath": "/docker-cli",
                        "readOnly": True,
                    },
                    {"name": "docker-socket", "mountPath": "/var/run"},
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "sfs", "mountPath": "/mnt/sfs"},
                ],
            }
        ],
        "volumes": [
            {"name": "bootstrap", "configMap": {"name": CONFIGMAP_NAME}},
            {"name": "docker-cli", "emptyDir": {"sizeLimit": DOCKER_CLI_VOLUME_SIZE}},
            {"name": "docker-socket", "emptyDir": {}},
            {"name": "docker-data", "emptyDir": {"sizeLimit": "40Gi"}},
            {"name": "workspace", "emptyDir": {"sizeLimit": "10Gi"}},
            {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": {
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": "q38-dp6-m-qualification",
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {
                "cyber-post-train.fleet.ai/create-once": "true",
                "cyber-post-train.fleet.ai/launch-authorized": "true",
                "cyber-post-train.fleet.ai/scoring-authorized": "false",
            },
        },
        "spec": {
            "activeDeadlineSeconds": 14400,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/experiment": "q38-dp6-m-qualification",
                    }
                },
                "spec": pod,
            },
        },
    }
    package = {
        "schema_version": PACKAGE_SCHEMA,
        "package_sha256": "sha256:" + hashlib.sha256(archive).hexdigest(),
        "release_receipt_sha256": release["receipt_sha256"],
        "harness_runtime_image": staged_image.identity(),
        "job": job,
        "configmap": configmap,
    }
    package["receipt_sha256"] = self_hosted.digest_without(package, "receipt_sha256")
    return package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--server-binding", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(
        args.repo_root,
        json.loads(args.submission.read_text()),
        json.loads(args.server_binding.read_text()),
        json.loads(args.release.read_text()),
    )
    manifest = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [value["configmap"], value["job"]],
    }
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
