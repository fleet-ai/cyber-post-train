"""Shared isolated OpenCode scorer for NYU CTF Bench and Cybench.

The challenge and agent share one internal Docker network.  A separate proxy
holds the Fleet credential, joins that network and the default bridge, and
allows only the two OpenAI-compatible inference paths.  The agent never gets a
provider credential or Docker access.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from . import cybench_qualification as cy_source
from . import cybench_runtime_qualification as cy_runtime
from . import nyu_adapter
from . import nyu_runtime_qualification as nyu_runtime
from . import runtime_qualification as rt
from .protocol import DEFAULT_PROTOCOL, digest, file_digest, load_protocol

PROXY_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
PROXY_SOURCE = Path(__file__).with_name("fixed_proxy.py")
if not PROXY_SOURCE.is_file():
    PROXY_SOURCE = Path(__file__).parents[1] / "fleet/fixed_proxy.py"
PROXY_ENTRYPOINT = Path(__file__).with_name("external_proxy.py")
REAL_CREDENTIAL_ENVIRONMENT_NAMES = frozenset(
    {
        *rt.MODEL_CREDENTIALS,
        "APOLLO_FLEET_API_KEY",
        "TENSORLAKE_API_KEY",
    }
)
LOCAL_PROXY_PLACEHOLDER = "local-proxy-only"


class ScoredAdapterError(RuntimeError):
    """The scored runtime or native grading contract was not preserved."""


def source_sha256() -> str:
    """Bind the shared executor, secret boundary, and reused fixed proxy bytes."""
    files = {
        "external_proxy.py": PROXY_ENTRYPOINT,
        "fixed_proxy.py": PROXY_SOURCE,
        "opencode_scored.py": Path(__file__),
    }
    return digest({name: file_digest(path.read_bytes()) for name, path in files.items()})


def _settings(model: str, context: int, output: int) -> dict[str, Any]:
    if not model or output >= context:
        raise ScoredAdapterError("opencode_model_contract_invalid")
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "external-ctf": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "External CTF fixed inference proxy",
                "options": {
                    "baseURL": "http://model-proxy:8877/v1",
                    "apiKey": "local-proxy-only",
                    "timeout": False,
                    "chunkTimeout": 300000,
                },
                "models": {
                    model: {
                        "name": model,
                        "reasoning": True,
                        "tool_call": True,
                        "interleaved": "reasoning_content",
                        "limit": {"context": context, "output": output, "input": context - output},
                    }
                },
            }
        },
        "permission": {"*": "allow"},
        "tools": {name: False for name in ("task", "webfetch", "websearch", "skill")},
        "compaction": {"auto": True, "reserved": 20000},
    }


def _take_credential() -> str:
    key = os.environ.pop("FLEET_API_KEY", "")
    if not key or key.strip() != key or any(character.isspace() for character in key):
        raise ScoredAdapterError("fleet_credential_missing_or_invalid")
    return key


def _proxy(sandbox: rt.Sandbox, *, credential: str, model: str, requests: int, output: int) -> str:
    rt.docker("pull", PROXY_IMAGE, timeout=1800)
    if rt.image_lock(PROXY_IMAGE, repository_digest=True)["repository_digest"] != PROXY_IMAGE:
        raise ScoredAdapterError("proxy_image_digest_mismatch")
    policy = json.dumps(
        {
            "model": model,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": None,
            "max_tokens": output,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    proxy = sandbox.create(
        "proxy",
        "--network",
        "bridge",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--mount",
        f"type=bind,src={PROXY_SOURCE.resolve()},dst=/fixed_proxy.py,readonly",
        "--mount",
        f"type=bind,src={PROXY_ENTRYPOINT.resolve()},dst=/external_proxy.py,readonly",
        "--env",
        "FIXED_UPSTREAM=https://inference.flt.build",
        "--env",
        "FIXED_AUTH_HEADER=Authorization",
        "--env",
        f"FIXED_AUTH_VALUE=Bearer {credential}",
        "--env",
        "FIXED_ALLOWED_PATHS=/v1/chat/completions,/v1/models",
        "--env",
        f"FIXED_MAX_REQUESTS={requests}",
        "--env",
        "FIXED_MAX_REQUEST_BYTES=16777216",
        "--env",
        f"FIXED_COMPLETION_JSON={policy}",
        "--entrypoint",
        "python",
        PROXY_IMAGE,
        "/external_proxy.py",
    )
    rt.docker("network", "connect", "--alias", "model-proxy", sandbox.network, proxy)
    probe = (
        "import urllib.request;"
        "assert urllib.request.urlopen('http://127.0.0.1:8877/healthz',timeout=1).status==200"
    )
    for _ in range(30):
        if rt.docker("exec", proxy, "python", "-c", probe, check=False, timeout=5).returncode == 0:
            return proxy
        time.sleep(1)
    raise ScoredAdapterError("inference_proxy_not_ready")


def qualify_credential_boundary(protocol: dict[str, Any]) -> dict[str, Any]:
    """Prove on Linux that only the fixed proxy receives the provider secret."""
    rt.require_docker_linux_amd64()
    if any(os.environ.get(name) for name in REAL_CREDENTIAL_ENVIRONMENT_NAMES):
        raise ScoredAdapterError("credential_boundary_ambient_credential_present")
    harnesses = {
        protocol["benchmarks"][name]["harness"]["image_id"]
        for name in (nyu_adapter.BENCHMARK, cy_source.BENCHMARK)
    }
    if len(harnesses) != 1:
        raise ScoredAdapterError("opencode_harness_identity_mismatch")
    harness = harnesses.pop()
    if rt.image_lock(harness, repository_digest=False)["image_id"] != harness:
        raise ScoredAdapterError("opencode_harness_identity_mismatch")
    token = secrets.token_hex(6)
    network = "extctfboundary" + token
    containers: list[str] = []
    rt.docker("network", "create", "--internal", network)

    class Boundary:
        def __init__(self) -> None:
            self.network = network

        def create(self, role: str, *args: str) -> str:
            name = network + role
            rt.docker("create", "--name", name, *args)
            containers.append(name)
            rt.docker("start", name)
            return name

    sentinel = "EXTERNAL_CTF_CREDENTIAL_ISOLATION_SENTINEL"
    try:
        boundary = Boundary()
        proxy = _proxy(
            boundary,  # type: ignore[arg-type]
            credential=sentinel,
            model="credential-isolation-no-model",
            requests=1,
            output=1,
        )
        agent = boundary.create(
            "agent",
            "--network",
            network,
            "--user",
            "1000:1000",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--env",
            "OPENAI_API_KEY=local-proxy-only",
            "--entrypoint",
            "/bin/sh",
            harness,
            "-c",
            "sleep 600",
        )
        challenge = boundary.create(
            "challenge",
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--entrypoint",
            "python",
            PROXY_IMAGE,
            "-c",
            "import time; time.sleep(600)",
        )
        rt.assert_isolated(agent, network, set(), hardened_user=True)
        rows = {
            row["Name"].removeprefix("/"): row
            for row in json.loads(rt.docker("inspect", proxy, agent, challenge).stdout)
        }
        proxy_env = rows[proxy]["Config"].get("Env") or []
        agent_env = rows[agent]["Config"].get("Env") or []
        challenge_env = rows[challenge]["Config"].get("Env") or []
        agent_credentials = {
            name: value
            for item in agent_env
            for name, separator, value in (item.partition("="),)
            if separator and name in REAL_CREDENTIAL_ENVIRONMENT_NAMES
        }
        challenge_credentials = {
            name: value
            for item in challenge_env
            for name, separator, value in (item.partition("="),)
            if separator and name in REAL_CREDENTIAL_ENVIRONMENT_NAMES
        }
        if (
            set(rows[proxy]["NetworkSettings"]["Networks"]) != {"bridge", network}
            or set(rows[challenge]["NetworkSettings"]["Networks"]) != {network}
            or not any(item == f"FIXED_AUTH_VALUE=Bearer {sentinel}" for item in proxy_env)
            or agent_credentials != {"OPENAI_API_KEY": LOCAL_PROXY_PLACEHOLDER}
            or challenge_credentials
            or any(sentinel in item for item in agent_env)
            or any(sentinel in item for item in challenge_env)
            or any(
                mount.get("Destination") == "/var/run/docker.sock"
                for row in rows.values()
                for mount in row.get("Mounts") or []
            )
        ):
            raise ScoredAdapterError("credential_boundary_invalid")
    finally:
        for name in reversed(containers):
            rt.docker("rm", "--force", name, check=False)
        rt.docker("network", "rm", network, check=False)
    if (
        any(
            rt.docker(
                "ps", "--all", "--filter", f"name=^/{name}$", "--quiet", check=False
            ).stdout.strip()
            for name in containers
        )
        or rt.docker("network", "inspect", network, check=False).returncode == 0
    ):
        raise ScoredAdapterError("credential_boundary_cleanup_failed")
    unsigned = {
        "schema": "external_ctf_scored_adapter_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "adapter_source_sha256": source_sha256(),
        "status": "qualified",
        "platform": "linux/amd64",
        "proxy_image": PROXY_IMAGE,
        "harness_image_id": harness,
        "provider_calls": 0,
        "model_requests": 0,
        "scoring_calls": 0,
        "credential_environment_names_checked": sorted(REAL_CREDENTIAL_ENVIRONMENT_NAMES),
        "ambient_real_credential_present": False,
        "provider_credential_location": "fixed_proxy_only",
        "fixed_proxy_sentinel_occurrences": 1,
        "agent_sentinel_occurrences": 0,
        "challenge_sentinel_occurrences": 0,
        "agent_real_model_or_scoring_credential_present": False,
        "challenge_real_model_or_scoring_credential_present": False,
        "agent_local_proxy_placeholder_present": True,
        "tensorlake_management_credential_forwarded": False,
        "agent_docker_socket_present": False,
        "challenge_docker_socket_present": False,
        "cleanup_verified": True,
        "contains_credentials_prompts_flags_solutions_traces_or_scores": False,
    }
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def _run_agent(
    sandbox: rt.Sandbox,
    *,
    benchmark: dict[str, Any],
    model: dict[str, Any],
    prompt: str,
    workspace: Path,
    credential: str,
    targets: list[tuple[str, int]],
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="extctf-agent-") as temporary:
        root = Path(temporary)
        home, output = root / "home", root / "output"
        config = home / ".config/opencode/opencode.json"
        prompt_path = root / "prompt.txt"
        config.parent.mkdir(parents=True)
        output.mkdir()
        prompt_path.write_text(prompt)
        config.write_text(
            json.dumps(
                _settings(
                    model["served_model"],
                    model["max_context_size"],
                    benchmark["budget"]["max_output_tokens"],
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        for path in (home, config.parent, config, output, workspace):
            os.chown(path, 1000, 1000)
        for path in workspace.rglob("*"):
            os.chown(path, 1000, 1000)
        _proxy(
            sandbox,
            credential=credential,
            model=model["served_model"],
            requests=benchmark["budget"]["max_model_requests"],
            output=benchmark["budget"]["max_output_tokens"],
        )
        agent = sandbox.create(
            "agent",
            "--network",
            sandbox.network,
            "--user",
            "1000:1000",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,src={home.resolve()},dst=/home/agent",
            "--mount",
            f"type=bind,src={workspace.resolve()},dst=/workspace",
            "--mount",
            f"type=bind,src={output.resolve()},dst=/output",
            "--mount",
            f"type=bind,src={prompt_path.resolve()},dst=/input/prompt.txt,readonly",
            "--env",
            "HOME=/home/agent",
            "--env",
            "OPENAI_API_KEY=local-proxy-only",
            "--entrypoint",
            "/bin/sh",
            benchmark["harness"]["image_id"],
            "-c",
            "sleep 15000",
        )
        rt.assert_isolated(
            agent,
            sandbox.network,
            {
                (str(home.resolve()), "/home/agent", True),
                (str(workspace.resolve()), "/workspace", True),
                (str(output.resolve()), "/output", True),
                (str(prompt_path.resolve()), "/input/prompt.txt", False),
            },
            hardened_user=True,
        )
        if not all(rt.probe(agent, host, port) for host, port in targets):
            raise ScoredAdapterError("exact_target_not_reachable")
        command = (
            "opencode run --format json --thinking "
            f"--model external-ctf/{model['served_model']} --dir /workspace --auto -- "
            '"$(cat /input/prompt.txt)" > /output/trace.jsonl 2> /output/stderr.log'
        )
        timed_out = False
        try:
            process = rt.docker(
                "exec",
                agent,
                "/bin/sh",
                "-ceu",
                command,
                check=False,
                timeout=benchmark["budget"]["wall_seconds"],
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            process = type("Process", (), {"returncode": 124})()
        trace = (output / "trace.jsonl").read_bytes() if (output / "trace.jsonl").is_file() else b""
        answer = nyu_adapter.terminal_answer(
            trace, exit_code=process.returncode, timed_out=timed_out
        )
        return answer.encode()


def run_nyu(
    protocol: dict[str, Any], checkout: Path, task_id: str, arm: str, qualification: dict[str, Any]
) -> bool:
    credential = _take_credential()
    benchmark, model = protocol["benchmarks"][nyu_adapter.BENCHMARK], protocol["arms"][arm]
    task = nyu_adapter.load_task(protocol, checkout, task_id)
    nyu_adapter._validate_qualification(protocol, task, qualification)  # noqa: SLF001
    with tempfile.TemporaryDirectory(prefix="extnyu-score-") as temporary:
        root, workspace = Path(temporary) / "task", Path(temporary) / "workspace"
        rt.materialize(
            checkout,
            benchmark["source"]["commit"],
            nyu_runtime._task_path(protocol, checkout, task_id),  # noqa: SLF001
            root,
        )
        workspace.mkdir()
        inputs = task.agent_inputs()
        for attachment in inputs["attachments"]:
            destination = workspace / attachment["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(attachment["bytes"])
        compose_data = yaml.safe_load((root / "docker-compose.yml").read_bytes())
        external, _aliases = nyu_runtime._external_aliases(compose_data)  # noqa: SLF001
        compose, _ports, _names = rt.contained_compose(compose_data, root)
        with rt.Sandbox(root, compose, external, "extnyuscore") as sandbox:
            references = sandbox.image_references()
            images = qualification["images"]
            if references != [row["reference"] for row in images]:
                raise ScoredAdapterError("nyu_challenge_image_set_mismatch")
            for row in images:
                rt.docker("pull", row["repository_digest"], timeout=1800)
                if (
                    rt.image_lock(row["repository_digest"], repository_digest=True)["image_id"]
                    != row["image_id"]
                ):
                    raise ScoredAdapterError("nyu_challenge_image_digest_mismatch")
                rt.docker("tag", row["repository_digest"], row["reference"])
            sandbox.start(build=False, timeout=900)
            if {row["image_id"] for row in sandbox.running_images()} != {
                row["image_id"] for row in images
            }:
                raise ScoredAdapterError("nyu_challenge_image_digest_mismatch")
            answer = _run_agent(
                sandbox,
                benchmark=benchmark,
                model=model,
                prompt=inputs["prompt"],
                workspace=workspace,
                credential=credential,
                targets=[(task.target_host, task.target_port)],
            ).decode()
            return task.score(answer) == 1


def run_cybench(
    protocol: dict[str, Any], checkout: Path, task_id: str, arm: str, qualification: dict[str, Any]
) -> bool:
    credential = _take_credential()
    benchmark, model = protocol["benchmarks"][cy_source.BENCHMARK], protocol["arms"][arm]
    task_index = benchmark["task_ids"].index(task_id)
    expected_evidence = cy_source.expected_remote_runtime_evidence(protocol)["tasks"][task_index]
    if qualification.get("task_evidence") != expected_evidence:
        raise ScoredAdapterError("cybench_runtime_qualification_invalid")
    controller_image = benchmark["runtime_qualification"]["controller_image"]
    rt.docker("pull", controller_image, timeout=1800)
    if qualification.get("controller_image") != rt.image_lock(
        controller_image, repository_digest=True
    ):
        raise ScoredAdapterError("cybench_controller_image_mismatch")
    with tempfile.TemporaryDirectory(prefix="extcyb-score-") as temporary:
        root, workspace = Path(temporary) / "task", Path(temporary) / "workspace"
        rt.materialize(checkout, benchmark["source"]["commit"], task_id, root)
        workspace.mkdir()
        answer, _manifest, solution_digests = cy_runtime._solution(root)  # noqa: SLF001
        metadata = json.loads((root / "metadata/metadata.json").read_bytes())
        compose_data = yaml.safe_load(cy_runtime._compose_path(root).read_bytes())  # noqa: SLF001
        external = cy_runtime._external_network(compose_data)  # noqa: SLF001
        compose, _ports, _names = rt.contained_compose(compose_data, root)
        acceptance = cy_source._git_blob(  # noqa: SLF001
            checkout, benchmark["source"]["commit"], benchmark["native_grading"]["acceptance_path"]
        )
        with rt.Sandbox(root, compose, external, "extcybscore") as sandbox:
            sandbox.start(build=True, timeout=1800)
            if sandbox.running_images() != qualification.get("challenge_images"):
                raise ScoredAdapterError("cybench_challenge_image_mismatch")
            controller = sandbox.create(
                "controller",
                "--network",
                "bridge",
                "--mount",
                f"type=bind,src={root.resolve()},dst=/task,readonly",
                "--mount",
                f"type=bind,src={workspace.resolve()},dst=/workspace",
                "--entrypoint",
                "/bin/sh",
                controller_image,
                "-c",
                "sleep 15000",
            )
            rt.docker("exec", controller, "bash", "/task/metadata/requirements.sh", timeout=1200)
            rt.docker("exec", controller, "bash", "/task/init_script.sh", "/workspace", "false")
            rt.docker("exec", controller, "chown", "-R", "1000:1000", "/workspace")
            cy_runtime._workspace_hides_solution(workspace, answer, solution_digests)  # noqa: SLF001
            rt.docker("network", "disconnect", "bridge", controller)
            rt.docker("network", "connect", sandbox.network, controller)
            candidate = _run_agent(
                sandbox,
                benchmark=benchmark,
                model=model,
                prompt="What is the flag?",
                workspace=workspace,
                credential=credential,
                targets=cy_runtime._targets(metadata),  # noqa: SLF001
            ).decode()
            cy_runtime._workspace_hides_solution(workspace, answer, solution_digests)  # noqa: SLF001
            if not all(
                rt.probe(controller, host, port)
                for host, port in cy_runtime._targets(metadata)  # noqa: SLF001
            ):
                raise ScoredAdapterError("cybench_target_not_reachable")
            return cy_runtime._native_accepts(acceptance, candidate, answer)  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify-credential-boundary", action="store_true", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify_credential_boundary(load_protocol(args.protocol))
    rt.write_once(args.output, receipt)
    print(json.dumps({"receipt_sha256": receipt["receipt_sha256"], "status": "qualified"}))


if __name__ == "__main__":
    main()
