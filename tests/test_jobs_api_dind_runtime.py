from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "evals/fleet/images/sglang-ray/Dockerfile.jobs-api-dind"
WORKFLOW = ROOT / ".github/workflows/publish-jobs-api-dind-runtime.yml"


def test_runtime_is_derived_from_exact_qualified_parent() -> None:
    text = DOCKERFILE.read_text()
    parent = (
        "ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@"
        "sha256:ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec"
    )
    assert f"FROM {parent}" in text
    assert f'fleet.cyber.parent-image="{parent}"' in text
    assert "DOCKER_VERSION=27.5.1" in text
    assert (
        "DOCKER_STATIC_SHA256=4f798b3ee1e0140eab5bf30b0edc4e84f4cdb53255a429dc3bbae9524845d640"
    ) in text
    assert (
        'fleet.cyber.opencode-image-id="'
        "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb"
        '"'
    ) in text


def test_publish_is_branch_scoped_amd64_and_immutable_tagged() -> None:
    text = WORKFLOW.read_text()
    assert "branches:\n      - 'codex/jobs-api-dind-runtime-main-v1'" in text
    assert "platforms: linux/amd64" in text
    assert "packages: write" in text
    assert "push: true" in text
    assert "jobs-api-dind-${{ github.sha }}" in text
