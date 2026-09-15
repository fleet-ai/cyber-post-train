from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "evals/fleet/images/sglang-ray/Dockerfile.jobs-api-dind"
WORKFLOW = ROOT / ".github/workflows/publish-jobs-api-dind-runtime.yml"


def test_runtime_reproduces_exact_qualified_base_and_ray() -> None:
    text = DOCKERFILE.read_text()
    base = "lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1"
    assert f"FROM {base}" in text
    assert f'fleet.cyber.base-image="{base}"' in text
    assert "RAY_VERSION=2.56.0" in text
    assert 'fleet.cyber.ray-version="2.56.0"' in text
    assert '"ray[default]==${RAY_VERSION}"' in text
    assert 'importlib.metadata.version("ray") == "2.56.0"' in text
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
