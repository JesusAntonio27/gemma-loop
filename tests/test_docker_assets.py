"""Static guarantees for the Docker build context (no daemon needed)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_IGNORE = ROOT / ".dockerignore"
DOCKER_IGNORE = ROOT / "docker" / ".dockerignore"

# Patterns the R1 review requires in both ignore files.
REQUIRED_PATTERNS = {
    ".env*",
    "*.pem",
    "*.key",
    "*token*",
    ".netrc",
    ".huggingface/",
    ".cache/",
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.pt",
    "checkpoints/",
    "artifacts",
    ".superpowers",
}


def _patterns(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_ignore_files_stay_in_sync():
    # Docker reads the root file; docker/.dockerignore is the review copy.
    assert _patterns(ROOT_IGNORE) == _patterns(DOCKER_IGNORE)


def test_ignore_files_exclude_secrets_and_weights():
    for path in (ROOT_IGNORE, DOCKER_IGNORE):
        patterns = _patterns(path)
        missing = REQUIRED_PATTERNS - patterns
        assert not missing, f"{path} is missing {sorted(missing)}"
