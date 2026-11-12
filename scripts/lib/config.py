"""Config loading: TOML catalogs, recursive merge, repo-relative paths, data hashing."""
from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HASH_PREFIX_CHARS = 16


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)


def load_toml(path: str | Path) -> dict:
    """Load a TOML file, resolving relative paths against the repository root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_path(str(path))
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with open(resolved, "rb") as handle:
        return tomllib.load(handle)


def deep_merge(base: dict, override: dict) -> dict:
    """Merge two mappings recursively. The override wins on leaf values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_dotted(data: dict, dotted: str, default=None):
    """Read a dotted key path, for example `caps.by_leg.mmlu`."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_optional_path(value: str, env_name: str | None = None) -> Path | None:
    """Resolve a data path, preferring an environment override. None when absent."""
    if env_name:
        override = os.environ.get(env_name)
        if override:
            candidate = Path(override).expanduser()
            return candidate if candidate.exists() else None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_path(value)
    return candidate if candidate.exists() else None


def missing_fields(entry: dict, required: tuple[str, ...]) -> list[str]:
    """Names from `required` that `entry` does not carry."""
    return [name for name in required if name not in entry]


def sha256_prefix(path: str | Path, chars: int = HASH_PREFIX_CHARS) -> str | None:
    """First `chars` hex digits of a file's SHA-256, or None when unreadable."""
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None
    return digest[:chars]


def file_md5(path: str | Path) -> str:
    """The md5 of a file, streamed so a large GGUF does not land in memory.

    Empty string when the file cannot be read: the manifest treats a missing digest as a
    mismatch, never as a pass.
    """
    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def resolve_md5(value: str | Path | None) -> str:
    """The md5 of a config file named by path, resolving relative paths against the repo."""
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_path(str(value))
    return file_md5(candidate)
