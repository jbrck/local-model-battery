"""Prerequisite resolution: which legs can run here, and what is missing when they cannot.

A missing dependency is a skipped leg with a reason, never a crash and never a silent zero.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from scripts.lib.config import get_dotted, resolve_optional_path, sha256_prefix
from scripts.lib.env_probe import HostFacts
from scripts.lib.legs import Leg
from scripts.lib.server_probe import ServerFacts

NLTK_DATA_DIRS = ("NLTK_DATA",)
NLTK_SUBDIR = "tokenizers/punkt"


@dataclass(frozen=True)
class Dependency:
    """One prerequisite: present, or absent with the reason."""

    name: str
    available: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class Judge:
    """The scoring model for the writing suite. It must not be the model under test.

    `pinned` is true when the standard supplied the model rather than the operator,
    which is what makes two runs' writing scores comparable.
    """

    model: str | None = None
    endpoint: str | None = None
    under_test: str | None = None
    pinned: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.model and self.endpoint)

    @property
    def is_self_judging(self) -> bool:
        return bool(self.model and self.under_test and self.model == self.under_test)


def _external_entry(standard: dict, entry_id: str) -> dict:
    for entry in get_dotted(standard, "data.external", []) or []:
        if entry.get("id") == entry_id:
            return entry
    return {}


def _tool_dependency(host: HostFacts, name: str) -> Dependency:
    fact = host.tool(name)
    if fact and fact.available:
        return Dependency(name, True, f"{fact.path} ({fact.version or 'version unknown'})")
    return Dependency(name, False, f"{name} is not on PATH")


def _path_dependency(standard: dict, entry_id: str, want_file: bool) -> Dependency:
    """Check an external data path, and its hash when the manifest pins one."""
    entry = _external_entry(standard, entry_id)
    if not entry:
        return Dependency(entry_id, False, f"{entry_id} is not declared in standard.toml")
    resolved = resolve_optional_path(str(entry.get("path") or ""),
                                     entry.get("path_env") or None)
    if resolved is None:
        hint = entry.get("fetch") or "see standard.toml"
        return Dependency(entry_id, False, f"not found at {entry.get('path')!r}; fetch: {hint}")
    if want_file and not resolved.is_file():
        return Dependency(entry_id, False, f"{resolved} is not a file")
    if not want_file and not resolved.is_dir():
        return Dependency(entry_id, False, f"{resolved} is not a directory")
    pinned = str(entry.get("sha256_16") or "")
    if pinned and resolved.is_file():
        actual = sha256_prefix(resolved)
        if actual != pinned:
            return Dependency(entry_id, False,
                              f"hash mismatch at {resolved}: {actual} != {pinned}")
    return Dependency(entry_id, True, str(resolved))


def _nltk_dependency() -> Dependency:
    if importlib.util.find_spec("nltk") is None:
        return Dependency("nltk_data", False, "the nltk module is not installed")
    for env_name in NLTK_DATA_DIRS:
        root = os.environ.get(env_name)
        if root and (Path(root) / NLTK_SUBDIR).exists():
            return Dependency("nltk_data", True, f"{root}/{NLTK_SUBDIR}")
    return Dependency("nltk_data", False,
                      "nltk is installed but no punkt corpus is on NLTK_DATA")


def resolve_dependency(name: str, standard: dict, host: HostFacts,
                       judge: Judge | None = None) -> Dependency:
    """Resolve one prerequisite by name. Unknown names are reported, not ignored."""
    if name == "docker_sandbox":
        return _tool_dependency(host, "docker")
    if name == "benchlocal_cli":
        return _tool_dependency(host, "benchlocal-cli")
    if name == "tool_eval_bench":
        return _tool_dependency(host, "tool-eval-bench")
    if name == "lcb_source":
        return _path_dependency(standard, "lcb_source", want_file=False)
    if name == "lcb_data":
        return _path_dependency(standard, "lcb_data", want_file=True)
    if name == "nltk_data":
        return _nltk_dependency()
    if name == "judge_endpoint":
        return _judge_dependency(judge)
    return Dependency(name, False, f"{name} has no check implemented")


def _judge_dependency(judge: Judge | None) -> Dependency:
    if judge is None or not judge.configured:
        return Dependency("judge_endpoint", False,
                          "no judge model configured, so the writing suite can generate but "
                          "cannot be scored")
    if judge.is_self_judging:
        return Dependency("judge_endpoint", False,
                          f"the judge is the model under test ({judge.model})")
    return Dependency("judge_endpoint", True, f"{judge.model} at {judge.endpoint}")


def resolve_prereqs(leg: Leg, standard: dict, host: HostFacts,
                    judge: Judge | None = None) -> tuple[Dependency, ...]:
    """Every prerequisite this leg declares, resolved."""
    return tuple(resolve_dependency(name, standard, host, judge) for name in leg.prereqs)


def unavailable(dependencies: tuple[Dependency, ...]) -> tuple[Dependency, ...]:
    return tuple(dependency for dependency in dependencies if not dependency.available)


def verify_bundled_data(standard: dict) -> list[str]:
    """Check every bundled dataset against its pinned hash. Returns problems."""
    problems = []
    for entry in get_dotted(standard, "data.bundled", []) or []:
        pinned = str(entry.get("sha256_16") or "")
        resolved = resolve_optional_path(str(entry.get("file") or ""))
        if resolved is None:
            problems.append(f"bundled dataset {entry.get('file')} is missing")
            continue
        actual = sha256_prefix(resolved)
        if pinned and actual != pinned:
            problems.append(f"bundled dataset {entry.get('file')} hash {actual} != {pinned}")
    return problems


def thinking_available(server: ServerFacts) -> tuple[bool, str]:
    """Whether the endpoint honours the thinking flag, and how we know."""
    if server.thinking_switch is True:
        return True, "the flag changes the output"
    if server.thinking_switch is False:
        return False, "the template ignores the flag"
    return True, "not measured; the flag is assumed to work"
