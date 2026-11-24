"""Filler text for depth tests: unique, seeded, built from the datasets the repo already ships.

A depth test needs a prompt of a given length that the model cannot shortcut. Repeated
text lets a speculative drafter win on every step and flatters the engine, so the filler
is real, non-repeating text: every long string in the bundled datasets, deduplicated,
shuffled by a seed, and joined until the target length is reached. The same data files and
seed give the same filler on any host, and the data files are already hash-pinned in
standard.toml.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Iterator

CORPUS_FILES = (
    "data/writing_suite_v1.json",
    "data/mmlu_test.json",
    "data/gpqa_diamond.json",
    "data/ifbench_prompts.json",
    "data/math500.json",
    "data/humaneval_plus.json",
)
MIN_PIECE_CHARS = 40
JOINER = "\n\n"


def _strings(node: object) -> Iterator[str]:
    """Every string inside a decoded JSON document."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def load_corpus(root: str, files: tuple[str, ...] = CORPUS_FILES) -> list[str]:
    """The sorted, deduplicated long strings of the corpus files under `root`."""
    pieces: set[str] = set()
    for name in files:
        with open(os.path.join(root, name)) as handle:
            pieces.update(text.strip() for text in _strings(json.load(handle))
                          if len(text.strip()) >= MIN_PIECE_CHARS)
    return sorted(pieces)


def corpus_chars(corpus: list[str]) -> int:
    """How many characters of filler one pass over the corpus can supply."""
    return sum(len(piece) + len(JOINER) for piece in corpus)


def corpus_hash(corpus: list[str]) -> str:
    """First 16 hex digits of SHA-256 over the pieces: proves two hosts built from the same text."""
    digest = hashlib.sha256()
    for piece in corpus:
        digest.update(piece.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def build_context(corpus: list[str], chars: int, seed: int) -> str:
    """A seeded shuffle of the corpus, cut to `chars` characters at a word boundary."""
    if chars <= 0:
        return ""
    if chars > corpus_chars(corpus):
        raise ValueError(f"asked for {chars} characters of filler, the corpus holds "
                         f"{corpus_chars(corpus)}")
    order = list(corpus)
    random.Random(seed).shuffle(order)
    text = JOINER.join(order)[:chars]
    cut = text.rfind(" ")
    return text[:cut] if cut > chars // 2 else text
