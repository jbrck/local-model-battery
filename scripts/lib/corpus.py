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
PASS_SEED_STRIDE = 1_000_003


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


def corpus_pass(corpus: list[str], seed: int, pass_number: int) -> str:
    """One pass over the corpus in a seeded order.

    Pass 0 keeps every piece intact. A later pass also shuffles the words inside each piece,
    so a second trip through the same text shares no run of words with the first. A drafter
    that copies from earlier context would otherwise find the repeat and read fast at depth.
    """
    rng = random.Random(seed + pass_number * PASS_SEED_STRIDE)
    order = list(corpus)
    rng.shuffle(order)
    if pass_number == 0:
        return JOINER.join(order)
    pieces = []
    for piece in order:
        words = piece.split()
        rng.shuffle(words)
        pieces.append(" ".join(words))
    return JOINER.join(pieces)


def build_context(corpus: list[str], chars: int, seed: int) -> str:
    """Seeded filler cut to `chars` characters at a word boundary.

    One pass holds about a million characters. A longer ask keeps going with more passes
    (see corpus_pass), so a million-token prompt is possible from the same data files.
    """
    if chars <= 0:
        return ""
    if not corpus:
        raise ValueError("the corpus is empty, so there is no filler to build")
    passes: list[str] = []
    have = 0
    while have < chars:
        text = corpus_pass(corpus, seed, len(passes))
        passes.append(text)
        have += len(text) + len(JOINER)
    text = JOINER.join(passes)[:chars]
    cut = text.rfind(" ")
    return text[:cut] if cut > chars // 2 else text
