"""The depth filler: reproducible, non-repeating, and honest about how much it can supply."""
import json
import os
import tempfile
import unittest

from scripts.lib.corpus import (JOINER, build_context, corpus_chars, corpus_hash, corpus_pass,
                                load_corpus)

PIECES = [f"piece {index:03d} " + "lorem ipsum dolor " * 4 for index in range(60)]


class LoadCorpusTest(unittest.TestCase):
    def test_nested_strings_are_collected_deduplicated_and_short_ones_dropped(self):
        long_one = "a sentence long enough to count as real filler text here"
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "one.json"), "w") as handle:
                json.dump({"q": [long_one, "short"], "n": {"deep": [long_one]}}, handle)
            with open(os.path.join(tmp, "two.json"), "w") as handle:
                json.dump([{"text": long_one + " and a second one"}], handle)
            corpus = load_corpus(tmp, ("one.json", "two.json"))
        self.assertEqual(corpus, sorted([long_one, long_one + " and a second one"]))

    def test_the_bundled_datasets_supply_a_corpus_big_enough_for_a_deep_rung(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus = load_corpus(root)
        self.assertGreater(corpus_chars(corpus), 1_000_000)


class BuildContextTest(unittest.TestCase):
    def test_the_same_seed_gives_the_same_text(self):
        self.assertEqual(build_context(PIECES, 800, seed=1), build_context(PIECES, 800, seed=1))

    def test_a_different_seed_gives_a_different_text(self):
        self.assertNotEqual(build_context(PIECES, 800, seed=1), build_context(PIECES, 800, seed=2))

    def test_the_text_is_cut_to_length_at_a_word_boundary(self):
        text = build_context(PIECES, 800, seed=3)
        self.assertLessEqual(len(text), 800)
        self.assertGreater(len(text), 700)
        self.assertNotIn(text[-1], " \n")

    def test_no_piece_repeats_inside_one_context(self):
        text = build_context(PIECES, corpus_chars(PIECES) - 10, seed=4)
        marks = [piece[:9] for piece in PIECES if piece[:9] in text]
        self.assertEqual(len(marks), len(set(marks)))
        self.assertEqual(text.count("piece 007 "), 1)

    def test_asking_for_more_than_one_pass_keeps_going_to_the_full_length(self):
        want = corpus_chars(PIECES) * 3
        text = build_context(PIECES, want, seed=1)
        self.assertLessEqual(len(text), want)
        self.assertGreater(len(text), want - 40)

    def test_a_second_pass_shares_no_run_of_words_with_the_first(self):
        pieces = [" ".join(f"w{index}x{word}" for word in range(30)) for index in range(40)]
        first = corpus_pass(pieces, 5, 0).split()
        second = corpus_pass(pieces, 5, 1).split()
        run = 6
        seen = {tuple(first[i:i + run]) for i in range(len(first) - run)}
        repeats = sum(tuple(second[i:i + run]) in seen for i in range(len(second) - run))
        self.assertEqual(repeats, 0)

    def test_a_second_pass_keeps_the_same_words(self):
        self.assertEqual(sorted(corpus_pass(PIECES, 5, 0).split()),
                         sorted(corpus_pass(PIECES, 5, 1).split()))

    def test_an_empty_corpus_cannot_supply_filler(self):
        with self.assertRaises(ValueError):
            build_context([], 100, seed=1)

    def test_zero_characters_is_an_empty_string(self):
        self.assertEqual(build_context(PIECES, 0, seed=1), "")

    def test_pieces_are_joined_with_the_declared_separator(self):
        self.assertIn(JOINER, build_context(PIECES, 400, seed=1))


class CorpusHashTest(unittest.TestCase):
    def test_the_hash_is_stable_and_sixteen_hex_digits(self):
        first = corpus_hash(PIECES)
        self.assertEqual(first, corpus_hash(list(PIECES)))
        self.assertRegex(first, r"^[0-9a-f]{16}$")

    def test_changing_one_piece_changes_the_hash(self):
        edited = list(PIECES)
        edited[0] += "!"
        self.assertNotEqual(corpus_hash(PIECES), corpus_hash(edited))

    def test_piece_boundaries_matter(self):
        self.assertNotEqual(corpus_hash(["ab", "c"]), corpus_hash(["a", "bc"]))


if __name__ == "__main__":
    unittest.main()
