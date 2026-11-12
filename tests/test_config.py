"""Config and standard tests. Stdlib only."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.config import (  # noqa: E402
    deep_merge,
    file_md5,
    get_dotted,
    load_toml,
    repo_path,
    sha256_prefix,
)

STANDARD = "standard.toml"

ALL_LEG_IDS = (
    "selftest", "math500", "humaneval_plus", "gpqa_diamond", "instr_v2", "refusal",
    "reasoning", "ifeval", "ifbench", "toolbench", "mmlu", "hermesagent20", "lcb",
    "writing",
)


class ConfigTest(unittest.TestCase):
    def test_load_toml_reads_relative_path_from_repo_root(self):
        data = load_toml("standard.toml")
        self.assertIn("standard_version", data)

    def test_load_toml_reports_the_resolved_path_when_missing(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_toml("does/not/exist.toml")
        self.assertIn("config file not found", str(caught.exception))

    def test_deep_merge_keeps_untouched_siblings(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        self.assertEqual(deep_merge(base, {"b": {"c": 9}}), {"a": 1, "b": {"c": 9, "d": 3}})

    def test_get_dotted_returns_default_for_missing_branch(self):
        self.assertEqual(get_dotted({"a": {"b": 1}}, "a.b"), 1)
        self.assertIsNone(get_dotted({"a": {"b": 1}}, "a.z"))
        self.assertEqual(get_dotted({"a": {"b": 1}}, "x.y", 5), 5)

    def test_repo_path_is_anchored_at_the_repository_root(self):
        self.assertTrue(
            str(repo_path("standard.toml")).endswith("local-model-battery/standard.toml"))

    def test_sha256_prefix_is_a_truncated_digest(self):
        payload = "battery\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            full = hashlib.sha256(payload.encode()).hexdigest()
            prefix = sha256_prefix(path)
            self.assertIsInstance(prefix, str)
            self.assertEqual(prefix, full[:16])
            self.assertEqual(len(prefix), 16)
        finally:
            os.unlink(path)

    def test_sha256_prefix_returns_none_for_unreadable_file(self):
        self.assertIsNone(sha256_prefix("/nonexistent/file.json"))

    def test_file_md5_matches_the_hashlib_digest(self):
        payload = b"battery data\n"
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            self.assertEqual(file_md5(path), hashlib.md5(payload).hexdigest())
        finally:
            os.unlink(path)

    def test_file_md5_streams_across_chunk_boundaries(self):
        # One full 1 MiB chunk plus a remainder, so the streaming loop takes two bites.
        payload = b"x" * (1024 * 1024 + 7)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            self.assertEqual(file_md5(path), hashlib.md5(payload).hexdigest())
        finally:
            os.unlink(path)

    def test_file_md5_is_empty_for_an_unreadable_file(self):
        # The manifest treats a missing digest as a mismatch, never as a pass.
        self.assertEqual(file_md5("/nonexistent/file.bin"), "")


class StandardTest(unittest.TestCase):
    def setUp(self):
        self.standard = load_toml(STANDARD)

    def test_version_is_declared_and_older_versions_are_named(self):
        self.assertEqual(self.standard["standard_version"], "v2")
        self.assertIn("v1", self.standard["incomparable_with"])

    def test_lcb_cap_plus_headroom_fits_the_standard_window(self):
        lcb_cap = self.standard["caps"]["lcb"]
        self.assertLessEqual(lcb_cap + 16384, self.standard["serve"]["ctx"])

    def test_seed_and_temperature_are_frozen_in_the_standard(self):
        self.assertIsInstance(self.standard["sampling"]["seed"], int)
        self.assertEqual(self.standard["sampling"]["temperature"], 0.0)

    def test_every_leg_has_a_budget_floor_and_an_item_count(self):
        for leg_id in ALL_LEG_IDS:
            self.assertIn(leg_id, self.standard["budget"]["floors"], leg_id)
            self.assertIn(leg_id, self.standard["budget"]["items"], leg_id)

    def test_limits_match_the_item_counts(self):
        items = self.standard["budget"]["items"]
        limits = self.standard["limits"]
        self.assertEqual(items["mmlu"], limits["mmlu"])
        self.assertEqual(items["lcb"], limits["lcb"])
        self.assertEqual(items["ifbench"], limits["ifbench"])
        self.assertEqual(items["refusal"], limits["refusal"])

    def test_bundled_data_entries_declare_a_hash_and_an_item_count(self):
        bundled = self.standard["data"]["bundled"]
        declared = {entry["file"] for entry in bundled}
        legs = load_toml("legs.toml")["leg"]
        wanted = {leg["data"] for leg in legs if leg.get("data")}
        self.assertEqual(declared, wanted)
        for entry in bundled:
            self.assertEqual(len(entry["sha256_16"]), 16, entry["file"])
            self.assertGreater(entry["items"], 0, entry["file"])

    def test_external_data_names_where_it_comes_from(self):
        for entry in self.standard["data"]["external"]:
            self.assertTrue(entry["fetch"], entry["id"])
            self.assertTrue(entry["license"], entry["id"])

    def test_lcb_data_pin_is_recorded(self):
        lcb = [e for e in self.standard["data"]["external"] if e["id"] == "lcb_data"][0]
        self.assertEqual(lcb["sha256_16"], "bb4c364f71921c44")


if __name__ == "__main__":
    unittest.main()
