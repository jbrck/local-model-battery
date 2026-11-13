"""A leg test that forgets --endpoint must fail fast, not call a live server."""
import os
import unittest

from tests import DEAD_ENDPOINT  # noqa: E402  imported first: it points the default at a dead port
from scripts.lib.harness import DEFAULT_ENDPOINT, leg_config, parse_leg_args


class EndpointGuardTest(unittest.TestCase):
    def test_the_test_run_points_the_default_endpoint_at_a_dead_port(self):
        self.assertEqual(os.environ.get("BATTERY_DEFAULT_ENDPOINT"), DEAD_ENDPOINT)

    def test_the_harness_reads_the_override(self):
        self.assertEqual(DEFAULT_ENDPOINT, DEAD_ENDPOINT)

    def test_a_leg_that_omits_endpoint_gets_the_dead_port(self):
        args = parse_leg_args("test leg", argv=["--model", "m"])
        self.assertEqual(args.endpoint, DEAD_ENDPOINT)

    def test_a_leg_that_omits_endpoint_builds_a_dead_client(self):
        args = parse_leg_args("test leg", argv=["--model", "m"])
        self.assertEqual(leg_config(args).endpoint, DEAD_ENDPOINT)

    def test_an_explicit_endpoint_still_wins(self):
        args = parse_leg_args("test leg", argv=["--model", "m", "--endpoint", "http://stub:9/v1"])
        self.assertEqual(args.endpoint, "http://stub:9/v1")


if __name__ == "__main__":
    unittest.main()
