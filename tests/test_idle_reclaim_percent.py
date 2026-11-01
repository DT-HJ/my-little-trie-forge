import contextlib
import io
import os
import unittest
from unittest.mock import patch

from install import launcher
from server import server as server_module

MODEL = "local/Qwen3.8-27B-q8c"


def _serve_args(value):
    return ["t", "d", "--tokenizer", "x", "--model", MODEL] + (
        ["--idle-reclaim-percent", value] if value is not None else []
    )


def _mltf_args(value):
    return ["serve", "--model", MODEL] + (
        ["--idle-reclaim-percent", value] if value is not None else []
    )


class IdleReclaimPercentTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SPLASH_API_KEY", None)
        os.environ.pop("SPLASH_DEFAULT_REASONING_EFFORT", None)

    def test_parsers_accept_the_supported_range(self):
        for value in ("1", "60", "100"):
            self.assertEqual(
                server_module.parse_args(_serve_args(value)).idle_reclaim_percent,
                int(value),
            )
            self.assertEqual(
                launcher.parse_args(_mltf_args(value)).idle_reclaim_percent,
                int(value),
            )
        self.assertIsNone(server_module.parse_args(_serve_args(None)).idle_reclaim_percent)
        self.assertIsNone(launcher.parse_args(_mltf_args(None)).idle_reclaim_percent)

    def test_parsers_reject_out_of_range_values(self):
        for value in ("0", "101", "-1", "1.5", "x"):
            for parse, args in (
                (server_module.parse_args, _serve_args),
                (launcher.parse_args, _mltf_args),
            ):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parse(args(value))

    def test_native_command_forwards_only_an_explicit_value(self):
        default = server_module._native_command(
            server_module.parse_args(_serve_args(None))
        )
        self.assertNotIn("--idle-reclaim-percent", default)
        command = server_module._native_command(
            server_module.parse_args(_serve_args("75"))
        )
        index = command.index("--idle-reclaim-percent")
        self.assertEqual(command[index + 1], "75")


if __name__ == "__main__":
    unittest.main()
