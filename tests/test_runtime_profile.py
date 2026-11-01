import contextlib
import http.client
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace

from server.runtime_profile import ProfileController, ProfileError
from server.server import FrontendServer, WaitingAdmission, _parse_max_memory
from tests.test_max_concurrent_requests import _app, _running

CEILING = 96 * 1024**3


def _write_applied(directory, **overrides):
    values = {
        "duty_percent": 100,
        "paused": 0,
        "memory_limit_bytes": CEILING,
        "memory_limit_pending_bytes": 0,
        "memory_ceiling_bytes": CEILING,
        "reclaim_seq": 0,
    }
    values.update(overrides)
    with open(os.path.join(directory, "applied"), "w") as handle:
        handle.write("".join(f"{k}={v}\n" for k, v in values.items()))


def _requested(directory):
    with open(os.path.join(directory, "profile")) as handle:
        return dict(line.split("=") for line in handle.read().split())


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory))
        self.server = SimpleNamespace(generation_admission=WaitingAdmission(3))
        self.controller = ProfileController(
            self.directory, _parse_max_memory, self.server
        )
        _write_applied(self.directory)

    def test_settings_reach_the_profile_file(self):
        result = self.controller.update(
            {"gpu_duty_percent": 30, "paused": True, "memory_limit": "48G"}
        )
        self.assertEqual(
            _requested(self.directory),
            {
                "duty_percent": "30",
                "paused": "1",
                "memory_limit_bytes": str(48 * 1024**3),
            },
        )
        self.assertEqual(result["applied"]["memory_ceiling_bytes"], CEILING)
        self.assertIn("notes", result)

    def test_settings_accumulate_and_reclaim_counts_up(self):
        self.controller.update({"gpu_duty_percent": 30})
        self.controller.update({"reclaim_now": True})
        self.controller.update({"reclaim_now": True})
        self.assertEqual(
            _requested(self.directory), {"duty_percent": "30", "reclaim_seq": "2"}
        )

    def test_memory_limit_null_returns_to_the_ceiling(self):
        self.controller.update({"memory_limit": None})
        self.assertEqual(_requested(self.directory), {"memory_limit_bytes": "0"})

    def test_request_limit_changes_the_gate(self):
        self.controller.update({"max_concurrent_requests": 1})
        self.assertEqual(self.server.generation_admission.capacity, 1)
        self.assertFalse(os.path.exists(os.path.join(self.directory, "profile")))

    def test_invalid_settings_change_nothing(self):
        for body, status in (
            ({}, 400),
            ({"gpu_duty_percent": 0}, 400),
            ({"gpu_duty_percent": 101}, 400),
            ({"gpu_duty_percent": True}, 400),
            ({"gpu_duty_percent": "5"}, 400),
            ({"paused": 1}, 400),
            ({"memory_limit": 0}, 400),
            ({"memory_limit": "lots"}, 400),
            ({"memory_limit": "97G"}, 400),
            ({"reclaim_now": False}, 400),
            ({"max_concurrent_requests": 5}, 400),
            ({"unknown": 1}, 400),
            ({"gpu_duty_percent": 30, "paused": "yes"}, 400),
        ):
            with self.subTest(body=body):
                with self.assertRaises(ProfileError) as caught:
                    self.controller.update(body)
                self.assertEqual(caught.exception.status, status)
        self.assertFalse(os.path.exists(os.path.join(self.directory, "profile")))
        self.assertEqual(self.server.generation_admission.capacity, 3)

    def test_memory_limit_needs_a_published_ceiling(self):
        os.remove(os.path.join(self.directory, "applied"))
        with self.assertRaises(ProfileError) as caught:
            self.controller.update({"memory_limit": "48G"})
        self.assertEqual(caught.exception.status, 503)

    def test_request_limit_needs_a_startup_limit(self):
        self.server.generation_admission = None
        with self.assertRaises(ProfileError) as caught:
            self.controller.update({"max_concurrent_requests": 2})
        self.assertEqual(caught.exception.status, 409)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory))
        _write_applied(self.directory)

    @contextlib.contextmanager
    def _server(self, api_key=None, profile=True):
        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=2, api_key=api_key
        )
        if profile:
            server.profile = ProfileController(
                self.directory, _parse_max_memory, server
            )
        with _running(server):
            yield server

    def _call(self, server, method, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=4
        )
        connection.request(
            method,
            "/admin/profile",
            None if body is None else json.dumps(body),
            {"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, json.loads(data)

    def test_get_and_post(self):
        with self._server() as server:
            status, body = self._call(server, "GET")
            self.assertEqual(status, 200)
            self.assertEqual(body["max_concurrent_requests"]["capacity"], 2)
            self.assertEqual(body["applied"]["gpu_duty_percent"], 100)
            status, body = self._call(
                server, "POST", {"gpu_duty_percent": 40, "max_concurrent_requests": 1}
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["requested"], {"duty_percent": 40})
            self.assertEqual(body["max_concurrent_requests"]["capacity"], 1)

    def test_bad_request_is_rejected_with_a_reason(self):
        with self._server() as server:
            status, body = self._call(server, "POST", {"gpu_duty_percent": 500})
            self.assertEqual(status, 400)
            self.assertIn("gpu_duty_percent", body["error"]["message"])

    def test_missing_profile_is_not_found(self):
        with self._server(profile=False) as server:
            self.assertEqual(self._call(server, "GET")[0], 404)

    def test_api_key_is_required_when_set(self):
        key = "k" * 32
        with self._server(api_key=key) as server:
            self.assertEqual(self._call(server, "GET")[0], 401)
            status, _ = self._call(
                server, "GET", headers={"Authorization": f"Bearer {key}"}
            )
            self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
