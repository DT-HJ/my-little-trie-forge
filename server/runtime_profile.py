"""Runtime profile: resource limits an operator can change while serving.

The HTTP front end validates a request and writes the native settings to a
private control directory; the native process applies them and publishes what
it applied. The request limit lives in this process and takes effect at once.
"""

import json
import os
import threading

PROFILE_FILE = "profile"
APPLIED_FILE = "applied"
NOTES = {
    "memory_limit": "applies when no request is active; the next request "
    "uses it",
    "reclaim_now": "runs when no request is active",
}


class ProfileError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _integer(body, key, low, high):
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ProfileError(400, f"{key} must be an integer from {low} to {high}")
    return value


class ProfileController:
    def __init__(self, directory, parse_memory, server, request_limits=(1, 4)):
        self.directory = directory
        self.parse_memory = parse_memory
        self.server = server
        self.request_limits = request_limits
        self.lock = threading.Lock()
        self.requested = {}

    def _applied(self):
        try:
            with open(os.path.join(self.directory, APPLIED_FILE), encoding="ascii") as handle:
                lines = handle.read().splitlines()
        except (OSError, UnicodeDecodeError):
            return None
        applied = {}
        for line in lines:
            key, separator, value = line.partition("=")
            if not separator or not value.isdigit():
                return None
            applied[key] = int(value)
        return applied

    def snapshot(self):
        applied = self._applied()
        admission = self.server.generation_admission
        return {
            "applied": None
            if applied is None
            else {
                "gpu_duty_percent": applied.get("duty_percent"),
                "paused": bool(applied.get("paused")),
                "memory_limit_bytes": applied.get("memory_limit_bytes"),
                "memory_limit_pending_bytes": applied.get("memory_limit_pending_bytes") or None,
                "memory_ceiling_bytes": applied.get("memory_ceiling_bytes"),
            },
            "max_concurrent_requests": None if admission is None else admission.stats(),
        }

    def _write(self):
        lines = [f"{key}={value}" for key, value in sorted(self.requested.items())]
        path = os.path.join(self.directory, PROFILE_FILE)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="ascii") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(temporary, path)

    def update(self, body):
        if not isinstance(body, dict) or not body:
            raise ProfileError(400, "send a JSON object with at least one setting")
        unknown = sorted(set(body) - {
            "gpu_duty_percent", "paused", "max_concurrent_requests",
            "memory_limit", "reclaim_now",
        })
        if unknown:
            raise ProfileError(400, f"unknown setting: {unknown[0]}")
        native = {}
        notes = []
        if "gpu_duty_percent" in body:
            native["duty_percent"] = _integer(body, "gpu_duty_percent", 1, 100)
        if "paused" in body:
            if not isinstance(body["paused"], bool):
                raise ProfileError(400, "paused must be true or false")
            native["paused"] = int(body["paused"])
        if "memory_limit" in body:
            value = body["memory_limit"]
            ceiling = (self._applied() or {}).get("memory_ceiling_bytes")
            if ceiling is None:
                raise ProfileError(503, "engine is not ready; retry shortly")
            if value is None or value == "auto":
                native["memory_limit_bytes"] = 0
            else:
                if isinstance(value, str):
                    try:
                        value = self.parse_memory(value)
                    except Exception:
                        value = None
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ProfileError(
                        400, "memory_limit must be null, a byte count or a size such as 48G"
                    )
                if value > ceiling:
                    raise ProfileError(
                        400,
                        f"memory_limit cannot exceed the startup ceiling of {ceiling} bytes; "
                        "restart with a larger --max-memory",
                    )
                native["memory_limit_bytes"] = value
            notes.append(NOTES["memory_limit"])
        if "reclaim_now" in body:
            if body["reclaim_now"] is not True:
                raise ProfileError(400, "reclaim_now must be true")
            notes.append(NOTES["reclaim_now"])
        limit = None
        if "max_concurrent_requests" in body:
            limit = _integer(body, "max_concurrent_requests", *self.request_limits)
            if self.server.generation_admission is None:
                raise ProfileError(
                    409, "start with --max-concurrent-requests to change the request limit"
                )
        with self.lock:
            if "reclaim_now" in body:
                native["reclaim_seq"] = self.requested.get("reclaim_seq", 0) + 1
            if native:
                self.requested.update(native)
                self._write()
            if limit is not None:
                self.server.generation_admission.set_capacity(limit)
        result = self.snapshot()
        result["requested"] = dict(self.requested)
        if notes:
            result["notes"] = notes
        return result
