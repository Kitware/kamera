"""Host power control for kamerad."""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Dict

from loguru import logger


class PowerManager(object):
    def __init__(self, supervisor_proxy, hostname=None):
        self._supervisor = supervisor_proxy
        self.hostname = hostname or os.environ.get(
            "NODE_HOSTNAME", socket.gethostname()
        )
        self.desired = "idle"
        self.actual = "idle"
        self.status = "ready"
        self.updated_at = time.time()

    def get_status(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "desired": self.desired,
            "actual": self.actual,
            "status": self.status,
            "updated_at": self.updated_at,
        }

    def request_shutdown(self) -> Dict[str, Any]:
        return self._request_power("shutdown", "host_shutdown", "shutting_down")

    def request_reboot(self) -> Dict[str, Any]:
        return self._request_power("reboot", "host_reboot", "rebooting")

    def _request_power(
        self, desired: str, supervisor_process: str, actual_state: str
    ) -> Dict[str, Any]:
        self.desired = desired
        self.actual = "pending"
        self.status = "requested {}".format(desired)
        self.updated_at = time.time()

        system_name = os.environ.get("SYSTEM_NAME", "").strip()
        if system_name:
            try:
                self._supervisor.supervisor.stopProcessGroup(system_name, True)
                logger.info("Stopped supervisor group {}", system_name)
            except Exception as exc:
                logger.warning(
                    "stopProcessGroup {} failed: {}", system_name, exc
                )

        try:
            self._supervisor.supervisor.startProcess(supervisor_process, False)
            logger.info("Started supervisor process {}", supervisor_process)
        except Exception as exc:
            self.actual = "failed"
            self.status = str(exc)
            self.updated_at = time.time()
            return {"ok": False, "error": str(exc)}

        # startProcess(wait=False) reports success even when the spawn fails,
        # so watch the one-shot briefly and surface spawn/exit failures.
        error = self._check_oneshot(supervisor_process)
        if error:
            logger.error("{} failed: {}", supervisor_process, error)
            self.actual = "failed"
            self.status = error
            self.updated_at = time.time()
            return {"ok": False, "error": error}

        self.actual = actual_state
        self.status = "executing {}".format(desired)
        self.updated_at = time.time()
        result = self.get_status()
        result["ok"] = True
        return result

    def _check_oneshot(self, name: str, timeout: float = 3.0) -> str:
        """Watch a one-shot supervisor process briefly after starting it.

        Returns an error string if it failed to spawn or exited nonzero,
        empty string on success. The shutdown/reboot commands normally exit 0
        within a second; if the process is still running at the deadline,
        assume the host is on its way down.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                info = self._supervisor.supervisor.getProcessInfo(name)
            except Exception as exc:
                return "getProcessInfo {} failed: {}".format(name, exc)
            state = info.get("statename")
            if state in ("FATAL", "BACKOFF", "UNKNOWN"):
                return "{} failed to spawn ({}): {}".format(
                    name, state, info.get("spawnerr") or self._tail_stderr(name)
                )
            if state == "EXITED":
                if info.get("exitstatus") == 0:
                    return ""
                return "{} exited with status {}: {}".format(
                    name, info.get("exitstatus"), self._tail_stderr(name)
                )
            time.sleep(0.2)
        return ""

    def _tail_stderr(self, name: str) -> str:
        try:
            return self._supervisor.supervisor.readProcessStderrLog(
                name, -1000, 0
            ).strip()
        except Exception:
            return ""
