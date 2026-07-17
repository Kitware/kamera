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

        self.actual = actual_state
        self.status = "executing {}".format(desired)
        self.updated_at = time.time()
        result = self.get_status()
        result["ok"] = True
        return result
