"""Host power control and Redis diagnostics for kamerad.

HTTP endpoints set desired/actual state in memory and trigger supervisor
one-shot shutdown/reboot programs. A background loop publishes the current
desired vs actual state to Redis for observability only (not command input).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any, Dict, Optional

import redis
from loguru import logger


class PowerManager(object):
    def __init__(
        self,
        supervisor_proxy,
        redis_host: Optional[str] = None,
        hostname: Optional[str] = None,
        diagnostics_period: float = 2.0,
    ):
        self._supervisor = supervisor_proxy
        self.hostname = hostname or os.environ.get(
            "NODE_HOSTNAME", socket.gethostname()
        )
        self._redis = (
            redis.Redis(host=redis_host, decode_responses=True)
            if redis_host
            else None
        )
        self._period = diagnostics_period
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.desired = "idle"
        self.actual = "idle"
        self.status = "ready"
        self.updated_at = time.time()

    def _redis_key(self, field: str) -> str:
        return "/sys/{}/power/{}".format(self.hostname, field)

    def publish_diagnostics(self) -> None:
        if not self._redis:
            return
        with self._lock:
            state = {
                "desired": self.desired,
                "actual": self.actual,
                "status": self.status,
                "updated_at": self.updated_at,
            }
        pipe = self._redis.pipeline()
        for field, value in state.items():
            key = self._redis_key(field)
            if isinstance(value, str):
                pipe.set(key, value)
            else:
                pipe.set(key, str(value))
        pipe.execute()

    def diagnostics_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.publish_diagnostics()
            except Exception as exc:
                logger.warning("power diagnostics publish failed: {}", exc)
            self._stop.wait(self._period)

    def start_diagnostics(self) -> None:
        if not self._redis:
            logger.warning(
                "REDIS_HOST not set; power diagnostics loop disabled on {}",
                self.hostname,
            )
            return
        thread = threading.Thread(
            target=self.diagnostics_loop, name="power-diagnostics", daemon=True
        )
        thread.start()
        logger.info(
            "Started power diagnostics loop for {} (redis={})",
            self.hostname,
            self._redis.connection_pool.connection_kwargs.get("host"),
        )

    def stop_diagnostics(self) -> None:
        self._stop.set()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
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
        with self._lock:
            self.desired = desired
            self.actual = "pending"
            self.status = "requested {}".format(desired)
            self.updated_at = time.time()
        self.publish_diagnostics()

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
            with self._lock:
                self.actual = "failed"
                self.status = str(exc)
                self.updated_at = time.time()
            self.publish_diagnostics()
            return {"ok": False, "error": str(exc)}

        with self._lock:
            self.actual = actual_state
            self.status = "executing {}".format(desired)
            self.updated_at = time.time()
        self.publish_diagnostics()
        result = self.get_status()
        result["ok"] = True
        return result
