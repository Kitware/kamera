# -*- coding: utf-8 -*-
"""rclpy backend for the wx GUI.

The GUI runs the wx main loop on the main thread and talks to ROS from wx
event handlers and worker threads. This module owns a single rclpy node
serviced by a background executor thread, and exposes the small imperative
surface the GUI needs (subscriptions, synchronous service calls, logging).
"""
from __future__ import division, print_function

import threading
import time as _time

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

_node = None            # type: Node
_executor = None
_spin_thread = None
_log = rclpy.logging.get_logger("wxpython_gui")
_throttle_marks = {}


class ServiceException(Exception):
    """Raised when a service is unavailable or the call fails/times out."""
    pass


class _ServiceNamespace(object):
    """Back-compat shim so call sites can reference ros.service.ServiceException."""
    ServiceException = ServiceException


service = _ServiceNamespace()


def init_node(name, anonymous=False):
    """Initialize rclpy, create the GUI node, and start the background spinner."""
    global _node, _executor, _spin_thread
    if _node is not None:
        return _node
    if anonymous:
        name = "%s_%d" % (name, int(_time.time() * 1e6) % 1000000)
    rclpy.init()
    _node = rclpy.create_node(name)
    _executor = MultiThreadedExecutor(num_threads=4)
    _executor.add_node(_node)
    _spin_thread = threading.Thread(target=_executor.spin, daemon=True)
    _spin_thread.start()
    return _node


def node():
    if _node is None:
        raise RuntimeError("rosnode.init_node() must be called first")
    return _node


def is_shutdown():
    return not rclpy.ok()


def now_sec():
    return node().get_clock().now().nanoseconds * 1e-9


def stamp_to_sec(stamp):
    """builtin_interfaces/Time -> float unix seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def Subscriber(topic, msg_type, callback, callback_args=None, queue_size=10):
    if callback_args is not None:
        wrapped = lambda msg: callback(msg, callback_args)
    else:
        wrapped = callback
    return node().create_subscription(msg_type, topic, wrapped, queue_size)


class ServiceProxy(object):
    """Synchronous service client mirroring rospy.ServiceProxy call semantics.

    Calls block the calling (wx/worker) thread while the background executor
    services the future. Raises ServiceException on unavailability or timeout.
    """

    def __init__(self, topic, srv_type, persistent=False, wait_timeout=2.0,
                 call_timeout=30.0):
        self._topic = topic
        self._srv_type = srv_type
        self._wait_timeout = wait_timeout
        self._call_timeout = call_timeout
        self._client = node().create_client(srv_type, topic)

    def call(self, *args, **kwargs):
        if args:
            # map positional args onto request fields in declaration order
            req = self._srv_type.Request()
            fields = list(req.get_fields_and_field_types().keys())
            for value, field in zip(args, fields):
                setattr(req, field, value)
            for key, value in kwargs.items():
                setattr(req, key, value)
        else:
            req = self._srv_type.Request(**kwargs)

        if not self._client.wait_for_service(timeout_sec=self._wait_timeout):
            raise ServiceException(
                "service [%s] unavailable" % self._topic)
        future = self._client.call_async(req)
        deadline = _time.time() + self._call_timeout
        while not future.done():
            if _time.time() > deadline:
                self._client.remove_pending_request(future)
                raise ServiceException(
                    "service [%s] call timed out" % self._topic)
            if not rclpy.ok():
                raise ServiceException(
                    "service [%s] interrupted by shutdown" % self._topic)
            _time.sleep(0.005)
        if future.exception() is not None:
            raise ServiceException(
                "service [%s] call failed: %s" % (self._topic, future.exception()))
        return future.result()

    __call__ = call


class Rate(object):
    def __init__(self, hz):
        self._period = 1.0 / hz
        self._last = _time.monotonic()

    def sleep(self):
        elapsed = _time.monotonic() - self._last
        remaining = self._period - elapsed
        if remaining > 0:
            _time.sleep(remaining)
        self._last = _time.monotonic()


def loginfo(msg, *args):
    _log.info(str(msg) % args if args else str(msg))


def logwarn(msg, *args):
    _log.warning(str(msg) % args if args else str(msg))


def logerr(msg, *args):
    _log.error(str(msg) % args if args else str(msg))


def logwarn_throttle(period, msg):
    key = str(msg)[:64]
    now = _time.monotonic()
    last = _throttle_marks.get(key, 0)
    if now - last >= period:
        _throttle_marks[key] = now
        _log.warning(str(msg))


def shutdown():
    global _node, _executor, _spin_thread
    if _executor is not None:
        _executor.shutdown()
    if _node is not None:
        _node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    _node = None
    _executor = None
    _spin_thread = None
