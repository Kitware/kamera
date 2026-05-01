#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable
import threading
import time
from collections import deque
from queue import Queue

from roskv.util import MyTimeoutError


class ConditionalRendezvous(object):
    """A limited-size deque which can block on waiting for a boolean to evaluate.
    This class also functions as a barrier / multi-threaded rendezvous"""

    def __init__(self, maxlen=1):
        self._value = None
        self._deque = deque(maxlen=maxlen)
        self._cond = threading.Condition()
        self._release = deque(maxlen=1)
        self._release.append(False)

    def release(self, window=0.01):
        print('release')
        now = time.time()
        until = now + window
        with self._cond:
            self._release.append(until)
            self._cond.notify_all()

    def put(self, value):
        print('put: {}'.format(value))
        with self._cond:
            self._deque.append(value)
            self._cond.notify_all()

    def contains(self, value):
        with self._cond:
            return value in self._deque

    def present(self, value, timeout=None, retries=None):
        # type: (Any, float, float) -> bool
        """Block/fail until value is present in the rendezvous"""
        return self.wait_for(lambda x: x == value, timeout=timeout, retries=retries)

    def absent(self, value, timeout=None, retries=None):
        # type: (Any, float, float) -> bool
        """Block/fail until value isn't presently in the rendezvous"""
        attempt = 0
        retries = 1e99 if retries is None else retries
        while attempt < retries:
            with self._cond:
                if value not in self._deque:
                    return True
                self._cond.wait(timeout=timeout)
                attempt += 1
        raise MyTimeoutError('Timed out after {} attempts'.format(attempt))

    def wait_for(self, func, timeout=None, retries=None):
        # type: (Any, float, float) -> bool
        """Block/fail until `func` evaluates true on at least one element"""
        attempt = 0
        retries = 1e99 if retries is None else retries
        while attempt < retries:
            with self._cond:
                out = filter(func, self._deque)
                if out:
                    return True
                until_time = self._release[0]
                if time.time() < until_time:
                    return True
                self._cond.wait(timeout=timeout)
                attempt += 1
        raise MyTimeoutError('Timed out after {} attempts'.format(attempt))
