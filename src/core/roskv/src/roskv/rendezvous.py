#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable
import threading
import time
from six.moves.queue import deque, Queue

from roskv.util import MyTimeoutError

class Governor(object):
    def __init__(self, period=0.01):
        self.period = period
        self._last = time.time()

    def check(self):
        """Check the elapsed time and block if it's fired too recently"""
        now = time.time()
        dt = now - self._last
        if dt < self.period:
            time.sleep(self.period - dt)
        self._last = time.time()


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




class _TestConditionalRendezvous(object):
    """A limited-size deque which can block on waiting for the presence or absence of
    a value"""

    def __init__(self, maxlen=1):
        self._value = None
        self._deque = deque(maxlen=maxlen)
        self._cond = threading.Condition()

    def put(self, value):
        logging.debug('Calling put {}'.format(value))
        with self._cond:
            logging.debug('Put: {}'.format(value))
            self._deque.append(value)
            self._cond.notify_all()

    def contains(self, value):
        with self._cond:
            return value in self._deque

    def present(self, value, timeout=None, retries=None):
        """Block/fail until value is present in the rendezvous"""
        logging.debug('Call present {}'.format(value))
        attempt = 0
        retries = 1e99 if retries is None else retries
        while attempt < retries:
            with self._cond:
                logging.debug('Attempt {}'.format(attempt))
                if value in self._deque:
                    logging.debug('ConditionalRendezvous contains {}'.format(value))
                    return True
                logging.debug('ConditionalRendezvous waiting for {}'.format(value))
                self._cond.wait(timeout=timeout)
                attempt += 1

        raise RuntimeError('ran out after {} attempts'.format(attempt))

    def absent(self, value):
        """Block/fail until value isn't presently in the rendezvous"""
        logging.debug('Call absent {}'.format(value))
        while True:
            with self._cond:
                if value not in self._deque:
                    logging.debug('ConditionalRendezvous lacks {}'.format(value))
                    return True
                logging.debug('ConditionalRendezvous waiting on lack of {}'.format(value))
                self._cond.wait()


if __name__ == '__main__':
    import time
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='(%(threadName)-9s) %(message)s', )


    def pigeon_produce(pg, value=5):
        logging.debug('Producer thread started {} ...'.format(value))
        pg.put(value)
        logging.debug('Producer put value {}...'.format(value))


    def pigeon_consume(pg, value=5):
        logging.debug('Consumer thread started {}...'.format(value))
        pg.present(value, 0.5, value)
        logging.debug('Consumer got value {}...'.format(value))


    def pigeon_absent(pg, value=5):
        logging.debug('Absent consumer thread started {}...'.format(value))
        pg.absent(value)
        logging.debug('Absent consumer got value {}...'.format(value))


    def consumer(cv):
        logging.debug('Consumer thread started ...')
        with cv:
            logging.debug('Consumer waiting ...')
            cv.wait()
            logging.debug('Consumer consumed the resource')


    def producer(cv):
        logging.debug('Producer thread started ...')
        with cv:
            logging.debug('Making resource available')
            logging.debug('Notifying to all consumers')
            cv.notifyAll()


    condition = threading.Condition()
    pg = _TestConditionalRendezvous()
    pg.put(4)
    # cs1 = threading.Thread(name='consumer1', target=consumer, args=(condition,))
    # cs2 = threading.Thread(name='consumer2', target=consumer, args=(condition,))
    # pd = threading.Thread(name='producer', target=producer, args=(condition,))
    pd4 = threading.Thread(name='producer4', target=pigeon_produce, args=(pg, 4))
    pd5 = threading.Thread(name='producer5', target=pigeon_produce, args=(pg, 5))
    pd6 = threading.Thread(name='producer6', target=pigeon_produce, args=(pg, 6))
    cs4 = threading.Thread(name='consumer4', target=pigeon_consume, args=(pg, 4))
    cs5 = threading.Thread(name='consumer5', target=pigeon_consume, args=(pg, 5))
    cs6 = threading.Thread(name='consumer6', target=pigeon_consume, args=(pg, 6))
    ab4 = threading.Thread(name='absent4', target=pigeon_absent, args=(pg, 4))
    ab5 = threading.Thread(name='absent5', target=pigeon_absent, args=(pg, 5))
    ab6 = threading.Thread(name='absent6', target=pigeon_absent, args=(pg, 6))
    # cs6 = threading.Thread(name='consumer2', target=pigeon_absent, args=(pg,6))

    ab4.start()
    cs5.start()
    cs6.start()
    time.sleep(1)

    # cs2.start()
    # time.sleep(2)
    pd5.start()
    time.sleep(0.1)
    print(pg.contains(5))
    ab5.start()
    time.sleep(1)
    pd6.start()
