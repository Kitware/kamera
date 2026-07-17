#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Tuple, List, Dict
from abc import abstractmethod, ABCMeta


class NullDefault(object):
    """A default placeholder for parameters that can take a meaningful None"""

    def __str__(self):
        return "<default>"


Jsonable = Union[dict, list, int, bool, str, tuple, float, NullDefault]

_default = NullDefault()


class Monostack(object):
    """A stack with 0 or 1 objects"""

    def __init__(self):
        self._obj = None

    def push(self, obj):
        """Write some object to serve as the current output of the service"""
        self._obj = obj

    def pop(self):
        obj = self._obj
        self._obj = None
        return obj

    def peek(self):
        return self._obj

    def __bool__(self):
        return self._obj is not None

    def __len__(self):
        if self._obj is None:
            return 0
        return 1


class ChildService(Monostack):
    def __init__(self, kv):
        self.kv = kv
        super(ChildService, self).__init__()

    @abstractmethod
    def set_health(self, status):
        # type: (str) -> bool
        """Set the output health state"""

    def healthy(self):
        self.set_health("passing")

    def unhealthy(self):
        self.set_health("critical")


class KV:
    """Interface for a bare-bones key-value store"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, key, default=_default, **kwargs):
        # type: (str, Jsonable, Jsonable) -> Jsonable
        pass

    @abstractmethod
    def put(self, key, val, **kwargs):
        # type: (str, Jsonable, Jsonable) -> Union[int, bool, str]
        pass

    @abstractmethod
    def delete(self, key, **kwargs):
        # type: (str, Jsonable) -> Union[int, bool, str]
        pass


class BaseEnvoy(KV):
    __metaclass__ = ABCMeta
