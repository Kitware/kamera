#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from typing import Any, Union, Dict
import json
import time

import redis
from roskv.base import BaseEnvoy, KV, NullDefault, Jsonable, ChildService
from roskv.util import redis_decode, redis_encode, loader23

_default = NullDefault()


class RedisEnvoy(redis.client.Redis, BaseEnvoy):
    """Redis-based key-value store. Extends redis-py directly with ROS-style
    nested-key helpers and an auto-generated client name."""

    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
        password=None,
        socket_timeout=None,
        socket_connect_timeout=None,
        socket_keepalive=None,
        socket_keepalive_options=None,
        connection_pool=None,
        unix_socket_path=None,
        encoding="utf-8",
        encoding_errors="strict",
        decode_responses=False,
        retry_on_timeout=False,
        ssl=False,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_cert_reqs="required",
        ssl_ca_certs=None,
        ssl_check_hostname=False,
        max_connections=None,
        single_connection_client=False,
        health_check_interval=0,
        client_name=None,
        username=None,
    ):
        if client_name is None:
            from uuid import uuid4
            import socket
            client_name = socket.gethostname() + "_" + str(uuid4())[:8]

        self.client_name = client_name
        super(RedisEnvoy, self).__init__(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            connection_pool=connection_pool,
            unix_socket_path=unix_socket_path,
            encoding=encoding,
            encoding_errors=encoding_errors,
            decode_responses=decode_responses,
            retry_on_timeout=retry_on_timeout,
            ssl=ssl,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_check_hostname=ssl_check_hostname,
            max_connections=max_connections,
            single_connection_client=single_connection_client,
            health_check_interval=health_check_interval,
            client_name=client_name,
            username=username,
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get(self, key, default=_default, **kwargs):
        # type: (str, Jsonable, Any) -> Jsonable
        """Get a value by exact key, or a nested dict by keypath prefix.
        Automatically tries to deserialize from json."""
        if self.exists(key):
            return loader23(super(RedisEnvoy, self).get(key))
        return self.get_dict(key, default=default, **kwargs)

    def get_dict(self, key, default=_default, flatten=False, **kwargs):
        # type: (str, Jsonable, bool, Any) -> Jsonable
        """Get a nested dict by keypath prefix.
        Automatically tries to deserialize from json."""
        keys = self.keys(key + "*")
        if not keys:
            if default is _default:
                raise KeyError(key)
            return default
        p = self.pipeline()
        for k in keys:
            p.get(k)
        vals = p.execute()
        return redis_decode(keys, vals, key, flatten, kwargs.get("as_json", True))

    def put(self, key, val, **kwargs):
        # type: (str, Union[Any, Dict], Any) -> bool
        """Insert a value. Dicts are automatically flattened into the keypath."""
        tups = redis_encode(key, val)
        if len(tups) == 0:
            return self.set(key, val, **kwargs)
        p = self.pipeline()
        for pair in tups:
            p.set(pair[0], pair[1])
        return p.execute()

    def delete(self, key, **kwargs):
        return super(RedisEnvoy, self).delete(key, **kwargs)

    def delete_dict(self, key, **kwargs):
        keys = self.keys(key + "*")
        if not keys:
            raise KeyError(key)
        p = self.pipeline()
        for k in keys:
            p.delete(k)
        return p.execute()

    @property
    def name(self):
        return self.client_name


class StateService(ChildService):
    """Posts a bit of state along with a health status to Redis."""

    def __init__(self, kv, node, name):
        #  type: (KV, str, str) -> None
        self.node = node
        self.name = name
        super(StateService, self).__init__(kv=kv)

    def update_state(self, state, status="passing"):
        self.push(state)
        self.set_health(status=status)

    def set_health(self, status):
        key = "health/checks/{}/{}".format(self.node, self.name)
        val = {"Status": status, "time": time.time(), "state": self.peek()}
        return self.kv.put(key, val)


if __name__ == "__main__":
    import sys
    try:
        addr = sys.argv[1]
        host, port = addr.split(":")
    except IndexError as exc:
        print("Warning: {}".format(exc), file=sys.stderr)
        host = "localhost"
        port = "6379"
    print("commence janky test on {}:{}".format(host, port), file=sys.stderr)
    dd = {
        "name": "bar_from_bd",
        "nest": {"spam": "eggs", "num": 42, "deep": {"a": 0}},
        "a_list": [1, 2, "three"],
    }
    kv = RedisEnvoy(host=host, port=port, client_name="foo")
    print("keys at start: {}".format(len(kv.keys())))
    key = "/TEST_DO_NOT_USE"
    kv.put(key, dd)
    out = kv.get(key)
    print("{}".format(kv.keys(key + "*")))
    print(out)
    print(out == dd)
    kv.delete_dict(key)
    print("keys at end  : {}".format(len(kv.keys())))
