#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Tuple, Union
from io import BytesIO
import json
from hashlib import md5
from collections.abc import Mapping
from json import JSONDecodeError


class MyTimeoutError(OSError):
    pass


def _flatten(d, prefix="", sep="/"):
    """Recursively flatten a nested dict with sep-joined keys."""
    items = {}
    for k, v in d.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, Mapping):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def _unflatten(flat, sep="/"):
    """Reconstruct a nested dict from a flat dict with sep-joined keys."""
    result = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = result
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def hash_genpy_msg(msg):
    # type: (genpy.message.Message) -> bytes
    buf = BytesIO()
    msg.serialize(buf)
    return md5(buf.getvalue()).hexdigest()


def simple_hash_jsonable(obj):
    """Works best on simple objects"""
    s = json.dumps(obj, sort_keys=True)
    m = md5(s.encode())
    return m.hexdigest()


def simple_hash_args(*args):
    """Works best on simple objects"""
    s = json.dumps(args, sort_keys=True)
    m = md5(s.encode())
    return m.hexdigest()


def loader23(s):
    # type: (bytes) -> str
    if s is None:
        return None
    if isinstance(s, bytes):
        return s.decode()
    return s


def dumper23(val):
    # type: (Union[str, bytes, dict]) -> bytes
    """Encode a value for storage in Redis"""
    if isinstance(val, str):
        return val.encode()
    elif isinstance(val, bytes):
        return val
    return json.dumps(val).encode()


def try_jloads(s):
    # type: (str) -> Any
    """Silently try to coerce from json"""
    if s is None:
        return None
    try:
        val = json.loads(s)
    except JSONDecodeError:
        val = loader23(s)
    return val


def flatten_fs(val, root="", sep="/"):
    """Flatten with forward slash separator.
    ROS-style KV space nesting with a leading forward slash for root"""
    if isinstance(val, Mapping):
        flat = _flatten(val, sep=sep)
        return {root + sep + k: v for k, v in flat.items()}
    return val


def redis_encode(key, val):
    # type: (str, Any) -> List[Tuple[str, str]]
    """Build a list of pairs suitable for pushing to redis"""
    if isinstance(val, Mapping):
        ddval = flatten_fs(val, key)
    else:
        ddval = {key: val}

    return [(dumper23(k), dumper23(v)) for k, v in ddval.items()]


def wildcard(pat, s):
    """Match like redis wildcards"""
    import re
    s = loader23(s)
    rpat = loader23(pat).replace('*', '.*').replace('?', '.')
    mat = re.search(rpat, s)
    return mat is not None


def filter_hosts_by_system(all_hosts, system_name=None):
    """Return host names belonging to the current system.

    Redis (/sys) can accumulate host entries from other systems; host
    names are suffixed with the system name, e.g. "center0taiga".
    """
    import os
    if system_name is None:
        system_name = os.environ.get("SYSTEM_NAME")
    all_hosts = sorted(all_hosts)
    if system_name:
        hosts = [h for h in all_hosts if h.endswith(system_name)]
    else:
        hosts = []
    if not hosts:
        # Fall back to every host if the naming convention doesn't match.
        hosts = all_hosts
    return hosts


def redis_decode(keys, values, key=None, flatten=False, as_json=True):
    # type: (List[str], List[str], Optional[str], bool, bool) -> dict
    """
    Unwraps and de-multiplexes a Redis response so nested values are returned as a nested dict.
    :param keys: list of keys
    :param values: list of return values
    :param key: keypath prefix key
    :param flatten: return the flat dict without unflattening
    :param as_json: Try to deserialize from json
    :return: nested dict from the keypath

    Examples:
    >>> keys_, vals = ['foo',], [b'{"nest": {"num": 42, "spam": "eggs"}}']
    >>> redis_decode(keys_, vals, 'foo')
    {'nest': {'num': 42, 'spam': 'eggs'}}
    >>> keys_ = ['nest1/nest2', 'nest1/nest3', 'nest1/nest3/nest33', 'nest1/nest3/nest45']
    >>> vals = ['abc', '123', '12345', '12345']
    >>> redis_decode(keys_, vals, 'nest1')
    {'nest2': 'abc', 'nest3': {'nest33': 12345, 'nest45': 12345}}
    >>> redis_decode(keys_, vals, 'nest1/nest3')
    {'nest33': 12345, 'nest45': 12345}

    """
    loader = try_jloads if as_json else loader23
    keys = [loader23(k) for k in keys]
    values = [loader(v) for v in values]

    flat = dict(zip(keys, values))
    if flatten:
        return flat
    bd = _unflatten(flat)
    if len(bd) == 1 and "" in bd:
        bd = bd[""]
    if key is None:
        return bd
    for subkey in key.split("/"):
        if not subkey:
            continue
        bd = bd[subkey]
    return bd


if __name__ == "__main__":
    dd = {"name": "bar_from_bd", "nest": {"spam": "eggs", "num": 42, "deep": {"a": 0}}, "a_list": [1, 2, "three"]}
    tmp = redis_encode("", dd)
    tmp2 = dict(tmp)
    out = redis_decode(tmp2.keys(), tmp2.values())
    assert dd == out
    assert sorted(tmp) == [
        (b"/a_list", b'[1, 2, "three"]'),
        (b"/name", b"bar_from_bd"),
        (b"/nest/deep/a", b"0"),
        (b"/nest/num", b"42"),
        (b"/nest/spam", b"eggs"),
    ]
