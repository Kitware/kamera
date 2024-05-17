#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, List, Dict, Optional, Tuple, Union
import json
from hashlib import md5

from six import string_types, binary_type, PY2, PY3, BytesIO
from benedict import benedict


if PY2:
    JSONDecodeError = ValueError
    from collections import Mapping
else:
    from json import JSONDecodeError
    from collections.abc import Mapping

class MyTimeoutError(OSError):
    pass

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
    # type: (binary_type) -> str
    """Python"""
    if s is None:
        return None
    if isinstance(s, binary_type):
        return s.decode()
    return s


def dumper23(val):
    # type: (Union[str, binary_type, dict]) -> binary_type
    """Python2/3 compatible encoder for redis"""
    if isinstance(val, string_types):
        return val.encode()
    elif isinstance(val, binary_type):
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
        bd = benedict(val).flatten(separator="/")
        return {root + sep + k: v for k, v in bd.items()}
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
    return rpat


def demux_consul(box_values, key=None):
    # type: (List[Dict[str, str]], Optional[str]) -> dict
    """
    Unwraps and de-multiplexes a response from Consul so nested values are returned as nested dict.
    :param box_values:
    :param key:
    :return:

    Examples:
    >>> bv = [{'Key': 'foo', 'Value': b'{"nest": {"num": 42, "spam": "eggs"}}'}]
    >>> demux_consul(bv, 'foo')
    {'nest': {'num': 42, 'spam': 'eggs'}}
    >>> bv = [{'Key': 'nest1/nest2', 'Value': b'abc'},{'Key': 'nest1/nest3', 'Value': b'123'}]
    >>> bv.extend([{'Key': 'nest1/nest3/nest33','Value': b'12345'}, {'Key': 'nest1/nest3/nest45','Value': b'12345'}])
    >>> demux_consul(bv)
    {'nest1': {'nest2': b'abc', 'nest3': {'nest33': 12345, 'nest45': 12345}}}
    >>> demux_consul(bv, 'nest1')
    {'nest2': b'abc', 'nest3': {'nest33': 12345, 'nest45': 12345}}

    """
    if len(box_values) == 1 and "/" not in box_values[0]["Key"]:
        return try_jloads(box_values[0]["Value"])

    bd = benedict({el["Key"]: try_jloads(el["Value"]) for el in box_values})
    bd = bd.unflatten(separator="/")
    if key is not None:
        return bd[key]
    return bd


def redis_decode(keys, values, key=None, flatten=False, as_json=True):
    # type: (List[str], List[str], Optional[str], bool, bool) -> dict
    """
    Unwraps and de-multiplexes a response from Consul so nested values are returned as nested dict.
    :param keys: list of keys
    :param values: list of return values
    :param key: keypath prefix key
    :param as_json: Try to deserialize from json
    :return: nested dict from the keypath

    Examples:
    >>> keys_, vals = ['foo',], [b'{"nest": {"num": 42, "spam": "eggs"}}']
    >>> redis_decode(keys_, vals, 'foo')
    {'nest': {'num': 42, 'spam': 'eggs'}}
    >>> keys_ = ['nest1/nest2', 'nest1/nest3', 'nest1/nest3/nest33', 'nest1/nest3/nest45']
    >>> vals = ['abc', '123', '12345', '12345']
    >>> redis_decode(keys_, vals)
    {'nest1': {'nest2': b'abc', 'nest3': {'nest33': 12345, 'nest45': 12345}}}
    >>> redis_decode(keys_, vals, 'nest1')
    {'nest2': b'abc', 'nest3': {'nest33': 12345, 'nest45': 12345}}
    >>> redis_decode(keys_, vals, 'nest1/nest3')
    {'nest33': 12345, 'nest45': 12345}

    """
    if as_json:
        loader = try_jloads
    else:
        loader = loader23
    ## I think this is wrong
    # if len(keys) == 1:
        # return loader(values[0])

    keys = [loader23(k) for k in keys]
    values = [loader(v) for v in values]

    bd = benedict(zip(keys, values))

    if flatten:
        return bd
    bd = bd.unflatten(separator="/")
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
