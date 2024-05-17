import json
from typing import Any
from collections import Mapping

from benedict import benedict
from six import string_types, binary_type
import consul
from roskv.base import KV, NullDefault, Jsonable

from src.core.roskv.src.roskv.util import demux_consul

_default = NullDefault()


class ConsulKV(KV):
    def __init__(
        self,
        host="127.0.0.1",
        port=8500,
        token=None,
        scheme="http",
        consistency="default",
        dc=None,
        verify=True,
        cert=None,
    ):
        self.c = consul.Consul(
            host=host, port=port, token=token, scheme=scheme, consistency=consistency, dc=dc, verify=verify, cert=cert
        )

    def get(self, key, default=_default, **kwargs):
        """
        Consul will return a None if empty, but ROS throws a KeyError here. We shall do the same.


        consul.kv.get Returns a tuple of (*index*, *value[s]*)

            *index* is the current Consul index, suitable for making subsequent
            calls to wait for changes since this query was last run.
            
        value is a dict with some additional Consul metadata that this interface doesn't really care about: 
         {u'CreateIndex': 58,
          u'Flags': 0,
          u'Key': u'foo',
          u'LockIndex': 0,
          u'ModifyIndex': 58,
          u'Value': 'spam'}

        :param key:
        :param default:
        :param kwargs:
        :return:
        """
        return self._get_mux(key, default, **kwargs)

    def _get_mux(self, key, default=_default, **kwargs):
        # type: (str, Any, Any) -> Jsonable
        key = key.strip('/')
        index, box_values = self.c.kv.get(key=key, recurse=True, **kwargs)
        if box_values is None:
            if _default is _default:
                raise KeyError(key)
            else:
                return _default

        return demux_consul(box_values, key)

    def put(self, key, val, **kwargs):
        """
        Insert value into KV.
        Todo: ROS param allows passing nests. These can be unrolled using benedict. For now, just dump to json as PoC

        Consul's put interface returns True on success. Not sure best way to handle it so just returning as is.
        :param key:
        :param val:
        :param kwargs:
        :return:
        """
        key = key.strip('/')
        if isinstance(val, Mapping):
            bd = benedict(val).flatten(separator='/')
            for k, v in bd.items():
                self._put(key + '/' + k, v, **kwargs)
            return True

        else:
            return self._put(key, val, **kwargs)

    def _put(self, key, val, **kwargs):
        if not isinstance(val, (string_types, binary_type)):
            val = json.dumps(val)

        return self.c.kv.put(key, val, **kwargs)

    def delete(self, key, **kwargs):
        key = key.strip('/')
        return self.c.kv.delete(key, **kwargs)
