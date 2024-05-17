from roskv.base import KV, NullDefault
from roskv.impl.rosparam_kv import RosParamKV
from roskv.impl.consul_kv import ConsulKV


_default = NullDefault()


class ConsulRosMirrorKV(KV):
    """Uses rosparam as backing store but mirrors to consul. Consul is 100% a follower.
    This is mostly for PoC
    """
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
        self.ckv = ConsulKV(
            host=host, port=port, token=token, scheme=scheme, consistency=consistency, dc=dc, verify=verify, cert=cert
        )
        self.rkv = RosParamKV()

    def get(self, key, default=_default, **kwargs):
        return self.rkv.get(key, _default, **kwargs)

    def put(self, key, val, **kwargs):
        self.ckv.put(key, val, **kwargs)
        return self.rkv.put(key, val, **kwargs)

    def delete(self, key, **kwargs):
        self.ckv.delete(key, **kwargs)
        return self.rkv.delete(key, **kwargs)
