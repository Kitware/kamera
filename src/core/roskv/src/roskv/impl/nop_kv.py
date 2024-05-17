from roskv.base import KV, NullDefault

_default = NullDefault()


class NopKV(KV):
    """Basic data store for mocking"""
    def __init__(self):
        self.data = {}

    def get(self, key, default=_default, **kwargs):
        if _default is _default:
            return self.data.get(key)
        return self.data.get(key, default)

    def put(self, key, val, **kwargs):
        self.data[key] = val

    def delete(self, key, **kwargs):
        del self.data[key]
