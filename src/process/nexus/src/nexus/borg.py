import threading
from collections import Mapping


class _Default(object):
    def __str__(self):
        return "<default>"


_default = _Default()


class Borg(Mapping):
    _shared_state = {'data': {}}

    def __init__(self):
        self.__dict__ = self._shared_state
        self._lock = threading.RLock()
        # borg state currently cannot be set from constructor

    def __setitem__(self, name, value):
        with self._lock:
            # print(f'<>{type(self).__name__}["{name}"]"')
            self.data.__setitem__(name, value)

    def __getitem__(self, name):
        with self._lock:
            return self.data.__getitem__(name)

    def get(self, key, default=_default):
        with self._lock:
            if default is _default:
                v = self.data.get(key)
                return v

            v = self.data.get(key, default)
            return v

    def __iter__(self):
        with self._lock:
            return self.data.__iter__()

    def __len__(self):
        with self._lock:
            return self.data.__len__()


def test_borg():
    """
    Example:
    >>> b1 = Borg()
    >>> b2 = Borg()
    >>> b1.data is b2.data
    True
    >>> b1['foo'] = 'bar'
    >>> dict(b2)
    >>> {'foo': 'bar'}

    :return:
    """