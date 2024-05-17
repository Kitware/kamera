import os
import redis


class RedisConfig(object):
    __slots__ = ["client_name", "host", "port", "unix_socket_path"]

    def __init__(
        self, client_name=None, host=None, port=None, unix_socket_path=None
    ):
        self.client_name = client_name
        self.host = host or os.environ.get('REDIS_HOST', "localhost")
        self.port = port or os.environ.get('REDIS_PORT', 6379)
        self.unix_socket_path = unix_socket_path or os.environ.get('REDIS_SOCK', None)

    def dict(self):
        return dict([(k, getattr(self, k, None)) for k in self.__slots__])
