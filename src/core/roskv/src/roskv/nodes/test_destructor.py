#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
from roskv.impl.redis_envoy import RedisEnvoy as Envoy


envoy = Envoy(host=sys.argv[1])


def main():
    print(envoy.name)


if __name__ == "__main__":
    main()
