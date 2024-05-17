#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
from benedict import benedict
import threading

import rospy

from roskv.impl.redis_envoy import RedisEnvoy


def menu_parser(description="roskv interface"):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-g", "--get", default=None, action="store", type=str, help="get a string/dict")
    parser.add_argument("-H", "--host", default="nuvo0", action="store", type=str, help="redis host")

    parser.add_argument("-D", "--debug", action="store_true", help="Start in debug mode")

    return parser


def do_get(key, host):
    envoy = RedisEnvoy(host, client_name="roskv")
    try:
        val = envoy.get(key)
    except KeyError:
        val = envoy.get_dict(key)
    print(val)


def main():
    parser = menu_parser()
    args = parser.parse_args()
    if args.get:
        do_get(args.get, args.host)


if __name__ == "__main__":
    main()
