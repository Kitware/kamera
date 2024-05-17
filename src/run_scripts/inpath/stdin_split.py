#!/usr/bin/env python3

from __future__ import print_function, division
from typing import Callable
import os
import sys
import codecs

import select
from enum import Enum

from six import BytesIO


def chunk2(s):
    result = []
    for i in range(0, len(s), 2):
        result.append(s[i : i + 2])
    return result


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="split input into chunks")

    parser.add_argument("-S", "--stream", action="store_true", help="run in streaming mode")
    parser.add_argument("-X", "--hex", action="store_true", help="dump hex with space")

    parser.add_argument(
        "-e", "--end", default="", action="store", type=str, help="End delimiter added between packets after timeout",
    )
    parser.add_argument(
        "-", "--timeout", default=0.1, action="store", type=float, help="Timeout to wait between select",
    )

    return parser


def full_typename(o):
    if isinstance(o, type):
        module = o.__module__
        name = o.__name__
    elif isinstance(o, Callable):
        module = o.__module__
        name = o.__name__

    else:
        module = o.__class__.__module__
        name = o.__class__.__name__

    if module is None or module == str.__class__.__module__:
        return name  # Avoid reporting __builtin__

    return module + "." + name


class ReaderState(Enum):
    unknown = 0
    reading_delim = 1
    reading_body = 2
    complete = 3


class Dumper(object):
    def __init__(self, end=b""):
        self.end = end

    def dump(self, bb):
        sys.stdout.buffer.write(bb)
        sys.stdout.buffer.write(self.end)
        sys.stdout.flush()


class HexDumper(Dumper):
    def dump(self, bb):
        hexl = codecs.encode(bb, 'hex_codec')
        parts = chunk2(hexl)
        sys.stdout.buffer.write(b' '.join(parts))
        sys.stdout.buffer.write(self.end)
        sys.stdout.flush()


def stdin_gen(timeout=0.1, end=b""):
    """Read in from stdin and insert an end sequence if `timeout` has elapsed"""
    buf = BytesIO()
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            for r in rlist:
                # gross but idk how to do it better at the moment
                pk = r.buffer.peek()
                _ = sys.stdin.buffer.read(len(pk))
                buf.write(pk)

        else:
            if len(buf.getvalue()):
                yield buf.getvalue() + end
                buf = BytesIO()


def main(timeout=0.1, end=b"---\r\n", hex_=False):
    if hex_:
        dumper = HexDumper(end=end)
    else:
        dumper = Dumper(end=end)
    try:
        for bb in stdin_gen(timeout=timeout):
            dumper.dump(bb)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    end = args.end.encode().decode('unicode_escape').encode()
    main(args.timeout, end, args.hex)
