#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import json
from select import select
from enum import Enum
from collections import deque

from six import StringIO
import yaml


class ReaderState(Enum):
    unknown = 0
    reading_delim = 1
    reading_body = 2
    complete = 3


class YamlBuf(object):
    def __init__(self):
        self._state = ReaderState.unknown
        self._buf = StringIO()
        self._msgs = deque()

    def finalize_buf(self):
        try:
            msgs = yaml.load_all(self._buf.getvalue(), yaml.Loader)
            for m in msgs:
                self._msgs.append(m)
            self._buf = StringIO()
            self._state = ReaderState.reading_delim
        except yaml.YAMLError as exc:
            print(exc)

    def write(self, s):
        if not s:
            return
        if self._state == ReaderState.unknown:
            self._buf.write(s)
            if s == "---\n":
                self._state = ReaderState.reading_body
            else:
                pass
        elif self._state == ReaderState.reading_delim:
            if s == "---\n":
                self._buf.write(s)
                self._state = ReaderState.reading_body
            else:
                print("Unexpected line while expecting delim (---) : {}".format(s), file=sys.stderr)
        elif self._state == ReaderState.reading_body:
            if s == "---\n":
                self.finalize_buf()
                self._buf.write(s)
                self._state = ReaderState.reading_body
            else:
                self._buf.write(s)
        elif self._state == ReaderState.complete:
            raise NotImplementedError("not used right now")
        else:
            raise ValueError("Invalid state: {}".format(self._state))

    @property
    def len(self):
        return len(self._msgs)

    def pop(self):
        try:
            return self._msgs.popleft()
        except IndexError:
            return None


def stdin_gen(timeout=0.1):
    yamlbuf = YamlBuf()
    while True:
        packet = StringIO()
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            s = sys.stdin.readline()
            yamlbuf.write(s)
            if yamlbuf.len:
                sys.stdout.write(json.dumps(yamlbuf.pop()) + "\n")

        else:
            if yamlbuf.len:
                sys.stdout.write(json.dumps(yamlbuf.pop()) + "\n")
        sys.stdout.flush()


def main():
    try:
        stdin_gen()
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
