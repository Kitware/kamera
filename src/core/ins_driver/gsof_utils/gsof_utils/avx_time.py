#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function
import os
import sys
import json
import errno
import select
import socket
from hashlib import md5

import six
import datetime
from dateutil.tz import gettz
import pytz
from boltons.fileutils import mkdir_p

from gsof_utils.parsers import process_buffer
from gsof_utils.structures import TimeSyncInfo
from gsof_utils.ntp import ntp_query
from gsof_utils.util import LowpassFIR as Lowpass


class AvxTimeClient(object):
    def __init__(self, addr, outfile, ntphost=None, verbosity=0):
        self.ntphost = ntphost
        self.addr = addr
        self.host, port = addr.split(":")
        self.port = int(port)
        self.addrtup = (self.host, self.port)
        self.outfile = outfile
        self.outhash = None
        self.rolling_offset = Lowpass(0.0, 0.1, size=120)
        mkdir_p(os.path.dirname(self.outfile))
        self.outdata = TimeSyncInfo()
        self.verbose = verbosity

    def vprint(self, *args):
        if self.verbose:
            print(*args)

    def update(self, info):
        # type: (TimeSyncInfo) -> None
        """Update the sync info data structure and write it to disk if it has changed
        todo: this
        """
        if self.ntphost:
            ntp_dict = ntp_query(self.ntphost, graceful=True, verbose=False)
            info.ntp_stratum = ntp_dict["stratum"]
            offset = ntp_dict["offset"]
            if offset != 0:
                self.rolling_offset.update(offset)
                rounded_offset = self.rolling_offset.state
            else:
                rounded_offset = offset

            # print(offset, rounded_offset)
            rounded_offset = float("{:.0e}".format(rounded_offset))  # we want to minimize the amount of file updates
            info.ntp_offset = rounded_offset
            info.ntp_valid = ntp_dict["ntp_valid"]
            info.ntp_addr = self.ntphost

        # python2 equality doesn't work on the data class
        info = dict(info)
        if self.outdata != info:
            self.outdata = info.copy()
            utc_now = datetime.datetime.now(gettz()).astimezone(pytz.UTC)
            info['time'] = utc_now.isoformat()
            jstring = json.dumps(info)
            print("{}".format(jstring))
            with open(self.outfile, 'w') as fp:
                fp.write(jstring + '\n')

    def run_continuous(self):
        while True:
            try:
                self.run()
            except KeyboardInterrupt:
                break
            except OSError as exc:
                if exc.errno != errno.ENOSR:
                    print('Is the daemon already running?')
                    raise
            except Exception as exc:
                print("{}: {}".format(exc.__class__.__name__, exc), file=sys.stderr)

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.setblocking(False)

        print("Starting on {}".format(self.addrtup))
        server.bind(self.addrtup)
        # Sockets from which we expect to read

        # Sockets to which we expect to write
        outputs = []
        while True:
            socket_list = [sys.stdin, server]

            # Wait for at least one of the sockets to be ready for processing
            readable, writable, exceptional = select.select(socket_list, outputs, [])
            for sock in readable:

                if sock == server:
                    utc_now = datetime.datetime.now(gettz()).astimezone(pytz.UTC)

                    # A "readable" server socket is ready to accept a connection
                    data, address = sock.recvfrom(4096)
                    # print('new connection from {}'.format(address), file=sys.stderr)
                    if not data:
                        print("disconnected")
                        break
                    else:
                        chunks = process_buffer(data)
                        # sys.stdout.write(repr(data))
                        # sys.stdout.flush()
                        if chunks["pps"]:
                            self.vprint("PPS: {}".format(chunks["pps"]))
                            continue
                        self.vprint("Now: {}".format(utc_now))
                        if chunks["rmc"]:
                            rmc = chunks["rmc"]
                            parser_lag = utc_now - rmc
                            self.vprint("RMC: {} lag: {}".format(rmc, parser_lag))
                        if chunks["parsed_gsofs"]:
                            for m in chunks["parsed_gsofs"]:
                                self.vprint(m)
                        offset = chunks.get("offset", None)
                        if offset:
                            self.vprint("GPS offset: {} ({})".format(offset, chunks["offset_kind"]))
                        info = TimeSyncInfo(gps_offset=offset.total_seconds(), gps_offset_kind=chunks["offset_kind"])
                        self.update(info)
                        # print()


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="split input into chunks")

    parser.add_argument("-S", "--stream", action="store_true", help="run in streaming mode")
    parser.add_argument("-X", "--hex", action="store_true", help="dump hex with space")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose print")
    parser.add_argument("-V", "--verbosity", default=0, action="store", type=int, help="Output verbosity level")
    parser.add_argument(
        "-e", "--end", default="", action="store", type=str, help="End delimiter added between packets after timeout",
    )
    parser.add_argument(
        "-o", "--output", default="/mnt/timestats.json", action="store", type=str, help="File to write summary to",
    )
    parser.add_argument(
        "-", "--timeout", default=0.1, action="store", type=float, help="Timeout to wait between select",
    )
    parser.add_argument("-H", "--host", default=None, action="store", type=str, help="Host to run RPC service on")
    parser.add_argument("-N", "--ntphost", default=None, action="store", type=str, help="NTP server address")
    parser.add_argument(
        "-P", "--port", default=12345, action="store", type=str, help="start of port range to run RPC service on"
    )

    return parser


def main():
    print('stdout avx_time v0.1.0 running on {}'.format(socket.gethostname()))
    # print('stderr avx_time v0.1.0', file=sys.stderr)
    parser = arg_parser()
    args = parser.parse_args()
    host = args.host or "localhost"
    addr = "{}:{}".format(host, args.port)
    verbosity = 99 if args.verbose else args.verbosity
    client = AvxTimeClient(addr, args.output, args.ntphost, verbosity=verbosity)
    client.run_continuous()


if __name__ == "__main__":
    main()
