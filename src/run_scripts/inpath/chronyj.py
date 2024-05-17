#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import json
import subprocess
from collections import OrderedDict


class SubprocessError(Exception):
    pass


chronyc_keys = [
    "ref_id",
    "ip",
    "stratum",
    "ref_time",
    "sys_time_vs_ntp",
    "last_offset",
    "rms_offset",
    "frequency_ppm",
    "residual_frequency_ppm",
    "skew_ppm",
    "root_delay",
    "root_dispersion",
    "update_interval",
    "leap_status",
]

essential_keys = ["ip", "rms_offset"]


def menu_parser():
    import argparse

    parser = argparse.ArgumentParser(description="wrapper around chronyc tracking")
    parser.add_argument("-u", "--user", default="user", action="store", help="ssh user")
    parser.add_argument("-a", "--all", action="store_true", help="run on all hosts")
    parser.add_argument("-H", "--host", default=None, action="store", type=str, help="host to target")
    parser.add_argument(
        "--host-list", default="guibox;nuvo0;nuvo1;nuvo2", action="store", type=str, help="list of hosts, split by ';'"
    )
    parser.add_argument("-t", "--thresh", default=1e-3, action="store", type=float, help="RMS offset max threshold")
    parser.add_argument("-k", "--check", action="store_true", help="Verify that settings look ok")
    parser.add_argument("-D", "--debug", action="store_true", help="Start in debug mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser


def run_subproc(cmdlist, timeout=3):
    ps = subprocess.Popen(cmdlist, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = ps.communicate()
    if ps.returncode:
        msg = "{cmdstr} exited with status {rc}: \n{err}".format(cmdstr=" ".join(cmdlist), rc=ps.returncode, err=err)
        print(msg, file=sys.stderr)
        raise SubprocessError(msg)
    return out


def chronyc_tracking(host=None, user=None, verbose=False):
    if host is None:
        cmd = []
    else:
        cmd = ["ssh", "{}@{}".format(user, host)]
    cmd.extend(["chronyc", "-c", "tracking"])
    if verbose:
        print(" ".join(cmd), file=sys.stderr)

    out = run_subproc(cmd)
    parts = out.strip().split(",")
    parts[2] = int(parts[2])
    for i in range(3, 13):
        parts[i] = float(parts[i])
    return filter_chronyc(zip(chronyc_keys, parts), verbose=verbose)


def filter_chronyc(data_list, verbose=False):
    if verbose:
        return OrderedDict(data_list)
    return OrderedDict([(el[0], el[1]) for el in data_list if el[0] in essential_keys])


def chronyc_multi(hosts, user="user", verbose=False):
    data = {}
    for host in hosts:
        data[host] = chronyc_tracking(host, user=user, verbose=verbose)
    return data


def check_offsets(data, thresh=1e-3):
    fails = []
    for host, rec in data.items():
        offset = rec['rms_offset']
        if offset > thresh:
            fails.append((host, offset))
    return fails


def main():
    args = menu_parser().parse_args()
    if args.all or args.check:
        hosts = args.host_list.split(";")
        data = chronyc_multi(hosts, user=args.user, verbose=args.verbose)
    else:
        data = {"localhost": chronyc_tracking(args.host, user=args.user, verbose=args.verbose)}

    print(json.dumps(data, indent=2))
    if args.check:
        fails = check_offsets(data, thresh=args.thresh)
        if fails:
            pairs = '\n'.join('{: >8}: {}'.format(*pair) for pair in fails)
            msg = 'One or more hosts are desynchronized, RMS offset is too high:\n{}'.format(pairs)
            raise ValueError(msg)


if __name__ == "__main__":
    main()
