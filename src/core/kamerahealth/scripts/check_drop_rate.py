#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import re
import sys
import datetime
from glob import glob
import errno
import math
import argparse
from pprint import pprint
from fractions import Fraction
from collections import Counter, defaultdict
import subprocess


class ManditoryInitializer(object):
    def __init__(self, **kwargs):
        sslots = set(self.__slots__)
        diffs = sslots.symmetric_difference(kwargs)
        if diffs:
            raise ValueError("missing fields: {}".format([x for x in diffs]))

        for k, v in kwargs.items():
            setattr(self, k, v)
            
class CountRecord(ManditoryInitializer):
    __slots__ = ["fov", "ext", "count", "secs" ]


def first(iterable, default=None, key=None):
    return next(filter(key, iterable), default)


def filter_view_dirs(dirs):
    return sorted(x for x in dirs if "_view" in x)

def vprint(*args):
    vprint_level = os.environ.get('VPRINTV', 0)
    if vprint_level:
        print(*args, file=sys.stderr)


def b3sum(fn):
    cmd = ["b3sum", fn]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    out, err = proc.communicate()

    if err:
        raise RuntimeError("b3sum failed: {}".format(err))

    cs, _name = out.split("  ")
    return cs


def parse_file_time(fn):
    # test0128_fl500_C_20210128_202624.001608_ir.tif
    ts = re.search(r"\d{8}_\d{6}\.\d{6}", fn).group()
    t = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S.%f")
    return t


def count_unique(files):
    checksums = Counter(b3sum(fn) for fn in files)
    return len(checksums)


def list_dirs(input_dir):
    return next(os.walk(input_dir))[1]

### === === === === === === === === ===



def get_fovs(input_dir):
    dirs = list_dirs(input_dir)
    fov_dirs = filter_view_dirs(dirs)
    if fov_dirs:
        return fov_dirs
    # we are at config level
    raise NotImplementedError("unable to process flight level just yet")
    

def count_ext(input_dir, dirname, ext, debug=False):
    gb = os.path.join(input_dir, dirname, "*" + ext)
    if debug:
        print(gb)
    files = sorted(glob(gb))
    if not files:
        print("No files for {}".format(gb))
        return {}
    ts_lo = parse_file_time(files[0])
    ts_hi = parse_file_time(files[-1])
    dt = ts_hi - ts_lo
    count = len(files)
    sec = dt.total_seconds()
    vprint("Start: {} End: {} DT: {}".format(ts_lo, ts_hi, dt))
    return count, sec

def count_per_fov(input_dir, fov_dirs, exts, debug=False):
    data = []
    for fov_dir in fov_dirs:
        for ext in exts:
            count, secs = count_ext(input_dir, fov_dir, ext, debug=debug)

            data.append(CountRecord(fov=fov_dir, ext=ext, count=count, secs=secs))

    return data

## deprecated
def process_ext(input_dir, dirname, ext, rate=None, actual_expect=None, thorough=False, debug=False):
    count, secs = count_ext
    data = compute_stats(count, secs, rate, actual_expect=actual_expect, debug=debug)
    return data

def compute_stats(count, sec, actual_expect, rate=None, thorough=False, debug=False):

    raw_rate = count / sec
    nearest_rate = rate or float(Fraction(raw_rate).limit_denominator(120))

    expected = actual_expect
    corrected_count = count
    drop_count = expected - corrected_count
    drop_rate = drop_count / expected
    stats = "{}/{}".format(count, expected)
    # if thorough:
    #     n_checksums = count_unique(files)
    # else:
    n_checksums = "-1"
    dd = {
        "count": count,
        "sec": sec,
        "rate": nearest_rate,
        "stats": stats,
        "drop_count": drop_count,
        "drop_rate": drop_rate,
        "expect": expected,
        "corrected_count": corrected_count,
        "n_unique": n_checksums,
    }
    return dd


def get_evt_counts(input_dir, mode='evt'):
    evts = {}
    for dirname in filter_view_dirs(list_dirs(input_dir)):
        gb = glob(os.path.join(input_dir, dirname, "*" + mode + ".json"))
        assert isinstance(gb, list)
        evts[dirname] = len(gb)

    assert evts, "Could not find evts in directory {}".format(input_dir)
    actual_expect = max(evts.values())
    if not all([x == actual_expect for x in evts.values()]):
        print("Warning: unbalanced count of {}.json".format(mode))
    print("{}.json count: {}".format(mode, evts))

    return actual_expect

# computing the expected number is a little tricky
def calc_expected(counts_data, exts):
    mode = 'evt.json' if 'evt.json' in exts else 'meta.json'
    select_rows = [rec for rec in counts_data if rec.ext == mode]
    evts = {rec.fov: rec.count for rec in select_rows}
    actual_expect = max(evts.values())
    if not all([x == actual_expect for x in evts.values()]):
        print("Warning: unbalanced count of {}".format(mode))
    print("{} count: {}".format(mode, evts))

    return actual_expect


def count_types(input_dir, exts="meta.json;ir.tif;uv.jpg;rgb.jpg", rate=None, thorough=False, mode='evt', debug=False):
    dirs = list_dirs(input_dir)
    exts = exts.split(";")
    results = defaultdict(dict)

    fov_dirs = get_fovs(input_dir)
    counts_data = count_per_fov(input_dir, fov_dirs, exts, debug=debug)
    actual_expect = calc_expected(counts_data, exts)
    # stats = [compute_stats(d[""])]
    stats = [(rec, compute_stats(rec.count, rec.secs, actual_expect)) for rec in counts_data]
    for rec, stat in stats:
        results[rec.fov][rec.ext] = stat

    return results



    ## === 


def print_format(results):
    lines = []
    lines.append(
        f"{'dirname': <12} {'ext': <9} {'count': >6} {'expect': >7} {'freq': >5} {'n_drop': >7} {'drop_r': >7} {'pass': >5} {'n_unique': >7}"
    )
    for dirname in results:
        for ext in results[dirname]:
            d = results[dirname][ext]
            if not d:
                continue
            passfail = d["drop_rate"] < 0.01
            passfail = 'pass' if passfail else 'FAIL'
            n_unique = d['n_unique']
            n_unique = '' if int(n_unique) < 0 else n_unique
            d.update({"ext": ext, "dirname": dirname, "pass": passfail, 'n_unique': n_unique})
            s = f"{dirname: <12} {ext: <9} {d['count']: >6d} {d['expect']: >7d} {d['rate']: >5.2f} {d['drop_count']: >7d} {d['drop_rate']: >9.5f} {passfail: >5} {d['n_unique']: >7}"
            lines.append(s)

    for line in lines:
        print(line)


def do_b3sum(args):
    print(b3sum(args.input_uri))


def do_count(args):
    mode = 'meta' if args.meta else 'evt'
    res = count_types(args.input_uri, args.exts, args.rate, args.thorough, mode=mode, debug=args.debug)
    # pprint(res)
    print_format(res)


def main():
    parser = menu_parser()
    args = parser.parse_args()
    if args.b3sum:
        return do_b3sum(args)
    else:
        return do_count(args)


def menu_parser(description="check drop rate"):

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input_uri", default=None, action="store", type=str, help="an input directory path")
    parser.add_argument("-r", "--rate", default=None, action="store", type=float, help="capture rate")
    parser.add_argument("--thorough", default=False, action="store_true", help="thorough scan")
    parser.add_argument("-b", "--b3sum", default=False, action="store_true", help="do b3sum")
    parser.add_argument("-m", "--meta", default=False, action="store_true", help="use meta as counts")
    parser.add_argument(
        "-e",
        "--exts",
        default="meta.json;ir.tif;uv.jpg;rgb.jpg",
        action="store",
        type=str,
        help="semicolon-separated list of extensions",
    )
    parser.add_argument("-D", "--debug", action="store_true", help="Start in debug mode")

    return parser


if __name__ == "__main__":
    main()
