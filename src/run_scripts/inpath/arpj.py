#!/usr/bin/env python
import sys
import json
import subprocess
from six import StringIO

import csv


def read_csv_records(s, columns):
    out = []
    reader = csv.reader(StringIO(s.replace("<incomplete>", "null null")), delimiter=" ")
    for row in reader:
        dd = dict(zip(columns, [x.strip("()").strip("[]") for x in row]))
        out.append(dd)

    return out


def keep_fields(list_of_dicts, columns):
    return [
        dict(filter(lambda p: p[0] in columns, rec.items())) for rec in list_of_dicts
    ]


def recmap(func, list_of_dicts):
    return [dict(map(func, rec.items())) for rec in list_of_dicts]


def run_cmd(cmd):
    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = ps.communicate()
    ps.wait()
    return out.decode(), err.decode()


def run_arp():
    cmd = ["arp", "-an"]
    out, err = run_cmd(cmd)
    if err:
        raise RuntimeError("`{}` failed with error: {}".format(" ".join(cmd), err))

    # seems linux specific, it's different on mac
    columns = ["?", "ip", "at", "mac", "ether", "on", "iface"]
    dd = read_csv_records(out, columns)
    return keep_fields(dd, ["ip", "mac", "iface"])


def run_arp_scan(iface):
    cmd = ["sudo", "arp-scan", "-l", "-I", iface]
    out, err = run_cmd(cmd)
    if err:
        raise RuntimeError("`{}` failed with error: {}".format(" ".join(cmd), err))

    lines = out.split('\n')
    columns = ['ip', 'mac', 'name']
    lines = [dict(zip(columns, el.split('\t'))) for el in lines[2:-4]]
    return lines


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dd = run_arp_scan(sys.argv[1])
    else:
        dd = run_arp()
    print(json.dumps(dd))
