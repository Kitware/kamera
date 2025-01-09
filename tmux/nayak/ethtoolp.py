#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import sys
import subprocess
import re
import json
import yaml
import argparse
import time
from pprint import pprint

assert sys.version_info.major >= 3, "must be python >=3"


PAT_SPACED = re.compile(r'(?=\w)(\w+)(?<=\w)')


def vprint(*args):
    vprint_level = os.environ.get('VPRINTV', 0)
    if vprint_level:
        print(*args, file=sys.stderr)


def get_space_separated_line(line):
    return re.findall(PAT_SPACED, line)


def menu(description='Wrapper around ethtool used to set interface speed to max'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config_uri", default=None, action="store", type=str, help="input config file")
    parser.add_argument('iface', nargs='?', default=None, type=str, help="Interface to be set")

    parser.add_argument("-s", "--speed", default=None, action="store", type=int, help="speed to set")
    parser.add_argument("-H", "--host", default=None, action="store", type=str, help="host name")
    parser.add_argument("-p", "--port", default=8987, action="store", type=int, help="port")
    parser.add_argument("-M", "--maximize-speed", "--max-speed", action="store_true", help="Set interface to use max possible speed")
    parser.add_argument("-+", "--health", action="store_true", help="Run a health check")
    parser.add_argument("-D", "--debug", action="store_true", help="Start in debug mode")
    parser.add_argument("-F", "--force", action="store_true", help="Force option even if it's already set")
    parser.add_argument("-a", "--all", action="store_true", help="Show all attributes")

    return parser


def run_cmd(cmd):
    # type: (list) -> (str, str)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    data = proc.stdout.read().decode()
    err = proc.stderr.read().decode()

    return data, err

def errcho(s):
    # type: (str) -> None
    sys.stderr.write('\033[31m' + s + '\n\033[0m')
    sys.stderr.flush()

def esplit(s, sep=" "):
    # type: (str, str) -> str
    try:
        return s.split(sep)

    except AttributeError:
        return s


def kmap(func, d):
    # type: (Callable, dict) -> dict
    return {k: func(v) for k, v in d.items()}

def parse_nmcli_line(line):
    # type: (str) -> dict
    parts = get_space_separated_line(line)
    device = parts[0]
    dtype = parts[1]
    state = parts[2]
    connection = ' '.join(parts[3:])
    return dict(device=device, dtype=dtype, state=state, connection=connection)

def nmcli_device_status():
    data, err = run_cmd(["nmcli", "device", "status"])
    if err:
        errcho('Subprocess `ethtool` failed with following error: \n{}'.format(err))
        sys.exit(1)

    recs = [parse_nmcli_line(line) for line in data.split('\n') if line][1:]
    return recs

def get_wired_devices():
    recs = nmcli_device_status()
    return [r for r in recs if 'wired' in r["connection"].lower()]


def ethtool_read(iface, show_all=False):
    cmd = ['sudo', 'ethtool', iface]

    data, err = run_cmd(cmd)
    if err:
        errcho('Subprocess `ethtool` failed with following error: \n{}'.format(err))
        sys.exit(1)
    data2 = data.replace(" \n\t" + " " * 24, "\n\t\t").replace("\t", " " * 4)
    dd = yaml.load(data2, yaml.Loader)
    if not dd:
        raise ValueError('failed to parse yaml:\n{}'.format(data))
    dd2 = {k: kmap(esplit, v) for k, v in dd.items()}
    dd3 = {k.replace('Settings for ', ''): v for k, v in dd2.items()}
    vprint(dd3)
    out = dd3[iface]
    if show_all:
        return out

    out = {k: v for k,v in out.items() if k in ["Speed", "Link detected", "Duplex", "Supported link modes"]}
    return out



def parse_speed(link_string, default=0):
    # type: (str, int) -> int
    res = re.findall('(\d+)', link_string)
    try:
        return int(res[0])
    except (IndexError, ValueError):
        return default

def get_current_speed(iface):
    iface_data = ethtool_read(iface)
    vprint(iface_data)
    current_speed = parse_speed(iface_data["Speed"][0])
    return current_speed


def set_speed_to_max(iface, maxspeed):
    # type: (str, int) -> str
    stdout, stderr = run_cmd(['sudo', 'ethtool', '-s', iface, 'speed', str(maxspeed), 'duplex', 'full'])
    print(stdout)
    return stdout

def main():
    parser = menu()
    args = parser.parse_args()
    vprint(args)
    iface = args.iface
    wired_devices = get_wired_devices()
    if not iface:
        print("E: must specify an interface", file=sys.stderr)
        pprint(wired_devices)
        sys.exit(1)

    iface_data = ethtool_read(iface, show_all=args.all)

    current_speed = parse_speed(iface_data["Speed"][0])
    speeds = [parse_speed(s) for s in iface_data["Supported link modes"]]
    max_speed = max(speeds)
    if not args.maximize_speed:
        sys.stdout.write(json.dumps(iface_data, indent=2) + '\n')
        sys.stdout.flush()
        return

    vprint("Current speed: {}".format(current_speed))
    if args.speed is not None:
        set_speed_to_max(iface, args.speed)
        print('{} Speed set to: {}'.format(iface, args.speed))
        return

    if args.maximize_speed:
        vprint("Maximizing speed. Max advertized speed: {}, currently ".format(max_speed, current_speed))
        if not args.force and current_speed == max_speed:
            print("Speed is currently maximized: {}: {}".format(iface, current_speed))
            return
        if not max_speed:
            errcho(iface_data)
            raise RuntimeError("Could not ascertain speed")

        set_speed_to_max(iface, max_speed)
        time.sleep(0.5)
        read_back_speed = get_current_speed(iface)
        print('{} Speed set to: {}'.format(iface, read_back_speed))
        if read_back_speed != max_speed:
            print("WARNING! Could not verify speed was set to max")
            sys.exit(1)

        return


if __name__ == '__main__':
    main()
