#!/usr/bin/env python
import os
import subprocess
from pathlib import Path
import time
import json
from collections import deque
from xmlrpc.client import ServerProxy

import redis
from loguru import logger
from kamerad.kam_types import server_parser, ServerArgs
from kamerad import arpj

from flask import Flask, request

# todo: global thread pool or maybe celery
app = Flask(__name__)
arp_scan_deque = deque(maxlen=1)
threads = {}
server = ServerProxy('http://localhost:9001/RPC2')


def run_cmd(cmd):
    # type: (list) -> (str, str)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    data = proc.stdout.read().decode()
    err = proc.stderr.read().decode()

    return proc.returncode, data, err


def loop_arp_scan(iface, period=10):
    global arp_scan_deque
    while True:
        dd = arpj.run_arp_scan(iface)
        print(dd)
        arp_scan_deque.append(dd)
        time.sleep(period)


@app.route("/check")
def check():
    return "true"


@app.route("/arp")
def arp():
    dd = arpj.run_arp()
    return json.dumps(dd)


@app.route("/arp-scan/<string:iface>", methods=["POST"])
def arp_scan_p(iface):
    dd = arpj.run_arp_scan(iface)
    return json.dumps(dd)


@app.route("/arp-scan/<string:iface>", methods=["GET"])
def arp_scan_g(iface):
    global arp_scan_deque
    try:
        return json.dumps(arp_scan_deque[0])
    except IndexError:
        return json.dumps([])


@app.route("/ismount", methods=["POST"])
def ismount_p():
    dname = request.get_data().decode()
    dpath = Path(dname)
    logger.debug("{} -> {}".format(dname, dpath))
    return json.dumps(os.path.ismount(dname))


@app.route("/diskinfo", methods=["POST"])
def disk_p():
    dname = request.get_data().decode()
    dname = dname
    dpath = Path(dname)
    # logger.debug("{} -> {}".format(dname, dpath))
    try:
        stats = os.statvfs(dname)
        bytes_free = stats.f_frsize * stats.f_bavail
    except FileNotFoundError:
        bytes_free = -1
    data = {
        "bytes_free": bytes_free,
        "realpath": os.path.realpath(dpath),
        "ismount": os.path.ismount(dpath),
        "isdir": os.path.isdir(dpath),
        "isfile": os.path.isfile(dpath),
        "exists": os.path.exists(dpath),
    }
    return json.dumps(data)


@app.route("/test_data", methods=["POST"])
def process_data():
    data = request.get_data()
    req_data = json.loads(data)
    logger.info(req_data)
    return json.dumps(req_data)


@app.route("/echo", methods=["GET", "POST"])
def echo():
    logger.debug(request)
    if request.method == "GET":
        return "GET".encode()
    return request.get_data()


@app.route("/mountall", methods=["POST"])
def mountall_p():
    logger.debug(request)
    process = "mount_nas"
    try:
        server.supervisor.stopProcess(process, True)
    except:
        pass
    server.supervisor.startProcess(process, True)
    code = 1
    out = "Finished"
    err = ""
    return json.dumps({"code": code, "out": out, "err": err})


# @app.route("/arp-scan/<string:iface>/start", methods=["POST"])
# def arp_scan_start(iface):
#     global threads
#     t = threading.Thread(target=loop_arp_scan, args=(iface, 10))
#     t.start()
#     threads[iface] = t
#     return 'true'


def entry(args: ServerArgs):
    app.run(host=args.host, port=args.port, debug=args.debug)


def cli_main():
    parser = server_parser("KAMERA health monitoring daemon.")
    args = ServerArgs.parse_obj(vars(parser.parse_args()))
    entry(args)


if __name__ == "__main__":
    cli_main()
