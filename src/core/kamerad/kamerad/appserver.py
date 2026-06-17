#!/usr/bin/env python
import json
import os
import subprocess
from collections import deque
from pathlib import Path
from xmlrpc.client import ServerProxy

from flask import Flask, request
from loguru import logger

from kamerad import arpj
from kamerad.kam_types import ServerArgs, server_parser
from kamerad.power import PowerManager

# todo: global thread pool or maybe celery
app = Flask(__name__)
arp_scan_deque = deque(maxlen=1)
threads = {}
server = ServerProxy("http://localhost:9001/RPC2")
power_manager = None


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
        import time

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
    except Exception:
        pass
    server.supervisor.startProcess(process, True)
    code = 1
    out = "Finished"
    err = ""
    return json.dumps({"code": code, "out": out, "err": err})


@app.route("/power/status", methods=["GET"])
def power_status():
    if power_manager is None:
        return json.dumps({"ok": False, "error": "power manager unavailable"}), 503
    return json.dumps(power_manager.get_status())


@app.route("/power/shutdown", methods=["POST"])
def power_shutdown():
    if power_manager is None:
        return json.dumps({"ok": False, "error": "power manager unavailable"}), 503
    logger.info("Shutdown requested for {}", power_manager.hostname)
    result = power_manager.request_shutdown()
    status = 200 if result.get("ok", True) else 500
    return json.dumps(result), status


@app.route("/power/reboot", methods=["POST"])
def power_reboot():
    if power_manager is None:
        return json.dumps({"ok": False, "error": "power manager unavailable"}), 503
    logger.info("Reboot requested for {}", power_manager.hostname)
    result = power_manager.request_reboot()
    status = 200 if result.get("ok", True) else 500
    return json.dumps(result), status


def entry(args: ServerArgs):
    global power_manager
    redis_host = os.environ.get("REDIS_HOST")
    power_manager = PowerManager(server, redis_host=redis_host)
    power_manager.publish_diagnostics()
    power_manager.start_diagnostics()
    app.run(host=args.host, port=args.port, debug=args.debug)


def cli_main():
    parser = server_parser("KAMERA health monitoring daemon.")
    args = ServerArgs.parse_obj(vars(parser.parse_args()))
    entry(args)


if __name__ == "__main__":
    cli_main()
