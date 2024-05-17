#!/usr/bin/env python
import os
import redis
import time
import json
import threading
from collections import deque

from kamerahealth import arpj, menu

from flask import Flask

# todo: global thread pool or maybe celery
app = Flask(__name__)
arp_scan_deque = deque(maxlen=1)
threads = {}


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


@app.route("/arp-scan/<string:iface>/start", methods=["POST"])
def arp_scan_start(iface):
    global threads
    t = threading.Thread(target=loop_arp_scan, args=(iface, 10))
    t.start()
    threads[iface] = t
    return 'true'


def main():
    parser = menu.server_parser("redis-backed health cheking daemon")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
