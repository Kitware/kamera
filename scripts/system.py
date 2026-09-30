import sys

from xmlrpc.client import ServerProxy, Fault

with open("/home/user/kw/SYSTEM_NAME") as f:
    SYSTEM_NAME = f.read().strip()

hosts = [ f"center0{SYSTEM_NAME}", f"left1{SYSTEM_NAME}", f"right2{SYSTEM_NAME}"]
group = "kamera"
pod = [
    "image_manager",
    f"{group}:fps_monitor",
    f"{group}:imageview",
    f"{group}:cam_rgb",
    f"{group}:cam_ir",
    f"{group}:cam_uv",
]
daemon = [
    "kamerad",
]

USAGE = "usage: system.py <host> {start,stop,restart} <cluster> | system.py <host> status"

if len(sys.argv) < 3:
    sys.exit(USAGE)
host = sys.argv[1].strip()
if host not in hosts:
    sys.exit("Invalid host %s!" % host)
action = sys.argv[2]
group2processes = {
    "pod": pod,
    "daemon": daemon,
    "central": [f"{group}:ins", f"{group}:daq"],
    "monitor": [
        f"{group}:cam_param_monitor",
        f"{group}:shapefile_monitor",
    ],
    "master": ["roscore"],
    "nas": ["mount_nas"],
    "detector": [f"{group}:detector"],
}
sup = ServerProxy("http://%s:9001/RPC2" % host)

if action == "status":
    try:
        infos = sup.supervisor.getAllProcessInfo()
    except OSError as e:
        sys.exit("%s: supervisor unreachable (%s)" % (host, e))
    for info in infos:
        name = info["name"] if info["group"] == info["name"] else "%s:%s" % (
            info["group"],
            info["name"],
        )
        print("%-16s %-32s %s" % (host, name, info["statename"]))
    raise SystemExit

if action not in ("start", "stop", "restart") or len(sys.argv) != 4:
    sys.exit(USAGE)
cluster = sys.argv[3]
if cluster not in group2processes:
    sys.exit("Invalid cluster %s! One of: %s" % (cluster, ", ".join(group2processes)))
# Sort so start is idempotent
processes = sorted(group2processes[cluster])


def stop(process):
    try:
        print("Stopping process %s." % process)
        sup.supervisor.stopProcess(process, True)
    except Fault as f:
        print(f)


def start(process):
    try:
        print("Starting process %s." % process)
        sup.supervisor.startProcess(process, True)
    except Fault as f:
        print(f)


print("Executing action %s on host %s with cluster %s." % (action, host, cluster))
for process in processes:
    if action in ("stop", "restart"):
        stop(process)
    if action in ("start", "restart"):
        start(process)
