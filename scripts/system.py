import sys
# import xmlrpclib # Python2
from xmlrpc.client import ServerProxy, Fault

with open("/home/user/kw/SYSTEM_NAME") as f:
    SYSTEM_NAME = f.read().strip()

if SYSTEM_NAME == "taiga":
    group = "taiga"
    hosts = ["nuvo0", "nuvo1", "nuvo2"]
    pod = ["image_manager", "kamerad", f"{group}:imageview", f"{group}:cam_rgb", f"{group}:cam_ir"]
else:
    group = "cas"
    hosts = ["cas0", "cas1", "cas2"]
    pod = ["image_manager", "kamerad", f"{group}:imageview", f"{group}:cam_uv", f"{group}:cam_rgb", f"{group}:cam_ir"]

host = sys.argv[1].strip()
if host not in hosts:
    print("Invalid host %s!" % host)
    raise SystemExit
action = sys.argv[2]
cluster = sys.argv[3]
group2processes= {"pod":pod,
                  "central":[f"{group}:ins", f"{group}:daq"],
                  "monitor":[f"{group}:cam_param_monitor", f"{group}:shapefile_monitor"],
                  "master": ["roscore"],
                  "nas": ["mount_nas"],
                  "detector": [f"{group}:detector"]
                 }
processes = group2processes[cluster]
# sup = xmlrpclib.Server('http://%s:9001/RPC2' % host)
sup = ServerProxy('http://%s:9001/RPC2' % host)

print("Executing action %s on host %s with cluster %s." % (action, host, cluster))
for process in processes:
    if action == "start":
        try:
            print("Starting process %s." % process)
            sup.supervisor.startProcess(process, True)
        except Fault as f:
            print(f)
    elif action == "stop":
        try:
            print("Stopping process %s." % process)
            sup.supervisor.stopProcess(process, True)
        except Fault as f:
            print(f)
    elif action == "restart":
        try:
            print("Restarting process %s." % process)
            try:
                sup.supervisor.stopProcess(process, True)
            except Exception as e:
                sup.supervisor.startProcess(process, True)
        except Fault as f:
            print(f)
    else:
        print("Invalid operation specified.")
