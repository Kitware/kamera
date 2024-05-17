import sys
# import xmlrpclib # Python2
from xmlrpc.client import ServerProxy, Fault

hosts = ["cas0", "cas1", "cas2", "nuvo0", "nuvo1", "nuvo2"]
host = sys.argv[1]
if host not in hosts:
    print("Invalid host!")
    raise SystemExit
action = sys.argv[2]
group = sys.argv[3]
group2processes= {"pod":["image_manager", "kamerad", "cas:imageview", "cas:cam_uv", "cas:cam_rgb", "cas:cam_ir"],
                  "central":["cas:ins", "cas:daq"],
                  "monitor":["cas:cam_param_monitor", "cas:shapefile_monitor"],
                  "master": ["roscore"],
                  "nas": ["mount_nas"],
                  "detector": ["cas:detector"]
                 }
processes = group2processes[group]
# sup = xmlrpclib.Server('http://%s:9001/RPC2' % host)
sup = ServerProxy('http://%s:9001/RPC2' % host)

print("Executing action %s on host %s with group %s." % (action, host, group))
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
