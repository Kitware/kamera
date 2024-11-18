from wxpython_gui.cfg import kv, DOCK_KAM_REPO_DIR, REAL_KAM_REPO_DIR
import xmlrpclib


class SystemCommandsCall(object):
    """ This contains the interface for sending supervisor calls to each system
        and starting / stopping nodes.
    """
    def __init__(self, hosts):
        self.hosts = hosts
        self.process_group = "taiga" if "nuvo0" in hosts else "cas"
        self.supers = { host:xmlrpclib.Server('http://%s:9001/RPC2' % host)
                        for host in self.hosts }
        self.commands = ['up', 'down', 'restart']
        self.cams = ['ir', 'rgb', 'uv']
        self.devices = ['ins', 'daq']
        self.halt = "%s/src/run_scripts/inpath/kamera.halt" % REAL_KAM_REPO_DIR
        self.processes = ""
        self.cmd = ""

    def run(self):
        print(self.cmd, self.host, self.processes)
        if self.cmd not in self.commands or self.host not in self.hosts:
            raise RuntimeWarning("Unsupported Runtime command!")
            return
        for p in self.processes:
            if "postproc" in p:
                process = p
                try:
                    if self.cmd == "up":
                        self.supers[self.host].supervisor.startProcess(process, False)
                    elif self.cmd == "down":
                        self.supers[self.host].supervisor.stopProcess(process, False)
                    elif self.cmd == "restart":
                        try:
                            print("Stopping process %s." % process)
                            self.supers[self.host].supervisor.stopProcess(process)
                        except:
                            print("Starting process %s." % process)
                            self.supers[self.host].supervisor.startProcess(process, False)
                        print("Starting process %s." % process)
                        self.supers[self.host].supervisor.startProcess(process, False)
                except Exception as e:
                    print(e)
                    continue
            else:
                process = "%s:%s" % (self.process_group, p)
                try:
                    if self.cmd == "up":
                        self.supers[self.host].supervisor.startProcess(process, False)
                    elif self.cmd == "down":
                        self.supers[self.host].supervisor.stopProcess(process, False)
                    elif self.cmd == "restart":
                        try:
                            print("Stopping process %s." % process)
                            self.supers[self.host].supervisor.stopProcess(process)
                        except:
                            print("Starting process %s." % process)
                            self.supers[self.host].supervisor.startProcess(process, False)
                        print("Starting process %s." % process)
                        self.supers[self.host].supervisor.startProcess(process, False)
                except Exception as f:
                    print(f)
                    continue

    def command_detector(self, host, cmd):
        self.host = host
        self.cmd = cmd
        self.processes = ["detector"]

    def command_cameras(self, host, cmd, cameras):
        self.host = host
        self.cmd = cmd
        if not isinstance(cameras, list):
            cameras = [cameras]
        self.processes = ["cam_%s" % cam for cam in cameras]

    def command_nexus(self, host, cmd):
        self.host = host
        self.cmd = cmd
        self.processes = ["imageview"]

    def command_pod(self, host, cmd):
        self.host = host
        self.cmd = cmd
        self.processes = ["cam_ir", "cam_uv", "cam_rgb", "imageview"]

    def command_central(self, host, cmd, devices):
        self.host = host
        self.cmd = cmd
        if not isinstance(devices, list):
            devices = [devices]
        self.processes = devices

    def command_halt(self):
        # WARNING will kill the GUI (itself)
        self.bash = [self.halt]
        self.run_bash()

    def command_postproc(self, host, command, postproc):
        print('Final command: (%s %s)' % (command, postproc))
        self.host = host
        self.cmd = command
        self.processes = ["postproc:%s" % postproc]

    def run_bash(self):
        bash_cmd = ' '.join(self.bash)
        if len(self.bash) > 0:
            return_code = subprocess.Popen(bash_cmd, shell=True)

class SystemCommands(object):
    def __init__(self, hosts):
        self.scc = SystemCommandsCall(hosts)
        self.cams = ['ir', 'rgb', 'uv']

    def run_command(self, target, host, command,
                    containers=None, d=None, postproc=None):
        if target == "detector":
            self.scc.command_detector(host, command)
            self.scc.run()
        elif target == "cameras":
            self.scc.command_cameras(host, command, containers)
            self.scc.run()
        elif target == "nexus":
            self.scc.command_nexus(host, command)
            self.scc.run()
        elif target == "pod":
            self.scc.command_pod(host, command)
            self.scc.run()
        elif target == "central":
            self.scc.command_central(host, command, containers)
            self.scc.run()
        elif target == "halt":
            self.scc.command_halt()
            self.scc.run()
        elif target == "postproc":
            self.scc.command_postproc(host, command, postproc)
            self.scc.run()
        else:
            raise RuntimeWarning("Unsupported target!")
