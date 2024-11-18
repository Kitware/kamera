import wxpython_gui.system_control_panel.form_builder_output_system_startup as form_builder_output_system_startup


class SystemStartup(form_builder_output_system_startup.MainFrame):
    def __init__(self, parent, system):
        """

        """
        # Initialize parent class
        form_builder_output_system_startup.MainFrame.__init__(self, parent)

        # Store the function that allow us to send logging requests.
        self.add_to_event_log = parent.add_to_event_log
        self.system = system
        self.hosts = system.scc.hosts

        self.Show()
        self.SetMinSize(self.GetSize())

    # -------------------------- All self.systems Commands ------------------------
    def on_start_all_nodes(self, event=None):
        for host in self.system.scc.hosts:
            self.system.run_command("pod", host, 'up')
            self.system.run_command("detector", host, 'up')
        self.system.run_command("central", self.hosts[0], 'up', 'daq')
        self.system.run_command("central", self.hosts[0], 'up', 'ins')

    def on_stop_all_nodes(self, event=None):
        self.add_to_event_log('command sent: stop entire system')
        for host in self.system.scc.hosts:
            self.system.run_command("pod", host, 'down')
            self.system.run_command("detector", host, 'down')
        self.system.run_command("central", self.hosts[0], 'down', 'daq')
        self.system.run_command("central", self.hosts[0], 'down', 'ins')

    def on_restart_all_nodes(self, event=None):
        self.add_to_event_log('command sent: restart all nodes')

        for host in self.system.scc.hosts:
            self.system.run_command("pod", host, 'restart')
            self.system.run_command("detector", host, 'restart')
        self.system.run_command("central", self.hosts[0], 'restart', 'daq')
        self.system.run_command("central", self.hosts[0], 'restart', 'ins')

    def on_restart_all_cameras(self, event=None):
        self.add_to_event_log('command sent: restart all cameras')
        for host in self.system.scc.hosts:
            self.system.run_command("cameras", host, 'restart', self.system.cams)

    def on_restart_all_rgb_cameras(self, event=None):
        self.add_to_event_log('command sent: restart all rgb cameras')
        for host in self.system.scc.hosts:
            self.system.run_command("cameras", host, 'restart', 'rgb')

    def on_restart_all_ir_cameras(self, event=None):
        self.add_to_event_log('command sent: restart all ir cameras')
        for host in self.system.scc.hosts:
            self.system.run_command("cameras", host, 'restart', 'ir')

    def on_restart_all_uv_cameras(self, event=None):
        self.add_to_event_log('command sent: restart all uv cameras')
        for host in self.system.scc.hosts:
            self.system.run_command("cameras", host, 'restart', 'uv')

    def on_restart_daq(self, event=None):
        self.add_to_event_log('command sent: restart daq')
        self.system.run_command("central", self.hosts[0], 'restart', 'daq')

    def on_restart_ins(self, event=None):
        self.add_to_event_log('command sent: restart ins')
        self.system.run_command("central", self.hosts[0], 'restart', 'ins')

    # ----------------------- Left-View (sys1) Commands ----------------------
    def on_start_all_nodes_sys1(self, event=None):
        self.add_to_event_log('command sent: start all nodes sys1')
        self.system.run_command("pod", self.hosts[1], 'up')
        self.system.run_command("detector", self.hosts[1], 'up')

    def on_stop_all_nodes_sys1(self, event=None):
        self.add_to_event_log('command sent: stop all nodes sys1')
        self.system.run_command("pod", self.hosts[1], 'down')
        self.system.run_command("detector", self.hosts[1], 'down')

    def on_restart_all_nodes_sys1(self, event=None):
        self.add_to_event_log('command sent: restart all nodes sys1')
        self.system.run_command("pod", self.hosts[1], 'restart')
        self.system.run_command("detector", self.hosts[1], 'restart')

    def on_restart_rgb_camera_sys1(self, event=None):
        self.add_to_event_log('command sent: restart rgb camera sys1')
        self.system.run_command("cameras", self.hosts[1], 'restart', 'rgb')

    def on_restart_ir_camera_sys1(self, event=None):
        self.add_to_event_log('command sent: restart ir camera sys1')
        self.system.run_command("cameras", self.hosts[1], 'restart', 'ir')

    def on_restart_uv_camera_sys1(self, event=None):
        self.add_to_event_log('command sent: restart uv camera sys1')
        self.system.run_command("cameras", self.hosts[1], 'restart', 'uv')

    def on_restart_nexus_sys1(self, event=None):
        self.add_to_event_log('command sent: restart nexus camera sys1')
        self.system.run_command('nexus', self.hosts[1], 'restart')

    # ----------------------- Center-View (sys0) Commands --------------------
    def on_start_all_nodes_sys0(self, event=None):
        self.add_to_event_log('command sent: start all nodes sys0')
        self.system.run_command("pod", self.hosts[0], 'up')
        self.system.run_command("detector", self.hosts[0], 'up')

    def on_stop_all_nodes_sys0(self, event=None):
        self.add_to_event_log('command sent: stop all nodes sys0')
        self.system.run_command("pod", self.hosts[0], 'down')
        self.system.run_command("detector", self.hosts[0], 'down')

    def on_restart_all_nodes_sys0(self, event=None):
        self.add_to_event_log('command sent: restart all nodes sys0')
        self.system.run_command("pod", self.hosts[0], 'restart')
        self.system.run_command("detector", self.hosts[0], 'restart')

    def on_restart_rgb_camera_sys0(self, event=None):
        self.add_to_event_log('command sent: restart rgb camera sys0')
        self.system.run_command("cameras", self.hosts[0], 'restart', 'rgb')

    def on_restart_ir_camera_sys0(self, event=None):
        self.add_to_event_log('command sent: restart ir camera sys0')
        self.system.run_command("cameras", self.hosts[0], 'restart', 'ir')

    def on_restart_uv_camera_sys0(self, event=None):
        self.add_to_event_log('command sent: restart uv camera sys0')
        self.system.run_command("cameras", self.hosts[0], 'restart', 'uv')

    def on_restart_nexus_sys0(self, event=None):
        self.add_to_event_log('command sent: restart nexus camera sys0')
        self.system.run_command('nexus', self.hosts[0], 'restart')

    # ----------------------- Right-View (sys2) Commands ---------------------
    def on_start_all_nodes_sys2(self, event=None):
        self.add_to_event_log('command sent: start all nodes sys2')
        self.system.run_command("pod", self.hosts[2], 'up')
        self.system.run_command("detector", self.hosts[2], 'up')

    def on_stop_all_nodes_sys2(self, event=None):
        self.add_to_event_log('command sent: stop all nodes sys2')
        self.system.run_command("pod", self.hosts[2], 'down')
        self.system.run_command("detector", self.hosts[2], 'down')

    def on_restart_all_nodes_sys2(self, event=None):
        self.add_to_event_log('command sent: restart all nodes sys2')
        self.system.run_command("pod", self.hosts[2], 'restart')
        self.system.run_command("detector", self.hosts[2], 'restart')

    def on_restart_rgb_camera_sys2(self, event=None):
        self.add_to_event_log('command sent: restart rgb camera sys2')
        self.system.run_command("cameras", self.hosts[2], 'restart', 'rgb')

    def on_restart_ir_camera_sys2(self, event=None):
        self.add_to_event_log('command sent: restart ir camera sys2')
        self.system.run_command("cameras", self.hosts[2], 'restart', 'ir')

    def on_restart_uv_camera_sys2(self, event=None):
        self.add_to_event_log('command sent: restart uv camera sys2')
        self.system.run_command("cameras", self.hosts[2], 'restart', 'uv')

    def on_restart_nexus_sys2(self, event=None):
        self.add_to_event_log('command sent: restart nexus camera sys2')
        self.system.run_command('nexus', self.hosts[2], 'restart')

    def on_cancel(self, event=None):
        """When the 'Cancel' button is selected.

        """
        self.Close()
