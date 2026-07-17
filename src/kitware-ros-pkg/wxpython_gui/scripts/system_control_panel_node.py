#!/usr/bin/env python3
"""
Library handling imagery simulation.
"""
from __future__ import division, print_function
import os

import wx
import rospy

from wxpython_gui.system_control_panel.gui import MainFrame
from roskv.impl.redis_envoy import RedisEnvoy
from roskv.util import filter_hosts_by_system


def main():
    node_name = "system_control_panel_node"
    name_space = rospy.get_namespace()

    envoy = RedisEnvoy(os.environ["REDIS_HOST"], client_name=node_name)
    enabled = envoy.get("/sys/enabled")
    all_hosts = envoy.get("/sys/arch/hosts")
    hosts = {h: all_hosts[h] for h in filter_hosts_by_system(all_hosts.keys())}

    topic_names = {}

    # From the camera driver itself (RGB) and view_server sync node (IR/UV).
    # Topic keys must match gui.MainFrame, which gates panels on /sys/enabled,
    # not /sys/channels (channels drives driver launch, not GUI visibility).
    for host, d in hosts.items():
        fov = d["fov"]
        fov_enabled = enabled.get(fov, {})
        if fov_enabled.get("rgb"):
            topic_names["_".join([fov, "rgb", "srv", "topic"])] = "/".join(
                ["", host, "rgb", "rgb_view_service", "get_image_view"]
            )
        for channel in ("ir", "uv"):
            if not fov_enabled.get(channel):
                continue
            topic_names["_".join([fov, channel, "srv", "topic"])] = "/".join(
                ["", host, "synched", "%s_view_service" % channel]
            )
    # ------------------------------------------------------------------------

    # ------------------- Add To Event Log Service Topics --------------------
    for host, d in hosts.items():
        fov = d["fov"]
        topic_names["_".join([fov, "sys", "event", "log", "srv"])] = "/".join(
            ["", host, "add_to_event_log"]
        )
    # ------------------------------------------------------------------------

    topic_names["nav_odom_topic"] = "/ins"

    window_title = "System Control Panel"

    app = wx.App(False)
    frame = MainFrame(None, node_name, topic_names, False, window_title)
    frame.Show(True)
    app.MainLoop()
    return True


if __name__ == "__main__":
    # We get non-deterministic failures on initial launch. This keeps retrying
    # until a clean exit happens.
    resp = False
    while not resp:
        resp = main()
