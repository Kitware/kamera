#!/usr/bin/env python
"""
ckwg +31
Copyright 2017 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Library handling imagery simulation.

"""
from __future__ import division, print_function
import os

import wx
import rospy

from wxpython_gui.system_control_panel.gui import MainFrame
from roskv.impl.redis_envoy import RedisEnvoy


def main():
    node_name = "system_control_panel_node"
    name_space = rospy.get_namespace()

    envoy = RedisEnvoy(os.environ["REDIS_HOST"], client_name=node_name)
    channels = envoy.get("/sys/channels").keys()
    hosts = envoy.get("/sys/arch/hosts")

    topic_names = {}

    # From the camera driver itself
    for host, d in hosts.items():
        channel = "rgb"
        fov = d["fov"]
        topic_names["_".join([fov, channel, "srv", "topic"])] = "/".join(
            ["", host, channel, "%s_view_service" % channel, "get_image_view"]
        )

    # From the view_server sync node
    for host, d in hosts.items():
        channel = "rgb"
        for channel in channels:
            if channel == "rgb":
                continue
            fov = d["fov"]
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
