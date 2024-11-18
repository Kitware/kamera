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
import wx
from wxpython_gui.system_control_panel.gui import MainFrame
import rospy

def main():
    node_name = 'system_control_panel_node'
    name_space = rospy.get_namespace()

    topic_names = {}

    # --------------------- Remote Image Server Topics -----------------------
    topic_names['left_rgb_srv_topic'] = '/nuvo1/rgb/rgb_view_service/get_image_view'
    topic_names['center_rgb_srv_topic'] = '/nuvo0/rgb/rgb_view_service/get_image_view'
    topic_names['right_rgb_srv_topic'] = '/nuvo2/rgb/rgb_view_service/get_image_view'

    # From the camera driver itself
    topic_names['left_rgb_srv_topic'] = '/nuvo1/rgb/rgb_driver/get_image_view'
    topic_names['center_rgb_srv_topic'] = '/nuvo0/rgb/rgb_driver/get_image_view'
    topic_names['right_rgb_srv_topic'] = '/nuvo2/rgb/rgb_driver/get_image_view'

    topic_names['left_uv_srv_topic'] = '/nuvo1/uv/uv_view_service/get_image_view'
    topic_names['center_uv_srv_topic'] = '/nuvo0/uv/uv_view_service/get_image_view'
    topic_names['right_uv_srv_topic'] = '/nuvo2/uv/uv_view_service/get_image_view'

    topic_names['left_ir_srv_topic'] = '/nuvo1/synched/ir_view_service'
    topic_names['center_ir_srv_topic'] = '/nuvo0/synched/ir_view_service'
    topic_names['right_ir_srv_topic'] = '/nuvo2/synched/ir_view_service'

    #topic_names['left_rgb_srv_topic'] = '/nuvo1/synched/rgb_view_service'
    #topic_names['center_rgb_srv_topic'] = '/nuvo0/synched/rgb_view_service'
    #topic_names['right_rgb_srv_topic'] = '/nuvo2/synched/rgb_view_service'

    topic_names['left_uv_srv_topic'] = '/nuvo1/synched/uv_view_service'
    topic_names['center_uv_srv_topic'] = '/nuvo0/synched/uv_view_service'
    topic_names['right_uv_srv_topic'] = '/nuvo2/synched/uv_view_service'
    # ------------------------------------------------------------------------

    # ------------------- Add To Event Log Service Topics --------------------
    topic_names['left_sys_event_log_srv'] = '/nuvo1/add_to_event_log'
    topic_names['center_sys_event_log_srv'] = '/nuvo0/add_to_event_log'
    topic_names['right_sys_event_log_srv'] = '/nuvo2/add_to_event_log'
    # ------------------------------------------------------------------------

    # ------------------------ Exposure Value Topics -------------------------
    topic_names['left_rgb_exposure'] = '/nuvo1/rgb/rgb_driver/exposure'
    topic_names['center_rgb_exposure'] = '/nuvo0/rgb/rgb_driver/exposure'
    topic_names['right_rgb_exposure'] = '/nuvo2/rgb/rgb_driver/exposure'

    topic_names['left_uv_exposure'] = '/nuvo1/uv/uv_driver/exposure'
    topic_names['center_uv_exposure'] = '/nuvo0/uv/uv_driver/exposure'
    topic_names['right_uv_exposure'] = '/nuvo2/uv/uv_driver/exposure'
    # ------------------------------------------------------------------------

    # --------------------------- Disk Space Topics --------------------------
    topic_names['sys0_disk_space'] = '/nuvo0/disk_free_bytes'
    topic_names['sys1_disk_space'] = '/nuvo1/disk_free_bytes'
    topic_names['sys2_disk_space'] = '/nuvo2/disk_free_bytes'
    # ------------------------------------------------------------------------

    topic_names['frame_rate_topic'] = '/daq/trigger_freq'


    topic_names['det_topic1'] = ''.join([name_space,
                                         'sprokit_detector_adapter/detections_out'])

    topic_names['nav_odom_topic'] = '/ins'

    window_title = 'System Control Panel'

    app = wx.App(False)
    frame = MainFrame(None, node_name, topic_names, False, window_title)
    frame.Show(True)
    app.MainLoop()
    return True

if __name__ == '__main__':
    # We get non-deterministic failures on initial launch. This keeps retrying
    # until a clean exit happens.
    resp = False
    while not resp:
        resp = main()
