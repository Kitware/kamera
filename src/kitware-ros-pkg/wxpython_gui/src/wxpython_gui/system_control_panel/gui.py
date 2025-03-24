#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Tuple
import os
import sys
import itertools
import re
import subprocess
import threading
import time
import datetime
import redis
import json
import copy
import psutil
from collections import OrderedDict
from functools import partial
from six import StringIO, string_types
from six.moves import queue
import requests

# import redis

# GUI imports
import wx
from wx.lib.wordwrap import wordwrap

# Vision / math
import cv2
import numpy as np
from PIL import Image as PILImage

# Geo
import pygeodesy
import shapely
import shapely.geometry
import shapefile

# ROS imports
import rospy
import std_msgs.msg
from custom_msgs.msg import GSOF_INS, Stat
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

# from roskv.impl.nop_kv import NopKV as ImplKV
# from roskv.impl.rosparam_kv import RosParamKV as ImplKV
from roskv.impl.redis_envoy import RedisEnvoy as ImplEnvoy
from roskv.rendezvous import ConditionalRendezvous  # , Governor
from roskv.util import simple_hash_args, MyTimeoutError

# Kamera imports
from custom_msgs.srv import (
    AddToEventLog,
    RequestCompressedImageView,
    RequestImageMetadata,
    RequestImageView,
    SetArchiving,
    CamSetAttr,
    CamGetAttr,
    CamGetAttrResponse,
)
import sys
from wxpython_gui.camera_models import load_from_file

# Relative Imports
import form_builder_output
import form_builder_output_effort_metadata
import form_builder_output_imagery_inspection
import form_builder_output_event_log_note
import form_builder_output_hot_key_list
import form_builder_output_collection_mode
import form_builder_output_log_panel
import form_builder_output_system_startup
import form_builder_output_camera_configuration

# Absolute imports
from wxpython_gui.cfg import (
    SYS_CFG,
    APP_GRAY,
    BRIGHT_RED,
    COLLECT_GREEN,
    ERROR_RED,
    FLAT_GRAY,
    WARN_AMBER,
    BRIGHT_GREEN,
    VERDANT_GREEN,
    WTF_PURPLE,
    SHAPE_COLLECT_BLUE,
    TEXTCTRL_DARK,
    LICENSE_STR,
    save_config_settings,
    save_camera_config,
    get_arch_path,
    host_from_fov,
    pull_gui_state,
    format_status,
    kv,
    geod,
)
from wxpython_gui.utils import diffpair, apply_ins_spoof
from wxpython_gui.CameraConfiguration import CameraConfiguration
from wxpython_gui.CollectionModeFrame import CollectionModeFrame
from wxpython_gui.MetadataEntryFrame import MetadataEntryFrame
from wxpython_gui.EventLogNoteFrame import EventLogNoteFrame
from wxpython_gui.LogFrame import LogFrame
from wxpython_gui.ImageInspectionFrame import ImageInspectionFrame
from wxpython_gui.HotKeyList import HotKeyList
from wxpython_gui.RemoteImagePanelFit import RemoteImagePanelFit
from wxpython_gui.SystemStartup import SystemStartup
from wxpython_gui.SystemCommands import SystemCommands
from wxpython_gui.DetectorState import set_detector_state, detector_state, EPodStatus

import gui_utils


G_time_note_started = None  # type: datetime.datetime

# Prosilicas
PR_GAIN_MIN_PARAM = "GainAutoMin"
PR_GAIN_MAX_PARAM = "GainAutoMax"
PR_EXPOSURE_MIN_PARAM = "ExposureAutoMin"
PR_EXPOSURE_MAX_PARAM = "ExposureAutoMax"
PR_GAIN_MIN = 0
PR_EXPOSURE_MIN = 0.03
# Use prosilica gain scale to simplify things
GAIN_MAX = 32

# Phase One
P1_GAIN_MIN_PARAM = "ISO_Min"
P1_GAIN_MAX_PARAM = "ISO_Max"
P1_EXPOSURE_MAX_PARAM = "Shutter_Speed_Min"
P1_EXPOSURE_MIN_PARAM = "Shutter_Speed_Max"
P1_EXPOSURE_STOPS = [
    0.0625,
    0.125,
    0.2,
    0.25,
    0.3125,
    0.4,
    0.5,
    0.625,
    0.8,
    1.0,
    1.25,
    1.5625,
    2.0,
    2.5,
    3.125,
    4.0,
    5.0,
    6.25,
    8.0,
    10,
    12.5,
    16.7,
    20,
    25,
    33.3,
    40,
    66.7,
    76.9,
    100,
    125,
    166.7,
    200,
    250,
    333.3,
    400,
    500,
    600,
    800,
    1000,
]
P1_GAIN_MIN = 0
# P1_GAIN_MAX = 1600
P1_EXPOSURE_MIN = (1 / 16000.0) * 1e3


def nearest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


class MainFrame(form_builder_output.MainFrame):
    # constructor
    def __init__(
        self,
        parent,
        node_name,
        topic_names,
        compressed=False,
        window_title="System Control Panel",
    ):
        """
        :param node_name:
        :type node_name: str

        :param topic_names: Dictionary of topic names.
        :type topic_names: dict

        """
        # initialize parent class
        form_builder_output.MainFrame.__init__(self, parent)
        self.SetTitle(window_title)
        icon = wx.EmptyIcon()
        icon.CopyFromBitmap(
            wx.Bitmap(
                os.path.expanduser("~/noaa_kamera/src/cfg/seal-icon.png"),
                wx.BITMAP_TYPE_ANY,
            )
        )
        self.SetIcon(icon)

        # self.rc = redis.Redis()

        # self.m_menuItem29.Disable()
        # self.m_menuItem31.Disable()
        # Match "fl00..." type dirs
        self.flight_re = re.compile("fl[0-9]*$")
        self.postprocessing_q = queue.Queue()
        self.last_status_str = ""
        # Commands to human readable
        self.c2s = {
            "homography": "Fine-Tune Tracking",
            "detections": "Detection Summary",
            "flight_summary": "Flight Summary",
        }

        self.hosts = sorted(SYS_CFG["arch"]["hosts"].keys())
        print("HOSTS: ")
        print(self.hosts)
        self.last_busy = {h: False for h in self.hosts}
        self.last_collecting = None

        # Initialize other frames.
        self._hot_key_list = False
        self._system_startup_frame = False
        self._metadata_entry_frame = False
        self._camera_config_frame = False
        self._image_inspection_frame = False
        self._log_panel = False
        self._event_log_note_frame = False
        self._collection_mode_frame = False
        self._collect_in_region = None
        self._last_collect_in_region_check_time = -np.inf
        self._last_fixed_overlap_check_time = -np.inf
        self._last_fixed_overlap_error_time = -np.inf

        self.topic_names = topic_names
        self.last_popup = datetime.datetime.now()
        self.last_msg_txt = ""

        self._sys_cfg = (
            None  # this is cached from the last good call to GetStringSelection()
        )

        # Setting wx.CB_READONLY doesn't allow you to SetEditable(True) later.
        self.effort_combo_box.SetEditable(False)
        self.camera_config_combo.SetEditable(False)

        self._collecting = None
        self.collecting = SYS_CFG["arch"]["is_archiving"]  # Triggers a GUI update.

        self._effort_metadata_dict = {}
        self._camera_configuration_dict = {}

        self._claheClipLimit = SYS_CFG["ir_contrast_strength"]

        self._debug = kv.get_dict("/debug", {})
        # print("Debug state: {}".format(self._debug))
        self._debug_enable = self._debug.get("enable", False)
        self._spoof_gps = kv.get("/debug/spoof_gps", False) and self._debug_enable
        self._spoof_events = int(kv.get("/debug/spoof_events", 0))
        self.last_ins_time = 0.0
        if self._spoof_gps:
            print("SPOOF GPS ENABLED")
        if self._spoof_events == 1:
            print("SPOOF EVENTS ENABLED")

        # Propogate GUI state from cached config
        self.set_camera_config_dict(SYS_CFG["camera_cfgs"])
        self.load_config_settings()
        self.update_project_flight_params()

        self.update_show_hide()

        # Set up ROS connections.
        rospy.init_node(node_name, anonymous=True)

        # --------------------------- Image Streams --------------------------
        # These will all be compressed images fit to the panel.
        self.remote_image_panels = []

        self.enabled = kv.get("/sys/enabled")
        # RGB images.
        if self.enabled["left"]["rgb"]:
            ret = RemoteImagePanelFit(
                self.left_rgb_panel,
                topic_names["left_rgb_srv_topic"],
                None,
                self.left_rgb_status_text,
                compressed,
                self.left_rgb_histogram_panel,
                attrs={"chan": "rgb", "fov": "left"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["center"]["rgb"]:
            ret = RemoteImagePanelFit(
                self.center_rgb_panel,
                topic_names["center_rgb_srv_topic"],
                None,
                self.center_rgb_status_text,
                compressed,
                self.center_rgb_histogram_panel,
                attrs={"chan": "rgb", "fov": "center"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["right"]["rgb"]:
            ret = RemoteImagePanelFit(
                self.right_rgb_panel,
                topic_names["right_rgb_srv_topic"],
                None,
                self.right_rgb_status_text,
                compressed,
                self.right_rgb_histogram_panel,
                attrs={"chan": "rgb", "fov": "right"},
            )
            self.remote_image_panels.append(ret)

        # IR images.
        if self.enabled["left"]["ir"]:
            ret = RemoteImagePanelFit(
                self.left_ir_panel,
                topic_names["left_ir_srv_topic"],
                None,
                self.left_ir_status_text,
                compressed,
                self.left_ir_histogram_panel,
                attrs={"chan": "ir", "fov": "left"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["center"]["ir"]:
            ret = RemoteImagePanelFit(
                self.center_ir_panel,
                topic_names["center_ir_srv_topic"],
                None,
                self.center_ir_status_text,
                compressed,
                self.center_ir_histogram_panel,
                attrs={"chan": "ir", "fov": "center"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["right"]["ir"]:
            ret = RemoteImagePanelFit(
                self.right_ir_panel,
                topic_names["right_ir_srv_topic"],
                None,
                self.right_ir_status_text,
                compressed,
                self.right_ir_histogram_panel,
                attrs={"chan": "ir", "fov": "right"},
            )
            self.remote_image_panels.append(ret)

        # UV images.
        if self.enabled["left"]["uv"]:
            ret = RemoteImagePanelFit(
                self.left_uv_panel,
                topic_names["left_uv_srv_topic"],
                None,
                self.left_uv_status_text,
                compressed,
                self.left_uv_histogram_panel,
                attrs={"chan": "uv", "fov": "left"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["center"]["uv"]:
            ret = RemoteImagePanelFit(
                self.center_uv_panel,
                topic_names["center_uv_srv_topic"],
                None,
                self.center_uv_status_text,
                compressed,
                self.center_uv_histogram_panel,
                attrs={"chan": "uv", "fov": "center"},
            )
            self.remote_image_panels.append(ret)

        if self.enabled["right"]["uv"]:
            ret = RemoteImagePanelFit(
                self.right_uv_panel,
                topic_names["right_uv_srv_topic"],
                None,
                self.right_uv_status_text,
                compressed,
                self.right_uv_histogram_panel,
                attrs={"chan": "uv", "fov": "right"},
            )
            self.remote_image_panels.append(ret)

        # ----------------------------- INS ----------------------------------
        self.lat0 = None
        self.lon0 = None
        self.h0 = None
        self.ins_state_sub = rospy.Subscriber(
            topic_names["nav_odom_topic"], GSOF_INS, self.ins_state_ros, queue_size=1
        )

        self.raw_msg_sub = rospy.Subscriber(
            "/rawmsg", std_msgs.msg.String, self.cb_raw_message_popup, queue_size=1
        )
        # --------------------------------------------------------------------

        # ------------------------- Add To Event Log--------------------------
        self._left_sys_event_log_srv = rospy.ServiceProxy(
            topic_names["left_sys_event_log_srv"], AddToEventLog, persistent=False
        )

        self._center_sys_event_log_srv = rospy.ServiceProxy(
            topic_names["center_sys_event_log_srv"], AddToEventLog, persistent=False
        )

        self._right_sys_event_log_srv = rospy.ServiceProxy(
            topic_names["right_sys_event_log_srv"], AddToEventLog, persistent=False
        )
        # --------------------------------------------------------------------

        # ------------------------- Missed Frame --------------------------
        self.camera_setting_rgb_uv_combo.Bind(wx.EVT_COMBOBOX, self.on_modal_selection)
        # Call once to hide/show proper options
        self.on_modal_selection(None)

        self.camera_setting_subsys.Bind(
            wx.EVT_COMBOBOX, self.on_camera_setting_subsys_selection
        )
        self.on_camera_setting_subsys_selection(None)

        # ----------------------------- Hot Keys -----------------------------
        entries = []

        # Bind ctrl+s to start/stop recording.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.reverse_collecting_state, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("S"), random_id)
        # Bind ctrl+d to start detectors.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.on_start_detectors, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("D"), random_id)

        # Bind ctrl+f to stop detectors.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.on_stop_detectors, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("F"), random_id)
        # Bind ctrl+h to hot key menu.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.on_hot_key_help, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("H"), random_id)

        # Bind ctrl+e to set context to exposure entry.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.set_focus_to_exposure, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("E"), random_id)

        # Bind ctrl+n to add note to log.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.on_add_to_event_log, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("N"), random_id)

        # Bind ctrl+o to next previous camera configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.previous_camera_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("O"), random_id)

        # Bind ctrl+p to next camera configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.next_camera_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("P"), random_id)

        # Bind ctrl+i to next previous effort configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.previous_effort_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("I"), random_id)

        # Bind ctrl+k to next effort configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self.next_effort_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("K"), random_id)

        accel = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(accel)

        # --------------------------------------------------------------------

        # rospy.add_client_shutdown_hook(self.on_close_button)
        self.system = SystemCommands(self.hosts)

        self.Bind(wx.EVT_CLOSE, self.when_closed)
        self.Bind(wx.EVT_SIZE, self.on_resize)

        # So that we can check that the node is still alive.
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        FAST_TIMER_MS = 200
        self.timer.Start(FAST_TIMER_MS)

        # So that we can check that the node is still alive.
        self.slow_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_slow_timer, self.slow_timer)
        self.slow_timer.Start(1000)
        # TODO
        # self.lastLdet = kv.get("/nuvo1/detector/health//frame")
        # self.lastCdet = kv.get("/nuvo0/detector/health//frame")
        # self.lastRdet = kv.get("/nuvo2/detector/health//frame")
        self.delay = 0
        self.Show()
        print(self.GetSize())
        self.SetMinSize((1500, 980))
        # self.detector_gauge.Hide()

        if self.collecting:
            self._disable_state_controls()
        # Add in pause before displaying imagery

    def system_sanity_check(self):
        fails = 0
        unmounts = []
        nas_unmounts = []
        for host in self.hosts:
            try:
                diskinfo = requests.post(
                    "http://{}:8987/diskinfo".format(host),
                    data=SYS_CFG["local_ssd_mnt"],
                    timeout=0.1,
                )
            except requests.exceptions.ConnectionError:
                rospy.logwarn("Could not access disk info from system %s." % host)
                continue
            diskinfo = json.loads(diskinfo.text)
            mounted = diskinfo["ismount"]
            try:
                if not mounted:
                    unmounts.append(host)
                else:
                    if 0 == int(host[-1]):
                        self.center_sys_space_static_text.SetLabel(
                            "Disk Space: %0.2f GB"
                            % (float(diskinfo["bytes_free"]) / 1e9)
                        )
                        self.center_sys_space_static_text.SetForegroundColour(
                            COLLECT_GREEN
                        )
                    elif 1 == int(host[-1]):
                        self.left_sys_space_static_text.SetLabel(
                            "Disk Space: %0.2f GB"
                            % (float(diskinfo["bytes_free"]) / 1e9)
                        )
                        self.left_sys_space_static_text.SetForegroundColour(
                            COLLECT_GREEN
                        )
                    else:
                        self.right_sys_space_static_text.SetLabel(
                            "Disk Space: %0.2f GB"
                            % (float(diskinfo["bytes_free"]) / 1e9)
                        )
                        self.right_sys_space_static_text.SetForegroundColour(
                            COLLECT_GREEN
                        )
            except Exception as e:
                rospy.logerr("Failed to connect to host {}: {}".format(host, e))
                return
            try:
                diskinfo = requests.post(
                    "http://{}:8987/diskinfo".format(host),
                    SYS_CFG["nas_mnt"],
                    timeout=0.1,
                )
                diskinfo = json.loads(diskinfo.text)
                # rospy.loginfo("{}: {}".format(host, diskinfo))
                mounted = diskinfo["ismount"]
                if not mounted:
                    nas_unmounts.append(host)
                else:
                    if 0 == int(host[-1]):
                        self.nas_disk_space.SetLabel(
                            "NAS Disk Space: %0.2f GB"
                            % (float(diskinfo["bytes_free"]) / 1e9)
                        )
                        self.nas_disk_space.SetForegroundColour(COLLECT_GREEN)
            except Exception as exc:
                # rospy.logerr("unable to connect to host {}: {}".format(host, exc))
                fails += 1
        for host in unmounts:
            if 0 == int(host[-1]):
                self.center_sys_space_static_text.SetLabel("Disk Space: Err")
                self.center_sys_space_static_text.SetForegroundColour(ERROR_RED)
            elif 1 == int(host[-1]):
                self.left_sys_space_static_text.SetLabel("Disk Space: Err")
                self.left_sys_space_static_text.SetForegroundColour(ERROR_RED)
            else:
                self.right_sys_space_static_text.SetLabel("Disk Space: Err")
                self.right_sys_space_static_text.SetForegroundColour(ERROR_RED)
        if len(unmounts) > 0:
            errmsg = "ERROR: One or more hosts has an sdd mount issue: {}\n".format(
                unmounts
            )
            rospy.logerr(errmsg)
            for host in unmounts:
                res = requests.post(
                    "http://{}:8987/mountall".format(host), data="/mnt/data"
                )

        if len(nas_unmounts) > 0:
            self.nas_disk_space.SetLabel("NAS Disk Space: Err")
            self.nas_disk_space.SetForegroundColour(ERROR_RED)
            errmsg = "ERROR: One or more hosts has an NAS mount issue: {}\n".format(
                nas_unmounts
            )
            rospy.logerr(errmsg)
            for host in nas_unmounts:
                res = requests.post(
                    "http://{}:8987/mountall".format(host), data=SYS_CFG["nas_mnt"]
                )

    def on_modal_selection(self, event):
        self.on_camera_setting_subsys_selection(event)
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        if mode == "IR":
            self.exposure_max_value_txt_ctrl.Disable()
            self.exposure_min_value_txt_ctrl.Disable()
            self.gain_min_value_txt_ctrl.Disable()
            self.gain_max_value_txt_ctrl.Disable()
            self.ir_nuc_time.Enable()
        else:
            self.exposure_max_value_txt_ctrl.Enable()
            self.exposure_min_value_txt_ctrl.Enable()
            self.gain_min_value_txt_ctrl.Enable()
            self.gain_max_value_txt_ctrl.Enable()
            self.ir_nuc_time.Disable()

    def set_camera_parameter(self, hosts, mode, param, val):
        mode = mode.lower()
        if "ISO" in param:
            # Scaling for Phase One
            val = int(val * 50)
        if mode == "rgb" and "Shutter" in param:
            print("Casting exposure to phase one stop.")
            # Set exposure to stop for Phase One
            val = float(val) * 1e3
            if val not in P1_EXPOSURE_STOPS:
                newval = nearest(P1_EXPOSURE_STOPS, val)
                msg = (
                    "Exposure value is not valid for Phase One, rounding %0.4fms to %0.4fms."
                    % (val, newval)
                )
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                p1val = newval * 1e-3
                print(val, newval)
                val = str(p1val)
                if "Max" in param:
                    self.exposure_min_value_txt_ctrl.SetValue(str(newval))
                elif "Min" in param:
                    self.exposure_max_value_txt_ctrl.SetValue(str(newval))
            else:
                val = str(val * 1e-3)

        for host in hosts:
            SYS_CFG["requested_geni_params"][host][mode][param] = val
            self.add_to_event_log(
                "Set parameter '%s' on %s:%s to %s." % (param, host, mode, val)
            )

    def check_box_val(self, val, typ=int):
        try:
            if val == "":
                return None
            val = typ(val)
        except ValueError:
            msg = "Settings value must be a valid number"
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return None
        return val

    def on_camera_setting_subsys_selection(self, event):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection().lower()
        fov = self.camera_setting_subsys.GetString(
            self.camera_setting_subsys.GetCurrentSelection()
        ).lower()
        if fov == "all":
            host = "null"
        else:
            host = host_from_fov(fov)
        topic_base = "/".join(["", "sys", "actual_geni_params", host, mode, ""])
        if mode == "ir":
            nuc_time = kv.get(topic_base + "CorrectionAutoDeltaTime", None)
            nuc_time = nuc_time if nuc_time is not None else 0
            self.ir_nuc_time.SetValue(str(nuc_time))
        elif mode == "rgb" or mode == "uv":
            if mode == "rgb":
                exp_factor = float(1e-3)
                gain_factor = 1 / 50.0
                GAIN_MAX_PARAM = P1_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = P1_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = P1_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = P1_EXPOSURE_MIN_PARAM
            else:
                exp_factor = float(1e3)
                gain_factor = 1
                GAIN_MAX_PARAM = PR_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = PR_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = PR_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = PR_EXPOSURE_MIN_PARAM
            gain_min = kv.get(topic_base + GAIN_MIN_PARAM, None)
            gain_max = kv.get(topic_base + GAIN_MAX_PARAM, None)
            exp_min = kv.get(topic_base + EXPOSURE_MIN_PARAM, None)
            exp_max = kv.get(topic_base + EXPOSURE_MAX_PARAM, None)
            gain_min = (
                int(float(gain_min) * gain_factor) if gain_min is not None else ""
            )
            gain_max = (
                int(float(gain_max) * gain_factor) if gain_max is not None else ""
            )
            exp_max = float(exp_max) / exp_factor if exp_max is not None else ""
            exp_min = float(exp_min) / exp_factor if exp_min is not None else ""
            self.exposure_min_value_txt_ctrl.SetValue(str(exp_min))
            self.exposure_max_value_txt_ctrl.SetValue(str(exp_max))
            self.gain_min_value_txt_ctrl.SetValue(str(gain_min))
            self.gain_max_value_txt_ctrl.SetValue(str(gain_max))

    def on_set_camera_parameter(self, event):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        host = self.camera_setting_subsys.GetString(
            self.camera_setting_subsys.GetCurrentSelection()
        ).lower()
        if host == "all":
            hosts = self.hosts
        else:
            hosts = [host_from_fov(host)]
        gain_min = None
        gain_max = None
        exp_min = None
        exp_max = None
        focus_pos = None
        if mode == "IR":
            nuc_time = self.check_box_val(self.ir_nuc_time.GetValue())
            if nuc_time is None:
                pass
            elif nuc_time > 30 or nuc_time < 0:
                msg = "NUC value must be >= 0 and <= 30 (in minutes)."
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            elif nuc_time == 0:
                # Received a 0 time
                # Turn off delta time and enable delta temp
                param = "CorrectionAutoUseDeltaTime"
                self.set_camera_parameter(hosts, mode, param, val=0)
                param = "CorrectionAutoUseDeltaTemp"
                self.set_camera_parameter(hosts, mode, param, val=1)
                return
            else:
                # Received a certain delta T to use, disable delta temp and enable delta time
                # then set delta time
                param = "CorrectionAutoUseDeltaTime"
                self.set_camera_parameter(hosts, mode, param, val=1)
                param = "CorrectionAutoUseDeltaTemp"
                self.set_camera_parameter(hosts, mode, param, val=0)
                param = "CorrectionAutoDeltaTime"
                self.set_camera_parameter(hosts, mode, param, val=nuc_time)
        elif mode == "RGB" or mode == "UV":
            if mode == "RGB":
                factor = float(1e-3)
                GAIN_MAX_PARAM = P1_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = P1_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = P1_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = P1_EXPOSURE_MIN_PARAM
                EXPOSURE_MIN = P1_EXPOSURE_MIN
            else:
                factor = 1e3
                GAIN_MAX_PARAM = PR_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = PR_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = PR_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = PR_EXPOSURE_MIN_PARAM
                EXPOSURE_MIN = PR_EXPOSURE_MIN
            # factor = 1e3 Prosilica
            exp_min = self.check_box_val(
                self.exposure_min_value_txt_ctrl.GetValue(), float
            )
            exp_max = self.check_box_val(
                self.exposure_max_value_txt_ctrl.GetValue(), float
            )
            if exp_min is None and exp_max is None:
                pass
            elif (exp_min is None and exp_max is not None) or (
                exp_max is None and exp_min is not None
            ):
                msg = "Both auto exposure min/max value must be set."
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                pass
            elif exp_min < EXPOSURE_MIN or exp_min > SYS_CFG["arch"]["max_exposure_ms"]:
                msg = "Exposure minimum value must be >= %s and <= %sms." % (
                    EXPOSURE_MIN,
                    SYS_CFG["arch"]["max_exposure_ms"],
                )
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            else:
                param = EXPOSURE_MIN_PARAM
                exp_min_ms = exp_min * factor
                self.set_camera_parameter(hosts, mode, param, val=exp_min_ms)
                if (
                    exp_max < EXPOSURE_MIN
                    or exp_max > SYS_CFG["arch"]["max_exposure_ms"]
                    or exp_max < exp_min
                ):
                    msg = (
                        "Exposure maximum value must be >= %s and <= %sms"
                        " and greater than min exposure."
                        % (EXPOSURE_MIN, SYS_CFG["arch"]["max_exposure_ms"])
                    )
                    dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
                else:
                    exp_max_ms = exp_max * factor
                    param = EXPOSURE_MAX_PARAM
                    self.set_camera_parameter(hosts, mode, param, val=exp_max_ms)

            gain_min = self.check_box_val(self.gain_min_value_txt_ctrl.GetValue())
            gain_max = self.check_box_val(self.gain_max_value_txt_ctrl.GetValue())
            if gain_min is None and gain_max is None:
                pass
            elif (gain_min is None and gain_max is not None) or (
                gain_max is None and gain_min is not None
            ):
                msg = "Both auto gain min/max value must be set."
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                pass
            elif gain_min < 0 or gain_min > GAIN_MAX:
                msg = "Gain minimum value must be >= 0 and <= %s." % GAIN_MAX
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            else:
                param = GAIN_MIN_PARAM
                self.set_camera_parameter(hosts, mode, param, val=gain_min)
                if gain_max is None:
                    pass
                elif gain_max < 0 or gain_max > GAIN_MAX or gain_max < gain_min:
                    msg = (
                        "Gain maximum value must be >= 0 and <= %s and greater than gain_min."
                        % GAIN_MAX_PARAM
                    )
                    dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
                else:
                    param = GAIN_MAX_PARAM
                    self.set_camera_parameter(hosts, mode, param, val=gain_max)
        else:
            msg = "Invalid mode selected, must be IR, UV, or RGB."
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        if len(hosts) > 1:
            host = "all"
        else:
            host = hosts[0]

    def on_exp_key_press(self, event):
        # type: (wx.KeyEvent) -> None
        """13 is enter. wx.stc.STC_KEY_RETURN must be a later version or something"""
        if event.GetKeyCode() in [13, 370]:
            self.on_set_exposure(event)
        else:
            event.Skip()

    def log_state(self):
        data = {
            key: SYS_CFG[key]
            for key in SYS_CFG.keys()
            if key
            in ["collection_mode", "collection_mode_parameter", "flight_number_str"]
        }
        effort_name = self.effort_combo_box.GetStringSelection()
        data["effort_metadata_dict"] = SYS_CFG["effort_metadata_dict"][effort_name]
        datas = json.dumps(data)
        self.add_to_event_log(datas)

    def on_slow_timer(self, event):
        detector_state.sync()
        self.update_static_text()
        d_desired = {}
        d_status = {}
        self.system_sanity_check()
        for host in self.hosts:
            fov = SYS_CFG["arch"]["hosts"][host]["fov"]
            text_attr = getattr(self, "{}_sys_detector_frames".format(fov))
            desired = detector_state.desired[host] or EPodStatus.Unknown
            if desired is EPodStatus.Off:
                text_attr.SetForegroundColour(TEXTCTRL_DARK)
                continue
            status = detector_state.decide_status(host, desired)
            d_desired[host] = desired
            d_status[host] = status
            if status is EPodStatus.Unknown:
                text_attr.SetForegroundColour(WTF_PURPLE)
            elif status is EPodStatus.Pending:
                text_attr.SetForegroundColour(WARN_AMBER)
            elif status.is_ok():
                text_attr.SetForegroundColour(VERDANT_GREEN)
            else:
                text_attr.SetForegroundColour(ERROR_RED)

        print(d_status.values())

        if not len(d_desired):
            self.detectors_gauge.SetBackgroundColour(FLAT_GRAY)
        else:
            if any([status is EPodStatus.Failed for status in d_status.values()]):
                self.detectors_gauge.SetBackgroundColour(ERROR_RED)
            elif any([status is EPodStatus.Pending for status in d_status.values()]):
                self.detectors_gauge.SetBackgroundColour(WARN_AMBER)
            elif all([status.is_ok() for status in d_status.values()]):
                self.detectors_gauge.SetBackgroundColour(BRIGHT_GREEN)
            else:
                self.detectors_gauge.SetBackgroundColour(ERROR_RED)

        is_busy = {h: 0 for h in self.hosts}
        current_task = {}
        host_last_status = {}
        for host in is_busy:
            for command in SYS_CFG["postproc_commands"]:
                try:
                    busy = int(kv.get("/postproc/%s/%s/busy" % (host, command)))
                except KeyError as e:
                    busy = 0
                if busy == 1:
                    current_task[host] = command
                is_busy[host] += int(busy)

        # self.m_status_bar.SetBackgroundColour(APP_GRAY)
        # self.m_status_bar.SetBackgroundColour(VERDANT_GREEN)
        # self.m_status_bar.SetBackgroundColour(ERROR_RED)
        status_str = ""
        busy = False
        for host in sorted(is_busy):
            if is_busy[host] > 0:
                command = current_task[host]
                current_dir = kv.get("/postproc/%s/%s/flight_dir" % (host, command))
                status_str += "|| %s Processing %s on %s. ||" % (
                    host,
                    self.c2s[command],
                    current_dir,
                )
                busy = True
            else:
                try:
                    command, flight_dir = self.postprocessing_q.get(timeout=0.01)
                except queue.Empty:
                    command = flight_dir = None
                # If host was busy last check, report status when it finished
                if self.last_busy[host]:
                    try:
                        status = kv.get("/postproc/%s/%s/status" % (host, command))
                    except:
                        status = "%s waiting for job." % host
                    status_str += "|| %s ||" % status
                if command is not None:
                    kv.put("/postproc/%s/%s/flight_dir" % (host, command), flight_dir)
                    self.system.run_command(
                        "postproc", host, "restart", postproc=command, d=flight_dir
                    )
                    status_str += "|| Submitting job %s to %s. ||" % (command, host)
                    self.add_to_event_log(
                        "command sent: Create {}"
                        " for flight {}.".format(self.c2s[command], flight_dir)
                    )
                else:
                    status_str += "|| %s waiting for job. ||" % host
        if busy:
            self.m_status_bar.SetBackgroundColour(SHAPE_COLLECT_BLUE)
        else:
            self.m_status_bar.SetBackgroundColour(APP_GRAY)

        if status_str != self.last_status_str:
            self.m_status_bar.SetStatusText(status_str)
            self.last_status_str = status_str
        self.delay += 1
        if self.delay == 2:
            for panel in self.remote_image_panels:
                print("Starting image threads")
                panel.start_image_thread()

    def on_timer(self, event):
        """Manages all updates that should happen at fixed rate."""
        tic = time.time()
        if rospy.is_shutdown():
            self.on_close_button(None)

        self.update_collect_colors()

        # Check to see if imagery has been received recently.
        for panel in self.remote_image_panels:
            # Refresh images if needed
            panel.update_all_if_needed()
            chan = panel.attrs["chan"]
            if panel.last_update is None:
                panel.status_static_text.SetLabel(format_status())
                panel.status_static_text.SetForegroundColour(BRIGHT_RED)
            else:
                dt = time.time() - panel.last_update
                if dt > 10:
                    panel.status_static_text.SetLabel(format_status(dt=dt))
                    panel.status_static_text.SetForegroundColour(BRIGHT_RED)

        # Check to see if imagery has been received recently.
        if self._image_inspection_frame:
            if self._image_inspection_frame.full_view_rp:
                try:
                    zoom = self._image_inspection_frame.full_view_rp.zoom_panel
                    fit = self._image_inspection_frame.full_view_rp.fit_panel
                    # Refresh images if needed.
                    zoom.update_all_if_needed()
                    fit.update_all_if_needed()
                except AttributeError as e:
                    rospy.logwarn(e)
                    # self._image_inspection_frame.Close()
                    # self._image_inspection_frame = None
                    pass

    def dump_sizes(self):
        for fmt in ["m_panel_{}_{}", "{}_{}_panel"]:
            print(fmt)
            for fov in ["left", "center", "right"]:
                for chan in ["rgb", "uv", "ir"]:
                    panel = getattr(self, fmt.format(fov, chan))

                    _width, _height = panel.GetSize()
                    print("{} x {} ".format(_width, _height), end="\t")
                print("")
            print("---")

    def on_resize(self, event):
        event.Skip()

    # -------------------------------- Properties -----------------------------
    @property
    def effort_metadata_dict(self):
        # type: () -> dict
        return SYS_CFG["effort_metadata_dict"]

    @property
    def camera_configuration_dict(self):
        # type: () -> dict
        return SYS_CFG["camera_cfgs"]

    @property
    def flight_number_str(self):
        # type: () -> str
        return str(SYS_CFG["flight_number_str"])

    @flight_number_str.setter
    def flight_number_str(self, val):
        SYS_CFG["flight_number_str"] = val

    @property
    def collecting(self):
        """Specifies whether data is currently being collected."""
        return self._collecting

    @collecting.setter
    def collecting(self, val):
        # type: (bool) -> None
        self._collecting = val

    def update_collect_colors(self, is_collecting=None, collect_in_region=None):
        # Grab directly from Redis since it could change via shapefile node
        is_archiving = int(kv.get("/sys/arch/is_archiving"))
        is_collecting = False if is_archiving == 0 else True
        collect_in_region = (
            False if SYS_CFG["arch"]["use_archive_region"] == 0 else True
        )
        if self.last_collecting is not None:
            if is_collecting != self.last_collecting:
                if is_collecting:
                    self.start_collecting()
                else:
                    self.stop_collecting()
        self.last_collecting = is_collecting
        self._collecting = False if is_collecting is None else is_collecting
        self._collect_in_region = (
            False if collect_in_region is None else collect_in_region
        )

        if is_collecting == True:
            self.recording_gauge.SetBackgroundColour((0, 255, 0))
            self.flight_data_panel.SetBackgroundColour(COLLECT_GREEN)

        elif is_collecting == False:
            self.recording_gauge.SetBackgroundColour((200, 200, 200))
            if collect_in_region is None or collect_in_region is False:
                self.flight_data_panel.SetBackgroundColour(APP_GRAY)
            else:
                self.flight_data_panel.SetBackgroundColour(SHAPE_COLLECT_BLUE)
        else:
            raise Exception("invalid value encountered for is_collecting")

    @property
    def collect_in_region(self):
        # type: () -> Optional[shapely.geometry.base.BaseGeometry]
        return self._collect_in_region

    # ------------------------------------------------------------------------

    def load_config_settings(self):
        # Build up the effort_combo_box.
        self.effort_combo_box.Unbind(wx.EVT_COMBOBOX)
        selection = SYS_CFG["arch"]["effort"]
        i = 0
        for effort_name in sorted(self.effort_metadata_dict.keys(), reverse=True):
            if effort_name is not None and effort_name != "null":
                try:
                    _ = self.effort_metadata_dict[effort_name]["project_name"]
                except (KeyError, TypeError):
                    continue
                self.effort_combo_box.Append(effort_name)
                if selection == effort_name:
                    self.effort_combo_box.SetSelection(i)
                i += 1

        # Build up sys config box
        self.effort_combo_box.Bind(wx.EVT_COMBOBOX, self.on_effort_selection)

        try:
            self.observer_text_ctrl.SetValue(SYS_CFG["observer"])
        except KeyError as e:
            print(e)
        self.flight_number_text_ctrl.Unbind(wx.EVT_TEXT)
        self.flight_number_text_ctrl.SetValue(self.flight_number_str)
        self.flight_number_text_ctrl.Bind(wx.EVT_TEXT, self.on_update_flight_number)
        self.update_collect_colors()
        print("Loaded config.")

    def update_static_text(self):
        try:
            center_det_frames = detector_state.health[self.hosts[0]]["frame"]
        except KeyError:
            center_det_frames = 0
        try:
            left_det_frames = detector_state.health[self.hosts[1]]["frame"]
        except KeyError:
            left_det_frames = 0
        try:
            right_det_frames = detector_state.health[self.hosts[2]]["frame"]
        except KeyError:
            right_det_frames = 0
        fmt = "Detector Frames: {}"
        self.left_sys_detector_frames.SetLabel(fmt.format(left_det_frames))
        self.center_sys_detector_frames.SetLabel(fmt.format(center_det_frames))
        self.right_sys_detector_frames.SetLabel(fmt.format(right_det_frames))

    def add_to_console_log(self, msg, msg_type="Info"):
        """Add message to the console log displayed in this GUI."""
        if self._log_panel:
            self._log_panel.add_message(msg_type, msg)

    def set_cam_attr(self, fov, chan, param, val):
        try:
            set_cam_attr(fov, chan, param, val, log_cb=self._add_to_event_log)
        except Exception as e:
            self.exception_window(e, "SetCamAttr Error")

    def set_ir_attr(self, fov, chan, param, val):
        try:
            set_ir_attr(fov, chan, param, val, log_cb=self._add_to_event_log)
        except Exception as e:
            rospy.logerr(e)  # , 'SetCamAttr Error')

    def start_collecting(self, event=None):
        self._disable_state_controls()
        self.log_state()

        effort_name = self.effort_combo_box.GetStringSelection()

        observer = self.observer_text_ctrl.GetValue()
        if observer != "":
            self.add_to_event_log("Observer: %s" % observer)
        collecting = True
        self.collecting = collecting
        if not SYS_CFG["arch"]["allow_ir_nuc"]:
            self.set_camera_parameter(self.hosts, "ir", "CorrectionAutoEnabled", 0)
        save_camera_config(self.get_sys_cfg())
        self.update_project_flight_params(collecting=collecting)
        self.add_to_event_log("Started collecting for effort: %s" % effort_name)

    def on_update_observer(self, event):
        SYS_CFG["observer"] = self.observer_text_ctrl.GetValue()

    def stop_collecting(self, event=None):
        self._enable_state_controls()
        effort_name = self.effort_combo_box.GetStringSelection()
        collecting = False
        self.collecting = collecting
        # Always make sure NUCing is turned on when not collecting
        self.set_camera_parameter(self.hosts, "ir", "CorrectionAutoEnabled", 1)
        self.update_project_flight_params(collecting=collecting)
        self.add_to_event_log("Stopped collecting for effort: %s" % effort_name)

    def exception_window(self, e, msg="An error occured"):
        dlg = wx.MessageDialog(self, "Error: {}".format(e), msg, wx.OK | wx.ICON_ERROR)
        dlg.ShowModal()
        dlg.Destroy()

    def get_sys_cfg(self):
        """For some silly reason, self.camera_config_combo.GetStringSelection() fails and
        just returns empty string periodically. No idea why. So we cache the past good value
        and hope that it's acceptable."""
        tmp_sys_cfg = self.camera_config_combo.GetStringSelection()
        if tmp_sys_cfg:
            self._sys_cfg = tmp_sys_cfg
            return tmp_sys_cfg

        try:
            curr_str = SYS_CFG["arch"]["sys_cfg"]
            if curr_str != "":
                self._sys_cfg = curr_str
                return self._sys_cfg
        except:
            pass

        # it fell through and somehow we still don't know what the string should be.
        # No choice but to use a sentinel value
        # print("{}: ERROR: sys_cfg is unset. This should rarely or never happen".format(time.time()))
        return "undefined_sys_cfg_delete_this"

    def get_project_flight(self):
        # type: () -> (str, str, str)
        effort_name = self.effort_combo_box.GetStringSelection()
        if effort_name == "":
            effort_name = "ON"
        sys_cfg = self.get_sys_cfg()
        project = SYS_CFG["effort_metadata_dict"][effort_name]["project_name"]
        flight = self.flight_number_str
        return project, flight, effort_name, sys_cfg

    def update_project_flight_params(self, collecting=None):
        project, flight, effort_name, sys_cfg = self.get_project_flight()
        redis_dict = {
            "project": project,
            "flight": flight,
            "effort": effort_name,
            "sys_cfg": sys_cfg,
            "base": SYS_CFG["arch"]["base"],
        }
        if collecting is not None:
            rospy.set_param("/sys/arch/is_archiving", int(collecting))
            redis_dict.update({"is_archiving": int(collecting)})
        sysdir = get_arch_path()
        SYS_CFG["syscfg_dir"] = sysdir
        SYS_CFG["arch"].update(redis_dict)
        return project, flight

    def on_effort_selection(self, event=None):
        """Called when an effort is selected."""
        effort_name = self.effort_combo_box.GetStringSelection()
        project_name = self.effort_metadata_dict[effort_name]["project_name"]
        self.add_to_event_log("Selected effort: %s" % effort_name)
        self.update_project_flight_params()

        if event is not None:
            event.Skip()

    def add_to_event_log(self, event_log_msg):
        """Sends message to event log."""
        # Only center will be in change of event logs
        self.update_project_flight_params()
        # sys_cfg_path = SYS_CFG["syscfg_dir"]
        # if not os.path.isdir(sys_cfg_path):
        #    os.makedirs(sys_cfg_path)
        self._add_to_event_log(*(event_log_msg, 0))
        self.add_to_console_log(event_log_msg, "add event log")

    def _add_to_event_log(self, note_str, sys_ind=0):
        """
        :param note_str: String to directly record in the event log (no

            symbol should be included).
        :type note_str: str

        :param sys_ind: Index indicating which system to send the message to.
        :type sys_ind:

        """
        if sys_ind in [0, -1]:
            srv = self._center_sys_event_log_srv
            system_name = "Center-View"
        elif sys_ind == 1:
            raise RuntimeError("Only centre view should receive log callback")
        elif sys_ind == 2:
            raise RuntimeError("Only centre view should receive log callback")
        else:
            raise Exception("Invalid 'sys_ind' %s" % str(sys_ind))

        effort_name = self.effort_combo_box.GetStringSelection()
        try:
            project = self.effort_metadata_dict[effort_name]["project_name"]
        except (KeyError, TypeError):
            project = "null"
        self.update_project_flight_params()
        sys_cfg_path = SYS_CFG["syscfg_dir"]
        try:
            # print(system_name, ':', note_str)
            project, flight, effort_name, sys_cfg = self.get_project_flight()
            collection_mode = SYS_CFG["collection_mode"]

            if flight == "":
                msg = "Must set the flight"
                dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return

            global G_time_note_started
            if isinstance(G_time_note_started, datetime.datetime):
                note_str += "| Time Note clicked: {}".format(
                    G_time_note_started.isoformat()[:22]
                )
                G_time_note_started = None  # unset so it doesn't conflict later

            resp = srv(
                project=project,
                flight=flight,
                effort=effort_name,
                collection_mode=collection_mode,
                note=note_str,
            )
            if not resp:
                print("No response.")
            self.add_to_console_log(note_str, msg_type="Info")
        except Exception as e:
            print(e)
            """
            msg = "'%s' system not responsive" % system_name
            if self.add_to_console_log:
                self.add_to_console_log(msg, msg_type='Error')
            dlg = wx.MessageDialog(self, msg, 'Error: {}'.format(e), wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            """

    # -------------------- Launch Post Proc Scripts --------------------------

    def on_detection_summary(self, event):
        dialog = wx.DirDialog(None, "Choose a Flight Directory", SYS_CFG["nas_mnt"])
        if dialog.ShowModal() == wx.ID_OK:
            flight_dir = dialog.GetPath()
        else:
            print("Invalid detection summary path selected, not setting.")
            return
        # If we've been given an effort directory, place all flight dirs in
        # the to process category
        ds = [d for d in self.search_dir(flight_dir)]
        if len(ds) == 0:
            icon = wx.ICON_ERROR
            dlg = wx.MessageDialog(
                self,
                "No Valid Flight Directories found in %s." % flight_dir,
                "Info",
                wx.OK | icon,
            )
            dlg.ShowModal()
            dlg.Destroy()
        else:
            for d in ds:
                self.postprocessing_q.put(["detections", d], timeout=0.1)

    def on_measure_image_to_image_homographies(self, event):
        dialog = wx.DirDialog(None, "Choose a Flight Directory", SYS_CFG["nas_mnt"])
        if dialog.ShowModal() == wx.ID_OK:
            flight_dir = dialog.GetPath()
        else:
            print("Invalid homography path selected, not setting.")
            return
        # If we've been given an effort directory, place all flight dirs in
        # the to process category
        ds = [d for d in self.search_dir(flight_dir)]
        if len(ds) == 0:
            icon = wx.ICON_ERROR
            dlg = wx.MessageDialog(
                self,
                "No Valid Flight Directories found in %s." % flight_dir,
                "Info",
                wx.OK | icon,
            )
            dlg.ShowModal()
            dlg.Destroy()
        else:
            for d in ds:
                self.postprocessing_q.put(["homography", d], timeout=0.1)

    def search_dir(self, d):
        # Recursively search through trying to find flight dirs
        for f in os.listdir(d):
            fd = os.path.join(d, f)
            if os.path.isdir(fd):
                if "_view" in f:
                    break
                if self.flight_re.match(f):
                    yield fd
                else:
                    self.search_dir(fd)
        f = os.path.basename(d)
        if self.flight_re.match(f):
            yield d

    def on_create_flight_summary(self, event):
        dialog = wx.DirDialog(None, "Choose a Flight Directory", SYS_CFG["nas_mnt"])
        if dialog.ShowModal() == wx.ID_OK:
            flight_dir = dialog.GetPath()
        else:
            print("Invalid flight summary path selected, not setting.")
            return
        # If we've been given an effort directory, place all flight dirs in
        # the to process category
        ds = [d for d in self.search_dir(flight_dir)]
        if len(ds) == 0:
            icon = wx.ICON_ERROR
            dlg = wx.MessageDialog(
                self,
                "No Valid Flight Directories found in %s." % flight_dir,
                "Info",
                wx.OK | icon,
            )
            dlg.ShowModal()
            dlg.Destroy()
        else:
            for d in ds:
                self.postprocessing_q.put(["flight_summary", d], timeout=0.1)

    def on_view_queue(self, event):
        jobs = self.postprocessing_q.queue
        heading = "Current Processing Queue"
        s = ""
        for job in jobs:
            s += "Command: %s || Directory: %s\n" % (self.c2s[job[0]], job[1])
        if len(s) == 0:
            s = "No post processing jobs are in queue.\n"
        wx.MessageBox(s, heading, parent=self)

    def on_clear_queue(self, event):
        l = self.postprocessing_q.qsize()
        with self.postprocessing_q.mutex:
            self.postprocessing_q.queue.clear()
        self.add_to_event_log("Cleared queue of %s jobs." % l)

    def on_cancel_running_jobs(self, event):
        is_busy = {h: 0 for h in self.hosts}
        current_task = {}
        for host in is_busy:
            for command in SYS_CFG["postproc_commands"]:
                try:
                    busy = int(kv.get("/postproc/%s/%s/busy" % (host, command)))
                except KeyError as e:
                    busy = 0
                if busy == 1:
                    current_task[host] = command
                    self.add_to_event_log(
                        "WARNING: Killed running job %s on %s." % (command, host)
                    )
                    self.system.run_command("postproc", host, "down", postproc=command)
                    # Killed process, no longer busy!
                    kv.put("/postproc/%s/%s/busy" % (host, command), 0)

    # ------------------------------------------------------------------------

    # -------------------- Launch Image Inspection Frame ---------------------
    def on_dclick_left_rgb(self, event):
        self.open_image_inspection_panel("Left RGB")

    def on_dclick_center_rgb(self, event):
        self.open_image_inspection_panel("Center RGB")

    def on_dclick_right_rgb(self, event):
        self.open_image_inspection_panel("Right RGB")

    def on_dclick_left_ir(self, event):
        self.open_image_inspection_panel("Left IR")

    def on_dclick_center_ir(self, event):
        self.open_image_inspection_panel("Center IR")

    def on_dclick_right_ir(self, event):
        self.open_image_inspection_panel("Right IR")

    def on_dclick_left_uv(self, event):
        self.open_image_inspection_panel("Left UV")

    def on_dclick_center_uv(self, event):
        self.open_image_inspection_panel("Center UV")

    def on_dclick_right_uv(self, event):
        self.open_image_inspection_panel("Right UV")

    # ------------------------------------------------------------------------

    def cb_raw_message_popup(self, msg):
        wx.CallAfter(self.raw_message_popup, msg)

    def raw_message_popup(self, msg):
        print("/rawmsg: \n{}".format(msg))
        return self.message_popup_throttle(msg.data, throttle=10)

    def message_popup_throttle(self, txt, throttle=None):
        now = datetime.datetime.now()
        if throttle is None:
            pass
        else:
            td_throttle = datetime.timedelta(seconds=throttle)
            dt = now - self.last_popup
            if dt < td_throttle:
                rospy.logwarn("suppressed, dt too short: {} : {}".format(dt, txt))
                return
        icon = wx.ICON_ERROR if "error" in txt.lower() else wx.ICON_INFORMATION
        dlg = wx.MessageDialog(self, txt, "Info", wx.OK | icon)
        dlg.ShowModal()
        dlg.Destroy()
        self.last_popup = now

    def ins_state_ros(self, msg):
        """
        :param msg: INS POSAVX message.
        :type msg: POSAVX

        """
        # Throttle to 10hz
        if msg.header.stamp.to_sec() - self.last_ins_time < 0.1:
            return
        self.last_ins_time = msg.header.stamp.to_sec()
        wx.CallAfter(self.ins_state, msg)

    def ins_state(self, msg):
        """
        :param msg: INS message.
        :type msg: POSAVX

        """
        if self._spoof_gps:
            spoofed = kv.get_dict("/spoof/ins")
            apply_ins_spoof(msg, spoofed)
        t = msg.time
        lat = msg.latitude
        lon = msg.longitude
        h = msg.altitude

        if np.abs(lat) > 1e-5:
            if self.lat0 is None:
                self.lat0 = lat
                self.lon0 = lon
                self.h0 = h

            self.lat_txtctrl.SetValue("{0:.8f}".format(lat))
            self.lon_txtctrl.SetValue("{0:.8f}".format(lon))
            self.alt_txtctrl.SetValue("{0:.3f}".format(h))

        # Location of geod (i.e., mean sea level, which is
        # generally the ground for us) relative to the
        # ellipsoid. Positive value means that mean sea level is
        # above the WGS84 ellipsoid.
        offset = geod.height(lat, lon)

        self.alt_msl_txtctrl.SetValue("{0:.3f}".format(h - offset))
        speed_si = msg.total_speed  # meters/second
        self.speed_si = speed_si
        speed_kts = speed_si * 1.94384  # knots
        self.speed_txtctrl.SetValue("{0:.3f}".format(speed_kts))

        # ENU quaternion
        heading = msg.heading
        pitch = msg.pitch
        roll = msg.roll

        self.heading_txtctrl.SetValue("{0:.3f}".format(heading))
        self.pitch_txtctrl.SetValue("{0:.3f}".format(pitch))
        self.roll_txtctrl.SetValue("{0:.3f}".format(roll))

        t = str(datetime.datetime.utcfromtimestamp(np.round(t * 100) / 100))
        t = t[:22]
        # Hack so string length stays the same on a zero
        if len(t) == 19:
            t += ".00"
        self.ins_time_txtctrl.SetValue(t)

        if msg.align_status == 0:
            self.ins_status_flag_txtctrl.SetValue("gps only")
        elif msg.align_status == 1:
            self.ins_status_flag_txtctrl.SetValue("coarse leveling")
        elif msg.align_status == 2:
            self.ins_status_flag_txtctrl.SetValue("degraded")
        elif msg.align_status == 3:
            self.ins_status_flag_txtctrl.SetValue("aligned")
        elif msg.align_status == 4:
            self.ins_status_flag_txtctrl.SetValue("full nav")
        else:
            self.ins_status_flag_txtctrl.SetValue("")

        if self._spoof_gps:
            self.ins_control_panel.SetBackgroundColour(WARN_AMBER)
            self.gnss_status_flag_txtctrl.SetValue("gps spoofed!")
        self._spoof_events = int(kv.get("/debug/spoof_events", 0))
        if self._spoof_events == 1:
            self.ins_control_panel.SetBackgroundColour(WARN_AMBER)
            self.gnss_status_flag_txtctrl.SetValue("no fix! event spoof!")
            return
        else:
            self.ins_control_panel.SetBackgroundColour(APP_GRAY)
        if msg.gnss_status == 0:
            self.gnss_status_flag_txtctrl.SetValue("No Fix")
        elif msg.gnss_status == 1:
            self.gnss_status_flag_txtctrl.SetValue("SPS Mode")
        elif msg.gnss_status == 2:
            self.gnss_status_flag_txtctrl.SetValue("Differential")
        elif msg.gnss_status == 3:
            self.gnss_status_flag_txtctrl.SetValue("PPS Mode")
        elif msg.gnss_status == 4:
            self.gnss_status_flag_txtctrl.SetValue("Fixed RTK")
        elif msg.gnss_status == 5:
            self.gnss_status_flag_txtctrl.SetValue("Float RTK")
        elif msg.gnss_status == 6:
            self.gnss_status_flag_txtctrl.SetValue("Dead Reckon")
        else:
            self.gnss_status_flag_txtctrl.SetValue("")

    def on_update_flight_number(self, event=None):
        flight_number_str = self.flight_number_text_ctrl.GetValue()
        try:
            test = int(flight_number_str)
        except:
            self.flight_number_text_ctrl.SetValue(self.flight_number_str)
            msg = "Flight number must be an integer, e.g. 01, 02, 001, etc."
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        self.flight_number_str = flight_number_str
        self.add_to_console_log(self.flight_number_str, "on_update_flight_number")

    # ---------------------------- Show/Hide Images --------------------------
    def on_show_or_hide_left(self, event):
        """Toggle visilibity of left column of images."""
        SYS_CFG["show_left"] = not SYS_CFG["show_left"]
        self.update_show_hide()

    def on_show_or_hide_center(self, event):
        """Toggle visilibity of center column of images."""
        SYS_CFG["show_center"] = not SYS_CFG["show_center"]
        self.update_show_hide()

    def on_show_or_hide_right(self, event):
        """Toggle visilibity of right column of images."""
        SYS_CFG["show_right"] = not SYS_CFG["show_right"]
        self.update_show_hide()

    def on_show_or_hide_rgb(self, event):
        """Toggle visilibity of RGB row of images."""
        SYS_CFG["show_rgb"] = not SYS_CFG["show_rgb"]
        self.update_show_hide()

    def on_show_or_hide_ir(self, event):
        """Toggle visilibity of ir row of images."""
        SYS_CFG["show_ir"] = not SYS_CFG["show_ir"]
        self.update_show_hide()

    def on_show_or_hide_uv(self, event):
        """Toggle visilibity of uv row of images."""
        SYS_CFG["show_uv"] = not SYS_CFG["show_uv"]
        self.update_show_hide()

    def on_toggle_saturated_pixels(self, event):
        """Toggle visilibity of uv row of images."""
        SYS_CFG["show_saturated_pixels"] = not SYS_CFG["show_saturated_pixels"]

    def update_show_hide(self):
        """Update which image panes are to be shown or hidden."""
        # RGB
        if SYS_CFG["show_left"] and SYS_CFG["show_rgb"]:
            self.m_panel_left_rgb.Show()
        else:
            self.m_panel_left_rgb.Hide()

        if SYS_CFG["show_center"] and SYS_CFG["show_rgb"]:
            self.m_panel_center_rgb.Show()
        else:
            self.m_panel_center_rgb.Hide()

        if SYS_CFG["show_right"] and SYS_CFG["show_rgb"]:
            self.m_panel_right_rgb.Show()
        else:
            self.m_panel_right_rgb.Hide()

        # IR
        if SYS_CFG["show_left"] and SYS_CFG["show_ir"]:
            self.m_panel_left_ir.Show()
        else:
            self.m_panel_left_ir.Hide()

        if SYS_CFG["show_center"] and SYS_CFG["show_ir"]:
            self.m_panel_center_ir.Show()
        else:
            self.m_panel_center_ir.Hide()

        if SYS_CFG["show_right"] and SYS_CFG["show_ir"]:
            self.m_panel_right_ir.Show()
        else:
            self.m_panel_right_ir.Hide()

        # UV
        if SYS_CFG["show_left"] and SYS_CFG["show_uv"]:
            self.m_panel_left_uv.Show()
        else:
            self.m_panel_left_uv.Hide()

        if SYS_CFG["show_center"] and SYS_CFG["show_uv"]:
            self.m_panel_center_uv.Show()
        else:
            self.m_panel_center_uv.Hide()

        if SYS_CFG["show_right"] and SYS_CFG["show_uv"]:
            self.m_panel_right_uv.Show()
        else:
            self.m_panel_right_uv.Hide()

        if SYS_CFG["show_center"]:
            self.sys0_detector_frame1.Show()
            self.sys0_disk_usage_panel.Show()
        else:
            self.sys0_detector_frame1.Hide()
            self.sys0_disk_usage_panel.Hide()
        if SYS_CFG["show_left"]:
            self.sys1_detector_frames.Show()
            self.sys1_disk_usage_panel.Show()
        else:
            self.sys1_detector_frames.Hide()
            self.sys1_disk_usage_panel.Hide()
        if SYS_CFG["show_right"]:
            self.sys2_detector_frames.Show()
            self.sys2_disk_usage_panel.Show()
        else:
            self.sys2_detector_frames.Hide()
            self.sys2_disk_usage_panel.Hide()
        self.Layout()

    # ------------------------------------------------------------------------
    # ---------------------------- Detection Menu ----------------------------

    def do_start_a_detector(self, event=None, host="unset", log=True):
        cmdpipef = SYS_CFG[host]["detector"]["pipefile"]
        cmdpipef = SYS_CFG[host]["detector"]["pipefile"]
        if not os.path.exists(cmdpipef):
            msg = (
                "Pipefile for detector %s does not exist, not starting detector." % host
            )
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        set_detector_state(self.system, host, EPodStatus.Running)
        if log:
            self.add_to_event_log("command sent: start detector {}".format(host))

    def do_stop_a_detector(self, event=None, host="unset", log=True):
        set_detector_state(self.system, host, EPodStatus.Off)
        if log:
            self.add_to_event_log("command sent: stop detector {}".format(host))

    def on_start_detectors(self, event=None):
        self.add_to_event_log("command sent: start all detectors")
        for host in self.hosts:
            self.do_start_a_detector(host=host, log=False)

    def on_start_detector_sys0(self, event=None):
        self.do_start_a_detector(host=self.hosts[0])

    def on_start_detector_sys1(self, event=None):
        self.do_start_a_detector(host=self.hosts[1])

    def on_start_detector_sys2(self, event=None):
        self.do_start_a_detector(host=self.hosts[2])

    def on_stop_detectors(self, event=None):
        self.add_to_event_log("command sent: stop all detectors")
        for host in self.hosts:
            self.do_stop_a_detector(host=host, log=False)

    def on_stop_detector_sys0(self, event=None):
        self.do_stop_a_detector(host=self.hosts[0])

    def on_stop_detector_sys1(self, event=None):
        self.do_stop_a_detector(host=self.hosts[1])

    def on_stop_detector_sys2(self, event=None):
        self.do_stop_a_detector(host=self.hosts[2])

    # ------------------------- Methods for Hot Keys -------------------------
    def set_focus_to_exposure(self, event=None):
        self.exposure_value_txt_ctrl.SetFocus()

    def reverse_collecting_state(self, event=None):
        """If collecting, stop. If not collecting, start collecting."""
        if self.collecting:
            self.stop_collecting()
        else:
            self.start_collecting()

    # ------------------------------------------------------------------------

    def load_camera_model(self, name, fname):
        wildcard = "Camera model (*.yaml)"
        dialog = wx.FileDialog(
            None, "Choose a file", os.getcwd(), "", wildcard, wx.OPEN
        )
        if dialog.ShowModal() == wx.ID_OK:
            print(dialog.GetPath())

    # ------------------------------------------------------------------------

    def save_flight_summary(self, event=None):
        wildcard = "Camera model (*.yaml)"
        dialog = wx.FileDialog(None, "Choose a file", "/mnt", "", wildcard, wx.SAVE)
        if dialog.ShowModal() == wx.ID_OK:
            print(dialog.GetPath())

    def format_external_drive_N(self, event=None, drive=0):
        dlg = wx.MessageDialog(
            None,
            "Are you sure you want to format the "
            "external \nsys{} data drive?".format(drive),
            "Delete",
            wx.YES_NO | wx.ICON_QUESTION,
        )
        result = dlg.ShowModal()
        if result != wx.ID_YES:
            return
        dlg = wx.MessageDialog(
            None,
            "This ABSOLUTELY CANNOT BE UNDONE. Are you still sure"
            "you want to erase \nsys{} data drive?".format(drive),
            "Delete".format(drive),
            wx.YES_NO | wx.ICON_QUESTION,
        )
        result = dlg.ShowModal()
        if result != wx.ID_YES:
            return

    def on_format_external_drive_sys0(self, event=None):
        self.format_external_drive_N(drive=0)

    def on_format_external_drive_sys1(self, event=None):
        self.format_external_drive_N(drive=1)

    def on_format_external_drive_sys2(self, event=None):
        self.format_external_drive_N(drive=2)

    # ----------------------------- Open New Panels -------------------------
    def on_add_to_event_log(self, event):
        """Called when 'Add Note to Log' button is pressed."""
        global G_time_note_started
        G_time_note_started = (
            datetime.datetime.now()
        )  # this ONLY gets set here. logs when "Add Note" clicked
        if not self._event_log_note_frame:
            self._event_log_note_frame = EventLogNoteFrame(self)

        self._event_log_note_frame.Raise()

    def open_image_inspection_panel(self, stream):
        if not self._image_inspection_frame:
            print("Opening image inspection")
            self._image_inspection_frame = ImageInspectionFrame(
                self, self.topic_names, stream, compressed=False
            )

        self._image_inspection_frame.Raise()

    def on_system_startup_frame_raise(self, event=None):
        """Show list of hot keys."""
        if not self._system_startup_frame:
            self._system_startup_frame = SystemStartup(self, self.system)

        self._system_startup_frame.Raise()

    def on_open_log_panel(self, event=None):
        if not self._log_panel:
            self._log_panel = LogFrame(self)

        self._log_panel.Raise()

    def on_set_collection_mode(self, event=None):
        """Triggered when 'Set Collection Mode' button is pressed."""
        pull_gui_state()
        if not self._collection_mode_frame:
            self._collection_mode_frame = CollectionModeFrame(
                self,
                SYS_CFG["collection_mode"],
                SYS_CFG["shapefile_fname"],
                SYS_CFG["arch"]["use_archive_region"],
                SYS_CFG["arch"]["allow_ir_nuc"],
                SYS_CFG["arch"]["trigger_freq"],
                SYS_CFG["arch"]["overlap_percent"],
            )
        self._collection_mode_frame.Raise()

    def metadata_entry_update(self, event=None, change="updated"):
        self.add_to_event_log("Effort metadata {}. ".format(change))

    def next_effort_config(self, event=None):
        ind = self.effort_combo_box.GetSelection()
        if ind == wx.NOT_FOUND:
            ind = 0
        else:
            ind = ind + 1

        if ind == self.effort_combo_box.GetCount():
            ind = 0
        self.effort_combo_box.SetSelection(ind)
        self.on_effort_selection()

    def previous_effort_config(self, event=None):
        ind = self.effort_combo_box.GetSelection()
        if ind == wx.NOT_FOUND:
            ind = 0
        else:
            ind = ind - 1

        if ind < 0:
            ind = self.effort_combo_box.GetCount() - 1
        self.effort_combo_box.SetSelection(ind)
        self.on_effort_selection()

    def on_new_effort_metadata_entry(self, event=None):
        # Set up the metadata_entry frame.
        if not self._metadata_entry_frame:
            self._metadata_entry_frame = MetadataEntryFrame(
                self, self.effort_metadata_dict, self.effort_combo_box
            )
        self.add_to_console_log("created metadata entry", "metadata")
        self._metadata_entry_frame.Raise()
        self.metadata_entry_update(change="created")

    def on_edit_effort_metadata(self, event=None):
        if self.effort_combo_box.GetCount() == 0:
            return

        effort_name = self.effort_combo_box.GetStringSelection()

        # Set up the metadata_entry frame.
        if not self._metadata_entry_frame:
            self._metadata_entry_frame = MetadataEntryFrame(
                self, self.effort_metadata_dict, self.effort_combo_box, effort_name
            )

        self._metadata_entry_frame.Raise()
        self.metadata_entry_update(change="edited")

    def on_delete_effort_metadata(self, event=None):
        if self.effort_combo_box.GetCount() == 0:
            return

        if self.effort_combo_box.GetCount() == 1:
            msg = "Cannot have zero entries! Please create another effort before deleting this effort."
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return

        dlg = wx.MessageDialog(
            None,
            "Are you sure you want to delete?",
            "Delete",
            wx.YES_NO | wx.ICON_QUESTION,
        )
        result = dlg.ShowModal()
        if result != wx.ID_YES:
            return

        self.metadata_entry_update(change="deleted")
        effort_name = self.effort_combo_box.GetStringSelection()
        ind = self.effort_combo_box.GetSelection()

        del self.effort_metadata_dict[effort_name]

        if self.effort_combo_box.GetCount() == 1:
            return
        else:
            if ind > 0:
                self.effort_combo_box.SetSelection(ind - 1)
            else:
                self.effort_combo_box.SetSelection(ind + 1)

            self.effort_combo_box.Delete(ind)
        self.on_effort_selection()

    # ------------------------- Camera Configuration -------------------------
    def on_edit_camera_configuration(self, event=None):
        # Set up the metadata_entry frame.
        if not self._camera_config_frame:
            self._camera_config_frame = CameraConfiguration(
                self, SYS_CFG["arch"]["sys_cfg"]
            )

        self._camera_config_frame.Raise()

    def set_camera_config_dict(self, config_dict=None):
        curr_str = self.get_sys_cfg()
        SYS_CFG["arch"]["sys_cfg"] = curr_str

        self.camera_config_combo.SetEditable(True)
        self.camera_config_combo.Clear()

        ind = None
        n = 0
        for camera_config_combo in sorted(
            self.camera_configuration_dict.keys(), reverse=True
        ):
            if camera_config_combo == curr_str:
                ind = n
            self.camera_config_combo.Append(camera_config_combo)
            n += 1

        self.camera_config_combo.SetEditable(False)

        if ind is not None:
            self.camera_config_combo.SetSelection(ind)
        else:
            self.camera_config_combo.SetSelection(0)
            SYS_CFG["arch"]["sys_cfg"] = self.camera_config_combo.GetStringSelection()

    def next_camera_config(self, event=None):
        ind = self.camera_config_combo.GetSelection()
        if ind == wx.NOT_FOUND:
            ind = 0
        else:
            ind = ind + 1

        if ind == self.camera_config_combo.GetCount():
            ind = 0

        self.camera_config_combo.SetSelection(ind)

    def previous_camera_config(self, event=None):
        ind = self.camera_config_combo.GetSelection()
        if ind == wx.NOT_FOUND:
            ind = 0
        else:
            ind = ind - 1

        if ind < 0:
            ind = self.camera_config_combo.GetCount() - 1

        self.camera_config_combo.SetSelection(ind)

    def on_camera_config_combo(self, event=None):
        """ """
        # Add log of this event to event log.
        curr_str = self.get_sys_cfg()
        self.add_to_event_log("system_config: %s" % curr_str)
        cc = SYS_CFG["camera_cfgs"][curr_str]
        syscfg_dir = SYS_CFG["syscfg_dir"]
        vfov = load_from_file(cc["center_rgb_yaml_path"]).fov()[1]
        SYS_CFG["rgb_vfov"] = vfov
        save_camera_config(curr_str)
        self.update_project_flight_params()

        if cc["center_sys_pipe"] != "null":
            SYS_CFG[self.hosts[0]]["detector"]["pipefile"] = cc["center_sys_pipe"]
        if cc["left_sys_pipe"] != "null":
            SYS_CFG[self.hosts[1]]["detector"]["pipefile"] = cc["left_sys_pipe"]
        if cc["right_sys_pipe"] != "null":
            SYS_CFG[self.hosts[2]]["detector"]["pipefile"] = cc["right_sys_pipe"]

    # ------------------------- END Camera Configuration -------------------------

    def on_ir_nuc(self, event=None):
        """Request that the IR cameras execute NUC."""
        for host in self.hosts:
            topic = os.path.join("/", host, "ir", "nuc")
            service = rospy.ServiceProxy(topic, CamSetAttr, persistent=False)
            # These values aren't used currently, just an overload to trigger nuc
            msg = "/{}/{}/{}:={}".format(host, "ir", "nuc", "manual")
            try:
                resp = service.call(name="nuc", value="manual")
            except rospy.service.ServiceException:
                errmsg = "Attempted to set `{}`, but system did not respond".format(msg)
                rospy.logerr(errmsg)
                continue
            if not resp:
                errmsg = "Attempted to set `{}`, but it failed".format(msg)
                rospy.logerr(errmsg)
                continue
            rospy.loginfo(msg)
            self.add_to_event_log(
                "command sent: Request to manual NUC cameras {}. ".format(msg)
            )

    def on_hot_key_help(self, event=None):
        """Show list of hot keys."""
        if not self._hot_key_list:
            self._hot_key_list = HotKeyList(self)

        self._hot_key_list.Raise()

    def on_menu_item_about(self, event):
        about_panel = wx.Panel(self, wx.ID_ANY)
        info = wx.AboutDialogInfo()
        info.Name = "KAMERA System Control Panel"
        info.Version = "2.0.0"
        info.Copyright = "(C) 2023 Kitware"
        info.Description = wordwrap(
            "This GUI allows control of the KAMERA system.",
            350,
            wx.ClientDC(self.ins_control_panel),
        )
        info.WebSite = ("http://www.kitware.com", "Kitware")
        info.Developers = ["Matt Brown, Adam Romlein, Mike McDermott"]
        info.License = wordwrap(LICENSE_STR, 500, wx.ClientDC(about_panel))
        # Show the wx.AboutBox
        wx.AboutBox(info)

    # ------------------------------------------------------------------------

    def _enable_state_controls(self):
        for child in self.flight_data_panel.GetChildren():
            child.Enable()
        self.camera_config_combo.Enable()
        self.close_button.Enable()

    def _disable_state_controls(self):
        for child in self.flight_data_panel.GetChildren():
            if child.GetLabel() in [
                "Set Collection Mode",
                "Add Note to Log",
                "Data Collection",
                "Stop Collecting",
                "Effort",
                "Start Detectors",
                "Stop Detectors",
            ]:
                continue
            if isinstance(child, wx.ComboBox):  # Effort dropdown element
                continue
            child.Disable()
        self.camera_config_combo.Disable()
        # self.close_button.Disable()

    # ------------------------------------------------------------------------

    def on_close_button(self, event=None):
        self.Close()
        if event is not None:
            event.Skip()

    def when_closed(self, event=None):
        # self.when_closed.Unbind(wx.EVT_CLOSE)
        # self.on_resize.Unbind(wx.EVT_SIZE)
        save_config_settings()
        self.timer.Unbind(wx.EVT_TIMER)
        self.ins_state_sub.unregister()
        try:
            self._metadata_entry_frame.Close()
        except:
            pass

        try:
            self._image_inspection_frame.Close()
        except:
            pass

        event.Skip()


# === === === === === === end MainFrame === === === === === === ===
