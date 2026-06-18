#!/usr/bin/env python3
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
import queue
import requests

# import redis

# GUI imports
import wx
import wx.adv
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
from custom_msgs.msg import GSOF_INS
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

# from roskv.impl.nop_kv import NopKV as ImplKV
# from roskv.impl.rosparam_kv import RosParamKV as ImplKV
from roskv.impl.redis_envoy import RedisEnvoy as ImplEnvoy
from roskv.rendezvous import ConditionalRendezvous  # , Governor
from roskv.util import simple_hash_args, MyTimeoutError, filter_hosts_by_system

# Kamera imports
from custom_msgs.srv import (
    AddToEventLog,
    RequestCompressedImageView,
    RequestImageMetadata,
    RequestImageView,
    CamSetAttr,
)
from wxpython_gui.camera_models import load_from_file

# Sibling modules within the system_control_panel package
import wxpython_gui.system_control_panel.form_builder_output as form_builder_output
import wxpython_gui.system_control_panel.form_builder_output_effort_metadata as form_builder_output_effort_metadata
import wxpython_gui.system_control_panel.form_builder_output_imagery_inspection as form_builder_output_imagery_inspection
import wxpython_gui.system_control_panel.form_builder_output_event_log_note as form_builder_output_event_log_note
import wxpython_gui.system_control_panel.form_builder_output_hot_key_list as form_builder_output_hot_key_list
import wxpython_gui.system_control_panel.form_builder_output_collection_mode as form_builder_output_collection_mode
import wxpython_gui.system_control_panel.form_builder_output_log_panel as form_builder_output_log_panel
import wxpython_gui.system_control_panel.form_builder_output_system_startup as form_builder_output_system_startup
import wxpython_gui.system_control_panel.form_builder_output_camera_configuration as form_builder_output_camera_configuration

# Absolute imports
from wxpython_gui.cfg import (
    SYS_CFG,
    APP_GRAY,
    BRIGHT_RED,
    COLLECT_GREEN,
    COLLECT_ALERT_RED,
    ERROR_RED,
    FLAT_GRAY,
    DISABLED_GRAY,
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
    get_detector_pipefile,
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

import wxpython_gui.system_control_panel.gui_utils as gui_utils


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
P1_SHUTTER_MODE_PARAM = "Shutter_Mode"
P1_APERTURE_MIN_PARAM = "Aperture_Min"
P1_APERTURE_MAX_PARAM = "Aperture_Max"
P1_EXPOSURE_COMP_PARAM = "Exposure_Comp."
P1_SHUTTER_MODE_LS = "1"
P1_SHUTTER_MODE_ES = "2"
# Shutter Mode (LS/ES) label + dropdown hidden for now; exposure bias stays visible.
_SHOW_RGB_SHUTTER_MODE_CONTROLS = False
P1_LS_SHUTTER_DENOMS = [16000, 13000, 10000, 8000, 6500, 5000]
P1_ES_SHUTTER_DENOMS = [
    4000,
    3200,
    2500,
    2000,
    1600,
    1250,
    1000,
    800,
    640,
    500,
    400,
    320,
    250,
    200,
    160,
    125,
    100,
    80,
    60,
    50,
    40,
    30,
    25,
    20,
    15,
    13,
    10,
]
P1_ISO_STOPS = [
    200,
    250,
    320,
    400,
    500,
    640,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3200,
    4000,
    5000,
    6400,
]
P1_APERTURE_STOPS = [
    4.0,
    4.5,
    5.0,
    5.6,
    6.3,
    7.1,
    8.0,
    9.0,
    10,
    11,
    12,
    14,
    16,
    18,
    20,
    22,
]
P1_GAIN_MIN = 0
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
        # Recompute enlarged title fonts so they are not clipped under Phoenix.
        wx.CallAfter(gui_utils.unclip_static_text, self)

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

        all_hosts = sorted(SYS_CFG["arch"]["hosts"].keys())
        self.hosts = filter_hosts_by_system(all_hosts)
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
        self.effort_combo_box.SetBackgroundColour(wx.WHITE)
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
        self._collect_health_ssd_ok = True
        self._collect_health_nas_ok = True
        self._collect_health_cameras_ok = True
        self._collect_stream_stale_sec = 10.0
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
        self._setup_rgb_controls()
        self._set_camera_param_btn_default_bg = self.m_button10.GetBackgroundColour()
        self._set_camera_param_btn_default_fg = self.m_button10.GetForegroundColour()
        self._bind_camera_param_dropdown_handlers()
        self.camera_setting_rgb_uv_combo.Bind(wx.EVT_COMBOBOX, self.on_modal_selection)
        self.rgb_shutter_mode_combo.Bind(wx.EVT_COMBOBOX, self.on_rgb_shutter_mode)
        # Call once to hide/show proper options
        self.on_modal_selection(None)

        self.camera_setting_subsys.Bind(
            wx.EVT_COMBOBOX, self.on_camera_setting_subsys_selection
        )
        self.on_camera_setting_subsys_selection(None)

        # ----------------------------- Hot Keys -----------------------------
        entries = []
        # Retain the id refs so their reserved ids aren't recycled (NewIdRef
        # releases the id once the ref is garbage collected).
        self._accel_ids = []

        # Bind ctrl+s to start/stop recording.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.reverse_collecting_state, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("S"), random_id)
        # Bind ctrl+d to start detectors.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.on_start_detectors, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("D"), random_id)

        # Bind ctrl+f to stop detectors.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.on_stop_detectors, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("F"), random_id)
        # Bind ctrl+h to hot key menu.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.on_hot_key_help, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("H"), random_id)

        # Bind ctrl+e to set context to exposure entry.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.set_focus_to_exposure, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("E"), random_id)

        # Bind ctrl+n to add note to log.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.on_add_to_event_log, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("N"), random_id)

        # Bind ctrl+o to next previous camera configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.previous_camera_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("O"), random_id)

        # Bind ctrl+p to next camera configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.next_camera_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("P"), random_id)

        # Bind ctrl+i to next previous effort configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.previous_effort_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("I"), random_id)

        # Bind ctrl+k to next effort configuration.
        entries.append(wx.AcceleratorEntry())
        random_id = wx.NewIdRef()
        self._accel_ids.append(random_id)
        self.Bind(wx.EVT_MENU, self.next_effort_config, id=random_id)
        entries[-1].Set(wx.ACCEL_CTRL, ord("K"), random_id)

        accel = wx.AcceleratorTable(entries)
        self.SetAcceleratorTable(accel)

        # --------------------------------------------------------------------

        # rospy.add_client_shutdown_hook(self.on_close_button)
        self.system = SystemCommands(self.hosts)

        self.Bind(wx.EVT_CLOSE, self.when_closed)
        self._did_initial_unclip = False
        self.Bind(wx.EVT_SIZE, self.on_resize)

        # Distinct ids so the two EVT_TIMER bindings don't collide; keep the
        # refs alive so the ids aren't recycled.
        self._fast_timer_id = wx.NewIdRef()
        self._slow_timer_id = wx.NewIdRef()
        self.timer = wx.Timer(self, id=self._fast_timer_id)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        FAST_TIMER_MS = 200
        self.timer.Start(FAST_TIMER_MS)

        # So that we can check that the node is still alive.
        self.slow_timer = wx.Timer(self, id=self._slow_timer_id)
        self.Bind(wx.EVT_TIMER, self.on_slow_timer, self.slow_timer)
        self.slow_timer.Start(1000)
        # TODO
        # self.lastLdet = kv.get("/nuvo1/detector/health//frame")
        # self.lastCdet = kv.get("/nuvo0/detector/health//frame")
        # self.lastRdet = kv.get("/nuvo2/detector/health//frame")
        self.delay = 0
        self.detectors_gauge.SetBackgroundColour(FLAT_GRAY)
        self.Show()
        print(self.GetSize())
        self._fit_to_display()
        # self.detector_gauge.Hide()

        if self.collecting:
            self._disable_state_controls()
        # Add in pause before displaying imagery

    def _fit_to_display(self):
        """Keep the window within the usable area of its display.

        On smaller monitors (e.g. 1920x1080) the design size is taller than the
        work area, which pushes the bottom of the side panel (detector /
        archiving controls, Close button) off-screen. Clamp the window size and
        position to the display's client area, and cap the minimum size so the
        window can always be shrunk to fit.
        """
        idx = wx.Display.GetFromWindow(self)
        if idx == wx.NOT_FOUND:
            idx = 0
        area = wx.Display(idx).GetClientArea()

        min_w = min(1500, area.width)
        min_h = min(980, area.height)
        self.SetMinSize((min_w, min_h))

        w, h = self.GetSize()
        self.SetSize(min(w, area.width), min(h, area.height))
        self.SetPosition((area.x, area.y))
        self.Layout()

    def system_sanity_check(self):
        # Run the disk / NAS checks off the UI thread so unreachable hosts
        # can't freeze the GUI. Skip if the previous check is still in flight.
        if getattr(self, "_sanity_check_running", False):
            return
        self._sanity_check_running = True
        worker = threading.Thread(
            target=self._system_sanity_worker, args=(list(self.hosts),)
        )
        worker.daemon = True
        worker.start()

    def _system_sanity_worker(self, hosts):
        """Background-thread network I/O for system_sanity_check.

        Collects results and marshals UI updates back onto the main thread via
        wx.CallAfter. Every request has a timeout so an offline host can't hang
        the check.
        """
        timeout = 0.5
        ssd_mnt = SYS_CFG["local_ssd_mnt"]
        nas_mnt = SYS_CFG["nas_mnt"]
        try:
            results = {}
            unmounts = []
            nas_unmounts = []
            for host in hosts:
                entry = {"ssd_gb": None, "ssd_err": False, "nas_gb": None}
                try:
                    resp = requests.post(
                        "http://{}:8987/diskinfo".format(host),
                        data=ssd_mnt,
                        timeout=timeout,
                    )
                    info = json.loads(resp.text)
                    if info["ismount"]:
                        entry["ssd_gb"] = float(info["bytes_free"]) / 1e9
                    else:
                        entry["ssd_err"] = True
                        unmounts.append(host)
                except (requests.exceptions.RequestException, ValueError, KeyError):
                    rospy.logwarn(
                        "Could not access disk info from system %s." % host
                    )
                    results[host] = entry
                    continue
                try:
                    resp = requests.post(
                        "http://{}:8987/diskinfo".format(host),
                        data=nas_mnt,
                        timeout=timeout,
                    )
                    info = json.loads(resp.text)
                    if info["ismount"]:
                        entry["nas_gb"] = float(info["bytes_free"]) / 1e9
                    else:
                        nas_unmounts.append(host)
                except (requests.exceptions.RequestException, ValueError, KeyError):
                    pass
                results[host] = entry

            # Attempt to remount any host that responded but wasn't mounted.
            if unmounts:
                rospy.logerr(
                    "ERROR: One or more hosts has an ssd mount issue: {}".format(
                        unmounts
                    )
                )
                for host in unmounts:
                    try:
                        requests.post(
                            "http://{}:8987/mountall".format(host),
                            data="/mnt/data",
                            timeout=timeout,
                        )
                    except requests.exceptions.RequestException:
                        pass
            if nas_unmounts:
                rospy.logerr(
                    "ERROR: One or more hosts has a NAS mount issue: {}".format(
                        nas_unmounts
                    )
                )
                for host in nas_unmounts:
                    try:
                        requests.post(
                            "http://{}:8987/mountall".format(host),
                            data=nas_mnt,
                            timeout=timeout,
                        )
                    except requests.exceptions.RequestException:
                        pass

            wx.CallAfter(self._apply_system_sanity, results, nas_unmounts)
        finally:
            self._sanity_check_running = False

    def _apply_system_sanity(self, results, nas_unmounts):
        """Apply system_sanity_check results to the UI (main thread only)."""
        ssd_ok = True
        for host in self.hosts:
            entry = results.get(host, {})
            if entry.get("ssd_err") or entry.get("ssd_gb") is None:
                ssd_ok = False
                break
        center_entry = results.get(self.hosts[0], {})
        nas_ok = not nas_unmounts and center_entry.get("nas_gb") is not None
        self._collect_health_ssd_ok = ssd_ok
        self._collect_health_nas_ok = nas_ok

        for host, entry in results.items():
            try:
                fov = SYS_CFG["arch"]["hosts"][host]["fov"]
            except KeyError:
                continue
            ssd_label = getattr(self, "{}_sys_space_static_text".format(fov), None)
            if ssd_label is not None:
                if entry["ssd_err"]:
                    ssd_label.SetLabel("Disk Space: Err")
                    ssd_label.SetForegroundColour(ERROR_RED)
                elif entry["ssd_gb"] is not None:
                    ssd_label.SetLabel("Disk Space: %0.2f GB" % entry["ssd_gb"])
                    ssd_label.SetForegroundColour(COLLECT_GREEN)
                gui_utils.refit_label(ssd_label)
            # The NAS is shared, so it's only displayed once (center host).
            if fov == "center" and entry["nas_gb"] is not None:
                self.nas_disk_space.SetLabel(
                    "NAS Space: %0.2f GB" % entry["nas_gb"]
                )
                self.nas_disk_space.SetForegroundColour(COLLECT_GREEN)
                gui_utils.refit_label(self.nas_disk_space)
        if nas_unmounts:
            self.nas_disk_space.SetLabel("NAS Err: " + "".join(nas_unmounts))
            self.nas_disk_space.SetForegroundColour(ERROR_RED)
            gui_utils.refit_label(self.nas_disk_space)
        self.update_collect_colors()

    def _ins_stream_healthy(self):
        if self.last_ins_time <= 0:
            return False
        return time.time() - self.last_ins_time <= self._collect_stream_stale_sec

    def _collecting_critical_health_ok(self):
        if not self._collect_health_ssd_ok:
            return False
        if not self._collect_health_nas_ok:
            return False
        if not self._collect_health_cameras_ok:
            return False
        return True

    def _flight_data_header_widgets(self):
        return (
            self.m_staticText14211,
            self.m_staticText33,
            self.m_staticText34,
            self.flight_number_text_ctrl,
            self.m_staticText331,
            self.observer_text_ctrl,
            self.m_staticText18171311,
            self.effort_combo_box,
        )

    def _collecting_panel_colour(self):
        if not self._collecting_critical_health_ok():
            return wx.Colour(*COLLECT_ALERT_RED)
        if self._spoof_events == 1 or not self._ins_stream_healthy():
            return self._spoof_events_colour()
        return wx.Colour(*COLLECT_GREEN)

    def _apply_flight_data_header_background(self, colour):
        for widget in self._flight_data_header_widgets():
            if widget is self.effort_combo_box:
                continue
            widget.SetBackgroundColour(colour)
            widget.Refresh()
        self.effort_combo_box.SetBackgroundColour(wx.WHITE)
        self.effort_combo_box.Refresh()

    def _reset_flight_data_header_backgrounds(self):
        for widget in self._flight_data_header_widgets():
            if widget in (self.flight_number_text_ctrl, self.observer_text_ctrl):
                if widget.IsEditable():
                    widget.SetBackgroundColour(wx.WHITE)
                else:
                    widget.SetBackgroundColour(wx.Colour(*DISABLED_GRAY))
            elif widget is self.effort_combo_box:
                widget.SetBackgroundColour(wx.WHITE)
            else:
                widget.SetBackgroundColour(wx.Colour(*APP_GRAY))
            widget.Refresh()

    def _reset_flight_data_panel_child_backgrounds(self):
        """Clear collecting-state colours from child widgets."""
        self._reset_flight_data_header_backgrounds()
        for child in self.flight_data_panel.GetChildren():
            if child in self._flight_data_header_widgets():
                continue
            if child in (self.recording_gauge, self.detectors_gauge):
                continue
            if isinstance(child, wx.StaticLine):
                continue
            if isinstance(child, wx.TextCtrl):
                if child.IsEditable():
                    child.SetBackgroundColour(wx.WHITE)
                else:
                    child.SetBackgroundColour(wx.Colour(*DISABLED_GRAY))
            elif isinstance(child, (wx.Button, wx.ComboBox)):
                child.SetBackgroundColour(
                    wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)
                )
            else:
                child.SetBackgroundColour(wx.Colour(*APP_GRAY))
            child.Refresh()

    def _spoof_events_colour(self):
        return wx.Colour(*WARN_AMBER)

    def _set_panel_background(self, panel, colour, propagate=False):
        panel.SetBackgroundColour(colour)
        if propagate:
            for child in panel.GetChildren():
                if isinstance(child, wx.StaticLine):
                    continue
                child.SetBackgroundColour(colour)
        panel.Refresh()

    def _reset_navigation_panel_background(self):
        if self._spoof_gps:
            self._set_panel_background(
                self.ins_control_panel, self._spoof_events_colour(), propagate=True
            )
            return
        self._set_panel_background(
            self.ins_control_panel, wx.Colour(*APP_GRAY), propagate=True
        )
        for child in self.ins_control_panel.GetChildren():
            if isinstance(child, wx.TextCtrl):
                child.SetBackgroundColour(wx.WHITE)

    def _apply_event_spoof_panel_style(self, active):
        colour = self._spoof_events_colour()
        if active:
            self._set_panel_background(
                self.ins_control_panel, colour, propagate=True
            )
        else:
            self._reset_navigation_panel_background()

    def _set_field_enabled(self, ctrl, enabled):
        """Enable/disable an input field and make the disabled state obvious.

        For text fields we use read-only instead of a full Disable() so the
        frozen value stays fully legible (GTK greys out the text of disabled
        controls, hiding the value). A darker background plus dark text makes
        the read-only/frozen state obvious while keeping the value visible.
        """
        if isinstance(ctrl, wx.TextCtrl):
            ctrl.SetEditable(enabled)
            ctrl.SetForegroundColour(
                wx.BLACK if enabled else wx.Colour(*TEXTCTRL_DARK)
            )
        else:
            ctrl.Enable(enabled)
        ctrl.SetBackgroundColour(
            wx.WHITE if enabled else wx.Colour(*DISABLED_GRAY)
        )
        ctrl.Refresh()

    def _camera_param_dropdowns(self):
        return (
            self.rgb_shutter_mode_combo,
            self.exposure_min_combo,
            self.exposure_max_combo,
            self.gain_min_combo,
            self.gain_max_combo,
            self.aperture_min_combo,
            self.aperture_max_combo,
        )

    def _camera_param_dropdown_state(self):
        state = []
        for combo in self._camera_param_dropdowns():
            if not combo.IsEnabled():
                continue
            sel = combo.GetSelection()
            if sel == wx.NOT_FOUND:
                state.append((id(combo), None))
            else:
                state.append((id(combo), combo.GetStringSelection()))
        return tuple(state)

    def _bind_camera_param_dropdown_handlers(self):
        for combo in self._camera_param_dropdowns():
            combo.Bind(wx.EVT_COMBOBOX, self._on_camera_param_dropdown_changed)

    def _on_camera_param_dropdown_changed(self, event):
        self._update_set_camera_param_button_highlight()
        event.Skip()

    def _save_applied_camera_param_state(self):
        self._applied_camera_param_state = self._camera_param_dropdown_state()
        self._update_set_camera_param_button_highlight()

    def _update_set_camera_param_button_highlight(self):
        applied = getattr(self, "_applied_camera_param_state", ())
        pending = self._camera_param_dropdown_state() != applied
        if pending:
            self.m_button10.SetBackgroundColour(wx.Colour(*WARN_AMBER))
            self.m_button10.SetForegroundColour(wx.Colour(*TEXTCTRL_DARK))
        else:
            self.m_button10.SetBackgroundColour(self._set_camera_param_btn_default_bg)
            self.m_button10.SetForegroundColour(self._set_camera_param_btn_default_fg)
        self.m_button10.Refresh()

    @staticmethod
    def _format_aperture_stop(stop):
        if stop == int(stop):
            return str(int(stop))
        return "%g" % stop

    @staticmethod
    def _shutter_denom_label(denom):
        return "1/%d" % denom

    @staticmethod
    def _shutter_denom_seconds(denom):
        return 1.0 / denom

    @staticmethod
    def _p1_shutter_denoms_for_mode(mode):
        if mode == P1_SHUTTER_MODE_LS:
            return P1_LS_SHUTTER_DENOMS
        return P1_ES_SHUTTER_DENOMS

    @staticmethod
    def _p1_shutter_mode_from_value(raw):
        if raw in (None, ""):
            return P1_SHUTTER_MODE_ES
        return str(int(float(raw)))

    @staticmethod
    def _p1_shutter_mode_label(mode):
        return "LS" if mode == P1_SHUTTER_MODE_LS else "ES"

    @staticmethod
    def _p1_iso_from_param(raw):
        if raw in (None, ""):
            return None
        return nearest(P1_ISO_STOPS, int(float(raw)))

    @staticmethod
    def _set_stop_combo(combo, stops, value):
        if value in (None, ""):
            combo.SetSelection(wx.NOT_FOUND)
            return
        stop = nearest(stops, float(value))
        combo.SetSelection(stops.index(stop))

    @staticmethod
    def _get_stop_combo(combo, stops):
        sel = combo.GetSelection()
        if sel == wx.NOT_FOUND:
            return None
        return stops[sel]

    def _exposure_input_sizer(self):
        return self.exposure_min_value_txt_ctrl.GetContainingSizer()

    def _camera_panel_sizer(self):
        return self.exposure_min_value_txt_ctrl.GetParent().GetSizer()

    def _get_p1_shutter_mode(self):
        return (
            P1_SHUTTER_MODE_LS
            if self.rgb_shutter_mode_combo.GetStringSelection() == "LS"
            else P1_SHUTTER_MODE_ES
        )

    def _p1_shutter_stops(self):
        denoms = self._p1_shutter_denoms_for_mode(self._get_p1_shutter_mode())
        return [self._shutter_denom_seconds(d) for d in denoms]

    def _populate_p1_shutter_combos(self, preserve=True):
        denoms = self._p1_shutter_denoms_for_mode(self._get_p1_shutter_mode())
        labels = [self._shutter_denom_label(d) for d in denoms]
        old_min = (
            self._get_stop_combo(self.exposure_min_combo, self._p1_exposure_stops)
            if preserve and hasattr(self, "_p1_exposure_stops")
            else None
        )
        old_max = (
            self._get_stop_combo(self.exposure_max_combo, self._p1_exposure_stops)
            if preserve and hasattr(self, "_p1_exposure_stops")
            else None
        )
        self._p1_exposure_stops = [self._shutter_denom_seconds(d) for d in denoms]
        for combo in (self.exposure_min_combo, self.exposure_max_combo):
            combo.Clear()
            for label in labels:
                combo.Append(label)
        if old_min is not None:
            self._set_stop_combo(self.exposure_min_combo, self._p1_exposure_stops, old_min)
        if old_max is not None:
            self._set_stop_combo(self.exposure_max_combo, self._p1_exposure_stops, old_max)

    def _setup_rgb_controls(self):
        self._suppress_rgb_shutter_mode_confirm = False
        self._rgb_shutter_mode_confirmed = self.rgb_shutter_mode_combo.GetStringSelection()
        self._p1_exposure_stops = []
        self._populate_p1_shutter_combos(preserve=False)
        for combo in (self.gain_min_combo, self.gain_max_combo):
            combo.Clear()
            for stop in P1_ISO_STOPS:
                combo.Append(str(stop))
            combo.SetSelection(wx.NOT_FOUND)
        for combo in (self.aperture_min_combo, self.aperture_max_combo):
            combo.Clear()
            for stop in P1_APERTURE_STOPS:
                combo.Append(self._format_aperture_stop(stop))
            combo.SetSelection(wx.NOT_FOUND)

    def _update_camera_setting_widgets(self, mode):
        use_rgb = mode == "RGB"
        exposure_sizer = self._exposure_input_sizer()
        gain_sizer = self.gain_min_value_txt_ctrl.GetContainingSizer()
        panel_sizer = self._camera_panel_sizer()

        exposure_sizer.Show(self.exposure_min_value_txt_ctrl, not use_rgb)
        exposure_sizer.Show(self.exposure_max_value_txt_ctrl, not use_rgb)
        exposure_sizer.Show(self.exposure_min_combo, use_rgb)
        exposure_sizer.Show(self.exposure_max_combo, use_rgb)

        gain_sizer.Show(self.gain_min_value_txt_ctrl, not use_rgb)
        gain_sizer.Show(self.gain_max_value_txt_ctrl, not use_rgb)
        gain_sizer.Show(self.gain_min_combo, use_rgb)
        gain_sizer.Show(self.gain_max_combo, use_rgb)

        panel_sizer.Show(self.rgb_shutter_mode_row, use_rgb)
        self.rgb_shutter_mode_row.Show(
            self.rgb_shutter_section, _SHOW_RGB_SHUTTER_MODE_CONTROLS
        )
        panel_sizer.Show(self.rgb_aperture_label, use_rgb)
        panel_sizer.Show(self.rgb_aperture_row, use_rgb)

        if use_rgb:
            self.m_staticText42.SetLabel("Shutter Speed")
            self.m_staticText422.SetLabel("ISO (Gain)")
        else:
            self.m_staticText42.SetLabel("Auto Exposure (ms)")
            self.m_staticText422.SetLabel("Auto Gain (0-32)")

        self.camera_panel.Layout()

    def _set_exposure_min_value(self, value):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        if mode == "RGB":
            self._set_stop_combo(
                self.exposure_min_combo, self._p1_shutter_stops(), value
            )
        else:
            self.exposure_min_value_txt_ctrl.SetValue(
                "" if value in (None, "") else str(value)
            )

    def _set_exposure_max_value(self, value):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        if mode == "RGB":
            self._set_stop_combo(
                self.exposure_max_combo, self._p1_shutter_stops(), value
            )
        else:
            self.exposure_max_value_txt_ctrl.SetValue(
                "" if value in (None, "") else str(value)
            )

    def _set_gain_min_value(self, value):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        if mode == "RGB":
            self._set_stop_combo(self.gain_min_combo, P1_ISO_STOPS, value)
        else:
            self.gain_min_value_txt_ctrl.SetValue(
                "" if value in (None, "") else str(value)
            )

    def _set_gain_max_value(self, value):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        if mode == "RGB":
            self._set_stop_combo(self.gain_max_combo, P1_ISO_STOPS, value)
        else:
            self.gain_max_value_txt_ctrl.SetValue(
                "" if value in (None, "") else str(value)
            )

    def _set_aperture_min_value(self, value):
        self._set_stop_combo(self.aperture_min_combo, P1_APERTURE_STOPS, value)

    def _set_aperture_max_value(self, value):
        self._set_stop_combo(self.aperture_max_combo, P1_APERTURE_STOPS, value)

    def _set_rgb_shutter_mode_selection(self, label):
        self._suppress_rgb_shutter_mode_confirm = True
        self.rgb_shutter_mode_combo.SetStringSelection(label)
        self._rgb_shutter_mode_confirmed = label
        self._suppress_rgb_shutter_mode_confirm = False

    def on_rgb_shutter_mode(self, event):
        selection = self.rgb_shutter_mode_combo.GetStringSelection()
        if (
            not self._suppress_rgb_shutter_mode_confirm
            and selection == "LS"
            and self._rgb_shutter_mode_confirmed != "LS"
        ):
            dlg = wx.MessageDialog(
                self,
                "Enabling the physical shutter greatly increases wear and tear on the camera, are you sure?",
                "Confirm Shutter Mode",
                wx.YES_NO | wx.ICON_WARNING,
            )
            if dlg.ShowModal() != wx.ID_YES:
                dlg.Destroy()
                self._set_rgb_shutter_mode_selection(self._rgb_shutter_mode_confirmed)
                return
            dlg.Destroy()
        self._rgb_shutter_mode_confirmed = selection
        self._populate_p1_shutter_combos(preserve=True)
        if event is not None:
            event.Skip()

    def on_modal_selection(self, event):
        self.on_camera_setting_subsys_selection(event)
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection()
        self._update_camera_setting_widgets(mode)
        rgb_fields = (
            self.exposure_min_combo,
            self.exposure_max_combo,
            self.gain_min_combo,
            self.gain_max_combo,
            self.rgb_shutter_mode_combo,
            self.aperture_min_combo,
            self.aperture_max_combo,
        )
        if mode == "IR":
            self._set_field_enabled(self.exposure_max_value_txt_ctrl, False)
            self._set_field_enabled(self.exposure_min_value_txt_ctrl, False)
            for ctrl in rgb_fields:
                self._set_field_enabled(ctrl, False)
            self._set_field_enabled(self.gain_min_value_txt_ctrl, False)
            self._set_field_enabled(self.gain_max_value_txt_ctrl, False)
            self._set_field_enabled(self.rgb_exposure_comp_txt_ctrl, False)
            self._set_field_enabled(self.ir_nuc_time, True)
        else:
            self._set_field_enabled(self.exposure_max_value_txt_ctrl, True)
            self._set_field_enabled(self.exposure_min_value_txt_ctrl, True)
            for ctrl in rgb_fields:
                self._set_field_enabled(ctrl, mode == "RGB")
            self._set_field_enabled(self.gain_min_value_txt_ctrl, True)
            self._set_field_enabled(self.gain_max_value_txt_ctrl, True)
            self._set_field_enabled(self.rgb_exposure_comp_txt_ctrl, mode == "RGB")
            self._set_field_enabled(self.ir_nuc_time, False)

    def set_camera_parameter(self, hosts, mode, param, val):
        mode = mode.lower()
        if "ISO" in param:
            val = int(val)
        if mode == "rgb" and "Shutter" in param and "Mode" not in param:
            val_sec = float(val)
            stops = self._p1_shutter_stops()
            if val_sec not in stops:
                val_sec = nearest(stops, val_sec)
                if "Max" in param:
                    self._set_exposure_min_value(val_sec)
                elif "Min" in param:
                    self._set_exposure_max_value(val_sec)
            val = str(val_sec)

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

    @staticmethod
    def _param_values_equal(a, b):
        if a in (None, "") and b in (None, ""):
            return True
        if a in (None, "") or b in (None, ""):
            return False
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return str(a) == str(b)

    def _camera_setting_hosts(self, fov):
        if fov == "all":
            return self.hosts
        return [host_from_fov(fov)]

    def _kv_param_unified(self, hosts, mode, param):
        values = []
        for host in hosts:
            topic_base = "/".join(["", "sys", "actual_geni_params", host, mode, ""])
            values.append(kv.get(topic_base + param, None))
        if not values:
            return None
        first = values[0]
        for val in values[1:]:
            if not self._param_values_equal(first, val):
                return None
        return first

    def _clear_rgb_shutter_mode_selection(self):
        self._suppress_rgb_shutter_mode_confirm = True
        self.rgb_shutter_mode_combo.SetSelection(wx.NOT_FOUND)
        self._rgb_shutter_mode_confirmed = ""
        self._suppress_rgb_shutter_mode_confirm = False

    def on_camera_setting_subsys_selection(self, event):
        mode = self.camera_setting_rgb_uv_combo.GetStringSelection().lower()
        fov = self.camera_setting_subsys.GetString(
            self.camera_setting_subsys.GetCurrentSelection()
        ).lower()
        hosts = self._camera_setting_hosts(fov)
        if mode == "ir":
            nuc_time = self._kv_param_unified(
                hosts, mode, "CorrectionAutoDeltaTime"
            )
            if nuc_time in (None, ""):
                self.ir_nuc_time.SetValue("")
            else:
                self.ir_nuc_time.SetValue(str(nuc_time))
        elif mode == "rgb" or mode == "uv":
            if mode == "rgb":
                shutter_mode_raw = self._kv_param_unified(
                    hosts, mode, P1_SHUTTER_MODE_PARAM
                )
                if shutter_mode_raw not in (None, ""):
                    self._set_rgb_shutter_mode_selection(
                        self._p1_shutter_mode_label(
                            self._p1_shutter_mode_from_value(shutter_mode_raw)
                        )
                    )
                else:
                    self._clear_rgb_shutter_mode_selection()
                self._populate_p1_shutter_combos(preserve=False)
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
            gain_min = self._kv_param_unified(hosts, mode, GAIN_MIN_PARAM)
            gain_max = self._kv_param_unified(hosts, mode, GAIN_MAX_PARAM)
            exp_min = self._kv_param_unified(hosts, mode, EXPOSURE_MIN_PARAM)
            exp_max = self._kv_param_unified(hosts, mode, EXPOSURE_MAX_PARAM)
            if mode == "rgb":
                gain_min = self._p1_iso_from_param(gain_min)
                gain_max = self._p1_iso_from_param(gain_max)
                exp_min = float(exp_min) if exp_min not in (None, "") else ""
                exp_max = float(exp_max) if exp_max not in (None, "") else ""
                self._set_exposure_min_value(exp_min)
                self._set_exposure_max_value(exp_max)
                self._set_gain_min_value(gain_min)
                self._set_gain_max_value(gain_max)
                aperture_min = self._kv_param_unified(
                    hosts, mode, P1_APERTURE_MIN_PARAM
                )
                aperture_max = self._kv_param_unified(
                    hosts, mode, P1_APERTURE_MAX_PARAM
                )
                aperture_min = (
                    float(aperture_min) if aperture_min not in (None, "") else ""
                )
                aperture_max = (
                    float(aperture_max) if aperture_max not in (None, "") else ""
                )
                self._set_aperture_min_value(aperture_min)
                self._set_aperture_max_value(aperture_max)
                exposure_comp = self._kv_param_unified(
                    hosts, mode, P1_EXPOSURE_COMP_PARAM
                )
                if exposure_comp in (None, ""):
                    self.rgb_exposure_comp_txt_ctrl.SetValue("")
                else:
                    self.rgb_exposure_comp_txt_ctrl.SetValue(str(float(exposure_comp)))
            else:
                gain_min = (
                    int(float(gain_min) * gain_factor)
                    if gain_min not in (None, "")
                    else ""
                )
                gain_max = (
                    int(float(gain_max) * gain_factor)
                    if gain_max not in (None, "")
                    else ""
                )
                exp_max = float(exp_max) / exp_factor if exp_max not in (None, "") else ""
                exp_min = float(exp_min) / exp_factor if exp_min not in (None, "") else ""
                self._set_exposure_min_value(exp_min)
                self._set_exposure_max_value(exp_max)
                self.gain_min_value_txt_ctrl.SetValue(str(gain_min))
                self.gain_max_value_txt_ctrl.SetValue(str(gain_max))

        self._save_applied_camera_param_state()

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
                self._save_applied_camera_param_state()
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
                GAIN_MAX_PARAM = P1_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = P1_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = P1_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = P1_EXPOSURE_MIN_PARAM
                shutter_mode = self._get_p1_shutter_mode()
                self.set_camera_parameter(
                    hosts, mode, P1_SHUTTER_MODE_PARAM, val=shutter_mode
                )
                exp_min = self._get_stop_combo(
                    self.exposure_min_combo, self._p1_shutter_stops()
                )
                exp_max = self._get_stop_combo(
                    self.exposure_max_combo, self._p1_shutter_stops()
                )
                gain_min = self._get_stop_combo(self.gain_min_combo, P1_ISO_STOPS)
                gain_max = self._get_stop_combo(self.gain_max_combo, P1_ISO_STOPS)
                aperture_min = self._get_stop_combo(
                    self.aperture_min_combo, P1_APERTURE_STOPS
                )
                aperture_max = self._get_stop_combo(
                    self.aperture_max_combo, P1_APERTURE_STOPS
                )
            else:
                factor = 1e3
                GAIN_MAX_PARAM = PR_GAIN_MAX_PARAM
                GAIN_MIN_PARAM = PR_GAIN_MIN_PARAM
                EXPOSURE_MAX_PARAM = PR_EXPOSURE_MAX_PARAM
                EXPOSURE_MIN_PARAM = PR_EXPOSURE_MIN_PARAM
                EXPOSURE_MIN = PR_EXPOSURE_MIN
                exp_min = self.check_box_val(
                    self.exposure_min_value_txt_ctrl.GetValue(), float
                )
                exp_max = self.check_box_val(
                    self.exposure_max_value_txt_ctrl.GetValue(), float
                )
                gain_min = self.check_box_val(self.gain_min_value_txt_ctrl.GetValue())
                gain_max = self.check_box_val(self.gain_max_value_txt_ctrl.GetValue())
                aperture_min = None
                aperture_max = None
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
            elif mode == "RGB":
                if exp_max < exp_min:
                    msg = (
                        "Shutter speed maximum must allow equal or longer exposure "
                        "than the minimum."
                    )
                    dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
                self.set_camera_parameter(
                    hosts, mode, EXPOSURE_MIN_PARAM, val=exp_min
                )
                self.set_camera_parameter(
                    hosts, mode, EXPOSURE_MAX_PARAM, val=exp_max
                )
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
            elif mode == "RGB":
                if gain_max < gain_min:
                    msg = "ISO maximum must be >= ISO minimum."
                    dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
                self.set_camera_parameter(hosts, mode, GAIN_MIN_PARAM, val=gain_min)
                self.set_camera_parameter(hosts, mode, GAIN_MAX_PARAM, val=gain_max)
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

            if mode == "RGB":
                if (aperture_min is None and aperture_max is not None) or (
                    aperture_max is None and aperture_min is not None
                ):
                    msg = "Both aperture min/max value must be set."
                    dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
                    return
                if aperture_min is not None and aperture_max is not None:
                    if aperture_max < aperture_min:
                        msg = "Aperture maximum must be >= aperture minimum."
                        dlg = wx.MessageDialog(
                            self, msg, "Error", wx.OK | wx.ICON_ERROR
                        )
                        dlg.ShowModal()
                        dlg.Destroy()
                        return
                    self.set_camera_parameter(
                        hosts, mode, P1_APERTURE_MIN_PARAM, val=aperture_min
                    )
                    self.set_camera_parameter(
                        hosts, mode, P1_APERTURE_MAX_PARAM, val=aperture_max
                    )
                exposure_comp = self.check_box_val(
                    self.rgb_exposure_comp_txt_ctrl.GetValue(), float
                )
                if exposure_comp is not None:
                    self.set_camera_parameter(
                        hosts, mode, P1_EXPOSURE_COMP_PARAM, val=exposure_comp
                    )
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
        self._save_applied_camera_param_state()

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
            elif status is EPodStatus.Stalled:
                text_attr.SetForegroundColour(WARN_AMBER)
            elif status.is_ok():
                text_attr.SetForegroundColour(VERDANT_GREEN)
            else:
                text_attr.SetForegroundColour(ERROR_RED)

        active_statuses = [
            status for status in d_status.values()
            if status is not EPodStatus.Unknown
        ]
        if not d_desired or not active_statuses:
            self.detectors_gauge.SetBackgroundColour(FLAT_GRAY)
        elif any(status is EPodStatus.Failed for status in active_statuses):
            self.detectors_gauge.SetBackgroundColour(ERROR_RED)
        elif any(status is EPodStatus.Pending for status in active_statuses):
            self.detectors_gauge.SetBackgroundColour(WARN_AMBER)
        elif any(status is EPodStatus.Stalled for status in active_statuses):
            self.detectors_gauge.SetBackgroundColour(WARN_AMBER)
        elif all(status.is_ok() for status in active_statuses):
            self.detectors_gauge.SetBackgroundColour(BRIGHT_GREEN)
        else:
            self.detectors_gauge.SetBackgroundColour(FLAT_GRAY)

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

        # The startup unclip runs before GTK finalizes panel sizes, so labels
        # center against stale widths. Re-run once, after the window is fully
        # realized (the first timer tick is guaranteed to be late enough).
        if not self._did_initial_unclip:
            self._did_initial_unclip = True
            gui_utils.unclip_static_text(self)

        cameras_ok = True
        now = time.time()
        for panel in self.remote_image_panels:
            # Refresh images if needed
            panel.update_all_if_needed()
            if panel.last_update is None:
                cameras_ok = False
                panel.status_static_text.SetLabel(format_status())
                panel.status_static_text.SetForegroundColour(BRIGHT_RED)
                gui_utils.refit_label(panel.status_static_text)
            else:
                dt = now - panel.last_update
                if dt > self._collect_stream_stale_sec:
                    cameras_ok = False
                    panel.status_static_text.SetLabel(format_status(dt=dt))
                    panel.status_static_text.SetForegroundColour(BRIGHT_RED)
                    gui_utils.refit_label(panel.status_static_text)
        self._collect_health_cameras_ok = (
            cameras_ok if self.remote_image_panels else True
        )

        self.update_collect_colors()

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
                    self._reset_flight_data_panel_child_backgrounds()
        self.last_collecting = is_collecting
        self._collecting = False if is_collecting is None else is_collecting
        self._collect_in_region = (
            False if collect_in_region is None else collect_in_region
        )
        self._spoof_events = int(kv.get("/debug/spoof_events", 0))

        if is_collecting == True:
            self.recording_gauge.SetBackgroundColour((0, 255, 0))
            panel_colour = self._collecting_panel_colour()
            self.flight_data_panel.SetBackgroundColour(panel_colour)
            self._apply_flight_data_header_background(panel_colour)

        elif is_collecting == False:
            self.recording_gauge.SetBackgroundColour((200, 200, 200))
            if collect_in_region is None or collect_in_region is False:
                self.flight_data_panel.SetBackgroundColour(wx.Colour(*APP_GRAY))
            else:
                self.flight_data_panel.SetBackgroundColour(
                    wx.Colour(*SHAPE_COLLECT_BLUE)
                )
        else:
            raise Exception("invalid value encountered for is_collecting")

        self._apply_event_spoof_panel_style(self._spoof_events == 1)

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
        self.flight_number_text_ctrl.Unbind(wx.EVT_TEXT_ENTER)
        self.flight_number_text_ctrl.Unbind(wx.EVT_KILL_FOCUS)
        self.flight_number_text_ctrl.SetValue(self.flight_number_str)
        self.flight_number_text_ctrl.Bind(wx.EVT_TEXT, self.on_flight_number_text)
        self.flight_number_text_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_flight_number_commit)
        self.flight_number_text_ctrl.Bind(wx.EVT_KILL_FOCUS, self.on_flight_number_commit)
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
        self.open_image_inspection_panel("Left View RGB")

    def on_dclick_center_rgb(self, event):
        self.open_image_inspection_panel("Center View RGB")

    def on_dclick_right_rgb(self, event):
        self.open_image_inspection_panel("Right View RGB")

    def on_dclick_left_ir(self, event):
        self.open_image_inspection_panel("Left View IR")

    def on_dclick_center_ir(self, event):
        self.open_image_inspection_panel("Center View IR")

    def on_dclick_right_ir(self, event):
        self.open_image_inspection_panel("Right View IR")

    def on_dclick_left_uv(self, event):
        self.open_image_inspection_panel("Left View UV")

    def on_dclick_center_uv(self, event):
        self.open_image_inspection_panel("Center View UV")

    def on_dclick_right_uv(self, event):
        self.open_image_inspection_panel("Right View UV")

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
            spoofed = kv.get_dict("/debug", {}).get("spoof", {}).get("ins", {})
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

        self._spoof_events = int(kv.get("/debug/spoof_events", 0))
        if self._spoof_events == 1:
            self._apply_event_spoof_panel_style(True)
            self.gnss_status_flag_txtctrl.SetValue("no fix! event spoof!")
            return
        self._apply_event_spoof_panel_style(False)
        if self._spoof_gps:
            self.gnss_status_flag_txtctrl.SetValue("gps spoofed!")
        elif msg.gnss_status == 0:
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

    def on_flight_number_text(self, event=None):
        val = self.flight_number_text_ctrl.GetValue()
        if val == "":
            self.flight_number_text_ctrl.SetForegroundColour(wx.Colour(180, 180, 180))
        elif self._flight_number_valid(val):
            self.flight_number_text_ctrl.SetForegroundColour(wx.NullColour)
        else:
            self.flight_number_text_ctrl.SetForegroundColour(wx.Colour(200, 0, 0))
        self.flight_number_text_ctrl.Refresh()

    def on_flight_number_commit(self, event=None):
        val = self.flight_number_text_ctrl.GetValue()
        if self._flight_number_valid(val):
            self.flight_number_str = val
            self.flight_number_text_ctrl.SetForegroundColour(wx.NullColour)
            self.flight_number_text_ctrl.Refresh()
            self.add_to_console_log(self.flight_number_str, "on_flight_number_commit")
        else:
            msg = "Flight number must be an integer, e.g. 01, 02, 001, etc."
            dlg = wx.MessageDialog(self, msg, "Invalid Flight Number", wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()
            self.flight_number_text_ctrl.SetValue(self.flight_number_str)
            self.flight_number_text_ctrl.SetForegroundColour(wx.NullColour)
            self.flight_number_text_ctrl.Refresh()
        if event:
            event.Skip()

    @staticmethod
    def _flight_number_valid(val):
        try:
            int(val)
            return bool(val)
        except (ValueError, TypeError):
            return False

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
        # Titles get a stale best size while hidden; re-fit them once shown.
        wx.CallAfter(gui_utils.unclip_static_text, self)

    # ------------------------------------------------------------------------
    # ---------------------------- Detection Menu ----------------------------

    def resolve_detector_pipefile(self, host):
        """Return the detector pipefile for a host from the active camera config."""
        return get_detector_pipefile(host, self.get_sys_cfg())

    def do_start_a_detector(self, event=None, host="unset", log=True):
        cmdpipef = self.resolve_detector_pipefile(host)
        if not cmdpipef:
            msg = (
                "No detector pipefile configured for %s; "
                "select a camera configuration with a detector model."
                % host
            )
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
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
            None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN
        )
        if dialog.ShowModal() == wx.ID_OK:
            print(dialog.GetPath())

    # ------------------------------------------------------------------------

    def save_flight_summary(self, event=None):
        wildcard = "Camera model (*.yaml)"
        dialog = wx.FileDialog(None, "Choose a file", "/mnt", "", wildcard, wx.FD_SAVE)
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
        else:
            self._camera_config_frame.Show()

        self._camera_config_frame.Raise()
        wx.CallAfter(gui_utils.unclip_static_text, self._camera_config_frame)

    def set_camera_config_dict(self, config_dict=None, select_str=None):
        curr_str = select_str if select_str is not None else self.get_sys_cfg()
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
        SYS_CFG["arch"]["sys_cfg"] = self.get_sys_cfg()

    def previous_camera_config(self, event=None):
        ind = self.camera_config_combo.GetSelection()
        if ind == wx.NOT_FOUND:
            ind = 0
        else:
            ind = ind - 1

        if ind < 0:
            ind = self.camera_config_combo.GetCount() - 1

        self.camera_config_combo.SetSelection(ind)
        SYS_CFG["arch"]["sys_cfg"] = self.get_sys_cfg()

    def on_camera_config_combo(self, event=None):
        """ """
        # Add log of this event to event log.
        curr_str = self.get_sys_cfg()
        self.add_to_event_log("system_config: %s" % curr_str)
        cc = SYS_CFG["camera_cfgs"][curr_str]
        center_rgb_yaml = (cc.get("center_rgb_yaml_path") or "").strip()
        if center_rgb_yaml and os.path.isfile(center_rgb_yaml):
            SYS_CFG["rgb_vfov"] = load_from_file(center_rgb_yaml).fov()[1]
        SYS_CFG["arch"]["sys_cfg"] = curr_str
        save_camera_config(curr_str, camera_cfgs=SYS_CFG["camera_cfgs"])
        self.update_project_flight_params()

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
        info = wx.adv.AboutDialogInfo()
        info.Name = "KAMERA System Control Panel"
        info.Version = "2.0.0"
        info.Copyright = "(C) 2026 Kitware"
        info.Description = wordwrap(
            "This GUI allows control of the KAMERA system.",
            350,
            wx.ClientDC(self.ins_control_panel),
        )
        info.WebSite = ("http://www.kitware.com", "Kitware")
        info.Developers = ["Matt Brown, Adam Romlein, Mike McDermott"]
        info.License = wordwrap(LICENSE_STR, 500, wx.ClientDC(about_panel))
        wx.adv.AboutBox(info)

    # ------------------------------------------------------------------------

    def _enable_state_controls(self):
        for child in self.flight_data_panel.GetChildren():
            if isinstance(child, wx.TextCtrl):
                self._set_field_enabled(child, True)
            else:
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
            if isinstance(child, wx.TextCtrl):
                self._set_field_enabled(child, False)
            else:
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
