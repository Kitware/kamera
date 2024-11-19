# -*- coding: utf-8 -*-
import json
import os
import pygeodesy
import redis
import re
import time
import yaml
from collections import OrderedDict
from functools import reduce

import rospy
from cv_bridge import CvBridge, CvBridgeError

from roskv.impl.redis_envoy import RedisEnvoy as ImplEnvoy
import wxpython_gui
from wxpython_gui.utils import check_default, MissedFrameStore


# Figure out relative positions
HERE_DIR = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.realpath(os.path.join(HERE_DIR, "../.."))
DOCK_KAM_REPO_DIR = os.path.realpath(os.path.join(PKG_DIR, "../../.."))
REAL_KAM_REPO_DIR = os.path.realpath(os.path.join(PKG_DIR, "../../.."))

# =================== LOADING SYS CONFIG =======================
# This will *always* be loaded from disk and then pushed into redis
# These never-changing values are placed under "/sys/arch", and will never
# have to be backed up
system_name = os.getenv("SYSTEM_NAME")
cfg_file = "%s/src/cfg/%s/config.yaml" % (REAL_KAM_REPO_DIR, system_name)
with open(cfg_file, "r") as stream:
    try:
        USER_CFG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# Need a redis instance to push/pull from
kv = ImplEnvoy(host=USER_CFG["redis_host"])
# vanilla instance to put shapefile bytes in
redis = redis.Redis(USER_CFG["redis_host"])
# Force Redis values to match these hardcoded values on disk
# kv.put("/sys/arch", cfg)
# =================== FINISHED SYS CONFIG =======================


# Attempting to create a dict monitored by Redis
class Cfg(dict):
    def __init__(self, *args, **kwargs):
        self.ns = kwargs.pop("ns", "/")
        self.update(*args, **kwargs)

    def __delitem__(self, key):
        # tic = time.time()
        if isinstance(self[key], dict):
            kv.delete_dict("%s/%s" % (self.ns, key))
        kv.delete("%s/%s" % (self.ns, key))
        super(Cfg, self).__delitem__(key)

    def __getitem__(self, key):
        # print("Getting item %s" % key)
        val = super(Cfg, self).__getitem__(key)
        return val

    #    def __delitem__(self, key):
    #        kv

    def __setitem__(self, key, val):
        # tic = time.time()
        kv.put("%s/%s" % (self.ns, key), val)
        # toc = time.time()
        # print("Time to access was %s" % (toc - tic))
        if isinstance(val, dict):
            val = Cfg(val, ns="%s/%s" % (self.ns, key))
        super(Cfg, self).__setitem__(key, val)

    def __repr__(self):
        dictrepr = super(Cfg, self).__repr__()
        return "%s(%s)" % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError("update expected at most 1 arguments, got %d" % len(args))
        other = dict(*args, **kwargs)
        for key in other:
            self[key] = other[key]


# =================== LOADING GUI CONFIG =======================
# This will attempt to load from redis first, and backfill from disk
# Values are placed under "/sys/gui", with sys params under "/sys/gui/arch"
# Location of the system configuration file dictionary.
config_filename = os.path.join(USER_CFG["gui_cfg_dir"], "system_state.json")
# create from template if it doesn't exist
if not os.path.isfile(config_filename):
    wxpython_gui.utils.make_path(config_filename, from_file=True)
    with open(os.path.join(PKG_DIR, "config/default_system_state.json"), "r") as infile:
        with open(config_filename, "w") as outfile:
            outfile.write(infile.read())
            print("Created config from scratch: {}".format(config_filename))
try:
    GUI_ARCH_KV = kv.get_dict("/sys")
except Exception as e:
    GUI_ARCH_KV = {}
with open(config_filename, "r") as input_file:
    GUI_ARCH_DEFAULT = json.load(input_file)

# Since we're grabbing everything, trim out camera configs for insert
try:
    kv.delete_dict("/sys/camera_cfgs")
except:
    pass
# Fill in any missing value in redis from disk
GUI_ARCH = check_default(GUI_ARCH_KV, GUI_ARCH_DEFAULT)


def save_config_settings():
    print("Saving config settings.")
    with open(config_filename, "w") as output_file:
        json.dump(SYS_CFG, output_file, indent=4, sort_keys=True)


# Fill in values that will change more frequently, not ones that are hardcoded
# in the user-config.yml.
# kv.put("/sys", GUI_ARCH)
# =================== FINISHED GUI CONFIG =======================

# =================== LOADING CAMERA CONFIG =======================
# This will attempt to load from redis first, and backfill from disk
# Location of the camera configuration file dictionary.
camera_config_filename = os.path.join(
    USER_CFG["gui_cfg_dir"], "camera_configurations.json"
)
try:
    CAMERA_CFGS_KV = kv.get_dict("/sys/camera_cfgs")
except:
    CAMERA_CFGS_KV = {}
try:
    with open(camera_config_filename, "r") as input_file:
        CAMERA_CFGS_DEFAULT = json.load(input_file)
except Exception as e:
    print(e)
    CAMERA_CFGS_DEFAULT = {}
# Fill in any missing value in redis from disk
CAMERA_CFGS = {}
CAMERA_CFGS["camera_cfgs"] = check_default(CAMERA_CFGS_KV, CAMERA_CFGS_DEFAULT)
# kv.put("/sys/camera_cfgs", CAMERA_CFGS)
# =================== FINISHED CAMERA CONFIG =======================


def merge_two_dicts(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_two_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                print("Warning: conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


# Combine entries from user-config.yml, camera_configurations.json, and system_state.json
SYS_ARCH = {}
SYS_ARCH = merge_two_dicts(SYS_ARCH, CAMERA_CFGS)
SYS_ARCH = merge_two_dicts(SYS_ARCH, GUI_ARCH)
# This is required since both files contain "arch" keys and otherwise
# update won't merge properly
USER_CFG["arch"] = merge_two_dicts(USER_CFG["arch"], GUI_ARCH["arch"])
USER_CFG = merge_two_dicts(USER_CFG, GUI_ARCH)
SYS_ARCH = merge_two_dicts(SYS_ARCH, USER_CFG)
# kv.put("/sys", SYS_ARCH)

# Get the global configuration combining all 3 configurations above
SYS_CFG = Cfg(ns="/sys")
SYS_CFG.update(SYS_ARCH)


# =================== DEFINE GLOBALS ===============================
missed_frame_store = MissedFrameStore()
# Need a vanilla one for binary insert
ros_immediate = rospy.Duration(nsecs=1)
# Instantiate CvBridge
bridge = CvBridge()

TEXTCTRL_GRAY = (255, 23, 23)
TEXTCTRL_WHITE = (255, 255, 255)
TEXTCTRL_DARK = (20, 20, 20)
APP_GRAY = (220, 218, 213)  # Default application background
FLAT_GRAY = (200, 200, 200)
COLLECT_GREEN = (55, 120, 25)
SHAPE_COLLECT_BLUE = (
    52,
    100,
    212,
)  # shape file is set and "primed" but not currently in it
VERDANT_GREEN = (0, 170, 45)
BRIGHT_GREEN = (0, 255, 0)
BRIGHT_RED = (255, 0, 0)
ERROR_RED = (192, 0, 0)
WARN_AMBER = (255, 192, 90)
WARN_AMBER2 = (224, 128, 0)
WARN_AMBER2 = (224, 128, 0)
WTF_PURPLE = (96, 48, 192)

# Location of the geod file.
geod_filename = os.path.join(DOCK_KAM_REPO_DIR, "assets/geods/egm84-15.pgm")
geod = pygeodesy.geoids.GeoidPGM(geod_filename)
PAT_BRACED = re.compile(r"\{(\w+)\}")


LICENSE_STR = "".join(
    [
        "Copyright 2018 by Kitware, Inc.\n",
        "All rights reserved.\n\n",
        "Redistribution and use in source and binary forms, with or without ",
        "modification, are permitted provided that the following conditions are met:",
        "\n\n",
        "* Redistributions of source code must retain the above copyright notice, ",
        "this list of conditions and the following disclaimer.",
        "\n\n",
        "* Redistributions in binary form must reproduce the above copyright notice, ",
        "this list of conditions and the following disclaimer in the documentation ",
        "and/or other materials provided with the distribution.",
        "\n\n",
        "* Neither name of Kitware, Inc. nor the names of any contributors may be ",
        "used to endorse or promote products derived from this software without ",
        "specific prior written permission.",
        "\n\n",
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ",
        "'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED ",
        "TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR ",
        "PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE ",
        "LIABLE FORANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR ",
        "CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF ",
        "SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS ",
        "INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN ",
        "CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ",
        "ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE ",
        "POSSIBILITY OF SUCH DAMAGE.",
    ]
)

# =================== END GLOBALS ===============================

# Path helper functions


def get_template_keys(tmpl):
    return re.findall(PAT_BRACED, tmpl)


def conformKwargsToFormatter(tmpl, kwargs):
    # type: (str, dict) -> dict
    required_keys = set(get_template_keys(tmpl))
    missing_keys = required_keys.difference(kwargs)
    fmt_dict = {k: v for k, v in kwargs.items() if k in required_keys}
    fmt_dict.update({k: "({})".format(k) for k in missing_keys})
    return fmt_dict


def get_arch_path():
    arch_dict = SYS_CFG["arch"]
    tmpl = SYS_CFG["arch"]["base_template"]
    fmt_dict = conformKwargsToFormatter(tmpl, arch_dict)
    return tmpl.format(**fmt_dict)


def sync_dicts(truth, sync):
    assert len(truth.keys()) == len(sync.keys())
    for k, v in truth.items():
        if isinstance(v, dict):
            sync_dicts(truth[k], sync[k])
        if v != sync[k]:
            sync[k] = v


def pull_gui_state():
    print("Pulling gui state")
    try:
        d1 = kv.get_dict("/sys/arch")
    except Exception as e:
        print(e)
        d1 = {}
    d2 = SYS_CFG["arch"]
    d3 = check_default(d2, d1)
    sync_dicts(d3, d2)


def save_camera_config(curr_cfg=None):
    print("Saving camera configuration")
    # Save to local cache
    if not os.path.isfile(camera_config_filename):
        wxpython_gui.utils.make_path(camera_config_filename, from_file=True)
    with open(camera_config_filename, "w") as outfile:
        json.dump(SYS_CFG["camera_cfgs"], outfile, indent=4, sort_keys=True)
        print("Saved config: {}".format(camera_config_filename))

    # Saving specific camera config to dir
    if curr_cfg is not None:
        SYS_CFG["syscfg_dir"] = dirname = get_arch_path()
        # Always save to NAS on guibox
        dirname = "/mnt/flight_data/" + "/".join(dirname.split("/")[3:])
        if dirname is not None:
            fname = "%s/sys_config.json" % dirname
        else:
            return
        if not os.path.isfile(fname):
            wxpython_gui.utils.make_path(fname, from_file=True)
        with open(fname, "w") as outfile:
            json.dump(SYS_CFG["camera_cfgs"][curr_cfg], outfile, indent=4, sort_keys=True)
            print("Saved sys config: {}".format(fname))
        return dirname
    else:
        return camera_config_filename


def format_status(
    timeval=None,
    num_dropped=0,
    exposure_us=None,
    gain=None,
    dt=0,
    fps=0.0,
    chan=None,
    total=None,
    processed=None,
):
    # type: (datetime.datetime, int, int, int, float, float, str) -> unicode
    """
    Render the status message.
    ☒☀⚠⍙

    :param timeval: Time of last valid message
    :param num_dropped: Count of dropped frames
    :param exposure_us: Exposure value, in microseconds
    :param gain:
    :param dt: Time elapsed since last good frame
    :return:
    """
    if not timeval:
        extra = " for {:.1f}s".format(dt) if dt else " ☒"

        return "☒ No image stream\n" + extra
    time_str = str(timeval.time())[3:11]

    drop_str = (
        "{} dropped".format(num_dropped)
        if num_dropped < 10000
        else "{:.0e} dropped".format(num_dropped)
    )
    gain_str = "Gain:?" if gain is None else "Gain:{}".format(gain)
    if total is not None and processed is not None:
        drop_str += " | DB: {}/{}".format(processed, total)
    total_str = "" if total is None else "{}".format(total)
    processed_str = "" if processed is None else "{}".format(processed)
    expo_str = (
        "Exp: ? ms"
        if exposure_us is None
        else "Exp: {:0.2f} ms".format(float(exposure_us) * 1e-3)
    )
    if chan == "ir":
        fmt = "{fps: 4.2f} fps\n{drop}"
        out = fmt.format(fps=fps, drop=drop_str)
    else:
        fmt = "{gain} | {fps: 4.2f} fps\n{expo}\n{drop}"
        out = fmt.format(
            time=time_str, gain=gain_str, fps=fps, expo=expo_str, drop=drop_str
        )
    return out


def channel_format_status(fov, chan, timeval=None, dt=0):
    driver = "%s_driver" % chan
    for host, cfgs in SYS_CFG["arch"]["hosts"].items():
        if cfgs["fov"] == fov:
            final = host
    a = "actual_geni_params"
    param_ns = "/".join(["", "sys", a, final, chan])
    missed_topic = "/".join(["", fov, chan, "missed"])
    num_dropped = missed_frame_store.get(missed_topic, 0)
    fps = missed_frame_store.get_fps(fov, chan) or 0.0
    exposure_us = None
    gain = None
    total = None
    processed = None
    if chan == "uv":
        gain = kv.get(param_ns + "/GainValue", None)
        exposure_us = kv.get(param_ns + "/ExposureValue", None)
    elif chan == "rgb":
        gain = int(float(kv.get(param_ns + "/ISO", None)) / 50.0)
        # convert float point seconds to us
        exposure_us = float(kv.get(param_ns + "/Shutter_Speed", None)) * 1e6
        total = int(kv.get("/sys/" + final + "/p1debayerq/total"))
        processed = int(kv.get("/sys/" + final + "/p1debayerq/processed"))
    try:
        dt = float(kv.get(param_ns + "/last_msg_time", None))
    except:
        dt = 0
    return format_status(
        timeval=timeval,
        num_dropped=num_dropped,
        exposure_us=exposure_us,
        gain=gain,
        fps=fps,
        chan=chan,
        dt=dt,
        total=total,
        processed=processed,
    )


def host_from_fov(fov):
    # type: (str) -> str
    hosts = SYS_CFG["arch"]["hosts"]
    for host, attrs in hosts.items():
        if fov == attrs["fov"]:
            return host
    raise KeyError("FOV not found: '{}'".format(fov))
