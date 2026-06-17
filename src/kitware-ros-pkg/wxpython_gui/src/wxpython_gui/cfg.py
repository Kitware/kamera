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
from roskv.util import filter_hosts_by_system
import wxpython_gui
import wxpython_gui.utils  # bind the submodule for wxpython_gui.utils.make_path()


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
        ns_key = "%s/%s" % (self.ns, key)
        if isinstance(val, dict):
            # An empty dict has no leaves to flatten; kv.put would fall back to
            # SET-ing a raw dict, which Redis rejects. Skip the write and keep
            # the empty namespace in memory only.
            if val:
                kv.put(ns_key, val)
            val = Cfg(val, ns=ns_key)
        else:
            kv.put(ns_key, val)
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


# =================== CONFIG RESOLUTION =======================
# SYS_CFG is built by deep_merge-ing tiers low precedence to high; last wins:
#   1. default_system_state.json        factory defaults
#   2. <gui_cfg_dir>/system_state.json  last session (the only thing saved back)
#   3. Redis /sys/*                      live runtime values
#   4. config.yaml (USER_CFG)            static truth; ALWAYS WINS for its keys
# camera_configurations.json owns SYS_CFG["camera_cfgs"].


def deep_merge(base, override):
    """Recursively merge ``override`` into ``base`` (later wins) and return it."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            deep_merge(base[key], val)
        else:
            base[key] = val
    return base


# --- Tiers 1 & 2: factory defaults, then the cache (seeded from default) -----
config_filename = os.path.join(USER_CFG["gui_cfg_dir"], "system_state.json")
default_system_state_file = os.path.join(
    REAL_KAM_REPO_DIR, "src/cfg", system_name, "default_system_state.json"
)
try:
    with open(default_system_state_file, "r") as infile:
        DEFAULT_STATE = json.load(infile)
except Exception as e:
    print(e)
    DEFAULT_STATE = {}
if not os.path.isfile(config_filename):
    wxpython_gui.utils.make_path(config_filename, from_file=True)
    with open(config_filename, "w") as outfile:
        json.dump(DEFAULT_STATE, outfile, indent=4, sort_keys=True)
        print("Created config from scratch: {}".format(config_filename))
try:
    with open(config_filename, "r") as input_file:
        CACHE_STATE = json.load(input_file)
except Exception as e:
    print(e)
    CACHE_STATE = {}

# --- Tier 3: live Redis state ------------------------------------------------
try:
    REDIS_LIVE = kv.get_dict("/sys")
except Exception as e:
    print(e)
    REDIS_LIVE = {}
REDIS_LIVE.pop("camera_cfgs", None)  # presets come from the file below

# --- Camera presets (camera_configurations.json is authoritative) ------------
# Mirrors the system_state cache: seed the runtime file from the in-repo default
# the first time the GUI runs, then load it.
camera_config_filename = os.path.join(
    USER_CFG["gui_cfg_dir"], "camera_configurations.json"
)
default_camera_config_file = os.path.join(
    REAL_KAM_REPO_DIR, "src/cfg", system_name, "default_camera_configurations.json"
)
if not os.path.isfile(camera_config_filename):
    try:
        wxpython_gui.utils.make_path(camera_config_filename, from_file=True)
        with open(default_camera_config_file, "r") as infile:
            with open(camera_config_filename, "w") as outfile:
                outfile.write(infile.read())
    except Exception as e:
        print(e)
try:
    with open(camera_config_filename, "r") as input_file:
        CAMERA_PRESETS = json.load(input_file)
except Exception as e:
    print(e)
    CAMERA_PRESETS = {}
try:
    kv.delete_dict("/sys/camera_cfgs")  # clear stale presets from Redis
except Exception:
    pass

# --- Resolve, lowest precedence first ----------------------------------------
SYS_ARCH = {}
deep_merge(SYS_ARCH, DEFAULT_STATE)  # 1. factory defaults
deep_merge(SYS_ARCH, CACHE_STATE)    # 2. operator's last session
deep_merge(SYS_ARCH, REDIS_LIVE)     # 3. live runtime values
# 4. config.yaml OWNS the static keys it declares: replace rather than merge, so
#    a key removed from the yaml (e.g. a dropped channel) can't survive from a
#    lower tier. "arch" mixes static + mutable session subkeys, so only its
#    static subkeys are replaced.
for key, val in USER_CFG.items():
    if key != "arch":
        SYS_ARCH[key] = val
SYS_ARCH.setdefault("arch", {})
for sub, val in USER_CFG.get("arch", {}).items():
    SYS_ARCH["arch"][sub] = val
SYS_ARCH["camera_cfgs"] = CAMERA_PRESETS

# Publish to Redis under /sys. First drop the static subtrees so keys removed
# from config.yaml don't linger (put only sets keys, never deletes); the update
# then republishes the authoritative values. arch.hosts is the only static dict
# under "arch" -- the rest of /sys/arch holds mutable state we must not wipe.
for key, val in USER_CFG.items():
    if key != "arch" and isinstance(val, dict):
        try:
            kv.delete_dict("/sys/%s" % key)
        except Exception:
            pass
try:
    kv.delete_dict("/sys/arch/hosts")
except Exception:
    pass
SYS_CFG = Cfg(ns="/sys")
SYS_CFG.update(SYS_ARCH)


# Keys excluded from the saved session state: YAML-owned (static), camera
# presets, and live per-host / camera Redis echoes.
_STATIC_TOP_KEYS = set(USER_CFG.keys())
_STATIC_ARCH_KEYS = set(USER_CFG.get("arch", {}).keys())
_HOST_KEYS = set(USER_CFG.get("arch", {}).get("hosts", {}).keys())
_RUNTIME_ONLY_KEYS = {"actual_geni_params", "camera_cfgs"}


def extract_state(cfg):
    """Return only the operator-mutable session state from ``cfg``."""
    state = {}
    for key, val in cfg.items():
        # arch is YAML-owned but mixes static and mutable subkeys, so split it.
        if key == "arch" and isinstance(val, dict):
            arch_state = {k: v for k, v in val.items() if k not in _STATIC_ARCH_KEYS}
            if arch_state:
                state["arch"] = arch_state
            continue
        if key in _STATIC_TOP_KEYS or key in _HOST_KEYS or key in _RUNTIME_ONLY_KEYS:
            continue
        state[key] = val
    return state


def save_config_settings():
    """Snapshot the mutable session state to the on-disk cache.

    Live values already stream into Redis via ``Cfg.__setitem__``; this just
    persists the subset so it survives a reboot or Redis flush.
    """
    state = extract_state(SYS_CFG)
    print("Saving config settings (session state).")
    with open(config_filename, "w") as output_file:
        json.dump(state, output_file, indent=4, sort_keys=True)
# =================== FINISHED CONFIG RESOLUTION =======================


# =================== DEFINE GLOBALS ===============================
# Need a vanilla one for binary insert
ros_immediate = rospy.Duration(nsecs=1)
# Instantiate CvBridge
bridge = CvBridge()

TEXTCTRL_GRAY = (255, 23, 23)
TEXTCTRL_WHITE = (255, 255, 255)
TEXTCTRL_DARK = (20, 20, 20)
APP_GRAY = (220, 218, 213)  # Default application background
FLAT_GRAY = (200, 200, 200)
DISABLED_GRAY = (150, 150, 150)  # Darker fill for disabled input fields
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


LICENSE_STR = (
    "Licensed under the Apache License, Version 2.0 (the \"License\"); "
    "you may not use this file except in compliance with the License. "
    "You may obtain a copy of the License at "
    "http://www.apache.org/licenses/LICENSE-2.0"
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


def pull_gui_state():
    """Refresh the in-memory arch state from the live Redis values."""
    print("Pulling gui state")
    try:
        live_arch = kv.get_dict("/sys/arch")
    except Exception as e:
        print(e)
        return
    deep_merge(SYS_CFG["arch"], live_arch)


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
            json.dump(
                SYS_CFG["camera_cfgs"][curr_cfg], outfile, indent=4, sort_keys=True
            )
            print("Saved sys config: {}".format(fname))
        return dirname
    else:
        return camera_config_filename


def format_status(
    timeval=None,
    num_dropped=None,
    exposure_us=None,
    gain=None,
    dt=0,
    fps=None,
    chan=None,
    total=None,
    processed=None,
):
    # type: (datetime.datetime, int, int, int, float, float, str) -> str
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

    if num_dropped is None:
        drop_str = "? dropped"
    elif num_dropped < 10000:
        drop_str = "{} dropped".format(num_dropped)
    else:
        drop_str = "{:.0e} dropped".format(num_dropped)
    gain_str = (
        "ISO: ?"
        if chan == "rgb" and gain is None
        else "Gain: ?"
        if gain is None
        else ("ISO: {}" if chan == "rgb" else "Gain: {}").format(gain)
    )
    fps_str = "? fps" if fps is None else "{:4.2f} fps".format(fps)
    if total is not None and processed is not None:
        # display N/N, even if internally it's N-1/N
        if processed < total:
            total = total - 1 if total > 0 else total
        drop_str += " | DB: {}/{}".format(processed, total)
    processed_str = "" if processed is None else "{}".format(processed)
    expo_str = (
        "Exp: ? ms"
        if exposure_us is None
        else "Exp: {:0.2f} ms".format(float(exposure_us) * 1e-3)
    )
    if chan == "ir":
        fmt = "{fps}\n{drop}"
        out = fmt.format(fps=fps_str, drop=drop_str)
    else:
        fmt = "{gain} | {fps} | {expo}\n{drop}"
        out = fmt.format(
            time=time_str, gain=gain_str, fps=fps_str, expo=expo_str, drop=drop_str
        )
    return out


def channel_format_status(fov, chan, timeval=None, dt=0):
    driver = "%s_driver" % chan
    # get current host from fov
    host = host_from_fov(fov)
    a = "actual_geni_params"
    param_ns = "/".join(["", "sys", a, host, chan])
    drop_ns = "/".join(["", "sys", "arch", host, chan, "dropped"])
    dropped_val = kv.get(drop_ns, None)
    num_dropped = int(dropped_val) if dropped_val is not None else None
    fps_ns = "/".join(["", "sys", "arch", host, chan, "fps"])
    fps_val = kv.get(fps_ns, None)
    fps = float(fps_val) if fps_val is not None else None
    exposure_us = None
    gain = None
    total = None
    processed = None
    if chan == "uv":
        gain = kv.get(param_ns + "/GainValue", None)
        exposure_us = kv.get(param_ns + "/ExposureValue", None)
    elif chan == "rgb":
        iso = kv.get(param_ns + "/ISO", None)
        if iso is not None:
            gain = int(float(iso))
        shutter = kv.get(param_ns + "/Shutter_Speed", None)
        if shutter is not None:
            # convert float point seconds to us
            exposure_us = float(shutter) * 1e6
        total = kv.get("/sys/" + host + "/p1debayerq/total", None)
        total = int(total) if total is not None else None
        processed = kv.get("/sys/" + host + "/p1debayerq/processed", None)
        processed = int(processed) if processed is not None else None
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
    # Skip stale hosts left in redis from other systems.
    for host in filter_hosts_by_system(hosts.keys()):
        if fov == hosts[host]["fov"]:
            return host
    raise KeyError("FOV not found: '{}'".format(fov))


def get_detector_pipefile(host, sys_cfg=None):
    # type: (str, Optional[str]) -> Optional[str]
    """Resolve a host's detector pipefile from the active camera config.

    The pipefile is defined per-FOV in camera_cfgs as "<fov>_sys_pipe"; it is
    read from there directly rather than duplicated into per-host state.
    Returns None when unset or invalid.
    """
    if sys_cfg is None:
        sys_cfg = SYS_CFG["arch"].get("sys_cfg")
    try:
        fov = SYS_CFG["arch"]["hosts"][host]["fov"]
        pipe = SYS_CFG["camera_cfgs"][sys_cfg]["{}_sys_pipe".format(fov)]
    except KeyError:
        return None
    return pipe if (pipe and pipe != "null") else None
