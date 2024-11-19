#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import re
import errno
import datetime

# import threading
import time
import json
import yaml

from roskv.impl.redis_envoy import RedisEnvoy, StateService


try:
    import rospy
    import genpy

    # from profilehooks import timecall
    from std_msgs.msg import Header
    from std_msgs.msg import UInt64 as MsgUInt64
    from std_msgs.msg import String as MsgString

    from msgdispatch.archive import ArchiveSchemaDispatch
    from custom_msgs.srv import SetArchiving, AddToEventLog
    from custom_msgs.msg import GSOF_INS, GSOF_EVT

except ImportError as exc:
    import sys

    print(
        "cannot import rospy or messages. if this is not a test environment, this is bad!",
        file=sys.stderr,
    )
    if not os.environ.get("IGNORE_ROS_IMPORT", False):
        raise exc

# from custom_msgs.srv import EraseDataDisk

PAT_BRACED = re.compile(r"\{(\w+)\}")
PAT_DOUBLE_SLASH = re.compile(r"//")


def strip_double_slash(s):
    while re.search(PAT_DOUBLE_SLASH, s):
        s = re.sub(PAT_DOUBLE_SLASH, "/", s)
    return s


def get_template_keys(tmpl):
    return re.findall(PAT_BRACED, tmpl)


def conformKwargsToFormatter(tmpl, kwargs):
    # type: (str, dict) -> dict
    """Return a dictionary which can be used to format the string. Fill any missing
    values with placeholders, e.g. {"keyname": "(keyname)"}"""
    required_keys = set(get_template_keys(tmpl))
    missing_keys = required_keys.difference(kwargs)
    fmt_dict = {k: v for k, v in kwargs.items() if k in required_keys}
    fmt_dict.update({k: "({})".format(k) for k in missing_keys})
    return fmt_dict


def msg_as_dict(msg):
    if isinstance(msg, genpy.rostime.TVal):
        return msg.to_sec()
    elif isinstance(msg, genpy.message.Message):
        return {str(k): msg_as_dict(getattr(msg, k)) for k in msg.__slots__}
    elif isinstance(msg, dict):
        return {str(k): v for k, v in msg.items()}
    return msg


def pathsafe_timestamp(now=None, show_micros=False, show_millis=False):
    # type: (Optional[datetime.datetime], bool, bool) -> str
    """
    Filesystem name safe timestamp string.
    :param now: datetime object of time to process
    :param show_micros: - Format with microseconds
    :return:
    """
    now = now or datetime.datetime.now()
    show_micros = show_millis or show_micros
    sfmt = "%Y%m%d_%H%M%S{}".format(".%f" * show_micros)
    s = now.strftime(sfmt)
    if show_millis:
        s = s[:-3]
    return s


def make_path(path, from_file=False, verbose=False):
    # type: (str, bool, bool) -> str
    """
    Make a path, ignoring already-exists error. Python 2/3 compliant.
    Catch any errors generated, and skip it if it's EEXIST.
    :param path: Path to create
    :type path: str, pathlib.Path
    :param from_file: if true, treat path as a file path and create the basedir
    :return: Path created or exists
    """
    path = str(path)  # coerce pathlib.Path
    if path == "":
        raise ValueError("Path is empty string, cannot make dir.")

    if from_file:
        path = os.path.dirname(path)
    try:
        os.makedirs(path)
        if verbose:
            print("Created path: {}".format(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        if verbose:
            print("Tried to create path, but exists: {}".format(path))

    return path


class _Fmt(object):
    filename_template = "{proj}_{flight}{fabr}{ts}{note}_{mode}.{ext}"  # fabr is field abridged/abbreviated

    fov_aliases = {
        "center_view": "center_view",
        "left_view": "left_view",
        "right_view": "right_view",
        "default_view": "default_view",
        "center": "center_view",
        "left": "left_view",
        "right": "right_view",
        "global": "global",
        "ins": "ins",
        "meta": "meta",
    }

    @staticmethod
    def msg_as_dict_headless(msg):
        """unused I think"""
        dd = {k: getattr(msg, k) for k in msg.__slots__}
        dd.pop("header", None)
        return dd

    @staticmethod
    def get_image_msg_meta(msg):
        fields = ["height", "width", "encoding", "is_bigendian", "step"]
        d = {"header": dict()}
        for field in fields:
            x = getattr(msg, field, None)
            d.update({field: str(x)})
        fields = ["seq", "stamp", "frame_id"]
        for field in fields:
            x = getattr(msg.header, field, None)
            d["header"].update({field: str(x)})
        return d

    @staticmethod
    def get_field_abr(field):
        # type: (str) -> str
        """Ambigous abbreviations, this use should be discouraged"""
        return field[0].upper()

    # @pysnooper.snoop()
    @staticmethod
    def fmt_filename(
        proj, flight, ts, field=None, mode="{mode}", ext="{ext}", note=None
    ):
        # type: (str, str, str, str, str, str, str) -> str
        if field is None or field == "log":
            fabr = ""
        else:
            fabr = "_" + _Fmt.get_field_abr(field)

        if note is None:
            note = ""
        else:
            note = "_" + note

        if ts:
            ts = "_" + ts

        out = fmt.filename_template.format(
            proj=proj, flight=flight, fabr=fabr, note=note, ts=ts, mode=mode, ext=ext
        )
        return out

    @staticmethod
    def fmt_log_filename():
        pass


fmt = _Fmt()


class ArchiverBase(object):

    def __init__(self, agent_name=None, bytes_halt_archiving=1e9, verbosity=0):
        """
        Class for managing the archiving of data coming from the system.
        By convention, paths are '/' terminated.
        """
        node_host = rospy.get_namespace().strip("/")
        cam_fov = rospy.get_param(
            os.path.join("/cfg", "hosts", node_host, "fov"), "node"
        )

        self.node_host = node_host
        self._redis_host = os.environ.get("REDIS_HOST", "nuvo0")

        ## deprecated
        self.verbosity = verbosity
        self._name_system = rospy.get_param("/system_name", "default_system")
        self._name_sync = "sync"
        self._name_ins = "ins"
        self._name_meta = "meta"
        self._init_time = datetime.datetime.now()
        self._init_time_str = pathsafe_timestamp(self._init_time)

        self.archive_service = None
        self.log_service = None
        self.erase_service = None

        self.bytes_halt_archiving = bytes_halt_archiving
        rospy.Subscriber("/ins", GSOF_INS, self.ins_callback)
        rospy.Subscriber("/event", GSOF_EVT, self.evt_callback)

        self.pub_diskfree = rospy.Publisher("disk_free_bytes", MsgUInt64, queue_size=1)
        self.counter = 0
        self.image_formats = {}
        default_types = {
            "rgb": "jpg",
            "ir": "tif",
            "uv": "jpg",
            "evt": "json",
            "ins": "json",
        }
        for chan in ["rgb", "uv", "ir", "evt", "ins"]:
            self.image_formats[chan] = rospy.get_param(
                os.path.join("/cfg/file_formats", chan), default_types[chan]
            )

        self.last_ins = GSOF_INS()
        self.latch_ins = GSOF_INS()
        print("\n\n VERSION CHECK 1 \n")

        self.envoy = RedisEnvoy(self._redis_host, client_name=node_host + "-nexus")
        self.state_service = StateService(self.envoy.kv, node_host, "nexus")

        # Get Redis Params
        arch = self.envoy.kv.get("/sys/arch")
        self.set_arch(arch)
        self._is_archiving = self.envoy.kv.get("/sys/arch/is_archiving")
        self._project = self.envoy.get("/sys/arch/project")
        fl = self.envoy.get("/sys/arch/flight")
        fl_number = "".join([d for d in str(fl) if d.isdigit()])
        self._flight = "fl{:0>2}".format(fl_number)
        self._effort = self.envoy.get("/sys/arch/effort")
        self._field = os.environ.get("CAM_FOV", "default_view")
        self._collection_mode = self.envoy.get("/sys/collection_mode")
        self._template = self.envoy.get("/sys/arch/template")
        self._base = self.envoy.kv.get("/sys/arch/base")
        self._data_mount_point = self.envoy.kv.get(
            "/sys/arch/base", "/mnt/archiver_default"
        )
        self.disk_check(self._base, every_nth=1)

        if verbosity > 0:
            print("Archive Manager: Verbosity:", verbosity)

    def __str__(self):
        myid = str(id(self))
        return "Archiver {}/{}:{}".format(
            self.project, self.flight, myid[:2] + myid[-2:]
        )

    @property
    def is_archiving(self):
        return self._is_archiving

    @property
    def project(self):
        return self._project

    @property
    def flight(self):
        return self._flight

    @property
    def fov(self):
        myfov = fmt.fov_aliases.get(self._field)
        return myfov

    @property
    def arch_dict(self):
        arch = self.envoy.get_dict("/sys/arch")
        self.set_arch(arch)
        return arch

    def ins_callback(self, msg):
        self.last_ins = msg

    def evt_callback(self, msg):
        self.latch_ins = self.last_ins

    def set_arch(self, arch_dict):
        self._arch_dict = arch_dict
        self._project = str(arch_dict["project"])
        self._effort = str(arch_dict["effort"])
        fl_number = "".join([d for d in str(arch_dict["flight"]) if d.isdigit()])
        self._flight = "fl{:0>2}".format(fl_number)
        self._is_archiving = arch_dict["is_archiving"]
        self._template = arch_dict["template"]
        self._base = arch_dict["base"]
        self._sys_cfg = arch_dict["sys_cfg"]

    def get_arch(self):
        return {
            "project": self._project,
            "effort": self._effort,
            "flight": self._flight,
            "is_archiving": self._is_archiving,
        }

    @property
    def now_gps(self):
        # type: () -> datetime.datetime
        """Returns GPS time of last received GPS packet"""
        return datetime.datetime.utcfromtimestamp(self.last_ins.time)

    def fmt_filename(self, ts, field=None, mode="{mode}", ext="{ext}", note=None):
        # type: (str, str, str, str, str) -> str
        return fmt.fmt_filename(
            self._project,
            self._flight,
            ts=ts,
            field=field,
            mode=mode,
            ext=ext,
            note=note,
        )

    @property
    def data_mount_point(self):
        # type: () -> str
        return self._data_mount_point

    def nomsg_set_archiving(self, state):
        """
        Set archiving state by a function call. This is discouraged, ideally a message should be used
        This is basically so gis archiving can be enabled independently
        :param state: set archiving state
        :return:
        """
        self._is_archiving = state

    def advertise_services(
        self,
        namespace=None,
        name_archive="set_archiving",
        name_log="add_to_event_log",
        name_erase="erase_data_disk",
    ):
        # type: (str, str, str, str) -> None

        if namespace is None:
            namespace = rospy.get_namespace()

        else:
            if namespace not in ["/", "~"] and namespace[-1] != "/":
                namespace += "/"
        name_archive = namespace + name_archive
        name_log = namespace + name_log
        name_erase = namespace + name_erase
        self.archive_service = rospy.Service(
            name_archive, SetArchiving, self.call_set_archiving
        )
        self.log_service = rospy.Service(
            name_log, AddToEventLog, self.call_add_to_event_log
        )
        # self.erase_service = rospy.Service(name_log, EraseDataDisk, self.call_erase_disk)
        rospy.loginfo("Subscribing {} to {}".format(namespace, name_archive))
        rospy.loginfo("Subscribing {} to {}".format(namespace, name_log))

    #        self.erase_service = rospy.Service(name_log, EraseDataDisk, self.call_erase_disk)

    def fmt_flight_path(self):
        # type: () -> str
        now = datetime.datetime.utcfromtimestamp(time.time())
        fpath = self.fmt_sync_path(now)
        flight_dir = os.path.dirname(os.path.dirname(fpath))
        return flight_dir

    @staticmethod
    def _fmt_sync_path(fn_template, fmt_dict, cam_fov, timestamp, note=None):
        time_long = pathsafe_timestamp(now=timestamp, show_micros=True)
        fmt_dict = fmt_dict.copy()
        # leave mode and ext to be formatted by the image dump
        fmt_dict.update(
            dict(
                cf=cam_fov[0].upper(),
                time=time_long,
                cam_fov=cam_fov,
                mode="{mode}",
                ext="{ext}",
            )
        )
        fmt_dict = conformKwargsToFormatter(fn_template, fmt_dict)
        filename = fn_template.format(**fmt_dict)
        return strip_double_slash(filename)

    def fmt_sync_path(self, timestamp, note=None):
        # type: (datetime.datetime, str) -> str
        """
        Generate the path for dumping synchronized data (including timestamp)
        General format is:
        $base/$projYYYY/$flXX/$view/$timestamp/Proj_fl##_pos_yyyymmdd_hhmmss.ms_rgb
        Args:
            timestamp: datetime object of time to use
            note: this isn't really used any more, it was used to flag a filename as using system time.

        Returns:
            Fully qualified path with {mode} and {ext} substitution points
        """
        fn_template = self._template
        fmt_dict = self.arch_dict.copy()
        # leave mode and ext to be formatted by the image dump
        cam_fov = self._field
        return self._fmt_sync_path(fn_template, fmt_dict, cam_fov, timestamp, note)

    def fmt_async_path(self, field, timestamp, note=None):
        # type: (str, datetime.datetime, str) -> str
        """
        Generate the path for dumping asynchronous data (such as long)
        General format is:
        $base/$projYYYY/$flXX/$view/Proj_fl##_pos_yyyymmdd_hhmmss.ms_rgb
        Args:
            field: field-of-view or general-field
            timestamp: datetime object of time to use
            note: additional string to affix to timestamp

        Returns:
            Fully qualified path with {mode} and {ext} substitution points
        """
        fn_template = self._template
        fmt_dict = self.arch_dict.copy()
        # leave mode and ext to be formatted by the image dump
        fmt_dict.update({"sys_cfg": ""})
        return self._fmt_sync_path(fn_template, fmt_dict, field, timestamp, note)

    def fmt_log_path(self, field):
        # type: (str) -> str
        """
        Generate the path for appending INDIVIDUAL logs to a singular file - probably should not use this
        General format is:
        $base/$projYYYY/$flXX/$view/Proj_fl##_pos_yyyymmdd_hhmmss.ms.json
        Args:
            field: field-of-view or general-field

        Returns:
            Fully qualified path with {mode} and {ext} substitution points
        """
        timestamp_short = pathsafe_timestamp(self._init_time, show_millis=True)
        # leave mode and ext to be formatted by the file dump
        fn_template = self._template
        fmt_dict = self.arch_dict.copy()
        # leave mode and ext to be formatted by the image dump
        fmt_dict.update({"sys_cfg": "", "cam_fov": ""})
        return self._fmt_sync_path(fn_template, fmt_dict, field, self._init_time, None)

    def fmt_flight_file_path(self, field=None):
        # type: (str) -> str
        """
        Generate the path for appending logs to a singular file per flight
        General format is:
        $base/$projYYYY/$flXX/Proj_fl##_pos_yyyymmdd_hhmmss.ms.txt

        Returns:
            Fully qualified path with {mode} and {ext} substitution points
        """
        # init_time_short = pathsafe_timestamp(self._init_time, show_millis=True)
        # leave mode and ext to be formatted by the file dump
        filename = self.fmt_filename("", field)

        # flight path now returns camera configuration, log should
        # be written to top level
        return os.path.join(os.path.dirname(self.fmt_flight_path()), filename)

    def fmt_meta_path(self, now=None):
        # type: (datetime.datetime) -> str
        return self.fmt_async_path(field="meta", timestamp=now)

    def get_meta_path_now(self):
        return self.fmt_async_path(field="meta", timestamp=datetime.datetime.now())

    def get_raw_ins_path(self, field="raw"):
        # type: (str) -> str
        """
        Timestamp is pegged to start of INS node (and consequently start of its archiver object)
        :return: Fully qualified path
        """
        filename = "ins_{}_{}.dat".format(field, self.init_time_str)
        meta = self.fmt_async_path(field=field, timestamp=None)
        basename = os.path.dirname(os.path.dirname(meta))
        return os.path.join(basename, "ins_raw", filename)

    def get_ins_path(self, timestamp=None):
        # type: (datetime.datetime) -> str
        """
        Parsed stream of ins data (not synced).
        /$basepath/$proj/$flight/ins/$ts_ins.yml
        :return: Fully qualified path
        """
        field = "ins"
        timestamp_long = pathsafe_timestamp(now=timestamp, show_micros=True)
        filename = self.fmt_filename(timestamp_long, field)
        return os.path.join(self.fmt_flight_path(), field, filename)

    def dump_json(self, data, time=None, mode="meta", make_dir=True):
        # type: (dict, datetime, str, bool) -> str
        # "{base}/{project}/fl{flight}/{cam_fov}_view/{project}_fl{flight}_{cf}_{time}_{mode}.{ext}"
        fn_template = self._template
        time_long = pathsafe_timestamp(now=time, show_micros=True)
        fmt_dict = self.arch_dict.copy()
        fmt_dict.update(
            dict(
                cf=self._field[0].upper(),
                time=time_long,
                cam_fov=self._field,
                mode=mode,
                ext="json",
            )
        )
        fmt_dict = conformKwargsToFormatter(fn_template, fmt_dict)
        filename = fn_template.format(**fmt_dict)
        if make_dir:
            make_path(filename, from_file=True)
        else:
            if not os.path.exists(os.path.dirname(filename)):
                rospy.logerr(
                    "Archiving directory not created yet and `make_dir` set to false. "
                    "Could not write: {}".format(filename)
                )
                return filename
        with open(filename, "w") as fp:
            json.dump(data, fp)
        return filename

    def dump_log_json(self, data):
        # type: (dict) -> str
        fn_template = self.fmt_log_path("meta")
        filename = fn_template.format(mode="meta", ext="json")
        make_path(filename, from_file=True)
        with open(filename, "a") as fp:
            json.dump(data, fp)
            fp.write("\n")
        print("wrote log: {}".format(filename))
        return filename

    def dump_log_yaml(self, data):
        # type: (dict) -> str
        """Write yaml-like text to a rolling log"""
        data.update({"gps_time": self.now_gps})
        fn_template = self.fmt_flight_file_path(field="log")
        filename = fn_template.format(mode="log", ext="txt")
        make_path(filename, from_file=True)
        with open(filename, "a") as fp:
            yaml.dump(data, fp)
            fp.write("\n---\n")
        print("wrote log: {}".format(filename))
        return filename

    def update_project_flight(self, project, flight, effort="", collection_mode="?"):
        if not project:
            rospy.logwarn("Missing project string, setting to default")
            project = "arch_core_svc_no_project"

        if not flight:
            rospy.logwarn("Missing flight string, setting to default")
            flight = "00"

        if not effort:
            rospy.logwarn("Missing effort string, setting to default")
            effort = "arch_core_svc_no_effort"

        self._project = project
        self._flight = "fl{:0>2}".format(flight)
        self._effort = effort
        self._collection_mode = collection_mode
        return project, flight, effort

    def call_set_archiving(self, req):
        # type: (SetArchiving) -> bool
        rospy.loginfo("!! DEPRECATED !! call_set_archiving v2: {}".format(req))
        return True

    def call_add_to_event_log(self, req):
        # type: (AddToEventLog) -> bool
        rospy.loginfo("maybe deprecating? call_add_to_event_log: {}".format(req))
        self.update_project_flight(
            req.project, req.flight, req.effort, req.collection_mode
        )
        now = pathsafe_timestamp(show_micros=True)
        data = {
            "note": req.note,
            "sys_time": now,
            "project": req.project,
            "flight": req.flight,
            "effort": req.effort,
            "collection_mode": req.collection_mode,
        }
        self.dump_log_yaml(data)
        return True

    def update_schema(self, msg):
        # type: (ArchiveSchemaDispatch) -> None
        rospy.loginfo("Set schema: \n{}".format(str(msg)))
        self._project = msg.project
        fl_number = "".join([d for d in msg.flight if d.isdigit()])
        self._flight = "fl{:0>2}".format(fl_number)

    @staticmethod
    def call_erase_disk(req):
        print(req)
        parent_host = rospy.get_namespace().strip("/")
        rospy.logwarn("Deleting disk on {}: ".format(parent_host))

    def disk_check(self, dirname, every_nth=8):
        """
        Run the disk check protocol and publish
        Runs only every nth call
        :param dirname:
        :param every_nth:
        :return:
        """
        rospy.loginfo("disk check on {} (every {}th)".format(dirname, every_nth))
        if every_nth > 1:
            self.counter += 1
            if self.counter % every_nth:
                return
        rospy.loginfo("checking!!!!!!!")
        stats = os.statvfs(dirname)
        bytes_free = stats.f_frsize * stats.f_bavail
        diskmsg = MsgUInt64()
        diskmsg.data = bytes_free
        self.pub_diskfree.publish(diskmsg)
        self.halt_if_disk_low(bytes_free)
        self.halt_if_unmounted()

    def halt_if_disk_low(self, bytes_free):
        if bytes_free < self.bytes_halt_archiving:
            self.nomsg_set_archiving(False)
            txt = "ERROR {}: Archiving shut off due to low disk".format(self.node_host)
            self.fail(txt)

    def halt_if_unmounted(self):
        # if not os.path.ismount(self._base): # this is neat but doesn't work in containers with mounts
        sentinel = os.path.join(self._base, ".flight_data_mounted")
        if not os.path.isfile(sentinel):
            self.nomsg_set_archiving(False)
            txt = (
                "ERROR {}: NAS mount failed. Restart KAMERA immediately. If this continues, ensure the "
                "file '{}' exists on the NAS".format(self.node_host, sentinel)
            )
            self.fail(txt)
        else:
            rospy.loginfo("Mount OK: {}".format(sentinel))

    def fail(self, txt):
        rospy.logwarn(txt)
        # if self.is_archiving:
        self.rawmsg_publish_err(txt)
        self.envoy.put("/sys/arch/is_archiving", 0)

    def rawmsg_publish_err(self, txt):
        msg = MsgString(txt)
        rospy.logerr_throttle(1, txt)
        self.rawmsg_pub = rospy.Publisher("/rawmsg", MsgString, queue_size=99)
        self.rawmsg_pub.publish(msg)

    @property
    def basepath(self):
        """
        This is the root path all other subdirectories stem from.
        :return:
        """
        return self._data_mount_point

    @property
    def init_time_str(self):
        return self._init_time_str

    @property
    def name_system(self):
        return self._name_system

    @property
    def name_sync(self):
        return self._name_sync

    @property
    def name_ins(self):
        return self._name_ins

    @property
    def name_meta(self):
        return self._name_meta

    @property
    def effort(self):
        return self._effort

    @property
    def collection_mode(self):
        return self._collection_mode


if __name__ == "__main__":
    raise NotImplementedError("Not for direct running at this time")
