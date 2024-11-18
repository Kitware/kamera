import re
import os
import errno
import time
from collections import OrderedDict

import rospy

def make_path(path, from_file=False, verbose=False):
    """
    Make a path, ignoring already-exists error. Python 2/3 compliant.
    Catch any errors generated, and skip it if it's EEXIST.
    :param path: Path to create
    :type path: str, pathlib.Path
    :param from_file: if true, treat path as a file path and create the basedir
    :return:
    """
    path = str(path)  # coerce pathlib.Path
    if path == '':
        raise ValueError("Path is empty string, cannot make dir.")

    if from_file:
        path = os.path.dirname(path)
    try:
        os.makedirs(path)
        if verbose:
            print('Created path: {}'.format(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        if verbose:
            print('Tried to create path, but exists: {}'.format(path))

def get_template_keys(tmpl):
    return re.findall(PAT_BRACED, tmpl)

def conformKwargsToFormatter(tmpl, kwargs):
    # type: (str, dict) -> dict
    required_keys = set(get_template_keys(tmpl))
    missing_keys = required_keys.difference(kwargs)
    fmt_dict = {k:v for k,v in kwargs.items() if k in required_keys}
    fmt_dict.update({k:'({})'.format(k) for k in missing_keys})
    return fmt_dict

def get_arch_path(key='/sys/arch/'):
    arch_dict = kv.get_dict(key)
    tmpl = arch_dict['base_template']
    fmt_dict = conformKwargsToFormatter(tmpl, arch_dict)
    return tmpl.format(**fmt_dict)

def verbose_call_service(srv, **kwargs):
    print('Calling {}\n{}'.format(srv.resolved_name, kwargs))
    return srv(**kwargs)

def omap(func, maybe_itr):
    # type: (Callable, Optional[Any]) -> Optional[Any]
    """Map over optional"""
    if maybe_itr is None:
        return None
    if isinstance(maybe_itr, string_types):
        return func(maybe_itr)
    elif isinstance(maybe_itr, Iterable):
        return lmap(func, maybe_itr)
    return func(maybe_itr)


def diffpair(a, b):
    if a is None or b is None:
        return None
    return a - b


def apply_ins_spoof(msg, spoof_dict):
    for k in spoof_dict:
        if k == 'time':
            continue
        try:
            val = spoof_dict[k]
            setattr(msg, k, val)
        except KeyError:
            pass

def check_default(provided_dict, default_dict):
    """ Any missing configuration values in <provided_dict>
        get filled in from <default_dict>
    """
    assert isinstance(default_dict, dict)
    assert isinstance(provided_dict, dict)
    for key in default_dict:
        if key not in provided_dict:
            provided_dict[key] = default_dict[key]
        elif isinstance(default_dict[key], dict):
            check_default(provided_dict[key], default_dict[key])
    return provided_dict

def save_cfg(cfg, fname):
    with open(fname, "w") as of:
        json.dump(cfg, of)

class MissedFrameStore(OrderedDict):
    """This keeps track of missed frames count and fps
    /center/rgb/missed
    """
    ok_size = 20
    zero_vals = {'missed': 0, 'fps': 0.0, 'ok_time': []}

    def cb_any_missed_frame(self, msg, topic):
        val = self.get(topic, 0)
        self[topic] = val + 1

    def cb_missed_fov_chan(self, msg, fov_chan):
        fov, chan = fov_chan
        topic = '/'.join(['', fov, chan, 'missed'])
        val = self.get(topic, 0)
        self[topic] = val + 1

    def cb_ok_img_fov_chan(self, msg, fov, chan):
        # type: (Any, str, str) -> None
        now = time.time()
        ok_topic = '/'.join(['', fov, chan, 'ok_time'])
        fps_topic = '/'.join(['', fov, chan, 'fps'])
        ok_times = self.get(ok_topic, [])
        if not isinstance(ok_times, list):
            print("oops, ok_times was corrupted on topic {}: {}".format(ok_topic, ok_times))
            ok_times = []
        ok_times.append(now)
        ok_times = ok_times[-MissedFrameStore.ok_size:]
        self[ok_topic] = ok_times
        if len(ok_times) < 2:
            self[fps_topic] = None
            return
        period_list = []
        for i, v in enumerate(ok_times[:-1]):
            dt = ok_times[i + 1] - v
            if dt:
                period_list.append(dt)
        mean_period = sum(period_list) / len(period_list)
        fps = 1.0 / mean_period

        self[fps_topic] = fps

    def get_fps(self, fov, chan):
        # type: (str, str) -> Optional[float]
        fps_topic = '/'.join(['', fov, chan, 'fps'])
        return self.get(fps_topic, None)

    def reset_count(self):
        for k in self.keys():
            tail = k.split('/')[-1]
            if tail != 'missed':
                continue
            v = MissedFrameStore.zero_vals[tail]
            self[k] = v
