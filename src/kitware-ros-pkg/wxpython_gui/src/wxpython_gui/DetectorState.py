from __future__ import division, print_function
import time
from enum import Enum
from wxpython_gui.cfg import SYS_CFG, kv
from wxpython_gui.utils import diffpair


class EPodStatus(Enum):
    Unknown = 'unknown'
    Pending = 'pending'
    Running = 'running'
    Succeeded = 'succeeded'
    Failed = 'failed'
    Stalled = 'stalled'
    Off = 'off'

    def is_ok(self):
        return self.value in ['pending', 'running', 'succeeded', 'stalled']

    def is_transitioning(self):
        return self.value in ['pending', 'succeeded']

class DetectorState(object):
    def __init__(self, kv, hosts, short_thresh=2.0, med_thresh=10.0, long_thresh=120.0):
        # type: (ImplEnvoy, List[str], float, float) -> None
        self.kv = kv
        self.hosts = hosts
        self.short_thresh = short_thresh
        self.med_thresh = med_thresh
        self.long_thresh = long_thresh
        self.desired = {h: None for h in hosts}  # type: Dict[str, Optional[EPodStatus]]
        self.actual = {h: None for h in hosts}  # type: Dict[str, Optional[EPodStatus]]
        self.health = {h: {} for h in hosts}  # type: Dict[str, dict]
        self.command_pipefile = {h: None for h in hosts}  # type: Dict[str, Optional[str]]
        self.last_frame_bump = {h: time.time() for h in hosts}  # type: Dict[str, Optional[float]]
        self.last_healthy_time = {h: None for h in hosts}  # type: Dict[str, Optional[float]]
        self.last_frame = {h: None for h in hosts}  # type: Dict[str, Optional[int]]
        self.dt = {h: None for h in hosts}  # type: Dict[str, Optional[float]]

    def set_det_attr_state(self, host, attr, state):
        # type: (str, str, EPodStatus) -> EPodStatus
        v = state.value
        SYS_CFG[host]["detector"][attr] = v
        return self.get_det_attr_state(host, attr)

    def get_det_attr_state(self, host, attr):
        # type: (str, str) -> EPodStatus
        key = '/sys/{}/detector/{}'.format(host, attr)
        ov = self.kv.get(key)
        try:
            return EPodStatus(ov)
        except ValueError:
            return EPodStatus.Unknown

    def set_desired(self, host, state):
        # type: (str, EPodStatus) -> EPodStatus
        status = self.set_det_attr_state(host, 'desired', state)
        self.desired[host] = status
        now = time.time()
        SYS_CFG[host]["detector"]["last_change_desired"] = now
        self.last_change_desired = now
        return status

    def bump(self, host):
        now = time.time()
        self.last_frame_bump[host] = now

    def decide_is_running(self, host):
        # type: (str) -> EPodStatus
        """Decide whether to call the detector running or not, based on available information.
        """
        now = time.time()
        if now < (self.last_frame_bump[host] + self.short_thresh):
            if not self.pipe_matches(host):
                # the pipefiles don't match, so report as bad
                return EPodStatus.Failed
            else:
                return EPodStatus.Running
        elif now < (self.last_frame_bump[host] + self.med_thresh):
            return EPodStatus.Stalled
        elif now < (self.last_change_desired + self.long_thresh):
            return EPodStatus.Pending
        else:
            return EPodStatus.Failed

    def decide_status(self, host, desired):
        # type: (str, EPodStatus) -> EPodStatus
        """
        decides if the detector is doing what it ought to.

        """
        # it's doing the thing
        if desired in [EPodStatus.Off, EPodStatus.Unknown]:
            return desired

        elif desired is EPodStatus.Running:
            running = self.decide_is_running(host)
            if running.is_ok():
                return running
            return EPodStatus.Failed
        else:
            raise ValueError("{}".format(desired))


    def pipe_matches(self, host):
        try:
            pipe1 = self.health[host]['pipefile']
            pipe2 = self.command_pipefile[host]
            return pipe1 == pipe2
        except KeyError:
            return False

    def sync(self):
        # type: () -> None
        now = time.time()
        for host in self.hosts:
            # Reported health from the detectors
            try:
                health = self.kv.get('/{}/detector/health//'.format(host))
                cmdpipef = SYS_CFG[host]["detector"]["pipefile"]
            except KeyError as e:
                continue
            try:
                health = health[""]
            except KeyError:
                pass
            try:
                lht = health['time']
            except:
                lht = 0.0
            try:
                lframe = health['frame']
            except:
                lframe = 0
            dframe = diffpair(lframe, self.last_frame[host])
            if dframe:
                self.bump(host)
            desired = self.get_det_attr_state(host, 'desired')
            desired = desired or EPodStatus.Unknown
            actual = self.decide_status(host, desired)
            self.desired[host] = desired
            self.set_det_attr_state(host, 'actual', actual)
            self.health[host] = health
            self.command_pipefile[host] = cmdpipef
            self.last_frame[host] = lframe
            self.actual[host] = actual


def set_detector_state(system, host, desired):
    # type: (str, EPodStatus) -> None
    assert host in SYS_CFG["arch"]['hosts']
    detector_state.set_desired(host, desired)
    if desired is EPodStatus.Running:
        verb = 'up'
    elif desired is EPodStatus.Off:
        verb = 'down'
    else:
        raise ValueError("not valid: {}".format(desired))
    system.run_command("detector", host, verb)

detector_state = DetectorState(kv, SYS_CFG["arch"]["hosts"])
