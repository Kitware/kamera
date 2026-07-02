#! /usr/bin/python

import os
import threading
import time

import rclpy
from rclpy.node import Node

from roskv.impl.redis_envoy import RedisEnvoy
from custom_msgs.srv import CamGetAttr, CamSetAttr
from roskv.util import filter_hosts_by_system

try:
    from phase_one.srv import GetPhaseOneParameter, SetPhaseOneParameter
except ImportError:
    # Phase One driver is only installed on systems flying a Phase One RGB
    # camera; nayak/taiga use the prosilica driver instead.
    GetPhaseOneParameter = SetPhaseOneParameter = None

p1setsrv = "set_phaseone_parameter"
p1getsrv = "get_phaseone_parameter"
setsrv = "set_camera_attr"
getsrv = "get_camera_attr"

P1_SHUTTER_MODE_LABELS = {"LS": "1", "ES": "2"}


def normalize_p1_shutter_mode(val):
    """Map Phase One shutter mode labels and numeric values to '1' or '2'."""
    if val in (None, ""):
        return None
    s = str(val).strip()
    if s in P1_SHUTTER_MODE_LABELS:
        return P1_SHUTTER_MODE_LABELS[s]
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return s


class CamParamMonitor(Node):
    """ A class to monitor state that is set in Redis,
        and the value reported by the cameras,
        and attempts to sync the 2.
    """
    def __init__(self):
        super().__init__("cam_param_monitor")
        redis_host = os.environ["REDIS_HOST"]
        print("Redis host: %s" % redis_host)
        self.envoy = RedisEnvoy(redis_host, client_name="cam_param_monitor")
        self.hosts = filter_hosts_by_system(
            self.envoy.get("/sys/arch/hosts").keys()
        )
        self.modes = self.envoy.get("/sys/channels").keys()
        self._clients = {}
        print("hosts: ")
        print(self.hosts)
        print("modes: ")
        print(self.modes)

    def start_threads(self):
        t = threading.Thread(target=self.check_cam_params)
        t.daemon = True
        t.start()

    def _get_client(self, topic, srv_type):
        client = self._clients.get(topic)
        if client is None:
            client = self.create_client(srv_type, topic)
            self._clients[topic] = client
        return client

    def _call(self, topic, srv_type, wait_time, request):
        """Synchronous service call; returns None on unavailability/timeout."""
        client = self._get_client(topic, srv_type)
        if not client.wait_for_service(timeout_sec=wait_time):
            return None
        future = client.call_async(request)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                client.remove_pending_request(future)
                return None
            time.sleep(0.01)
        return future.result()

    def get_param_val(self, host, mode, param, requested_val):
        driver = "%s_driver" % mode
        wait_time = 0.1
        if mode == "ir":
            topic = '/'.join(['', host, mode, getsrv])
            resp = self._call(topic, CamGetAttr, wait_time,
                              CamGetAttr.Request(name=param))
        elif mode == "uv":
            topic = '/'.join(['', host, mode, driver, getsrv])
            resp = self._call(topic, CamGetAttr, wait_time,
                              CamGetAttr.Request(name=param))
        elif mode == "rgb":
            if GetPhaseOneParameter is None:
                topic = '/'.join(['', host, mode, driver, getsrv])
                resp = self._call(topic, CamGetAttr, wait_time,
                                  CamGetAttr.Request(name=param))
            else:
                topic = '/'.join(['', host, mode, driver, p1getsrv])
                resp = self._call(topic, GetPhaseOneParameter, wait_time,
                                  GetPhaseOneParameter.Request(name=param))
        else:
            return
        if resp is None:
            return
        getsrv_val = None
        dtype = None
        try:
            getsrv_val = resp.value
            if getsrv_val == "error":
                self.get_logger().info(str(resp))
                self.get_logger().info(topic)
                self.get_logger().error("|GET| Failed to get parameter %s!" % param)
                return
            if mode == "rgb" and GetPhaseOneParameter is not None:
                # Phase one params have the type in the return string
                getsrv_val = ''.join(getsrv_val.split(' ')[1:])
                # A random s is sometimes in shutter speed
                if "s" in getsrv_val and param != "Shutter Mode":
                    getsrv_val = getsrv_val[:-1]
                if param == "Shutter Mode":
                    getsrv_val = normalize_p1_shutter_mode(getsrv_val)
                    requested_val = normalize_p1_shutter_mode(requested_val)
                elif isinstance(requested_val, float):
                    try:
                        getsrv_val = float(getsrv_val)
                    except Exception:
                        num, den = map(int, getsrv_val.split('/'))
                        getsrv_val = float(num / den)
                elif isinstance(requested_val, int):
                    try:
                        getsrv_val = float(getsrv_val)
                        getsrv_val = int(getsrv_val)
                    except Exception:
                        num, den = map(int, getsrv_val.split('/'))
                        getsrv_val = float(num / den)
            else:
                dtype = resp.dtype
                if isinstance(requested_val, float):
                    getsrv_val = float(getsrv_val)
                elif isinstance(requested_val, int):
                    getsrv_val = int(getsrv_val.rstrip('\x00'))
        except Exception as e:
            self.get_logger().warning(
                f"|GET| value coercion failed on {param}, resp value: %s" % resp.value)
            self.get_logger().error(str(e))
            return
        if getsrv_val is None:
            return
        if mode == "rgb":
            param = '_'.join(param.split(' '))
        self.envoy.set("/sys/actual_geni_params/%s/%s/%s"
                       % (host, mode, param), getsrv_val)
        if param == "GainValue" or param == "ExposureValue"\
                or param == "ISO" or param == "Shutter_Speed"\
                or param == "Sensor_Temperature":
            # Special case for a "read-only" value
            return
        if mode == "rgb":
            # Convert back to phase one spaces
            param = ' '.join(param.split('_'))
        # Value has changed, will return value to set
        if getsrv_val != requested_val:
            print("Param: %s, getsrv_val: %s, requested_val: %s" %
                  (param, getsrv_val, requested_val))
            self.get_logger().warning("Setting parameter %s on %s because it differs."
                                      % (param, host + "/" + mode))
            # Return real value to set
            return str(param), str(requested_val), dtype
        return None

    def set_params(self, host, mode, params_to_set, requested_params):
        # Set all params that differ from those in redis db
        driver = f"{mode}_driver"
        use_p1 = mode == "rgb" and SetPhaseOneParameter is not None
        if mode == "ir":
            topic = '/'.join(['', host, mode, setsrv])
        elif use_p1:
            topic = '/'.join(['', host, mode, driver, p1setsrv])
        elif mode in ("uv", "rgb"):
            topic = '/'.join(['', host, mode, driver, setsrv])
        else:
            return
        if use_p1:
            req_str = ','.join([f"{name}={v}" for name, (v, d) in
                                params_to_set.items()])
            if len(params_to_set):
                self.get_logger().info("|SET| Setting the following params on P1:")
                print(req_str)
                resp = self._call(topic, SetPhaseOneParameter, 1.0,
                                  SetPhaseOneParameter.Request(parameters=req_str))
                if resp is None:
                    self.get_logger().warning(
                        "|SET| Failed to set params for system %s camera %s." %
                        (host, mode))
        else:
            for name, (v, d) in params_to_set.items():
                self.get_logger().info("|SET| Parameters: {} {} {}".format(name, v, d))
                resp = self._call(topic, CamSetAttr, 1.0,
                                  CamSetAttr.Request(name=name, value=v, dtype=d or ""))
                if resp is None:
                    self.get_logger().warning(
                        "|SET| Failed to set params for system %s camera %s." %
                        (host, mode))
        return

    def check_cam_params(self):
        # Check all params once every 3 s
        while rclpy.ok():
            tic = time.time()
            for host in self.hosts:
                for mode in self.modes:
                    try:
                        requested_params = self.envoy.get_dict(
                            "/sys/requested_geni_params/%s/%s" % (host, mode))
                    except KeyError as e:
                        print(e)
                        continue
                    params_to_set = {}
                    for param, requested_val in requested_params.items():
                        param = ' '.join(param.split('_'))
                        if param == "ExposureAuto":
                            # Write-only for some reason
                            continue
                        ret = self.get_param_val(host, mode, param, requested_val)
                        if ret is not None:
                            params_to_set[ret[0]] = (ret[1], ret[2])
                    self.set_params(host, mode, params_to_set,
                                    requested_params)
            self.get_logger().info("Time to set parameters was %0.4fs." % (time.time() - tic))
            time.sleep(3.0)


def main(args=None):
    rclpy.init(args=args)
    cpm = CamParamMonitor()
    cpm.start_threads()
    try:
        rclpy.spin(cpm)
    except KeyboardInterrupt:
        pass
    finally:
        cpm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
