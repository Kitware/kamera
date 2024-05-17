#! /usr/bin/python

import io
import os
import requests
import shapefile
import shapely
import shapely.geometry
import threading
import time

import rospy

from roskv.impl.redis_envoy import RedisEnvoy
from phase_one.srv import GetPhaseOneParameter, SetPhaseOneParameter
from custom_msgs.srv import CamGetAttr, CamSetAttr

HOSTS = ["cas0", "cas1", "cas2"]
MODES = ["ir", "rgb", "uv"]
p1setsrv = "set_phaseone_parameter"
p1getsrv = "get_phaseone_parameter"
setsrv = "set_camera_attr"
getsrv = "get_camera_attr"

class CamParamMonitor(object):
    """ A class to monitor state that is set in Redis,
        and the value reported by the cameras,
        and attempts to sync the 2.
    """
    def __init__(self):
        redis_host = os.environ["REDIS_HOST"]
        print("Redis host: %s" % redis_host)
        self.envoy = RedisEnvoy(os.environ["REDIS_HOST"],
                                client_name="cam_param_monitor")

    def start_threads(self):
        t = threading.Thread(target=self.check_cam_params)
        t.daemon = True
        t.start()

    def get_param_val(self, host, mode, param, requested_val):
        #rospy.logwarn(f"|GET| Starting call on {host} for {param}.")
        driver = "%s_driver" % mode
        wait_time = 0.1
        try:
            if mode == "ir":
                topic = '/'.join(['', host, mode, getsrv])
                rospy.wait_for_service(topic, timeout=wait_time)
                srv = rospy.ServiceProxy(topic, CamGetAttr,
                                         persistent=False)
            elif mode == "uv":
                topic = '/'.join(['', host, mode, driver,
                                getsrv])
                rospy.wait_for_service(topic, timeout=wait_time)
                srv = rospy.ServiceProxy(topic, CamGetAttr,
                                         persistent=False)
            elif mode == "rgb":
                topic = '/'.join(['', host, mode, driver,
                                p1getsrv])
                rospy.wait_for_service(topic, timeout=wait_time)
                srv = rospy.ServiceProxy(topic, GetPhaseOneParameter,
                                         persistent=False)
            resp = srv(name=param)
        except (rospy.exceptions.ROSException, rospy.service.ServiceException) as e:
            #rospy.logerr("|GET| Service exception!")
            #rospy.logerr(e)
            #print(topic)
            #print(param)
            resp = None
            return
        getsrv_val = None
        dtype = None
        try:
            getsrv_val = resp.value
            #rospy.loginfo("|GET| getsrv_val: ", getsrv_val)
            if getsrv_val == "error":
                rospy.loginfo(resp)
                rospy.loginfo(topic)
                rospy.logerr("|GET| Failed to get parameter %s!" % param)
                return
            if mode == "rgb":
                # Phase one params have the type in the return string
                getsrv_val = ''.join(getsrv_val.split(' ')[1:])
                # A random s is sometimes in shutter speed
                if "s" in getsrv_val:
                    getsrv_val = getsrv_val[:-1]
                if isinstance(requested_val, float):
                    try:
                        getsrv_val = float(getsrv_val)
                    except:
                        num,den = map(int, getsrv_val.split( '/' ))
                        getsrv_val = float(num / den)
                elif isinstance(requested_val, int):
                    try:
                        getsrv_val = float(getsrv_val)
                        getsrv_val = int(getsrv_val)
                    except:
                        num,den = map(int, getsrv_val.split( '/' ))
                        getsrv_val = float(num / den)
                        getsrv_val = int(getsrv_val)
            else:
                dtype = resp.dtype
                if isinstance(requested_val, float):
                    getsrv_val = float(getsrv_val)
                elif isinstance(requested_val, int):
                    getsrv_val = int(getsrv_val.rstrip('\x00'))
        except Exception as e:
            rospy.logwarn(f"|GET| value coercion failed on {param}, resp value: %s" % resp.value)
            rospy.logerr(e)
            return
        if not getsrv_val:
            return
        if mode == "rgb":
            param = '_'.join(param.split(' '))
        self.envoy.kv.set("/sys/actual_geni_params/%s/%s/%s"
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
            rospy.logwarn("Setting parameter %s on %s because it differs."
                            % (param, host + "/" + mode))
            # Return real value to set
            return str(param), str(requested_val), dtype
        #rospy.loginfo("|GET| Finished.")
        return None

    def set_params(self, host, mode, params_to_set, requested_params):
        # Set all params that differ from those in redis db
        driver = f"{mode}_driver"
        if mode == "ir":
            topic = '/'.join(['', host, mode, setsrv])
            srv = rospy.ServiceProxy(topic, CamSetAttr,
                                     persistent=False)
        elif mode == "uv":
            topic = '/'.join(['', host, mode, driver, setsrv])
            srv = rospy.ServiceProxy(topic, CamSetAttr,
                                    persistent=False)
        elif mode == "rgb":
            topic = '/'.join(['', host, mode, driver, p1setsrv])
            srv = rospy.ServiceProxy(topic, SetPhaseOneParameter,
                                    persistent=False)
        if mode == "rgb":
            req = ','.join([ f"{name}={v}" for name, (v, d) in
                            params_to_set.items() ])
            resp = None
            if len(params_to_set):
                try:
                    rospy.loginfo("|SET| Setting the following params on P1:")
                    print(req)
                    resp = srv(parameters=req)
                except Exception as e:
                    rospy.logwarn("|SET| Failed to set params for system %s camera %s." %
                            (host, mode))
                    rospy.logerr(e)
        else:
            for name, (v, d) in params_to_set.items():
                try:
                    rospy.loginfo("|SET| Parameters: {} {} {}".format(name, v, d))
                    resp = srv(name=name, value=v, dtype=d)
                except Exception as e:
                    rospy.logwarn("|SET| Failed to set params for system %s camera %s." %
                            (host, mode))
                    rospy.logerr(e)
        #rospy.loginfo("|SET| Finished.")
        return

    def check_cam_params(self):
        # Check all params once every 3 s
        ros_rate = rospy.Rate(0.333)
        while not rospy.is_shutdown():
            tic = time.time()
            for host in HOSTS:
                for mode in MODES:
                    try:
                        requested_params = self.envoy.get_dict(
                            "/sys/requested_geni_params/%s/%s" % (host, mode))
                    except KeyError:
                        continue
                    params_to_set = {}
                    for param, requested_val in requested_params.items():
                        param = ' '.join(param.split('_'))
                        ret = self.get_param_val(host, mode, param, requested_val)
                        if ret is not None:
                            params_to_set[ret[0]] = (ret[1], ret[2])
                    self.set_params(host, mode, params_to_set,
                                    requested_params)
            rospy.loginfo("Time to set parameters was %0.4fs." % (time.time() - tic))
            ros_rate.sleep()


def main():
    rospy.init_node("cam_param_monitor")
    CPM = CamParamMonitor()
    CPM.start_threads()
    rospy.spin()

if __name__ == "__main__":
    main()
