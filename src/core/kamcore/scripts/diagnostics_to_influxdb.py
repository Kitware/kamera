#! /usr/bin/python

import os
import requests

import rospy
from diagnostic_msgs.msg import DiagnosticArray

from roskv.impl.redis_envoy import RedisEnvoy


class D2I(object):
    """ A class to perform the operations of taking in diagnostics data from
    ROS nodes and input into a give influxdb timeseries database.
    """
    def __init__(self, host, org, bucket, token):
        self.envoy = RedisEnvoy(os.environ["REDIS_HOST"],
                client_name="diagnostic2influxdb")
        self.influxdb_url = "http://" + host + ":8086/api/v2/write?org=" + org + "&bucket=" + bucket
        self.influxdb_header = {"Authorization": "Token " + token}
        sub = rospy.Subscriber("/diagnostics", DiagnosticArray,
                               callback=self.diagnostics_to_influxdb,
                               queue_size=100)

    def diagnostics_to_influxdb(self, array):
        """ Takes and input DiagnosticArray msg and places into the database
        of the class.

        :param array: The input ros msg.
        :type array: DiagnosticArray.msg.
        """
        rospy.loginfo_throttle(10, "Processing incoming diagnostics array.")
        for msg in array.status:
            tag_value = msg.hardware_id
            measurement = msg.name.replace(" ", "_")
            payload = ""
            for i, pair in enumerate(msg.values):
                tag_key = "origin"
                key = pair.key.replace(" ", "_")
                if key == "Info" or key == "Intrinsics":
                    continue
                value = pair.value.replace(" ", "_")
                if key == "Actual_frequency_(Hz)":
                    parts = measurement.split("/")
                    host = parts[0]
                    chan = parts[1]
                    topic = '/'.join(['', 'sys', "actual_geni_params", host, chan, "fps"])
                    print("Setting %s to %s fps" % (topic, value))
                    self.envoy.kv.set(topic, value)
                payload += measurement + "," + tag_key + "=" + tag_value + " "
                payload += key + "=" + value + "\n"
            payload=payload[:-1]
            try:
                rospy.loginfo_throttle(60, payload)
                response = requests.post(self.influxdb_url, headers=self.influxdb_header, data=payload)
                if response == 400:
                    rospy.logerr("Malformed payload:")
                    rospy.logerr(payload)
            except requests.exceptions.RequestException as e:
                rospy.logerr('Diagnostics to influxdb error: ')
                rospy.logerr(e)
                return

def main():
    rospy.init_node("diagnostics_to_influxdb")
    host = rospy.get_param("~host")
    org = rospy.get_param("~org")
    bucket = rospy.get_param("~bucket")
    token = rospy.get_param("~token")
    D2I(host, org, bucket, token)
    rospy.loginfo("Waiting for incoming messages on /diagnostics ...")
    rospy.spin()

if __name__ == "__main__":
    main()
