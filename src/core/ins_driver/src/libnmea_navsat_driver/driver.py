# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Eric Perko
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the names of the authors nor the names of their
#    affiliated organizations may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import List
import math

import rospy

from sensor_msgs.msg import NavSatFix, NavSatStatus, TimeReference
from geometry_msgs.msg import TwistStamped, QuaternionStamped
from custom_msgs.msg import PASHR, EVT
from tf.transformations import quaternion_from_euler

from libnmea_navsat_driver.checksum_utils import check_nmea_checksum
import libnmea_navsat_driver.parser
from . import nmea_class
from vprint import aprint


class PashrPub(nmea_class.NMEA):
    def __init__(self, name, queue_size=1):
        super(PashrPub, self).__init__(name=name, msg=PASHR, queue_size=queue_size)

    def from_dict(self, data):
        pashr = self.msg()
        self.format_header(pashr, data)
        pashr.time = data['utc_time']
        pashr.heading = data['heading']
        pashr.roll = data['roll']
        pashr.pitch = data['pitch']
        pashr.gnss_status = data['gnss_status']
        pashr.imu_alignment_status = data['imu_alignment_status']
        return pashr


class EvtPub(nmea_class.NMEA):
    def __init__(self, name, queue_size=1):
        super(EvtPub, self).__init__(name=name, msg=EVT, queue_size=queue_size)

    def from_dict(self, data):
        evt = self.msg()
        self.format_header(evt, data)
        evt.time = data['utc_time']
        evt.event = data['event']
        evt.event_counter = data['event_counter']
        return evt


class HeadingPub(nmea_class.NMEA):
    def __init__(self, name, queue_size=1):
        super(HeadingPub, self).__init__(name=name, msg=QuaternionStamped, queue_size=queue_size)

    def from_dict(self, data):
        current_heading = self.msg()
        self.format_header(current_heading, data)
        heading = data.get('heading')
        q = quaternion_from_euler(0, 0, math.radians(heading))
        current_heading.quaternion.x = q[0]
        current_heading.quaternion.y = q[1]
        current_heading.quaternion.z = q[2]
        current_heading.quaternion.w = q[3]
        return current_heading


class TimeRefPub(nmea_class.NMEA):
    def __init__(self, name, queue_size=1):
        super(TimeRefPub, self).__init__(name=name, msg=TimeReference, queue_size=queue_size)

    def from_dict(self, data):
        timestamp = data['utc_time']
        if math.isnan(timestamp):
            return None

        current_time_ref = self.msg_from_header(data)
        current_time_ref.time_ref = rospy.Time.from_sec(data['utc_time'])
        source = data.get('time_ref_source', None)
        if source:
            current_time_ref.source = source
        return current_time_ref


class RosNMEADriver(object):

    def __init__(self):
        self.fix_pub = rospy.Publisher('fix', NavSatFix, queue_size=1)
        self.vel_pub = rospy.Publisher('vel', TwistStamped, queue_size=1)
        self.heading_pub = HeadingPub('heading', queue_size=1)
        self.time_ref_pub = rospy.Publisher('time_reference', TimeReference, queue_size=1)
        self.time_ref_pub2 = TimeRefPub('time_reference', queue_size=1)
        self.pashr_pub = PashrPub('pashr', queue_size=1)
        self.evt_pub = EvtPub('evt', queue_size=5)

        self.time_ref_source = rospy.get_param('~time_ref_source', None)
        self.use_RMC = rospy.get_param('~useRMC', False)

        # epe = estimated position error
        self.default_epe_quality0 = rospy.get_param('~epe_quality0', 1000000)
        self.default_epe_quality1 = rospy.get_param('~epe_quality1', 4.0)
        self.default_epe_quality2 = rospy.get_param('~epe_quality2', 0.1)
        self.default_epe_quality4 = rospy.get_param('~epe_quality4', 0.02)
        self.default_epe_quality5 = rospy.get_param('~epe_quality5', 4.0)
        self.default_epe_quality9 = rospy.get_param('~epe_quality9', 3.0)
        self.using_receiver_epe = False

        self.lon_std_dev = float("nan")
        self.lat_std_dev = float("nan")
        self.alt_std_dev = float("nan")

        """Format for this dictionary is the fix type from a GGA message as the key, with
        each entry containing a tuple consisting of a default estimated
        position error, a NavSatStatus value, and a NavSatFix covariance value."""
        self.gps_qualities = {
          # Unknown
          -1: [
              self.default_epe_quality0,
              NavSatStatus.STATUS_NO_FIX,
              NavSatFix.COVARIANCE_TYPE_UNKNOWN
              ],
          # Invalid
          0: [
              self.default_epe_quality0,
              NavSatStatus.STATUS_NO_FIX,
              NavSatFix.COVARIANCE_TYPE_UNKNOWN
              ],
          # SPS
          1: [
              self.default_epe_quality1,
              NavSatStatus.STATUS_FIX,
              NavSatFix.COVARIANCE_TYPE_APPROXIMATED
              ],
          # DGPS
          2: [
              self.default_epe_quality2,
              NavSatStatus.STATUS_SBAS_FIX,
              NavSatFix.COVARIANCE_TYPE_APPROXIMATED
              ],
          # RTK Fix
          4: [
              self.default_epe_quality4,
              NavSatStatus.STATUS_GBAS_FIX,
              NavSatFix.COVARIANCE_TYPE_APPROXIMATED
              ],
          # RTK Float
          5: [
              self.default_epe_quality5,
              NavSatStatus.STATUS_GBAS_FIX,
              NavSatFix.COVARIANCE_TYPE_APPROXIMATED
              ],
          # WAAS
          9: [
              self.default_epe_quality9,
              NavSatStatus.STATUS_GBAS_FIX,
              NavSatFix.COVARIANCE_TYPE_APPROXIMATED
              ]
          }

    def set_std_from_epe(self, default_epe):
        # use default epe std_dev unless we've received a GST sentence with epes
        if not self.using_receiver_epe or math.isnan(self.lon_std_dev):
            self.lon_std_dev = default_epe
        if not self.using_receiver_epe or math.isnan(self.lat_std_dev):
            self.lat_std_dev = default_epe
        if not self.using_receiver_epe or math.isnan(self.alt_std_dev):
            self.alt_std_dev = default_epe * 2

    def covar_from_hdop(self, hdop):
        position_covariance = [0, ] * 9
        position_covariance[0] = (hdop * self.lon_std_dev) ** 2
        position_covariance[4] = (hdop * self.lat_std_dev) ** 2
        position_covariance[8] = (2 * hdop * self.alt_std_dev) ** 2  # FIXME
        return position_covariance

    # Returns True if we successfully did something with the passed in
    # nmea_string
    def add_sentence(self, nmea_string, frame_id=None, timestamp=None):
        if nmea_string[0] != '$':
            return False
        
        if not check_nmea_checksum(nmea_string):
            rospy.logwarn("Received a sentence with an invalid checksum. " +
                          "Sentence was: %s" % repr(nmea_string))
            aprint('invalid checksum')
            return False

        parsed_sentence = libnmea_navsat_driver.parser.parse_nmea_sentence(nmea_string)
        if not parsed_sentence:
            rospy.logdebug("Failed to parse NMEA sentence. Sentence was: %s" % nmea_string)
            aprint('failed to parse: {}'.format(nmea_string))
            return False

        if frame_id is None:
            frame_id = self.get_frame_id()

        # aprint('\n' + str(parsed_sentence))

        if timestamp:
            current_time = timestamp
        else:
            current_time = rospy.get_rostime()
        current_fix = NavSatFix()
        current_fix.header.stamp = current_time
        current_fix.header.frame_id = frame_id
        current_time_ref = TimeReference()
        current_time_ref.header.stamp = current_time
        current_time_ref.header.frame_id = frame_id
        if self.time_ref_source:
            current_time_ref.source = self.time_ref_source
        else:
            current_time_ref.source = frame_id

        header = {'stamp': current_time, 'frame_id': frame_id}

        # GGA with no RMC
        if not self.use_RMC and 'GGA' in parsed_sentence:
            aprint('Branch 1')
            current_fix.position_covariance_type = \
                NavSatFix.COVARIANCE_TYPE_APPROXIMATED

            data = parsed_sentence['GGA']
            aprint(data)
            fix_type = data['fix_type']
            if not (fix_type in self.gps_qualities):
              fix_type = -1
            gps_qual = self.gps_qualities[fix_type]
            default_epe = gps_qual[0]
            current_fix.status.status = gps_qual[1]
            current_fix.status.service = NavSatStatus.SERVICE_GPS
            current_fix.position_covariance_type = gps_qual[2]

            data.update({})
            current_fix.latitude = data['latitude']
            current_fix.longitude = data['longitude']
            current_fix.altitude = data['altitude']

            self.set_std_from_epe(default_epe)
            positional_covar = self.covar_from_hdop(data['hdop'])
            for n in [0, 4, 8]:
                current_fix.position_covariance[n] = positional_covar[n]

            data.update({'header': header, 'time_ref_source': 'GGA'})
            self.fix_pub.publish(current_fix)
            self.time_ref_pub2.publish_from_dict(data)


        elif 'RMC' in parsed_sentence:
            aprint('Branch 2')
            data = parsed_sentence['RMC']

            # Only publish a fix from RMC if the use_RMC flag is set.
            if self.use_RMC:
                if data['fix_valid']:
                    current_fix.status.status = NavSatStatus.STATUS_FIX
                else:
                    current_fix.status.status = NavSatStatus.STATUS_NO_FIX

                current_fix.status.service = NavSatStatus.SERVICE_GPS

                current_fix.latitude = data['latitude']
                current_fix.longitude = data['longitude']

                current_fix.altitude = float('NaN')
                current_fix.position_covariance_type = \
                    NavSatFix.COVARIANCE_TYPE_UNKNOWN

                data.update({'header': header, 'time_ref_source': 'RMC'})
                self.fix_pub.publish(current_fix)
                self.time_ref_pub2.publish_from_dict(data)

            # Publish velocity from RMC regardless, since GGA doesn't provide it.
            if data['fix_valid']:
                current_vel = TwistStamped()
                current_vel.header.stamp = current_time
                current_vel.header.frame_id = frame_id
                current_vel.twist.linear.x = data['speed'] * \
                    math.sin(data['true_course'])
                current_vel.twist.linear.y = data['speed'] * \
                    math.cos(data['true_course'])
                self.vel_pub.publish(current_vel)
        elif 'GST' in parsed_sentence:
            data = parsed_sentence['GST']

            # Use receiver-provided error estimate if available
            self.using_receiver_epe = True
            self.lon_std_dev = data['lon_std_dev']
            self.lat_std_dev = data['lat_std_dev']
            self.alt_std_dev = data['alt_std_dev']
        elif 'HDT' in parsed_sentence:
            data = parsed_sentence['HDT']
            data['header'] = header
            self.heading_pub.publish_from_dict(data)
        elif 'PASHR' in parsed_sentence:
            data = parsed_sentence['PASHR']
            data.update({'header': header, 'time_ref_source': 'PASHR'})
            self.pashr_pub.publish_from_dict(data)
            self.time_ref_pub2.publish_from_dict(data)

        elif 'EVT' in parsed_sentence:
            data = parsed_sentence['EVT']
            data.update({'header': header, 'time_ref_source': 'EVT'})
            self.evt_pub.publish_from_dict(data)
            # self.time_ref_pub2.publish_from_dict(data)

        else:
            return {}

        return parsed_sentence

    def publish_packets(self, dict_of_packets):
        pass

    def add_multi(self, data_list):
        # type: (List[str]) -> int
        """
        Add multiple data packets and publish them
        Args:
            data_list:

        Returns:
            count of packets published
        """
        for sentence in data_list:
            self.add_sentence(sentence)

    @staticmethod
    def get_frame_id():
        """Helper method for getting the frame_id with the correct TF prefix"""
        frame_id = rospy.get_param('~frame_id', 'gps')
        if frame_id[0] != "/":
            """Add the TF prefix"""
            prefix = ""
            prefix_param = rospy.search_param('tf_prefix')
            if prefix_param:
                prefix = rospy.get_param(prefix_param)
                if prefix[0] != "/":
                    prefix = "/%s" % prefix
            return "%s/%s" % (prefix, frame_id)
        else:
            return frame_id
