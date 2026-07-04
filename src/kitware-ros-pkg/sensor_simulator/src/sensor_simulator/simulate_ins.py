#! /usr/bin/python
from __future__ import division, print_function
import time

import numpy as np

# ROS imports
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as MsgTime
from custom_msgs.msg import GsofIns


def _time_msg_from_sec(t):
    sec = int(t)
    return MsgTime(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def main(args=None):
    rclpy.init(args=args)
    node = Node('simulate_ptz_camera')
    log = node.get_logger()

    lat = node.declare_parameter('lat', 0.0).value
    lon = node.declare_parameter('lon', 0.0).value
    height = node.declare_parameter('height', 0.0).value

    yaw0 = node.declare_parameter('nominal_yaw', 0.0).value
    pitch0 = node.declare_parameter('nominal_pitch', 0.0).value
    roll0 = node.declare_parameter('nominal_roll', 0.0).value

    yaw_range = node.declare_parameter('yaw_range', 1.0).value
    pitch_range = node.declare_parameter('pitch_range', 1.0).value
    roll_range = node.declare_parameter('roll_range', 1.0).value
    motion_rate = node.declare_parameter('motion_rate', 1.0).value
    pub_rate = node.declare_parameter('pub_rate', 10.0).value

    topic = node.declare_parameter('topic', '/ins').value

    log.info('lat (deg): %s' % str(lat))
    log.info('lon (deg): %s' % str(lon))
    log.info('height (m): %s' % str(height))
    log.info('nominal_yaw (deg): %s' % str(yaw0))
    log.info('nominal_pitch (deg): %s' % str(pitch0))
    log.info('nominal_roll (deg): %s' % str(roll0))
    log.info('yaw_range (deg): %s' % str(yaw_range))
    log.info('pitch_range (deg): %s' % str(pitch_range))
    log.info('roll_range (deg): %s' % str(roll_range))
    log.info('Motion rate (deg/s): %s' % str(motion_rate))
    log.info('Publish rate: %s' % str(pub_rate))
    log.info('Odometry topic: %s' % str(topic))
    # ------------------------------------------------------------------------

    ins_state_pub = node.create_publisher(GsofIns, topic, 1)

    t0 = time.time()

    def tick():
        t = time.time() - t0
        yaw = yaw0 + yaw_range * np.sin(t * motion_rate / yaw_range * 2 * np.pi)
        pitch = pitch0 + pitch_range * np.sin(t * motion_rate / pitch_range * 2 * np.pi)
        roll = roll0 + roll_range * np.sin(t * motion_rate / roll_range * 2 * np.pi)

        print('yaw:', yaw, 'pitch:', pitch, 'roll:', roll)
        msg = GsofIns()
        msg.latitude = float(lat)
        msg.longitude = float(lon)
        msg.altitude = float(height)
        msg.align_status = 4
        msg.gnss_status = 1
        msg.north_velocity = 50.0
        msg.east_velocity = 20.0
        msg.down_velocity = 1.0
        msg.total_speed = float(np.sqrt(msg.north_velocity**2 +
                                        msg.east_velocity**2 +
                                        msg.down_velocity**2))

        msg.heading = float(yaw)
        msg.pitch = float(pitch)
        msg.roll = float(roll)
        msg.track_angle = 5.0
        now = time.time()
        msg.time = now
        msg.header.stamp = _time_msg_from_sec(now)
        ins_state_pub.publish(msg)

    node.create_timer(1.0 / pub_rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
