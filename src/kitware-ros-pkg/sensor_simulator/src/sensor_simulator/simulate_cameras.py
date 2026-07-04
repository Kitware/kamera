#! /usr/bin/python
from __future__ import division, print_function

# ROS imports
import rclpy
from rclpy.node import Node

# kitware-ros-pkg imports
import sensor_simulator.camera_simulator as camera_simulator


def main(args=None):
    rclpy.init(args=args)
    node = Node('simulate_cameras')

    frame_rate = node.declare_parameter('frame_rate', 4.0).value

    # Define camera simulator
    ir_cam = camera_simulator.CameraSimulator(node, res_x=540, res_y=512,
                                              encoding='mono8',
                                              image_topic='ir/image_raw')

    eo_cam = camera_simulator.CameraSimulator(node, res_x=6576, res_y=4384,
                                              encoding='bayer_grbg8',
                                              image_topic='rgb/image_raw')

    uv_cam = camera_simulator.CameraSimulator(node, res_x=6576, res_y=4384,
                                              encoding='mono8',
                                              image_topic='uv/image_raw')

    node.get_logger().info('Publishing images at %0.1f Hz' % frame_rate)

    def tick():
        ir_cam.publish_test_image()
        eo_cam.publish_test_image()
        uv_cam.publish_test_image()

    node.create_timer(1.0 / frame_rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
