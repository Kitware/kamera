#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from custom_msgs.msg import GsofEvt
from std_msgs.msg import Header


class EventSpoofer(Node):
    def __init__(self):
        super().__init__("event_spoofer")
        self.spoof_pub = self.create_publisher(GsofEvt, "/event", 1)
        self.sub = self.create_subscription(Header, "/trig", self.pub, 10)

    def pub(self, hmsg):
        t = hmsg.stamp
        msg = GsofEvt()
        msg.header.stamp = t
        msg.gps_time = t
        msg.sys_time = t
        msg.time = t.sec + t.nanosec * 1e-9
        self.spoof_pub.publish(msg)
        self.get_logger().info("Published event msg.")


def main(args=None):
    print("Initializing spoof node.")
    rclpy.init(args=args)
    node = EventSpoofer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
