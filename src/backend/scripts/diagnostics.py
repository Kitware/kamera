#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import rospy
import sys



def netcat(hostname, port, content):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    # s.sendall(content)
    s.shutdown(socket.SHUT_WR)
    while 1:
        data = s.recv(1024)
        if data == "":
            break
        print("Received: {}".format(repr(data)))
    print("Connection closed.")
    s.close()





if __name__ == '__main__':
    rospy.init_node('ins_socket_driver')
    try:
        host = rospy.get_param('~ip', '0.0.0.0')
        port = rospy.get_param('~port', 10110)
        buffer_size = rospy.get_param('~buffer_size', 4096)
        timeout = rospy.get_param('~timeout_sec', 2)
        spoof = rospy.get_param('~spoof')
        replay_path = rospy.get_param('~replay')
    except KeyError as e:
        rospy.logerr("Parameter %s not found" % e)
        sys.exit(1)


    client = AvxClient()
    if spoof > 0 :
        client.spoof(spoof)
    elif replay_path:
        client.replay(replay_path)
    else:
        client.run(host, port, buffer_size, timeout)

