#!/usr/bin/python3
import io
import cv2
import flask
import time
import numpy as np
from flask import Flask, jsonify, request, send_file

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from phase_one.srv import GetCompressedImageView, GetImageView


app = Flask(__name__)
bridge = CvBridge()


@app.get('/')
def index():
    return "<p>(∩ ` -´)⊃━━☆ﾟ.*･｡ﾟ</p>"


@app.get('/image_<w>_<h>')
def get_image(w, h):
    tic = time.time()
    topic = "/StandAlone/get_image_view"
    srv = rospy.ServiceProxy(topic, GetImageView,
                             persistent=False)
    try:
        resp = srv(output_width=int(w), output_height=int(h))
    except rospy.service.ServiceException as e:
        print(topic)
        rospy.logerr(e)
        resp = None
    toc = time.time()
    print("Time to receive incoming image was %.3fs" % (toc - tic))
    if resp and resp.success:
        cv_image = bridge.imgmsg_to_cv2(resp.image,
                desired_encoding='passthrough')
        image_binary = cv2.imencode(".jpeg",
                             cv_image[:,:,::-1])[1].tobytes()
    else:
        image_binary = cv2.imencode(".jpeg", np.zeros([100,100,3
                                    ],dtype=np.uint8))[1].tobytes()
    response = flask.make_response(image_binary)
    # could be png here
    response.headers.set('Content-Type', 'image/jpeg')
    toc = time.time()
    print("Time to process incoming image was %.3fs" % (toc - tic))
    return response


def main():
    print("Starting ROS node.")
    rospy.init_node("img_server", anonymous=False)
    print("Starting App.")
    app.run(host="0.0.0.0", port=5000, use_reloader=False, debug=True)
    print("Finished.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
