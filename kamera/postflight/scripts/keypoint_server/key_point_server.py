from bottle import route, template, run, static_file, request, BaseRequest
import os
import argparse

BaseRequest.MEMFILE_MAX = 6 * 1024 * 1024

image_dir = "<your_dir>"
right_img_file = "kamera_calibration_fl09_L_20240612_204629.686820_ir.png"
left_img_file = "kamera_calibration_fl09_L_20240612_204629.686820_rgb.jpg"


@route("/")
def index():
    return template("keypoint.tpl")


@route("/image-dir", method="POST")
def set_image_dir():
    global image_dir

    json = request.json
    image_dir = json["image-dir"]
    print(f"Set img dir to '{image_dir}'")


@route("/right-img-file", method="POST")
def set_right_img_file():
    global right_img_file
    json = request.json
    right_img_file = json["right-img-file"]
    print(f"Set right img file to '{right_img_file}'")


@route("/left-img-file", method="POST")
def set_left_img_file():
    global left_img_file
    json = request.json
    left_img_file = json["left-img-file"]
    print(f"Set left img file to '{left_img_file}'")


@route("/image/left")
def left_image():
    resp = static_file(left_img_file, root=image_dir)
    resp.set_header("Cache-Control", "no-store")
    return resp


@route("/image/right")
def right_image():
    resp = static_file(right_img_file, root=image_dir)
    resp.set_header("Cache-Control", "no-store")
    return resp


@route("/image/<fn>")
def image(fn):
    return static_file(fn, root=dirname)


@route("/points", method="POST")
def points():
    global points
    json = request.json
    points = []
    for z in zip(json["leftPoints"], json["rightPoints"]):
        points.append(z[0] + z[1])


@route("/points", method="GET")
def get_points():
    return {"points": points}


if __name__ == "__main__":
    dirname = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
