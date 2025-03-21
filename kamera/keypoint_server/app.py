import json
import os
import argparse
from bottle import route, template, run, static_file, request, BaseRequest

BaseRequest.MEMFILE_MAX = 6 * 2048 * 2048

image_dir = ""
left_img_file = "./kamera_calibration_fl09_C_20240612_204041.621432_rgb.jpg"
right_img_file = "./kamera_calibration_fl09_C_20240612_204041.621432_ir.png"


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
    return static_file(fn, root=image_dir)


@route("/points", method="POST")
def to_points():
    global points
    json = request.json
    points = []
    for z in zip(json["leftPoints"], json["rightPoints"]):
        points.append(z[0] + z[1])


@route("/save_points", method="POST")
def save_points():
    global points
    data = request.json
    with open("points.json", "w") as f:
        json.dump(data, f)


@route("/points", method="GET")
def get_points():
    return {"points": points}


def run_keypoint_server(
    host: str,
    port: int,
    image_root: str | os.PathLike,
    left_img_fname: str | os.PathLike,
    right_img_fname: str | os.PathLike,
) -> None:
    global left_img_file, right_img_file, image_dir
    left_img_file = left_img_fname
    right_img_file = right_img_fname
    image_dir = image_root
    run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
