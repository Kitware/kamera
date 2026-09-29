"""Batch-run the 2025 YOLO11s IR hotspot detector over folders of 16-bit tifs.

The model is a TorchScript export run with plain PyTorch (see the README for
how it is produced). Detections are written as a VIAME CSV.
"""

import argparse
import csv
import json
import logging
import os
import time
from glob import glob

import cv2
import torch
import torchvision

from despeckle_and_norm_001 import (
    DIFF_THRESHOLD,
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    despeckle_16bit,
    robust_normalize_to_8bit,
)

# Postprocessing defaults of the YOLO11 predict() the model was validated with
IOU_THRESHOLD = 0.7
MAX_DET = 300
PAD_VALUE = 114

VIAME_HEADER = (
    "# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame "
    "Identifier, 4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y), 8: Detection or Length "
    "Confidence, 9: Target Length (0 or -1 if invalid), 10-11+: Repeated Species, "
    "Confidence Pairs or Attributes\n"
)


def load_model(path, device):
    """Load the TorchScript model and the metadata stored with it at export."""
    extra_files = {"config.txt": ""}
    model = torch.jit.load(path, map_location=device, _extra_files=extra_files)
    model.eval()
    meta = json.loads(extra_files["config.txt"])
    names = {int(k): v for k, v in meta["names"].items()}
    dtype = torch.half if meta.get("half") else torch.float
    return model, meta["imgsz"], names, dtype


def letterbox(img, new_shape):
    """Resize keeping aspect ratio, then pad to new_shape (h, w), centered."""
    h, w = img.shape[:2]
    gain = min(new_shape[0] / h, new_shape[1] / w)
    unpad_w, unpad_h = round(w * gain), round(h * gain)
    if (unpad_w, unpad_h) != (w, h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape[1] - unpad_w) / 2
    dh = (new_shape[0] - unpad_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_VALUE
    )
    return img, gain, (left, top)


@torch.inference_mode()
def detect(model, img_8bit, imgsz, dtype, conf_threshold):
    """Run the model on a single-channel 8-bit image.

    Returns xyxy boxes in image pixels, scores and class ids, best first.
    """
    padded, gain, (left, top) = letterbox(img_8bit, imgsz)
    device = next(model.parameters()).device
    x = torch.from_numpy(padded).to(device, dtype) / 255
    # The model takes 3 channels; the grayscale image is replicated into each
    x = x[None, None].repeat(1, 3, 1, 1)

    pred = model(x)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    # (4 + num_classes, anchors) -> (anchors, 4 + num_classes): cx, cy, w, h, ...
    pred = pred[0].T.float()
    scores, classes = pred[:, 4:].max(1)
    keep = scores > conf_threshold
    xywh, scores, classes = pred[keep, :4], scores[keep], classes[keep]
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)

    keep = torchvision.ops.batched_nms(boxes, scores, classes, IOU_THRESHOLD)[:MAX_DET]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    # Undo the letterbox and clip to the original image
    boxes -= torch.tensor((left, top, left, top), device=boxes.device)
    boxes /= gain
    h, w = img_8bit.shape[:2]
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="TorchScript export of the model")
    parser.add_argument("folders", nargs="*", help="folders of 16-bit .tif images")
    parser.add_argument("--manifest", help="text file with one folder path per line")
    parser.add_argument(
        "-o",
        "--output",
        default="detections_viame.csv",
        help="VIAME CSV to write (default: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="minimum detection confidence (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: %(default)s)",
    )
    parser.add_argument("--log", help="also write the log to this file")
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as f:
            args.folders += [line.strip() for line in f if line.strip()]
    if not args.folders:
        parser.error("give at least one folder or a --manifest")
    return args


def main():
    args = parse_args()

    handlers = [logging.StreamHandler()]
    if args.log:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.info(f"Loading model from: {args.model}")
    model, imgsz, names, dtype = load_model(args.model, args.device)
    logging.info(f"Found {len(args.folders)} folders.")

    with open(args.output, mode="w", newline="") as f:
        f.write(VIAME_HEADER)
        f.write(
            "# metadata,exec_time: 0,exported_by: python_yolo_script,"
            f"exported_at: {time.ctime()},,,,,,,,\n"
        )
        writer = csv.writer(f)

        global_detection_id = 1
        total_images_processed = 0

        for folder_idx, folder_path in enumerate(args.folders):
            logging.info(
                f"Processing folder [{folder_idx + 1}/{len(args.folders)}]: "
                f"{folder_path}"
            )

            if not os.path.exists(folder_path):
                logging.error(f"Folder does not exist, skipping: {folder_path}")
                continue

            tif_files = sorted(glob(os.path.join(folder_path, "*.tif")))
            if not tif_files:
                logging.warning(f"No .tif files found in {folder_path}")
                continue

            for i, tif_path in enumerate(tif_files):
                try:
                    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(
                            "cv2.imread returned None (corrupted or unreadable file)."
                        )

                    clean_16bit = despeckle_16bit(img, DIFF_THRESHOLD)
                    img_8bit = robust_normalize_to_8bit(
                        clean_16bit, LOWER_PERCENTILE, UPPER_PERCENTILE
                    )
                    boxes, scores, classes = detect(
                        model, img_8bit, imgsz, dtype, args.conf
                    )

                    filename = os.path.basename(tif_path)
                    for (x_min, y_min, x_max, y_max), conf, cls_id in zip(
                        boxes, scores, classes
                    ):
                        writer.writerow(
                            [
                                global_detection_id,
                                filename,
                                i,  # Frame ID (index within the folder)
                                f"{x_min:.3f}",
                                f"{y_min:.3f}",
                                f"{x_max:.3f}",
                                f"{y_max:.3f}",
                                f"{conf:.5f}",
                                0,
                                names[int(cls_id)],
                                f"{conf:.5f}",
                            ]
                        )
                        global_detection_id += 1

                    total_images_processed += 1

                except Exception as e:
                    logging.error(f"Failed processing image {tif_path} - ERROR: {e}")

    logging.info(
        f"Pipeline complete! Successfully processed {total_images_processed} images."
    )
    logging.info(f"VIAME Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
