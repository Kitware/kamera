"""Despeckle and normalize 16-bit IR tifs to 8-bit PNGs.

This is the preprocessing the 2025 YOLO11s IR model was trained on;
2025_IR_batch_norm_yolo11s.py imports it so inference matches training.
"""

import argparse
import glob
import os

import cv2
import numpy as np

# STAGE 1 SETTINGS: Despeckling (Bad Pixel Removal)
# Threshold: How much a pixel must deviate from neighbors to be "bad"
# 1000 is a good starting point for 16-bit thermal (0-65535)
DIFF_THRESHOLD = 500

# STAGE 2 SETTINGS: Robust Normalization
# We clip the bottom 0.001% and top 0.001% of the histogram.
# This prevents a few hot/cold pixels from ruining the contrast.
LOWER_PERCENTILE = 0.001
UPPER_PERCENTILE = 99.999


def despeckle_16bit(image, threshold):
    """
    Removes dead pixels by comparing raw image to a median-blurred version.
    Surgical approach: only modifies pixels exceeding the threshold.
    """
    # 1. Create reference using Median Blur (ignores outliers)
    median_blurred = cv2.medianBlur(image, 3)

    # 2. Identify Bad Pixels
    diff = cv2.absdiff(image, median_blurred)
    bad_pixel_mask = diff > threshold

    # 3. Fix ONLY the bad pixels
    cleaned_img = image.copy()
    cleaned_img[bad_pixel_mask] = median_blurred[bad_pixel_mask]

    return cleaned_img


def robust_normalize_to_8bit(image, low_p, high_p):
    """
    Clips outliers using percentiles and linearly stretches the rest.
    Best for maintaining 'Physics' (Heat = Bright) without noise amplification.
    """
    # 1. Determine clipping points based on statistics, not absolute min/max
    p_low = np.nanpercentile(image, low_p)
    p_high = np.nanpercentile(image, high_p)

    # 2. Clip the data (removing the 'tails' of the distribution)
    # This prevents the histogram from being stretched by extreme values
    img_clipped = np.clip(image, p_low, p_high)

    # 3. Linear Stretch to 0-255
    # Use float32 to avoid integer division errors
    img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-5)

    # Convert to 8-bit integer
    img_8bit = (img_norm * 255).astype(np.uint8)

    return img_8bit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", help="folder of 16-bit .tif images")
    parser.add_argument("output_folder", help="folder to write the 8-bit PNGs to")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    tif_files = sorted(glob.glob(os.path.join(args.input_folder, "*.tif")))
    print(f"Processing {len(tif_files)} images...")

    for i, tif_path in enumerate(tif_files):
        # Load 16-bit thermal image
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Warning: Could not read {tif_path}")
            continue

        # --- STEP 1: Remove Bad Pixels ---
        clean_16bit = despeckle_16bit(img, DIFF_THRESHOLD)

        # --- STEP 2: Robust Normalization (16-bit -> 8-bit) ---
        final_8bit = robust_normalize_to_8bit(
            clean_16bit, LOWER_PERCENTILE, UPPER_PERCENTILE
        )

        # Save as PNG
        filename = os.path.basename(tif_path).replace(".tif", ".png")
        cv2.imwrite(os.path.join(args.output_folder, filename), final_8bit)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images...")

    print(f"Done! Processed images saved to '{args.output_folder}'.")


if __name__ == "__main__":
    main()
