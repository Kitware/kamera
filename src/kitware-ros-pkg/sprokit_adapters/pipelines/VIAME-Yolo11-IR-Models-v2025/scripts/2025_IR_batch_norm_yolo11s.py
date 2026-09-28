import cv2
import numpy as np
import os
import csv
import time
import logging
import glob
from ultralytics import YOLO

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MANIFEST_FILE = r"manifest.txt" # Text file with one folder path per line
MODEL_PATH = r"Y:\NMML_Polar\Analytics\IceSeal_DetectionModel\Yolov11_IR_2025\train\runs\detect\seals_run1\weights\best.pt"
OUTPUT_CSV = 'master_detections_viame.csv'
ERROR_LOG = 'processing_errors.log'

# Model & Normalization Params
CONFIDENCE_THRESHOLD = 0.01
DIFF_THRESHOLD = 500
LOWER_PERCENTILE = 0.001
UPPER_PERCENTILE = 99.999

# ==========================================
# --- SETUP LOGGING ---
# ==========================================
# Logs to both the console and a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG),
        logging.StreamHandler()
    ]
)

# ==========================================
# --- IMAGE PROCESSING FUNCTIONS ---
# ==========================================
def despeckle_16bit(image, threshold):
    median_blurred = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, median_blurred)
    bad_pixel_mask = diff > threshold
    cleaned_img = image.copy()
    cleaned_img[bad_pixel_mask] = median_blurred[bad_pixel_mask]
    return cleaned_img

def robust_normalize_to_8bit(image, low_p, high_p):
    p_low = np.nanpercentile(image, low_p)
    p_high = np.nanpercentile(image, high_p)
    img_clipped = np.clip(image, p_low, p_high)
    img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-5)
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # YOLO models natively expect 3 channels (RGB/BGR). 
    # We convert the 1-channel grayscale to a 3-channel image for inference compatibility.
    img_8bit_3c = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    return img_8bit_3c

# ==========================================
# --- MAIN PIPELINE ---
# ==========================================
def run_pipeline():
    logging.info(f"Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        logging.critical(f"Failed to load YOLO model: {e}")
        return

    # 1. Read Manifest
    if not os.path.exists(MANIFEST_FILE):
        logging.critical(f"Manifest file not found: {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, 'r') as f:
        # Read lines, strip whitespace, ignore empty lines
        folders = [line.strip() for line in f if line.strip()]

    logging.info(f"Found {len(folders)} folders in manifest.")

    # 2. Setup CSV Output
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        f.write("# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame Identifier, 4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y), 8: Detection or Length Confidence, 9: Target Length (0 or -1 if invalid), 10-11+: Repeated Species, Confidence Pairs or Attributes\n")
        current_time = time.ctime()
        f.write(f"# metadata,exec_time: 0,exported_by: python_yolo_script,exported_at: {current_time},,,,,,,,\n")
        writer = csv.writer(f)

        global_detection_id = 1
        total_images_processed = 0

        # 3. Process Folders
        for folder_idx, folder_path in enumerate(folders):
            logging.info(f"Processing folder [{folder_idx+1}/{len(folders)}]: {folder_path}")
            
            if not os.path.exists(folder_path):
                logging.error(f"Folder does not exist, skipping: {folder_path}")
                continue

            tif_files = glob.glob(os.path.join(folder_path, "*.tif"))
            if not tif_files:
                logging.warning(f"No .tif files found in {folder_path}")
                continue

            # 4. Process Images in Folder
            for i, tif_path in enumerate(tif_files):
                try:
                    # Load image
                    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError("cv2.imread returned None (corrupted or unreadable file).")

                    # Normalize in-memory
                    clean_16bit = despeckle_16bit(img, DIFF_THRESHOLD)
                    final_8bit_3c = robust_normalize_to_8bit(clean_16bit, LOWER_PERCENTILE, UPPER_PERCENTILE)

                    # Run YOLO Inference directly on the numpy array
                    # save=False ensures no drawn images are saved to disk
                    # verbose=False stops it from printing inference times for every single image
                    results = model.predict(source=final_8bit_3c, save=False, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    
                    # Because we passed a single image, results is a list of length 1
                    result = results[0] 
                    filename = os.path.basename(tif_path)

                    # Extract detections
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                        writer.writerow([
                            global_detection_id,    
                            filename,               
                            i,                      # Frame ID (using loop index)
                            f"{x_min:.3f}",         
                            f"{y_min:.3f}",         
                            f"{x_max:.3f}",         
                            f"{y_max:.3f}",         
                            f"{conf:.5f}",          
                            0,                      
                            cls_name,               
                            f"{conf:.5f}"           
                        ])
                        global_detection_id += 1

                    total_images_processed += 1

                except Exception as e:
                    logging.error(f"Failed processing image {tif_path} - ERROR: {e}")

    logging.info(f"Pipeline complete! Successfully processed {total_images_processed} images.")
    logging.info(f"VIAME Results saved to: {os.path.abspath(OUTPUT_CSV)}")
    logging.info(f"Any errors have been written to: {os.path.abspath(ERROR_LOG)}")

if __name__ == "__main__":
    run_pipeline()