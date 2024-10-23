#Normalization 
#(Normalization code is adapted from Matt B.  Sourced from somewhere in Matt’s Kamera scripts, the rest I wrote and used to help make preprocessing data for COLMAP easier/more automated)  You will likely need to modify this a bit to work with the organization of the data you have.  This script processes ir/rgb/uv images that are all in the same directory(since our directory for LEFT contains all EO/IR/RGB images taken by the LEFT camera, same for CENTER and RIGHT) and then copies them to separate directories as required by COLMAP.

import os
import shutil

import cv2
import numpy as np

def clahe_normalize(image_filepath, stretch_percentiles=[0.1, 99.9]):
    im = cv2.imread(image_filepath, -1)
    img = im.astype(np.float)
    img -= np.percentile(img.ravel(), stretch_percentiles[0])
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), stretch_percentiles[1])/255
    img[img > 255] = 255
    img = np.round(img).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5, 5))
    if img.ndim == 3:
        HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        HLS[:, :, 1] = clahe.apply(HLS[:, :, 1])
        img = cv2.cvtColor(HLS, cv2.COLOR_HLS2BGR)
    else:
        img = clahe.apply(img)

    return img


def process_dir(input_dir, ext_pattern, output_dir, preproc_fn=None):
    if input_dir == output_dir:
        raise Exception('careful not to write to the archive!')
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    images = []
    for fn in files:
        if fn.endswith(ext_pattern):
            images.append(os.path.abspath(os.path.join(input_dir,fn)))
    total = len(images)
    for idx, im_fp in enumerate(images):
        print(f'{idx}/{total}')
        dest_fp = os.path.abspath(os.path.join(output_dir, os.path.basename(im_fp)))
        if os.path.basename(dest_fp)[-4:] == '.tif':
            dest_fp = dest_fp[:-4] + '.png'
        if os.path.isfile(dest_fp):
            print(f'Skipping {dest_fp} already exists')
            continue
        if preproc_fn:
            im = preproc_fn(im_fp)
            cv2.imwrite(dest_fp, im)
            print(f'Preprocessed {os.path.basename(im_fp)}.  Wrote {dest_fp}')
        else:
            shutil.copy(im_fp, dest_fp)
            print(f'Copied {dest_fp}')


# Process all IR Images
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/center_view', 'ir.tif', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap_ir/images0/N68RF_30deg_C_ir', preproc_fn=clahe_normalize)
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/left_view', 'ir.tif', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap_ir/images0/N68RF_30deg_L_ir', preproc_fn=clahe_normalize)
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/right_view', 'ir.tif', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap_ir/images0/N68RF_30deg_R_ir', preproc_fn=clahe_normalize)

# Process all UV Images
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/center_view', 'uv.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_C_uv', preproc_fn=clahe_normalize)
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/left_view', 'uv.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_L_uv', preproc_fn=clahe_normalize)
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/right_view', 'uv.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_R_uv', preproc_fn=clahe_normalize)

# Process Color (Right now just copies all since no normalization happens) but just an example
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/center_view', 'rgb.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_C_rgb')
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/left_view', 'rgb.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_L_rgb')
process_dir('Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/fl09/images_30deg_N68RF/right_view', 'rgb.jpg', 'Y:/NMML_Polar_Imagery/KAMERA_Calibration/2024_IceSeals/camera_model_development/fl09/colmap/images0/N68RF_30deg_R_rgb')



