#3.3 Batch Model Analyze script
#This most likely works on windows but I’ve only tested it on my linux machine.

import os
import subprocess

if os.name == 'nt':
    # Windows
    COLMAP_TARGET = os.path.normpath(r"C:\software\COLMAP-3.6-windows-cuda\COLMAP.bat")
else:
    COLMAP_TARGET = 'colmap'
    
def model_analyze(path):
    path = os.path.abspath(os.path.normpath(path))
    
    result = subprocess.run([
        COLMAP_TARGET, 
        'model_analyzer', 
        '--path', 
        path]
        , stdout=subprocess.PIPE)
    print('\n=================')
    print("Model: %s" % path)
    print(result.stdout.decode('utf-8'))



sparse_folder_path = r"Y:\NMML_Polar_Imagery\KAMERA_Calibration\2024_IceSeals\camera_model_development\fl09\colmap_ir\sparse"

models = sorted([os.path.join(sparse_folder_path, x) for x in os.listdir(sparse_folder_path)])
print('All Models: ' + str(models))
for s in models:
    model_analyze(s)

