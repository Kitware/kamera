# Pass the path to the Colmap data folder.

docker pull colmap/colmap:latest

docker run -it \
    --gpus all \
    --network="host" \
    -e "DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -w /working \
    -v $1:/working \
    colmap/colmap:latest \
    colmap feature_extractor \
   	--database_path database.db \
   	--image_path images0 \
   	--ImageReader.camera_model=OPENCV \
   	--ImageReader.single_camera=0 \
   	--ImageReader.single_camera_per_folder=1 \
   	--ImageReader.existing_camera_id=-1 \
   	--ImageReader.default_focal_length_factor=1.2 \
   	--SiftExtraction.num_threads=-1 \
   	--SiftExtraction.use_gpu=1 \
   	--SiftExtraction.gpu_index=-1 \
   	--SiftExtraction.max_image_size=3200 \
   	--SiftExtraction.max_num_features=8192 \
   	--SiftExtraction.first_octave=-1 \
   	--SiftExtraction.num_octaves=11 \
   	--SiftExtraction.octave_resolution=3 \
   	--SiftExtraction.peak_threshold=0.0066666666666666671 \
   	--SiftExtraction.edge_threshold=10 \
   	--SiftExtraction.estimate_affine_shape=0 \
   	--SiftExtraction.max_num_orientations=2 \
   	--SiftExtraction.upright=0 \
   	--SiftExtraction.domain_size_pooling=0 \
   	--SiftExtraction.dsp_min_scale=0.16666666666666666 \
   	--SiftExtraction.dsp_max_scale=3 \
   	--SiftExtraction.dsp_num_scales=10
