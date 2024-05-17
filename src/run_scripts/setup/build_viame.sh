#!/usr/bin/env bash

# Allow us to force building a specific version
VIAME_BRANCH=${VIAME_BRANCH:-kamera/master}
NO_VIAME_CMAKE=${NO_VIAME_CMAKE:-}

for VNAME in BUILD_DIR SRC_BASE
do
  if [[ -z "${!VNAME}" ]]
  then
    echo "ERROR: Expected $VNAME environment variable that is missing."
    exit 1
  else
    echo "INFO: ENV ${VNAME} = ${!VNAME}"
  fi
done

echo $BUILD_DIR

cd ${SRC_BASE}/viame \
    && git checkout kamera/master \
    && git submodule update --init --recursive

# Nerf the ITK python bindings. This cuts down build time from hours
sed -i 's/-DITK_WRAP_PYTHON:BOOL=${fletch_BUILD_WITH_PYTHON}/-DITK_WRAP_PYTHON:BOOL=OFF/g' \
    "${SRC_BASE}/viame/packages/fletch/CMake/External_ITK.cmake"

# Enter anaconda env to use Py 3.5
source "/root/anaconda3/bin/activate"

if [[ -z $NO_VIAME_CMAKE ]]; then
    cmake \
            -B /build/viame \
            -S /src/viame \
            -DCMAKE_BUILD_TYPE:STRING=Release \
            -DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
            -DVIAME_ENABLE_BURNOUT:BOOL=OFF \
            -DVIAME_DOWNLOAD_MODELS=ON \
            -DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL:BOOL=ON \
            -DVIAME_DOWNLOAD_MODELS-HABCAM:BOOL=OFF \
            -DVIAME_DOWNLOAD_MODELS-MOUSS:BOOL=OFF \
            -DVIAME_DOWNLOAD_MODELS-PYTORCH:BOOL=OFF \
            -DVIAME_ENABLE_CAMTRAWL:BOOL=OFF \
            -DVIAME_ENABLE_CUDA:BOOL=ON \
            -DVIAME_ENABLE_DOCS:BOOL=OFF \
            -DVIAME_ENABLE_FFMPEG:BOOL=OFF \
            -DVIAME_ENABLE_FASTER_RCNN:BOOL=OFF \
            -DVIAME_ENABLE_FLASK:BOOL=OFF \
            -DVIAME_ENABLE_ITK:BOOL=ON \
            -DVIAME_ENABLE_KWANT:BOOL=OFF \
            -DVIAME_ENABLE_MATLAB:BOOL=OFF \
            -DVIAME_ENABLE_OPENCV:BOOL=ON \
            -DVIAME_ENABLE_PYTHON:BOOL=ON \
            -DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
            -DVIAME_ENABLE_SEAL_TK:BOOL=OFF \
            -DVIAME_ENABLE_SMQTK:BOOL=OFF \
            -DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
            -DVIAME_ENABLE_VIVIA:BOOL=OFF \
            -DVIAME_ENABLE_VXL:BOOL=ON \
            -DVIAME_ENABLE_YOLO:BOOL=ON \
            -DVIAME_ENABLE_TENSORFLOW:BOOL=ON \
            -DVIAME_ENABLE_PYTORCH-INTERNAL:BOOL=ON \
            -DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
            -DVIAME_ENABLE_PYTORCH:BOOL=ON \
            -DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
            -DVIAME_ENABLE_CAFFE:BOOL=OFF \
            -DEXTERNAL_OpenCV:PATH=/opt/ros/kinetic/share/OpenCV-3.3.1-dev

else
    printf "\n~~~\nSkipping cmake\n~~~\n"
fi

cd ${BUILD_DIR}/viame &&\
    time make -j$((`nproc`-1))

cd ${BUILD_DIR}/viame &&\
    time make -j$((`nproc`-1)) install kwiver

cd ${BUILD_DIR}/viame &&\
    time make -j$((`nproc`-1)) install &> /dev/null

printf "#!/bin/bash
export VIAME_INSTALL=${BUILD_DIR}/viame/install
echo $VIAME_INSTALL
" > /usr/local/bin/get_viame_install

# This link has to be sourced at its actual location to work right
# source `readlink /link_to_setup_viame`
ln -svf /build/viame/install/setup_viame.sh /link_to_setup_viame
