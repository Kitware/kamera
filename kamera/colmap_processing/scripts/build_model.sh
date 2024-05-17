#!/usr/bin/env bash
#2021 (C) Eugene.Borovikov@Kitware.com: build COLMAP sparse and dense models.

# input parameters

BaseDN=$1
echo BaseDN=$BaseDN

DBFN=$2
echo DBFN=$DBFN

ImgLstFN=$3
echo ImgLstFN=$ImgLstFN

VocabFN=$4
echo VocabFN=$VocabFN

# feature extraction
echo colmap feature_extractor \
	--database_path $DBFN \
	--image_path $BaseDN \
	--image_list_path $ImgLstFN \
	--ImageReader.camera_model=OPENCV \
	--ImageReader.single_camera_per_folder=1

# vocab-tree matcher
echo colmap vocab_tree_matcher \
	--database_path $DBFN \
	--VocabTreeMatching.vocab_tree_path $VocabFN

SparseDN=$5
echo SparseDN=$SparseDN
echo mkdir -p $SparseDN

# sparse model mapper & bundle-adjuster
echo colmap mapper \
	--database_path $DBFN \
	--image_path $BaseDN \
	--output_path $SparseDN \
	--Mapper.multiple_models=0

DenseDN=$6
echo DenseDN=$DenseDN
echo mkdir -p $DenseDN

# image undistorter for subsequent dense reconstruction
echo colmap image_undistorter \
	--image_path $BaseDN \
	--input_path $SparseDN/0 \
	--output_path $DenseDN \
	--output_type COLMAP \
	--max_image_size 2000

# stereo match
echo colmap patch_match_stereo \
	--workspace_path $DenseDN \
	--workspace_format COLMAP \
	--PatchMatchStereo.geom_consistency true

# stereo fusion
echo colmap stereo_fusion \
	--workspace_path $DenseDN \
	--workspace_format COLMAP \
	--input_type geometric \
	--output_path $DenseDN/fused.ply

# optional poisson mesh
echo colmap poisson_mesher \
	--input_path $DenseDN/fused.ply \
	--output_path $DenseDN/poisson.ply

# optional delaunay mesh
echo colmap delaunay_mesher \
	--input_path $DenseDN \
	--output_path $DenseDN/delaunay.ply
