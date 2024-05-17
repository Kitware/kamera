#!/usr/bin/env bash
#2021 (C) Eugene.Borovikov@Kitware.com: register camera images to an existing model, e.g. to calibrate them, assuming colmap is installed an configured.

BaseDN=$1 # base directory name
echo BaseDN=$BaseDN

VocabFN=$2 # SIFT vocabulary file name
echo VocabFN=$VocabFN

echo colmap feature_extractor \
	--database_path $BaseDN/colmap/database.db \
	--image_path $BaseDN/images \
	--image_list_path $BaseDN/images/cam.lst \
	--ImageReader.camera_model OPENCV

# vocab-tree matcher
echo colmap vocab_tree_matcher \
	--database_path $BaseDN/colmap/database.db \
	--VocabTreeMatching.vocab_tree_path $VocabFN \
	--VocabTreeMatching.match_list_path $BaseDN/images/cam.lst
	
# alternative exhaustive_matcher
echo colmap exhaustive_matcher \
	--database_path $BaseDN/colmap/database.db

InMdlPath=$BaseDN/colmap/sparse/org
OutMdlPath=$InMdlPath\_cam

echo mkdir -p $OutMdlPath
echo cp $InMdlPath/project.ini $OutMdlPath

echo colmap image_registrator \
	--database_path $BaseDN/colmap/database.db \
	--input_path $InMdlPath \
	--output_path $OutMdlPath
