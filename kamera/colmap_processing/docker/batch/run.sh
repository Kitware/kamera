#!/bin/bash
set -e
mp4="$1"
DIR=$(dirname "${mp4}")
FILE=$(basename "${mp4}")
BASE="${FILE%.*}"
colmap_dir="${DIR}/${BASE}"
LOG=$DIR/log.txt
[[ -f $LOG ]] || touch $LOG

echo "Starting processing of $mp4"
s0=`date +%s`

# Create folder of the same name as the mp4
mkdir -p $colmap_dir/sparse $colmap_dir/snapshots $colmap_dir/images0

# Extract mp4 into images
echo "Extracting mp4 into images at: $colmap_dir/images0"
docker run --rm -it -v "$DIR:/tmp/workdir" jrottenberg/ffmpeg -i "/tmp/workdir/$FILE" "/tmp/workdir/${BASE}/images0/frame%06d.png"
end=`date +%s`
runtime=$( echo "$end - $s0" | bc -l )
start=`date +%s`
echo "Finished ffmpeg frame extraction on  $mp4 in $runtime seconds." | tee -a $LOG

echo "Running feature extraction."
./feature_extractor.sh $colmap_dir
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
start=`date +%s`
echo "Finished feature extraction on  $mp4 in $runtime seconds." | tee -a $LOG

echo "Running the exhaustive feature matcher."
./exhaustive_matcher.sh $colmap_dir
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
start=`date +%s`
echo "Finished exhaustive matcher on  $mp4 in $runtime seconds." | tee -a $LOG

echo "Running the mapper."
./mapper.sh $colmap_dir
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
start=`date +%s`
echo "Finished mapping on  $mp4 in $runtime seconds." | tee -a $LOG

end=`date +%s`
runtime=$( echo "$end - $s0" | bc -l )

echo "Finished processing $mp4 in $runtime seconds." | tee -a $LOG
echo "=============================================" | tee -a $LOG
