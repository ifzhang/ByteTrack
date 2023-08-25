#!/bin/bash
# usage: 
# 0. modify the path parameter and check every step carefully
# 1. chmod+x run_tracking.sh
# 2. ./run_tracking.sh
# Set the path to the videos and the tracking script
VIDEOS_PATH="./videos/bdd/"
SCRIPT_PATH="./tools/demo_track.py"
# Set this to the directory where the .txt files are saved
OUTPUT_PATH="./YOLOX_outputs/yolox_s_mix_det/track_vis/"  

# Iterate through all .mov videos in the directory
for video in $VIDEOS_PATH/*.mov; do
    video_name=$(basename "$video" .mov)
    output_file="${OUTPUT_PATH}/${video_name}.txt"

    # Check if the output file already exists
    if [[ -f "$output_file" ]]; then
        echo "Output for $video_name already exists. Skipping..."
        continue
    fi

    # Run the tracking script for the current video
    python $SCRIPT_PATH  video -f exps/example/mot/yolox_s_mix_det.py --path "$video" --ckpt './models/bytetrack_s_mot17.pth.tar' --fp16 --fuse --save_result
    echo "Processed video: $video_name"
    
    #python tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py -c ./models/bytetrack_s_mot17.pth.tar --path ./videos/BDD/b1f4491b-9958bd99.mov --fp16 --fuse --save_result

    # Optionally, rename the output (assuming the original script produces a timestamped output)
    # mv "path_to_original_output" "${video_name}.txt"
done

# Post-process renaming
python tools/post_rename.py

# store txt
zip -r txt_archive.zip $OUTPUT_PATH/*.txt
# Create archives of  .mov videos using zip

DATE=$(date +"%Y_%m_%d")
NEW_FOLDER="$OUTPUT_PATH/$DATE"

mkdir -p "$NEW_FOLDER"
find "$OUTPUT_PATH" -type f -name "*.mp4" -exec mv {} "$NEW_FOLDER/" \;

# zip -r video_archive.zip $OUTPUT_PATH/*/*.mp4
zip -r video_archive.zip "$NEW_FOLDER"

echo "Archiving completed!"



