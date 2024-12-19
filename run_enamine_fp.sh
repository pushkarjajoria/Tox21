#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/Github/rename_gpus.sh

# Input arguments
file_path_input=$1      # Path(s) to directories containing .parquet files
dir_output=$2           # Batch size for processing
batch_len=$3             # Job type: 'odd' or 'even'
threadpool=$4

# Echo the command to show variable values
echo "Running command: python /nethome/pjajoria/Github/Tox21Noisy/create_enamine_fingerprint_dataset.py \"$file_path_input\" \"$dir_output\" \"$batch_len\" \"$threadpool\" "

# Run the processing script
python /nethome/pjajoria/Github/Tox21Noisy/create_enamine_fingerprint_dataset.py "$file_path_input" "$dir_output" "$batch_len" "$threadpool"
