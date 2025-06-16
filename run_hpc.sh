#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/Github/rename_gpus.sh

# Input arguments
directory_paths=$1      # Path(s) to directories containing .parquet files
batch_size=$2           # Batch size for processing
job_type=$3             # Job type: 'odd' or 'even'
identifier=$4

# Install the required library
#pip install rdkit
# pip install --upgrade numpy 

# Echo the command to show variable values
echo "Running command: python /nethome/pjajoria/Github/Tox21Noisy/find_similar_molecules_tensor.py \"$directory_paths\" \"$batch_size\" --job_type \"$job_type\" > output_$identifier.log"

# Run the processing script
python3.9 -u /nethome/pjajoria/Github/Tox21Noisy/find_similar_molecules_tensor.py "$directory_paths" "$batch_size" --job_type "$job_type"
