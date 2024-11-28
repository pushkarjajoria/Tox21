#!/bin/bash
source /nethome/pjajoria/Github/rename_gpus.sh

input_file=$1
output_file=$2

pip install rdkit==2024.3.3
# Run the processing script
python /nethome/pjajoria/Github/Tox21Noisy/create_enamine_fingerprint_dataset.py $input_file $output_file 'false'
