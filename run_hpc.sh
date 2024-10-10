#!/bin/bash
source /nethome/pjajoria/Github/rename_gpus.sh
export HF_HOME=/nethome/pjajoria/.cache/huggingface
export HF_HUB_CACHE=/nethome/pjajoria/.cache/huggingface/hub

echo "Variable $HF_HOME and $HF_HUB_CACHE"
ls $HF_HOME
ls $HF_HUB_CACHE
python Github/Tox21Noisy/train.py