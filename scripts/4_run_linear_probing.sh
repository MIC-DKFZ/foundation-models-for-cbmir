#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiments_dir>"
    exit 1
fi

EXPERIMENTS_DIR=$1

python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/ViTB16/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/ViTMAEB/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/DINOv2ViTB14/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/MedCLIP/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/BiomedCLIPVitB16_224/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/ResNet50/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/MedSAMViTB/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/SAMViTB/config.json
python3 src/linear_probing.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radiology_ai/CLIPOpenAIViTL14/config.json
