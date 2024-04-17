#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiments_dir>"
    exit 1
fi

EXPERIMENTS_DIR=$1

python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/ViTB16/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/ViTMAEB/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/DINOv2ViTB14/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/MedCLIP/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/BiomedCLIPVitB16_224/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/ResNet50/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/MedSAMViTB/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/SAMViTB/config.json
python3 src/retrieval.py ${EXPERIMENTS_DIR}/chexpert_mimic_nih14_radimagenet/CLIPOpenAIViTL14/config.json
