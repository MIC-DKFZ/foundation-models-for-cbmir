#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_dir> <cbmir_dir>"
    exit 1
fi

DATA_DIR=$1
CBMIR_DIR=$2

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv ViTB16 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv ViTB16 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv ViTB16 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv ViTB16 1340

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv ViTMAEB 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv ViTMAEB 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv ViTMAEB 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv ViTMAEB 1340

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv DINOv2ViTB14 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv DINOv2ViTB14 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv DINOv2ViTB14 1340
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv DINOv2ViTB14 1340

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv MedCLIP 760
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv MedCLIP 760
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv MedCLIP 760
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv MedCLIP 760

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv BiomedCLIPVitB16_224 1720
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv BiomedCLIPVitB16_224 1720
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv BiomedCLIPVitB16_224 1720
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv BiomedCLIPVitB16_224 1720

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv ResNet50 1360
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv ResNet50 1360
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv ResNet50 1360
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv ResNet50 1360

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv MedSAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv MedSAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv MedSAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv MedSAMViTB 6

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv SAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv SAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv SAMViTB 6
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv SAMViTB 6

python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/nih14.csv CLIPOpenAIViTL14 640
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/radimagenet.csv CLIPOpenAIViTL14 640
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/mimic.csv CLIPOpenAIViTL14 640
python3 src/embeddings/create_embeddings.py ${CBMIR_DIR}/embeddings ${CBMIR_DIR}/checkpoints  ${DATA_DIR} ${CBMIR_DIR}/datasets/CheXpert.csv CLIPOpenAIViTL14 640

