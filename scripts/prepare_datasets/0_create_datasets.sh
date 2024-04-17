#!/bin/bash

dataset_dir="/path/to/datasets"

python3 scripts/prepare_datasets/CheXpert/prepare_CheXpert.py $dataset_dir/CheXpert/ datasets/CheXpert.csv
python3 scripts/prepare_datasets/mimic/prepare_mimic.py $dataset_dir/mimic/ datasets/mimic.csv
python3 scripts/prepare_datasets/nih14/prepare_nih14.py $dataset_dir/nih14/ datasets/nih14.csv
python3 scripts/prepare_datasets/RadImageNet/prepare_radimagenet.py $dataset_dir/radiology_ai/ datasets/radimagenet.csv
