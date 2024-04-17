import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def prepare_dataset(
    split_file: Path, data_file: Path, output_file: Path, base_path: Path
):
    split_df = pd.read_csv(split_file)
    data_df = pd.read_csv(data_file)

    df_data = []
    not_found = []

    # Iterate over all rows
    for index, row in tqdm(split_df.iterrows(), total=len(split_df)):
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        dicom_id = row["dicom_id"]
        split = row["split"]

        # Filter data_df by subject_id and study_id and extract labels
        row_df = data_df[
            (data_df["subject_id"] == subject_id) & (data_df["study_id"] == study_id)
        ]

        file_path = f"mimic/files/mimic-cxr-jpg/2.0.0/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        if len(row_df) == 0:
            print(f"Could not find {dicom_id} {study_id} {subject_id}")
            not_found.append((dicom_id, study_id, subject_id, file_path))
            continue

        assert (base_path.parent / file_path).exists(), f"File {file_path} does not exist"            

        labels = row_df.iloc[0, 2:].to_numpy()
        labels[labels != 1] = np.nan

        df_data.append(
            [
                file_path,
                "mimic",
                split if split != "validate" else "val",
                "XR",
                "chest",
                *labels,
            ]
        )

    columns = ["filepath", "dataset", "split", "modality", "anatomic_region"] + [
        "XR/chest/" + c for c in list(data_df.columns[2:])
    ]
    new_df = pd.DataFrame(df_data, columns=columns).fillna(0.0).drop(["XR/chest/Support Devices"], axis=1)
    label_columns = new_df.columns[5:]
    # Only keep single label samples
    new_df = new_df[new_df[label_columns].sum(axis=1) == 1]
    new_df.to_csv(output_file, index=False)

    return not_found

def main(base_path, output_path):
    # File paths
    split_file = base_path / "files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
    data_file = base_path / "files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"

    # Prepare dataset
    prepare_dataset(split_file, data_file, output_path, base_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("csv_path", type=Path, help="Path to save the output file")
    args = parser.parse_args()

    main(args.data_path, args.csv_path)


    