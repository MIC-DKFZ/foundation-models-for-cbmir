from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def split_train_val(data_file: Path, ratio=0.9, random_seed=42):
    data_df = pd.read_csv(data_file).fillna(0.0).drop(["Support Devices"], axis=1)
    label_columns = data_df.columns[5:]
    # Only keep single label samples
    data_df = data_df[data_df[label_columns].sum(axis=1) == 1]

    # Retrieve unique patients from data_df['Path']
    data_df["Patient ID"] = data_df["Path"].apply(lambda x: Path(x).parts[2])

    # Splitting by Patient ID
    train_ids, val_ids = train_test_split(
        data_df["Patient ID"].unique(), train_size=ratio, random_state=random_seed
    )

    # split data_df into train_df and val_df
    train_df = data_df[data_df["Patient ID"].isin(train_ids)].drop(
        columns=["Patient ID"]
    )
    val_df = data_df[data_df["Patient ID"].isin(val_ids)].drop(columns=["Patient ID"])

    return train_df, val_df


def prepare_dataset(data_df: pd.DataFrame, data_dir: Path, split_label):
    # Convert labels to int and set all < 1 to NaN
    data_df[data_df.columns[5:]] = data_df[data_df.columns[5:]].astype(float)
    data_df.iloc[:, 5:] = data_df.iloc[:, 5:].where(data_df.iloc[:, 5:] >= 1, np.nan)

    # this will only change the content for the test set
    data_df = data_df.fillna(0.0)
    if "Support Devices" in data_df.columns: 
        print(f'Dropped column Support Devices for {split_label} set.')
        data_df = data_df.drop(["Support Devices"], axis=1)
    label_columns = data_df.columns[5:]
    # Only keep single label samples
    data_df = data_df[data_df[label_columns].sum(axis=1) == 1]

    # rename columns: 'Path' -> 'filepath'
    data_df.rename(columns={"Path": "filepath"}, inplace=True)

    # add 'XR/chest/' to all label column names
    data_df.rename(
        columns={c: "XR/chest/" + c for c in data_df.columns[5:]}, inplace=True
    )

    # rename all filepaths
    data_df["filepath"] = data_df["filepath"].apply(
        lambda x: str(Path(data_dir.name) / x)
    )

    # Reorder columns
    columns_order = [
        "filepath",
        "dataset",
        "split",
        "modality",
        "anatomic_region",
    ] + list(data_df.columns[5:])

    # Add new columns
    data_df["dataset"] = "CheXpert"
    data_df["modality"] = "XR"
    data_df["anatomic_region"] = "chest"
    data_df["split"] = split_label

    return data_df[columns_order]


def main(base_path: Path, output_path: Path):
    # File paths
    train_file = base_path / "CheXpert-v1.0/train.csv"
    test_file = base_path / "CheXpert-v1.0/valid.csv"

    train_df, val_df = split_train_val(train_file)

    train_df = prepare_dataset(train_df, base_path, "train")
    val_df = prepare_dataset(val_df, base_path, "val")
    test_df = prepare_dataset(pd.read_csv(test_file), base_path, "test")

    # Combine and save to a single file
    combined_df = pd.concat([train_df, val_df, test_df]).fillna(0.0)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("csv_path", type=Path, help="Path to save the output file")
    args = parser.parse_args()

    main(args.data_path, args.csv_path)
