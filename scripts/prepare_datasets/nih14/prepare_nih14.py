import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def get_test_df(data_file, test_list):
    with open(test_list, "r") as file:
        test_image_indices = {line.strip() for line in file}
    test_df = pd.read_csv(data_file)
    test_df = test_df[test_df["Image Index"].isin(test_image_indices)]
    return test_df


def split_train_val(data_file: Path, trainval_list: Path, ratio=0.9, random_seed=42):
    with open(trainval_list, "r") as file:
        image_indices = {line.strip() for line in file}

    data_df = pd.read_csv(data_file)
    data_df = data_df[data_df["Image Index"].isin(image_indices)]

    # Splitting by Patient ID
    train_ids, val_ids = train_test_split(
        data_df["Patient ID"].unique(), train_size=ratio, random_state=random_seed
    )
    train_df = data_df[data_df["Patient ID"].isin(train_ids)]
    val_df = data_df[data_df["Patient ID"].isin(val_ids)]

    return train_df, val_df


def prepare_dataset(data_df: pd.DataFrame, data_dir: Path, pathologies, split_label):
    pathology_mapping = {
        **{p: "XR/chest/" + p for p in pathologies},
        "Effusion": "XR/chest/Pleural Effusion",
        "Pleural_Thickening": "XR/chest/Pleural Thickening",
    }

    for pathology in pathologies:
        data_df[pathology_mapping.get(pathology, pathology)] = (
            data_df["Finding Labels"].str.contains(pathology).astype(int)
        )

    filepaths = {
        f.name: f.relative_to(data_dir.parent) for f in data_dir.rglob("*.png")
    }
    data_df["filepath"] = data_df["Image Index"].map(filepaths).fillna("File Missing")

    data_df["dataset"] = "nih14"
    data_df["modality"] = "XR"
    data_df["anatomic_region"] = "chest"
    data_df["split"] = split_label

    columns_order = [
        "filepath",
        "dataset",
        "split",
        "modality",
        "anatomic_region",
    ] + list(pathology_mapping.values())
    data_df = data_df[columns_order]
    label_columns = data_df.columns[5:]
    # Only keep single label samples
    data_df = data_df[
        data_df[label_columns].sum(axis=1) == 1
    ]
    return data_df


# Common pathologies
pathologies = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

def main(base_path, output_path):
    # File paths
    data_file = base_path / "Data_Entry_2017.csv"

    # Prepare train, validation, and test datasets
    train_val_list = base_path / "train_val_list.txt"
    train_df, val_df = split_train_val(data_file, train_val_list)
    train_df = prepare_dataset(train_df, base_path, pathologies, "train")
    val_df = prepare_dataset(val_df, base_path, pathologies, "val")

    test_list = base_path / "test_list.txt"
    test_df = get_test_df(data_file, test_list)
    test_df = prepare_dataset(test_df, base_path, pathologies, "test")

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