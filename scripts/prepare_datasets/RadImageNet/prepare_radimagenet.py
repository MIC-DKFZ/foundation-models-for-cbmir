import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

from tqdm import tqdm


def extract_details(filepath: Path) -> tuple:
    """
    Extracts modality, anatomic region, and label from a given file path.

    Args:
    filepath (Path): The path of the file.

    Returns:
    tuple: A tuple containing the modality, anatomic region, and label.
    """
    mapping = {
        "US/aorta": "US/aorta/normal",
        "US/bladder": "US/bladder/normal",
        "US/cbd": "US/cbd/normal",
        "US/fibroid": "US/fibroid/normal",
        "US/gb": "US/gb/normal",
        "US/ivc": "US/ivc/normal",
        "US/kidney": "US/kidney/normal",
        "US/liver": "US/liver/normal",
        "US/ovary": "US/ovary/normal",
        "US/pancreas": "US/pancreas/normal",
        "US/portal_vein": "US/portal_vein/normal",
        "US/spleen": "US/spleen/normal",
        "US/thyroid": "US/thyroid/normal",
        "US/thyroid_nodule": "US/thyroid/nodule",
        "US/uterus": "US/uterus/normal",
    }

    extracted_path = str(Path(*filepath.parts[1:3]))
    if extracted_path in mapping:
        filepath = Path(str(filepath).replace(extracted_path, mapping[extracted_path]))

    modality = filepath.parts[1]
    anatomic_region = filepath.parts[2]
    label = str(Path(*filepath.parts[1:-1]))

    return modality, anatomic_region, label


def read_file_to_list(file_path: str) -> list:
    """
    Reads a file and returns its content as a list of strings.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    list: A list of strings, each representing a line in the file.
    """
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def process_csv(input_csv: str, ignore_list_file: str, split_type: str) -> pd.DataFrame:
    ignore_list = read_file_to_list(ignore_list_file)

    df = pd.read_csv(input_csv)
    print(f"Number of samples before processing: {len(df)}")

    label_data = defaultdict(set)
    modality_data = {}
    anatomic_region_data = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = row["filename"]
        if filepath in ignore_list:
            continue

        modality, anatomic_region, label = extract_details(Path(filepath))
        modality_data[filepath] = modality
        anatomic_region_data[filepath] = anatomic_region
        label_data[filepath].add(label)

    unique_labels = sorted(
        set(label for labels in label_data.values() for label in labels)
    )
    one_hot_encoded_data = {label: [] for label in unique_labels}
    one_hot_encoded_data.update(
        {
            "filepath": [],
            "dataset": [],
            "modality": [],
            "anatomic_region": [],
            "split": [],
        }
    )

    for filepath, labels in label_data.items():
        one_hot_encoded_data["filepath"].append(filepath)
        one_hot_encoded_data["dataset"].append("radiology_ai")
        one_hot_encoded_data["modality"].append(modality_data[filepath])
        one_hot_encoded_data["anatomic_region"].append(anatomic_region_data[filepath])
        one_hot_encoded_data["split"].append(split_type)
        for label in unique_labels:
            one_hot_encoded_data[label].append(1 if label in labels else 0)

    new_df = pd.DataFrame(one_hot_encoded_data)
    desired_order = [
        "filepath",
        "dataset",
        "split",
        "modality",
        "anatomic_region",
    ] + unique_labels
    return new_df[desired_order]


# Prepare datasets
def construct_file_path(base_dir: Path, split_type: str) -> Path:
    return base_dir / f"RadiologyAI_{split_type}.csv"


def main(base_path, output_path):
    # Base directory
    ignore_list_file = Path(__file__).parent / "ignore_list.txt"

    # Prepare datasets for each split and combine
    combined_df = pd.concat(
        [
            process_csv(
                construct_file_path(base_path, "train"), ignore_list_file, "train"
            ),
            process_csv(construct_file_path(base_path, "val"), ignore_list_file, "val"),
            process_csv(
                construct_file_path(base_path, "test"), ignore_list_file, "test"
            ),
        ]
    ).fillna(0.0)

    # Save to a single file
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("csv_path", type=Path, help="Path to save the output file")
    args = parser.parse_args()

    main(args.data_path, args.csv_path)
