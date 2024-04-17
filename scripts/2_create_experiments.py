import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm
import argparse


def combine_csvs(config: SimpleNamespace) -> None:

    if (
        Path(config.train_csv).exists()
        and Path(config.val_csv).exists()
        and Path(config.test_csv).exists()
    ):
        print(
            f"Skipping combining CSVs as {config.train_csv}, {config.val_csv}, and {config.test_csv} already exist"
        )
        return

    # Create a DataFrame from the first CSV file
    combined_df = pd.concat(
        [pd.read_csv(csv_path) for csv_path in config.csv_paths]
    ).fillna(0.0)

    label_columns = combined_df.columns[5:]

    # Filter out where the sum of label columns is > 1
    sanity_check_df = combined_df[combined_df[label_columns].sum(axis=1) == 1]
    assert len(sanity_check_df) == len(
        combined_df
    ), "There are samples with more than one label"

    # train_df is where split == 'train'
    train_df = combined_df[combined_df["split"] == "train"]

    # val_df is where split == 'val'
    val_df = combined_df[combined_df["split"] == "val"]

    # test_df is where split == 'test'
    test_df = combined_df[combined_df["split"] == "test"]

    train_df.to_csv(config.train_csv, index=False)
    val_df.to_csv(config.val_csv, index=False)
    test_df.to_csv(config.test_csv, index=False)


def main(experiments_dir: Path, embeddings_dir: Path, datasets_dir: Path):
    models = [
        "ViTB16",
        "ViTMAEB",
        "DINOv2ViTB14",
        "MedCLIP",
        "BiomedCLIPVitB16_224",
        "ResNet50",
        "MedSAMViTB",
        "SAMViTB",
        "ViTMAEB",
        "CLIPOpenAIViTL14",
    ]
    for model in tqdm(models):

        config = {
            "comment": f"{model} with CheXpert, MIMIC, NIH14, and RadImageNet",
            "experiments_dir": str(
                experiments_dir / "chexpert_mimic_nih14_radimagenet" / model
            ),
            "embeddings_dir": str(embeddings_dir / model),
            "datasets_dir": str(datasets_dir),
        }

        config = {
            **config,
            "h5_paths": [
                str(Path(config["embeddings_dir"]) / d)
                for d in ["nih14.h5", "mimic.h5", "radiology_ai.h5", "CheXpert.h5"]
            ],
            "csv_paths": [
                str(Path(config["datasets_dir"]) / d)
                for d in [
                    "nih14.csv",
                    "mimic.csv",
                    "radimagenet.csv",
                    "CheXpert.csv",
                ]
            ],
            "index_path": str(Path(config["experiments_dir"]) / "faiss.index"),
            "metadata_path": str(Path(config["experiments_dir"]) / "metadata.json"),
            "train_csv": str(Path(config["experiments_dir"]).parent / "train.csv"),
            "val_csv": str(Path(config["experiments_dir"]).parent / "val.csv"),
            "test_csv": str(Path(config["experiments_dir"]).parent / "test.csv"),
            "knn_csv": str(Path(config["experiments_dir"]) / "knn.csv"),
            "config_path": str(Path(config["experiments_dir"]) / "config.json"),
            "k": 100,
            "epochs": 100,
            "batch_size": 512,
            "lr": 0.001,
            "patience": 20,
            "num_workers": 10,
            "device": "cuda",
        }

        if Path(config["config_path"]).exists():
            print(f"Skipping {config['config_path']} as it already exists")
            continue

        # create experiments_dir
        Path(config["experiments_dir"]).mkdir(parents=True, exist_ok=True)

        combine_csvs(SimpleNamespace(**config))

        # load train.csv
        combined_df = pd.read_csv(config["train_csv"])
        config["label_columns"] = combined_df.columns[5:].tolist()

        # store config in experiments_dir
        print(f"Writing config to {config['config_path']}")
        with open(config["config_path"], "w") as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiments")
    parser.add_argument("experiments_dir", type=Path)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("datasets_dir", type=Path)

    args = parser.parse_args()

    main(
        experiments_dir=args.experiments_dir,
        embeddings_dir=args.embeddings_dir,
        datasets_dir=args.datasets_dir,
    )
