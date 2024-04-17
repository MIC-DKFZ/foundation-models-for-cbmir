import argparse
import importlib
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import BaseModel


class ImageDataset(Dataset):
    """
    Dataset to handle images for generating embeddings.
    Arguments:
        dataframe (pd.DataFrame): DataFrame containing the data.
        datasets_dir (Path): Directory path where datasets are stored.
        image_column (str): Column name in DataFrame that contains the image file path.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        datasets_dir: Path,
        image_column: str,
        transform=None,
    ):
        self.data_frame = dataframe
        self.datasets_dir = datasets_dir
        self.image_column = image_column
        self.transform = transform
        self.name = dataframe["dataset"].unique()[0]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        rel_img_path = self.data_frame.loc[idx, self.image_column]
        img_path = self.datasets_dir / rel_img_path
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, rel_img_path


def create_embeddings(
    embeddings_path: Path, model: BaseModel, dataset: ImageDataset, batch_size: int
):
    """
    Generate embeddings for the dataset using the provided model and store them in a .h5 file.
    Arguments:
        embeddings_path (Path): Path to store embeddings.
        model (BaseModel): Model used to generate embeddings.
        dataset (ImageDataset): Dataset from which to generate embeddings.
        batch_size (int): Batch size for data loader.
    """
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=min(6, os.cpu_count()),
        batch_size=batch_size,
    )

    h5_path = embeddings_path / f"{dataset.name}.h5"
    print(f"Saving embeddings to {h5_path}")
    with h5py.File(h5_path, "w") as h5f:
        for inputs, img_paths in data_loader:
            embeddings = model.get_image_embeddings(inputs).numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            for i, path in enumerate(img_paths):
                h5f.create_dataset(path, data=embeddings[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embeddings Creator")
    parser.add_argument("dataset_csv", type=Path, help="Path to the dataset csv file")
    parser.add_argument(
        "model", type=str, help="Class name of the model to use for embeddings creation"
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        help="Path where the embeddings are stored",
        default="embeddings/",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        help="Path where the model checkpoints of the pretrained models are stored",
        default="checkpoints/",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to the base directory containing the dataset",
        default="data/",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite if embeddings already exist",
    )

    args = parser.parse_args()

    embeddings_path = args.embeddings_path / args.model
    embeddings_path.mkdir(exist_ok=True, parents=True)

    model_module = importlib.import_module("models")
    model = getattr(model_module, args.model)(checkpoint_path=args.checkpoint_path)

    if not hasattr(model, "process_image"):
        # We expect the model to have a method called 'process_image' for image preprocessing
        raise ValueError(
            "Model does not have a 'process_image' method for image preprocessing."
        )

    dataset = ImageDataset(
        dataframe=pd.read_csv(args.dataset_csv),
        datasets_dir=args.data_dir,
        image_column="filepath",
        transform=model.process_image,
    )

    # output path
    h5_path = embeddings_path / f"{dataset.name}.h5"
    if h5_path.exists() and not args.force:
        print(f"Embeddings already exist at {h5_path}. Use --force to overwrite.")
        exit()

    print("==> Evaluating ...")
    create_embeddings(embeddings_path, model, dataset, args.batch_size)
