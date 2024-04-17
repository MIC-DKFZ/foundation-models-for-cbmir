import argparse
import datetime
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from types import SimpleNamespace
import time
from json import load

import torch.optim as optim


class H5Dataset(Dataset):
    """
    A custom PyTorch Dataset for loading data from H5 files into memory.

    Attributes:
        features (Tensor): The features loaded into memory.
        labels (Tensor): The labels loaded into memory.
    """

    def __init__(self, config, df: pd.DataFrame):
        self.df = df
        self.config = config
        h5_embeddings_path = Path(config.experiments_dir) / "embeddings.h5"

        self.combine_embeddings(h5_embeddings_path)

        self.features = {}
        with h5py.File(h5_embeddings_path, "r") as features_h5f:
            for idx in tqdm(range(len(self.df)), desc="Loading features"):
                img_path = self.df.iloc[idx]["filepath"]
                self.features[img_path] = torch.tensor(features_h5f[img_path][:]).type(
                    torch.float32
                )
        print(f"Loaded {len(self.features)} features")

    def combine_embeddings(self, output_path: Path, force=False):
        if (not output_path.exists()) or force:
            print(f"Saving embeddings to {output_path}")
            with h5py.File(output_path, "w") as features_h5f:
                for file_path in tqdm(self.config.h5_paths):
                    with h5py.File(file_path, "r") as h5f_in:
                        for key in h5f_in.keys():
                            features_h5f.copy(h5f_in[key], key)
            print(f"Saved embeddings to {output_path}")

    def __len__(self):
        return len(self.df)

    def embed_dim(self):
        return self.features[self.df["filepath"].iloc[0]].shape[0]

    def num_labels(self):
        return len(self.config.label_columns)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        features = self.features[img_path][:].type(torch.float32)
        labels = torch.tensor(self.df.iloc[idx][self.config.label_columns]).type(
            torch.float32
        )
        return features, labels, img_path


class LinearClassifier(nn.Module):
    """
    A simple linear classifier module in PyTorch.

    Args:
        dim (int): The size of the input feature dimension.
        num_labels (int): The number of output labels.
    """

    def __init__(self, dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)


def train(model, loader, optimizer, epoch, writer, config):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, _ in tqdm(loader, desc=f"Epoch {epoch} Training"):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, targets_max = torch.max(targets.data, 1)
        total += targets_max.size(0)
        correct += (predicted == targets_max).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    writer.add_scalar("loss", epoch_loss, epoch)
    writer.add_scalar("accuracy", epoch_acc, epoch)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, epoch, writer, config):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, _ in tqdm(loader, desc=f"Epoch {epoch} Validation"):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, targets_max = torch.max(targets.data, 1)
        total += targets_max.size(0)
        correct += (predicted == targets_max).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    writer.add_scalar("loss", epoch_loss, epoch)
    writer.add_scalar("accuracy", epoch_acc, epoch)
    return epoch_loss, epoch_acc


@torch.no_grad()
def test(model, test_loader, config):
    model.eval()
    predictions = []

    for inputs, targets, filepaths in test_loader:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        outputs = model(inputs)

        filepaths_np = np.array(filepaths).reshape(-1, 1)
        # apply sigmoid to outputs
        outputs_np = torch.softmax(outputs, dim=1).cpu().numpy()
        combined = np.concatenate((filepaths_np, outputs_np), axis=1)

        predictions.extend(combined)

    return predictions


def main(config):
    print(
        f"Starting linear probing experiment for {Path(config.experiments_dir)} at {datetime.datetime.now()}"
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    linear_probing_dir = Path(config.experiments_dir) / "linear_probing" / timestamp
    linear_probing_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = linear_probing_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer_train = SummaryWriter(
        log_dir=linear_probing_dir / "logs" / "train" / timestamp
    )
    writer_val = SummaryWriter(
        log_dir=linear_probing_dir / "logs" / "validate" / timestamp
    )

    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)

    # Load Datasets
    train_dataset = H5Dataset(config, train_df)
    val_dataset = H5Dataset(config, val_df)

    # Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Linear Classifier
    linear_classifier = LinearClassifier(
        dim=train_dataset.embed_dim(), num_labels=train_dataset.num_labels()
    ).to(config.device)

    # Optimizer
    optimizer = optim.AdamW(
        linear_classifier.parameters(), lr=config.lr, weight_decay=0.01
    )

    # Early Stopping and Checkpoint Initialization
    best_val_acc = 0.0
    patience_counter = 0

    # Training and Evaluation Loop
    for epoch in range(config.epochs):
        start_time = time.time()
        train(linear_classifier, train_loader, optimizer, epoch, writer_train, config)

        val_loss, val_acc = validate(
            linear_classifier, val_loader, epoch, writer_val, config
        )

        print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")

        torch.save(
            linear_classifier.state_dict(),
            checkpoints_dir / f"checkpoint_{epoch}.pth",
        )
        # Checkpoint and Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save checkpoint
            torch.save(
                linear_classifier.state_dict(),
                checkpoints_dir / "best_checkpoint.pth",
            )
            print("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping")
                break

    writer_train.close()
    writer_val.close()

    # Load Best Model for Test Evaluation
    linear_classifier.load_state_dict(
        torch.load(checkpoints_dir / "best_checkpoint.pth")
    )
    test_dataset = H5Dataset(config, pd.read_csv(config.test_csv))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    predictions = test(linear_classifier, test_loader, config)

    linear_probing_results = linear_probing_dir / "linear_probing_results.csv"
    print(f"Saving predictions to {linear_probing_results}")
    pd.DataFrame(predictions, columns=["query", *config.label_columns]).to_csv(
        linear_probing_results, index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear classification on precomputed features")

    parser.add_argument(
        "config",
        # "--config",
        # default="/home/s037r/shared/cluster/data/cbir/experiments/BiomedCLIP/chexpert_mimic_nih14_radiology_ai/config.json",
        type=Path,
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = SimpleNamespace(**load(f))
