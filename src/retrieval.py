import argparse
import faiss
import h5py
import json
import numpy as np
import pandas as pd
from json import load
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from types import SimpleNamespace


def load_filepaths(csv_path: Path) -> List[str]:
    # Load CSV and return list of img_paths
    df = pd.read_csv(csv_path)
    return df["filepath"].tolist()


def load_index(index_path: Path, metadata_path: Path):
    print("Loading index and metadata...")
    # Load index and metadata
    index = faiss.read_index(str(index_path))
    with open(str(metadata_path), "r") as f:
        metadata = json.load(f)
    print("Index and metadata loaded!")
    return index, metadata


def create_index(
    config,
    filepaths: List[str],
    force=False,
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    if (
        not force
        and Path(config.index_path).exists()
        and Path(config.metadata_path).exists()
    ):
        print("Index and metadata already exist!")
        return load_index(config.index_path, config.metadata_path)
    else:
        index = None
        # Placeholder for metadata (img_paths)
        metadata = []

        print("Creating index...")
        for i, h5_path in enumerate(config.h5_paths):
            print(f"({i+1}/{len(config.h5_paths)}) Indexing {h5_path}...")
            with h5py.File(h5_path, "r") as h5f:
                for img_path in tqdm(filepaths, desc="Indexing", unit="image"):
                    if img_path in h5f:
                        embedding = h5f[img_path][:].reshape(1, -1)
                        if index is None:
                            # Create FAISS index
                            d = embedding.shape[1]
                            index = faiss.IndexFlatIP(d)
                        index.add(embedding)
                        metadata.append(img_path)
        print("Index created!")

        # Store index and metadata
        faiss.write_index(index, config.index_path)
        with open(config.metadata_path, "w") as f:
            json.dump(metadata, f)

        return index, metadata


def h5_to_dict(h5_paths: List[Path], filepaths: List[str]) -> Dict[str, np.ndarray]:
    embeddings = {}
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5f:
            for img_path in filepaths:
                if img_path in h5f:
                    embeddings[img_path] = h5f[img_path][:]
    return embeddings


def knn_retrieval(
    index: faiss.IndexFlatL2,
    metadata: List[str],
    query_dict: Dict[str, np.ndarray],
    knn_csv: Path,
    k: int,
    batch_size: int = 4096,
) -> pd.DataFrame:
    # Prepare DataFrame in advance
    num_queries = len(query_dict)
    column_names = ["query"] + [f"{i+1}" for i in range(k)]
    results_df = pd.DataFrame(columns=column_names, index=range(num_queries))

    # Process in batches
    query_items = list(query_dict.items())
    for batch_start in tqdm(
        range(0, num_queries, batch_size), desc="Evaluating", unit="batch"
    ):
        batch_end = min(batch_start + batch_size, num_queries)
        batch_queries = query_items[batch_start:batch_end]
        query_paths, query_embeddings = zip(*batch_queries)
        query_embeddings = np.array(query_embeddings)

        # Perform search for the entire batch
        _, indices = index.search(query_embeddings, k)
        for i, idx in enumerate(indices):
            top_k_results = [metadata[index] for index in idx]
            results_df.iloc[batch_start + i] = [query_paths[i]] + top_k_results

    # Save to CSV
    results_df.to_csv(knn_csv, index=False)
    return results_df


def main(config, force):
    if Path(config.knn_csv).exists() and not force:
        print("KNN CSV already exists! Exiting...")
        return

    index_filepaths = load_filepaths(config.train_csv)
    query_filepaths = load_filepaths(config.test_csv) + load_filepaths(config.val_csv)

    index, metadata = create_index(config, index_filepaths)

    # Load query embeddings
    query_dict: Dict[str, np.ndarray] = h5_to_dict(config.h5_paths, query_filepaths)

    # For each query, return the top k results and store in the knn CSV
    knn_retrieval(index, metadata, query_dict, config.knn_csv, config.k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval on embeddings")
    parser.add_argument("config", type=Path, help="Paths to config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute index and metadata",
        default=False,
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = SimpleNamespace(**load(f))

    main(config, args.force)
