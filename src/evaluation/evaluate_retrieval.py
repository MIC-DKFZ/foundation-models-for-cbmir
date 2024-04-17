import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm
import argparse


def get_modified_mean_precision_at_n(data, N, label_occurrences):
    # Calculate precision at N for each query
    def precision_at_n(row):
        # Get the actual number of available class instances
        available_instances = label_occurrences[row.name]
        # Use the minimum of N and available instances for precision calculation
        n = min(N, available_instances)
        # Count how many of the top n neighbors are of the same class as the query
        correct_predictions = sum(row[:n])
        # Calculate precision
        return correct_predictions / n

    # Apply the precision_at_n function to each row and calculate the mean
    precision = data.apply(precision_at_n, axis=1)
    micro_precision_mean = precision.mean()
    micro_precision_std = precision.std()
    # Calculate the mean precision for each class

    precision_per_class_mean = precision.groupby(data.index).mean()
    precision_per_class_std = precision.groupby(data.index).std()
    macro_precision_mean = precision_per_class_mean.mean()
    macro_precision_std = precision_per_class_std.mean()
    return (
        micro_precision_mean,
        micro_precision_std,
        macro_precision_mean,
        macro_precision_std,
    )


def main(experiments_dir: Path):
    # find all config.json files
    config_files = list(experiments_dir.glob("**/config.json"))

    # print all config.json files
    for config_file in config_files:
        print(config_file)

    for config_file in tqdm(config_files):
        # Load config
        with open(config_file, "r") as f:
            config = SimpleNamespace(**json.load(f))

        path = Path(config.experiments_dir) / f"retrieval.csv"
        if path.exists():
            print(f"Skipping {config_file} as {path} already exists")
            continue

        if not Path(config.knn_csv).exists():
            print(f"Skipping {config_file} as {config.knn_csv} does not exist")
            continue

        # Read the retrieved CSV file and set the index to the filepath
        query_df = pd.read_csv(config.test_csv).set_index("filepath")

        index_df = pd.read_csv(config.train_csv).set_index("filepath")

        knn_df = pd.read_csv(config.knn_csv).set_index("query")
        knn_df = knn_df.loc[query_df.index]

        label_occurrences = index_df[config.label_columns].sum().astype(int)

        # Create a query_df[filepath] to label mapping
        query_df_mapping = query_df[config.label_columns].idxmax(axis=1)
        index_df_mapping = index_df[config.label_columns].idxmax(axis=1)

        # map the index of knn_df to the label
        knn_df.index = knn_df.index.map(query_df_mapping.get)
        # map each element of knn_df to the label (this step takes some time)
        knn_df = knn_df.applymap(index_df_mapping.get)

        # Boolean mask to check if the index is equal to the column
        knn_df = knn_df.apply(lambda x: x == knn_df.index, axis=0)

        results = {
            "micro_precision_mean": [],
            "micro_precision_std": [],
            "macro_precision_mean": [],
            "macro_precision_std": [],
        }
        for N in tqdm(range(1, 100)):
            (
                micro_precision_mean,
                micro_precision_std,
                macro_precision_mean,
                macro_precision_std,
            ) = get_modified_mean_precision_at_n(knn_df, N, label_occurrences)
            results["micro_precision_mean"].append(micro_precision_mean)
            results["micro_precision_std"].append(micro_precision_std)
            results["macro_precision_mean"].append(macro_precision_mean)
            results["macro_precision_std"].append(macro_precision_std)

        print(f"Writing results to CSV: {path}")
        pd.DataFrame(results).to_csv(path)

    # Single Chart across models
    results = {}
    for config_file in tqdm(config_files):
        # Load config
        with open(config_file, "r") as f:
            config = SimpleNamespace(**json.load(f))

        if not (Path(config.experiments_dir) / "retrieval.csv").exists():
            print(f"Skipping {config_file} as retrieval.csv does not exist")
            continue

        retrieval_precisions = pd.read_csv(
            Path(config.experiments_dir) / "retrieval.csv"
        )
        model = Path(config.experiments_dir).name

        results[model] = dict(
            micro_precison_at_1=retrieval_precisions["micro_precision_mean"].iloc[0],
            micro_precision_at_3=retrieval_precisions["micro_precision_mean"].iloc[2],
            micro_precision_at_5=retrieval_precisions["micro_precision_mean"].iloc[4],
            micro_precision_at_10=retrieval_precisions["micro_precision_mean"].iloc[9],
            macro_precision_at_1=retrieval_precisions["macro_precision_mean"].iloc[0],
            macro_precision_at_3=retrieval_precisions["macro_precision_mean"].iloc[2],
            macro_precision_at_5=retrieval_precisions["macro_precision_mean"].iloc[4],
            macro_precision_at_10=retrieval_precisions["macro_precision_mean"].iloc[9],
        )

    pd.DataFrame(results).T.to_csv(experiments_dir / "retrieval.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for retrieval pathology"
    )
    parser.add_argument(
        "experiments_dir", type=Path, help="Path to the experiments directory"
    )
    args = parser.parse_args()

    main(args.experiments_dir)
