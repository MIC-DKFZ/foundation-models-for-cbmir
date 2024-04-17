import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def knn_classification(knn_df, query_df, index_df, N, label_occurrences):
    # a the row contains the top 100 label paths
    # We look their labels up in the index_df and sum them up
    # we return the propabilties per class
    def kNN(row):
        # get row label
        label = pd.to_numeric(query_df.loc[row.name].iloc[4:]).idxmax()
        # Get the actual number of available class instances
        available_instances = label_occurrences[label]
        # Use the minimum of N and available instances for precision calculation
        n = min(N, available_instances)
        # Map each entry of row[:n] using index_mapping
        y_pred = index_mapping.reindex(row[:n]).to_numpy().mean(axis=0)
        return y_pred

    # index mapping
    index_mapping = index_df.iloc[:, 4:]

    # Apply the kNN to each row and calculate the mean
    # predictions now contains the probabilities for each class
    predictions = knn_df.apply(kNN, axis=1, result_type="expand")
    predictions.columns = index_df.columns[4:]

    return predictions


def get_best_k(knn_df, val_df, index_df, config):
    label_occurrences = index_df[config.label_columns].sum().astype(int)
    val_path = Path(config.experiments_dir) / "knn_validation_val.csv"

    if val_path.exists():
        val_results_df = pd.read_csv(val_path)
        best_k = val_results_df.micro_f1.idxmax()
        print(f"Best k from previous run: {best_k}")
        return best_k
    else:
        print(f"File {val_path} does not exist. Calculating best k.")

    #####################################
    # Determine best k on valiation set #
    #####################################
    val_results = defaultdict(list)
    ground_truths = val_df.iloc[:, 4:].reindex(knn_df.index).loc[val_df.index]

    for k in tqdm(range(1, knn_df.shape[-1])):
        predictions = knn_classification(
            knn_df.loc[val_df.index], val_df, index_df, k, label_occurrences
        )

        y_true = ground_truths.to_numpy()
        y_pred = predictions.to_numpy()

        temp_res = evaluate(y_true, y_pred)

        # append to results
        for key, value in temp_res.items():
            val_results[key].append(value)

    val_results_df = pd.DataFrame(val_results)
    print(f"Writing val results to CSV: {val_path}")
    val_results_df.to_csv(val_path)

    # get argmax for micro_f1
    best_k = val_results_df.micro_f1.idxmax()

    return best_k


def evaluate(y_true, y_pred):
    test_results = dict(
        micro_roc_auc=roc_auc_score(
            y_true,
            y_pred,
            average="micro",
            multi_class="ovr",
        ),
        micro_f1=f1_score(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            average="micro",
        ),
        macro_roc_auc=roc_auc_score(
            y_true,
            y_pred,
            average="macro",
            multi_class="ovr",
        ),
        macro_f1=f1_score(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            average="macro",
        ),
        macro_precision=precision_score(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            average="macro",
        ),
        macro_recall=recall_score(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            average="macro",
        ),
        macro_accuracy=accuracy_score(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
        ),
    )
    return test_results


def test_knn_classification(knn_df, index_df, best_k, config):
    label_occurrences = index_df[config.label_columns].sum().astype(int)
    test_df = pd.read_csv(config.test_csv).set_index("filepath")

    test_results = dict()

    predictions = knn_classification(
        knn_df.loc[test_df.index],
        test_df,
        index_df,
        best_k,
        label_occurrences,
    )
    ground_truths = test_df.iloc[:, 4:].reindex(knn_df.index).loc[test_df.index]

    y_true = ground_truths.to_numpy()
    y_pred = predictions.to_numpy()

    temp_results = evaluate(y_true, y_pred)
    test_results = {
        **test_results,
        **temp_results,
    }

    test_path = Path(config.experiments_dir) / "knn_classification.csv"
    print(f"Writing test results to CSV: {test_path}")
    pd.DataFrame([test_results]).to_csv(test_path, index=None)
    return test_results


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

        if not Path(config.knn_csv).exists():
            print(f"Skipping {config_file} as {config.knn_csv} does not exist")
            continue

        if (config_file.parent / f"knn_classification.csv").exists():
            # Assuming that also all other files exist
            print(f"Skipping {config_file} as knn_classification.csv already exists")
            continue

        knn_df = pd.read_csv(config.knn_csv).set_index("query")

        # Read the retrieved CSV file and set the index to the filepath
        val_df = pd.read_csv(config.val_csv).set_index("filepath")
        index_df = pd.read_csv(config.train_csv).set_index("filepath")

        # Determine best k on the validation set
        best_k = get_best_k(knn_df, val_df, index_df, config)

        ########################
        # Evaluate on test set #
        ########################
        test_knn_classification(knn_df, index_df, best_k, config)

    global_results = dict()

    for config_file in tqdm(config_files):

        if not (config_file.parent / "knn_classification.csv").exists():
            print(f"Skipping {config_file} as knn_classification.csv does not exist")
            continue

        model = config_file.parent.name

        global_results[model] = pd.read_csv(
            config_file.parent / "knn_classification.csv"
        ).iloc[0]

    pd.DataFrame(global_results).T.to_csv(experiments_dir / "knn_classification.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments_dir", type=Path, help="Path to the experiments directory"
    )
    args = parser.parse_args()

    main(args.experiments_dir)
