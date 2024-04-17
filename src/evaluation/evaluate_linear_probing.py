import argparse
import json
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


def test_linear_probing(predictions_df, test_df, config):

    test_results = dict()

    # Filter out queries that are not in the predictions
    predictions = predictions_df[predictions_df.index.isin(test_df.index)]

    assert len(predictions) == len(test_df)

    ground_truths = test_df.iloc[:, 4:]
    # Check if the indices are the same
    assert (ground_truths.index == predictions.index).all()

    y_true = ground_truths.to_numpy()
    y_pred = predictions.to_numpy()

    temp_results = evaluate(y_true, y_pred)

    test_results = {
        **test_results,
        **temp_results,
    }

    test_path = Path(config.experiments_dir) / "linear_probing.csv"

    print(f"Writing test results to CSV: {test_path}")
    pd.DataFrame([test_results]).to_csv(test_path, index=None)
    return test_results


def main(experiments_dir: Path):
    # find all config.json files
    config_files = list(experiments_dir.glob("**/config.json"))
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = SimpleNamespace(**json.load(f))

        linear_probing_result_candidates = list(
            Path(config.experiments_dir).glob(f"**/linear_probing_results.csv")
        )
        if not linear_probing_result_candidates:
            print(f"No linear_probing_results.csv found for {config_file}. Skipping...")
            continue

        if (config_file.parent / f"linear_probing.csv").exists():
            # Assuming that also all other files exist
            print(f"Skipping {config_file} as linear_probing.csv already exists")
            continue

        linear_probing_result = linear_probing_result_candidates[-1]

        predictions_df = pd.read_csv(linear_probing_result).set_index("query")
        test_df = pd.read_csv(config.test_csv).set_index("filepath")

        test_linear_probing(predictions_df, test_df, config)

    global_results = dict()
    for config_file in tqdm(config_files):
        if not (config_file.parent / f"linear_probing.csv").exists():
            print(f"Skipping {config_file} as linear_probing.csv does not exist")
            continue

        model = config_file.parent.parent.name
        global_results[model] = pd.read_csv(
            config_file.parent / f"linear_probing.csv"
        ).iloc[0]

    pd.DataFrame(global_results).T.to_csv(experiments_dir / f"linear_probing.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments_dir", type=Path, help="Path to the experiments directory"
    )
    args = parser.parse_args()

    main(args.experiments_dir)
