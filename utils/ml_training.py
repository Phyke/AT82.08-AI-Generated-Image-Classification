import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utils.types import FeatureMatrix_float32


def shuffle_indexes(
    X_train: FeatureMatrix_float32 | pd.DataFrame,
    y_train: FeatureMatrix_float32 | pd.Series,
) -> tuple:
    rng = np.random.RandomState(42)
    shuffle_indices = rng.permutation(len(X_train))
    X_train = X_train.iloc[shuffle_indices]
    y_train = y_train.iloc[shuffle_indices]
    return X_train, y_train


def save_and_get_model_size_joblib(model: object, filename: str) -> float:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # Save model
    joblib.dump(model, filename, compress=("lzma", 3))
    # Measure size using pathlib
    return Path(filename).stat().st_size / (1024**2)  # in MB


def run_ml_experiments(  # noqa: PLR0913
    X_train: FeatureMatrix_float32,
    y_train: FeatureMatrix_float32,
    X_test: NDArray[np.uint8],
    y_test: NDArray[np.uint8],
    path_output: Path,
    feature_set_name: str,
) -> dict[str, dict]:
    models = {
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
        "LogisticRegression": LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=42),
        "LGBMClassifier": LGBMClassifier(n_jobs=-1, random_state=42, force_col_wise=True, verbose=-1),
    }

    results = {}
    for name, model in models.items():
        # Train
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        total_time_train = end_train - start_train
        # Predict / metrics
        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        total_time_pred = end_pred - start_pred
        report_str = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
            target_names=["Real", "Fake"],
            labels=[0, 1],
            output_dict=False,
        )

        report = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
            target_names=["Real", "Fake"],
            labels=[0, 1],
            output_dict=True,
        )

        # Model size (joblib)
        size_mb = save_and_get_model_size_joblib(model, f"outputs/models/{feature_set_name}/{name}.joblib")
        print("\n--------------------------------------------------")
        print(f"{name}:")
        print(f"Training time: {total_time_train:.2f} seconds for {len(X_train)} samples")
        print(f"Prediction time: {total_time_pred:.2f} seconds for {len(X_test)} samples")
        print(f"Model size (joblib): {size_mb:.3f} MB")
        print(report_str)
        results[name] = {
            "training_time": total_time_train,
            "prediction_time": total_time_pred,
            "report": report,
            "report_str": report_str,
            "model_size_mb": size_mb,
        }

    with (path_output / "models" / feature_set_name / "metrics.json").open("w") as f:
        json.dump(results, f, indent=4)

    return results


def extract_metrics(results: dict) -> tuple:
    models = list(results.keys())
    accuracy = [results[m]["report"]["accuracy"] for m in models]
    precision = [results[m]["report"]["macro avg"]["precision"] for m in models]
    recall = [results[m]["report"]["macro avg"]["recall"] for m in models]
    f1 = [results[m]["report"]["macro avg"]["f1-score"] for m in models]
    training_time = [results[m]["training_time"] for m in models]
    model_sizes = [results[m]["model_size_mb"] for m in models]
    prediction_time = [results[m]["prediction_time"] for m in models]
    return models, accuracy, precision, recall, f1, training_time, model_sizes, prediction_time


def plot_classification_metrics(results: dict) -> None:
    models, accuracy, precision, recall, f1, *_ = extract_metrics(results)
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precision, width, label="Precision")
    plt.bar(x + 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 1.5 * width, f1, width, label="F1-score")

    plt.xticks(x, models, rotation=45)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_and_prediction_times(results: dict) -> None:
    models, *_, training_time, prediction_time = extract_metrics(results)

    plt.figure(figsize=(10, 5))
    plt.bar(models, training_time)
    plt.bar(models, prediction_time, bottom=training_time)
    plt.xticks(rotation=45)
    plt.ylabel("Training Time (seconds)")
    plt.title("Model Training Time Comparison")
    plt.tight_layout()
    plt.show()


def plot_model_sizes(results: dict) -> None:
    models, *_, model_sizes = extract_metrics(results)[0], extract_metrics(results)[-1]

    # corrected extraction
    models, _, _, _, _, _, model_sizes = extract_metrics(results)

    plt.figure(figsize=(10, 5))
    plt.bar(models, model_sizes)
    plt.xticks(rotation=45)
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size Comparison")
    plt.tight_layout()
    plt.show()
