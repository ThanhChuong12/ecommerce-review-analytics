"""Script for training and benchmarking Text Baseline Ensemble models.

Evaluates combinations of SMOTE and automated weight calculation strategies
on text classification tasks, generates learning curves, and saves metrics.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Set UTF-8 encoding for standard streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ai_engine.models.text_baseline import TextEnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
_DATA_TRAIN = "data/processed/processed_labeled_text_train.csv"
_DATA_VAL = "data/processed/processed_labeled_text_val.csv"
_DATA_TEST = "data/processed/processed_labeled_text_test.csv"

_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "models", "baselines")
_METRICS_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "metrics")
_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "plots")

_TEXT_COL = "cleaned_text"
_LABEL_COL = "sentiment_label"
_RANDOM_STATE = 42


@dataclass
class ExperimentConfig:
    """Configuration options for a single experiment run."""
    exp_id: str
    name: str
    use_smote: bool
    weights: Optional[List[float]]
    save_name: str


@dataclass
class ExperimentResult:
    """Evaluation metrics and results of a completed experiment."""
    exp_id: str
    name: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    final_weights: Optional[List[float]]
    elapsed_seconds: float
    save_path: str


class DatasetLoader:
    """Handles loading and basic validation of the dataset."""

    def __init__(self, text_col: str = _TEXT_COL, label_col: str = _LABEL_COL) -> None:
        self.text_col = text_col
        self.label_col = label_col

    def load(self, data_path: str) -> Tuple[pd.Series, pd.Series]:
        """Loads features and labels from the CSV file, removing empty rows."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("Loading dataset from: %s", data_path)
        df = pd.read_csv(data_path)

        required_cols = {self.text_col, self.label_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns in dataset: {missing}")

        initial_len = len(df)
        df = df.dropna(subset=[self.text_col, self.label_col])
        dropped = initial_len - len(df)
        if dropped:
            logger.warning("Dropped %d null rows in text or label columns.", dropped)

        logger.info("Loaded %d clean samples.", len(df))
        return df[self.text_col], df[self.label_col]


class ExperimentRunner:
    """Executes experiments and calculates evaluation metrics."""

    def __init__(self, artifacts_dir: str, random_state: int = _RANDOM_STATE) -> None:
        self.artifacts_dir = artifacts_dir
        self.random_state = random_state

    def run(
        self,
        config: ExperimentConfig,
        X_train: pd.Series,
        y_train: pd.Series,
        X_test: pd.Series,
        y_test: pd.Series,
    ) -> ExperimentResult:
        """Trains the ensemble model, evaluates performance, and saves artifacts."""
        separator = "=" * 70
        logger.info("\n%s\n  %s — %s\n%s", separator, config.exp_id, config.name, separator)

        model = TextEnsembleModel(
            use_smote=config.use_smote,
            weights=config.weights,
            random_state=self.random_state,
        )

        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t_start

        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(
            "\nClassification Report — %s:\n%s",
            config.name,
            classification_report(y_test, y_pred),
        )

        save_path = os.path.join(self.artifacts_dir, config.save_name)
        model.save(save_path)

        return ExperimentResult(
            exp_id=config.exp_id,
            name=config.name,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            accuracy=accuracy,
            final_weights=model.weights,
            elapsed_seconds=elapsed,
            save_path=save_path,
        )


class LearningCurvePlotter:
    """Computes and plots learning curves (Loss, F1-macro, Accuracy) over training set sizes."""

    def __init__(self, model_config: ExperimentConfig, random_state: int = _RANDOM_STATE) -> None:
        self.config = model_config
        self.random_state = random_state

    def compute_and_plot(
        self,
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series,
        y_val: pd.Series,
        plot_dir: str,
    ) -> None:
        """Trains the model on fractions of the training data and plots metric curves."""
        logger.info("Computing learning curves for the best model: %s", self.config.name)

        train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        train_sizes = []

        metrics = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_acc": [], "val_acc": []
        }

        # Identify unique classes for log loss validation
        classes = np.unique(y_train)

        for frac in train_fractions:
            n_samples = int(len(X_train) * frac)
            train_sizes.append(n_samples)

            X_sub = X_train.iloc[:n_samples]
            y_sub = y_train.iloc[:n_samples]

            model = TextEnsembleModel(
                use_smote=self.config.use_smote,
                weights=self.config.weights,
                random_state=self.random_state,
            )
            model.fit(X_sub, y_sub)

            # Performance on the training subset
            y_train_pred = model.predict(X_sub)
            y_train_proba = model.predict_proba(X_sub)

            # Performance on the validation set
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)

            metrics["train_f1"].append(f1_score(y_sub, y_train_pred, average="macro"))
            metrics["val_f1"].append(f1_score(y_val, y_val_pred, average="macro"))

            metrics["train_acc"].append(accuracy_score(y_sub, y_train_pred))
            metrics["val_acc"].append(accuracy_score(y_val, y_val_pred))

            metrics["train_loss"].append(log_loss(y_sub, y_train_proba, labels=classes))
            metrics["val_loss"].append(log_loss(y_val, y_val_proba, labels=classes))

        # Generate plots
        os.makedirs(plot_dir, exist_ok=True)

        # Plot 1: Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, metrics["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
        plt.plot(train_sizes, metrics["val_loss"], label="Val Loss", marker="o", color="#ff7f0e")
        plt.title(f"Baseline Loss Curve ({self.config.exp_id})", fontsize=14, pad=15)
        plt.xlabel("Training Samples", fontsize=12)
        plt.ylabel("Log Loss (Cross-Entropy)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        loss_plot_path = os.path.join(plot_dir, "baseline_loss_curve.png")
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        logger.info("Saved Loss Curve to %s", loss_plot_path)

        # Plot 2: F1-Macro & Accuracy Curve
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, metrics["train_f1"], label="Train F1-Macro", marker="o", color="#1f77b4")
        plt.plot(train_sizes, metrics["val_f1"], label="Val F1-Macro", marker="s", color="#ff7f0e")
        plt.plot(train_sizes, metrics["train_acc"], label="Train Accuracy", linestyle="--", marker="x", color="#2ca02c")
        plt.plot(train_sizes, metrics["val_acc"], label="Val Accuracy", linestyle="--", marker="d", color="#d62728")
        plt.title(f"Baseline Learning Curves ({self.config.exp_id})", fontsize=14, pad=15)
        plt.xlabel("Training Samples", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        metrics_plot_path = os.path.join(plot_dir, "baseline_learning_curve.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        logger.info("Saved Learning Curve to %s", metrics_plot_path)


class ConfusionMatrixPlotter:
    """Generates and saves confusion matrix heatmaps."""

    @staticmethod
    def plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        title: str,
        output_path: str,
    ) -> None:
        """Generates a styled confusion matrix heatmap and saves it to disk."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Map labels to Vietnamese standard ordering if needed (negative -> neutral -> positive)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 12}
        )
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("Actual Label", fontsize=12)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Saved Confusion Matrix to %s", output_path)


class BenchmarkReporter:
    """Generates comparison reports and tables for benchmark results."""

    @staticmethod
    def print_table(results: List[ExperimentResult]) -> None:
        """Prints a structured comparison table of all results to stdout."""
        col_widths = [8, 40, 10, 12, 10, 12]
        header = ["Exp ID", "Name", "Macro-F1", "Weighted-F1", "Accuracy", "Time (s)"]
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        row_fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"

        print("\n" + "=" * 90)
        print("  BENCHMARK COMPARISON — Weighted Soft-Voting Ensemble".center(90))
        print("=" * 90)
        print(sep)
        print(row_fmt.format(*header))
        print(sep)
        for r in results:
            print(
                row_fmt.format(
                    r.exp_id,
                    r.name[:40],
                    f"{r.macro_f1:.4f}",
                    f"{r.weighted_f1:.4f}",
                    f"{r.accuracy:.4f}",
                    f"{r.elapsed_seconds:.1f}s",
                )
            )
        print(sep)
        best = max(results, key=lambda r: r.macro_f1)
        print(f"\n  [BEST] Best macro-F1: {best.exp_id} - {best.name} ({best.macro_f1:.4f})")
        if best.final_weights:
            print(f"     Ensemble weights used [LR, SVM, RF]: {best.final_weights}")
        print("=" * 90 + "\n")


# Benchmark experiments configuration list
EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        exp_id="EXP-1",
        name="No SMOTE + Auto-Weights (F1-CV)",
        use_smote=False,
        weights=None,
        save_name="ensemble_no_smote_auto_weights.pkl",
    ),
    ExperimentConfig(
        exp_id="EXP-2",
        name="SMOTE + Auto-Weights (F1-CV)",
        use_smote=True,
        weights=None,
        save_name="ensemble_smote_auto_weights.pkl",
    ),
    ExperimentConfig(
        exp_id="EXP-3",
        name="No SMOTE + Equal Weights [1,1,1] (Control)",
        use_smote=False,
        weights=[1.0, 1.0, 1.0],
        save_name="ensemble_no_smote_equal_weights.pkl",
    ),
    ExperimentConfig(
        exp_id="EXP-4",
        name="SMOTE + Equal Weights [1,1,1] (Control)",
        use_smote=True,
        weights=[1.0, 1.0, 1.0],
        save_name="ensemble_smote_equal_weights.pkl",
    ),
]


def main() -> None:
    """Benchmark suite entry point."""
    logger.info("=" * 70)
    logger.info("  STARTING — Weighted Soft-Voting Ensemble Benchmark")
    logger.info("=" * 70)

    loader = DatasetLoader()
    try:
        X_train, y_train = loader.load(_DATA_TRAIN)
        X_val, y_val = loader.load(_DATA_VAL)
        X_test, y_test = loader.load(_DATA_TEST)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Failed to load benchmark datasets: %s", exc)
        sys.exit(1)

    logger.info("Label distribution (Train):\n%s", y_train.value_counts().to_string())
    logger.info("Train/Val/Test split: %d train | %d val | %d test", len(X_train), len(X_val), len(X_test))

    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

    runner = ExperimentRunner(artifacts_dir=_ARTIFACTS_DIR)
    results: List[ExperimentResult] = []

    for config in EXPERIMENTS:
        result = runner.run(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(result)

    BenchmarkReporter.print_table(results)

    # Identify the best experiment model based on macro F1-score
    best = max(results, key=lambda r: r.macro_f1)
    logger.info("Best model configuration: %s → saved to %s", best.exp_id, best.save_path)

    # -------------------------------------------------------------------------
    # Evaluation Plots for the Best Model Configuration
    # -------------------------------------------------------------------------
    # Generate and save Learning Curves (F1, Loss, Accuracy) for the best setup
    best_config = next(c for c in EXPERIMENTS if c.exp_id == best.exp_id)
    plotter = LearningCurvePlotter(best_config)
    plotter.compute_and_plot(X_train, y_train, X_val, y_val, plot_dir=_PLOTS_DIR)

    # Load the trained best model to evaluate and plot the confusion matrix on the test set
    best_model = TextEnsembleModel.load(best.save_path)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    # Define standard sentiment class ordering from negative to positive
    sentiment_labels = ["tiêu cực", "trung lập", "tích cực"]
    cm_output_path = os.path.join(_PLOTS_DIR, "baseline_confusion_matrix.png")
    ConfusionMatrixPlotter.plot(
        y_true=y_test.values,
        y_pred=y_test_pred,
        labels=sentiment_labels,
        title=f"Baseline Confusion Matrix ({best.exp_id})",
        output_path=cm_output_path,
    )

    # Calculate and save quantitative test metrics to JSON
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "f1_macro": float(f1_score(y_test, y_test_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_test_pred, average="weighted")),
        "precision_macro": float(precision_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, y_test_proba, labels=sentiment_labels)),
    }

    os.makedirs(_METRICS_DIR, exist_ok=True)
    metrics_output_path = os.path.join(_METRICS_DIR, "baseline_test_metrics.json")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved baseline test metrics to %s", metrics_output_path)


if __name__ == "__main__":
    main()