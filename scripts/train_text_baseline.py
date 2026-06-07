"""Script for training and benchmarking Text Baseline Ensemble models.

Evaluates combinations of SMOTE and automated weight calculation strategies
on text classification tasks.
"""

from __future__ import annotations

import io
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
from sklearn.metrics import classification_report, f1_score, accuracy_score

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
_DATA_TEST = "data/processed/processed_labeled_text_test.csv"
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "models", "baselines")
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
        X_test, y_test = loader.load(_DATA_TEST)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Failed to load benchmark datasets: %s", exc)
        sys.exit(1)

    logger.info("Label distribution (Train):\n%s", y_train.value_counts().to_string())
    logger.info("Train/Test split: %d train samples | %d test samples", len(X_train), len(X_test))

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

    best = max(results, key=lambda r: r.macro_f1)
    logger.info("Best model configuration: %s → saved to %s", best.exp_id, best.save_path)


if __name__ == "__main__":
    main()