"""Training & Benchmarking Script — Text Ensemble Baseline.

Runs four benchmark experiments to evaluate the impact of:
  - SMOTE over-sampling  vs.  cost-sensitive class weights only
  - Automatic F1-weighted voting  vs.  equal (uniform) weights

Experiments
-----------
+--------+----------+--------------+
| Exp ID | SMOTE    | Weights      |
+========+==========+==============+
| EXP-1  | No       | Auto (F1-CV) |
| EXP-2  | Yes      | Auto (F1-CV) |
| EXP-3  | No       | Equal [1,1,1]|
| EXP-4  | Yes      | Equal [1,1,1]|
+--------+----------+--------------+

Artifacts are saved to ``<project_root>/artifacts/models/baselines/``.

Usage
-----
From the project root directory::

    python ai_engine/scripts/train_text_baseline.py

"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of how the script is invoked
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ai_engine.models.text_baseline import TextEnsembleModel  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_PATH: str = os.path.join(_PROJECT_ROOT, "data", "processed", "processed_labeled_reviews.csv")
_ARTIFACTS_DIR: str = os.path.join(_PROJECT_ROOT, "artifacts", "models", "baselines")
_TEXT_COL: str = "cleaned_text"
_LABEL_COL: str = "sentiment_label"
_TEST_SIZE: float = 0.20
_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment.

    Attributes:
        exp_id (str): Short identifier, e.g. ``"EXP-1"``.
        name (str): Human-readable experiment description.
        use_smote (bool): Whether to enable SMOTE over-sampling.
        weights (Optional[List[float]]): Explicit ensemble weights, or
            ``None`` to trigger automatic weight computation.
        save_name (str): Filename for the serialised model artifact.
    """

    exp_id: str
    name: str
    use_smote: bool
    weights: Optional[List[float]]
    save_name: str


@dataclass
class ExperimentResult:
    """Stores evaluation metrics for a completed experiment.

    Attributes:
        exp_id (str): Experiment identifier.
        name (str): Experiment name.
        macro_f1 (float): Macro-averaged F1 score.
        weighted_f1 (float): Weighted-averaged F1 score.
        accuracy (float): Classification accuracy.
        final_weights (Optional[List[float]]): Weights actually used in the
            trained ensemble (relevant when ``weights=None`` → auto-computed).
        elapsed_seconds (float): Wall-clock training duration in seconds.
        save_path (str): Path where the model artifact was saved.
    """

    exp_id: str
    name: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    final_weights: Optional[List[float]]
    elapsed_seconds: float
    save_path: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> Tuple[pd.Series, pd.Series]:
    """Loads the processed CSV and returns feature and label series.

    Args:
        data_path (str): Absolute path to the processed CSV file.

    Returns:
        Tuple[pd.Series, pd.Series]: ``(X, y)`` where *X* is the text column
        and *y* is the sentiment label column.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
        KeyError: If expected columns are missing in the CSV.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from: %s", data_path)
    df = pd.read_csv(data_path)

    missing_cols = {_TEXT_COL, _LABEL_COL} - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=[_TEXT_COL, _LABEL_COL])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN in '%s' or '%s'.", dropped, _TEXT_COL, _LABEL_COL)

    logger.info("Loaded %d samples.", len(df))
    return df[_TEXT_COL], df[_LABEL_COL]


def print_comparison_table(results: List[ExperimentResult]) -> None:
    """Renders a formatted comparison table to stdout.

    Args:
        results (List[ExperimentResult]): List of completed experiment results.
    """
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
    print(f"\n  ★ Best macro-F1: {best.exp_id} — {best.name} ({best.macro_f1:.4f})")
    if best.final_weights:
        print(f"     Ensemble weights used [LR, SVM, RF]: {best.final_weights}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Experiments configuration
# ---------------------------------------------------------------------------

EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        exp_id="EXP-1",
        name="No SMOTE + Auto-Weights (F1-CV)",
        use_smote=False,
        weights=None,           # triggers compute_auto_weights()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    artifacts_dir: str,
) -> ExperimentResult:
    """Trains, evaluates, and saves a single experiment.

    Args:
        config (ExperimentConfig): Experiment hyper-parameters.
        X_train (pd.Series): Training text features.
        y_train (pd.Series): Training labels.
        X_test (pd.Series): Test text features.
        y_test (pd.Series): Test labels.
        artifacts_dir (str): Directory where the model artifact is saved.

    Returns:
        ExperimentResult: Evaluation metrics and metadata for the run.
    """
    separator = "=" * 70
    logger.info("\n%s\n  %s — %s\n%s", separator, config.exp_id, config.name, separator)

    model = TextEnsembleModel(
        use_smote=config.use_smote,
        weights=config.weights,
        random_state=_RANDOM_STATE,
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

    save_path = os.path.join(artifacts_dir, config.save_name)
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


def main() -> None:
    """Entry point for the ensemble benchmark training pipeline.

    Steps:
        1. Load and validate the processed dataset.
        2. Perform a stratified 80/20 train/test split.
        3. Run all configured experiments sequentially.
        4. Print a formatted comparison table.
        5. Report the best-performing configuration.
    """
    logger.info("=" * 70)
    logger.info("  STARTING — Weighted Soft-Voting Ensemble Benchmark")
    logger.info("=" * 70)

    # 1. Load data
    try:
        X, y = load_data(_DATA_PATH)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    logger.info("Label distribution:\n%s", y.value_counts().to_string())

    # 2. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train/Test split: %d train samples | %d test samples",
        len(X_train),
        len(X_test),
    )

    # 3. Ensure artifacts directory exists
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

    # 4. Run experiments
    results: List[ExperimentResult] = []
    for config in EXPERIMENTS:
        result = run_experiment(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            artifacts_dir=_ARTIFACTS_DIR,
        )
        results.append(result)

    # 5. Summary
    print_comparison_table(results)

    best = max(results, key=lambda r: r.macro_f1)
    logger.info(
        "Best model: %s → saved to %s",
        best.exp_id,
        best.save_path,
    )


if __name__ == "__main__":
    main()