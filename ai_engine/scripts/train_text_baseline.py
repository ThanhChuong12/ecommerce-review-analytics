import os
import sys
import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Thêm root path để import được module ai_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ai_engine.models.text_baseline import TextBaselineModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> Tuple[pd.Series, pd.Series]:
    """Loads and preprocesses the dataset.

    Args:
        data_path (str): The path to the processed csv.

    Returns:
        Tuple[pd.Series, pd.Series]: Features (X) and target labels (y).
        
    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    # Ensure missing text or labels are dropped
    df = df.dropna(subset=['cleaned_text', 'sentiment_label'])
    
    return df['cleaned_text'], df['sentiment_label']

def main() -> None:
    """Main pipeline for training baseline text classification models."""
    logger.info("--- STARTING TEXT BASELINE TRAINING PIPELINE ---")
    
    # 1. Load cleaned data from Phase 1
    data_path = 'data/processed/processed_labeled_reviews.csv'
    try:
        X, y = load_data(data_path)
    except FileNotFoundError:
        return

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # 2. Train/Test Split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Data split successfully with stratified sampling.")

    # Output directory for saving model artifacts
    artifacts_dir = 'artifacts/models'
    os.makedirs(artifacts_dir, exist_ok=True)

    # 3. Multi-Model Training Block for A/B Comparison
    configurations = [
        {
            'name': 'Logistic Regression (Cost-Sensitive)',
            'classifier_type': 'logreg',
            'use_smote': False,
            'save_name': 'text_lr_balanced.pkl'
        },
        {
            'name': 'Linear SVM (Cost-Sensitive)',
            'classifier_type': 'svm',
            'use_smote': False,
            'save_name': 'text_svm_balanced.pkl'
        },
        {
            'name': 'Logistic Regression (SMOTE)',
            'classifier_type': 'logreg',
            'use_smote': True,
            'save_name': 'text_lr_smote.pkl'
        }
    ]

    for config in configurations:
        logger.info(f"\n{'='*60}\n--- TRAINING {config['name'].upper()} ---\n{'='*60}")
        
        # Initialize and Train
        model = TextBaselineModel(
            classifier_type=config['classifier_type'], 
            use_smote=config['use_smote']
        )
        model.train(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        logger.info(f"Classification Report - {config['name']}:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Save Artifact
        save_path = os.path.join(artifacts_dir, config['save_name'])
        model.save_model(save_path)

if __name__ == "__main__":
    main()