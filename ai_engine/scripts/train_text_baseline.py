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

    # 3. Train Logistic Regression Model (Cost-Sensitive Learning, No SMOTE by default)
    logger.info("\n--- TRAINING LOGISTIC REGRESSION ---")
    lr_model = TextBaselineModel(classifier_type='lr', use_smote=False)
    lr_model.train(X_train, y_train)
    
    # Evaluation
    y_pred_lr = lr_model.predict(X_test)
    logger.info("Classification Report (Logistic Regression - Balanced Weights):")
    logger.info("\n" + classification_report(y_test, y_pred_lr))
    
    # Save Model
    lr_model.save_model('ai_engine/models/weights/text_lr_balanced.pkl')

    # 4. Train Linear SVM Model (For comparison)
    logger.info("\n--- TRAINING LINEAR SVM ---")
    svm_model = TextBaselineModel(classifier_type='svm', use_smote=False)
    svm_model.train(X_train, y_train)
    
    # Evaluation
    y_pred_svm = svm_model.predict(X_test)
    logger.info("Classification Report (SVM - Balanced Weights):")
    logger.info("\n" + classification_report(y_test, y_pred_svm))
    
    # Save Model
    svm_model.save_model('ai_engine/models/weights/text_svm_balanced.pkl')

if __name__ == "__main__":
    main()