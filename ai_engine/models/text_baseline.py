import os
import joblib
import logging
import pandas as pd
from typing import Any, Optional
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextBaselineModel:
    """A baseline machine learning model for text classification.
    
    This class encapsulates a text processing and classification pipeline
    using TF-IDF, optional SMOTE for oversampling, and linear classifiers.
    Prioritizes cost-sensitive learning via class_weight='balanced'.
    
    Attributes:
        classifier_type (str): The type of classifier to use ('lr' for Logistic Regression, 'svm' for Linear SVM).
        random_state (int): The seed used by the random number generator.
        use_smote (bool): Whether to use SMOTE for oversampling. Defaults to False.
        pipeline (ImbPipeline): The scikit-learn compatible pipeline.
    """

    def __init__(self, classifier_type: str = 'lr', random_state: int = 42, use_smote: bool = False) -> None:
        """Initializes the TextBaselineModel.

        Args:
            classifier_type (str, optional): The classifier algorithm ('lr' or 'svm'). Defaults to 'lr'.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            use_smote (bool, optional): If True, incorporates SMOTE into the pipeline. Defaults to False.
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.use_smote = use_smote
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> ImbPipeline:
        """Constructs the modeling pipeline with TF-IDF, optional sampling, and a classifier.

        Returns:
            ImbPipeline: The configured imbalanced-learn pipeline.
            
        Raises:
            ValueError: If an unsupported classifier type is provided.
        """
        logger.info(f"Building pipeline with classifier_type='{self.classifier_type}', use_smote={self.use_smote}")
        
        # 1. Initialize TF-IDF Vectorizer (Unigram & Bigram)
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=5,       
            max_df=0.8      
        )

        # 2. Initialize Classifier with explicit cost-sensitive learning
        if self.classifier_type == 'svm':
            clf = LinearSVC(random_state=self.random_state, class_weight='balanced', dual="auto")
        elif self.classifier_type in ['lr', 'logreg']:
            clf = LogisticRegression(random_state=self.random_state, class_weight='balanced', max_iter=1000, solver='lbfgs', n_jobs=-1)
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

        # 3. Assemble Pipeline
        steps: list[tuple[str, Any]] = [('tfidf', tfidf)]
        
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            steps.append(('smote', smote))
            
        steps.append(('classifier', clf))
        
        return ImbPipeline(steps=steps)

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Trains the model pipeline on the provided training data.

        Args:
            X_train (pd.Series): The training text features.
            y_train (pd.Series): The training target labels.
        """
        logger.info(f"Training {self.classifier_type.upper()} model...")
        self.pipeline.fit(X_train, y_train)
        logger.info("Training completed.")

    def predict(self, X_test: pd.Series) -> np.ndarray:
        """Predicts the labels for the given test data.

        Args:
            X_test (pd.Series): The testing text features to predict on.

        Returns:
            np.ndarray: The predicted labels.
        """
        return self.pipeline.predict(X_test)

    def save_model(self, filepath: str) -> None:
        """Saves the trained pipeline to a local file.

        Args:
            filepath (str): The destination path to save the model.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.pipeline, filepath)
            logger.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise

    @classmethod
    def load_model(cls, filepath: str) -> 'TextBaselineModel':
        """Loads a model instance from a saved file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            TextBaselineModel: The reinstated TextBaselineModel instance.
        """
        logger.info(f"Loading model from {filepath}")
        try:
            pipeline = joblib.load(filepath)
            instance = cls()
            instance.pipeline = pipeline
            return instance
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise