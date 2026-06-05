"""
PhoBERT Sentiment Classification – Model Deployment and Inference Wrapper.

This module provides ``PhoBertSentimentModel``, a production-grade wrapper class
for loading, deploying, and running real-time inference on the trained PhoBERT 
sentiment model checkpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class PhoBertSentimentModel:
    """Wrapper class for deploying and inferring with the trained PhoBERT Sentiment model.

    Provides a clean, scikit-learn-like API (predict, predict_proba) that acts
    identically to the baseline models, facilitating seamless model switching 
    in the web platform or evaluation pipelines.
    """

    def __init__(self, model_dir: str) -> None:
        """Initialises the PhoBERT sentiment wrapper.

        Args:
            model_dir: Path to the directory containing saved model weights,
                config, and tokenizer files (e.g., "ai_engine/models/weights/phobert_best").
        
        Raises:
            FileNotFoundError: If the model directory or essential files do not exist.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")

        logger.info("Loading PhoBERT model and tokenizer from %s ...", model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device for inference: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode for inference stability

        # Propagate mappings from configuration
        self.id2label: Dict[int, str] = self.model.config.id2label
        self.label2id: Dict[str, int] = self.model.config.label2id
        logger.info("Model loaded successfully. Mappings: %s", self.id2label)

    def predict(self, texts: Union[str, List[str]], max_length: int = 256) -> Union[str, List[str]]:
        """Predicts the sentiment label for one or multiple reviews.

        Args:
            texts: A single raw string review or a list of review strings.
            max_length: Maximum tokenization sequence length. Defaults to 256.

        Returns:
            The predicted label string (e.g. "tích cực", "tiêu cực", "trung lập"),
            or a list of label strings if the input was a list.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        probs = self.predict_proba(texts, max_length=max_length)
        pred_ids = np.argmax(probs, axis=-1)
        preds = [self.id2label[int(pid)] for pid in pred_ids]

        return preds[0] if is_single else preds

    def predict_proba(self, texts: Union[str, List[str]], max_length: int = 256) -> np.ndarray:
        """Predicts class probability distributions for one or multiple reviews.

        Args:
            texts: A single raw string review or a list of review strings.
            max_length: Maximum tokenization sequence length. Defaults to 256.

        Returns:
            A numpy array of shape (num_samples, num_classes) containing predicted
            probabilities for each class. If input is a single string, returns a
            1D array of shape (num_classes,).
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Light preprocessing suitable for PhoBERT (collapsing redundant spaces)
        cleaned_texts = [
            " ".join(str(t).strip().split()) if t is not None else ""
            for t in texts
        ]

        encodings = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move tensors to active inference device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Softmax to obtain actual probability distributions
            probs = F.softmax(outputs.logits, dim=-1)

        probs_np = probs.cpu().numpy()
        return probs_np[0] if is_single else probs_np
