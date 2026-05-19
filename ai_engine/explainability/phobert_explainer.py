"""
Explainable AI (XAI) Module for PhoBERT.

This module uses Captum (Integrated Gradients) to explain the predictions of a 
PhoBERT text classification model. It identifies the top words that contribute 
to a specific class prediction (e.g., 'Negative'), allowing the frontend to 
highlight them.

Key Upgrades:
- Sub-word Aggregation: Aggregates BPE/WordPiece sub-words (e.g., '@@') back 
  into complete words before returning them.
- API-Ready Structure: Uses dataclasses to return a structured list of explanations.
- Device Management: Ensures seamless tensor movement between CPU and CUDA.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """API-ready data structure for word attribution scores."""
    word: str
    score: float


class PhoBertExplainer:
    """Explainable AI module for PhoBERT using Captum's Integrated Gradients."""

    def __init__(
        self, 
        model_path: str = "vinai/phobert-base-v2",
        num_labels: int = 3
    ) -> None:
        """
        Initialize the explainer.
        
        Args:
            model_path: Path to the pre-trained PhoBERT model or HuggingFace repo.
            num_labels: Number of classification labels.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing PhoBertExplainer on %s...", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        ).to(self.device)
        self.model.eval()

        # Target the embeddings layer for Integrated Gradients
        self.lig = LayerIntegratedGradients(
            self._forward_func, self.model.roberta.embeddings.word_embeddings
        )

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass wrapper for Captum."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def explain(self, text: str, target_class: int = 1, top_k: int = 3) -> List[ExplanationResult]:
        """
        Explain the model's prediction for a given text.

        Args:
            text: The raw input review text.
            target_class: The class index to attribute towards (default 1 for 'Negative').
            top_k: Number of top contributing words to return.

        Returns:
            A list of ExplanationResult containing the top_k words and their scores.
        """
        if not text or not text.strip():
            return []

        # 1. Clean and Tokenize
        clean_text = re.sub(r"\s+", " ", text).strip()
        inputs = self.tokenizer(
            clean_text, return_tensors="pt", truncation=True, max_length=256
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Baseline (reference) input using PAD tokens
        ref_input_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id).to(self.device)
        ref_input_ids[0, 0] = self.tokenizer.cls_token_id
        ref_input_ids[0, -1] = self.tokenizer.sep_token_id

        # 2. Compute Attributions using Integrated Gradients
        try:
            attributions, _ = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                additional_forward_args=(attention_mask,),
                target=target_class,
                return_convergence_delta=True,
            )
        except Exception as e:
            logger.error("Captum attribution failed: %s", e)
            return []

        # Summarize across the embedding dimensions
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        # Normalize scores
        norm = torch.norm(attributions)
        if norm > 0:
            attributions = attributions / norm

        attributions = attributions.cpu().detach().numpy()
        input_ids_list = input_ids[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids_list)

        # 3. Sub-word Aggregation
        word_scores: List[ExplanationResult] = []
        current_word = ""
        current_score = 0.0

        special_tokens = {
            self.tokenizer.cls_token, 
            self.tokenizer.sep_token, 
            self.tokenizer.pad_token
        }

        for token, score in zip(tokens, attributions):
            if token in special_tokens:
                continue

            # Handle PhoBERT's sub-word marker '@@'
            if token.endswith("@@"):
                current_word += token[:-2]
                current_score += float(score)
            else:
                current_word += token
                current_score += float(score)
                
                # PhoBERT uses '_' for spaces within compounded words
                clean_word = current_word.replace("_", " ")
                word_scores.append(ExplanationResult(word=clean_word, score=current_score))
                
                # Reset for next word
                current_word = ""
                current_score = 0.0

        # Sort by score in descending order to get the highest positive contribution 
        # towards the target_class prediction
        word_scores.sort(key=lambda x: x.score, reverse=True)
        return word_scores[:top_k]
