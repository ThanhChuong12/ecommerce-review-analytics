"""Unit tests for Cross-Modal Fusion Engine v2.0.

Run:
    python -m unittest tests/test_fusion_engine.py -v
"""

import unittest
from ai_engine.fusion.fusion_engine import (
    TrustScoreCalculator,
    FusionInput,
    TextProbs,
    ImageProbs,
    ImageMeta,
    AuthMeta,
)


class TestFusionEngine(unittest.TestCase):
    """Test cases for TrustScoreCalculator."""

    def setUp(self):
        self.calculator = TrustScoreCalculator()

    def test_spam_detection_gatekeeper(self):
        """Step 1: Spam detected should instantly terminate and severely penalize trust score."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.9, negative=0.0, neutral=0.1),
            image_probs=ImageProbs(defect=0.0, no_defect=1.0),
            image_meta=ImageMeta(is_irrelevant=False),
            auth_meta=AuthMeta(is_spam=True)
        )
        result = self.calculator.calculate(inputs)
        
        self.assertEqual(result.final_score, 5.0)
        self.assertFalse(result.is_conflict)
        self.assertIn("RISK: Fraudulent Review (Spam/Seeding)", result.flags)
        self.assertEqual(result.reason_code, "SPAM_DETECTED")

    def test_missing_image_modality(self):
        """Step 2: Missing image should redistribute weight to text modality (weight_text=0.8)."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=1.0, negative=0.0, neutral=0.0),
            image_probs=None,
            image_meta=None,
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 1.0 * 100 = 100.0
        # final_score = (100.0 * 0.8) + (100.0 * 0.2) = 100.0
        self.assertEqual(result.final_score, 100.0)
        self.assertFalse(result.is_conflict)
        self.assertEqual(result.flags, [])
        self.assertEqual(result.reason_code, "MISSING_IMAGE")

        # Test with neutral sentiment text
        inputs_neutral = FusionInput(
            text_probs=TextProbs(positive=0.0, negative=0.0, neutral=1.0),
            image_probs=None,
            image_meta=None,
            auth_meta=AuthMeta(is_spam=False)
        )
        result_neutral = self.calculator.calculate(inputs_neutral)
        # score_text = 1.0 * 50 = 50.0
        # final_score = (50.0 * 0.8) + (100.0 * 0.2) = 40.0 + 20.0 = 60.0
        self.assertEqual(result_neutral.final_score, 60.0)

    def test_irrelevant_image_clip_intervention(self):
        """Step 2: Irrelevant image should ignore image probs and redistribute weight to text."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=1.0, negative=0.0, neutral=0.0),
            image_probs=ImageProbs(defect=0.9, no_defect=0.1), # Highly defective, but irrelevant!
            image_meta=ImageMeta(is_irrelevant=True),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 100.0
        # final_score = (100.0 * 0.8) + 20 = 100.0
        self.assertEqual(result.final_score, 100.0)
        self.assertFalse(result.is_conflict)
        self.assertIn("WARNING: Irrelevant Product Image", result.flags)
        self.assertEqual(result.reason_code, "IRRELEVANT_IMAGE")

    def test_normal_multimodal_path(self):
        """Step 4: Check score calculation when all signals are normal and relevant."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.8, negative=0.1, neutral=0.1),
            image_probs=ImageProbs(defect=0.1, no_defect=0.9),
            image_meta=ImageMeta(is_irrelevant=False),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 0.8 * 100 + 0.1 * 50 = 85.0
        # image_score = 0.9 * 100 = 90.0
        # final_score = (85 * 0.4) + (90 * 0.4) + (100 * 0.2) = 34.0 + 36.0 + 20.0 = 90.0
        self.assertEqual(result.final_score, 90.0)
        self.assertFalse(result.is_conflict)
        self.assertEqual(result.flags, [])
        self.assertEqual(result.reason_code, "HIGH_TRUST")

    def test_conflict_positive_text_defect_image(self):
        """Step 3: Conflict 1 (Positive text > 0.6 and Defect image > 0.6) -> Penalize 50% score."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.7, negative=0.1, neutral=0.2),
            image_probs=ImageProbs(defect=0.8, no_defect=0.2),
            image_meta=ImageMeta(is_irrelevant=False),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 0.7 * 100 + 0.2 * 50 = 80.0
        # image_score = 0.2 * 100 = 20.0
        # base_score = (80 * 0.4) + (20 * 0.4) + (100 * 0.2) = 32.0 + 8.0 + 20.0 = 60.0
        # penalized score = 60.0 * 0.5 = 30.0
        self.assertEqual(result.final_score, 30.0)
        self.assertTrue(result.is_conflict)
        self.assertIn("CONFLICT: Suspicious praise due to defective product image", result.flags)
        self.assertEqual(result.reason_code, "MULTIMODAL_CONFLICT")

    def test_conflict_negative_text_no_defect_image(self):
        """Step 3: Conflict 2 (Negative text > 0.6 and No-Defect image > 0.8) -> Warning flag, no penalty."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.1, negative=0.8, neutral=0.1),
            image_probs=ImageProbs(defect=0.1, no_defect=0.9),
            image_meta=ImageMeta(is_irrelevant=False),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 0.1 * 100 + 0.1 * 50 = 15.0
        # image_score = 0.9 * 100 = 90.0
        # final_score = (15 * 0.4) + (90 * 0.4) + (100 * 0.2) = 6.0 + 36.0 + 20.0 = 62.0 (no penalty)
        self.assertEqual(result.final_score, 62.0)
        self.assertTrue(result.is_conflict)
        self.assertIn("NOTICE: Customer complaint but no visible product defects", result.flags)
        self.assertEqual(result.reason_code, "MULTIMODAL_CONFLICT")


if __name__ == "__main__":
    unittest.main()
