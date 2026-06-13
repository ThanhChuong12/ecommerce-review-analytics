"""Unit tests for Cross-Modal Fusion Engine v2.0 (MobileNetV3 version).

Run:
    python -m unittest tests/test_fusion_engine.py -v
"""

import unittest
from ai_engine.fusion.fusion_engine import (
    TrustScoreCalculator,
    FusionInput,
    TextProbs,
    ImageProbs,
    AuthMeta,
)


class TestFusionEngine(unittest.TestCase):
    """Test cases for TrustScoreCalculator."""

    def setUp(self):
        self.calculator = TrustScoreCalculator()

    def test_spam_detection_gatekeeper(self):
        """Step 1: Spam detected should instantly terminate and severely penalize trust score.
        With spam_score=0.0 (default), severity defaults to 1.0, penalty_score=5.0 (backward compat).
        """
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.9, negative=0.0, neutral=0.1),
            image_probs=ImageProbs(intact=1.0, damaged=0.0, wrong_item=0.0, irrelevant=0.0),
            auth_meta=AuthMeta(is_spam=True)
        )
        result = self.calculator.calculate(inputs)
        
        self.assertEqual(result.final_score, 5.0)
        self.assertFalse(result.is_conflict)
        self.assertIn("RISK: Fraudulent Review (Spam/Seeding)", result.flags)
        self.assertEqual(result.reason_code, "SPAM_DETECTED")
        self.assertEqual(result.prediction_confidence, 1.0)

    def test_spam_detection_severity_modulated(self):
        """Step 1: Spam detected with mild severity should yield a higher penalty score."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.9, negative=0.0, neutral=0.1),
            image_probs=ImageProbs(intact=1.0, damaged=0.0, wrong_item=0.0, irrelevant=0.0),
            auth_meta=AuthMeta(is_spam=True, spam_score=0.4)
        )
        result = self.calculator.calculate(inputs)
        
        # severity = 0.4
        # penalty_score = 5.0 + (1.0 - 0.4) * (20.0 - 5.0) = 5.0 + 0.6 * 15.0 = 14.0
        self.assertEqual(result.final_score, 14.0)
        self.assertFalse(result.is_conflict)
        self.assertIn("RISK: Fraudulent Review (Spam/Seeding)", result.flags)
        self.assertEqual(result.reason_code, "SPAM_DETECTED")
        self.assertEqual(result.prediction_confidence, 1.0)

    def test_missing_image_modality(self):
        """Step 2: Missing image should redistribute weight to text modality."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=1.0, negative=0.0, neutral=0.0),
            image_probs=None,
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 50.0 + 50.0 * (1.0 - 0.0) = 100.0
        # final_score = (100.0 * 0.8) + (100.0 * 0.2) = 100.0
        self.assertEqual(result.final_score, 100.0)
        self.assertFalse(result.is_conflict)
        self.assertEqual(result.flags, [])
        self.assertEqual(result.reason_code, "MISSING_IMAGE")
        self.assertEqual(result.prediction_confidence, 1.0)

        # Test with neutral sentiment text
        inputs_neutral = FusionInput(
            text_probs=TextProbs(positive=0.0, negative=0.0, neutral=1.0),
            image_probs=None,
            auth_meta=AuthMeta(is_spam=False)
        )
        result_neutral = self.calculator.calculate(inputs_neutral)
        # score_text = 50.0 + 50.0 * (0.0 - 0.0) = 50.0
        # final_score = (50.0 * 0.8) + (100.0 * 0.2) = 40.0 + 20.0 = 60.0
        self.assertEqual(result_neutral.final_score, 60.0)
        self.assertEqual(result_neutral.prediction_confidence, 1.0)

    def test_irrelevant_image_intervention(self):
        """Step 2: Irrelevant image should ignore image probs and redistribute weight to text."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=1.0, negative=0.0, neutral=0.0),
            image_probs=ImageProbs(intact=0.1, damaged=0.0, wrong_item=0.0, irrelevant=0.9), # Irrelevant is highest!
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # score_text = 100.0
        # final_score = (100.0 * 0.8) + 20 = 100.0
        self.assertEqual(result.final_score, 100.0)
        self.assertFalse(result.is_conflict)
        self.assertIn("WARNING: Irrelevant Product Image", result.flags)
        self.assertEqual(result.reason_code, "IRRELEVANT_IMAGE")
        self.assertEqual(result.prediction_confidence, 1.0)

    def test_normal_multimodal_path(self):
        """Step 4: Check score calculation when all signals are normal and relevant."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.8, negative=0.1, neutral=0.1),
            image_probs=ImageProbs(intact=0.9, damaged=0.1, wrong_item=0.0, irrelevant=0.0),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # text_conf = 0.7, img_conf = 0.8
        # raw_text_w = 0.34, raw_img_w = 0.36, raw_auth_w = 0.2 -> sum = 0.9
        # eff_text_w = 0.34/0.9, eff_img_w = 0.36/0.9, eff_auth_w = 0.2/0.9
        # score_text = 85.0
        # defect_prob = 0.1, image_score = max(0, 90.0 - 5.0) = 85.0
        # score_auth = 100.0
        # final_score = (85*0.34 + 85*0.36 + 100*0.2) / 0.9 = 79.5/0.9 = 88.33
        self.assertEqual(result.final_score, 88.33)
        self.assertFalse(result.is_conflict)
        self.assertEqual(result.flags, [])
        self.assertEqual(result.reason_code, "HIGH_TRUST")
        # weighted confidence: (0.7*0.34/0.9 + 0.8*0.36/0.9) / (0.70/0.9) = 0.526/0.70 = 0.7514
        self.assertEqual(result.prediction_confidence, 0.7514)

    def test_conflict_positive_text_defect_image(self):
        """Step 5: Conflict 1 (Positive text > 0.6 and Defect image > 0.6) -> Penalize 50% score."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.7, negative=0.1, neutral=0.2),
            image_probs=ImageProbs(intact=0.2, damaged=0.8, wrong_item=0.0, irrelevant=0.0),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # text_conf = 0.5, img_conf = 0.6
        # raw_text_w = 0.30, raw_img_w = 0.32, raw_auth_w = 0.2 -> sum = 0.82
        # score_text = 80.0
        # defect_prob = 0.8, image_score = max(0, 20.0 - 40.0) = 0.0
        # score_auth = 100.0
        # base_score = (80*0.30 + 0*0.32 + 100*0.2) / 0.82 = 44.0/0.82 = 53.66
        # penalized score = 53.66 * 0.5 = 26.83
        self.assertEqual(result.final_score, 26.83)
        self.assertTrue(result.is_conflict)
        self.assertIn("CONFLICT: Suspicious praise due to defective product image", result.flags)
        self.assertEqual(result.reason_code, "MULTIMODAL_CONFLICT")
        # weighted confidence: (0.5*0.30/0.82 + 0.6*0.32/0.82) / (0.62/0.82) = 0.5516
        self.assertEqual(result.prediction_confidence, 0.5516)

    def test_conflict_negative_text_no_defect_image(self):
        """Step 5: Conflict 2 (Negative text > 0.6 and No-Defect image > 0.8) -> 0.8x lighter penalty."""
        inputs = FusionInput(
            text_probs=TextProbs(positive=0.1, negative=0.8, neutral=0.1),
            image_probs=ImageProbs(intact=0.9, damaged=0.1, wrong_item=0.0, irrelevant=0.0),
            auth_meta=AuthMeta(is_spam=False)
        )
        result = self.calculator.calculate(inputs)
        
        # text_conf = 0.7, img_conf = 0.8
        # raw_text_w = 0.34, raw_img_w = 0.36, raw_auth_w = 0.2 -> sum = 0.9
        # score_text = 15.0
        # defect_prob = 0.1, image_score = max(0, 90.0 - 5.0) = 85.0
        # score_auth = 100.0
        # base_score = (15*0.34 + 85*0.36 + 100*0.2) / 0.9 = 55.7/0.9 = 61.89
        # penalized = 61.89 * 0.8 = 49.51
        self.assertEqual(result.final_score, 49.51)
        self.assertTrue(result.is_conflict)
        self.assertIn("NOTICE: Customer complaint but no visible product defects", result.flags)
        self.assertEqual(result.reason_code, "MULTIMODAL_CONFLICT")
        self.assertEqual(result.prediction_confidence, 0.7514)


if __name__ == "__main__":
    unittest.main()
