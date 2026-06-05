"""
Unit tests for LLMFallbackClient response parsing.
"""

import unittest

from ai_engine.llm_integration.llm_client import LLMFallbackClient


class TestLLMFallbackClientParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMFallbackClient(provider="unknown") 

    def test_parse_valid_json(self) -> None:
        raw = '{"sentiment": "tích cực"}'
        result = self.client._parse_response(raw)
        self.assertEqual(result, {"sentiment": "tích cực"})

    def test_parse_embedded_json(self) -> None:
        raw = 'prefix {"sentiment": "tiêu cực"} suffix'
        result = self.client._parse_response(raw)
        self.assertEqual(result, {"sentiment": "tiêu cực"})

    def test_parse_label_in_text(self) -> None:
        raw = "Kết quả phân tích là: tích cực."
        result = self.client._parse_response(raw)
        self.assertEqual(result, {"sentiment": "trung lập"})

    def test_parse_invalid_default(self) -> None:
        raw = "unknown"
        result = self.client._parse_response(raw)
        self.assertEqual(result, {"sentiment": "trung lập"})

    def test_analyze_unknown_provider_default(self) -> None:
        result = self.client.analyze("San pham binh thuong")
        self.assertEqual(result, {"sentiment": "trung lập"})


if __name__ == "__main__":
    unittest.main()
