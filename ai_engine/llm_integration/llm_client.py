"""
Multi-provider LLM client with cascading fallback routing.

Contains two specialized clients:
1. LLMFallbackClient: (Legacy) Predicts 'tích cực', 'tiêu cực', 'trung lập' for zero-shot sentiment fallback.
2. LLMRecommendationClient: (New) Uses Chain-of-Thought to generate a recommendation action based on Fusion Engine Trust Scores.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMClient:
    """
    Base class providing cascading fallback routing across Gemini, Groq, and OpenAI.
    """
    def __init__(self, provider_chain: Optional[List[str]] = None, timeout: float = 15.0) -> None:
        self.provider_chain = provider_chain or ["gemini", "groq", "openai"]
        self.timeout = timeout
        self.system_prompt = ""
        self.temperature = 0.0
        self.max_tokens = 30
        self.response_format: Optional[str] = None

    def _call_provider(self, provider: str, user_prompt: str) -> str:
        """Route the call to the specified provider."""
        if provider == "gemini":
            return self._call_gemini(user_prompt)
        elif provider == "groq":
            return self._call_groq(user_prompt)
        elif provider == "openai":
            return self._call_openai(user_prompt)
        raise ValueError(f"Unsupported provider: {provider}")

    def _call_gemini(self, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is not installed") from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        genai.configure(api_key=api_key)
        
        generation_config = {"temperature": self.temperature}
        if self.response_format == "json_object":
            generation_config["response_mime_type"] = "application/json"
        else:
            generation_config["max_output_tokens"] = self.max_tokens

        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=self.system_prompt,
        )

        response = model.generate_content(user_prompt)
        return getattr(response, "text", "") or ""

    def _call_groq(self, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured in .env")

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        kwargs: Dict[str, Any] = {
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        
        if self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _call_openai(self, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured in .env")

        client = OpenAI(api_key=api_key)
        kwargs: Dict[str, Any] = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        
        if self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class LLMFallbackClient(BaseLLMClient):
    """
    Client for fallback sentiment analysis (Used by sentiment_analysis.py).
    Predicts basic labels: tích cực, tiêu cực, trung lập.
    """
    _allowed_labels = {"tích cực", "tiêu cực", "trung lập"}

    def __init__(self, provider_chain: Optional[List[str]] = None, timeout: float = 15.0) -> None:
        super().__init__(provider_chain, timeout)
        self.temperature = 0.0
        self.max_tokens = 20
        self.system_prompt = (
            "Bạn là một chuyên gia phân tích cảm xúc bình luận e-commerce tiếng Việt. "
            "Nhiệm vụ của bạn là trả về đúng một JSON object duy nhất theo mẫu: "
            '{"sentiment": "tích cực" | "tiêu cực" | "trung lập"}. '
            "Không giải thích, không thêm bất kỳ ký tự nào khác."
        )

    def analyze(self, text: str) -> Dict[str, str]:
        """Execute the cascading fallback loop to classify text sentiment."""
        if not text:
            return {"sentiment": "trung lập"}

        user_prompt = (
            "Hãy phân loại cảm xúc của bình luận sau và chỉ trả JSON đúng định dạng yêu cầu.\n"
            f"Bình luận: {text}"
        )
        
        raw_response = None
        for provider in self.provider_chain:
            try:
                raw_response = self._call_provider(provider, user_prompt)
                if raw_response:
                    break
            except Exception as exc:
                logger.warning("Provider '%s' failed: %s. Routing to next...", provider, exc)

        if not raw_response:
            logger.error("All LLM providers in the fallback chain failed.")
            return {"sentiment": "trung lập"}

        return self._parse_response(raw_response)

    def _parse_response(self, raw: str) -> Dict[str, str]:
        if not raw:
            return {"sentiment": "trung lập"}

        payload = None
        raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    payload = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if isinstance(payload, dict):
            sentiment = str(payload.get("sentiment", "")).strip().lower()
            if sentiment in self._allowed_labels:
                return {"sentiment": sentiment}

        lowered = raw.lower()
        for label in self._allowed_labels:
            if label in lowered:
                return {"sentiment": label}

        return {"sentiment": "trung lập"}


def ask_llm(text: str) -> str:
    """Backward-compatible helper for basic sentiment fallback."""
    return LLMFallbackClient().analyze(text).get("sentiment", "trung lập")


class LLMRecommendationClient(BaseLLMClient):
    """
    Client for CoT Multimodal Analysis.
    Takes the Trust Score and Conflict Flags from Fusion Engine and outputs 
    a final platform action recommendation.
    """

    def __init__(self, provider_chain: Optional[List[str]] = None, timeout: float = 15.0) -> None:
        super().__init__(provider_chain, timeout)
        self.temperature = 0.2
        self.response_format = "json_object"
        
        self.cot_system_prompt = """Bạn là hệ thống AI phân tích rủi ro thương mại điện tử chuyên sâu.
Nhiệm vụ: Dựa trên điểm Trust Score và phân tích đa phương thức (Văn bản + Hình ảnh + Spam), hãy đưa ra gợi ý hành động (recommendation_action) cho bình luận này.

BẮT BUỘC ÁP DỤNG SUY LUẬN TỪNG BƯỚC (Chain-of-Thought) TRƯỚC KHI KẾT LUẬN:
- step_1 (Trích xuất): Liệt kê các tín hiệu nhận được từ đầu vào (Điểm Trust Score, mâu thuẫn hình/chữ, đánh dấu spam, nội dung bình luận).
- step_2 (Phân tích): Đánh giá độ tin cậy của bình luận. Nếu có mâu thuẫn (VD: text khen nhưng ảnh hỏng) hoặc điểm thấp, đây có thể là bình luận giả mạo/buff đơn.
- step_3 (Kết luận): Căn cứ vào phân tích, đưa ra quyết định duyệt hoặc từ chối.

Bạn PHẢI trả về ĐÚNG MỘT JSON object duy nhất theo định dạng sau, không kèm bất kỳ markdown hay chữ giải thích nào bên ngoài:
{
    "step_1": "<trích xuất dữ liệu>",
    "step_2": "<phân tích độ tin cậy>",
    "recommendation_action": "DUYỆT" | "CẢNH BÁO" | "XÓA"
}"""
        self.system_prompt = self.cot_system_prompt

    def analyze_review(self, text: str, fusion_result: Dict[str, Any]) -> Dict[str, str]:
        """Analyze the review text and fusion engine output using LLM CoT."""
        # Handle dict or dataclass
        score = fusion_result.get('final_score', 0) if isinstance(fusion_result, dict) else getattr(fusion_result, 'final_score', 0)
        conflict = fusion_result.get('is_conflict', False) if isinstance(fusion_result, dict) else getattr(fusion_result, 'is_conflict', False)
        reason = fusion_result.get('reason_code', 'UNKNOWN') if isinstance(fusion_result, dict) else getattr(fusion_result, 'reason_code', 'UNKNOWN')
        
        user_prompt = f"""Hãy phân tích bình luận sau:
Nội dung bình luận: "{text}"
Kết quả hệ thống Fusion:
- Trust Score: {score}/100
- Xung đột Đa phương thức: {conflict}
- Mã lý do: {reason}"""

        raw_response = None
        for provider in self.provider_chain:
            try:
                raw_response = self._call_provider(provider, user_prompt)
                if raw_response:
                    break
            except Exception as exc:
                logger.warning("Provider '%s' failed: %s. Routing to next...", provider, exc)

        if not raw_response:
            logger.error("All LLM providers failed.")
            return {"recommendation_action": "CẢNH BÁO", "error": "All providers failed"}

        return self._parse_json_response(raw_response)

    def _parse_json_response(self, raw: str) -> Dict[str, str]:
        """Safely parses the JSON output from the LLM."""
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            payload = json.loads(raw)
            action = payload.get("recommendation_action", "CẢNH BÁO")
            if action not in ["DUYỆT", "CẢNH BÁO", "XÓA"]:
                action = "CẢNH BÁO"
            payload["recommendation_action"] = action
            return payload
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM: %s", raw)
            return {"recommendation_action": "CẢNH BÁO", "error": "Invalid JSON response"}