"""
Multi-provider LLM client for sentiment fallback analysis.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFallbackClient:
    """
    Strategy-style client that switches between Gemini, OpenAI, and Grok (xAI).

    The provider is chosen via the LLM_PROVIDER environment variable. The
    response is forced into a strict JSON payload:
    {"sentiment": "tích cực" | "tiêu cực" | "trung lập"}
    """

    _allowed_labels = {"tích cực", "tiêu cực", "trung lập"}

    def __init__(self, provider: str | None = None, timeout: float = 15.0) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.timeout = timeout
        self.system_prompt = (
            "Bạn là một chuyên gia phân tích cảm xúc bình luận e-commerce tiếng Việt. "
            "Nhiệm vụ của bạn là trả về đúng một JSON object duy nhất theo mẫu: "
            "{\"sentiment\": \"tích cực\" | \"tiêu cực\" | \"trung lập\"}. "
            "Không giải thích, không thêm bất kỳ ký tự nào khác."
        )

    def analyze(self, text: str) -> Dict[str, str]:
        """
        Analyze sentiment via the selected LLM provider.

        Returns a safe default if the provider fails, times out, or produces
        invalid JSON.
        """
        if not text:
            return {"sentiment": "trung lập"}

        try:
            if self.provider == "gemini":
                raw = self._call_gemini(text)
            elif self.provider == "openai":
                raw = self._call_openai(text)
            elif self.provider in {"grok", "xai", "x-ai"}:
                raw = self._call_grok(text)
            else:
                logger.error("Unsupported LLM provider: %s", self.provider)
                return {"sentiment": "trung lập"}
        except Exception as exc:
            logger.error("LLM fallback failed: %s", exc)
            return {"sentiment": "trung lập"}

        return self._parse_response(raw)

    def _user_prompt(self, text: str) -> str:
        return (
            "Hãy phân loại cảm xúc của bình luận sau và chỉ trả JSON đúng định dạng yêu cầu.\n"
            f"Bình luận: {text}"
        )

    def _call_gemini(self, text: str) -> str:
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-generativeai is not installed") from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.1,
            "top_k": 1,
            "max_output_tokens": 20,
        }
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=self.system_prompt,
        )

        response = model.generate_content(self._user_prompt(text))
        return getattr(response, "text", "") or ""

    def _call_openai(self, text: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._user_prompt(text)},
            ],
            temperature=0.0,
            max_tokens=30,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    def _call_grok(self, text: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise RuntimeError("GROK_API_KEY is not configured")

        model = os.getenv("GROK_MODEL", "grok-2-latest")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._user_prompt(text)},
            ],
            temperature=0.0,
            max_tokens=30,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

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
                    payload = None

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
    """
    Backward-compatible helper returning a sentiment string.
    """
    return LLMFallbackClient().analyze(text).get("sentiment", "trung lập")