"""
Multi-provider LLM client with cascading fallback routing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFallbackClient:
    """
    Client that implements a cascading fallback strategy.
    
    It will attempt to call LLM providers in the exact order specified in 
    the provider_chain. If a provider fails, it logs the warning and seamlessly 
    routes the request to the next available provider.
    """

    _allowed_labels = {"tích cực", "tiêu cực", "trung lập"}

    def __init__(self, provider_chain: List[str] | None = None, timeout: float = 15.0) -> None:
        # Define the routing priority: Gemini first, Groq second
        self.provider_chain = provider_chain or ["gemini", "groq"]
        self.timeout = timeout
        self.system_prompt = (
            "Bạn là một chuyên gia phân tích cảm xúc bình luận e-commerce tiếng Việt. "
            "Nhiệm vụ của bạn là trả về đúng một JSON object duy nhất theo mẫu: "
            "{\"sentiment\": \"tích cực\" | \"tiêu cực\" | \"trung lập\"}. "
            "Không giải thích, không thêm bất kỳ ký tự nào khác."
        )

    def analyze(self, text: str) -> Dict[str, str]:
        """
        Execute the cascading fallback loop across configured providers.
        """
        if not text:
            return {"sentiment": "trung lập"}

        raw_response = None

        # Iterate through the providers based on priority
        for provider in self.provider_chain:
            try:
                if provider == "gemini":
                    raw_response = self._call_gemini(text)
                elif provider == "groq":
                    raw_response = self._call_groq(text)
                elif provider == "openai":
                    raw_response = self._call_openai(text)
                else:
                    logger.warning("Unsupported provider in chain: %s", provider)
                    continue

                # If successful (no exception raised and response exists), break the loop
                if raw_response:
                    break
                    
            except Exception as exc:
                # Log the failure and let the loop naturally proceed to the next provider
                logger.warning("Provider '%s' failed: %s. Routing to the next...", provider, exc)
                continue

        # If the loop finishes but raw_response is still empty, all providers failed
        if not raw_response:
            logger.error("All LLM providers in the fallback chain failed.")
            return {"sentiment": "trung lập"}

        return self._parse_response(raw_response)

    def _user_prompt(self, text: str) -> str:
        return (
            "Hãy phân loại cảm xúc của bình luận sau và chỉ trả JSON đúng định dạng yêu cầu.\n"
            f"Bình luận: {text}"
        )

    def _call_gemini(self, text: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-generativeai is not installed") from exc

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.1,
            "top_k": 1,
            "max_output_tokens": 20,
        }
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=self.system_prompt,
        )

        response = model.generate_content(self._user_prompt(text))
        return getattr(response, "text", "") or ""

    def _call_groq(self, text: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured in .env")

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
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

    def _call_openai(self, text: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured in .env")

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
    Backward-compatible helper.
    """
    return LLMFallbackClient().analyze(text).get("sentiment", "trung lập")

# Cập nhật Chain-of-Thought (CoT) Prompt cho Gemini