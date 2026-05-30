"""
Multi-provider LLM client with cascading fallback routing and quota management.

Contains two specialized clients:
1. LLMFallbackClient: Predicts 'tích cực', 'tiêu cực', 'trung lập' for
   zero-shot sentiment fallback.  Enforces a **global call budget** via the
   :class:`LLMBudget` singleton to prevent quota exhaustion during large batch
   labeling runs.
2. LLMRecommendationClient: Uses Chain-of-Thought to generate a recommendation
   action based on Fusion Engine Trust Scores.
"""

from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global LLM call-budget guard
# ---------------------------------------------------------------------------

class LLMBudget:
    """Thread-safe singleton that caps the total number of LLM API calls.

    Usage::

        # At notebook / script startup — set the cap for the whole run
        LLMBudget.configure(max_calls=500)

        # Query status at any time
        remaining = LLMBudget.remaining()
        print(LLMBudget.summary())

    The budget is shared across **all** ``LLMFallbackClient`` instances so that
    parallel ``DataFrame.apply`` calls cannot collectively exceed the limit.

    Attributes:
        _max_calls:  Maximum permitted calls in this session (default: 300).
        _used_calls: Running count of calls consumed so far.
        _lock:       Re-entrant lock for thread safety.
        _exhausted_label: Sentinel returned when the budget is exhausted
                          instead of making a real API call.
    """

    _max_calls: int = 300
    _used_calls: int = 0
    _lock: Lock = Lock()
    _exhausted_label: str = "trung lập"  # safe default when quota is gone

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def configure(cls, max_calls: int, exhausted_label: str = "trung lập") -> None:
        """Set the budget for the current run.

        Call this **once** before the labeling loop starts (e.g. at the top of
        the notebook cell that initialises ``NextGenReviewAnalyzer``).

        Args:
            max_calls: Maximum number of LLM API calls allowed in this session.
            exhausted_label: Sentiment label returned when the budget is
                             exhausted.  Defaults to ``"trung lập"``.
        """
        with cls._lock:
            cls._max_calls = max_calls
            cls._used_calls = 0
            cls._exhausted_label = exhausted_label
        logger.info(
            "[LLMBudget] Budget configured: max_calls=%d, exhausted_label='%s'",
            max_calls,
            exhausted_label,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the used-call counter to zero (useful between notebook runs)."""
        with cls._lock:
            cls._used_calls = 0
        logger.info("[LLMBudget] Counter reset to 0.")

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    @classmethod
    def remaining(cls) -> int:
        """Return the number of API calls still available."""
        with cls._lock:
            return max(0, cls._max_calls - cls._used_calls)

    @classmethod
    def is_exhausted(cls) -> bool:
        """Return ``True`` when the budget has been fully consumed."""
        return cls.remaining() == 0

    @classmethod
    def consume(cls) -> bool:
        """Attempt to consume one call from the budget.

        Returns:
            ``True`` if the call is permitted (budget decremented).
            ``False`` if the budget is already exhausted.
        """
        with cls._lock:
            if cls._used_calls >= cls._max_calls:
                return False
            cls._used_calls += 1
            return True

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable budget status string."""
        with cls._lock:
            pct = (cls._used_calls / cls._max_calls * 100) if cls._max_calls else 0
            return (
                f"[LLMBudget] {cls._used_calls}/{cls._max_calls} calls used "
                f"({pct:.1f}%) — {max(0, cls._max_calls - cls._used_calls)} remaining"
            )


# ---------------------------------------------------------------------------
# Base provider client
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """Base class providing cascading fallback routing across Gemini, Groq, and OpenAI."""

    def __init__(
        self,
        provider_chain: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> None:
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

        generation_config: Dict[str, Any] = {"temperature": self.temperature}
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


# ---------------------------------------------------------------------------
# LLMFallbackClient — sentiment labeling with budget guard
# ---------------------------------------------------------------------------

class LLMFallbackClient(BaseLLMClient):
    """Client for fallback sentiment analysis with a global call-budget guard.

    Every call to :meth:`analyze` first checks :class:`LLMBudget`.  If the
    budget is exhausted the method returns immediately without making any API
    request, using the ``exhausted_label`` configured on the budget.

    A short exponential back-off is applied between provider retries to
    reduce the risk of hitting per-minute rate limits.

    Args:
        provider_chain: Ordered list of providers to try (default: gemini → groq
                        → openai).
        timeout: Per-request timeout in seconds.
        retry_delay: Base delay (seconds) before retrying the next provider.
                     Each successive retry doubles the delay (capped at 8 s).
    """

    _allowed_labels = {"tích cực", "tiêu cực", "trung lập"}

    def __init__(
        self,
        provider_chain: Optional[List[str]] = None,
        timeout: float = 15.0,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(provider_chain, timeout)
        self.temperature = 0.0
        self.max_tokens = 20
        self.retry_delay = retry_delay
        self.system_prompt = (
            "Bạn là một chuyên gia phân tích cảm xúc bình luận e-commerce tiếng Việt. "
            "Nhiệm vụ của bạn là trả về đúng một JSON object duy nhất theo mẫu: "
            '{"sentiment": "tích cực" | "tiêu cực" | "trung lập"}. '
            "Không giải thích, không thêm bất kỳ ký tự nào khác."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, payload: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """Classify the sentiment of a review with budget enforcement.

        Accepts either a plain string or a dict with ``"text"`` (and optionally
        ``"rating"``) so callers can pass rating context without a separate
        argument.

        Args:
            payload: Review text (str) **or** a dict containing at least the
                     key ``"text"`` and optionally ``"rating"``.

        Returns:
            ``{"sentiment": <label>}`` where label is one of ``"tích cực"``,
            ``"tiêu cực"``, or ``"trung lập"``.
        """
        # --- Normalise input -------------------------------------------------
        if isinstance(payload, dict):
            text: str = str(payload.get("text", "")).strip()
            rating: Optional[int] = payload.get("rating")
        else:
            text = str(payload).strip()
            rating = None

        if not text:
            return {"sentiment": "trung lập"}

        # --- Budget gate -----------------------------------------------------
        if LLMBudget.is_exhausted():
            logger.warning(
                "[LLMBudget] Budget exhausted — skipping LLM call. %s",
                LLMBudget.summary(),
            )
            return {"sentiment": LLMBudget._exhausted_label}

        # --- Build prompt (include rating hint when available) ---------------
        rating_hint = f"\nSố sao đánh giá: {rating}/5." if rating is not None else ""
        user_prompt = (
            "Hãy phân loại cảm xúc của bình luận sau và chỉ trả JSON đúng định dạng yêu cầu."
            f"{rating_hint}\n"
            f"Bình luận: {text}"
        )

        # --- Cascading provider loop with back-off ---------------------------
        raw_response: Optional[str] = None
        delay = self.retry_delay

        for provider in self.provider_chain:
            try:
                raw_response = self._call_provider(provider, user_prompt)
                if raw_response:
                    break
            except Exception as exc:
                logger.warning(
                    "[LLMFallback] Provider '%s' failed: %s. Waiting %.1fs before next...",
                    provider,
                    exc,
                    delay,
                )
                time.sleep(min(delay, 8.0))
                delay *= 2  # exponential back-off, capped at 8 s

        # --- Consume one budget unit after a real call attempt ---------------
        LLMBudget.consume()

        if not raw_response:
            logger.error("[LLMFallback] All providers failed. %s", LLMBudget.summary())
            return {"sentiment": "trung lập"}

        result = self._parse_response(raw_response)

        # Log every Nth call so the notebook operator can monitor progress
        used = LLMBudget._used_calls
        if used % 50 == 0 or LLMBudget.remaining() <= 10:
            logger.info("[LLMBudget] %s", LLMBudget.summary())

        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, str]:
        """Extract the sentiment label from the raw LLM response string."""
        if not raw:
            return {"sentiment": "trung lập"}

        raw = raw.strip()
        payload = None

        # Try direct JSON parse
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: locate the first {...} block
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

        # Last resort: keyword scan
        lowered = raw.lower()
        for label in self._allowed_labels:
            if label in lowered:
                return {"sentiment": label}

        return {"sentiment": "trung lập"}


# ---------------------------------------------------------------------------
# Backward-compatible helper
# ---------------------------------------------------------------------------

def ask_llm(text: str) -> str:
    """Backward-compatible helper for basic sentiment fallback."""
    return LLMFallbackClient().analyze(text).get("sentiment", "trung lập")


# ---------------------------------------------------------------------------
# LLMRecommendationClient — CoT multimodal analysis (unchanged)
# ---------------------------------------------------------------------------

class LLMRecommendationClient(BaseLLMClient):
    """Client for CoT Multimodal Analysis.

    Takes the Trust Score and Conflict Flags from Fusion Engine and outputs
    a final platform action recommendation.
    """

    def __init__(
        self,
        provider_chain: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> None:
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

    def analyze_review(
        self, text: str, fusion_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Analyze the review text and fusion engine output using LLM CoT."""
        score = (
            fusion_result.get("final_score", 0)
            if isinstance(fusion_result, dict)
            else getattr(fusion_result, "final_score", 0)
        )
        conflict = (
            fusion_result.get("is_conflict", False)
            if isinstance(fusion_result, dict)
            else getattr(fusion_result, "is_conflict", False)
        )
        reason = (
            fusion_result.get("reason_code", "UNKNOWN")
            if isinstance(fusion_result, dict)
            else getattr(fusion_result, "reason_code", "UNKNOWN")
        )

        user_prompt = (
            f'Hãy phân tích bình luận sau:\nNội dung bình luận: "{text}"\n'
            f"Kết quả hệ thống Fusion:\n"
            f"- Trust Score: {score}/100\n"
            f"- Xung đột Đa phương thức: {conflict}\n"
            f"- Mã lý do: {reason}"
        )

        raw_response: Optional[str] = None
        for provider in self.provider_chain:
            try:
                raw_response = self._call_provider(provider, user_prompt)
                if raw_response:
                    break
            except Exception as exc:
                logger.warning(
                    "[LLMRecommendation] Provider '%s' failed: %s. Routing to next...",
                    provider,
                    exc,
                )

        if not raw_response:
            logger.error("[LLMRecommendation] All providers failed.")
            return {"recommendation_action": "CẢNH BÁO", "error": "All providers failed"}

        return self._parse_json_response(raw_response)

    def _parse_json_response(self, raw: str) -> Dict[str, str]:
        """Safely parse the JSON output from the LLM."""
        raw = raw.strip()
        for marker in ("```json", "```"):
            if raw.startswith(marker):
                raw = raw[len(marker):]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            payload = json.loads(raw)
            action = payload.get("recommendation_action", "CẢNH BÁO")
            if action not in {"DUYỆT", "CẢNH BÁO", "XÓA"}:
                action = "CẢNH BÁO"
            payload["recommendation_action"] = action
            return payload
        except json.JSONDecodeError:
            logger.error("[LLMRecommendation] Failed to parse JSON: %s", raw)
            return {"recommendation_action": "CẢNH BÁO", "error": "Invalid JSON response"}