"""Zero-Shot Content-Based Reranker.

Reranks candidate recommendation products using PhoBERT semantic similarity,
preliminary trust score heuristic, and price deviation penalty without explicit fine-tuning.

Scoring formula:
    Score(p_i) = alpha * CosineSim(E(p_0), E(p_i)) + beta * Trust(p_i) - gamma * PriceDev(p_i, p_0)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALPHA: float = 0.50
BETA: float = 0.35
GAMMA: float = 0.15
SCORE_THRESHOLD: float = 0.30


def _parse_price(price_raw: Any) -> float:
    """Parse a raw price representation into a floating-point numeric value."""
    if not price_raw:
        return 0.0
    price_str = str(price_raw).lower().strip()
    if "tr" in price_str or "triệu" in price_str:
        digits = re.findall(r"[\d,\.]+", price_str)
        if digits:
            return float(digits[0].replace(",", ".")) * 1_000_000
    clean = re.sub(r"[^\d]", "", price_str)
    return float(clean) if clean else 0.0


def _parse_sold(sold_raw: Any) -> int:
    """Parse raw sold strings (e.g., 'Đã bán 1.2k') into an integer count."""
    if not sold_raw:
        return 0
    sold_str = str(sold_raw).lower().strip()
    try:
        if "k" in sold_str:
            digits = re.findall(r"[\d\.]+", sold_str)
            return int(float(digits[0]) * 1000) if digits else 0
        digits = re.findall(r"\d+", sold_str.replace(".", "").replace(",", ""))
        return int(digits[0]) if digits else 0
    except Exception:
        return 0


def _compute_trust(rating: Any, sold: int) -> float:
    """Calculate a normalized preliminary trust score in the range [0, 1]."""
    try:
        rating_val = float(rating) if rating else 3.0
    except (ValueError, TypeError):
        rating_val = 3.0

    rating_norm = max(0.0, min(1.0, (rating_val - 1.0) / 4.0))
    sold_norm = 1.0 - math.exp(-sold / 500.0) if sold > 0 else 0.0
    sold_norm = max(0.0, min(1.0, sold_norm))

    return rating_norm * 0.6 + sold_norm * 0.4


def _cosine_sim(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors."""
    sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1)
    return float(sim.item())


def _embed_text(
    text: str,
    tokenizer,
    backbone: torch.nn.Module,
    device: torch.device,
    max_length: int = 64,
) -> torch.Tensor:
    """Extract sentence embedding vector using mean pooling across non-padding tokens."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)

    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    else:
        hidden = outputs[0]

    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (hidden * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return (sum_hidden / count).squeeze(0)


class ZeroShotReranker:
    """Reranks product recommendation candidates using pre-trained PhoBERT embeddings."""

    def __init__(
        self,
        tokenizer,
        backbone: torch.nn.Module,
        device: torch.device,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        threshold: float = SCORE_THRESHOLD,
    ):
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

    def _get_dominant_badge(
        self,
        cosine_contrib: float,
        trust_contrib: float,
        price_penalty: float,
        rating_val: float,
        sold_val: int,
    ) -> str:
        """Assign user-facing reasoning badge based on dominant score component."""
        if cosine_contrib >= trust_contrib and cosine_contrib >= 0.25:
            if rating_val >= 4.7 and sold_val >= 500:
                return "Tương tự & Bán chạy"
            return "Sản phẩm tương tự"
        if trust_contrib >= cosine_contrib and trust_contrib >= 0.25:
            if sold_val >= 1000:
                return "Mua nhiều nhất"
            if rating_val >= 4.8:
                return "Đánh giá cực tốt"
            return "Đánh giá cao"
        if price_penalty < 0.05:
            return "Giá hợp lý"

        if rating_val >= 4.5 and sold_val >= 200:
            return "Bán chạy & Uy tín"
        if rating_val >= 4.0:
            return "Khách mua hài lòng"
        return "Cùng phân khúc"

    def rerank(
        self,
        origin_name: str,
        origin_price: Any,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank candidates using composite similarity, trust, and price penalty score."""
        if not candidates:
            return []

        logger.info("Starting zero-shot reranking for %d candidates", len(candidates))

        try:
            origin_vec = _embed_text(origin_name, self.tokenizer, self.backbone, self.device)
        except Exception as e:
            logger.warning("Failed to embed origin product text: %s", e)
            return candidates

        origin_price_float = _parse_price(origin_price)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for cand in candidates:
            name_cand = cand.get("name", "")
            if not name_cand:
                continue

            try:
                cand_vec = _embed_text(name_cand, self.tokenizer, self.backbone, self.device)
                cosine = max(0.0, _cosine_sim(origin_vec, cand_vec))
            except Exception as e:
                logger.debug("Failed to embed candidate '%s': %s", name_cand[:40], e)
                cosine = 0.3

            sold_int = _parse_sold(cand.get("sold", 0))
            trust_norm = _compute_trust(cand.get("rating", 0), sold_int)

            cand_price_float = _parse_price(cand.get("price", 0))
            if origin_price_float > 0 and cand_price_float > 0:
                price_deviation = abs(cand_price_float - origin_price_float) / origin_price_float
                price_penalty = min(1.0, price_deviation)
            else:
                price_penalty = 0.0

            alpha_contrib = self.alpha * cosine
            beta_contrib = self.beta * trust_norm
            gamma_penalty = self.gamma * price_penalty

            final_score = max(0.0, min(1.0, alpha_contrib + beta_contrib - gamma_penalty))

            try:
                rating_raw = float(cand.get("rating") or 0)
            except (ValueError, TypeError):
                rating_raw = 0.0
            rating_val_display = rating_raw if rating_raw > 0 else 3.0

            mini_trust_display = round(
                min(100.0, rating_val_display * 20.0 + min(20.0, trust_norm * 20.0)),
                1,
            )

            reason_badge = self._get_dominant_badge(
                cosine_contrib=alpha_contrib,
                trust_contrib=beta_contrib,
                price_penalty=gamma_penalty,
                rating_val=rating_val_display,
                sold_val=sold_int,
            )

            if final_score < self.threshold:
                logger.debug(
                    "Filtered out candidate '%s' (Score=%.3f < threshold=%.3f)",
                    name_cand[:40], final_score, self.threshold,
                )
                continue

            enriched_cand = {
                **cand,
                "rerank_score": round(final_score, 4),
                "cosine_score": round(cosine, 4),
                "trust_score_norm": round(trust_norm, 4),
                "trustScore": mini_trust_display,
                "reason": reason_badge,
            }
            scored.append((final_score, enriched_cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [item for _, item in scored]
        logger.info(
            "Reranking completed: %d/%d candidates passed threshold (%.2f)",
            len(result), len(candidates), self.threshold,
        )
        return result


def rerank_candidates(
    origin_name: str,
    origin_price: Any,
    candidates: List[Dict[str, Any]],
    tokenizer,
    backbone: torch.nn.Module,
    device: torch.device,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    threshold: float = SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Helper function for standalone zero-shot reranking execution."""
    reranker = ZeroShotReranker(
        tokenizer=tokenizer,
        backbone=backbone,
        device=device,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold=threshold,
    )
    return reranker.rerank(origin_name, origin_price, candidates)
