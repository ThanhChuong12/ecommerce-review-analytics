"""
Zero-Shot Content-Based Reranker
=================================
Sắp xếp lại danh sách sản phẩm đề xuất dựa trên tương đồng ngữ nghĩa,
điểm uy tín (Trust Score), và hình phạt lệch giá — hoàn toàn không cần training.

Công thức học thuật:
    Score(pᵢ) = α·CosineSim(E(p₀), E(pᵢ)) + β·Trust(pᵢ) - γ·|Price(pᵢ) - Price(p₀)| / Price(p₀)

Trong đó:
    E(·)         : Hàm trích xuất vector từ PhoBERT (Mean Pooling CLS token)
    Trust(pᵢ)    : Điểm uy tín chuẩn hóa [0,1] từ Rating và Sold count
    CosineSim    : Độ tương đồng cosine giữa vector sản phẩm gốc và ứng viên
    α, β, γ      : Trọng số điều chỉnh (mặc định: α=0.5, β=0.35, γ=0.15)

Threshold mặc định: Score < 0.3 → loại khỏi danh sách (lọc nhiễu).

Badge reasoning:
    - CosineSim đóng góp lớn nhất → "Sản phẩm tương tự"
    - Trust đóng góp lớn nhất     → "Đánh giá cao"
    - Không có penalty giá lớn   → "Giá hợp lý"
"""

from __future__ import annotations

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
ALPHA: float = 0.50   # Cosine similarity weight
BETA:  float = 0.35   # Trust score weight
GAMMA: float = 0.15   # Price deviation penalty weight
SCORE_THRESHOLD: float = 0.30  # Remove candidates with score below threshold

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_price(price_raw: Any) -> float:
    """Parse raw price string into float."""
    if not price_raw:
        return 0.0
    price_str = str(price_raw).lower().strip()
    # Handle million indicator (tr)
    if "tr" in price_str or "triệu" in price_str:
        digits = re.findall(r"[\d,\.]+", price_str)
        if digits:
            return float(digits[0].replace(",", ".")) * 1_000_000
    # Strip non-digit characters
    clean = re.sub(r"[^\d]", "", price_str)
    return float(clean) if clean else 0.0


def _parse_sold(sold_raw: Any) -> int:
    """Parse sold counts like 'Đã bán 1.2k' into int."""
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
    """
    Calculate normalized trust score in [0, 1].
    
    Formula:
        trust_raw = rating_norm * 0.6 + sold_norm * 0.4
    
    rating_norm : min-max scale [1, 5] -> [0, 1]
    sold_norm   : sigmoid saturation at 2000 units
    """
    try:
        rating_val = float(rating) if rating else 3.0
    except (ValueError, TypeError):
        rating_val = 3.0
    
    # Scale rating from [1, 5] to [0, 1]
    rating_norm = max(0.0, min(1.0, (rating_val - 1.0) / 4.0))
    
    # Scale sold counts: sigmoid saturation at 2000 -> 1.0
    sold_norm = 1.0 - math.exp(-sold / 500.0) if sold > 0 else 0.0
    sold_norm = max(0.0, min(1.0, sold_norm))
    
    return rating_norm * 0.6 + sold_norm * 0.4


def _cosine_sim(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors."""
    sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1)
    return float(sim.item())


# ─── Core Embedding Function ──────────────────────────────────────────────────

def _embed_text(
    text: str,
    tokenizer,
    backbone: torch.nn.Module,
    device: torch.device,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Extract semantic vector [hidden_dim] using PhoBERT Mean Pooling.
    Uses mean pooling across non-padding tokens for stable sentence embedding.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get hidden states (last_hidden_state)
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
    else:
        # Some models return tuple
        hidden = outputs[0]
    
    # Mean Pooling
    mask_expanded = attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
    sum_hidden = (hidden * mask_expanded).sum(dim=1)      # [1, hidden_dim]
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)      # [1, 1]
    embedding = (sum_hidden / count).squeeze(0)            # [hidden_dim]
    
    return embedding


# ─── Main Reranker Class ──────────────────────────────────────────────────────

class ZeroShotReranker:
    """
    Zero-Shot Content-Based Reranker using PhoBERT.
    
    No training required, leverages pre-loaded PhoBERT backbone.
    """
    
    def __init__(
        self,
        tokenizer,
        backbone: torch.nn.Module,
        device: torch.device,
        alpha: float = ALPHA,
        beta: float  = BETA,
        gamma: float = GAMMA,
        threshold: float = SCORE_THRESHOLD,
    ):
        self.tokenizer = tokenizer
        self.backbone  = backbone
        self.device    = device
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.threshold = threshold
    
    def _get_dominant_badge(
        self,
        cosine_contrib: float,
        trust_contrib: float,
        price_penalty: float,
        rating_val: float,
        sold_val: int,
    ) -> str:
        """
        Determine reasoning badge based on highest score contributor.
        """
        if cosine_contrib >= trust_contrib and cosine_contrib >= 0.25:
            # Semantic similarity dominant factor
            if rating_val >= 4.7 and sold_val >= 500:
                return "Tương tự & Bán chạy"
            return "Sản phẩm tương tự"
        elif trust_contrib >= cosine_contrib and trust_contrib >= 0.25:
            # Trust score dominant factor
            if sold_val >= 1000:
                return "Mua nhiều nhất"
            elif rating_val >= 4.8:
                return "Đánh giá cực tốt"
            return "Đánh giá cao"
        elif price_penalty < 0.05:
            # Price penalty is close to 0
            return "Giá hợp lý"
        else:
            # Fallback based on rating/sold
            if rating_val >= 4.5 and sold_val >= 200:
                return "Bán chạy & Uy tín"
            elif rating_val >= 4.0:
                return "Khách mua hài lòng"
            return "Cùng phân khúc"
    
    def rerank(
        self,
        origin_name: str,
        origin_price: Any,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate recommendations based on:
            Score(pᵢ) = α·CosineSim(E(p₀),E(pᵢ)) + β·Trust(pᵢ) - γ·|PriceDeviation|
        
        Args:
            origin_name    : Original product name
            origin_price   : Original product price
            candidates     : Candidate dictionaries
        
        Returns:
            Reranked list sorted by score descending
        """
        if not candidates:
            return []
        
        logger.info("[Reranker] Bắt đầu zero-shot reranking với %d ứng viên", len(candidates))
        
        # -- 1. Embed original product
        try:
            origin_vec = _embed_text(origin_name, self.tokenizer, self.backbone, self.device)
        except Exception as e:
            logger.warning("[Reranker] Không embed được sản phẩm gốc: %s", e)
            # Fallback: return unchanged candidates
            return candidates
        
        origin_price_float = _parse_price(origin_price)
        
        # -- 2. Compute Score for each candidate
        scored: List[Tuple[float, Dict[str, Any]]] = []
        
        for cand in candidates:
            name_cand = cand.get("name", "")
            if not name_cand:
                continue
            
            # 2a. Semantic similarity
            try:
                cand_vec = _embed_text(name_cand, self.tokenizer, self.backbone, self.device)
                cosine = max(0.0, _cosine_sim(origin_vec, cand_vec))  # clamp to [0,1]
            except Exception as e:
                logger.debug("[Reranker] Embed thất bại cho '%s': %s", name_cand[:40], e)
                cosine = 0.3  # Default value on error
            
            # 2b. Trust score (normalized)
            sold_int = _parse_sold(cand.get("sold", 0))
            trust_norm = _compute_trust(cand.get("rating", 0), sold_int)
            
            # 2c. Price deviation penalty
            cand_price_float = _parse_price(cand.get("price", 0))
            if origin_price_float > 0 and cand_price_float > 0:
                price_deviation = abs(cand_price_float - origin_price_float) / origin_price_float
                price_penalty = min(1.0, price_deviation)  # cap at 100% deviation
            else:
                price_deviation = 0.0
                price_penalty = 0.0
            
            # 2d. Composite Score
            alpha_contrib = self.alpha * cosine
            beta_contrib  = self.beta  * trust_norm
            gamma_penalty = self.gamma * price_penalty
            
            final_score = alpha_contrib + beta_contrib - gamma_penalty
            final_score = max(0.0, min(1.0, final_score))  # clamp [0,1]
            
            # 2e. Trust score display (0-100)
            # Use default rating of 3.0 when there are no ratings
            try:
                rating_raw = float(cand.get("rating") or 0)
            except (ValueError, TypeError):
                rating_raw = 0.0
            rating_val_display = rating_raw if rating_raw > 0 else 3.0

            mini_trust_display = round(
                min(100.0, rating_val_display * 20.0 + min(20.0, trust_norm * 20.0)),
                1
            )

            # 2f. Reasoning badge
            reason_badge = self._get_dominant_badge(
                cosine_contrib=alpha_contrib,
                trust_contrib=beta_contrib,
                price_penalty=gamma_penalty,
                rating_val=rating_val_display,
                sold_val=sold_int,
            )
            
            # 2g. Filter by score threshold
            if final_score < self.threshold:
                logger.debug(
                    "[Reranker] Loại '%s' vì Score=%.3f < threshold=%.3f",
                    name_cand[:40], final_score, self.threshold
                )
                continue
            
            enriched_cand = {
                **cand,
                "rerank_score":    round(final_score, 4),
                "cosine_score":    round(cosine, 4),
                "trust_score_norm": round(trust_norm, 4),
                "trustScore":      mini_trust_display,
                "reason":          reason_badge,
            }
            scored.append((final_score, enriched_cand))
        
        # -- 3. Sort candidates descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        result = [item for _, item in scored]
        logger.info(
            "[Reranker] Kết quả: %d/%d sản phẩm vượt threshold (%.2f). Scores: %s",
            len(result),
            len(candidates),
            self.threshold,
            [f"{s:.3f}" for s, _ in scored],
        )
        return result


# -- Standalone helper

def rerank_candidates(
    origin_name: str,
    origin_price: Any,
    candidates: List[Dict[str, Any]],
    tokenizer,
    backbone: torch.nn.Module,
    device: torch.device,
    alpha: float = ALPHA,
    beta: float  = BETA,
    gamma: float = GAMMA,
    threshold: float = SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Helper wrapper to rerank candidates directly from main.py.
    """
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
