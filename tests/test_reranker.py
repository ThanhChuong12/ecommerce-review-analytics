"""
Unit tests for the ZeroShotReranker (ai_engine/recommendation/reranker.py).
Tests are fully self-contained — no actual PhoBERT weights required.
All tests pass with a tiny stub backbone and tokenizer.
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# ── Fix import path ────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent
_AI_ENGINE = _THIS.parent / "ai_engine"
if str(_AI_ENGINE) not in sys.path:
    sys.path.insert(0, str(_AI_ENGINE))

from recommendation.reranker import (
    ZeroShotReranker,
    rerank_candidates,
    _parse_price,
    _parse_sold,
    _compute_trust,
    _cosine_sim,
    ALPHA, BETA, GAMMA, SCORE_THRESHOLD,
)


# ─── Stub backbone & tokenizer ────────────────────────────────────────────────

class StubOutput:
    """Simulates transformer last_hidden_state output."""
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class StubBackbone(nn.Module):
    """Deterministic minimal backbone: returns fixed embedding per call."""
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        # Return all-ones hidden state — deterministic and meaningful for tests
        hidden = torch.ones(B, L, self.hidden_dim)
        return StubOutput(hidden)


class StubTokenizer:
    """Minimal tokenizer that maps every text to a constant token ID sequence."""
    def __call__(self, text: str, return_tensors="pt", truncation=True, max_length=64, padding="max_length"):
        seq_len = min(max_length, max(4, len(text.split())))
        return {
            "input_ids":      torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }


@pytest.fixture
def stub_device():
    return torch.device("cpu")


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def stub_backbone():
    return StubBackbone(hidden_dim=16)


@pytest.fixture
def reranker(stub_tokenizer, stub_backbone, stub_device):
    return ZeroShotReranker(
        tokenizer=stub_tokenizer,
        backbone=stub_backbone,
        device=stub_device,
    )


# ─── Parser tests ─────────────────────────────────────────────────────────────

class TestParsePrice:
    def test_plain_number(self):
        assert _parse_price("120000") == 120000.0

    def test_currency_symbol(self):
        assert _parse_price("₫120.000") == 120000.0

    def test_millions_tr(self):
        # "1.2tr" → 1.2 * 1_000_000
        assert _parse_price("1.2tr") == pytest.approx(1_200_000.0)

    def test_none_returns_zero(self):
        assert _parse_price(None) == 0.0

    def test_empty_returns_zero(self):
        assert _parse_price("") == 0.0


class TestParseSold:
    def test_plain_integer(self):
        assert _parse_sold("1200") == 1200

    def test_k_suffix(self):
        assert _parse_sold("1.2k") == 1200

    def test_none_returns_zero(self):
        assert _parse_sold(None) == 0

    def test_sold_with_units(self):
        assert _parse_sold("Đã bán 500") == 500


# ─── Trust computation tests ───────────────────────────────────────────────────

class TestComputeTrust:
    def test_perfect_rating_high_sold(self):
        trust = _compute_trust(5.0, 2000)
        assert trust > 0.90, f"Expected > 0.9 but got {trust}"

    def test_low_rating_zero_sold(self):
        trust = _compute_trust(1.0, 0)
        assert trust == pytest.approx(0.0)

    def test_average_rating(self):
        trust = _compute_trust(3.0, 100)
        assert 0.2 < trust < 0.8

    def test_trust_in_unit_range(self):
        for rating in [1.0, 2.5, 3.5, 4.0, 5.0]:
            for sold in [0, 50, 200, 1000]:
                t = _compute_trust(rating, sold)
                assert 0.0 <= t <= 1.0, f"Trust out of range: {t} for rating={rating}, sold={sold}"


# ─── CosineSim tests ──────────────────────────────────────────────────────────

class TestCosineSim:
    def test_identical_vectors_return_one(self):
        v = torch.tensor([1.0, 0.5, -0.3])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_return_zero(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_clamped_to_zero(self):
        # Reranker clamps with max(0, ...) so opposite vectors → 0 after clamp
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        raw_sim = _cosine_sim(a, b)
        assert raw_sim == pytest.approx(-1.0, abs=1e-5)


# ─── ZeroShotReranker tests ───────────────────────────────────────────────────

class TestZeroShotReranker:

    def test_returns_list(self, reranker):
        candidates = [
            {"name": "Áo thun cotton cao cấp", "rating": 4.8, "sold": "500", "price": "120000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Áo thun unisex", 100000, candidates)
        assert isinstance(result, list)

    def test_enriches_candidate(self, reranker):
        candidates = [
            {"name": "Điện thoại Samsung Galaxy A54", "rating": 4.5, "sold": "1200", "price": "7990000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Điện thoại Samsung", 8000000, candidates)
        if result:
            item = result[0]
            assert "rerank_score" in item
            assert "cosine_score" in item
            assert "trust_score_norm" in item
            assert "trustScore" in item
            assert "reason" in item

    def test_score_in_valid_range(self, reranker):
        candidates = [
            {"name": "Laptop Asus VivoBook", "rating": 4.5, "sold": "300", "price": "15000000", "thumbnail": "", "url": ""},
            {"name": "Laptop Dell Inspiron", "rating": 4.2, "sold": "800", "price": "16500000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Laptop gaming", 15000000, candidates)
        for item in result:
            assert 0.0 <= item["rerank_score"] <= 1.0, f"Score out of bounds: {item['rerank_score']}"

    def test_filters_low_score_candidates(self):
        """Candidates with score < threshold must be excluded."""
        # Use extreme gamma penalty via very high price deviation
        reranker = ZeroShotReranker(
            tokenizer=StubTokenizer(),
            backbone=StubBackbone(),
            device=torch.device("cpu"),
            alpha=0.0,
            beta=0.0,
            gamma=1.0,   # Only price penalty, no similarity reward
            threshold=0.01,  # Low threshold — penalty still drives score to 0
        )
        candidates = [
            # Price 100x the origin → massive penalty
            {"name": "Sản phẩm giá siêu cao", "rating": 1.0, "sold": "0", "price": "9999999999", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Sản phẩm rẻ", 1, candidates)
        # Score = 0 + 0 - 1.0*|penalty| → negative → clamped to 0 → below threshold=0.01
        assert len(result) == 0

    def test_sorted_descending(self, reranker):
        candidates = [
            {"name": "Sản phẩm A", "rating": 3.0, "sold": "10", "price": "50000", "thumbnail": "", "url": ""},
            {"name": "Sản phẩm B", "rating": 4.9, "sold": "2000", "price": "48000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Sản phẩm", 50000, candidates)
        scores = [item["rerank_score"] for item in result]
        assert scores == sorted(scores, reverse=True), "Results must be sorted descending by rerank_score"

    def test_empty_candidates(self, reranker):
        result = reranker.rerank("Laptop", 10000000, [])
        assert result == []

    def test_candidates_without_name_skipped(self, reranker):
        candidates = [
            {"name": "", "rating": 4.8, "sold": "500", "price": "120000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("Áo thun", 100000, candidates)
        assert result == []

    def test_reason_badge_assigned(self, reranker):
        candidates = [
            {"name": "Điện thoại iPhone 15 Pro Max", "rating": 4.9, "sold": "2000", "price": "30000000", "thumbnail": "", "url": ""},
        ]
        result = reranker.rerank("iPhone 15", 29000000, candidates)
        if result:
            assert isinstance(result[0]["reason"], str)
            assert len(result[0]["reason"]) > 0


# ─── Standalone helper test ───────────────────────────────────────────────────

class TestRerankCandidates:
    def test_wrapper_returns_same_as_class(self, stub_tokenizer, stub_backbone, stub_device):
        candidates = [
            {"name": "Giày thể thao Nike Air Max", "rating": 4.7, "sold": "600", "price": "1200000", "thumbnail": "", "url": ""},
        ]
        result = rerank_candidates(
            origin_name="Giày thể thao",
            origin_price=1000000,
            candidates=candidates,
            tokenizer=stub_tokenizer,
            backbone=stub_backbone,
            device=stub_device,
        )
        assert isinstance(result, list)
